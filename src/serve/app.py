"""Online recommendation service.

The request path is the same two stages as the offline pipeline, which is the
point -- the offline numbers only mean something if the thing serving traffic
computes features the same way:

    1. user tower forward pass          (~1 ms, the only neural work online)
    2. ANN lookup + repeat/knn/popular  (candidate generation)
    3. feature store join               (precomputed aggregates, small joins)
    4. LightGBM scoring                 (~200 candidates)

The item tower never runs online. Item vectors are precomputed into the index
at refresh time; that asymmetry is the entire reason two-tower is the standard
retrieval architecture rather than a cross-encoder.

Run with:  uvicorn src.serve.app:app --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CONFIG
from src.data.splits import load_catalog, load_events, make_temporal_split
from src.features.ranking_features import FeatureStore, build_candidates
from src.features.sequences import DENSE_NAMES, build_inference_state
from src.models.baselines import (
    ItemKNNRecommender,
    PopularityRecommender,
    RepeatPurchaseRecommender,
)
from src.models.trainer import encode_users
from src.models.two_tower import ItemFeatures, TwoTowerModel
from src.retrieval.vector_index import ItemVectorIndex

PER_SOURCE = {"repeat": 60, "two_tower": 120, "item_knn": 60, "popularity": 20}


class RecommendRequest(BaseModel):
    customer_id: int = Field(..., ge=0, description="Customer to score")
    k: int = Field(10, ge=1, le=100, description="Slate size")
    explain: bool = Field(False, description="Attach a human-readable reason")


class Recommendation(BaseModel):
    item_id: int
    score: float
    category: str
    price: float
    sources: list[str]
    reason: str | None = None


class RecommendResponse(BaseModel):
    customer_id: int
    recommendations: list[Recommendation]
    latency_ms: float
    cold_start: bool
    model_version: str


@dataclass
class ServingState:
    """Everything loaded once at startup and shared across requests."""

    catalog: pd.DataFrame = None
    two_tower: TwoTowerModel = None
    index: ItemVectorIndex = None
    ranker: lgb.Booster = None
    feature_names: list[str] = None
    feature_store: FeatureStore = None
    repeat: RepeatPurchaseRecommender = None
    knn: ItemKNNRecommender = None
    popular: PopularityRecommender = None
    history: pd.DataFrame = None
    as_of_day: int = 0
    n_items: int = 0
    # User-tower inputs for every customer, materialised once at startup.
    # Rebuilding these per request means re-scanning the whole event log on
    # each call -- it was worth ~2.9s of p50 latency before this was hoisted.
    user_history: np.ndarray = None
    user_history_len: np.ndarray = None
    user_dense: np.ndarray = None
    catalog_meta: pd.DataFrame = None
    latencies: list[float] = field(default_factory=list)
    requests: int = 0


STATE = ServingState()
MODEL_VERSION = "two_stage_v1"


def _load() -> None:
    events, catalog = load_events(), load_catalog()
    split = make_temporal_split(events)
    history = events[events.day <= split.valid_end_day]

    STATE.catalog = catalog
    STATE.n_items = len(catalog)
    STATE.as_of_day = split.valid_end_day
    STATE.history = history

    blob = torch.load(CONFIG.artifact_dir / "two_tower.pt", weights_only=False)
    model = TwoTowerModel(
        n_users=blob["n_users"], n_items=blob["n_items"],
        item_features=ItemFeatures.from_catalog(catalog),
        n_dense=len(DENSE_NAMES), cfg=CONFIG.two_tower,
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    STATE.two_tower = model

    STATE.index = ItemVectorIndex.load(CONFIG.artifact_dir / "item_index")
    STATE.ranker = lgb.Booster(model_file=str(CONFIG.artifact_dir / "ranker.txt"))
    STATE.feature_names = STATE.ranker.feature_name()

    # Aggregates are computed once here, exactly as the training job computes
    # them, and then only joined per request.
    STATE.feature_store = FeatureStore.build(history, catalog, split.valid_end_day)
    STATE.repeat = RepeatPurchaseRecommender(backfill=False).fit(history, catalog)
    STATE.knn = ItemKNNRecommender().fit(history, catalog)
    STATE.popular = PopularityRecommender().fit(history, catalog)

    # Same reasoning as the feature store: the user-tower inputs are derived
    # from history that only changes at refresh time, so they are built once
    # for the whole customer base and indexed per request.
    all_customers = np.arange(CONFIG.data.n_customers, dtype=np.int64)
    STATE.user_history, STATE.user_history_len, STATE.user_dense = (
        build_inference_state(
            history, STATE.n_items, all_customers,
            history_len=CONFIG.two_tower.history_len,
            item_price=catalog.price.to_numpy(),
        )
    )
    STATE.catalog_meta = catalog.set_index("item_id")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    STATE.latencies.clear()


app = FastAPI(
    title="Walmart Personalization Service",
    description="Two-stage retrieval and ranking for customer product slates.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    ready = STATE.ranker is not None
    return {
        "status": "ok" if ready else "loading",
        "model_version": MODEL_VERSION,
        "catalog_size": STATE.n_items,
        "as_of_day": STATE.as_of_day,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    if not STATE.latencies:
        return {"requests": 0}
    lat = np.array(STATE.latencies)
    return {
        "requests": STATE.requests,
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
            "max": round(float(lat.max()), 2),
        },
    }


def _recommend(customer_id: int, k: int) -> tuple[list[dict], bool]:
    customers = np.array([customer_id], dtype=np.int64)
    catalog = STATE.catalog

    known = customer_id in STATE.repeat.history_
    if not known:
        # Cold start: no history to build a user vector from, so serve the
        # popularity slate rather than a confidently-wrong personalised one.
        top = STATE.popular.ranking_[:k]
        rows = [
            {"item_id": int(i), "score": float(k - rank), "sources": ["popularity"]}
            for rank, i in enumerate(top)
        ]
        return rows, True

    sources: dict[str, dict[int, np.ndarray]] = {
        "repeat": STATE.repeat.recommend(customers, PER_SOURCE["repeat"]),
        "item_knn": STATE.knn.recommend(customers, PER_SOURCE["item_knn"]),
        "popularity": STATE.popular.recommend(customers, PER_SOURCE["popularity"]),
    }
    row = slice(customer_id, customer_id + 1)
    user_vec = encode_users(
        STATE.two_tower, customers,
        STATE.user_history[row], STATE.user_history_len[row], STATE.user_dense[row],
    )
    ids, sims = STATE.index.search(user_vec, PER_SOURCE["two_tower"])
    sources["two_tower"] = {customer_id: ids[0]}

    cands = build_candidates(
        customers, sources, {"two_tower": {customer_id: sims[0]}}, PER_SOURCE
    )
    frame = STATE.feature_store.transform(cands)
    scores = STATE.ranker.predict(frame[STATE.feature_names].astype(np.float32))

    order = np.argsort(-scores)[:k]
    rows = []
    for pos in order:
        row = frame.iloc[pos]
        rows.append({
            "item_id": int(row.item_id),
            "score": float(scores[pos]),
            "sources": [s for s in PER_SOURCE if row[f"in_{s}"] == 1],
        })
    return rows, False


def _reason(frame_row: pd.Series) -> str:
    """A short, factual reason string built from the feature values."""
    if frame_row.get("ui_n_purchase", 0) > 0:
        days = int(frame_row.get("ui_days_since", 0))
        n = int(frame_row["ui_n_purchase"])
        if frame_row.get("ui_due_ratio", 0) >= 1.0:
            return f"You buy this regularly and it has been {days} days"
        return f"Purchased {n} time(s) before, most recently {days} days ago"
    if frame_row.get("uc_share", 0) > 0.15:
        return f"Popular in a category you shop often"
    return "Similar to products you have bought"


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    t0 = time.perf_counter()
    if STATE.ranker is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if req.customer_id >= CONFIG.data.n_customers:
        raise HTTPException(
            status_code=404, detail=f"unknown customer {req.customer_id}"
        )

    rows, cold = _recommend(req.customer_id, req.k)
    meta = STATE.catalog_meta

    recs = [
        Recommendation(
            item_id=r["item_id"],
            score=round(r["score"], 4),
            category=str(meta.loc[r["item_id"], "category_name"]),
            price=float(meta.loc[r["item_id"], "price"]),
            sources=r["sources"],
        )
        for r in rows
    ]

    latency = (time.perf_counter() - t0) * 1000
    STATE.latencies.append(latency)
    STATE.requests += 1
    return RecommendResponse(
        customer_id=req.customer_id,
        recommendations=recs,
        latency_ms=round(latency, 2),
        cold_start=cold,
        model_version=MODEL_VERSION,
    )
