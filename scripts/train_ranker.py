"""End-to-end two-stage pipeline: multi-source retrieval -> learned ranking.

Timeline discipline is the whole point of this script, so it is spelled out:

    days 0..105    train      retrieval models are fitted here
    days 106..112  valid      labels for the RANKER's training set
    days 113..119  test       labels for the final report, touched once

The ranker's training rows are built from features as of day 105 with labels
from the valid week. The test rows are built from features as of day 112 with
labels from the test week. The retrieval models stay fitted on train only --
a week-stale retrieval model is what production actually serves between
retrains, so scoring against a freshly-refit one would flatter us.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from src.config import CONFIG
from src.data.splits import load_catalog, load_events, make_temporal_split
from src.evaluation.metrics import evaluate_ranking, format_leaderboard, leaderboard
from src.features.ranking_features import (
    FEATURE_COLUMNS,
    SOURCES,
    attach_labels,
    build_candidates,
    build_feature_frame,
)
from src.features.sequences import DENSE_NAMES, build_inference_state
from src.models.baselines import (
    ItemKNNRecommender,
    PopularityRecommender,
    RepeatPurchaseRecommender,
)
from src.models.ranker import rank_candidates, train_ranker
from src.models.trainer import encode_users
from src.models.two_tower import ItemFeatures, TwoTowerModel
from src.retrieval.vector_index import ItemVectorIndex

PER_SOURCE = {"repeat": 60, "two_tower": 120, "item_knn": 60, "popularity": 20}
BATCH = 2_000


def load_two_tower(n_users: int, n_items: int, catalog) -> TwoTowerModel:
    blob = torch.load(CONFIG.artifact_dir / "two_tower.pt", weights_only=False)
    model = TwoTowerModel(
        n_users=blob["n_users"], n_items=blob["n_items"],
        item_features=ItemFeatures.from_catalog(catalog),
        n_dense=len(DENSE_NAMES),
        cfg=CONFIG.two_tower,
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def batched(model, customers: np.ndarray, k: int) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for start in range(0, len(customers), BATCH):
        out.update(model.recommend(customers[start : start + BATCH], k))
    return out


def retrieval_sources(
    history: pd.DataFrame,
    catalog: pd.DataFrame,
    customers: np.ndarray,
    two_tower: TwoTowerModel,
    index: ItemVectorIndex,
    n_items: int,
    as_of_day: int,
) -> tuple[dict, dict]:
    """Run every candidate source against a given history cutoff."""
    sources: dict[str, dict[int, np.ndarray]] = {}
    scores: dict[str, dict[int, np.ndarray]] = {}

    repeat = RepeatPurchaseRecommender(backfill=False).fit(history, catalog)
    sources["repeat"] = batched(repeat, customers, PER_SOURCE["repeat"])

    knn = ItemKNNRecommender().fit(history, catalog)
    sources["item_knn"] = batched(knn, customers, PER_SOURCE["item_knn"])

    pop = PopularityRecommender().fit(history, catalog)
    sources["popularity"] = batched(pop, customers, PER_SOURCE["popularity"])

    hist, hlen, dense = build_inference_state(
        history, n_items, customers,
        history_len=CONFIG.two_tower.history_len,
        item_price=catalog.price.to_numpy(),
    )
    user_vecs = encode_users(two_tower, customers, hist, hlen, dense)
    ids, sims = index.search(user_vecs, PER_SOURCE["two_tower"])
    sources["two_tower"] = {int(c): ids[i] for i, c in enumerate(customers)}
    scores["two_tower"] = {int(c): sims[i] for i, c in enumerate(customers)}
    return sources, scores


def build_stage(
    name: str,
    history: pd.DataFrame,
    catalog: pd.DataFrame,
    labels: dict[int, np.ndarray],
    two_tower: TwoTowerModel,
    index: ItemVectorIndex,
    n_items: int,
    as_of_day: int,
) -> pd.DataFrame:
    customers = np.array(sorted(labels.keys()), dtype=np.int64)
    t0 = time.perf_counter()
    sources, scores = retrieval_sources(
        history, catalog, customers, two_tower, index, n_items, as_of_day
    )
    cands = build_candidates(customers, sources, scores, PER_SOURCE)
    frame = build_feature_frame(cands, history, catalog, as_of_day)
    frame = attach_labels(frame, labels)
    frame = frame.sort_values(["customer_id", "item_id"]).reset_index(drop=True)

    hit_rate = frame.groupby("customer_id").label.max().mean()
    recall_ceiling = float(np.mean([
        len(set(frame.item_id[frame.customer_id == c]) & set(labels[c])) / len(labels[c])
        for c in customers[:500]
    ]))
    print(f"  {name}: {len(frame):,} rows, {len(customers):,} customers, "
          f"{len(frame) / len(customers):.0f} cand/customer, "
          f"positives {frame.label.mean():.4f}, "
          f"customer coverage {hit_rate:.4f}, "
          f"candidate recall ceiling ~{recall_ceiling:.4f} "
          f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    return frame


def main() -> None:
    events, catalog = load_events(), load_catalog()
    split = make_temporal_split(events)
    n_items = len(catalog)
    n_users = int(events.customer_id.max()) + 1

    two_tower = load_two_tower(n_users, n_items, catalog)
    index = ItemVectorIndex.load(CONFIG.artifact_dir / "item_index")

    print("building ranker training rows (features<=day %d, labels=valid week)"
          % split.train_end_day, flush=True)
    train_frame = build_stage(
        "ranker-train", split.train, catalog, split.valid_labels,
        two_tower, index, n_items, split.train_end_day,
    )

    print("building ranker test rows (features<=day %d, labels=test week)"
          % split.valid_end_day, flush=True)
    known = events[events.day <= split.valid_end_day]
    test_frame = build_stage(
        "ranker-test", known, catalog, split.test_labels,
        two_tower, index, n_items, split.valid_end_day,
    )

    # Hold out whole customers for early stopping -- splitting rows within a
    # customer would leak the group structure LambdaRank is grouped on.
    rng = np.random.default_rng(CONFIG.ranker.seed)
    cust = train_frame.customer_id.unique()
    holdout = set(rng.choice(cust, size=int(0.15 * len(cust)), replace=False).tolist())
    is_holdout = train_frame.customer_id.isin(holdout)
    fit_frame = train_frame[~is_holdout].reset_index(drop=True)
    es_frame = train_frame[is_holdout].reset_index(drop=True)

    print(f"\ntraining LambdaRank on {len(fit_frame):,} rows "
          f"({fit_frame.customer_id.nunique():,} customers), "
          f"early stopping on {es_frame.customer_id.nunique():,}", flush=True)
    artifacts = train_ranker(fit_frame, es_frame, FEATURE_COLUMNS, CONFIG.ranker)
    print(f"  best iteration: {artifacts.best_iteration}")

    imp = artifacts.importance("gain")
    print("\n  top features by gain:")
    for _, r in imp.head(15).iterrows():
        print(f"    {r.feature:<22} {r.importance:>12,.0f}")

    t0 = time.perf_counter()
    preds = rank_candidates(artifacts, test_frame, max(CONFIG.evaluation.k_values))
    rank_s = time.perf_counter() - t0

    res = evaluate_ranking(
        preds, split.test_labels, k_values=CONFIG.evaluation.k_values,
        catalog_size=n_items, item_popularity=catalog.base_popularity.to_numpy(),
        item_price=catalog.price.to_numpy(), name="two_stage_ranker",
    )
    res.metrics["infer_ms_per_1k"] = round(
        1000 * rank_s / len(split.test_labels) * 1000, 1
    )

    board = leaderboard([res])
    cols = [f"recall@{k}" for k in CONFIG.evaluation.k_values] + [
        "ndcg@20", "map@20", "hit_rate@20", "coverage@20", "novelty@20"
    ]
    print("\n" + format_leaderboard(board, cols))

    artifacts.model.save_model(str(CONFIG.artifact_dir / "ranker.txt"),
                              num_iteration=artifacts.best_iteration)
    imp.to_csv(CONFIG.report_dir / "ranker_importance.csv", index=False)
    board.to_csv(CONFIG.report_dir / "ranker.csv", index=False)
    # Persist the scored test frame so downstream analysis works on exactly
    # what the ranker produced, rather than reconstructing an approximation.
    scored = test_frame[["customer_id", "item_id", "label"]].copy()
    scored["score"] = artifacts.model.predict(
        test_frame[artifacts.feature_names].astype(np.float32),
        num_iteration=artifacts.best_iteration,
    )
    for source in SOURCES:
        scored[f"in_{source}"] = test_frame[f"in_{source}"].to_numpy()
    scored["ui_n_purchase"] = test_frame["ui_n_purchase"].to_numpy()
    scored.to_parquet(
        CONFIG.report_dir / "ranker_test_scored.parquet", index=False
    )
    with open(CONFIG.artifact_dir / "ranker_features.json", "w") as f:
        json.dump(artifacts.feature_names, f, indent=2)
    print(f"\nartifacts -> {CONFIG.artifact_dir}")


if __name__ == "__main__":
    main()
