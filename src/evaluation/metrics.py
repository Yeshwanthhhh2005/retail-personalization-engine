"""Offline metrics for top-K retrieval and ranking.

Two families, and the distinction matters when arguing for a launch:

  * Accuracy  -- recall / NDCG / MAP. Did we put the right things in the slate?
  * Health    -- coverage, novelty, gini. Are we merely re-selling the head?

A model that wins on accuracy while collapsing coverage is usually a model
that has learned popularity. Reporting both keeps that honest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvalResult:
    name: str
    metrics: dict[str, float]

    def row(self) -> dict[str, float | str]:
        return {"model": self.name, **self.metrics}


def _hit_matrix(
    ranked: np.ndarray, labels: list[np.ndarray], k: int
) -> np.ndarray:
    """(n_customers, k) boolean: was slot j a true future purchase?"""
    topk = ranked[:, :k]
    hits = np.zeros(topk.shape, dtype=bool)
    for i, truth in enumerate(labels):
        if len(truth):
            hits[i] = np.isin(topk[i], truth, assume_unique=False)
    return hits


def evaluate_ranking(
    predictions: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    k_values: tuple[int, ...] = (5, 10, 20, 50),
    catalog_size: int | None = None,
    item_popularity: np.ndarray | None = None,
    item_price: np.ndarray | None = None,
    name: str = "model",
) -> EvalResult:
    """Score a set of per-customer ranked lists against held-out purchases.

    Customers present in `labels` but missing from `predictions` are scored as
    complete misses rather than dropped -- a model that declines to serve part
    of the population should pay for it in the headline number.
    """
    customers = sorted(labels.keys())
    max_k = max(k_values)

    ranked = np.full((len(customers), max_k), -1, dtype=np.int64)
    truth: list[np.ndarray] = []
    for i, c in enumerate(customers):
        pred = predictions.get(c, np.empty(0, dtype=np.int64))
        n = min(len(pred), max_k)
        if n:
            ranked[i, :n] = pred[:n]
        truth.append(labels[c])

    n_truth = np.array([len(t) for t in truth], dtype=np.float64)
    metrics: dict[str, float] = {}

    for k in k_values:
        hits = _hit_matrix(ranked, truth, k)
        n_hits = hits.sum(axis=1)

        metrics[f"recall@{k}"] = float(np.mean(n_hits / np.maximum(n_truth, 1)))
        metrics[f"precision@{k}"] = float(np.mean(n_hits / k))
        metrics[f"hit_rate@{k}"] = float(np.mean(n_hits > 0))

        # NDCG with binary gains, ideal list truncated at min(|truth|, k).
        discount = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = (hits * discount).sum(axis=1)
        ideal_len = np.minimum(n_truth, k).astype(int)
        idcg = np.array([discount[:n].sum() if n else 1.0 for n in ideal_len])
        metrics[f"ndcg@{k}"] = float(np.mean(dcg / np.maximum(idcg, 1e-12)))

        # MAP: precision at each hit position, averaged over min(|truth|, k).
        positions = np.arange(1, k + 1)
        cum_hits = np.cumsum(hits, axis=1)
        prec_at_hit = np.where(hits, cum_hits / positions, 0.0).sum(axis=1)
        metrics[f"map@{k}"] = float(
            np.mean(prec_at_hit / np.maximum(ideal_len, 1))
        )

    # --- Slate health, reported at the primary K only. -------------------
    k = max(k_values) if max(k_values) <= 20 else 20
    slate = ranked[:, :k]
    served = slate[slate >= 0]

    if catalog_size:
        metrics[f"coverage@{k}"] = float(len(np.unique(served)) / catalog_size)
    if item_popularity is not None and len(served):
        p = np.clip(item_popularity[served], 1e-12, None)
        metrics[f"novelty@{k}"] = float(np.mean(-np.log2(p)))
    if len(served):
        counts = np.bincount(served)
        counts = np.sort(counts[counts > 0]).astype(np.float64)
        n = len(counts)
        metrics[f"gini@{k}"] = float(
            (2 * np.arange(1, n + 1) - n - 1).dot(counts) / (n * counts.sum())
        )
    if item_price is not None:
        hits = _hit_matrix(ranked, truth, k)
        hit_items = np.where(hits, slate, 0)
        metrics[f"revenue_hit@{k}"] = float(
            np.mean((item_price[hit_items] * hits).sum(axis=1))
        )

    return EvalResult(name=name, metrics=metrics)


def leaderboard(results: list[EvalResult], sort_by: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame([r.row() for r in results])
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    return df.reset_index(drop=True)


def format_leaderboard(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    cols = ["model"] + (columns or [c for c in df.columns if c != "model"])
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")
