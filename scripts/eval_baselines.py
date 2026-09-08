"""Fit and score the non-neural baselines on the held-out test week."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data.splits import load_catalog, load_events, make_temporal_split
from src.evaluation.metrics import evaluate_ranking, format_leaderboard, leaderboard
from src.models.baselines import (
    ALSRecommender,
    BlendRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RepeatPurchaseRecommender,
)

BATCH = 2_000


def recommend_batched(model, customer_ids: np.ndarray, k: int) -> dict[int, np.ndarray]:
    """Score in batches so the dense customer x item block stays bounded."""
    out: dict[int, np.ndarray] = {}
    for start in range(0, len(customer_ids), BATCH):
        out.update(model.recommend(customer_ids[start : start + BATCH], k))
    return out


def main() -> None:
    events, catalog = load_events(), load_catalog()
    split = make_temporal_split(events)
    print(split.summary().to_string(index=False), "\n")

    customers = np.array(sorted(split.test_labels.keys()), dtype=np.int32)
    popularity = catalog.base_popularity.to_numpy()
    price = catalog.price.to_numpy()
    k_values = CONFIG.evaluation.k_values
    max_k = max(k_values)

    models = [
        PopularityRecommender(),
        RepeatPurchaseRecommender(),
        ItemKNNRecommender(),
        ALSRecommender(),
    ]

    fitted, results = [], []
    for model in models:
        t0 = time.perf_counter()
        model.fit(split.train, catalog)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = recommend_batched(model, customers, max_k)
        infer_s = time.perf_counter() - t0

        res = evaluate_ranking(
            preds, split.test_labels, k_values=k_values,
            catalog_size=len(catalog), item_popularity=popularity,
            item_price=price, name=model.name,
        )
        res.metrics["fit_s"] = round(fit_s, 1)
        res.metrics["infer_ms_per_1k"] = round(1000 * infer_s / len(customers) * 1000, 1)
        results.append(res)
        fitted.append(model)
        print(f"  {model.name:<12} fit {fit_s:6.1f}s  "
              f"recall@20 {res.metrics['recall@20']:.4f}")

    blend = BlendRecommender(
        [fitted[1], fitted[2], fitted[3]], weights=[1.0, 0.8, 0.8],
        name="blend_rrf",
    )
    t0 = time.perf_counter()
    preds = recommend_batched(blend, customers, max_k)
    res = evaluate_ranking(
        preds, split.test_labels, k_values=k_values, catalog_size=len(catalog),
        item_popularity=popularity, item_price=price, name=blend.name,
    )
    res.metrics["fit_s"] = 0.0
    res.metrics["infer_ms_per_1k"] = round(
        1000 * (time.perf_counter() - t0) / len(customers) * 1000, 1
    )
    results.append(res)

    board = leaderboard(results, sort_by="recall@20")
    cols = [f"recall@{k}" for k in k_values] + [
        "ndcg@20", "map@20", "hit_rate@20", "coverage@20", "novelty@20", "fit_s"
    ]
    print("\n" + format_leaderboard(board, cols))
    board.to_csv(CONFIG.report_dir / "baselines.csv", index=False)
    print(f"\nwritten to {CONFIG.report_dir / 'baselines.csv'}")


if __name__ == "__main__":
    main()
