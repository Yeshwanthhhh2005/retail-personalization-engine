"""Diagnostic analysis: where the lift actually comes from.

A single leaderboard number hides the mechanism. Two questions decide whether
this system is worth deploying, and neither is answered by recall@20 alone:

  1. Repeat vs. discovery. A grocery recommender can score well by only
     re-selling the known basket. That has value, but it grows nothing. We
     split the metric by whether the purchased item was already in the
     customer's history and report both halves.

  2. Who is being served badly. Aggregate recall averages over a population
     that ranges from 1-purchase customers to 100-purchase customers. If the
     lift is concentrated in heavy shoppers, the headline is misleading.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data.splits import load_catalog, load_events, make_temporal_split

EVENT_PURCHASE = 2
K = 20


def decompose(
    predictions: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    known: dict[int, set[int]],
    k: int = K,
) -> dict[str, float]:
    """Recall split into repeat purchases and genuine discovery."""
    repeat_hits = repeat_total = new_hits = new_total = 0
    for customer, truth in labels.items():
        slate = set(predictions.get(customer, np.empty(0, dtype=np.int64))[:k].tolist())
        seen = known.get(customer, set())
        for item in truth:
            item = int(item)
            if item in seen:
                repeat_total += 1
                repeat_hits += item in slate
            else:
                new_total += 1
                new_hits += item in slate
    return {
        "repeat_recall": repeat_hits / max(repeat_total, 1),
        "discovery_recall": new_hits / max(new_total, 1),
        "repeat_share_of_truth": repeat_total / max(repeat_total + new_total, 1),
    }


def by_segment(
    predictions: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    depth: dict[int, int],
    k: int = K,
) -> pd.DataFrame:
    """Recall bucketed by how much history the customer has."""
    edges = [0, 5, 15, 30, 60, 10_000]
    names = ["1-5", "6-15", "16-30", "31-60", "60+"]
    rows = []
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        members = [c for c in labels if lo < depth.get(c, 0) <= hi]
        if not members:
            continue
        recalls = []
        for c in members:
            truth = set(int(i) for i in labels[c])
            slate = set(predictions.get(c, np.empty(0, dtype=np.int64))[:k].tolist())
            recalls.append(len(truth & slate) / len(truth))
        rows.append({
            "history_purchases": name,
            "customers": len(members),
            f"recall@{k}": float(np.mean(recalls)),
        })
    return pd.DataFrame(rows)


SOURCES = ("repeat", "two_tower", "item_knn", "popularity")


def source_contribution(scored: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """Which candidate source supplied the items that actually converted.

    `exclusive_hits` is the column that matters when deciding whether a source
    earns its operational cost: it counts hits that NO other source proposed.
    A source with high total hits but near-zero exclusive hits is redundant --
    it is re-finding what something cheaper already found.
    """
    ranked = scored.sort_values(["customer_id", "score"], ascending=[True, False])
    top = ranked.groupby("customer_id", sort=False).head(k)

    flags = {s: top[f"in_{s}"].to_numpy().astype(bool) for s in SOURCES}
    n_sources = np.sum([flags[s] for s in SOURCES], axis=0)
    hit = top["label"].to_numpy().astype(bool)

    rows = []
    for s in SOURCES:
        present = flags[s]
        exclusive = present & (n_sources == 1)
        rows.append({
            "source": s,
            "slate_share": float(present.mean()),
            "hits": int((present & hit).sum()),
            "precision": float((present & hit).sum() / max(present.sum(), 1)),
            "exclusive_slots": int(exclusive.sum()),
            "exclusive_hits": int((exclusive & hit).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    events, catalog = load_events(), load_catalog()
    split = make_temporal_split(events)

    history = events[events.day <= split.valid_end_day]
    purchases = history[history.event_type == EVENT_PURCHASE]
    known = {
        int(c): set(g.tolist())
        for c, g in purchases.groupby("customer_id").item_id.unique().items()
    }
    depth = purchases.groupby("customer_id").size().to_dict()

    # The scored test frame is exactly what the deployed ranker produced.
    scored = pd.read_parquet(CONFIG.report_dir / "ranker_test_scored.parquet")
    ranked = scored.sort_values(["customer_id", "score"], ascending=[True, False])
    preds = {
        int(c): g.item_id.to_numpy(dtype=np.int64)[:K]
        for c, g in ranked.groupby("customer_id", sort=False)
    }

    print("=" * 70)
    print("REPEAT VS DISCOVERY")
    print("=" * 70)
    split_metrics = decompose(preds, split.test_labels, known)
    for key, value in split_metrics.items():
        print(f"  {key:<24} {value:.4f}")

    print("\n" + "=" * 70)
    print(f"RECALL@{K} BY CUSTOMER HISTORY DEPTH")
    print("=" * 70)
    segments = by_segment(preds, split.test_labels, depth)
    print(segments.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 70)
    print(f"CANDIDATE SOURCE CONTRIBUTION (top {K})")
    print("=" * 70)
    contrib = source_contribution(scored)
    print(contrib.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = {
        "repeat_vs_discovery": split_metrics,
        "by_segment": segments.to_dict(orient="records"),
        "source_contribution": contrib.to_dict(orient="records"),
    }
    with open(CONFIG.report_dir / "analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    segments.to_csv(CONFIG.report_dir / "segments.csv", index=False)
    contrib.to_csv(CONFIG.report_dir / "source_contribution.csv", index=False)
    print(f"\nwritten to {CONFIG.report_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
