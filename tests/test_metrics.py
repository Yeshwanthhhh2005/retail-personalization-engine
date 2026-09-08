"""Hand-checked cases for the metric layer.

Every model number in this project is only as trustworthy as these functions,
so the expected values below are computed by hand in the comments rather than
snapshotted from the implementation.
"""
from __future__ import annotations

import math

import numpy as np

from src.evaluation.metrics import evaluate_ranking

TOL = 1e-9


def _run(predictions, labels, k_values=(3,), **kw):
    return evaluate_ranking(
        {c: np.asarray(v) for c, v in predictions.items()},
        {c: np.asarray(v) for c, v in labels.items()},
        k_values=k_values,
        **kw,
    ).metrics


def test_perfect_single_hit():
    m = _run({0: [1, 2, 3]}, {0: [1]}, k_values=(1,))
    assert abs(m["recall@1"] - 1.0) < TOL
    assert abs(m["ndcg@1"] - 1.0) < TOL
    assert abs(m["map@1"] - 1.0) < TOL
    assert abs(m["precision@1"] - 1.0) < TOL


def test_partial_hits_at_k3():
    # truth {1,2}, ranked [3,1,2] -> hits at positions 2 and 3.
    m = _run({0: [3, 1, 2]}, {0: [1, 2]}, k_values=(3,))
    assert abs(m["recall@3"] - 1.0) < TOL
    assert abs(m["precision@3"] - 2 / 3) < TOL

    dcg = 1 / math.log2(3) + 1 / math.log2(4)      # 1.13093
    idcg = 1 / math.log2(2) + 1 / math.log2(3)     # 1.63093
    assert abs(m["ndcg@3"] - dcg / idcg) < 1e-9

    # precision at each hit: 1/2 at pos 2, 2/3 at pos 3; averaged over |truth|=2
    assert abs(m["map@3"] - (0.5 + 2 / 3) / 2) < 1e-9


def test_complete_miss():
    m = _run({0: [7, 8, 9]}, {0: [1, 2]}, k_values=(3,))
    for key in ("recall@3", "precision@3", "ndcg@3", "map@3", "hit_rate@3"):
        assert m[key] == 0.0


def test_unserved_customer_counts_as_miss():
    # Customer 1 has a label but no prediction: must drag the mean down, not
    # be silently dropped from the denominator.
    m = _run({0: [1]}, {0: [1], 1: [5]}, k_values=(1,))
    assert abs(m["recall@1"] - 0.5) < TOL
    assert abs(m["hit_rate@1"] - 0.5) < TOL


def test_recall_normalises_by_basket_not_k():
    # truth has 4 items, k=2 -> at most 2/4 recall even when both slots hit.
    m = _run({0: [1, 2, 3, 4]}, {0: [1, 2, 3, 4]}, k_values=(2,))
    assert abs(m["recall@2"] - 0.5) < TOL
    assert abs(m["precision@2"] - 1.0) < TOL
    assert abs(m["ndcg@2"] - 1.0) < TOL  # ideal list is also truncated at 2


def test_ordering_matters_for_ndcg_not_recall():
    early = _run({0: [1, 9, 9]}, {0: [1]}, k_values=(3,))
    late = _run({0: [9, 9, 1]}, {0: [1]}, k_values=(3,))
    assert abs(early["recall@3"] - late["recall@3"]) < TOL
    assert early["ndcg@3"] > late["ndcg@3"]
    assert abs(late["ndcg@3"] - 1 / math.log2(4)) < TOL


def test_coverage_and_gini():
    # Two customers, disjoint slates over a catalogue of 10 -> coverage 4/10.
    m = _run(
        {0: [0, 1], 1: [2, 3]}, {0: [0], 1: [2]},
        k_values=(2,), catalog_size=10,
    )
    assert abs(m["coverage@2"] - 0.4) < TOL
    assert abs(m["gini@2"]) < TOL  # every served item appears exactly once

    # Same item to everyone -> coverage collapses, gini stays 0 (one item).
    m2 = _run(
        {0: [0, 1], 1: [0, 1]}, {0: [0], 1: [0]},
        k_values=(2,), catalog_size=10,
    )
    assert abs(m2["coverage@2"] - 0.2) < TOL


def test_novelty_prefers_rare_items():
    pop = np.array([0.5, 0.5, 0.001, 0.001])
    head = _run({0: [0, 1]}, {0: [0]}, k_values=(2,), item_popularity=pop)
    tail = _run({0: [2, 3]}, {0: [2]}, k_values=(2,), item_popularity=pop)
    assert tail["novelty@2"] > head["novelty@2"]
    assert abs(head["novelty@2"] - 1.0) < TOL  # -log2(0.5)


def test_revenue_counts_only_hits():
    price = np.array([10.0, 100.0, 1000.0])
    m = _run(
        {0: [0, 1]}, {0: [1]}, k_values=(2,), item_price=price,
    )
    assert abs(m["revenue_hit@2"] - 100.0) < TOL


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
