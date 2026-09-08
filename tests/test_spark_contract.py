"""Keeps the Spark and pandas feature paths from drifting apart.

The Spark pipeline cannot run in this environment (no JVM), so it cannot be
tested by execution. What CAN be tested without a JVM is the contract between
the two implementations: that they name the same features, and that the
semantics the Spark module documents in its fillna policy are the semantics
the pandas module actually implements.

That is worth testing precisely because a drift here is silent -- the Spark
job would keep running and quietly serve a different feature distribution
than the model was trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import spark_pipeline
from src.features.ranking_features import (
    FEATURE_COLUMNS,
    build_feature_frame,
)


def test_spark_module_imports_without_pyspark():
    """The TYPE_CHECKING guard must keep pyspark out of import time.

    If this fails, importing anything from src.features pulls in a JVM
    dependency and the training path breaks on machines without Spark.
    """
    assert spark_pipeline.EVENT_PURCHASE == 2
    assert callable(spark_pipeline.assemble_candidate_features)


def test_declared_features_exist_in_the_pandas_contract():
    declared = (
        spark_pipeline.USER_ITEM_FEATURES
        + spark_pipeline.USER_FEATURES
        + spark_pipeline.ITEM_FEATURES
    )
    missing = [f for f in declared if f not in FEATURE_COLUMNS]
    assert not missing, f"Spark declares features the ranker never consumes: {missing}"


def _toy_data():
    catalog = pd.DataFrame({
        "item_id": np.arange(4, dtype=np.int32),
        "category_id": np.array([0, 0, 1, 1], dtype=np.int16),
        "category_name": ["A", "A", "B", "B"],
        "brand_id": np.array([0, 1, 0, 1], dtype=np.int16),
        "price": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        "is_replenishable": [True, True, False, False],
        "base_popularity": np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
    })
    # Customer 0 buys item 0 on days 0 and 10; never touches item 3.
    history = pd.DataFrame({
        "customer_id": np.array([0, 0, 0], dtype=np.int32),
        "item_id": np.array([0, 0, 1], dtype=np.int32),
        "day": np.array([0, 10, 5], dtype=np.int16),
        "event_type": np.array([2, 2, 2], dtype=np.int8),
        "quantity": np.array([1, 1, 1], dtype=np.int8),
        "event_seq": np.arange(3, dtype=np.int64),
    })
    candidates = pd.DataFrame({
        "customer_id": np.array([0, 0], dtype=np.int32),
        "item_id": np.array([0, 3], dtype=np.int32),
    })
    return candidates, history, catalog


def test_unseen_item_recency_is_far_past_not_zero():
    """The fillna policy both modules document, asserted on real output.

    Filling an unseen item's recency with 0 would tell the model "bought
    today" -- the exact inverse of the truth, and a bug that raises offline
    metrics while destroying online behaviour.
    """
    candidates, history, catalog = _toy_data()
    frame = build_feature_frame(candidates, history, catalog, as_of_day=12)

    seen = frame[frame.item_id == 0].iloc[0]
    unseen = frame[frame.item_id == 3].iloc[0]

    assert seen["ui_days_since"] == 2      # last bought day 10, as_of 12
    assert unseen["ui_days_since"] == 9_999
    assert seen["is_known_item"] == 1
    assert unseen["is_known_item"] == 0
    assert unseen["ui_n_purchase"] == 0


def test_repurchase_cadence_is_computed_from_history():
    candidates, history, catalog = _toy_data()
    frame = build_feature_frame(candidates, history, catalog, as_of_day=12)
    seen = frame[frame.item_id == 0].iloc[0]
    # Two purchases spanning days 0..10 -> mean gap 10/2 = 5, due ratio 2/5.
    assert abs(seen["ui_mean_gap"] - 5.0) < 1e-9
    assert abs(seen["ui_due_ratio"] - 0.4) < 1e-9


def test_feature_builder_refuses_to_read_past_the_cutoff():
    """The leak guard must actually fire."""
    candidates, history, catalog = _toy_data()
    try:
        build_feature_frame(candidates, history, catalog, as_of_day=5)
    except AssertionError as exc:
        assert "leaks" in str(exc)
    else:
        raise AssertionError("expected a leak assertion when history > as_of_day")


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
