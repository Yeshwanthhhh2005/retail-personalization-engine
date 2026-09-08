"""Temporal splitting.

Interaction data must never be split at random. A random split lets the model
see a customer's Friday basket while predicting their Tuesday one, which
inflates every offline metric and predicts nothing about an A/B test. We cut
on time instead: train on the past, validate on the next week, test on the
week after that -- the same shape as a production retrain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CONFIG, SplitConfig

EVENT_PURCHASE = 2


@dataclass
class TemporalSplit:
    """Event frames plus the per-customer ground truth for each window."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    valid_labels: dict[int, np.ndarray]
    test_labels: dict[int, np.ndarray]
    train_end_day: int
    valid_end_day: int

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "window": name,
                "days": f"{int(df.day.min())}-{int(df.day.max())}",
                "events": len(df),
                "purchases": int((df.event_type == EVENT_PURCHASE).sum()),
                "customers": df.customer_id.nunique(),
                "items": df.item_id.nunique(),
            }
            for name, df in (
                ("train", self.train), ("valid", self.valid), ("test", self.test)
            )
        ])


def _labels(
    events: pd.DataFrame, eligible: set[int] | np.ndarray
) -> dict[int, np.ndarray]:
    """Ground truth = the distinct items a customer purchased in the window."""
    purchases = events[events.event_type == EVENT_PURCHASE]
    purchases = purchases[purchases.customer_id.isin(eligible)]
    grouped = purchases.groupby("customer_id").item_id.unique()
    return {int(c): np.asarray(v, dtype=np.int32) for c, v in grouped.items()}


def make_temporal_split(
    events: pd.DataFrame, cfg: SplitConfig | None = None
) -> TemporalSplit:
    cfg = cfg or CONFIG.split
    max_day = int(events.day.max())
    valid_end = max_day - cfg.test_days
    train_end = valid_end - cfg.valid_days

    train = events[events.day <= train_end]
    valid = events[(events.day > train_end) & (events.day <= valid_end)]
    test = events[events.day > valid_end]

    # A customer is only scorable if we have enough history to build features
    # from. Everyone else is a cold-start case, handled by a separate path in
    # the service rather than being quietly scored and counted as a miss.
    train_counts = train.groupby("customer_id").size()
    eligible = set(
        train_counts[train_counts >= cfg.min_train_events_per_customer].index.tolist()
    )

    return TemporalSplit(
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        test=test.reset_index(drop=True),
        valid_labels=_labels(valid, eligible),
        test_labels=_labels(test, eligible),
        train_end_day=train_end,
        valid_end_day=valid_end,
    )


def load_events() -> pd.DataFrame:
    return pd.read_parquet(CONFIG.data_dir / "events.parquet")


def load_catalog() -> pd.DataFrame:
    return pd.read_parquet(CONFIG.data_dir / "catalog.parquet")


def load_customers() -> pd.DataFrame:
    return pd.read_parquet(CONFIG.data_dir / "customers.parquet")


if __name__ == "__main__":
    split = make_temporal_split(load_events())
    print(split.summary().to_string(index=False))
    print(f"\nscorable customers  valid: {len(split.valid_labels):,}"
          f"  test: {len(split.test_labels):,}")
    sizes = np.array([len(v) for v in split.test_labels.values()])
    print(f"test basket size    mean: {sizes.mean():.2f}  median: {np.median(sizes):.0f}"
          f"  p90: {np.percentile(sizes, 90):.0f}")
