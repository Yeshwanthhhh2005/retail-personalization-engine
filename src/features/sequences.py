"""Causal training sequences for the two-tower retrieval model.

The single most common way to get a recommender wrong is to build a user
feature from data that includes the label. Everything here is constructed
strictly from events *before* the target event, per customer, in event order.
The result is a set of (user, history, target) triples plus the dense
recency/frequency/monetary features as they stood at that moment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EVENT_PURCHASE = 2


@dataclass
class SequenceData:
    """Arrays ready for batching. `pad_index` == n_items marks empty slots."""

    user: np.ndarray          # (N,)   int32
    target: np.ndarray        # (N,)   int32
    history: np.ndarray       # (N, L) int32, left-padded with pad_index
    history_len: np.ndarray   # (N,)   int32
    dense: np.ndarray         # (N, D) float32
    day: np.ndarray           # (N,)   int16
    pad_index: int
    dense_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.user)


DENSE_NAMES = (
    "log_n_prior",        # how much history we have
    "days_since_last",    # recency
    "log_mean_price",     # basket price level so far
    "log_days_active",    # tenure within the window
    "distinct_ratio",     # exploration: distinct items / total purchases
)


def build_sequences(
    events: pd.DataFrame,
    n_items: int,
    history_len: int = 20,
    min_prior: int = 1,
    item_price: np.ndarray | None = None,
) -> SequenceData:
    """One training row per purchase that has at least `min_prior` predecessors."""
    purchases = events[events.event_type == EVENT_PURCHASE]
    purchases = purchases.sort_values(["customer_id", "day", "event_seq"], kind="stable")

    cust = purchases.customer_id.to_numpy(dtype=np.int32)
    item = purchases.item_id.to_numpy(dtype=np.int32)
    day = purchases.day.to_numpy(dtype=np.int32)
    price = (
        item_price[item] if item_price is not None else np.ones(len(item), dtype=np.float32)
    )

    pad = n_items
    # Customer block boundaries: the data is sorted, so a change in id starts one.
    starts = np.flatnonzero(np.r_[True, cust[1:] != cust[:-1]])
    ends = np.r_[starts[1:], len(cust)]

    users, targets, hists, hlens, denses, days = [], [], [], [], [], []

    for start, end in zip(starts, ends):
        n = end - start
        if n <= min_prior:
            continue
        c_items = item[start:end]
        c_days = day[start:end]
        c_price = price[start:end]

        first_day = c_days[0]
        seen: set[int] = set(c_items[:min_prior].tolist())
        price_sum = float(c_price[:min_prior].sum())

        for t in range(min_prior, n):
            past = c_items[max(0, t - history_len) : t]
            h = np.full(history_len, pad, dtype=np.int32)
            h[history_len - len(past) :] = past          # left-pad, recent last

            users.append(c_items.dtype.type(0))          # placeholder, filled below
            targets.append(c_items[t])
            hists.append(h)
            hlens.append(len(past))
            days.append(c_days[t])
            denses.append((
                np.log1p(t),
                float(c_days[t] - c_days[t - 1]),
                np.log1p(price_sum / t),
                np.log1p(float(c_days[t] - first_day)),
                len(seen) / t,
            ))
            seen.add(int(c_items[t]))
            price_sum += float(c_price[t])

        # Fill the user id for every row this customer produced.
        n_rows = n - min_prior
        users[-n_rows:] = [cust[start]] * n_rows

    return SequenceData(
        user=np.asarray(users, dtype=np.int32),
        target=np.asarray(targets, dtype=np.int32),
        history=np.asarray(hists, dtype=np.int32),
        history_len=np.asarray(hlens, dtype=np.int32),
        dense=np.asarray(denses, dtype=np.float32),
        day=np.asarray(days, dtype=np.int16),
        pad_index=pad,
        dense_names=DENSE_NAMES,
    )


def build_inference_state(
    events: pd.DataFrame,
    n_items: int,
    customer_ids: np.ndarray,
    history_len: int = 20,
    item_price: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The user-tower inputs as of the end of `events`, for scoring.

    Mirrors `build_sequences` exactly but emits one row per customer using
    their full trailing history -- this is what the online service computes.
    """
    purchases = events[events.event_type == EVENT_PURCHASE]
    purchases = purchases.sort_values(["customer_id", "day", "event_seq"], kind="stable")
    last_day = int(events.day.max())

    price = item_price if item_price is not None else np.ones(n_items, dtype=np.float32)
    pad = n_items

    history = np.full((len(customer_ids), history_len), pad, dtype=np.int32)
    hlen = np.zeros(len(customer_ids), dtype=np.int32)
    dense = np.zeros((len(customer_ids), len(DENSE_NAMES)), dtype=np.float32)

    grouped_items = purchases.groupby("customer_id").item_id.apply(np.asarray)
    grouped_days = purchases.groupby("customer_id").day.apply(np.asarray)

    for i, c in enumerate(customer_ids):
        c = int(c)
        if c not in grouped_items.index:
            continue
        items = grouped_items.loc[c]
        days = grouped_days.loc[c]
        past = items[-history_len:]
        history[i, history_len - len(past) :] = past
        hlen[i] = len(past)

        n = len(items)
        dense[i] = (
            np.log1p(n),
            float(last_day - days[-1]),
            np.log1p(float(price[items].mean())),
            np.log1p(float(days[-1] - days[0])),
            len(np.unique(items)) / n,
        )
    return history, hlen, dense
