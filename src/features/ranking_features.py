"""Candidate generation and feature assembly for the ranking stage.

Retrieval decides *what could be relevant*; ranking decides *what to show*.
Splitting them is not ceremony -- it is what lets a 4k (or 40M) item catalogue
be scored in milliseconds, and it is what lets the repeat-purchase signal and
the discovery signal compete on a common scale instead of one drowning the
other.

Candidates come from several sources on purpose. Each has a blind spot:

  repeat      strong, but can only ever re-sell the known basket
  two_tower   generalises to unseen items, weaker on exact re-buys
  item_knn    good on complements, noisy in the tail
  popularity  the safety net that guarantees a full slate

The union is what the ranker sees. Feature construction is strictly causal:
everything is computed from events at or before `as_of_day`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EVENT_VIEW, EVENT_CART, EVENT_PURCHASE = 0, 1, 2

SOURCES = ("repeat", "two_tower", "item_knn", "popularity")


@dataclass
class CandidateSet:
    """Long-format candidates: one row per (customer, item) pair."""

    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)


def build_candidates(
    customers: np.ndarray,
    sources: dict[str, dict[int, np.ndarray]],
    scores: dict[str, dict[int, np.ndarray]] | None = None,
    max_per_source: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Union per-source ranked lists into one candidate frame.

    Keeps each source's rank and score as features -- "this item was #1 for
    the repeat model but #180 for the two-tower" is exactly the kind of
    disagreement the ranker should be able to exploit.
    """
    max_per_source = max_per_source or {}
    rows_c, rows_i = [], []
    src_rank = {s: [] for s in sources}
    src_score = {s: [] for s in sources}

    for c in customers:
        c = int(c)
        pooled: dict[int, dict[str, tuple[int, float]]] = {}
        for name, preds in sources.items():
            cand = preds.get(c, np.empty(0, dtype=np.int64))
            limit = max_per_source.get(name, len(cand))
            cand = cand[:limit]
            sc = None
            if scores and name in scores:
                sc = scores[name].get(c)
            for rank, item in enumerate(cand):
                entry = pooled.setdefault(int(item), {})
                entry[name] = (rank, float(sc[rank]) if sc is not None else 0.0)

        for item, entry in pooled.items():
            rows_c.append(c)
            rows_i.append(item)
            for name in sources:
                rank, score = entry.get(name, (-1, 0.0))
                src_rank[name].append(rank)
                src_score[name].append(score)

    frame = pd.DataFrame({
        "customer_id": np.asarray(rows_c, dtype=np.int32),
        "item_id": np.asarray(rows_i, dtype=np.int32),
    })
    for name in sources:
        frame[f"rank_{name}"] = np.asarray(src_rank[name], dtype=np.int16)
        frame[f"score_{name}"] = np.asarray(src_score[name], dtype=np.float32)
        frame[f"in_{name}"] = (frame[f"rank_{name}"] >= 0).astype(np.int8)
        # Reciprocal rank is a better-behaved feature than raw rank: the
        # difference between rank 1 and 2 matters, 180 vs 181 does not.
        frame[f"rr_{name}"] = np.where(
            frame[f"rank_{name}"] >= 0, 1.0 / (1.0 + frame[f"rank_{name}"]), 0.0
        ).astype(np.float32)
    frame["n_sources"] = frame[[f"in_{s}" for s in sources]].sum(axis=1).astype(np.int8)
    return frame


def _user_item_history(history: pd.DataFrame, as_of_day: int) -> pd.DataFrame:
    """Per (customer, item): counts by event type, recency, repurchase cadence."""
    # Indicator columns rather than lambdas inside agg: a per-group Python
    # call over a million rows is the difference between seconds and minutes.
    work = history[["customer_id", "item_id", "day", "quantity"]].copy()
    etype = history.event_type.to_numpy()
    work["_purchase"] = (etype == EVENT_PURCHASE).astype(np.int32)
    work["_cart"] = (etype == EVENT_CART).astype(np.int32)
    work["_view"] = (etype == EVENT_VIEW).astype(np.int32)

    agg = work.groupby(["customer_id", "item_id"], sort=False).agg(
        ui_n_events=("day", "size"),
        ui_last_day=("day", "max"),
        ui_first_day=("day", "min"),
        ui_n_purchase=("_purchase", "sum"),
        ui_n_cart=("_cart", "sum"),
        ui_n_view=("_view", "sum"),
        ui_qty=("quantity", "sum"),
    ).reset_index()

    agg["ui_days_since"] = as_of_day - agg["ui_last_day"]
    span = (agg["ui_last_day"] - agg["ui_first_day"]).clip(lower=0)
    # Mean gap between purchases of this item by this customer. The ratio of
    # elapsed time to that cadence is the replenishment signal: "they buy milk
    # every 6 days and it has been 7" is the single most actionable feature in
    # a grocery basket.
    agg["ui_mean_gap"] = np.where(
        agg["ui_n_purchase"] > 1, span / agg["ui_n_purchase"].clip(lower=1), np.nan
    )
    agg["ui_due_ratio"] = agg["ui_days_since"] / agg["ui_mean_gap"].replace(0, np.nan)
    agg["ui_cart_rate"] = agg["ui_n_cart"] / agg["ui_n_events"].clip(lower=1)
    agg["ui_purchase_rate"] = agg["ui_n_purchase"] / agg["ui_n_events"].clip(lower=1)
    return agg.drop(columns=["ui_first_day"])


def _user_profile(history: pd.DataFrame, catalog: pd.DataFrame, as_of_day: int) -> pd.DataFrame:
    purchases = history[history.event_type == EVENT_PURCHASE].merge(
        catalog[["item_id", "price", "category_id"]], on="item_id", how="left"
    )
    prof = purchases.groupby("customer_id").agg(
        u_n_purchase=("item_id", "size"),
        u_n_distinct=("item_id", "nunique"),
        u_mean_price=("price", "mean"),
        u_max_price=("price", "max"),
        u_last_day=("day", "max"),
        u_first_day=("day", "min"),
        u_n_categories=("category_id", "nunique"),
    ).reset_index()
    prof["u_days_since"] = as_of_day - prof["u_last_day"]
    prof["u_active_days"] = (prof["u_last_day"] - prof["u_first_day"]).clip(lower=1)
    prof["u_freq"] = prof["u_n_purchase"] / prof["u_active_days"]
    prof["u_repeat_ratio"] = 1.0 - prof["u_n_distinct"] / prof["u_n_purchase"].clip(lower=1)
    return prof.drop(columns=["u_first_day", "u_last_day"])


def _user_category_affinity(history: pd.DataFrame, catalog: pd.DataFrame,
                            as_of_day: int) -> pd.DataFrame:
    purchases = history[history.event_type == EVENT_PURCHASE].merge(
        catalog[["item_id", "category_id"]], on="item_id", how="left"
    )
    agg = purchases.groupby(["customer_id", "category_id"]).agg(
        uc_n=("item_id", "size"),
        uc_last_day=("day", "max"),
    ).reset_index()
    total = agg.groupby("customer_id").uc_n.transform("sum")
    agg["uc_share"] = agg["uc_n"] / total
    agg["uc_days_since"] = as_of_day - agg["uc_last_day"]
    return agg.drop(columns=["uc_last_day"])


def _user_brand_affinity(history: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    purchases = history[history.event_type == EVENT_PURCHASE].merge(
        catalog[["item_id", "brand_id"]], on="item_id", how="left"
    )
    agg = purchases.groupby(["customer_id", "brand_id"]).size().reset_index(name="ub_n")
    total = agg.groupby("customer_id").ub_n.transform("sum")
    agg["ub_share"] = agg["ub_n"] / total
    return agg


def _item_profile(history: pd.DataFrame, catalog: pd.DataFrame,
                  as_of_day: int, half_life: float = 21.0) -> pd.DataFrame:
    purchases = history[history.event_type == EVENT_PURCHASE]
    w = np.exp(-np.log(2.0) * (as_of_day - purchases.day.to_numpy()) / half_life)
    pop = np.bincount(
        purchases.item_id.to_numpy(), weights=w, minlength=len(catalog)
    )
    n_buyers = purchases.groupby("item_id").customer_id.nunique()
    n_events = purchases.groupby("item_id").size()

    prof = catalog[["item_id", "price", "category_id", "brand_id", "is_replenishable"]].copy()
    prof["i_pop"] = np.log1p(pop)
    prof["i_n_buyers"] = np.log1p(prof.item_id.map(n_buyers).fillna(0.0))
    prof["i_repeat_intensity"] = (
        prof.item_id.map(n_events).fillna(0.0) / prof.item_id.map(n_buyers).fillna(1.0).clip(lower=1)
    )
    prof["i_log_price"] = np.log1p(prof["price"])
    prof["is_replenishable"] = prof["is_replenishable"].astype(np.int8)
    return prof


FEATURE_COLUMNS = [
    # candidate-source signals
    *[f"rr_{s}" for s in SOURCES],
    *[f"in_{s}" for s in SOURCES],
    *[f"score_{s}" for s in SOURCES],
    "n_sources",
    # user x item
    "ui_n_purchase", "ui_n_cart", "ui_n_view", "ui_qty", "ui_days_since",
    "ui_mean_gap", "ui_due_ratio", "ui_cart_rate", "ui_purchase_rate",
    # user
    "u_n_purchase", "u_n_distinct", "u_mean_price", "u_days_since",
    "u_freq", "u_repeat_ratio", "u_n_categories",
    # user x category / brand
    "uc_n", "uc_share", "uc_days_since", "ub_n", "ub_share",
    # item
    "i_pop", "i_n_buyers", "i_repeat_intensity", "i_log_price", "is_replenishable",
    # interactions
    "price_ratio", "is_known_item", "cat_due_ratio",
]


@dataclass
class FeatureStore:
    """Precomputed aggregates, joined onto candidates on demand.

    This split is what keeps training and serving honest. The aggregates are
    expensive and static between retrains, so they are computed once here; the
    per-request path is only a set of small joins. Crucially, the offline
    pipeline and the online service call the *same* `transform`, so a feature
    cannot be defined one way in training and another way in serving -- the
    most expensive class of bug in a production recommender, and one that
    offline metrics never reveal.
    """

    item: pd.DataFrame
    user_item: pd.DataFrame
    user: pd.DataFrame
    user_category: pd.DataFrame
    user_brand: pd.DataFrame
    as_of_day: int

    def __post_init__(self) -> None:
        # Pre-index every block on its join keys. A `merge` of a 200-row
        # request against the 700k-row user-item table rebuilds the hash side
        # every call; an indexed `reindex` is a lookup of exactly 200 keys.
        # Online this was the difference between 133 ms and ~4 ms per request.
        self._item = self.item.set_index("item_id")
        self._user_item = self.user_item.set_index(["customer_id", "item_id"])
        self._user = self.user.set_index("customer_id")
        self._user_category = self.user_category.set_index(
            ["customer_id", "category_id"]
        )
        self._user_brand = self.user_brand.set_index(["customer_id", "brand_id"])

    @classmethod
    def build(
        cls, history: pd.DataFrame, catalog: pd.DataFrame, as_of_day: int
    ) -> "FeatureStore":
        assert int(history.day.max()) <= as_of_day, (
            "feature history leaks past as_of_day"
        )
        return cls(
            item=_item_profile(history, catalog, as_of_day),
            user_item=_user_item_history(history, as_of_day),
            user=_user_profile(history, catalog, as_of_day),
            user_category=_user_category_affinity(history, catalog, as_of_day),
            user_brand=_user_brand_affinity(history, catalog),
            as_of_day=as_of_day,
        )

    @staticmethod
    def _attach(frame: pd.DataFrame, block: pd.DataFrame, keys) -> pd.DataFrame:
        """Left-join `block` onto `frame` by positional reindex."""
        aligned = block.reindex(keys)
        for col in block.columns:
            frame[col] = aligned[col].to_numpy()
        return frame

    def transform(self, candidates: pd.DataFrame) -> pd.DataFrame:
        frame = candidates.reset_index(drop=True).copy()
        customers = frame.customer_id.to_numpy()
        items = frame.item_id.to_numpy()

        # The item block supplies category_id and brand_id, so it must land
        # before the affinity blocks that key on them.
        frame = self._attach(frame, self._item, pd.Index(items))
        frame = self._attach(
            frame, self._user_item, pd.MultiIndex.from_arrays([customers, items])
        )
        frame = self._attach(frame, self._user, pd.Index(customers))
        frame = self._attach(
            frame, self._user_category,
            pd.MultiIndex.from_arrays([customers, frame.category_id.to_numpy()]),
        )
        frame = self._attach(
            frame, self._user_brand,
            pd.MultiIndex.from_arrays([customers, frame.brand_id.to_numpy()]),
        )
        return _finalise(frame)


def build_feature_frame(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    catalog: pd.DataFrame,
    as_of_day: int,
) -> pd.DataFrame:
    """Join every feature block onto the candidate frame.

    `history` must contain only events at or before `as_of_day`; the caller
    owns that guarantee and the pipeline asserts it.
    """
    return FeatureStore.build(history, catalog, as_of_day).transform(candidates)


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Derived interactions and the null policy, shared by both paths."""
    frame["is_known_item"] = (frame["ui_n_purchase"].fillna(0) > 0).astype(np.int8)
    frame["price_ratio"] = frame["price"] / frame["u_mean_price"].replace(0, np.nan)
    # How overdue is the whole category, not just this item -- catches
    # substitution (they buy *a* yoghurt weekly, not always the same one).
    frame["cat_due_ratio"] = frame["uc_days_since"] / frame["uc_n"].clip(lower=1)

    counts = ["ui_n_purchase", "ui_n_cart", "ui_n_view", "ui_qty",
              "uc_n", "uc_share", "ub_n", "ub_share"]
    frame[counts] = frame[counts].fillna(0.0)
    # Recency of something never seen is "infinitely long ago", not zero --
    # filling these with 0 would tell the model the opposite of the truth.
    for col in ("ui_days_since", "uc_days_since"):
        frame[col] = frame[col].fillna(9_999.0)
    frame["ui_due_ratio"] = frame["ui_due_ratio"].fillna(0.0)
    frame["ui_mean_gap"] = frame["ui_mean_gap"].fillna(-1.0)

    return frame


def attach_labels(
    frame: pd.DataFrame, labels: dict[int, np.ndarray]
) -> pd.DataFrame:
    """label = 1 if the customer actually purchased the item in the window."""
    truth = {
        (c, int(i)) for c, items in labels.items() for i in items
    }
    keys = list(zip(frame.customer_id.tolist(), frame.item_id.tolist()))
    frame = frame.copy()
    frame["label"] = np.fromiter((k in truth for k in keys), dtype=np.int8,
                                 count=len(keys))
    return frame
