"""Non-neural baselines.

These exist to make the deep model earn its place. In grocery retail the two
bars that actually matter are:

  * Popularity      -- cheap, robust, and embarrassingly hard to beat on
                       aggregate metrics because demand is genuinely skewed.
  * Repeat purchase -- "show them what they always buy". With a 42% repeat
                       rate this is not a strawman, it is most of the value.

A retrieval model that cannot clear both is not worth deploying, whatever its
architecture. All models share one interface so the evaluation harness and the
serving layer can treat them interchangeably.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

EVENT_VIEW, EVENT_CART, EVENT_PURCHASE = 0, 1, 2

# Implicit-feedback confidence per event type. A purchase is worth far more
# than a view, but a view is not worth zero.
EVENT_WEIGHTS = {EVENT_VIEW: 0.3, EVENT_CART: 1.0, EVENT_PURCHASE: 3.0}


class Recommender(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, catalog: pd.DataFrame) -> "Recommender": ...

    def recommend(self, customer_ids: np.ndarray, k: int) -> dict[int, np.ndarray]: ...


def build_interaction_matrix(
    train: pd.DataFrame,
    n_customers: int,
    n_items: int,
    half_life_days: float = 30.0,
) -> sparse.csr_matrix:
    """Customer x item implicit-confidence matrix with exponential recency decay.

    Recency decay matters more than it looks: a diaper purchase from 90 days
    ago is a different signal from one last week, and a flat count treats them
    the same.
    """
    weights = train.event_type.map(EVENT_WEIGHTS).to_numpy(dtype=np.float32)
    age = train.day.max() - train.day.to_numpy()
    decay = np.exp(-np.log(2.0) * age / half_life_days).astype(np.float32)
    data = weights * decay

    mat = sparse.coo_matrix(
        (data, (train.customer_id.to_numpy(), train.item_id.to_numpy())),
        shape=(n_customers, n_items),
        dtype=np.float32,
    ).tocsr()
    mat.sum_duplicates()
    return mat


def _top_k_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k indices of a dense score vector, highest first."""
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


class PopularityRecommender:
    """Recency-weighted global bestsellers. Identical slate for everyone."""

    name = "popularity"

    def __init__(self, half_life_days: float = 21.0):
        self.half_life_days = half_life_days
        self.ranking_: np.ndarray | None = None
        self.scores_: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, catalog: pd.DataFrame) -> "PopularityRecommender":
        purchases = train[train.event_type == EVENT_PURCHASE]
        age = purchases.day.max() - purchases.day.to_numpy()
        w = np.exp(-np.log(2.0) * age / self.half_life_days)
        scores = np.bincount(
            purchases.item_id.to_numpy(), weights=w, minlength=len(catalog)
        )
        self.scores_ = scores
        self.ranking_ = np.argsort(-scores).astype(np.int32)
        return self

    def recommend(self, customer_ids: np.ndarray, k: int) -> dict[int, np.ndarray]:
        slate = self.ranking_[:k]
        return {int(c): slate for c in customer_ids}


class RepeatPurchaseRecommender:
    """Rank a customer's own history by frequency x recency.

    Backfilled with global popularity so every customer gets a full slate --
    otherwise the model quietly under-serves light shoppers and the headline
    recall hides it.
    """

    name = "repeat"

    def __init__(self, half_life_days: float = 30.0, backfill: bool = True):
        self.half_life_days = half_life_days
        self.backfill = backfill
        self.history_: dict[int, np.ndarray] = {}
        self.popular_: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, catalog: pd.DataFrame) -> "RepeatPurchaseRecommender":
        purchases = train[train.event_type == EVENT_PURCHASE].copy()
        age = purchases.day.max() - purchases.day.to_numpy()
        purchases["w"] = np.exp(-np.log(2.0) * age / self.half_life_days)

        agg = (
            purchases.groupby(["customer_id", "item_id"], sort=False)["w"]
            .sum()
            .reset_index()
            .sort_values(["customer_id", "w"], ascending=[True, False])
        )
        self.history_ = {
            int(c): g.item_id.to_numpy(dtype=np.int32)
            for c, g in agg.groupby("customer_id", sort=False)
        }
        self.popular_ = (
            PopularityRecommender().fit(train, catalog).ranking_
        )
        return self

    def recommend(self, customer_ids: np.ndarray, k: int) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for c in customer_ids:
            c = int(c)
            own = self.history_.get(c, np.empty(0, dtype=np.int32))[:k]
            if self.backfill and len(own) < k:
                filler = self.popular_[~np.isin(self.popular_, own)]
                own = np.concatenate([own, filler[: k - len(own)]])
            out[c] = own.astype(np.int32)
        return out


class ItemKNNRecommender:
    """Item-item cosine similarity over the implicit matrix.

    Scores a customer as the sum of similarities from everything in their
    history. Shrinkage damps the co-occurrence spikes that two-item overlaps
    otherwise produce in the long tail.
    """

    name = "item_knn"

    def __init__(self, top_n_similar: int = 200, shrinkage: float = 25.0):
        self.top_n_similar = top_n_similar
        self.shrinkage = shrinkage
        self.sim_: sparse.csr_matrix | None = None
        self.matrix_: sparse.csr_matrix | None = None
        self.popular_: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, catalog: pd.DataFrame) -> "ItemKNNRecommender":
        n_items = len(catalog)
        n_customers = int(train.customer_id.max()) + 1
        mat = build_interaction_matrix(train, n_customers, n_items)
        self.matrix_ = mat

        co = (mat.T @ mat).tocsr()                       # item x item co-occurrence
        norms = np.sqrt(co.diagonal()) + 1e-8
        co.setdiag(0.0)
        co.eliminate_zeros()

        # Cosine with shrinkage: sim = co / (|i| |j| + shrinkage)
        co = co.tocoo()
        denom = norms[co.row] * norms[co.col] + self.shrinkage
        sim = sparse.csr_matrix(
            (co.data / denom, (co.row, co.col)), shape=(n_items, n_items)
        )

        # Keep only the top-N neighbours per item; the rest is noise and makes
        # the online lookup far heavier than it needs to be.
        self.sim_ = self._prune(sim, self.top_n_similar)
        self.popular_ = PopularityRecommender().fit(train, catalog).ranking_
        return self

    @staticmethod
    def _prune(sim: sparse.csr_matrix, top_n: int) -> sparse.csr_matrix:
        sim = sim.tocsr()
        rows, cols, vals = [], [], []
        indptr, indices, data = sim.indptr, sim.indices, sim.data
        for i in range(sim.shape[0]):
            start, end = indptr[i], indptr[i + 1]
            if end - start > top_n:
                local = np.argpartition(-data[start:end], top_n - 1)[:top_n]
            else:
                local = np.arange(end - start)
            rows.append(np.full(len(local), i, dtype=np.int32))
            cols.append(indices[start:end][local])
            vals.append(data[start:end][local])
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=sim.shape,
        )

    def recommend(self, customer_ids: np.ndarray, k: int) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        rows = self.matrix_[customer_ids]                # (n_batch, n_items)
        scores = (rows @ self.sim_).toarray()
        for i, c in enumerate(customer_ids):
            s = scores[i]
            if not s.any():
                out[int(c)] = self.popular_[:k]
                continue
            out[int(c)] = _top_k_from_scores(s, k).astype(np.int32)
        return out


class ALSRecommender:
    """Implicit-feedback matrix factorization (Hu, Koren & Volinsky 2008).

    Written out rather than imported so the confidence weighting and the
    regularisation are visible and tunable -- and because it is the honest
    linear reference point for the two-tower model.
    """

    name = "als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 12.0,
        iterations: int = 12,
        seed: int = 7,
    ):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None
        self.popular_: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, catalog: pd.DataFrame) -> "ALSRecommender":
        n_items = len(catalog)
        n_customers = int(train.customer_id.max()) + 1
        mat = build_interaction_matrix(train, n_customers, n_items)
        # Confidence c_ui = 1 + alpha * r_ui, preference p_ui = 1[r_ui > 0].
        conf = mat.copy()
        conf.data = self.alpha * conf.data

        rng = np.random.default_rng(self.seed)
        X = rng.normal(0, 0.01, (n_customers, self.factors))
        Y = rng.normal(0, 0.01, (n_items, self.factors))

        conf_csr = conf.tocsr()
        conf_csc_t = conf.T.tocsr()
        eye = self.regularization * np.eye(self.factors)

        # Every solve here is a stack of tiny 64x64 systems. Multi-threaded
        # BLAS spawns a thread team per system and thrashes -- on this box that
        # is ~300x slower than running the same batch on one thread. Pin it.
        with threadpool_limits(limits=1, user_api="blas"):
            for _ in range(self.iterations):
                X = self._solve(conf_csr, Y, eye)
                Y = self._solve(conf_csc_t, X, eye)

        self.user_factors_, self.item_factors_ = X, Y
        self.popular_ = PopularityRecommender().fit(train, catalog).ranking_
        return self

    @staticmethod
    def _batches(order: np.ndarray, nnz: np.ndarray, max_elements: int, max_rows: int):
        """Group nnz-sorted rows so each padded block stays within a budget.

        Batching by row count alone is a trap: item rows range from a handful
        of interactions to several thousand, so the batch containing the head
        items pads every row out to the widest one and allocates gigabytes.
        Budgeting on rows x width instead keeps every block the same size in
        memory regardless of where it falls in the distribution.
        """
        i, n = 0, len(order)
        while i < n:
            j = i
            while (
                j < n
                and (j - i + 1) * int(nnz[order[j]]) <= max_elements
                and (j - i + 1) <= max_rows
            ):
                j += 1
            j = max(j, i + 1)  # always make progress, even on a single wide row
            yield order[i:j]
            i = j

    @staticmethod
    def _solve(
        conf: sparse.csr_matrix,
        Y: np.ndarray,
        eye: np.ndarray,
        max_elements: int = 262_144,
        max_rows: int = 2_048,
    ) -> np.ndarray:
        """One ALS half-step, fully batched.

        Uses the standard YtY decomposition so only the non-zero entries of a
        row enter the correction term. Rows are processed in order of their
        non-zero count and each batch is padded to its own maximum, which keeps
        padding waste small even though item rows range from a handful of
        interactions to several thousand.
        """
        n_rows, n_factors = conf.shape[0], Y.shape[1]
        YtY = Y.T @ Y
        out = np.zeros((n_rows, n_factors))

        indptr, indices, data = conf.indptr, conf.indices, conf.data
        nnz = np.diff(indptr)
        order = np.argsort(nnz, kind="stable")
        order = order[nnz[order] > 0]  # empty rows keep their zero vector

        for rows in ALSRecommender._batches(order, nnz, max_elements, max_rows):
            width = int(nnz[rows].max())

            idx_pad = np.zeros((len(rows), width), dtype=np.int32)
            c_pad = np.zeros((len(rows), width), dtype=np.float64)
            mask = np.zeros((len(rows), width), dtype=np.float64)
            for i, r in enumerate(rows):
                lo, hi = indptr[r], indptr[r + 1]
                n = hi - lo
                idx_pad[i, :n] = indices[lo:hi]
                c_pad[i, :n] = data[lo:hi]
                mask[i, :n] = 1.0

            Yi = Y[idx_pad]                                   # (b, width, f)
            # A = YtY + Yi^T diag(c) Yi + lambda I
            A = np.matmul((Yi * c_pad[:, :, None]).transpose(0, 2, 1), Yi)
            A += YtY + eye
            # b = Yi^T (c + 1), padded slots zeroed by the mask
            rhs = ((c_pad + 1.0) * mask)[:, :, None]
            b = np.matmul(Yi.transpose(0, 2, 1), rhs)[:, :, 0]
            out[rows] = np.linalg.solve(A, b)

        return out

    def recommend(self, customer_ids: np.ndarray, k: int) -> dict[int, np.ndarray]:
        scores = self.user_factors_[customer_ids] @ self.item_factors_.T
        return {
            int(c): _top_k_from_scores(scores[i], k).astype(np.int32)
            for i, c in enumerate(customer_ids)
        }


class BlendRecommender:
    """Reciprocal-rank fusion of several recommenders.

    RRF needs no score calibration between models, which is exactly the
    problem when blending a cosine score with a popularity count.
    """

    def __init__(self, models: list[Recommender], weights: list[float] | None = None,
                 k_rrf: float = 60.0, name: str = "blend"):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.k_rrf = k_rrf
        self.name = name

    def fit(self, train: pd.DataFrame, catalog: pd.DataFrame) -> "BlendRecommender":
        for m in self.models:
            m.fit(train, catalog)
        return self

    def recommend(self, customer_ids: np.ndarray, k: int) -> dict[int, np.ndarray]:
        per_model = [m.recommend(customer_ids, k * 3) for m in self.models]
        out: dict[int, np.ndarray] = {}
        for c in customer_ids:
            c = int(c)
            fused: dict[int, float] = {}
            for w, preds in zip(self.weights, per_model):
                for rank, item in enumerate(preds.get(c, [])):
                    fused[int(item)] = fused.get(int(item), 0.0) + w / (self.k_rrf + rank + 1)
            ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
            out[c] = np.array([i for i, _ in ranked], dtype=np.int32)
        return out
