"""Vector index over the item embeddings.

The two-tower model only pays off if the item side is precomputed and served
from an ANN index: encode the catalogue once per refresh, and the online cost
per request drops to one user-tower forward pass plus a sub-millisecond
lookup. This wraps FAISS with the flat/IVF choice made explicit, because at
4k items flat is exact and free, while the same code path at 40M items needs
IVF-PQ and a recall-vs-latency decision someone has to own.

The same index backs the GenAI retrieval layer -- one embedding store, two
consumers -- which is the point of keeping it behind a small interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class IndexStats:
    n_vectors: int
    dim: int
    kind: str
    build_secs: float


class ItemVectorIndex:
    """Inner-product index over L2-normalised vectors (== cosine similarity)."""

    def __init__(self, dim: int, use_ivf: bool = False, nlist: int = 256, nprobe: int = 16):
        self.dim = dim
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.nprobe = nprobe
        self.index: faiss.Index | None = None
        self.item_ids: np.ndarray | None = None
        self.vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray, item_ids: np.ndarray | None = None) -> IndexStats:
        import time

        t0 = time.perf_counter()
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        # Normalise defensively: the towers already do it, but a caller passing
        # raw factors would otherwise get inner-product ranking that silently
        # rewards long vectors.
        faiss.normalize_L2(vectors)

        n = len(vectors)
        # IVF needs enough vectors per centroid to train sensibly; below that
        # a flat index is both exact and faster.
        if self.use_ivf and n >= self.nlist * 39:
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist,
                                       faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
            index.nprobe = self.nprobe
            kind = f"ivf{self.nlist}/nprobe{self.nprobe}"
        else:
            index = faiss.IndexFlatIP(self.dim)
            kind = "flat"

        index.add(vectors)
        self.index = index
        self.vectors = vectors
        self.item_ids = (
            np.arange(n, dtype=np.int64) if item_ids is None
            else np.asarray(item_ids, dtype=np.int64)
        )
        return IndexStats(n, self.dim, kind, round(time.perf_counter() - t0, 2))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (item_ids, scores), both (n_queries, k)."""
        if self.index is None:
            raise RuntimeError("index not built")
        q = np.ascontiguousarray(queries.astype(np.float32))
        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, k)
        return self.item_ids[idx], scores

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        np.save(path.with_suffix(".ids.npy"), self.item_ids)
        np.save(path.with_suffix(".vecs.npy"), self.vectors)

    @classmethod
    def load(cls, path: Path) -> "ItemVectorIndex":
        path = Path(path)
        obj = cls(dim=1)
        obj.index = faiss.read_index(str(path.with_suffix(".faiss")))
        obj.item_ids = np.load(path.with_suffix(".ids.npy"))
        obj.vectors = np.load(path.with_suffix(".vecs.npy"))
        obj.dim = obj.vectors.shape[1]
        return obj


def recall_against_exact(
    index: ItemVectorIndex, queries: np.ndarray, k: int, sample: int = 500
) -> float:
    """ANN recall vs. brute force -- the number to watch when moving to IVF.

    Flat indexes return 1.0 by construction; the check earns its keep the day
    someone swaps in IVF-PQ to fit the catalogue in memory.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(queries), size=min(sample, len(queries)), replace=False)
    q = np.ascontiguousarray(queries[idx].astype(np.float32))
    faiss.normalize_L2(q)

    approx, _ = index.search(q, k)
    exact_scores = q @ index.vectors.T
    exact = np.argpartition(-exact_scores, k - 1, axis=1)[:, :k]
    exact = index.item_ids[exact]

    return float(np.mean([
        len(set(a.tolist()) & set(e.tolist())) / k for a, e in zip(approx, exact)
    ]))
