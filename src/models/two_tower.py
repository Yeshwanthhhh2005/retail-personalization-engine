"""Two-tower neural retrieval.

Architecture
------------
    user tower : [id embedding | mean-pooled history | dense RFM] -> MLP -> z_u
    item tower : [id embedding | category | brand | price bucket] -> MLP -> z_i
    score      : cosine(z_u, z_i) / temperature

Both towers emit L2-normalised vectors, so the item tower can be evaluated
once per catalogue refresh, pushed into a vector index, and the online cost
collapses to one user-tower forward pass plus an ANN lookup. That property --
not accuracy -- is why two-tower is the standard retrieval architecture at
catalogue scale.

Training uses in-batch sampled softmax with the logQ correction from Yi et al.
(RecSys 2019). Without that correction the in-batch negatives are drawn in
proportion to popularity, the loss over-penalises head items, and retrieval
drifts into the long tail -- visible as recall collapsing while coverage looks
suspiciously excellent.

The item id embedding table is shared between the history encoder and the item
tower. Tying them roughly halves the parameters that matter and consistently
trains faster here than keeping two separate tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.config import TwoTowerConfig


@dataclass
class ItemFeatures:
    """Static per-item content features, aligned to item_id order."""

    category: np.ndarray      # (n_items,) int
    brand: np.ndarray         # (n_items,) int
    price_bucket: np.ndarray  # (n_items,) int
    n_categories: int
    n_brands: int
    n_price_buckets: int = 20

    @staticmethod
    def from_catalog(catalog, n_price_buckets: int = 20) -> "ItemFeatures":
        price = catalog.price.to_numpy()
        # Quantile buckets: log-price is still heavily skewed by Electronics.
        edges = np.quantile(price, np.linspace(0, 1, n_price_buckets + 1)[1:-1])
        bucket = np.searchsorted(edges, price)
        return ItemFeatures(
            category=catalog.category_id.to_numpy().astype(np.int64),
            brand=catalog.brand_id.to_numpy().astype(np.int64),
            price_bucket=bucket.astype(np.int64),
            n_categories=int(catalog.category_id.max()) + 1,
            n_brands=int(catalog.brand_id.max()) + 1,
            n_price_buckets=n_price_buckets,
        )


def _mlp(sizes: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        item_features: ItemFeatures,
        n_dense: int,
        cfg: TwoTowerConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_items = n_items
        self.pad_index = n_items

        d = cfg.embed_dim
        side = max(8, d // 4)

        # Shared across the history encoder and the item tower.
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=n_items)
        self.user_emb = nn.Embedding(n_users, d)
        self.category_emb = nn.Embedding(item_features.n_categories, side)
        self.brand_emb = nn.Embedding(item_features.n_brands, side)
        self.price_emb = nn.Embedding(item_features.n_price_buckets, side)

        self.register_buffer("item_category", torch.as_tensor(item_features.category))
        self.register_buffer("item_brand", torch.as_tensor(item_features.brand))
        self.register_buffer("item_price_bucket", torch.as_tensor(item_features.price_bucket))

        self.dense_norm = nn.LayerNorm(n_dense)
        self.user_tower = _mlp([d + d + n_dense, *cfg.hidden], cfg.dropout)
        self.item_tower = _mlp([d + 3 * side, *cfg.hidden], cfg.dropout)

        for emb in (self.item_emb, self.user_emb, self.category_emb,
                    self.brand_emb, self.price_emb):
            nn.init.normal_(emb.weight, std=0.05)
        with torch.no_grad():
            self.item_emb.weight[self.pad_index].zero_()

    # --- towers ---------------------------------------------------------
    def encode_user(
        self, user: torch.Tensor, history: torch.Tensor,
        history_len: torch.Tensor, dense: torch.Tensor,
    ) -> torch.Tensor:
        h = self.item_emb(history)                                # (B, L, d)
        mask = (history != self.pad_index).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)    # masked mean
        x = torch.cat([self.user_emb(user), pooled, self.dense_norm(dense)], dim=-1)
        return F.normalize(self.user_tower(x), dim=-1)

    def encode_item(self, item: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            self.item_emb(item),
            self.category_emb(self.item_category[item]),
            self.brand_emb(self.item_brand[item]),
            self.price_emb(self.item_price_bucket[item]),
        ], dim=-1)
        return F.normalize(self.item_tower(x), dim=-1)

    @torch.no_grad()
    def encode_all_items(self, batch: int = 2048) -> torch.Tensor:
        self.eval()
        out = []
        for start in range(0, self.n_items, batch):
            ids = torch.arange(start, min(start + batch, self.n_items),
                               device=self.item_emb.weight.device)
            out.append(self.encode_item(ids))
        return torch.cat(out)

    # --- loss -----------------------------------------------------------
    def in_batch_loss(
        self,
        user_vec: torch.Tensor,
        target: torch.Tensor,
        log_q: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sampled softmax over the other targets in the batch as negatives."""
        item_vec = self.encode_item(target)
        logits = (user_vec @ item_vec.T) / self.cfg.temperature

        if log_q is not None and self.cfg.logq_correction:
            # Correct for sampling negatives in proportion to popularity.
            logits = logits - log_q[target].unsqueeze(0)

        # Accidental hits: the same item appearing twice in a batch would
        # otherwise be trained as a negative against itself.
        same = target.unsqueeze(0) == target.unsqueeze(1)
        eye = torch.eye(len(target), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(same & ~eye, float("-inf"))

        labels = torch.arange(len(target), device=logits.device)
        return F.cross_entropy(logits, labels)


@dataclass
class TrainState:
    history: list[dict[str, float]] = field(default_factory=list)

    def log(self, **kw: float) -> None:
        self.history.append(kw)
        parts = "  ".join(
            f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in kw.items()
        )
        print(f"    {parts}", flush=True)
