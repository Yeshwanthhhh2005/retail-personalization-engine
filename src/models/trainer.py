"""Training loop for the two-tower retrieval model."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.config import TwoTowerConfig
from src.features.sequences import SequenceData
from src.models.two_tower import ItemFeatures, TwoTowerModel


@dataclass
class Batcher:
    """Index shuffler. The arrays are small enough to keep resident in RAM,
    so a DataLoader would add worker overhead and buy nothing here."""

    n: int
    batch_size: int
    seed: int = 0

    def epoch(self, epoch: int) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(self.n)
        return [
            order[i : i + self.batch_size]
            for i in range(0, self.n, self.batch_size)
        ]


def compute_log_q(targets: np.ndarray, n_items: int) -> torch.Tensor:
    """log P(item is sampled as an in-batch negative) ~ log of its frequency."""
    counts = np.bincount(targets, minlength=n_items).astype(np.float64)
    probs = np.clip(counts / counts.sum(), 1e-10, None)
    return torch.as_tensor(np.log(probs), dtype=torch.float32)


def train_two_tower(
    sequences: SequenceData,
    n_users: int,
    n_items: int,
    item_features: ItemFeatures,
    cfg: TwoTowerConfig,
    validate_fn=None,
    device: str = "cpu",
) -> tuple[TwoTowerModel, list[dict]]:
    """Fit the model, optionally scoring retrieval recall after each epoch.

    `validate_fn(model) -> dict` is called at the end of every epoch; the model
    with the best `recall@20` is kept, so a late over-fitting epoch cannot
    silently become the artifact we ship.
    """
    torch.manual_seed(cfg.seed)
    model = TwoTowerModel(
        n_users=n_users, n_items=n_items, item_features=item_features,
        n_dense=sequences.dense.shape[1], cfg=cfg,
    ).to(device)

    user = torch.as_tensor(sequences.user.astype(np.int64), device=device)
    target = torch.as_tensor(sequences.target.astype(np.int64), device=device)
    history = torch.as_tensor(sequences.history.astype(np.int64), device=device)
    hlen = torch.as_tensor(sequences.history_len.astype(np.int64), device=device)
    dense = torch.as_tensor(sequences.dense, device=device)
    log_q = compute_log_q(sequences.target, n_items).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    batcher = Batcher(len(sequences), cfg.batch_size, cfg.seed)
    total_steps = cfg.epochs * ((len(sequences) + cfg.batch_size - 1) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=0.25
    )

    history_log: list[dict] = []
    best_state, best_metric = None, -np.inf

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        losses = []
        for idx in batcher.epoch(epoch):
            batch = torch.as_tensor(idx, device=device)
            user_vec = model.encode_user(
                user[batch], history[batch], hlen[batch], dense[batch]
            )
            loss = model.in_batch_loss(user_vec, target[batch], log_q)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.detach().item())

        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "secs": round(time.perf_counter() - t0, 1),
        }
        if validate_fn is not None:
            record.update(validate_fn(model))
            metric = record.get("val_recall@20", -np.inf)
            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                record["best"] = True
        history_log.append(record)
        print("    " + "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in record.items()
        ), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history_log


@torch.no_grad()
def encode_users(
    model: TwoTowerModel,
    user_ids: np.ndarray,
    history: np.ndarray,
    history_len: np.ndarray,
    dense: np.ndarray,
    batch: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(user_ids), batch):
        sl = slice(start, start + batch)
        out.append(model.encode_user(
            torch.as_tensor(user_ids[sl].astype(np.int64), device=device),
            torch.as_tensor(history[sl].astype(np.int64), device=device),
            torch.as_tensor(history_len[sl].astype(np.int64), device=device),
            torch.as_tensor(dense[sl], device=device),
        ).cpu().numpy())
    return np.concatenate(out)
