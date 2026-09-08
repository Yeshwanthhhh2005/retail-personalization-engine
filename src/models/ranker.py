"""Second-stage ranker: LightGBM LambdaRank over pooled candidates.

Why a GBDT rather than another neural net: the ranking stage consumes dozens
of heterogeneous, differently-scaled tabular features (counts, ratios, days,
model scores). Trees handle that without normalisation, train in seconds on
CPU, and -- the part that matters in a business review -- produce feature
importances and monotone-ish behaviour that can be explained to a merchant.

LambdaRank optimises the ordering within each customer's candidate list
directly, which is the objective we actually care about, rather than a
pointwise probability that then has to be assumed monotone in relevance.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import RankerConfig


@dataclass
class RankerArtifacts:
    model: lgb.Booster
    feature_names: list[str]
    best_iteration: int

    def importance(self, kind: str = "gain") -> pd.DataFrame:
        return (
            pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importance(kind),
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def _groups(frame: pd.DataFrame) -> np.ndarray:
    """Candidate counts per customer, in the frame's existing row order."""
    customer = frame.customer_id.to_numpy()
    if np.any(customer[1:] < customer[:-1]):
        raise ValueError("frame must be sorted by customer_id for group ranking")
    _, counts = np.unique(customer, return_counts=True)
    return counts


def train_ranker(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    feature_names: list[str],
    cfg: RankerConfig,
) -> RankerArtifacts:
    features = [f for f in feature_names if f in train_frame.columns]

    train_set = lgb.Dataset(
        train_frame[features].astype(np.float32),
        label=train_frame["label"].to_numpy(),
        group=_groups(train_frame),
        feature_name=features,
        free_raw_data=False,
    )
    valid_sets, valid_names = [train_set], ["train"]
    if valid_frame is not None and len(valid_frame):
        valid_sets.append(lgb.Dataset(
            valid_frame[features].astype(np.float32),
            label=valid_frame["label"].to_numpy(),
            group=_groups(valid_frame),
            feature_name=features,
            reference=train_set,
            free_raw_data=False,
        ))
        valid_names.append("valid")

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20],
        "lambdarank_truncation_level": 30,
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_child_samples": cfg.min_child_samples,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "seed": cfg.seed,
        "verbosity": -1,
        "num_threads": 4,
    }

    callbacks = [lgb.log_evaluation(period=50)]
    if len(valid_sets) > 1:
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params, train_set, num_boost_round=cfg.n_estimators,
        valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
    )
    return RankerArtifacts(
        model=booster, feature_names=features,
        best_iteration=booster.best_iteration or cfg.n_estimators,
    )


def rank_candidates(
    artifacts: RankerArtifacts, frame: pd.DataFrame, k: int
) -> dict[int, np.ndarray]:
    """Score every candidate, then take each customer's top-k."""
    scores = artifacts.model.predict(
        frame[artifacts.feature_names].astype(np.float32),
        num_iteration=artifacts.best_iteration,
    )
    scored = pd.DataFrame({
        "customer_id": frame.customer_id.to_numpy(),
        "item_id": frame.item_id.to_numpy(),
        "score": scores,
    })
    scored = scored.sort_values(["customer_id", "score"], ascending=[True, False])
    return {
        int(c): g.item_id.to_numpy(dtype=np.int32)[:k]
        for c, g in scored.groupby("customer_id", sort=False)
    }
