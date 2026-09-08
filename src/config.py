"""Central configuration for the personalization stack.

Everything that is a knob lives here so the offline pipeline, the training
jobs and the online service read the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_lake"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

for _d in (DATA_DIR, ARTIFACT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    """Shape of the simulated retail event stream."""

    n_customers: int = 20_000
    n_items: int = 4_000
    n_categories: int = 40
    n_brands: int = 250
    n_stores: int = 120
    n_days: int = 120
    latent_dim: int = 24
    # Grocery is repeat-heavy: this is the probability a purchase in a session
    # is a re-buy of something the customer already owns.
    repeat_purchase_rate: float = 0.42
    # Zipf exponent shaping the item popularity prior. It drives the discovery
    # channel and the popularity baseline; note the affinity pool consumes a
    # rank-standardised version of it, which is invariant to this exponent.
    popularity_zipf: float = 0.9
    # Relative pull of the three forces deciding what a customer looks at.
    # Taste must dominate popularity or there is no personalization to learn:
    # tuned so the top-100 items take ~25% of pool demand (gini ~0.68) before
    # repeat-purchase amplification, rather than the ~95% an untuned mix gives.
    taste_weight: float = 6.0
    popularity_weight: float = 0.30
    price_weight: float = 1.2
    pool_temperature: float = 1.2
    pool_size: int = 80
    avg_sessions_per_customer: float = 9.0
    seed: int = 20260908


@dataclass(frozen=True)
class SplitConfig:
    """Temporal split. Never random-split interaction data: it leaks."""

    test_days: int = 7
    valid_days: int = 7
    min_train_events_per_customer: int = 3


@dataclass(frozen=True)
class TwoTowerConfig:
    embed_dim: int = 64
    history_len: int = 20
    hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.10
    lr: float = 3e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    epochs: int = 18
    temperature: float = 0.07
    # Sampled-softmax logQ correction (Yi et al., RecSys'19). Without it the
    # in-batch softmax over-penalises head items and retrieval collapses to
    # the long tail.
    logq_correction: bool = True
    seed: int = 7


@dataclass(frozen=True)
class RankerConfig:
    n_candidates: int = 200
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_child_samples: int = 40
    early_stopping_rounds: int = 40
    seed: int = 7


@dataclass(frozen=True)
class EvalConfig:
    k_values: tuple[int, ...] = (5, 10, 20, 50)
    primary_k: int = 20


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    two_tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    data_dir: Path = DATA_DIR
    artifact_dir: Path = ARTIFACT_DIR
    report_dir: Path = REPORT_DIR


CONFIG = Config()
