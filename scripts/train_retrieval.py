"""Train the two-tower retrieval model and score it against the baselines."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from src.config import CONFIG
from src.data.splits import load_catalog, load_events, make_temporal_split
from src.evaluation.metrics import evaluate_ranking, format_leaderboard, leaderboard
from src.features.sequences import build_inference_state, build_sequences
from src.models.trainer import encode_users, train_two_tower
from src.models.two_tower import ItemFeatures
from src.retrieval.vector_index import ItemVectorIndex, recall_against_exact


def main() -> None:
    cfg = CONFIG
    events, catalog = load_events(), load_catalog()
    split = make_temporal_split(events)
    n_items = len(catalog)
    n_users = int(events.customer_id.max()) + 1
    price = catalog.price.to_numpy()

    print("building causal sequences ...", flush=True)
    t0 = time.perf_counter()
    sequences = build_sequences(
        split.train, n_items=n_items,
        history_len=cfg.two_tower.history_len, item_price=price,
    )
    print(f"  {len(sequences):,} training rows in {time.perf_counter() - t0:.1f}s")
    print(f"  mean history length {sequences.history_len.mean():.1f}")

    item_features = ItemFeatures.from_catalog(catalog)

    # Validation users are scored from train-only history: exactly what the
    # service would know at prediction time.
    valid_customers = np.array(sorted(split.valid_labels.keys()), dtype=np.int64)
    v_hist, v_hlen, v_dense = build_inference_state(
        split.train, n_items, valid_customers,
        history_len=cfg.two_tower.history_len, item_price=price,
    )

    def validate(model) -> dict:
        item_vecs = model.encode_all_items().cpu().numpy()
        user_vecs = encode_users(model, valid_customers, v_hist, v_hlen, v_dense)
        scores = user_vecs @ item_vecs.T
        top = np.argpartition(-scores, 20, axis=1)[:, :20]
        row = np.arange(len(top))[:, None]
        top = top[row, np.argsort(-scores[row, top], axis=1)]
        preds = {int(c): top[i] for i, c in enumerate(valid_customers)}
        res = evaluate_ranking(
            preds, split.valid_labels, k_values=(20,), name="two_tower"
        )
        return {"val_recall@20": res.metrics["recall@20"],
                "val_ndcg@20": res.metrics["ndcg@20"]}

    print("\ntraining two-tower ...", flush=True)
    model, log = train_two_tower(
        sequences, n_users=n_users, n_items=n_items,
        item_features=item_features, cfg=cfg.two_tower, validate_fn=validate,
    )

    # --- build the serving-shaped artifacts ------------------------------
    print("\nbuilding vector index ...", flush=True)
    item_vecs = model.encode_all_items().cpu().numpy()
    index = ItemVectorIndex(dim=item_vecs.shape[1], use_ivf=False)
    stats = index.build(item_vecs)
    print(f"  {stats.kind} index, {stats.n_vectors:,} x {stats.dim} in {stats.build_secs}s")

    # --- score on the held-out test week ---------------------------------
    test_customers = np.array(sorted(split.test_labels.keys()), dtype=np.int64)
    # Test-time history may use everything up to the test window.
    known = events[events.day <= split.valid_end_day]
    t_hist, t_hlen, t_dense = build_inference_state(
        known, n_items, test_customers,
        history_len=cfg.two_tower.history_len, item_price=price,
    )
    user_vecs = encode_users(model, test_customers, t_hist, t_hlen, t_dense)

    print(f"  ANN recall vs exact: {recall_against_exact(index, user_vecs, 20):.4f}")

    t0 = time.perf_counter()
    max_k = max(cfg.evaluation.k_values)
    item_ids, _ = index.search(user_vecs, max_k)
    search_s = time.perf_counter() - t0
    preds = {int(c): item_ids[i] for i, c in enumerate(test_customers)}

    res = evaluate_ranking(
        preds, split.test_labels, k_values=cfg.evaluation.k_values,
        catalog_size=n_items, item_popularity=catalog.base_popularity.to_numpy(),
        item_price=price, name="two_tower",
    )
    res.metrics["fit_s"] = round(sum(r["secs"] for r in log), 1)
    res.metrics["infer_ms_per_1k"] = round(1000 * search_s / len(test_customers) * 1000, 1)

    board = leaderboard([res])
    cols = [f"recall@{k}" for k in cfg.evaluation.k_values] + [
        "ndcg@20", "map@20", "hit_rate@20", "coverage@20", "novelty@20"
    ]
    print("\n" + format_leaderboard(board, cols))

    # --- persist ---------------------------------------------------------
    torch.save(
        {"state_dict": model.state_dict(),
         "config": cfg.two_tower.__dict__,
         "n_users": n_users, "n_items": n_items},
        cfg.artifact_dir / "two_tower.pt",
    )
    index.save(cfg.artifact_dir / "item_index")
    np.save(cfg.artifact_dir / "item_vectors.npy", item_vecs)
    board.to_csv(cfg.report_dir / "two_tower.csv", index=False)
    with open(cfg.report_dir / "two_tower_training.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nartifacts -> {cfg.artifact_dir}")


if __name__ == "__main__":
    main()
