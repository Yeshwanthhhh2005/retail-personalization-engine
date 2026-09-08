"""Demonstrate the RAG explanation layer over real pipeline output."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data.splits import load_catalog, load_events, make_temporal_split
from src.genai.gateway import LLMGateway
from src.genai.rag import (
    ExplanationRequest,
    ProductKnowledgeBase,
    RecommendationExplainer,
    answer_catalog_question,
)

EXPLAIN_FEATURES = [
    "ui_n_purchase", "ui_days_since", "ui_mean_gap", "ui_due_ratio",
    "uc_share", "ub_share",
]


def main() -> None:
    events, catalog = load_events(), load_catalog()
    split = make_temporal_split(events)
    item_vectors = np.load(CONFIG.artifact_dir / "item_vectors.npy")

    kb = ProductKnowledgeBase(item_vectors).build(catalog, split.train)
    gateway = LLMGateway(audit_path=CONFIG.report_dir / "llm_audit.jsonl")
    explainer = RecommendationExplainer(kb, gateway, catalog)
    print(f"knowledge base: {len(kb.documents):,} product documents")
    print(f"provider      : {gateway.provider.name}"
          + ("  (set ANTHROPIC_API_KEY for live Claude calls)"
             if gateway.provider.name == "offline" else ""))

    # Pull real ranked candidates with their real feature values.
    frame = pd.read_parquet(CONFIG.report_dir / "ranker_test_candidates.parquet")
    hits = frame[frame.label == 1]

    purchases = split.train[split.train.event_type == 2]
    history = purchases.groupby("customer_id").item_id.apply(
        lambda s: s.tolist()[-5:]
    ).to_dict()

    print("\n" + "=" * 78)
    print("RECOMMENDATION EXPLANATIONS")
    print("=" * 78)

    for _, row in hits.head(4).iterrows():
        cid, iid = int(row.customer_id), int(row.item_id)
        feats = {}
        # Recompute the same features the ranker saw, for this pair.
        cust_hist = purchases[(purchases.customer_id == cid)]
        own = cust_hist[cust_hist.item_id == iid]
        feats["ui_n_purchase"] = float(len(own))
        feats["ui_days_since"] = float(
            split.train_end_day - own.day.max()
        ) if len(own) else 9999.0
        if len(own) > 1:
            feats["ui_mean_gap"] = float(
                (own.day.max() - own.day.min()) / max(len(own) - 1, 1)
            )
            feats["ui_due_ratio"] = feats["ui_days_since"] / max(feats["ui_mean_gap"], 1)
        cat = int(catalog.loc[catalog.item_id == iid, "category_id"].iloc[0])
        cats = catalog.set_index("item_id").category_id
        feats["uc_share"] = float(
            (cats[cust_hist.item_id].to_numpy() == cat).mean()
        ) if len(cust_hist) else 0.0
        brand = int(catalog.loc[catalog.item_id == iid, "brand_id"].iloc[0])
        brands = catalog.set_index("item_id").brand_id
        feats["ub_share"] = float(
            (brands[cust_hist.item_id].to_numpy() == brand).mean()
        ) if len(cust_hist) else 0.0

        req = ExplanationRequest(
            customer_id=cid, item_id=iid,
            reason_features=feats, history_items=history.get(cid, []),
        )
        resp = explainer.explain(req)
        print(f"\ncustomer {cid} -> item {iid} "
              f"({catalog.loc[catalog.item_id == iid, 'category_name'].iloc[0]})")
        print(f"  evidence : purchased {int(feats['ui_n_purchase'])}x, "
              f"{feats['ui_days_since']:.0f}d ago, "
              f"category share {100 * feats['uc_share']:.0f}%")
        print(f"  answer   : {resp.text}")
        print(f"  grounded : {resp.grounding['grounded']}  "
              f"cited={resp.grounding['cited_ids']}  "
              f"{resp.latency_ms:.0f}ms  ${resp.usage.cost_usd(resp.model):.6f}")

    print("\n" + "=" * 78)
    print("CATALOGUE Q&A (lexical RAG)")
    print("=" * 78)
    for question in (
        "Which replenishable dairy products sell the most units?",
        "Show me expensive electronics that customers rarely rebuy.",
    ):
        resp = answer_catalog_question(kb, gateway, question, k=4)
        print(f"\nQ: {question}")
        print(f"A: {resp.text}")
        print(f"   grounded={resp.grounding['grounded']} "
              f"cited={resp.grounding['cited_ids']}")

    print("\n" + "=" * 78)
    print("GOVERNANCE SUMMARY")
    print("=" * 78)
    print(json.dumps(gateway.summary(), indent=2))
    print(f"\naudit log -> {CONFIG.report_dir / 'llm_audit.jsonl'}")


if __name__ == "__main__":
    main()
