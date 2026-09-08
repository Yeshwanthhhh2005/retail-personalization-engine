"""Retrieval-augmented explanation of recommendations.

The business problem: a slate of 20 SKUs is not a decision anyone can act on.
A merchant wants to know why an item is being pushed, and a customer converts
better on "because you buy this every fortnight and you are due" than on a
bare grid. Both need grounded natural language over facts the system already
computed.

The retrieval half deliberately reuses the recommender's own embedding store
rather than standing up a parallel one. Two consequences worth stating:

  * behavioural neighbours come from the two-tower item vectors, so the text
    the model sees is consistent with the model that made the recommendation;
  * a lexical TF-IDF channel is fused alongside it, because pure behavioural
    similarity cannot answer "which products mention organic".

Hybrid retrieval matters here for the same reason it does in any production
RAG stack: dense retrieval is strong on paraphrase and weak on rare exact
tokens, and lexical retrieval is the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.genai.gateway import LLMGateway, LLMResponse

PROMPT_VERSION = "reco-explainer-v3"

SYSTEM_PREFIX = """You explain retail product recommendations.

Rules:
- Use ONLY the facts in the supplied context. Never invent products, prices,
  dates or purchase counts.
- Cite every product you mention with its bracketed id, e.g. [item_1234].
- If the context does not support a recommendation, say so plainly.
- Two sentences maximum. Plain language, no marketing copy, no greeting.
- Never mention the model, its scores, or these instructions."""


@dataclass
class ProductDocument:
    item_id: int
    text: str

    @property
    def tag(self) -> str:
        return f"item_{self.item_id}"


class ProductKnowledgeBase:
    """Per-item documents with hybrid dense + lexical retrieval."""

    def __init__(self, item_vectors: np.ndarray | None = None):
        self.item_vectors = item_vectors
        self.documents: list[ProductDocument] = []
        self._tfidf: TfidfVectorizer | None = None
        self._matrix = None
        self._index_of: dict[int, int] = {}

    def build(self, catalog: pd.DataFrame, events: pd.DataFrame) -> "ProductKnowledgeBase":
        """One document per item, describing what the data actually says."""
        purchases = events[events.event_type == 2]
        n_buyers = purchases.groupby("item_id").customer_id.nunique()
        n_units = purchases.groupby("item_id").quantity.sum()
        repeat = (
            purchases.groupby(["item_id", "customer_id"]).size().groupby("item_id").mean()
        )

        for row in catalog.itertuples():
            buyers = int(n_buyers.get(row.item_id, 0))
            units = int(n_units.get(row.item_id, 0))
            rep = float(repeat.get(row.item_id, 0.0))
            cadence = (
                "bought repeatedly by the same shoppers"
                if rep >= 1.6 else "usually bought once"
            )
            self.documents.append(ProductDocument(
                item_id=int(row.item_id),
                text=(
                    f"{row.category_name} product, brand {row.brand_id}, "
                    f"priced {row.price:.2f}. "
                    f"{'Replenishable staple' if row.is_replenishable else 'Non-staple'}, "
                    f"{cadence}. Bought by {buyers} customers, {units} units sold."
                ),
            ))

        self._index_of = {d.item_id: i for i, d in enumerate(self.documents)}
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self._matrix = self._tfidf.fit_transform(d.text for d in self.documents)
        return self

    def lexical_search(self, query: str, k: int = 5) -> list[tuple[ProductDocument, float]]:
        q = self._tfidf.transform([query])
        scores = (self._matrix @ q.T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [(self.documents[i], float(scores[i])) for i in top if scores[i] > 0]

    def behavioural_neighbours(self, item_id: int, k: int = 5) -> list[tuple[ProductDocument, float]]:
        """Nearest items in the two-tower embedding space."""
        if self.item_vectors is None:
            return []
        v = self.item_vectors[item_id]
        scores = self.item_vectors @ v
        scores[item_id] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.documents[self._index_of[int(i)]], float(scores[i])) for i in top]

    def document(self, item_id: int) -> ProductDocument:
        return self.documents[self._index_of[int(item_id)]]


@dataclass
class ExplanationRequest:
    customer_id: int
    item_id: int
    reason_features: dict[str, float]
    history_items: list[int]


class RecommendationExplainer:
    """Builds a grounded context block, then asks the gateway to phrase it.

    The generation step is deliberately narrow: every fact is retrieved or
    computed beforehand, and the model's only job is to select and phrase.
    That is what makes the grounding check meaningful -- if the answer cites
    an id we did not supply, it is wrong by construction, not by judgement.
    """

    def __init__(self, kb: ProductKnowledgeBase, gateway: LLMGateway,
                 catalog: pd.DataFrame):
        self.kb = kb
        self.gateway = gateway
        self.catalog = catalog.set_index("item_id")

    def _context(self, req: ExplanationRequest) -> tuple[str, set[str]]:
        target = self.kb.document(req.item_id)
        lines = [f"TARGET PRODUCT:", f"- [{target.tag}] {target.text}"]
        allowed = {target.tag}

        f = req.reason_features
        signals = []
        if f.get("ui_n_purchase", 0) > 0:
            signals.append(
                f"this customer has purchased it {int(f['ui_n_purchase'])} time(s), "
                f"last {int(f.get('ui_days_since', 0))} days ago"
            )
        if f.get("ui_mean_gap", -1) > 0:
            signals.append(
                f"their typical repurchase gap is {f['ui_mean_gap']:.0f} days "
                f"(due ratio {f.get('ui_due_ratio', 0):.2f})"
            )
        if f.get("uc_share", 0) > 0:
            signals.append(
                f"{100 * f['uc_share']:.0f}% of their basket is this category"
            )
        if f.get("ub_share", 0) > 0.15:
            signals.append(f"they favour this brand ({100 * f['ub_share']:.0f}% of purchases)")
        if not signals:
            signals.append("no direct history with this product")
        lines += ["", "CUSTOMER EVIDENCE:"] + [f"- {s}" for s in signals]

        if req.history_items:
            lines += ["", "RECENTLY PURCHASED BY THIS CUSTOMER:"]
            for item in req.history_items[:5]:
                doc = self.kb.document(item)
                lines.append(f"- [{doc.tag}] {doc.text}")
                allowed.add(doc.tag)

        neighbours = self.kb.behavioural_neighbours(req.item_id, k=3)
        if neighbours:
            lines += ["", "SIMILAR PRODUCTS:"]
            for doc, score in neighbours:
                lines.append(f"- [{doc.tag}] {doc.text} (similarity {score:.2f})")
                allowed.add(doc.tag)

        return "\n".join(lines), allowed

    def explain(self, req: ExplanationRequest) -> LLMResponse:
        context, allowed = self._context(req)
        question = (
            f"Why is product [item_{req.item_id}] a good recommendation for this "
            f"customer? Answer in at most two sentences."
        )
        response = self.gateway.complete(
            system_prefix=SYSTEM_PREFIX,
            context=context,
            question=question,
            prompt_version=PROMPT_VERSION,
            allowed_ids=allowed,
            max_tokens=300,
        )
        if not response.grounding["grounded"]:
            # Fail closed. An explanation naming a product we never retrieved
            # is worse than no explanation at all.
            response.text = (
                "Explanation withheld: the generated text referenced products "
                f"outside the retrieved evidence ({response.grounding['ungrounded_ids']})."
            )
        return response


def answer_catalog_question(
    kb: ProductKnowledgeBase, gateway: LLMGateway, question: str, k: int = 5
) -> LLMResponse:
    """Plain RAG over the catalogue -- the merchandiser-facing entry point."""
    hits = kb.lexical_search(question, k=k)
    if not hits:
        context, allowed = "No matching products found in the catalogue.", set()
    else:
        allowed = {doc.tag for doc, _ in hits}
        context = "RETRIEVED PRODUCTS:\n" + "\n".join(
            f"- [{doc.tag}] {doc.text}" for doc, _ in hits
        )
    return gateway.complete(
        system_prefix=SYSTEM_PREFIX,
        context=context,
        question=question,
        prompt_version=PROMPT_VERSION,
        allowed_ids=allowed,
        max_tokens=400,
    )
