# Retail Personalization Engine

A two-stage recommender for grocery-and-general-merchandise retail: multi-source
candidate retrieval feeding a learned ranker, with a vector index, an online
service, and a governed LLM layer for explaining recommendations.

Built against the requirements of a Walmart Global Tech Data Scientist posting
(Personalization team, R-2607505) — deep neural retrieval, embeddings and vector
databases, RAG with a managed LLM gateway, PySpark at scale, model deployment,
and owning the metrics that decide whether any of it ships. It is an independent
project built from a public job description; it uses no Walmart data, code, or
affiliation.

---

## Headline result

Evaluated on a held-out **future week** (days 113–119), scoring 8,943 customers
who were unseen in that window:

| model | recall@20 | NDCG@20 | MAP@20 | coverage@20 |
|---|---|---|---|---|
| **two-stage (retrieval → ranker)** | **0.5465** | **0.3922** | **0.2718** | 0.941 |
| RRF blend of all retrievers | 0.5015 | 0.2920 | 0.1717 | 0.923 |
| repeat-purchase | 0.4877 | 0.3591 | 0.2484 | 0.944 |
| ALS matrix factorization | 0.4607 | 0.2793 | 0.1671 | 0.531 |
| two-tower alone | 0.4393 | 0.3153 | 0.2117 | 0.906 |
| item-item kNN | 0.1972 | 0.0977 | 0.0466 | 0.992 |
| popularity | 0.0973 | 0.0505 | 0.0232 | 0.005 |

**+9.0% recall@20** over the strongest baseline and **+9.2% NDCG@20**, while
holding catalogue coverage at 94% — the system is not buying accuracy by
collapsing onto the head of the catalogue.

Online: **45 ms p50 / 73 ms p95** end-to-end per request, single CPU process.

---

## Why the baselines matter

In grocery, ~42% of purchases are re-buys. That makes "just show them what they
always buy" a genuinely strong strategy, not a strawman — `repeat` alone reaches
0.488 recall@20 and beats a well-tuned ALS. Any honest evaluation has to clear
that bar, and a surprising number of published recommender comparisons quietly
do not.

Note that `repeat` wins NDCG@20 among the single retrievers (0.359) while the
RRF blend wins recall@20 (0.502). They are good at different things: `repeat`
puts a few near-certain items at the very top, the blend finds more of the
basket further down. Neither dominates — which is precisely the argument for a
learned ranker over a hand-tuned fusion.

---

## Architecture

```
                    ┌─────────────── candidate sources ───────────────┐
   customer ───────▶│  repeat(60)  two-tower(120)  kNN(60)  pop(20)   │
                    └──────────────────────┬──────────────────────────┘
                                           │  union ≈ 165 candidates
                                           ▼
                              ┌────────────────────────┐
                              │  feature store join    │  42 features
                              │  (precomputed aggs)    │
                              └───────────┬────────────┘
                                          ▼
                              ┌────────────────────────┐
                              │  LightGBM LambdaRank   │
                              └───────────┬────────────┘
                                          ▼
                                    top-k slate
```

**Stage 1 — retrieval.** Four sources with complementary blind spots. The
two-tower model is the only one that generalises to items the customer has never
touched; `repeat` is the only one that reliably nails re-buys; kNN finds
complements; popularity guarantees a full slate for anyone.

**Stage 2 — ranking.** LightGBM LambdaRank over the pooled candidates. A GBDT
rather than another network because the features are heterogeneous tabular
signals (counts, ratios, day-gaps, model scores), it trains in seconds on CPU,
and its feature importances survive contact with a business review.

### The two-tower model

```
user tower : [user id | mean-pooled history | dense RFM] → MLP → L2-norm → z_u
item tower : [item id | category | brand | price bucket] → MLP → L2-norm → z_i
score      : cosine(z_u, z_i) / temperature
```

Trained with in-batch sampled softmax and the **logQ correction**
(Yi et al., RecSys 2019). Without that correction, in-batch negatives arrive in
proportion to popularity, the loss over-penalises head items, and retrieval
drifts into the long tail — recall collapses while coverage looks suspiciously
excellent. The item embedding table is shared between the history encoder and
the item tower.

The architectural payoff is asymmetry: **the item tower never runs online.** The
catalogue is encoded once per refresh into a FAISS index, so a request costs one
user-tower forward pass (2.2 ms) plus an ANN lookup (0.4 ms). That property, not
raw accuracy, is why two-tower is the standard retrieval architecture at
catalogue scale.

---

## Evaluation protocol

Interaction data is **split on time, never at random**:

```
days 0–105     train   retrieval models fitted here
days 106–112   valid   labels for the RANKER's training set
days 113–119   test    the final report — touched once
```

A random split lets a model see a customer's Friday basket while predicting
their Tuesday one. It inflates every offline metric and predicts nothing about
an A/B test.

Three further choices worth stating, because each is a place results get
accidentally inflated:

- **Retrieval models are deliberately left stale.** They stay fitted on days
  0–105 even when scoring the test week. A week-old retrieval model is what
  production actually serves between retrains; refitting them on the eve of the
  test window would flatter the result.
- **Unserved customers count as misses.** Customers with labels but no
  prediction are scored zero rather than dropped. A model that declines to serve
  part of the population pays for it in the headline number.
- **Feature construction asserts its own cutoff.** `FeatureStore.build` raises
  if the history frame contains any event after `as_of_day`.

Metrics are unit-tested against hand-computed values (`tests/test_metrics.py`)
rather than snapshotted from the implementation — every model number in this
repo rests on those functions.

---

## What the ranker actually learned

Top features by gain:

| feature | signal |
|---|---|
| `rr_repeat` | reciprocal rank from the repeat model |
| `in_repeat` | did the repeat source propose this at all |
| `ui_n_cart` | times this customer carted this item |
| `ui_n_view` | times viewed |
| `n_sources` | how many retrievers agreed |
| `ub_share` | share of the customer's basket from this brand |
| `rr_two_tower`, `score_two_tower` | the neural retrieval signal |
| `cat_due_ratio` | how overdue the whole category is |

`n_sources` ranking that highly is the two-stage design paying for itself:
cross-retriever agreement is a real signal, and it only exists because
candidates come from multiple sources.

The replenishment features (`ui_due_ratio`, `cat_due_ratio` — elapsed time
divided by that customer's own repurchase cadence) encode the domain fact that
makes grocery different from media: "they buy milk every 6 days and it has been
7" is more actionable than any similarity score.

---

## The result decomposed — and where it is weak

The headline number hides the mechanism, so `scripts/analyse_results.py` splits
it. This is the most important table in the repo:

| | recall@20 |
|---|---|
| items the customer had bought before | **0.9963** |
| items genuinely new to the customer | **0.1278** |
| *(repeats are 45.8% of test purchases)* | |

**The system is close to perfect at re-buys and weak at discovery.** The
headline 0.5465 is mostly the first column. That is worth saying plainly: a
retailer deploying this would see reliable basket completion and very little
genuine expansion, and anyone reading "+9% recall" without this breakdown would
draw the wrong conclusion about what they were buying.

It is also a defensible place to be — re-buy accuracy is what drives grocery
basket completion, and 46% of purchases are re-buys — but discovery is the
obvious next target, and it is a *ranker* problem more than a retrieval one: the
candidate sets already contain more new items than the ranker promotes, because
the repeat features dominate the learned ordering.

### Does each candidate source earn its cost?

| source | slate share | hits | precision | exclusive hits |
|---|---|---|---|---|
| two_tower | 0.897 | 11,535 | 0.072 | **335** |
| repeat | 0.492 | 10,750 | **0.122** | 326 |
| item_knn | 0.624 | 7,178 | 0.064 | 41 |
| popularity | 0.222 | 1,660 | 0.042 | 22 |

`exclusive hits` counts conversions that **no other source proposed** — the only
column that answers whether a retriever is pulling its weight rather than
re-finding what something cheaper already found.

- **two_tower earns its place**: the most exclusive hits, and it is the only
  source that reaches items with no prior interaction.
- **repeat has by far the best precision** — highest hit rate per slot.
- **item_knn is largely redundant**: 62% of slate slots for 41 exclusive hits.
  A candidate for removal on a latency budget.
- **popularity is nearly dead weight** at 22 exclusive hits, but it is the
  cold-start safety net, so it stays.

### Who is served worst

| history depth | customers | recall@20 |
|---|---|---|
| 1–5 purchases | 163 | 0.5197 |
| 6–15 | 3,570 | 0.5287 |
| 16–30 | 4,879 | 0.5590 |
| 31–60 | 330 | 0.5710 |

Quality degrades gracefully rather than falling off a cliff for light shoppers —
the gap between the thinnest and richest histories is ~5 points, not 30.

---

## GenAI layer

The JD asks for embeddings, vector databases, managed LLM gateways, RAG agents,
and governance. Rather than bolt on a chatbot, the layer answers a question the
recommender creates: **a slate of 20 SKUs is not a decision anyone can act on.**
A merchant wants to know why an item is being pushed; a customer converts better
on "you buy this every fortnight and you're due" than on a bare grid.

- **Retrieval reuses the recommender's own embedding store** — behavioural
  neighbours come from the two-tower item vectors, fused with a lexical TF-IDF
  channel. Dense retrieval is strong on paraphrase and weak on rare exact
  tokens; lexical is the reverse.
- **The gateway** (`src/genai/gateway.py`) owns provider binding (Claude via the
  official SDK, `claude-opus-5`), prompt versioning, prompt caching with the
  stable prefix isolated from volatile context, per-call token/cost accounting
  including the cache-read discount, and a JSONL audit trail.
- **Grounding guard.** Every generated answer may only cite item ids that were
  in the retrieved context. Anything else is flagged and the explanation is
  withheld. This is the cheap deterministic half of hallucination control, and
  it catches the failure mode that would put a fabricated SKU in front of a
  customer. It is unit-tested.
- **Offline provider.** With no `ANTHROPIC_API_KEY`, the gateway falls back to a
  deterministic stub so the pipeline and CI run end to end with no credentials.
  It is not a second vendor and every audit record says which provider ran.

---

## Serving

`src/serve/app.py` — FastAPI, same two stages as the offline pipeline.

The critical property is that the service and the training job call **the same
`FeatureStore.transform`**. Training/serving skew — a feature defined one way in
training and another way online — is the most expensive class of bug in a
production recommender, and offline metrics never reveal it.

Getting to 45 ms took two fixes that are worth recording because both were
invisible until profiled:

| | p50 |
|---|---|
| naive | 2865 ms |
| hoist per-customer state to startup | 163 ms |
| index the feature blocks, `reindex` instead of `merge` | **45 ms** |

The first was rebuilding user-tower inputs by re-scanning all 1.7M events on
every request. The second was that a pandas `merge` of a 200-row request against
a 700k-row aggregate table rebuilds the hash side every call; pre-indexing on
the join keys turns it into a 200-key lookup. That rewrite was verified
value-for-value against the original merge across all 42 features
(11,946 rows, zero mismatches) and by re-running the full pipeline to identical
metrics.

---

## Running it

```bash
pip install -r requirements.txt
python -m scripts.run_all          # ~15 min, CPU only
```

Or stage by stage:

```bash
python -m src.data.generate        # synthetic event stream
python -m scripts.eval_baselines   # baseline leaderboard
python -m scripts.train_retrieval  # two-tower + FAISS index
python -m scripts.train_ranker     # two-stage pipeline, final metrics
python -m scripts.analyse_results  # repeat/discovery decomposition
python -m scripts.run_genai_demo   # RAG explanations + governance summary

uvicorn src.serve.app:app --port 8000
python -m scripts.smoke_serve      # latency percentiles
```

Tests:

```bash
python -m tests.test_metrics
python -m tests.test_genai
python -m tests.test_spark_contract
```

---

## The data

No real customer data can ship with a public project, so `src/data/generate.py`
simulates 1.67M events over 20k customers, 4k items and 120 days, with the
properties that make retail personalization hard:

- Zipf-skewed popularity, so the popularity baseline is genuinely strong
- 42% repeat-purchase rate, so "already bought" is signal rather than leakage
- latent taste structure held out of the feature set, so models must recover it
- category seasonality and a view → cart → purchase funnel

Tuning this honestly mattered. The first version put the top-100 items at
**94.8%** of all purchases; personalization was drowned by popularity and the
exercise would have been meaningless. Rebalancing the taste/popularity mix
brought it to **26.1%**, with a purchase Gini of **0.683** and 3,974 of 4,000
items selling at least once — skewed like real retail, but not degenerate.

**Read the absolute numbers accordingly.** They characterise models on a
simulator whose structure I chose. What transfers is the relative comparison
under a shared protocol, the engineering, and the evaluation discipline — not
"0.5465 recall is achievable on Walmart traffic."

---

## Scale: the PySpark path

`src/features/spark_pipeline.py` expresses the same feature definitions against
Spark for the full event stream, with point-in-time correctness enforced before
aggregation, broadcast joins on the catalogue dimension, salted joins for the
skewed user-item aggregate, and `as_of_day` partitioning so daily runs append
one partition and backfills are idempotent.

**This module has never been executed.** The development machine has no JVM. It
is verified by inspection and by `tests/test_spark_contract.py`, which asserts
the Spark and pandas paths declare identical feature sets and that the null
policy they document is the one pandas implements — a drift here would be
silent, with the Spark job happily serving a different feature distribution than
the model trained on. Treat it as reviewed design, not as tested code.

---

## Deploying this on GCP

The JD names GCP specifically. The mapping:

| component | service | notes |
|---|---|---|
| event stream | Pub/Sub → BigQuery | raw events, partitioned by day |
| feature pipeline | Dataproc Serverless (PySpark) | daily; writes `as_of_day` partitions |
| feature store | Bigtable (online) + BigQuery (offline) | one definition, two stores |
| training | Vertex AI Training, custom container | two-tower on one A100; ranker on CPU |
| model registry | Vertex AI Model Registry | versioned with the eval report attached |
| vector index | Vertex AI Vector Search | swap for FAISS-flat; 4k items is trivial, 40M is not |
| serving | Cloud Run or GKE | the FastAPI app; item tower stays offline |
| LLM gateway | Vertex AI (Claude on Vertex) | same gateway class, `AnthropicVertex` client |
| orchestration | Vertex AI Pipelines / Composer | daily features, weekly retrain |
| monitoring | Cloud Monitoring + BigQuery | see below |

**At real catalogue scale the flat index stops being free.** 4k items is exact
brute force; tens of millions needs IVF-PQ and an explicit recall-vs-latency
decision. `recall_against_exact()` in `src/retrieval/vector_index.py` exists to
make that tradeoff measurable rather than assumed — it returns 1.0 today by
construction, and earns its keep the day someone swaps the index type.

**What I would monitor**, in priority order:

1. **Online/offline metric divergence** — the single best early warning that
   training/serving skew has crept in
2. **Slate coverage and Gini** — a model quietly collapsing onto the head shows
   up here long before it shows up in revenue
3. **Feature freshness and null rates** per feature, alerting on distribution
   shift, not just on nulls
4. **p99 latency** and ANN recall against exact
5. **Grounding-failure rate** and cost per call on the LLM path

---

## Honest limitations

- **Results are on simulated data.** See the caveat above.
- **The Spark pipeline is unexecuted.** No JVM available.
- **No A/B test.** Every number here is offline. Offline recall is a proxy for
  business value, not a substitute — the decisive experiment is a live holdout
  measuring revenue and basket size, and nothing in this repo can stand in for it.
- **Discovery is weak** (0.128 recall on genuinely new items). See the
  decomposition above; this is the first thing I would work on.
- **Cold start is handled bluntly.** Customers with no history get the
  popularity slate. A content-based user tower using demographics and store
  context would do better; the item tower already supports cold *items* through
  its category/brand/price features.
- **The ranker is trained on one week of labels.** More windows, pooled, would
  reduce variance.
- **Sequence modelling is left on the table.** History is mean-pooled. A
  transformer over the purchase sequence (SASRec/BERT4Rec-style) is the obvious
  next model, and the causal sequence builder already emits what it needs.

---

## Repo layout

```
src/
  config.py                  every tunable in one place
  data/generate.py           synthetic event stream
  data/splits.py             temporal splitting, ground truth
  features/sequences.py      causal training sequences
  features/ranking_features.py  FeatureStore, 42 features, null policy
  features/spark_pipeline.py    production path (unexecuted)
  models/baselines.py        popularity, repeat, kNN, ALS, RRF blend
  models/two_tower.py        the retrieval network
  models/trainer.py          training loop, best-epoch checkpointing
  models/ranker.py           LambdaRank wrapper
  retrieval/vector_index.py  FAISS wrapper, ANN-vs-exact recall
  evaluation/metrics.py      recall/NDCG/MAP + coverage/novelty/Gini
  genai/gateway.py           managed LLM gateway, cost + governance
  genai/rag.py               hybrid retrieval, grounded explanations
  serve/app.py               FastAPI service
scripts/                     runnable pipeline stages
tests/                       22 tests, no network or credentials required
```

~4,500 lines of Python.
