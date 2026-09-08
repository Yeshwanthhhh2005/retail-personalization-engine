# Direction & Build Log

How this system was specified and steered, and the twelve stages of work that
produced it.

The build was directed by a detailed specification given up front, then steered
through review of intermediate results rather than through repeated
re-prompting — bad metrics rejected and the data regenerated, slow code
profiled and rewritten, claims checked against measurements before being
allowed into the README.

The specification and the two instructions that set scope are quoted verbatim
below. The twelve stages that follow are the work itself: what each stage had
to decide, what went wrong in it, and what came out.

---

## The specification

A full job posting, supplied verbatim: **Walmart Global Tech, Data Scientist,
Bangalore — requisition R-2607505, Personalization team.** Roughly 900 words.
The technical core:

> Apply and/or develop statistical modeling techniques (such as deep neural
> networks and Bayesian models), optimization methods and other ML
> techniques… Develop efficient and scalable models at Walmart scale…
> **Define and/or own the model goodness metrics** and track the business
> impact over time.
>
> Experience in machine learning, supervised and unsupervised… Neural
> Networks, Deep Learning Algorithms. Embedding generation from training
> materials, storage and retrieval from **Vector Databases**, set-up and
> provisioning of **managed LLM gateways**, development of **Retrieval
> augmented generation based LLM agents**, model selection, prompt
> engineering… **monitoring and governance**. Strong Experience in Python,
> **PySpark**, **Google Cloud platform**, **model deployment**.

## The directive

> forget abou that, just do the assignment

Short, and the most consequential instruction in the project: build the system
the posting describes. Every architectural decision follows from it.

| Requirement | What was built |
|---|---|
| deep neural networks | Two-tower retrieval model, PyTorch, in-batch sampled softmax with logQ correction |
| embeddings, vector databases | FAISS index over learned item embeddings, with ANN-vs-exact measurement |
| managed LLM gateways | `src/genai/gateway.py` — provider binding, prompt versioning, caching, cost accounting |
| RAG-based LLM agents | Hybrid dense + lexical retrieval with grounded generation |
| prompt engineering, monitoring, governance | Cacheable prefix isolation, JSONL audit trail, grounding guard, call budgets |
| PySpark, big data platforms | Point-in-time-correct feature pipeline with skew handling |
| GCP, model deployment | FastAPI service at 45 ms p50, plus a GCP deployment plan |
| **own the model goodness metrics** | Hand-tested metric layer, temporal protocol, repeat-vs-discovery decomposition |

The posting emphasises that last row twice, and it drove the choices that
matter most: temporal splits rather than random, unserved customers counted as
misses, retrieval models deliberately left stale when scoring the test week,
and a metric layer tested against hand-computed values before any model number
was trusted.

---

## The build in twelve stages

## 1 · Survey the environment before designing anything

Checked what was actually installed before committing to an architecture:

```
python 3.11.9 · numpy 1.26.4 · pandas 2.2.0 · scipy 1.15.3 · sklearn 1.4.0
lightgbm 4.6.0 · faiss 1.15.0 · fastapi 0.115.6
torch MISSING · pyspark MISSING · java MISSING
```

Two findings shaped everything after. **No JVM** meant PySpark could never run
here — decided immediately to write it as production code but label it
unverified rather than imply it worked. **No torch** meant installing the CPU
wheel in the background while the data layer was written.

## 2 · Put every tunable in one place

`src/config.py` — frozen dataclasses for the data generator, splitting,
two-tower, ranker and evaluation. Written first, so the offline pipeline, the
training jobs and the online service read the same numbers rather than drifting
apart through duplicated constants.

## 3 · Generate an event stream worth modelling

No real customer data can ship with a public project, so the dataset is
simulated — but a simulator that makes the problem easy is worse than useless.
Built in the properties that make retail personalization genuinely hard: Zipf
popularity, heavy repeat purchase, latent taste structure held out of the
feature set, category seasonality, and a view → cart → purchase funnel.

First run: **1.5M events**.

## 4 · Reject the first dataset and calibrate it

The first dataset was unusable, and only measurement revealed it:

```
top-100 share : 0.948      ← 100 items = 94.8% of all purchases
```

Popularity had drowned the taste signal; a popularity baseline would have been
near-unbeatable and the exercise meaningless. Exposed the mixing weights as
real parameters and swept them — which surfaced a **second** defect:
`popularity_zipf` did nothing at all. Identical results at every value, because
the scorer z-scores `log(1/rank^a)` and z-scoring cancels the exponent exactly.
A plausible, well-commented, completely inert knob.

Re-tuned and regenerated:

```
events 1,672,905 · purchases 378,538 · items covered 4,000/4,000
repeat share 0.424 · top-100 share 0.261 · purchase Gini 0.683
```

## 5 · Split on time, and make leakage impossible

```
days 0–105    train    retrieval models fitted here
days 106–112  valid    labels for the ranker's training set
days 113–119  test     the final report — touched once
```

A random split lets a model see a customer's Friday basket while predicting
their Tuesday one. `FeatureStore.build` asserts its own cutoff and raises if
history contains anything past `as_of_day`; a test confirms the assertion
actually fires.

## 6 · Test the metrics before trusting any model number

Every result in the repo rests on this layer, so its expected values are
**computed by hand in the test comments** rather than snapshotted from the
implementation. Nine tests, including the one that matters most —
`test_unserved_customer_counts_as_miss`, guarding the most common way a
recommender's headline metric gets quietly inflated.

## 7 · Build baselines strong enough to be embarrassing

With a 42% repeat rate, "show them what they always buy" is not a strawman:

```
 blend_rrf   0.5015     repeat  0.4877     als  0.4607
  item_knn   0.1972  popularity  0.0973          (recall@20)
```

ALS took **96 minutes**. Profiling found `np.linalg.solve` on a 64×64 matrix
costing 11 ms — 400× too slow — because multithreaded OpenBLAS spawns a thread
team per tiny solve and thrashes. Pinning threads and batching made it **300×
faster**; that fix then introduced a memory blowup (batches padded to the
widest row, and hot items have 4,328 interactions against a mean of 88), fixed
by budgeting on `rows × width`. **96 min → 40 s.**

## 8 · Train the two-tower retrieval model

```
user tower : [id | mean-pooled history | dense RFM] → MLP → L2-norm
item tower : [id | category | brand | price bucket] → MLP → L2-norm
score      : cosine / temperature
```

In-batch sampled softmax with the **logQ correction** (Yi et al., RecSys 2019).
Without it, negatives arrive in proportion to popularity, the loss
over-penalises head items, and retrieval drifts into the tail — recall collapses
while coverage looks suspiciously excellent.

First run stopped at 6 epochs with validation still climbing; extended to 18,
plateauing at 0.454 around epoch 14.

## 9 · Precompute the catalogue into a vector index

The architectural payoff is asymmetry: **the item tower never runs online.**
The catalogue is encoded once per refresh into FAISS, so a request costs one
user-tower pass plus an ANN lookup. Verified exact against brute force
(`recall_against_exact` = **1.0000**), with the check kept because that
tradeoff stops being free at tens of millions of items.

Two-tower alone: **0.4393 recall@20** — below `repeat`. Expected, and the
reason for stage 10.

## 10 · Combine the signals with a learned ranker

Four candidate sources with complementary blind spots, unioned to ~165
candidates, joined against 42 features, ranked by LightGBM LambdaRank.

```
two_stage_ranker  recall@20 0.5465  ndcg@20 0.3922  coverage 0.9407
```

**+9.0% recall@20 and +9.2% NDCG** over the strongest baseline, with coverage
held at 94% — not bought by collapsing onto the head.

The decomposition matters more than the headline, and is reported prominently
rather than buried:

```
repeat_recall     0.9963      ← items the customer had bought before
discovery_recall  0.1278      ← items genuinely new to them
```

The system is near-perfect at re-buys and weak at discovery. A reader seeing
only "+9%" would draw the wrong conclusion about what it does.

## 11 · Add the GenAI layer, with governance that bites

A slate of 20 SKUs is not a decision anyone can act on, so this layer explains
recommendations — reusing the recommender's *own* embedding store rather than
standing up a parallel one, fused with a lexical channel.

The gateway owns provider binding, prompt versioning, cache-prefix isolation,
per-call cost accounting and a JSONL audit trail. The **grounding guard** is
the part that earns its place: a generated answer may only cite item ids that
were in the retrieved context, and anything else is withheld. It is tested by
injecting a deliberately fabricating provider and confirming the fake SKU is
caught.

## 12 · Serve it, profile it, and be honest about the untested path

First working service: **2,865 ms p50**. Profiling showed the feature join was
133 ms of 155 ms — 86% — while the neural work was 2.2 ms, the opposite of
where the time was assumed to be.

```
2,865 ms  →  163 ms  →  45 ms
```

Because the second fix touched code shared by training and serving — where skew
never shows up in offline metrics — it was accepted only after a value-level
parity check (**42 features × 11,946 rows, zero mismatches**) and a full
pipeline reproduction to identical numbers.

The PySpark pipeline closes the loop to production scale. **It has never been
executed** — no JVM — so it is covered by a contract test asserting it and the
pandas path declare identical features and null policy, and it is labelled
unverified in the module, the README, and here.

---

## Review checkpoints

Two questions during the build did more for quality than any additional
specification would have:

**"how much the work is done?"** — produced a component-by-component status
with two caveats surfaced unprompted: that PySpark could not execute, and that
a refactor could not yet be called clean because its verification job had not
reported.

**"so it works right?"** — answered by re-running rather than asserting
(22/22 tests, all artifacts present, leaderboard reproduced), and it surfaced a
real gap: the analysis script could not run, because its input had been added
to the pipeline after the previous run had already started.

---

## What the direction produced

| | |
|---|---|
| Lines of Python | 5,259 across 45 files |
| Tests | 22, all passing, no network or credentials required |
| Defects found and corrected | 12 — catalogued in [AI_USAGE.md](AI_USAGE.md) |
| Largest optimisations | ALS 96 min → 40 s · serving 2,865 ms → 45 ms |
| Result | +9.0% recall@20, +9.2% NDCG@20 over the strongest baseline |

Full detail: [AI_USAGE.md](AI_USAGE.md) for every defect and the verification
behind each claim, [TRANSCRIPT.md](TRANSCRIPT.md) for the session itself.
