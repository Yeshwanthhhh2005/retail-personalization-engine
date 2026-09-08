# Prompts & Direction

How this system was specified, decided, and steered — the prompts that drove it
and the decisions each one opened up.

Direction here was deliberately high-leverage: a detailed specification up
front, then steering through review of intermediate results rather than
through repeated re-prompting. Bad metrics were rejected and the data
regenerated; slow code was profiled and rewritten; claims were checked against
measurements before they were allowed into the README. The prompts are the
spine of that process — what follows is each one and the work it set in motion.

---

## 1 — The specification

The build was specified by a full job posting supplied verbatim: **Walmart
Global Tech, Data Scientist, Bangalore — requisition R-2607505,
Personalization team.** Roughly 900 words covering the position summary,
responsibilities, and required skills:

> Apply and/or develop statistical modeling techniques (such as deep neural
> networks and Bayesian models), optimization methods and other ML
> techniques… Develop efficient and scalable models at Walmart scale…
> Define and/or own the model goodness metrics and track the business impact
> over time.
>
> **What you'll bring:** Experience in machine learning, supervised and
> unsupervised: NLP, Classification, Data/Text Mining, Multi-modal models,
> Neural Networks, Deep Learning Algorithms. Embedding generation from
> training materials, storage and retrieval from Vector Databases, set-up and
> provisioning of managed LLM gateways, development of Retrieval augmented
> generation based LLM agents, model selection, prompt engineering and
> finetuning based on accuracy and user-feedback, monitoring and governance.
> Strong Experience in Python, PySpark, Google Cloud platform, model
> deployment. Strong Experience with big data platforms.

**Response:** The working directory was empty, so before writing anything I
established what artifact was actually wanted — interview preparation, resume
tailoring, a fit/gap analysis, or a build.

---

## 2 — Scope: treat the specification as the build

> forget abou that, just do the assignment

Short, and the single most consequential instruction in the project. It
resolved the ambiguity in one direction — **build the system the posting
describes** — and every architectural decision follows from it without further
specification.

The requirement-to-implementation mapping it produced:

| Requirement in the posting | What was built |
|---|---|
| deep neural networks | Two-tower retrieval model, PyTorch, in-batch sampled softmax with logQ correction |
| embeddings, vector databases | FAISS index over learned item embeddings, with ANN-vs-exact recall measurement |
| managed LLM gateways | `src/genai/gateway.py` — provider binding, prompt versioning, caching, cost accounting |
| RAG-based LLM agents | Hybrid dense + lexical retrieval, grounded explanation generation |
| prompt engineering, monitoring, governance | Cacheable prefix isolation, JSONL audit trail, grounding guard, call budgets |
| PySpark, big data platforms | `src/features/spark_pipeline.py` — point-in-time correctness, salted joins, skew handling |
| GCP, model deployment | FastAPI service at 45 ms p50, plus a component-by-component GCP deployment plan |
| **own the model goodness metrics** | Hand-tested metric layer, temporal evaluation protocol, repeat-vs-discovery decomposition |

That last row is the one the posting emphasises twice, and it drove the
choices that matter most: temporal splits rather than random, unserved
customers counted as misses, retrieval models deliberately left stale when
scoring the test week, and a metric layer unit-tested against hand-computed
values before any model number was trusted.

**Decisions this opened, resolved during the build:**

- *The simulator must not trivialise the problem.* The first generated dataset
  put the top 100 items at 94.8% of purchases — popularity would have drowned
  personalization. Rejected and re-tuned to 26.1%.
- *Baselines must be real.* With a 42% repeat-purchase rate, "show them what
  they always buy" scores 0.488 recall@20 and beats a tuned ALS. The neural
  model had to clear that bar, and on its own it did not — which is the
  argument for the two-stage design.
- *Environment constraints must be stated, not hidden.* No JVM was available,
  so the PySpark pipeline is labelled unverified rather than implied to work.

---

## 3 — Review checkpoint: status

> how much the work is done?

Produced a component-by-component accounting: what was verified, what was
outstanding, and — unprompted — two caveats. That PySpark could not execute,
and that a background job verifying a shared-code refactor had not yet
reported, so the refactor could not yet be called clean.

---

## 4 — Review checkpoint: verification

> so it works right?

The most useful question asked in the session, because it was answered by
re-running rather than by asserting: 22/22 tests, all seven artifacts present,
the full leaderboard reproduced.

It also surfaced a real gap — `analyse_results.py` could not run, because the
input file it reads had been added to the pipeline *after* the last pipeline
run had already started.

---

## 5 — Publication: naming

> i need to push it into git, what should i name it?

Produced the name `retail-personalization-engine`, and two decisions that
changed the deliverable:

- **Avoid `walmart-*` in the repository name.** The work was built from a
  public posting and uses no Walmart data or code; a name implying
  affiliation is a liability on a public profile, and the association belongs
  in the README instead.
- **Gitignore the 30 MB of generated data and model artifacts.** All of it
  rebuilds deterministically from a fixed seed, keeping the repository at
  ~280 KB of actual work.

---

## 6 — Publication: ship it

> good, push everythign into this repo
> https://github.com/Yeshwanthhhh2005/retail-personalization-engine.git

Completed the remaining work: the analysis script, `requirements.txt`,
`.gitignore`, `scripts/run_all.py`, and the README.

Running the analysis produced the most important finding in the project:

```
repeat_recall       0.9963      ← items the customer had bought before
discovery_recall    0.1278      ← items genuinely new to them
```

The headline 0.5465 recall@20 is almost entirely repeat purchases. That was
promoted to a prominent README section rather than buried, because the
headline alone would mislead a reader about what the system actually does —
and a reviewer who found it themselves would rightly discount everything else.

---

## 7 — Documentation

The final turns concerned this documentation set rather than the system
itself — requesting a README, an AI-usage disclosure, a session transcript,
test-run output, and this page, and then a revision of how the direction
behind the project was presented. They produced [AI_USAGE.md](AI_USAGE.md),
[TRANSCRIPT.md](TRANSCRIPT.md), [TEST_OUTPUT.md](TEST_OUTPUT.md), this file,
and `.github/workflows/tests.yml`.

*(Prompts 1–6 are quoted verbatim; these closing turns are summarised, as they
concern documentation rather than the system.)*

---

## What the direction produced

| | |
|---|---|
| Lines of Python | 5,259 across 45 files |
| Tests | 22, all passing, no network or credentials required |
| Defects found and corrected | 12 — catalogued in [AI_USAGE.md](AI_USAGE.md) |
| Largest optimisations | ALS 96 min → 40 s · serving 2,865 ms → 45 ms |
| Result | +9.0% recall@20 and +9.2% NDCG@20 over the strongest baseline |

The detailed record — every defect, what was accepted or rejected, and the
verification behind each claim — is in [AI_USAGE.md](AI_USAGE.md), with the
full session in [TRANSCRIPT.md](TRANSCRIPT.md).
