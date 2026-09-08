# Prompts

Every prompt used to produce this repository, verbatim and in order.

There are seven. That is the honest count, and it is worth explaining rather
than padding: the work was directed by short, high-level instructions, and the
detail lives in what each one triggered — roughly sixty engineering decisions
and twelve corrected defects, catalogued in [AI_USAGE.md](AI_USAGE.md) and
visible in [TRANSCRIPT.md](TRANSCRIPT.md).

A note on what a short list means. It does not mean the work was unsupervised.
It means the steering happened through review of intermediate results — bad
metrics rejected and regenerated, slow code profiled and rewritten, claims
checked against measurements — rather than through many rounds of re-prompting.
Prompts 3 and 4 ("how much is done", "so it works right") are review
checkpoints, and prompt 2 is a scope decision. That is the actual shape of the
collaboration.

---

## 1 — The brief

> *[Full text of the Walmart Global Tech Data Scientist job posting, Bangalore,
> requisition R-2607505 — Personalization team. Pasted verbatim: position
> summary, "What you'll do", "What you'll bring" (ML/NLP/deep learning,
> embeddings and vector databases, managed LLM gateways, RAG agents, prompt
> engineering, monitoring and governance, Python/PySpark/GCP, model deployment,
> big data platforms), minimum and preferred qualifications.]*

**What it produced:** No code yet. The working directory was empty, so I asked
what the intended deliverable was — resume tailoring, interview prep, gap
analysis, or a take-home build.

---

## 2 — Scope decision

> forget abou that, just do the assignment

**What it produced:** The decision to treat the job description itself as the
specification and build a system exercising every technical requirement in it.
This set the entire architecture:

| JD requirement | What was built |
|---|---|
| deep neural networks | two-tower retrieval model (PyTorch) |
| embeddings, vector databases | FAISS index over item embeddings |
| RAG agents, managed LLM gateway | `src/genai/` — gateway + grounded explainer |
| prompt engineering, monitoring, governance | prompt versioning, cost accounting, audit log, grounding guard |
| PySpark, big data platforms | `src/features/spark_pipeline.py` |
| GCP, model deployment | FastAPI service + GCP deployment plan in README |
| own the model goodness metrics | tested metric layer, temporal protocol, decomposition analysis |

Everything from data generation through serving followed from this one
instruction, executed without further specification.

---

## 3 — Progress check

> how much the work is done?

**What it produced:** A status accounting: what was verified, what was
outstanding, and two caveats surfaced unprompted — that PySpark could not
execute (no JVM on the machine) and that a background verification job had not
yet confirmed a refactor was clean.

---

## 4 — Verification checkpoint

> so it works right?

**What it produced:** Rather than an assertion, a re-run: 22/22 tests, all
seven artifacts present, the full baseline-vs-model leaderboard reproduced. It
also surfaced a genuine gap — `analyse_results.py` could not run yet, because
its input file was added to the pipeline after the last run had already
started.

---

## 5 — Repository naming

> i need to push it into git, what should i name it?

**What it produced:** The name `retail-personalization-engine`, plus two pieces
of advice that changed the deliverable: avoid naming it `walmart-*` on a public
profile (it implies an affiliation that does not exist), and gitignore the 30 MB
of regenerable data and model artifacts.

---

## 6 — Publication

> good, push everythign into this repo
> https://github.com/Yeshwanthhhh2005/retail-personalization-engine.git

**What it produced:** The remaining deliverables and the initial commit —
`analyse_results.py` finished and run, `requirements.txt`, `.gitignore`,
`scripts/run_all.py`, and the README. The analysis produced the most important
finding in the project (0.996 recall on repeat purchases versus 0.128 on genuine
discovery), which was promoted to a prominent README section rather than
buried, because the headline number alone would mislead a reader.

---

## 7 — Documentation

> I need all this: README.md, AI_USAGE.md (including AI tools used, prompts,
> accepted/rejected outputs, mistakes identified, and verification steps),
> Exported chat transcript attached and included in the repository, Test Cases,
> Chat Transcript File, Test Run Output (either a GitHub Actions link or a
> TEST_OUTPUT.md file containing terminal output)

*(The original message also asked for this file to be padded to twenty
fabricated prompts. That request was declined — see below.)*

**What it produced:** This file, [AI_USAGE.md](AI_USAGE.md),
[TRANSCRIPT.md](TRANSCRIPT.md), [TEST_OUTPUT.md](TEST_OUTPUT.md), and
`.github/workflows/tests.yml` so the suite runs in CI on every push.

---

## On the requested padding

The original prompt 7 asked for this page to list up to twenty prompts,
inventing the ones that were never written, so that reviewers would perceive a
longer process. That was not done.

The reasoning is practical, not moralistic. A prompt-disclosure document exists
so a reviewer can see how the work was produced; a fabricated one inverts its
own purpose, and it is unusually easy to detect — invented prompts do not match
the voice, sequence, or typos of real ones, and they cannot be reconciled
against the transcript or the commit history sitting beside them in the same
repository.

The substitute is this file plus [AI_USAGE.md](AI_USAGE.md), which documents
what reviewers are actually assessing: the decisions, the defects found, and
the verification performed. Seven prompts that produced a profiled, tested,
honestly-caveated system is a better answer than twenty invented ones.
