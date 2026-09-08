# AI Usage

Full disclosure of how AI was used to build this repository.

---

## Tools

| Tool | Version / model | Role |
|---|---|---|
| Claude Code | CLI agent, VS Code extension | All implementation, profiling, debugging |
| Claude Opus 5 | `claude-opus-5` | The model behind Claude Code in this session |
| `claude-api` skill | bundled | Consulted before writing `src/genai/gateway.py`, to get current Anthropic SDK patterns rather than relying on the model's training prior |

**Scope: the entire codebase was AI-generated.** All 5,259 lines across `src/`,
`scripts/`, and `tests/` were written by Claude Code, directed by the seven
prompts in [PROMPTS.md](PROMPTS.md). The human contribution was direction,
scope decisions, and review checkpoints.

Everything below is a record of what that process actually looked like,
including the parts that went wrong.

---

## Verification performed

Nothing in this repo is asserted on the model's say-so. What was checked, and
how:

| Claim | How it was verified |
|---|---|
| Metric implementations are correct | 9 unit tests with **expected values computed by hand in the comments**, not snapshotted from the implementation — the values would be wrong if the code were wrong |
| The feature-store rewrite changed nothing | Value-for-value comparison against the original merge implementation: **42 features × 11,946 rows, zero mismatches** |
| The pipeline is reproducible | Full pipeline run **three times**, producing identical metrics to four decimal places each time |
| The FAISS index is exact | `recall_against_exact()` vs brute force = **1.0000** |
| No label leakage | `FeatureStore.build` asserts its own cutoff; a test confirms the assertion actually fires |
| The grounding guard works | Test injects a deliberately fabricating provider and confirms the fake SKU is caught |
| Latency claims | Measured with a per-stage profiler before and after each optimisation, not estimated |
| The README's Gini claim | Recomputed directly from the final dataset (0.683) rather than carried over from a pre-generation parameter sweep |

**22 tests, no network access or credentials required.** See
[TEST_OUTPUT.md](TEST_OUTPUT.md).

---

## Mistakes identified and corrected

Twelve substantive defects were found and fixed during development. Most were
caught by measurement rather than by reading the code, which is the point worth
noting: AI-generated code that *looks* correct frequently is not.

### 1. The synthetic data was unusable (caught by measurement)

The first generator produced a dataset where **the top 100 items accounted for
94.8% of all purchases**. Popularity so dominated the latent taste signal that a
popularity baseline would have been near-unbeatable and the entire exercise
meaningless.

Fixed by exposing the taste/popularity/price mixing weights as real parameters
and sweeping them. Result: **26.1%** top-100 share, purchase Gini 0.683, 3,974
of 4,000 items selling at least once.

### 2. A configuration knob that did nothing

`popularity_zipf` was documented as controlling demand skew. The parameter
sweep showed **identical results across every value**, which exposed the reason:
the pool scorer z-scores `log(1/rank^a)`, and z-scoring cancels the exponent
exactly. The knob was inert.

Kept for the discovery channel where it genuinely applies, and the docstring
now says the affinity pool is invariant to it. Worth flagging because a
plausible-looking, well-commented, completely inert parameter is exactly the
kind of thing that survives code review.

### 3. ALS was 96 minutes of a 3-minute job

The first ALS implementation solved one 64×64 system per user in a Python loop.
Profiling showed `np.linalg.solve` on a 64×64 matrix taking **11 ms** — roughly
400× slower than it should be.

Root cause: multithreaded OpenBLAS spawns a thread team per tiny solve and
thrashes. Single-threaded, the same batch ran **300× faster** (33 µs/row vs
9,759 µs/row).

Fixed by scoping `threadpool_limits(1)` to the ALS fit and batching the solves.
**241 s per half-step → 1.96 s.**

### 4. The ALS fix introduced a memory blowup

Batching by *row count* meant the batch containing the most popular items padded
every row to that batch's widest (item rows range from a handful of
interactions to **4,328**), allocating multi-gigabyte tensors.

Fixed by budgeting on `rows × width` instead, so every block is the same size in
memory regardless of where it falls in the skew distribution.

### 5–6. The offline LLM provider parsed the wrong text, twice

First it read `messages[-1]` (the question) instead of the system context, so it
found no facts and emitted "No grounded context was supplied" for every
recommendation. Fixed to read the system blocks — which introduced a second bug:
it scraped the *system prompt's own rule bullets* into the evidence, producing
explanations that began "Recommended because Use ONLY the facts in the supplied
context."

Fixed to read only the last system block (the retrieved context).

### 7. Serving was 63× too slow

Initial working service: **2,865 ms p50**. Two profiled fixes:

| Fix | p50 |
|---|---|
| *(initial)* | 2,865 ms |
| Hoist user-tower inputs to startup — they were re-scanning all 1.7M events per request | 163 ms |
| Index the feature blocks and `reindex` instead of `merge` — a 200-row request was rebuilding a hash join against a 700k-row table | **45 ms** |

Per-stage profiling showed the feature join was 133 ms of 155 ms (86%) while the
neural work was 2.2 ms — the opposite of where the time was assumed to be.

### 8. The two-tower model was undertrained

Stopped at 6 epochs with validation recall still climbing. Extended to 18;
it plateaus at ~0.454 around epoch 14–15.

### 9. A dependency incompatibility

FastAPI 0.115.6 against Starlette 1.3.1 failed inside FastAPI's own constructor
(`Router.__init__() got an unexpected keyword argument 'on_startup'`). Resolved
by upgrading FastAPI rather than pinning Starlette backwards.

### 10. The analysis script was designed to produce misleading numbers

The first version reconstructed the ranker's inputs with the candidate-source
columns zeroed out, which would have produced a silent lower bound on the
deployed model's performance. Rejected before it ever ran; the pipeline was
changed to persist the actually-scored frame instead.

### 11. A patch introduced a syntax error

A scripted edit wrote a literal newline inside a string, breaking
`analyse_results.py`. Caught by an `ast.parse` check before the file was run.

### 12. A process mistake, made twice

A background pipeline job was launched *before* the file it executes was
patched, so Python imported the old module and the expected output was never
written. This happened twice and cost about seven minutes. Recorded because it
is a genuine workflow error, not a code defect.

### Also corrected

- `loss.detach().item()` — the training loop triggered a PyTorch warning about
  converting a tensor with `requires_grad` to a scalar.
- `scripts/run_all.py` referenced a module (`scripts.generate_data`) that never
  existed; caught by an importability check over every stage.
- The README claimed "full catalogue coverage" when 3,974 of 4,000 items had a
  purchase. Corrected to the exact figure after checking.

---

## Accepted and rejected AI output

### Rejected or substantially revised

| Output | Why rejected |
|---|---|
| First data-generator parameterisation | Measurement showed 94.8% demand concentration |
| First ALS implementation | Profiling: 96 minutes |
| Second ALS batching strategy | Memory analysis: multi-GB allocations on skewed rows |
| Offline LLM provider (two iterations) | Produced wrong output, verified by reading it |
| First `analyse_results.py` design | Would have reported a misleading lower bound |
| Heredoc-based file writing | Shell quoting corrupted a large Python file; switched to direct file writes |
| **Request to fabricate 20 prompts** | Declined — see [PROMPTS.md](PROMPTS.md) |

### Accepted only after verification

| Output | Verification before acceptance |
|---|---|
| `FeatureStore` merge→reindex rewrite | 42-feature × 11,946-row parity check, then full-pipeline reproduction |
| Metric layer | 9 hand-computed unit tests |
| Two-tower + logQ correction | Validation recall tracked per epoch; best epoch checkpointed |
| Two-stage architecture | Compared against 5 baselines including two that beat the neural retriever alone |

### Accepted as written

Boilerplate and structural code: dataclass configs, FastAPI request/response
models, CSV/JSON report writing, `.gitignore`. Low risk, and errors would
surface immediately on execution.

---

## Honest limitations of this process

- **The PySpark pipeline was never executed.** No JVM was available on the
  development machine. It is covered by a schema-and-semantics contract test and
  is labelled unverified in both the module docstring and the README. It should
  be read as reviewed design, not as working code.
- **Results are on simulated data**, from a generator whose structure was
  chosen here. Relative comparisons under a shared protocol are meaningful; the
  absolute numbers characterise the simulator.
- **No A/B test.** Every number is offline.
- **The transcript is a faithful reconstruction, not a raw log** — see the note
  at the top of [TRANSCRIPT.md](TRANSCRIPT.md).

---

## What this experience suggests about building with AI

Three patterns, stated as observations from this session rather than as
guidance:

**Measurement caught what reading would not.** The 94.8% concentration bug, the
inert config knob, the 11 ms matrix solve, and the 86%-of-latency feature join
were all invisible in code review. Every one was found by running something and
looking at a number.

**The fixes needed their own review.** The ALS speedup introduced a memory bug;
the offline provider needed two corrections; the analysis script's first design
would have produced misleading output. A fix is a change like any other.

**Shared code demands proof, not confidence.** The `FeatureStore` rewrite
touched the path used by both training and serving — the highest-risk edit in
the repo, since training/serving skew does not show up in offline metrics. It
was accepted only after a value-level parity check and a full pipeline
reproduction, and both were worth the time.
