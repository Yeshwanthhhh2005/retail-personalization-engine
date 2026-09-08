# Test Run Output

Terminal output from the test suite, captured on the development machine.
CI runs the same three suites on every push — see
[`.github/workflows/tests.yml`](.github/workflows/tests.yml) and the
**Actions** tab of this repository.

```
Python   3.11.9
Platform Windows 10
numpy 1.26.4 · pandas 2.2.0 · torch 2.9.1+cpu · lightgbm 4.6.0 · faiss 1.15.0
```

**22 tests, 0 failures.** No network access and no API credentials required —
the LLM tests run against the gateway's deterministic offline provider.

---

## Test suite

```console
$ python -m tests.test_metrics
PASS test_complete_miss
PASS test_coverage_and_gini
PASS test_novelty_prefers_rare_items
PASS test_ordering_matters_for_ndcg_not_recall
PASS test_partial_hits_at_k3
PASS test_perfect_single_hit
PASS test_recall_normalises_by_basket_not_k
PASS test_revenue_counts_only_hits
PASS test_unserved_customer_counts_as_miss

0 failure(s)

$ python -m tests.test_genai
PASS test_audit_log_is_written_as_jsonl
PASS test_cache_reads_are_discounted
PASS test_call_budget_is_enforced
PASS test_cost_arithmetic_matches_published_rates
PASS test_grounding_guard_flags_fabricated_ids
PASS test_grounding_passes_when_citations_are_supplied
PASS test_identical_requests_get_identical_request_ids
PASS test_stable_prefix_is_marked_cacheable_and_separate

0 failure(s)

$ python -m tests.test_spark_contract
PASS test_declared_features_exist_in_the_pandas_contract
PASS test_feature_builder_refuses_to_read_past_the_cutoff
PASS test_repurchase_cadence_is_computed_from_history
PASS test_spark_module_imports_without_pyspark
PASS test_unseen_item_recency_is_far_past_not_zero

0 failure(s)
```

---

## What each suite covers

### `tests/test_metrics.py` — 9 tests

The metric layer every model number in this repository depends on. Expected
values are **computed by hand in the test comments**, not snapshotted from the
implementation, so the tests fail if the implementation is wrong rather than
merely if it changes.

| Test | Guards against |
|---|---|
| `test_perfect_single_hit` | basic correctness at k=1 |
| `test_partial_hits_at_k3` | NDCG/MAP arithmetic against hand-computed 0.6934 / 0.5833 |
| `test_complete_miss` | all metrics zero when nothing hits |
| `test_unserved_customer_counts_as_miss` | **silently dropping unscored customers** — the most common way to inflate a recommender's headline metric |
| `test_recall_normalises_by_basket_not_k` | recall denominator confusion |
| `test_ordering_matters_for_ndcg_not_recall` | NDCG actually being rank-sensitive |
| `test_coverage_and_gini` | catalogue-health metrics |
| `test_novelty_prefers_rare_items` | novelty direction (rare > head) |
| `test_revenue_counts_only_hits` | revenue credited only for real conversions |

### `tests/test_genai.py` — 8 tests

Accounting and governance for the LLM gateway. Asserts on prompt construction
and cost arithmetic, never on model output, which is why it is deterministic.

| Test | Guards against |
|---|---|
| `test_cost_arithmetic_matches_published_rates` | wrong billing maths (1M in + 1M out = $30.00 on Opus 5) |
| `test_cache_reads_are_discounted` | cache-read discount applied correctly |
| `test_grounding_guard_flags_fabricated_ids` | **hallucinated SKUs reaching a customer** — injects a deliberately fabricating provider |
| `test_grounding_passes_when_citations_are_supplied` | false positives in that guard |
| `test_stable_prefix_is_marked_cacheable_and_separate` | volatile context leaking into the cached prefix and silently destroying cache hits |
| `test_call_budget_is_enforced` | runaway spend |
| `test_audit_log_is_written_as_jsonl` | missing audit trail |
| `test_identical_requests_get_identical_request_ids` | non-reproducible request identity |

### `tests/test_spark_contract.py` — 5 tests

The PySpark pipeline cannot execute here (no JVM), so what is tested is the
**contract** between it and the pandas implementation — a drift would be silent,
with the Spark job serving a different feature distribution than the model was
trained on.

| Test | Guards against |
|---|---|
| `test_spark_module_imports_without_pyspark` | a JVM dependency leaking into the training path |
| `test_declared_features_exist_in_the_pandas_contract` | the two paths naming different features |
| `test_unseen_item_recency_is_far_past_not_zero` | filling "never bought" with 0 days — telling the model the **exact inverse** of the truth |
| `test_repurchase_cadence_is_computed_from_history` | the replenishment feature's arithmetic |
| `test_feature_builder_refuses_to_read_past_the_cutoff` | **label leakage** — confirms the point-in-time assertion actually fires |

---

## Beyond unit tests

Three checks that are not unit tests but did more to establish correctness:

**Feature-store parity.** The merge→reindex rewrite of `FeatureStore.transform`
touches the code path shared by training and serving. It was verified
value-for-value against the original implementation across **42 features and
11,946 rows — zero mismatches** — and then by re-running the full pipeline to
identical metrics.

**Pipeline reproducibility.** The end-to-end pipeline was run three times,
producing identical metrics each time:

```
two_stage_ranker  recall@20 0.5465  ndcg@20 0.3922  map@20 0.2718  coverage 0.9407
```

**ANN exactness.** `recall_against_exact()` compares the FAISS index against
brute-force search: **1.0000** at the current catalogue size, and the check
exists to make the tradeoff measurable when the index type changes at scale.

---

## Reproducing

```bash
pip install -r requirements.txt

python -m tests.test_metrics
python -m tests.test_genai
python -m tests.test_spark_contract
```

The tests need no generated data. To reproduce the model results as well:

```bash
python -m scripts.run_all      # ~15 min, CPU only
```
