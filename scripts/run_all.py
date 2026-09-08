"""Reproduce every result in the README, in order, from a clean checkout.

    python -m scripts.run_all

Each stage depends on the artifacts of the one before it, so the order is not
negotiable. Total runtime is roughly 15 minutes on a CPU-only laptop; the
two-tower training dominates.
"""
from __future__ import annotations

import argparse
import importlib
import time

STAGES = [
    ("generate data",      "src.data.generate",       "~40s"),
    ("baselines",          "scripts.eval_baselines",  "~90s"),
    ("two-tower retrieval", "scripts.train_retrieval", "~11min"),
    ("two-stage ranker",   "scripts.train_ranker",    "~3min"),
    ("analysis",           "scripts.analyse_results", "~30s"),
    ("genai demo",         "scripts.run_genai_demo",  "~20s"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", type=int, default=0,
                        help="skip ahead, e.g. 3 to start at the ranker")
    args = parser.parse_args()

    started = time.perf_counter()
    for i, (label, module, estimate) in enumerate(STAGES):
        if i < args.from_stage:
            print(f"[{i}] SKIP  {label}")
            continue
        print(f"\n{'=' * 70}\n[{i}] {label}  ({estimate})\n{'=' * 70}", flush=True)
        t0 = time.perf_counter()
        importlib.import_module(module).main()
        print(f"\n[{i}] {label} done in {time.perf_counter() - t0:.0f}s", flush=True)

    print(f"\nall stages complete in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
