"""Tests for the LLM gateway's accounting and governance guarantees.

These assert on prompt construction, cost arithmetic and the grounding guard --
never on model output, which is why they run deterministically against the
offline provider with no credentials.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.genai.gateway import (
    DEFAULT_MODEL,
    LLMGateway,
    OfflineProvider,
    Usage,
)

TOL = 1e-9


class FabricatingProvider:
    """Returns an item id that was never supplied in the context."""

    name = "test-fabricator"

    def complete(self, system, messages, max_tokens, model):
        return "You should buy [item_99999], it is excellent.", Usage(10, 5)


def test_cost_arithmetic_matches_published_rates():
    # 1M input + 1M output on Opus 5 = $5 + $25.
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(usage.cost_usd("claude-opus-5") - 30.0) < TOL


def test_cache_reads_are_discounted():
    cached = Usage(input_tokens=0, cache_read_tokens=1_000_000)
    uncached = Usage(input_tokens=1_000_000)
    # Cache reads bill at ~10% of the input rate.
    assert abs(cached.cost_usd(DEFAULT_MODEL) - 0.5) < TOL
    assert abs(uncached.cost_usd(DEFAULT_MODEL) - 5.0) < TOL


def test_grounding_guard_flags_fabricated_ids():
    gw = LLMGateway(provider=FabricatingProvider())
    resp = gw.complete(
        system_prefix="rules", context="- [item_1] a real product",
        question="why?", prompt_version="v1", allowed_ids={"item_1"},
    )
    assert resp.grounding["grounded"] is False
    assert resp.grounding["ungrounded_ids"] == ["item_99999"]
    assert gw.summary()["grounding_failures"] == 1


def test_grounding_passes_when_citations_are_supplied():
    class Good:
        name = "good"

        def complete(self, system, messages, max_tokens, model):
            return "Try [item_1].", Usage(5, 3)

    gw = LLMGateway(provider=Good())
    resp = gw.complete(
        system_prefix="rules", context="- [item_1] a real product",
        question="why?", prompt_version="v1", allowed_ids={"item_1"},
    )
    assert resp.grounding["grounded"] is True
    assert gw.summary()["grounding_failures"] == 0


def test_stable_prefix_is_marked_cacheable_and_separate():
    """The volatile context must not share a block with the cached prefix."""
    captured = {}

    class Capture:
        name = "capture"

        def complete(self, system, messages, max_tokens, model):
            captured["system"] = system
            return "ok", Usage(1, 1)

    gw = LLMGateway(provider=Capture())
    gw.complete(system_prefix="STABLE", context="VOLATILE",
                question="q", prompt_version="v1")

    system = captured["system"]
    assert system[0]["text"] == "STABLE"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    # The changing half must come after the breakpoint, or every request
    # invalidates the cache it was supposed to hit.
    assert system[1]["text"] == "VOLATILE"
    assert "cache_control" not in system[1]


def test_call_budget_is_enforced():
    gw = LLMGateway(provider=OfflineProvider(), max_calls=2)
    for _ in range(2):
        gw.complete("s", "- [item_1] x", "q", "v1")
    try:
        gw.complete("s", "- [item_1] x", "q", "v1")
    except RuntimeError as exc:
        assert "budget exhausted" in str(exc)
    else:
        raise AssertionError("expected the call budget to be enforced")


def test_audit_log_is_written_as_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audit.jsonl"
        gw = LLMGateway(provider=OfflineProvider(), audit_path=path)
        gw.complete("s", "- [item_1] x", "q", "v1", allowed_ids={"item_1"})
        gw.complete("s", "- [item_2] y", "q", "v1", allowed_ids={"item_2"})

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        for key in ("request_id", "provider", "model", "prompt_version",
                    "cost_usd", "grounded", "latency_ms"):
            assert key in record


def test_identical_requests_get_identical_request_ids():
    gw = LLMGateway(provider=OfflineProvider())
    a = gw.complete("s", "- [item_1] x", "q", "v1")
    b = gw.complete("s", "- [item_1] x", "q", "v1")
    c = gw.complete("s", "- [item_2] y", "q", "v1")
    assert a.request_id == b.request_id
    assert a.request_id != c.request_id


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
