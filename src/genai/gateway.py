"""Managed LLM gateway.

Everything that talks to a model goes through here, so that the things an
organisation actually has to answer for -- what did we spend, what did we
send, what came back, can we reproduce it -- have exactly one place to live.

What the gateway owns:

  * Provider binding      Anthropic Claude via the official SDK.
  * Prompt versioning     every call records which prompt template produced it.
  * Caching               the stable system prefix is marked cacheable, so the
                          catalogue context is billed once per window, not once
                          per request.
  * Cost accounting       per-call token and dollar attribution, including the
                          cache-read discount.
  * Governance            structured audit records, latency, a grounding check
                          on the response, and an offline mode so the pipeline
                          is runnable and testable without a live key.

The offline provider is a deterministic stub, not a second vendor: it exists so
CI and this assignment run end to end with no credentials, and so tests assert
on prompt construction rather than on model output.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

# Claude Opus 5. Input $5.00 / output $25.00 per million tokens; cache reads
# bill at ~0.1x input and cache writes at ~1.25x.
DEFAULT_MODEL = "claude-opus-5"
PRICING = {
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def cost_usd(self, model: str) -> float:
        price = PRICING.get(model, PRICING[DEFAULT_MODEL])
        return (
            self.input_tokens * price["input"]
            + self.cache_read_tokens * price["input"] * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * price["input"] * CACHE_WRITE_MULTIPLIER
            + self.output_tokens * price["output"]
        ) / 1_000_000


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Usage
    latency_ms: float
    prompt_version: str
    request_id: str
    provider: str
    cached: bool = False
    grounding: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    name: str

    def complete(
        self, system: list[dict], messages: list[dict], max_tokens: int, model: str
    ) -> tuple[str, Usage]: ...


class AnthropicProvider:
    """Claude via the official SDK.

    Adaptive thinking is left on (the default on Opus 5) because the
    explanation task involves reconciling several numeric signals, and
    `effort` is the lever we tune rather than a token budget.
    """

    name = "anthropic"

    def __init__(self, effort: str = "low", timeout: float = 30.0):
        import anthropic  # imported lazily so offline mode needs no dependency

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=timeout)
        self.effort = effort

    def complete(
        self, system: list[dict], messages: list[dict], max_tokens: int, model: str
    ) -> tuple[str, Usage]:
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            # Explanations are short by design; the ceiling is a guard, not a
            # target. Effort stays low because this is a formatting-and-
            # grounding task, not a reasoning-heavy one.
            output_config={"effort": self.effort},
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        )
        return text, usage


class OfflineProvider:
    """Deterministic stand-in used when no API key is configured.

    It renders the grounded facts the prompt already contains into a fixed
    template. That keeps the pipeline runnable end to end, keeps CI free, and
    makes the prompt-construction tests deterministic -- but it is explicitly
    not a language model, and every audit record says so via `provider`.
    """

    name = "offline"

    def complete(
        self, system: list[dict], messages: list[dict], max_tokens: int, model: str
    ) -> tuple[str, Usage]:
        prompt = messages[-1]["content"]
        if isinstance(prompt, list):
            prompt = " ".join(b.get("text", "") for b in prompt)

        # The retrieved evidence is the LAST system block. Reading the whole
        # system text would scrape the rule bullets out of the prefix.
        context = system[-1].get("text", "") if system else ""
        facts = re.findall(r"- \[(item_\d+)\] ([^\n]+)", context)
        evidence = re.findall(r"^- (?!\[)([^\n]+)$", context, flags=re.MULTILINE)

        if facts:
            tag, description = facts[0]
            body = f"{description.strip()} [{tag}]"
            if evidence:
                body = f"Recommended because {evidence[0].strip()}. " + body
            if len(facts) > 1:
                body += f" Comparable to [{facts[1][0]}]."
        else:
            body = "No grounded context was supplied for this recommendation."

        system_text = "\n".join(b.get("text", "") for b in system)

        usage = Usage(
            input_tokens=len(system_text.split()) + len(str(prompt).split()),
            output_tokens=len(body.split()),
        )
        return body, usage


@dataclass
class AuditRecord:
    request_id: str
    timestamp: float
    provider: str
    model: str
    prompt_version: str
    prompt_hash: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float
    grounded: bool
    ungrounded_ids: list[str]
    truncated: bool


class LLMGateway:
    """Single entry point for model calls, with accounting and governance."""

    def __init__(
        self,
        provider: Provider | None = None,
        model: str = DEFAULT_MODEL,
        audit_path: Path | None = None,
        max_calls: int | None = None,
    ):
        self.model = model
        self.audit_path = audit_path
        self.max_calls = max_calls
        self.records: list[AuditRecord] = []
        self.provider = provider or self._auto_provider()

    @staticmethod
    def _auto_provider() -> Provider:
        """Use Claude when credentials exist, the offline stub otherwise."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return OfflineProvider()
        try:
            return AnthropicProvider()
        except Exception:
            # A missing SDK or a bad key must not take the pipeline down --
            # explanations are an enhancement, not a serving dependency.
            return OfflineProvider()

    def complete(
        self,
        system_prefix: str,
        context: str,
        question: str,
        prompt_version: str,
        allowed_ids: set[str] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Run one grounded completion.

        `system_prefix` is the stable, cacheable half of the prompt and
        `context` the retrieved evidence. They are separate blocks so the
        prefix stays byte-identical across requests -- a cache read is ~10% of
        the input price, and a single varying character in the prefix forfeits
        all of it.
        """
        if self.max_calls is not None and len(self.records) >= self.max_calls:
            raise RuntimeError(
                f"gateway call budget exhausted ({self.max_calls} calls)"
            )

        system = [
            {"type": "text", "text": system_prefix,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": context},
        ]
        messages = [{"role": "user", "content": question}]

        request_id = hashlib.sha256(
            f"{prompt_version}{context}{question}".encode()
        ).hexdigest()[:16]

        t0 = time.perf_counter()
        text, usage = self.provider.complete(system, messages, max_tokens, self.model)
        latency_ms = (time.perf_counter() - t0) * 1000

        grounding = self._check_grounding(text, allowed_ids)
        record = AuditRecord(
            request_id=request_id,
            timestamp=time.time(),
            provider=self.provider.name,
            model=self.model,
            prompt_version=prompt_version,
            prompt_hash=hashlib.sha256(system_prefix.encode()).hexdigest()[:12],
            latency_ms=round(latency_ms, 1),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cost_usd=round(usage.cost_usd(self.model), 6),
            grounded=grounding["grounded"],
            ungrounded_ids=grounding["ungrounded_ids"],
            truncated=usage.output_tokens >= max_tokens,
        )
        self.records.append(record)
        if self.audit_path:
            self._append_audit(record)

        return LLMResponse(
            text=text, model=self.model, usage=usage, latency_ms=latency_ms,
            prompt_version=prompt_version, request_id=request_id,
            provider=self.provider.name, grounding=grounding,
        )

    @staticmethod
    def _check_grounding(text: str, allowed_ids: set[str] | None) -> dict[str, Any]:
        """Verify every item the answer cites was actually in the context.

        This is the cheap, deterministic half of hallucination control: the
        model may only reference evidence we supplied, and any id outside that
        set is flagged rather than shown. It catches fabricated SKUs, which is
        the failure mode that would put a wrong product in front of a customer.
        """
        cited = set(re.findall(r"\[(item_\d+)\]", text))
        if allowed_ids is None:
            return {"grounded": True, "cited_ids": sorted(cited), "ungrounded_ids": []}
        ungrounded = sorted(cited - allowed_ids)
        return {
            "grounded": not ungrounded,
            "cited_ids": sorted(cited),
            "ungrounded_ids": ungrounded,
        }

    def _append_audit(self, record: AuditRecord) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"calls": 0}
        return {
            "calls": len(self.records),
            "provider": self.records[-1].provider,
            "model": self.model,
            "total_cost_usd": round(sum(r.cost_usd for r in self.records), 6),
            "total_input_tokens": sum(r.input_tokens for r in self.records),
            "total_output_tokens": sum(r.output_tokens for r in self.records),
            "cache_read_tokens": sum(r.cache_read_tokens for r in self.records),
            "p50_latency_ms": sorted(r.latency_ms for r in self.records)[
                len(self.records) // 2
            ],
            "grounding_failures": sum(1 for r in self.records if not r.grounded),
            "truncations": sum(1 for r in self.records if r.truncated),
        }
