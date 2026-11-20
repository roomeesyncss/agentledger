"""Span and usage data model."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

SpanKind = str  # "agent" | "llm" | "tool" | "retrieval" | "chain" | any custom label


@dataclass
class Usage:
    """Token usage for one or more model calls."""

    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass
class Span:
    """One timed unit of agent work: an LLM call, a tool call, a retrieval step..."""

    name: str
    kind: SpanKind
    trace_id: str
    span_id: str
    parent_id: str | None
    start_time: float
    end_time: float | None = None
    status: str = "ok"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    input: Any = None
    output: Any = None
    usage: Usage | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def set(self, **attributes: Any) -> Span:
        """Attach attributes, e.g. ``span.set(retriever="bm25", k=8)``."""
        self.attributes.update(attributes)
        return self

    def event(self, name: str, **attributes: Any) -> None:
        """Record a point-in-time event such as a retry or a guardrail trip."""
        self.events.append({"name": name, "time": time.time(), **attributes})

    def record_usage(
        self,
        *,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> Usage:
        """Add token usage to this span. Calls accumulate; cost is computed on finish."""
        u = self.usage or Usage()
        u.model = model or u.model
        u.input_tokens += input_tokens
        u.output_tokens += output_tokens
        u.cache_read_tokens += cache_read_tokens
        u.cache_write_tokens += cache_write_tokens
        self.usage = u
        return u

    def record_response(self, response: Any) -> Usage:
        """Record usage from an Anthropic Messages API response (duck-typed, no import).

        Reads ``response.model`` and ``response.usage.{input_tokens, output_tokens,
        cache_read_input_tokens, cache_creation_input_tokens}``, plus ``stop_reason``.
        """
        usage = getattr(response, "usage", None)

        def n(attr: str) -> int:
            return int(getattr(usage, attr, 0) or 0)

        stop = getattr(response, "stop_reason", None)
        if stop:
            self.attributes["stop_reason"] = stop
        return self.record_usage(
            model=getattr(response, "model", None),
            input_tokens=n("input_tokens"),
            output_tokens=n("output_tokens"),
            cache_read_tokens=n("cache_read_input_tokens"),
            cache_write_tokens=n("cache_creation_input_tokens"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration_ms"] = round(self.duration_ms, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Span:
        d = dict(d)
        d.pop("duration_ms", None)
        usage = d.pop("usage", None)
        span = cls(**d)
        span.usage = Usage(**usage) if usage else None
        return span
