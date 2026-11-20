"""Exporters, trace loading, summaries and the text tree view."""

from __future__ import annotations

import json
import sys
import threading
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .span import Span


class InMemoryExporter:
    """Collects spans in a list. Handy in tests."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, span: Span) -> None:
        self.spans.append(span)


class JSONLExporter:
    """Appends one JSON object per finished span. Crash-safe, append-only, greppable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        line = json.dumps(span.to_dict(), ensure_ascii=False, default=repr)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def __repr__(self) -> str:
        return f"JSONLExporter({str(self.path)!r})"


class ConsoleExporter:
    """Prints a trace tree each time a root span finishes."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream
        self._pending: dict[str, list[Span]] = defaultdict(list)
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        with self._lock:
            self._pending[span.trace_id].append(span)
            if span.parent_id is not None:
                return
            spans = self._pending.pop(span.trace_id)
        print(format_tree(spans), file=self.stream or sys.stderr)


def load_spans(path: str | Path) -> list[Span]:
    spans = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                spans.append(Span.from_dict(json.loads(line)))
    return spans


def group_traces(spans: Iterable[Span]) -> dict[str, list[Span]]:
    """Spans grouped by trace id, ordered by each trace's start time."""
    groups: dict[str, list[Span]] = defaultdict(list)
    for s in spans:
        groups[s.trace_id].append(s)
    ordered = sorted(groups.items(), key=lambda kv: min(s.start_time for s in kv[1]))
    return {k: sorted(v, key=lambda s: s.start_time) for k, v in ordered}


# -- summaries ---------------------------------------------------------------------------


@dataclass
class TraceSummary:
    spans: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def summarize(spans: Iterable[Span]) -> TraceSummary:
    spans = list(spans)
    out = TraceSummary(spans=len(spans))
    if not spans:
        return out
    start = min(s.start_time for s in spans)
    end = max((s.end_time or s.start_time) for s in spans)
    out.duration_ms = round((end - start) * 1000, 3)
    for s in spans:
        k = out.by_kind.setdefault(s.kind, {"count": 0, "errors": 0, "total_ms": 0.0})
        k["count"] += 1
        k["total_ms"] = round(k["total_ms"] + s.duration_ms, 3)
        if s.status == "error":
            out.errors += 1
            k["errors"] += 1
        u = s.usage
        if u:
            out.input_tokens += u.input_tokens
            out.output_tokens += u.output_tokens
            out.cache_read_tokens += u.cache_read_tokens
            out.cache_write_tokens += u.cache_write_tokens
            out.cost_usd += u.cost_usd or 0.0
            m = out.by_model.setdefault(
                u.model or "unknown",
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            m["calls"] += 1
            m["input_tokens"] += u.input_tokens
            m["output_tokens"] += u.output_tokens
            m["cost_usd"] = round(m["cost_usd"] + (u.cost_usd or 0.0), 8)
    out.cost_usd = round(out.cost_usd, 8)
    return out


# -- text tree ---------------------------------------------------------------------------


def fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"


def fmt_cost(usd: float) -> str:
    return f"${usd:.4f}" if usd else ""


def _subtree_cost(span: Span, children: dict[str | None, list[Span]]) -> float:
    own = span.usage.cost_usd if span.usage and span.usage.cost_usd else 0.0
    return own + sum(_subtree_cost(c, children) for c in children.get(span.span_id, []))


def format_tree(spans: Iterable[Span]) -> str:
    """Render one or more traces as an indented tree with timings, tokens and cost."""
    lines: list[str] = []
    for trace_spans in group_traces(spans).values():
        ids = {s.span_id for s in trace_spans}
        children: dict[str | None, list[Span]] = defaultdict(list)
        for s in trace_spans:
            parent = s.parent_id if s.parent_id in ids else None
            children[parent].append(s)

        rows: list[tuple[str, Span]] = []

        def walk(
            span: Span,
            prefix: str,
            is_last: bool,
            is_root: bool,
            rows: list[tuple[str, Span]] = rows,
            children: dict[str | None, list[Span]] = children,
        ) -> None:
            branch = "" if is_root else ("└─ " if is_last else "├─ ")
            rows.append((prefix + branch, span))
            kids = children.get(span.span_id, [])
            ext = "" if is_root else ("   " if is_last else "│  ")
            for i, c in enumerate(kids):
                walk(c, prefix + ext, i == len(kids) - 1, False)

        for root in children[None]:
            walk(root, "", True, True)

        width = max(len(p) + len(s.name) + len(s.kind) + 3 for p, s in rows)
        for prefix, s in rows:
            label = f"{prefix}{s.name} [{s.kind}]"
            parts = [label.ljust(width), fmt_ms(s.duration_ms).rjust(8)]
            if s.usage:
                parts.append(f"{s.usage.input_tokens:>6}→{s.usage.output_tokens:<5} tok")
            cost = _subtree_cost(s, children)
            if cost:
                parts.append(fmt_cost(cost))
            if s.status == "error":
                parts.append(f"✗ {s.error}")
            lines.append("  ".join(parts).rstrip())
        summary = summarize(trace_spans)
        lines.append(
            f"── {summary.spans} spans · {fmt_ms(summary.duration_ms)} · "
            f"{summary.input_tokens:,} in / {summary.output_tokens:,} out tokens · "
            f"${summary.cost_usd:.4f} · {summary.errors} error(s)"
        )
        lines.append("")
    return "\n".join(lines).rstrip("\n")
