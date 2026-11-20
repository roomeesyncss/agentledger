"""Self-contained HTML waterfall for traces. No JavaScript, no external assets."""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from collections.abc import Iterable
from html import escape

from .exporters import fmt_ms, group_traces, summarize
from .span import Span

_CSS = """
:root{--bg:#f7f7f8;--panel:#fff;--text:#18181b;--muted:#71717a;--line:#e4e4e7;--track:#f1f1f3;
--agent:#7c3aed;--llm:#2563eb;--tool:#d97706;--retrieval:#059669;--chain:#64748b;--err:#dc2626;
font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#0b0b0f;--panel:#15151b;--text:#ececf1;
--muted:#9b9ba7;--line:#26262f;--track:#1d1d25}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-size:14px}
main{max-width:1180px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:20px;margin:0 0 4px}.muted{color:var(--muted)}
.trace{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:20px;margin-bottom:24px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;
margin:16px 0 20px}
.stat{border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.stat b{display:block;font-size:18px;font-variant-numeric:tabular-nums}
.stat span{font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:minmax(220px,34%) 1fr 72px;align-items:center;gap:10px}
.axis{font-size:11px;color:var(--muted);padding:0 0 6px;border-bottom:1px solid var(--line)}
.axis .ticks{display:flex;justify-content:space-between}
details{border-bottom:1px solid var(--line)}
summary{list-style:none;cursor:pointer;padding:6px 0}summary::-webkit-details-marker{display:none}
summary:hover{background:var(--track)}
.label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.badge{display:inline-block;font-size:10px;font-weight:600;text-transform:uppercase;
letter-spacing:.04em;padding:1px 6px;border-radius:4px;color:#fff;margin-right:6px;
background:var(--k)}
.track{position:relative;height:14px;background:var(--track);border-radius:3px}
.bar{position:absolute;top:0;bottom:0;border-radius:3px;background:var(--k);min-width:2px}
.dur{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.err .label{color:var(--err)}.err .bar{outline:2px solid var(--err);outline-offset:1px}
pre{margin:4px 0 12px;padding:12px;background:var(--track);border-radius:8px;overflow:auto;
font-size:12px;line-height:1.45;max-height:360px}
.k-agent{--k:var(--agent)}.k-llm{--k:var(--llm)}.k-tool{--k:var(--tool)}
.k-retrieval{--k:var(--retrieval)}.k-other{--k:var(--chain)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--k);
margin-right:5px;vertical-align:-1px}
@media (max-width:640px){.grid{grid-template-columns:1fr 56px}.track{grid-column:1/-1;order:3}
.axis{display:none}}
"""

_KNOWN = {"agent", "llm", "tool", "retrieval"}


def _kind_class(kind: str) -> str:
    return f"k-{kind}" if kind in _KNOWN else "k-other"


def _tree_order(spans: list[Span]) -> list[tuple[Span, int]]:
    ids = {s.span_id for s in spans}
    children: dict[str | None, list[Span]] = defaultdict(list)
    for s in sorted(spans, key=lambda s: s.start_time):
        children[s.parent_id if s.parent_id in ids else None].append(s)
    out: list[tuple[Span, int]] = []

    def walk(s: Span, depth: int) -> None:
        out.append((s, depth))
        for c in children.get(s.span_id, []):
            walk(c, depth + 1)

    for root in children[None]:
        walk(root, 0)
    return out


def _details(s: Span) -> str:
    payload = {
        "kind": s.kind,
        "status": s.status,
        "error": s.error,
        "attributes": s.attributes or None,
        "usage": s.usage.__dict__ if s.usage else None,
        "input": s.input,
        "output": s.output,
        "events": s.events or None,
    }
    clean = {k: v for k, v in payload.items() if v is not None}
    return escape(json.dumps(clean, indent=2, ensure_ascii=False, default=repr))


def _render_trace(spans: list[Span]) -> str:
    summary = summarize(spans)
    t0 = min(s.start_time for s in spans)
    total = max(summary.duration_ms, 1e-6)
    roots = [s for s, d in _tree_order(spans) if d == 0]
    title = roots[0].name if roots else "trace"
    started = dt.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S")
    llm_calls = int(summary.by_kind.get("llm", {}).get("count", 0))

    stats = [
        (fmt_ms(summary.duration_ms), "duration"),
        (str(summary.spans), "spans"),
        (str(llm_calls), "LLM calls"),
        (f"{summary.input_tokens:,} / {summary.output_tokens:,}", "tokens in / out"),
        (f"${summary.cost_usd:.4f}", "est. cost"),
        (str(summary.errors), "errors"),
    ]
    stat_html = "".join(
        f'<div class="stat"><b>{escape(v)}</b><span>{k}</span></div>' for v, k in stats
    )
    ticks = "".join(f"<span>{fmt_ms(total * f)}</span>" for f in (0, 0.25, 0.5, 0.75, 1))

    rows = []
    for s, depth in _tree_order(spans):
        left = (s.start_time - t0) * 1000 / total * 100
        width = max(s.duration_ms / total * 100, 0.15)
        err = " err" if s.status == "error" else ""
        mark = " ✗" if s.status == "error" else ""
        rows.append(
            f'<details class="{_kind_class(s.kind)}{err}"><summary class="grid">'
            f'<div class="label" style="padding-left:{depth * 16}px" title="{escape(s.name)}">'
            f'<span class="badge">{escape(s.kind)}</span>{escape(s.name)}{mark}</div>'
            f'<div class="track"><div class="bar" style="left:{left:.3f}%;width:{width:.3f}%">'
            f'</div></div><div class="dur">{fmt_ms(s.duration_ms)}</div></summary>'
            f"<pre>{_details(s)}</pre></details>"
        )
    legend = "".join(
        f'<span class="{_kind_class(k)}"><i></i>{k}</span>'
        for k in ("agent", "llm", "tool", "retrieval", "chain")
    )
    return (
        f'<section class="trace"><h1>{escape(title)}</h1>'
        f'<div class="muted">trace {escape(spans[0].trace_id[:12])} · started {started}</div>'
        f'<div class="stats">{stat_html}</div>'
        f'<div class="grid axis"><div>span</div><div class="ticks">{ticks}</div><div></div></div>'
        f'{"".join(rows)}<div class="legend">{legend}</div></section>'
    )


def render_html(spans: Iterable[Span], *, title: str = "agentledger") -> str:
    """Render all traces in ``spans`` to a standalone HTML document."""
    traces = group_traces(spans)
    body = "".join(_render_trace(t) for t in traces.values()) or "<p>No spans recorded.</p>"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )
