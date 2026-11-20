"""Command-line interface.

agentledger view  traces.jsonl                  # tree with timings, tokens, cost
agentledger stats traces.jsonl [--json]         # aggregate by kind and model
agentledger html  traces.jsonl -o trace.html    # self-contained waterfall
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .exporters import fmt_ms, format_tree, load_spans, summarize
from .html import render_html


def _load(path: str):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"agentledger: {path}: no such file")
    return load_spans(p)


def cmd_view(args: argparse.Namespace) -> int:
    spans = _load(args.file)
    if args.last:
        trace_ids = list(
            dict.fromkeys(s.trace_id for s in sorted(spans, key=lambda s: s.start_time))
        )
        keep = set(trace_ids[-args.last :])
        spans = [s for s in spans if s.trace_id in keep]
    print(format_tree(spans))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    summary = summarize(_load(args.file))
    if args.json:
        json.dump(summary.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    print(f"spans      {summary.spans}  ({summary.errors} errors)")
    print(f"wall time  {fmt_ms(summary.duration_ms)}")
    print(
        f"tokens     {summary.input_tokens:,} in · {summary.output_tokens:,} out · "
        f"{summary.cache_read_tokens:,} cache-read · {summary.cache_write_tokens:,} cache-write"
    )
    print(f"cost       ${summary.cost_usd:.4f}")
    print("\nby kind")
    for kind, k in sorted(summary.by_kind.items(), key=lambda kv: -kv[1]["total_ms"]):
        count, errors = int(k["count"]), int(k["errors"])
        print(f"  {kind:<10} {count:>4}x  {fmt_ms(k['total_ms']):>8}  {errors} err")
    if summary.by_model:
        print("\nby model")
        for model, m in sorted(summary.by_model.items(), key=lambda kv: -kv[1]["cost_usd"]):
            print(f"  {model:<22} {int(m['calls']):>4}x  ${m['cost_usd']:.4f}")
    return 0


def cmd_html(args: argparse.Namespace) -> int:
    spans = _load(args.file)
    out = Path(args.output or Path(args.file).with_suffix(".html"))
    out.write_text(
        render_html(spans, title=f"agentledger · {Path(args.file).name}"), encoding="utf-8"
    )
    print(f"agentledger: wrote {out}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentledger", description="Inspect agent traces.")
    parser.add_argument("--version", action="version", version=f"agentledger {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("view", help="print traces as a tree")
    v.add_argument("file")
    v.add_argument("--last", type=int, help="only the N most recent traces")
    v.set_defaults(func=cmd_view)

    s = sub.add_parser("stats", help="aggregate timings, tokens and cost")
    s.add_argument("file")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stats)

    h = sub.add_parser("html", help="render a self-contained HTML timeline")
    h.add_argument("file")
    h.add_argument("-o", "--output")
    h.set_defaults(func=cmd_html)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
