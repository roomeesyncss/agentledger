<div align="center">

# agentledger

**Lightweight tracing, token & cost accounting for LLM agents.**
See every model call, tool call and retry as a timed tree, with cost rolled up per step. Export to JSONL or a self-contained HTML timeline.

[![CI](https://github.com/roomeesyncss/agentledger/actions/workflows/ci.yml/badge.svg)](https://github.com/roomeesyncss/agentledger/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

```text
research_agent [agent]         1.76s  $0.0457
├─ plan [llm]                  350ms    1840→212   tok  $0.0116
├─ search_docs [retrieval]     120ms
├─ fetch_url [tool]            180ms
├─ fetch_url [tool]            180ms  ✗ TimeoutError: upstream timed out after 180ms
├─ fetch_url [tool]            180ms  ✗ TimeoutError: upstream timed out after 180ms
├─ synthesise [llm]            600ms    5210→640   tok  $0.0340
└─ self_check [llm]            150ms     980→45    tok  $0.0001
── 8 spans · 1.76s · 8,030 in / 897 out tokens · $0.0457 · 2 error(s)
```

## Why

When an agent is slow, expensive or wrong, you need to know **which step** caused it. Full observability platforms are heavy to adopt; `print` debugging doesn't scale. agentledger is a small, zero-dependency library you can drop into any agent loop in minutes.

## Features

- **Nested spans** via `contextvars`. Parent/child links survive `await` and `asyncio.gather`.
- **`@trace` decorator** for sync and async functions, capturing args and return values (bounded, JSON-safe).
- **Token & cost accounting.** `span.record_response(response)` reads Anthropic-style usage, including cache reads and writes, and prices it. The pricing table is overridable.
- **Cost roll-up.** Each node in the tree shows the cost of its whole subtree.
- **Exporters.** JSONL (append-only, crash-safe), console tree, in-memory. A failing exporter never breaks your agent.
- **HTML timeline.** A single self-contained file (no JS, no CDN) with a waterfall per trace and expandable span details.
- **CLI.** `view`, `stats`, `html`.

## Install

```bash
pip install "git+https://github.com/roomeesyncss/agentledger"
```

## Quickstart

```python
from agentledger import ConsoleExporter, JSONLExporter, Tracer

tracer = Tracer([JSONLExporter("traces/run.jsonl"), ConsoleExporter()])


@tracer.trace(kind="tool")
def search_docs(query: str) -> list[str]: ...


with tracer.span("support_agent", kind="agent", input=question) as agent:
    docs = search_docs(question)
    with tracer.span("answer", kind="llm") as call:
        response = client.messages.create(...)
        call.record_response(response)  # tokens + cost
    agent.event("guardrail_passed")
```

Try it without an API key:

```bash
python examples/simulated_agent.py
agentledger view  traces/demo.jsonl
agentledger stats traces/demo.jsonl
agentledger html  traces/demo.jsonl -o traces/demo.html
```

### With Claude

[`examples/claude_tool_agent.py`](examples/claude_tool_agent.py) is a complete tool-using agent (document search + safe calculator) where every model turn and tool call is a span:

```bash
pip install "agentledger[claude] @ git+https://github.com/roomeesyncss/agentledger"
export ANTHROPIC_API_KEY=...
python examples/claude_tool_agent.py "What is 17% of Q3 revenue, and how does Q3 compare to Q2?"
agentledger html traces/claude.jsonl
```

## Pricing

Default prices (USD per million tokens) cover current Claude models. Unknown models still record tokens, just without cost. Override or add your own:

```python
from agentledger import Pricing, Tracer

tracer = Tracer(pricing={"my-finetune": Pricing(input=1.0, output=3.0)})
```

Model names match exactly first, then by longest prefix.

## Development

```bash
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check .
```

## Roadmap

- [ ] OpenTelemetry (OTLP) exporter
- [ ] Sampling for high-volume production traffic
- [ ] Trace diffing between two runs of the same agent

## Author

**Rooma Siddiqui**, AI Engineer (RAG pipelines, AI agents, backend & automation)
GitHub [@roomeesyncss](https://github.com/roomeesyncss)

## License

[MIT](LICENSE) © 2026 Rooma Siddiqui
