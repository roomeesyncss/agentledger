"""Offline demo: a simulated research agent that produces a realistic trace.

No API key needed. Run it, then explore the trace:

    python examples/simulated_agent.py
    agentledger view traces/demo.jsonl
    agentledger html traces/demo.jsonl -o traces/demo.html
"""

from __future__ import annotations

import time
from pathlib import Path

from agentledger import ConsoleExporter, JSONLExporter, Tracer

OUT = Path("traces/demo.jsonl")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("")  # start a fresh trace file for the demo

tracer = Tracer([JSONLExporter(OUT), ConsoleExporter()])


@tracer.trace(kind="retrieval")
def search_docs(query: str, k: int = 4) -> list[str]:
    time.sleep(0.12)
    return [f"doc-{i}: notes about {query}" for i in range(k)]


@tracer.trace(kind="tool")
def fetch_url(url: str) -> str:
    time.sleep(0.18)
    if "flaky" in url:
        raise TimeoutError("upstream timed out after 180ms")
    return f"<html><title>{url}</title>…</html>"


def llm_call(name: str, model: str, seconds: float, tokens_in: int, tokens_out: int, **kw) -> None:
    with tracer.span(name, kind="llm", model=model) as s:
        time.sleep(seconds)  # stands in for the network call
        s.record_usage(model=model, input_tokens=tokens_in, output_tokens=tokens_out, **kw)


def run(question: str) -> str:
    with tracer.span("research_agent", kind="agent", input=question) as agent:
        llm_call("plan", "claude-opus-5-5", 0.35, 1_840, 212)
        docs = search_docs(question)
        agent.set(docs_found=len(docs))

        for url in ("https://example.com/pricing", "https://flaky.example.com/changelog"):
            for attempt in (1, 2):
                try:
                    fetch_url(url)
                    break
                except TimeoutError:
                    agent.event("retry", url=url, attempt=attempt)

        llm_call("synthesise", "claude-opus-5-5", 0.6, 5_210, 640, cache_read_tokens=1_800)
        llm_call("self_check", "claude-haiku-5-5", 0.15, 980, 45)
        agent.output = "Usage-based pricing launched in v2.3; see example.com/pricing."
        return agent.output


if __name__ == "__main__":
    run("When did the product switch to usage-based pricing?")
    print(f"\nwrote {OUT}; try: agentledger html {OUT}")
