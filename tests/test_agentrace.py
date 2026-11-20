import asyncio
import io
import json
from types import SimpleNamespace

import pytest

import agentledger
from agentledger import (
    ConsoleExporter,
    InMemoryExporter,
    JSONLExporter,
    Pricing,
    Tracer,
    current_span,
    format_tree,
    load_spans,
    render_html,
    summarize,
)
from agentledger.cli import main


@pytest.fixture
def tracer():
    return Tracer([InMemoryExporter()])


def fake_response(model="claude-opus-5-5", inp=1000, out=500, cache_read=0, cache_write=0):
    usage = SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )
    return SimpleNamespace(model=model, usage=usage, stop_reason="end_turn")


def test_nesting_links_parent_and_trace(tracer):
    with tracer.span("agent", "agent") as root:
        assert current_span() is root
        with tracer.span("step", "llm") as child, tracer.span("tool", "tool") as leaf:
            pass
    assert current_span() is None
    assert root.parent_id is None
    assert child.parent_id == root.span_id and leaf.parent_id == child.span_id
    assert {s.trace_id for s in tracer.spans} == {root.trace_id}
    assert [s.name for s in tracer.spans] == ["tool", "step", "agent"]  # completion order
    assert root.duration_ms >= child.duration_ms >= 0


def test_separate_roots_get_separate_traces(tracer):
    with tracer.span("a"):
        pass
    with tracer.span("b"):
        pass
    a, b = tracer.spans
    assert a.trace_id != b.trace_id


def test_errors_are_recorded_and_reraised(tracer):
    with pytest.raises(ValueError), tracer.span("boom", "tool"):
        raise ValueError("bad input")
    (s,) = tracer.spans
    assert s.status == "error" and s.error == "ValueError: bad input"
    assert s.end_time is not None


def test_decorator_sync_bare_and_configured(tracer):
    class Agent:
        @tracer.trace
        def plan(self, goal: str) -> list[str]:
            return [goal, "verify"]

    @tracer.trace("lookup", kind="tool")
    def lookup(key: str, default: int = 0) -> int:
        return len(key) + default

    assert Agent().plan("ship") == ["ship", "verify"]
    assert lookup("abc", default=1) == 4
    plan, look = tracer.spans
    assert plan.name.endswith("Agent.plan") and plan.input == {"goal": "ship"}
    assert plan.output == ["ship", "verify"]
    assert look.name == "lookup" and look.kind == "tool"
    assert look.input == {"key": "abc", "default": 1} and look.output == 4


def test_decorator_capture_io_off(tracer):
    @tracer.trace(capture_io=False)
    def secret(token: str) -> str:
        return token

    secret("s3cr3t")
    (s,) = tracer.spans
    assert s.input is None and s.output is None


def test_async_children_keep_parent_across_gather(tracer):
    @tracer.trace(kind="tool")
    async def fetch(i: int) -> int:
        await asyncio.sleep(0.01)
        return i * 2

    async def agent():
        with tracer.span("agent", "agent") as root:
            results = await asyncio.gather(fetch(1), fetch(2), fetch(3))
        return root, results

    root, results = asyncio.run(agent())
    assert results == [2, 4, 6]
    children = [s for s in tracer.spans if s.name.endswith("fetch")]
    assert len(children) == 3 and all(c.parent_id == root.span_id for c in children)
    # Concurrent: the parent is far shorter than the sum of its children.
    assert root.duration_ms < sum(c.duration_ms for c in children)


def test_record_response_and_cost(tracer):
    with tracer.span("call", "llm") as s:
        s.record_response(fake_response(inp=1000, out=500, cache_read=10_000))
    u = s.usage
    assert (u.model, u.input_tokens, u.output_tokens, u.cache_read_tokens) == (
        "claude-opus-5-5",
        1000,
        500,
        10_000,
    )
    # $4/M in, $20/M out, $0.20/M cache read
    assert u.cost_usd == pytest.approx(0.004 + 0.010 + 0.002)
    assert s.attributes["stop_reason"] == "end_turn"


def test_pricing_prefix_override_and_unknown():
    t = Tracer(pricing={"my-model": Pricing(input=1.0, output=2.0)})
    with t.span("a", "llm") as a:
        a.record_usage(model="my-model-2026", input_tokens=1_000_000, output_tokens=1_000_000)
    with t.span("b", "llm") as b:
        b.record_usage(model="mystery", input_tokens=10)
    assert a.usage.cost_usd == pytest.approx(3.0)
    assert b.usage.cost_usd is None


def test_usage_accumulates(tracer):
    with tracer.span("multi", "llm") as s:
        s.record_usage(model="claude-haiku-5-5", input_tokens=100, output_tokens=10)
        s.record_usage(input_tokens=50, output_tokens=5)
    assert (s.usage.input_tokens, s.usage.output_tokens, s.usage.model) == (
        150,
        15,
        "claude-haiku-5-5",
    )


def test_safe_serialisation_bounds_data():
    t = Tracer(max_io_chars=10)
    assert t.safe("x" * 25) == "x" * 10 + "…(+15)"
    assert t.safe({"a": (1, 2), "b": {1, 2}}) in (
        {"a": [1, 2], "b": [1, 2]},
        {"a": [1, 2], "b": [2, 1]},
    )
    assert isinstance(t.safe(object()), str)
    deep = {"l1": {"l2": {"l3": {"l4": {"l5": 1}}}}}
    assert isinstance(t.safe(deep)["l1"]["l2"]["l3"]["l4"], str)


def test_events(tracer):
    with tracer.span("agent") as s:
        s.event("retry", attempt=2)
    assert s.events[0]["name"] == "retry" and s.events[0]["attempt"] == 2


def test_jsonl_roundtrip_and_tree(tmp_path):
    path = tmp_path / "t.jsonl"
    t = Tracer([JSONLExporter(path)])
    with t.span("agent", "agent"):
        with t.span("llm", "llm") as s:
            s.record_response(fake_response())
        with pytest.raises(KeyError), t.span("tool", "tool"):
            raise KeyError("missing")
    spans = load_spans(path)
    assert [s.name for s in spans] == ["llm", "tool", "agent"]
    assert spans[0].usage.cost_usd == pytest.approx(0.014)
    tree = format_tree(spans)
    assert tree.splitlines()[0].startswith("agent [agent]")
    assert "├─ llm [llm]" in tree and "└─ tool [tool]" in tree
    assert "✗ KeyError: 'missing'" in tree
    assert "$0.0140" in tree.splitlines()[0]  # cost rolls up to the root


def test_console_exporter_prints_once_per_trace():
    buf = io.StringIO()
    t = Tracer([ConsoleExporter(buf)])
    with t.span("root"), t.span("child"):
        pass
    out = buf.getvalue()
    assert out.count("root [chain]") == 1 and "└─ child [chain]" in out


def test_failing_exporter_does_not_break_agent():
    class Broken:
        def export(self, span):
            raise OSError("disk full")

    t = Tracer([Broken()])
    with pytest.warns(UserWarning, match="disk full"), t.span("work"):
        pass
    assert len(t.spans) == 1


def test_summarize(tracer):
    with tracer.span("agent", "agent"):
        with tracer.span("a", "llm") as a:
            a.record_response(fake_response(inp=100, out=10))
        with tracer.span("b", "llm") as b:
            b.record_response(fake_response(model="claude-haiku-5-5", inp=200, out=20))
    s = summarize(tracer.spans)
    assert (s.spans, s.errors, s.input_tokens, s.output_tokens) == (3, 0, 300, 30)
    assert s.by_kind["llm"]["count"] == 2
    assert set(s.by_model) == {"claude-opus-5-5", "claude-haiku-5-5"}


def test_html_is_self_contained_and_escaped(tracer):
    with tracer.span("<script>alert(1)</script>", "agent"), tracer.span("call", "llm") as s:
        s.record_response(fake_response())
    page = render_html(tracer.spans)
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    assert page.startswith("<!doctype html>") and "http" not in page.split("<body>")[0]
    assert page.count("<details") == 2 and "$0.0140" in page


def test_module_level_configure_resolves_at_call_time():
    @agentledger.trace(kind="tool")
    def work() -> int:
        return 1

    mem = InMemoryExporter()
    agentledger.configure(mem)
    work()
    with agentledger.span("manual"):
        pass
    assert [s.kind for s in mem.spans] == ["tool", "chain"]
    agentledger.configure()  # restore a clean global tracer


def test_cli_commands(tmp_path, capsys):
    path = tmp_path / "trace.jsonl"
    t = Tracer([JSONLExporter(path)])
    for _ in range(2):
        with t.span("agent", "agent"), t.span("call", "llm") as s:
            s.record_response(fake_response())

    assert main(["view", str(path), "--last", "1"]) == 0
    assert capsys.readouterr().out.count("agent [agent]") == 1

    assert main(["stats", str(path), "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["spans"] == 4 and stats["cost_usd"] == pytest.approx(0.028)

    assert main(["stats", str(path)]) == 0
    assert "claude-opus-5-5" in capsys.readouterr().out

    out = tmp_path / "t.html"
    assert main(["html", str(path), "-o", str(out)]) == 0
    assert out.read_text().count("<section") == 2
