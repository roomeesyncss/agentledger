"""A traced tool-using agent on Claude: every model call and tool call becomes a span.

pip install "agentledger[claude] @ git+https://github.com/roomeesyncss/agentledger"
export ANTHROPIC_API_KEY=...
python examples/claude_tool_agent.py "What is 17% of Q3 revenue, and how does Q3 compare to Q2?"
agentledger html traces/claude.jsonl
"""

from __future__ import annotations

import ast
import operator
import sys

import anthropic

from agentledger import ConsoleExporter, JSONLExporter, Tracer

MODEL = "claude-opus-5-5"
tracer = Tracer([JSONLExporter("traces/claude.jsonl"), ConsoleExporter()])

DOCS = {
    "q3-report": "Q3 revenue was $4.2M, up 18% quarter over quarter. Gross margin was 71%.",
    "q2-report": "Q2 revenue was $3.56M. Monthly churn fell to 2.1%.",
    "hiring-plan": "We plan to hire 6 engineers in Q4, mostly for the platform team.",
}

TOOLS = [
    {
        "name": "search_docs",
        "description": "Keyword search over internal company documents. "
        "Returns each matching document with its id.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords to look for"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression using + - * / ** % and parentheses. "
        "Use it for any arithmetic instead of computing in your head.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
]


@tracer.trace("search_docs", kind="tool")
def search_docs(query: str) -> str:
    terms = {t.lower() for t in query.split()}
    hits = [
        f"[{doc_id}] {text}"
        for doc_id, text in DOCS.items()
        if terms & set(text.lower().replace(".", " ").split()) or terms & {doc_id}
    ]
    return "\n".join(hits) or "No documents matched."


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numeric arithmetic is supported")


@tracer.trace("calculate", kind="tool")
def calculate(expression: str) -> str:
    if len(expression) > 200:
        raise ValueError("expression too long")
    return str(_eval(ast.parse(expression, mode="eval")))


HANDLERS = {"search_docs": search_docs, "calculate": calculate}


def run_agent(question: str, max_turns: int = 8) -> str:
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": question}]

    with tracer.span("tool_agent", kind="agent", input=question) as agent:
        for turn in range(1, max_turns + 1):
            with tracer.span(f"claude turn {turn}", kind="llm") as call:
                response = client.beta.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    output_config={"effort": "medium"},
                    # If a safety classifier declines, the API retries on a fallback model.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    tools=TOOLS,
                    messages=messages,
                )
                call.record_response(response)

            # Append the full content (thinking + tool_use blocks) so the history stays valid.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason == "refusal":
                agent.set(outcome="refusal")
                return "The model declined this request."

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:  # end_turn / max_tokens
                answer = "".join(b.text for b in response.content if b.type == "text")
                agent.output = answer
                return answer

            results = []
            for block in tool_uses:
                try:
                    content, is_error = HANDLERS[block.name](**block.input), False
                except Exception as exc:  # report tool failures back to the model
                    content, is_error = f"{type(exc).__name__}: {exc}", True
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

        raise RuntimeError(f"agent did not finish within {max_turns} turns")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is 17% of Q3 revenue, and how does Q3 compare to Q2?"
    print(run_agent(q))
