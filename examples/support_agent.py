"""Offline demo: a customer-support agent handling a refund ticket.

No API key needed. Model calls are simulated with realistic latencies and token counts.
Everything else is real logic: the refund policy check, a rate-limited order API with
exponential backoff, and a policy guardrail on the drafted reply.

    python examples/support_agent.py
    agentledger view  traces/support.jsonl
    agentledger stats traces/support.jsonl
    agentledger html  traces/support.jsonl -o traces/support.html
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path

from agentledger import ConsoleExporter, JSONLExporter, Tracer, current_span

OUT = Path("traces/support.jsonl")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("")  # fresh trace file per run

tracer = Tracer([JSONLExporter(OUT), ConsoleExporter()])

FAST_MODEL = "claude-haiku-5-5"  # cheap routing and checks
SMART_MODEL = "claude-opus-5-5"  # customer-facing writing

TICKET = {
    "id": "TCK-20931",
    "customer": "CUS-7781",
    "tier": "pro",
    "text": (
        "Hi, my order ORD-48213 arrived but the headphones crackle in the left ear. "
        "I'd like a refund rather than a replacement please."
    ),
}

KNOWLEDGE_BASE = {
    "refund-policy": "Defective items can be refunded within 30 days of delivery.",
    "replacements": "Replacements ship in 2-4 business days once the item is returned.",
    "shipping-delays": "Carrier delays over 7 days qualify for a shipping credit.",
    "payment-methods": "Refunds go back to the original payment method in 5-10 days.",
}


@dataclass
class Order:
    id: str
    item: str
    amount: float
    delivered_on: dt.date


class RateLimitError(Exception):
    pass


_order_api_calls = 0


def simulated_llm(
    name: str,
    model: str,
    seconds: float,
    tokens_in: int,
    tokens_out: int,
    cache_read: int = 0,
    **attrs,
) -> None:
    with tracer.span(name, kind="llm", model=model, **attrs) as s:
        time.sleep(seconds)  # stands in for the API round-trip
        s.record_usage(
            model=model,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cache_read_tokens=cache_read,
        )


@tracer.trace(kind="retrieval")
def search_kb(query: str, k: int = 2) -> list[str]:
    time.sleep(0.14)
    words = set(query.lower().split())
    ranked = sorted(
        KNOWLEDGE_BASE, key=lambda doc: -len(words & set(KNOWLEDGE_BASE[doc].lower().split()))
    )
    return ranked[:k]


@tracer.trace(kind="tool")
def get_order(order_id: str) -> Order:
    global _order_api_calls
    _order_api_calls += 1
    time.sleep(0.09)
    if _order_api_calls == 1:  # the orders service throttles the first burst
        raise RateLimitError("429 Too Many Requests (retry-after: 0.4s)")
    time.sleep(0.12)
    delivered = dt.date.today() - dt.timedelta(days=6)
    return Order(order_id, "Wireless headphones", 89.00, delivered)


def get_order_with_retry(order_id: str, attempts: int = 3) -> Order:
    delay = 0.4
    for attempt in range(1, attempts + 1):
        try:
            return get_order(order_id)
        except RateLimitError:
            if attempt == attempts:
                raise
            current_span().event("backoff", tool="get_order", attempt=attempt, sleep_s=delay)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


@tracer.trace(kind="tool")
def check_refund_eligibility(order: Order, reason: str) -> dict:
    days = (dt.date.today() - order.delivered_on).days
    eligible = reason == "defective" and days <= 30
    return {"eligible": eligible, "days_since_delivery": days, "policy": "refund-policy"}


@tracer.trace(kind="tool")
def create_refund(order: Order) -> str:
    time.sleep(0.16)
    return f"RF-{order.id[-5:]}-01"


def handle(ticket: dict) -> str:
    with tracer.span(
        "support_agent",
        kind="agent",
        input=ticket["text"],
        ticket=ticket["id"],
        customer=ticket["customer"],
        tier=ticket["tier"],
    ) as agent:
        simulated_llm("classify_ticket", FAST_MODEL, 0.32, 612, 24, intent="refund_request")
        articles = search_kb("refund defective item headphones")
        order = get_order_with_retry("ORD-48213")
        decision = check_refund_eligibility(order, reason="defective")
        agent.set(intent="refund_request", eligible=decision["eligible"], kb=articles)

        simulated_llm("draft_reply", SMART_MODEL, 0.95, 1_180, 286, cache_read=2_950)
        simulated_llm("policy_guardrail", FAST_MODEL, 0.26, 904, 9)
        agent.event("guardrail", result="pass")

        refund_id = create_refund(order) if decision["eligible"] else None
        agent.set(refund_id=refund_id)
        agent.output = (
            f"Refund {refund_id} for ${order.amount:.2f} issued; customer told to expect "
            "it on the original payment method in 5-10 days."
        )
        return agent.output


if __name__ == "__main__":
    handle(TICKET)
    print(f"\nwrote {OUT}. Open the timeline with: agentledger html {OUT}")
