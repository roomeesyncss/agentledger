"""Token pricing and cost calculation.

Prices are USD per million tokens. The defaults are list prices at the time of writing
(November 2026). Prices change, so override or extend them with ``Tracer(pricing=...)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .span import Usage


@dataclass(frozen=True)
class Pricing:
    input: float
    output: float
    cache_read: float | None = None  # default: 10% of input
    cache_write: float | None = None  # default: 125% of input

    def cost(self, usage: Usage) -> float:
        cache_read = self.cache_read if self.cache_read is not None else self.input * 0.10
        cache_write = self.cache_write if self.cache_write is not None else self.input * 1.25
        total = (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * cache_read
            + usage.cache_write_tokens * cache_write
        )
        return total / 1_000_000


DEFAULT_PRICING: dict[str, Pricing] = {
    "claude-opus-5-5": Pricing(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": Pricing(input=2.00, output=10.00, cache_read=0.10),
    "claude-haiku-5-5": Pricing(input=0.10, output=0.50),
}


def price_for(model: str | None, table: Mapping[str, Pricing]) -> Pricing | None:
    """Exact match first, then the longest table key that prefixes ``model``."""
    if not model:
        return None
    if model in table:
        return table[model]
    matches = [k for k in table if model.startswith(k)]
    return table[max(matches, key=len)] if matches else None
