"""Per-model token prices (USD per million tokens), for cost reporting.

Claude list prices; keep in sync with https://www.anthropic.com/pricing. Other
providers' prices vary by plan (Gemini's free tier costs nothing), so they are
configured instead (``FRA_LLM_PRICE_INPUT`` / ``FRA_LLM_PRICE_OUTPUT``). Unknown
models cost ``None`` rather than a guess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from filings_agent.llm.base import Usage

# model -> (input, output) USD per million tokens
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.00, 50.00),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def cost_usd(model: str, usage: Usage, prices: tuple[float, float] | None = None) -> float | None:
    """USD for ``usage``, at ``prices`` (input, output per million tokens) or list price."""
    prices = prices or PRICES.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    tokens_cost = (
        usage.input_tokens * input_price
        + usage.cache_read_input_tokens * input_price * CACHE_READ_MULTIPLIER
        + usage.cache_creation_input_tokens * input_price * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * output_price
    )
    return tokens_cost / 1_000_000
