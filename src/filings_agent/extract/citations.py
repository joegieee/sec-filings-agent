"""Check that a citation holds up: the quote is in the cited block and shows the number."""

from __future__ import annotations

import math
import re

_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_SEPARATOR = re.compile(r"\s*\|\s*")
_WS = re.compile(r"\s+")
# Typographic quotes and dashes -> ASCII, so a quote matches however it was typed.
_TRANSLATE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
)


def normalize_text(text: str) -> str:
    text = _SEPARATOR.sub(" | ", text.translate(_TRANSLATE))
    return _WS.sub(" ", text).strip().casefold()


def numbers_in(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUMBER.findall(text)]


def quote_in_block(quote: str, block_text: str) -> bool:
    q = normalize_text(quote)
    return bool(q) and q in normalize_text(block_text)


def shows_number(text: str, value: float) -> bool:
    """True if ``text`` prints ``value`` (sign aside: "(1,234)" shows -1234)."""
    target = abs(value)
    return any(math.isclose(n, target, rel_tol=1e-9, abs_tol=1e-9) for n in numbers_in(text))


def citation_valid(quote: str, block_text: str | None, value: float) -> bool:
    return (
        block_text is not None and quote_in_block(quote, block_text) and shows_number(quote, value)
    )
