"""Pick the blocks of a filing the extractor needs, instead of sending all of it.

A large bank's 10-K is ~500k characters; the figures we extract live in a
handful of places: the three primary statements, a financial-highlights table,
and the capital / net-interest-margin disclosures. Selecting those keeps each
call to roughly 10-20k tokens.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from filings_agent.parse.models import Block, ParsedFiling, SectionKind

DEFAULT_MAX_CHARS = 80_000

_NIM = re.compile(r"net interest margin|net yield on interest|net interest yield", re.IGNORECASE)
_CET1 = re.compile(r"\bCET1\b|common equity tier 1", re.IGNORECASE)
_PERCENT_LIKE = re.compile(r"\d\.\d{1,2}\s*%?|\d\s*%")
_HIGHLIGHT_TERMS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"net income",
        r"revenue",
        r"diluted",
        r"total assets",
        r"net interest income",
        r"\bCET1\b|common equity tier 1",
        r"book value",
    )
]


def _has_all(*patterns: str) -> Callable[[Block], bool]:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    return lambda b: all(c.search(b.text) for c in compiled)


_INCOME_STATEMENT = _has_all(r"net income", r"revenue", r"per (common )?share|diluted")
_BALANCE_SHEET = _has_all(r"total assets", r"total liabilities")
_CASH_FLOW = _has_all(r"operating activities", r"net cash")
_EQUITY = re.compile(r"(stockholders|shareholders)['\u2019]? equity", re.IGNORECASE)
_EPS = re.compile(r"diluted", re.IGNORECASE)


@dataclass(frozen=True)
class Context:
    blocks: list[Block]
    truncated: bool

    @property
    def chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


def _first_table(blocks: Sequence[Block], match: Callable[[Block], bool]) -> int | None:
    return next((i for i, b in enumerate(blocks) if b.kind == "table" and match(b)), None)


def _lead_in(blocks: Sequence[Block], index: int, limit: int = 2) -> list[int]:
    """Short text blocks right before a table: its title and '(in millions)' line."""
    picked: list[int] = []
    i = index - 1
    while i >= 0 and len(picked) < limit and blocks[i].kind == "text" and len(blocks[i].text) < 300:
        picked.append(i)
        i -= 1
    return picked


def _statement(
    blocks: Sequence[Block], match: Callable[[Block], bool], complete: re.Pattern[str]
) -> list[int]:
    """The first matching table, its lead-in and, if it is split, its continuation."""
    index = _first_table(blocks, match)
    if index is None:
        return []
    picked = [*_lead_in(blocks, index), index]
    if not complete.search(blocks[index].text):
        nxt = next((i for i in range(index + 1, len(blocks)) if blocks[i].kind == "table"), None)
        if nxt is not None and blocks[nxt].page - blocks[index].page <= 1:
            picked.append(nxt)
    return picked


def _disclosures(blocks: Sequence[Block], pattern: re.Pattern[str], limit: int) -> list[int]:
    """Blocks stating a ratio: tables first, then prose, earliest first."""

    def states_ratio(b: Block) -> bool:
        return any(
            pattern.search(line) and _PERCENT_LIKE.search(line) for line in b.text.split("\n")
        )

    hits = [i for i, b in enumerate(blocks) if pattern.search(b.text) and states_ratio(b)]
    tables = [i for i in hits if blocks[i].kind == "table"]
    texts = [i for i in hits if blocks[i].kind == "text"]
    picked = tables[: limit - 1] + texts[:1]
    return [j for i in picked for j in (*_lead_in(blocks, i, 1), i)]


def _highlights(blocks: Sequence[Block]) -> list[int]:
    for i, b in enumerate(blocks):
        if (
            b.kind == "table"
            and b.section == SectionKind.MDNA
            and sum(1 for t in _HIGHLIGHT_TERMS if t.search(b.text)) >= 3
        ):
            return [*_lead_in(blocks, i), i]
    return []


def select_context(parsed: ParsedFiling, *, max_chars: int = DEFAULT_MAX_CHARS) -> Context:
    blocks = parsed.blocks
    statements = [i for i, b in enumerate(blocks) if b.section == SectionKind.FINANCIAL_STATEMENTS]
    # Search the financial-statements section first, so the primary statements win
    # over look-alike tables in MD&A; fall back to the whole document.
    in_statements = set(statements)
    order = statements + [i for i in range(len(blocks)) if i not in in_statements]
    ordered = [blocks[i] for i in order]

    def remap(indexes: list[int]) -> list[int]:
        return [order[i] for i in indexes]

    # Highest priority first: when over budget, later groups are dropped.
    groups = [
        remap(_statement(ordered, _INCOME_STATEMENT, _EPS)),
        remap(_statement(ordered, _BALANCE_SHEET, _EQUITY)),
        remap(_statement(ordered, _CASH_FLOW, re.compile(r"operating activities"))),
        _disclosures(blocks, _CET1, 3),
        _disclosures(blocks, _NIM, 3),
        _highlights(blocks),
    ]
    chosen: set[int] = set()
    used = 0
    truncated = False
    for group in groups:
        new = [i for i in dict.fromkeys(group) if i not in chosen]
        size = sum(len(blocks[i].text) for i in new)
        if used + size > max_chars:
            truncated = True
            continue
        chosen.update(new)
        used += size
    return Context(blocks=[blocks[i] for i in sorted(chosen)], truncated=truncated)
