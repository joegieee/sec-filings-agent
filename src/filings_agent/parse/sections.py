"""Assign each block to a 10-K / 10-Q item.

Filers lay out their reports in different ways, so detection runs in layers:

1. **Headings.** Body headings such as ``Item 7. Management's Discussion...``.
   Table-of-contents tables are skipped. When an item heading appears more
   than once, the occurrence followed by the most text wins.
2. **Cross-references.** Some filers (JPMorgan, for one) satisfy an item with a
   one-line pointer such as *"appears on pages 48-161"* into the annual-report
   part of the document. Pointers into *this* document are resolved through
   the printed page numbers. Pointers to another document (*"pages 9-33 of the
   2023 Form 10-K"*) are not.
3. **Table of contents.** An item that has no body heading at all (JPMorgan's
   10-Qs) gets the page range its table-of-contents entries give it.

Each block's section kind comes from the item's title (``Risk Factors``,
``Financial Statements``...), falling back to the item number, because 10-K
and 10-Q number their items differently.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from filings_agent.parse.html import RawBlock
from filings_agent.parse.models import SectionKind

How = Literal["heading", "cross_reference", "table_of_contents"]

_ITEM = re.compile(r"^\s*item\s*(\d{1,2}[a-c]?)\b\s*[.:\-\u2013\u2014]?\s*(.*)$", re.IGNORECASE)
_PART = re.compile(r"^\s*part\s+(iv|iii|ii|i)\b", re.IGNORECASE)
_TRAILING_PAGE = re.compile(r"\|\s*(\d{1,3})\s*$")
_PAGE_RANGE = re.compile(
    r"pages?\s+(\d{1,3})\s*(?:[\u2013\u2014-]|through|to)\s*(\d{1,3})\b", re.IGNORECASE
)
_OTHER_DOCUMENT = re.compile(
    r"^\W*(?:of|in)\b[^.;]{0,60}?(?:form\s+10-k|annual\s+report|proxy\s+statement)",
    re.IGNORECASE,
)
_ITEM_15 = re.compile(r"\bitem\s+15\b", re.IGNORECASE)

# Sections a short cross-reference is worth resolving for.
_RESOLVABLE = frozenset(
    {
        SectionKind.RISK_FACTORS,
        SectionKind.MDNA,
        SectionKind.MARKET_RISK,
        SectionKind.FINANCIAL_STATEMENTS,
    }
)
_SHORT_SPAN_CHARS = 3000

_TEN_K_ITEMS = {
    "1": SectionKind.BUSINESS,
    "1A": SectionKind.RISK_FACTORS,
    "3": SectionKind.LEGAL,
    "7": SectionKind.MDNA,
    "7A": SectionKind.MARKET_RISK,
    "8": SectionKind.FINANCIAL_STATEMENTS,
    "9A": SectionKind.CONTROLS,
}
_TEN_Q_ITEMS = {
    ("I", "1"): SectionKind.FINANCIAL_STATEMENTS,
    ("I", "2"): SectionKind.MDNA,
    ("I", "3"): SectionKind.MARKET_RISK,
    ("I", "4"): SectionKind.CONTROLS,
    ("II", "1"): SectionKind.LEGAL,
    ("II", "1A"): SectionKind.RISK_FACTORS,
}
_PART_I_KINDS = frozenset(
    {
        SectionKind.FINANCIAL_STATEMENTS,
        SectionKind.MDNA,
        SectionKind.MARKET_RISK,
        SectionKind.CONTROLS,
    }
)


@dataclass(frozen=True)
class Assignment:
    kind: SectionKind
    item: str | None
    how: How | None


@dataclass
class _Heading:
    index: int
    part: str | None
    item: str
    kind: SectionKind

    @property
    def key(self) -> tuple[str | None, str]:
        return (self.part, self.item)


@dataclass
class _TocEntry:
    part: str | None
    item: str
    kind: SectionKind
    pages: list[int] = field(default_factory=list)


def classify(title: str, item: str, part: str | None, *, quarterly: bool) -> SectionKind:
    """Section kind from an item's title, falling back to its number."""
    t = title.lower().replace("\u2019", "'")
    if "exhibits" in t or "changes in and disagreements" in t:
        return SectionKind.OTHER
    for needle, kind in (
        ("risk factors", SectionKind.RISK_FACTORS),
        ("discussion and analysis", SectionKind.MDNA),
        ("quantitative and qualitative", SectionKind.MARKET_RISK),
        ("financial statements", SectionKind.FINANCIAL_STATEMENTS),
        ("controls and procedures", SectionKind.CONTROLS),
        ("legal proceedings", SectionKind.LEGAL),
    ):
        if needle in t:
            return kind
    if re.match(r"\W*business\b", t):
        return SectionKind.BUSINESS
    if quarterly:
        return _TEN_Q_ITEMS.get((part or "I", item), SectionKind.OTHER)
    return _TEN_K_ITEMS.get(item, SectionKind.OTHER)


def item_label(part: str | None, item: str) -> str:
    return f"Part {part}, Item {item}" if part else f"Item {item}"


def _first_line(block: RawBlock) -> str:
    return block.text.split("\n", 1)[0].replace(" | ", " ")


def _is_toc(block: RawBlock) -> bool:
    if block.kind != "table":
        return False
    rows = block.text.split("\n")
    if len(rows) < 3:
        return False
    item_rows = sum(1 for r in rows if _ITEM.match(r.replace(" | ", " ")))
    paged_rows = sum(1 for r in rows if _TRAILING_PAGE.search(r))
    return item_rows >= 2 or paged_rows >= len(rows) / 2


def _part_for(kind: SectionKind, current: str | None) -> str:
    if kind in _PART_I_KINDS:
        return "I"
    if kind in (SectionKind.LEGAL, SectionKind.RISK_FACTORS):
        return "II"
    return current or "I"


def _find_headings(
    blocks: Sequence[RawBlock], *, quarterly: bool
) -> tuple[list[_Heading], list[_TocEntry]]:
    headings: list[_Heading] = []
    toc: list[_TocEntry] = []
    part: str | None = None
    for i, block in enumerate(blocks):
        if _is_toc(block):
            toc.extend(_parse_toc(block, quarterly=quarterly))
            continue
        first = _first_line(block)
        part_match = _PART.match(first)
        if part_match and len(first) <= 120:
            part = part_match.group(1).upper()
            continue
        match = _ITEM.match(first)
        limit = 300 if block.kind == "text" else 600
        if not match or len(first) > 250 or len(block.text) > limit:
            continue
        item = match.group(1).upper()
        title = f"{match.group(2)} {block.text[len(first) : 300]}".strip()
        if not title and i + 1 < len(blocks) and len(blocks[i + 1].text) <= 200:
            title = blocks[i + 1].text
        provisional = classify(title, item, part, quarterly=quarterly)
        item_part = _part_for(provisional, part) if quarterly else None
        kind = classify(title, item, item_part, quarterly=quarterly)
        headings.append(_Heading(i, item_part, item, kind))
    return headings, toc


def _parse_toc(block: RawBlock, *, quarterly: bool) -> list[_TocEntry]:
    entries: list[_TocEntry] = []
    part: str | None = None
    for row in block.text.split("\n"):
        flat = row.replace(" | ", " ")
        part_match = _PART.match(flat)
        if part_match:
            part = part_match.group(1).upper()
            continue
        page = _TRAILING_PAGE.search(row)
        match = _ITEM.match(flat)
        if match:
            item = match.group(1).upper()
            title = _TRAILING_PAGE.sub("", row).replace(" | ", " ")
            entry_part = (part or "I") if quarterly else None
            kind = classify(title, item, entry_part, quarterly=quarterly)
            entries.append(_TocEntry(entry_part, item, kind))
        if page and entries:
            entries[-1].pages.append(int(page.group(1)))
    return entries


def _span_chars(blocks: Sequence[RawBlock], start: int, end: int) -> int:
    return sum(len(b.text) for b in blocks[start:end])


def _estimated_labels(blocks: Sequence[RawBlock], labels: dict[int, str]) -> list[int | None]:
    """Printed page number per block, interpolated across pages that lack one."""
    pages = sorted({b.page for b in blocks})
    estimate: dict[int, int] = {}
    last: tuple[int, int] | None = None  # (physical page, printed number)
    for page in pages:
        label = labels.get(page)
        if label is not None and label.isdigit():
            last = (page, int(label))
            estimate[page] = int(label)
        elif last is not None:
            estimate[page] = last[1] + (page - last[0])
    return [estimate.get(b.page) for b in blocks]


def _internal_page_range(text: str) -> tuple[int, int] | None:
    for match in _PAGE_RANGE.finditer(text):
        start, end = int(match.group(1)), int(match.group(2))
        if end < start or _OTHER_DOCUMENT.match(text[match.end() : match.end() + 120]):
            continue
        return start, end
    return None


def assign_sections(
    blocks: Sequence[RawBlock], labels: dict[int, str], *, form: str
) -> list[Assignment]:
    quarterly = form.upper().startswith("10-Q")
    headings, toc = _find_headings(blocks, quarterly=quarterly)

    # Keep, for each item, the heading occurrence followed by the most text.
    starts = [h.index for h in headings] + [len(blocks)]
    best: dict[tuple[str | None, str], tuple[int, _Heading]] = {}
    for n, heading in enumerate(headings):
        size = _span_chars(blocks, heading.index, starts[n + 1])
        if heading.key not in best or size > best[heading.key][0]:
            best[heading.key] = (size, heading)
    chosen = sorted((h for _, h in best.values()), key=lambda h: h.index)

    result: list[Assignment] = [Assignment(SectionKind.OTHER, None, None)] * len(blocks)
    spans: dict[tuple[str | None, str], tuple[int, int]] = {}
    bounds = [h.index for h in chosen] + [len(blocks)]
    for n, heading in enumerate(chosen):
        start, end = bounds[n], bounds[n + 1]
        spans[heading.key] = (start, end)
        label = item_label(heading.part, heading.item)
        for i in range(start, end):
            result[i] = Assignment(heading.kind, label, "heading")

    estimated = _estimated_labels(blocks, labels)

    # Short items that point elsewhere in this document.
    overrides: list[tuple[int, int, int, Assignment]] = []  # (width, lo, hi, assignment)
    for heading in chosen:
        start, end = spans[heading.key]
        if heading.kind not in _RESOLVABLE or _span_chars(blocks, start, end) >= _SHORT_SPAN_CHARS:
            continue
        span_text = " ".join(b.text for b in blocks[start:end])
        label = item_label(heading.part, heading.item)
        page_range = _internal_page_range(span_text)
        if page_range is not None:
            lo, hi = page_range
            overrides.append((hi - lo, lo, hi, Assignment(heading.kind, label, "cross_reference")))
        elif heading.kind == SectionKind.FINANCIAL_STATEMENTS and _ITEM_15.search(span_text):
            # "...listed in Part IV, Item 15(a)(1)": the statements sit under Item 15.
            target = next((h for h in chosen if h.item == "15"), None)
            if target is not None:
                t_start, t_end = spans[target.key]
                for i in range(t_start, t_end):
                    result[i] = Assignment(heading.kind, label, "cross_reference")
    for _, lo, hi, assignment in sorted(overrides, key=lambda o: -o[0]):
        for i, number in enumerate(estimated):
            if number is not None and lo <= number <= hi:
                result[i] = assignment

    # Items with no body heading: fall back to their table-of-contents pages.
    chosen_keys = set(spans)
    toc_starts = sorted({min(e.pages) for e in toc if e.pages})
    for entry in toc:
        key = (entry.part, entry.item)
        if key in chosen_keys or not entry.pages or entry.kind not in _RESOLVABLE:
            continue
        lo = min(entry.pages)
        later = [s for s in toc_starts if s > lo]
        hi = later[0] - 1 if later else 10**6
        assignment = Assignment(entry.kind, item_label(entry.part, entry.item), "table_of_contents")
        for i, number in enumerate(estimated):
            if result[i].how is None and number is not None and lo <= number <= hi:
                result[i] = assignment
        chosen_keys.add(key)
    return result
