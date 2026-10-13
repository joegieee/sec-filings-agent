"""Filing documents to section-tagged plain-text blocks with citation anchors."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from filings_agent.parse.html import find_page_labels, html_to_blocks
from filings_agent.parse.models import (
    Block,
    ParsedFiling,
    SectionKind,
    SectionSummary,
    block_id,
)
from filings_agent.parse.sections import assign_sections

# Bump whenever parsing output changes: stored parses from older versions are redone,
# and block IDs (which citations reference) may shift.
PARSER_VERSION = 1

__all__ = ["PARSER_VERSION", "Block", "ParsedFiling", "SectionKind", "parse_document"]


def parse_document(
    content: bytes, *, accession_number: str, cik: int, form: str, document_url: str
) -> ParsedFiling:
    raw = html_to_blocks(content)
    labels, furniture = find_page_labels(raw)
    raw = [b for i, b in enumerate(raw) if i not in furniture]
    assignments = assign_sections(raw, labels, form=form)

    blocks = [
        Block(
            id=block_id(i),
            kind="table" if b.kind == "table" else "text",
            text=b.text,
            page=b.page,
            page_label=labels.get(b.page),
            anchor=b.anchor,
            section=a.kind,
            item=a.item,
        )
        for i, (b, a) in enumerate(zip(raw, assignments, strict=True))
    ]

    groups: dict[tuple[SectionKind, str | None, str], list[Block]] = defaultdict(list)
    for block, a in zip(blocks, assignments, strict=True):
        if a.how is not None:
            groups[(a.kind, a.item, a.how)].append(block)
    sections = [
        SectionSummary(
            kind=kind,
            item=item,
            blocks=len(members),
            chars=sum(len(m.text) for m in members),
            first_page=members[0].page,
            how=how,
        )
        for (kind, item, how), members in sorted(groups.items(), key=lambda g: g[1][0].page)
    ]
    return ParsedFiling(
        parser_version=PARSER_VERSION,
        accession_number=accession_number,
        cik=cik,
        form=form,
        source_sha256=hashlib.sha256(content).hexdigest(),
        document_url=document_url,
        blocks=blocks,
        sections=sections,
    )
