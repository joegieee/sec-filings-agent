"""Parsed-filing data model: an ordered list of text blocks, each tagged with its section."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class SectionKind(StrEnum):
    """Canonical sections, shared by 10-K and 10-Q (whose item numbers differ)."""

    BUSINESS = "business"
    RISK_FACTORS = "risk_factors"
    MDNA = "mdna"
    MARKET_RISK = "market_risk"
    FINANCIAL_STATEMENTS = "financial_statements"
    CONTROLS = "controls"
    LEGAL = "legal"
    OTHER = "other"


SECTION_TITLES: dict[SectionKind, str] = {
    SectionKind.BUSINESS: "Business",
    SectionKind.RISK_FACTORS: "Risk Factors",
    SectionKind.MDNA: "Management's Discussion and Analysis",
    SectionKind.MARKET_RISK: "Market Risk Disclosures",
    SectionKind.FINANCIAL_STATEMENTS: "Financial Statements",
    SectionKind.CONTROLS: "Controls and Procedures",
    SectionKind.LEGAL: "Legal Proceedings",
    SectionKind.OTHER: "Other",
}


class Block(BaseModel):
    """One paragraph or table of the filing, as plain text (no XBRL markup).

    ``id`` is stable for a given document and parser version, and is what
    citations point at. ``anchor`` is the nearest preceding HTML ``id`` in the
    source document, so a citation can deep-link into the filing on EDGAR.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    kind: Literal["text", "table"]
    text: str
    page: int  # physical page in the document, 1-based (counted from page breaks)
    page_label: str | None = None  # printed page number, when the page has one
    anchor: str | None = None
    section: SectionKind = SectionKind.OTHER
    item: str | None = None  # e.g. "Item 7" (10-K) or "Part I, Item 2" (10-Q)


class SectionSummary(BaseModel):
    kind: SectionKind
    item: str | None
    blocks: int
    chars: int
    first_page: int
    how: Literal["heading", "cross_reference", "table_of_contents"]


class ParsedFiling(BaseModel):
    parser_version: int
    accession_number: str
    cik: int
    form: str
    source_sha256: str
    document_url: str
    blocks: list[Block]
    sections: list[SectionSummary]

    def block(self, block_id: str) -> Block | None:
        index = _block_index(block_id)
        if index is not None and index < len(self.blocks) and self.blocks[index].id == block_id:
            return self.blocks[index]
        return next((b for b in self.blocks if b.id == block_id), None)

    def section_blocks(self, kind: SectionKind) -> list[Block]:
        return [b for b in self.blocks if b.section == kind]

    def citation_url(self, block: Block) -> str:
        return f"{self.document_url}#{block.anchor}" if block.anchor else self.document_url


def block_id(index: int) -> str:
    return f"b{index:05d}"


def _block_index(block_id: str) -> int | None:
    if len(block_id) == 6 and block_id[0] == "b" and block_id[1:].isdigit():
        return int(block_id[1:])
    return None
