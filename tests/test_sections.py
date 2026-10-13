from __future__ import annotations

from filings_agent.parse.html import RawBlock
from filings_agent.parse.models import SectionKind
from filings_agent.parse.sections import assign_sections, classify


def _blocks(spec: list[tuple[int, str] | tuple[int, str, str]]) -> list[RawBlock]:
    out = []
    for entry in spec:
        page, text = entry[0], entry[1]
        kind = entry[2] if len(entry) == 3 else "text"
        out.append(RawBlock(kind, text, page, None))
    return out


def test_classify_uses_title_then_number() -> None:
    assert classify("Risk Factors", "1A", None, quarterly=False) == SectionKind.RISK_FACTORS
    assert classify("MANAGEMENT\u2019S DISCUSSION AND ANALYSIS", "7", None, quarterly=False) == (
        SectionKind.MDNA
    )
    assert classify("Exhibits, Financial Statement Schedules", "15", None, quarterly=False) == (
        SectionKind.OTHER
    )
    assert classify("", "8", None, quarterly=False) == SectionKind.FINANCIAL_STATEMENTS
    assert classify("", "1", "I", quarterly=True) == SectionKind.FINANCIAL_STATEMENTS
    assert classify("", "1", "II", quarterly=True) == SectionKind.LEGAL


def test_longest_occurrence_of_a_heading_wins() -> None:
    blocks = _blocks(
        [
            (1, "Item 7. Management's Discussion and Analysis"),  # a TOC line in prose form
            (1, "Item 8. Financial Statements"),
            (2, "Item 7. Management's Discussion and Analysis"),
            (2, "MD&A body. " * 50),
            (3, "Item 8. Financial Statements"),
            (3, "Statements body. " * 50),
        ]
    )
    result = assign_sections(blocks, {}, form="10-K")
    assert [a.kind for a in result] == [
        SectionKind.OTHER,
        SectionKind.OTHER,
        SectionKind.MDNA,
        SectionKind.MDNA,
        SectionKind.FINANCIAL_STATEMENTS,
        SectionKind.FINANCIAL_STATEMENTS,
    ]


def test_cross_reference_to_pages_of_this_document() -> None:
    # JPMorgan style: the 10-K items point into the annual report that follows them.
    blocks = _blocks(
        [
            (1, "Item 7. Management's Discussion and Analysis."),
            (1, "Management's discussion and analysis appears on pages 3-4."),
            (1, "Item 8. Financial Statements and Supplementary Data."),
            (1, "The Consolidated Financial Statements appear on pages 5\u20135."),
            (1, "Item 15. Exhibits."),
            (2, "Exhibit list"),
            (3, "Management's discussion and analysis"),
            (4, "More MD&A"),
            (5, "Consolidated statements of income", "table"),
            (6, "Glossary"),
        ]
    )
    labels = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6"}
    result = assign_sections(blocks, labels, form="10-K")
    kinds = [a.kind for a in result]
    assert kinds[6:8] == [SectionKind.MDNA, SectionKind.MDNA]
    assert kinds[8] == SectionKind.FINANCIAL_STATEMENTS
    assert kinds[9] == SectionKind.OTHER
    assert {result[i].how for i in (6, 7, 8)} == {"cross_reference"}
    assert result[6].item == "Item 7"


def test_cross_reference_to_another_document_is_ignored() -> None:
    blocks = _blocks(
        [
            (1, "Item 3. Quantitative and Qualitative Disclosures About Market Risk."),
            (1, "Refer to pages 2-3 of JPMorgan Chase's 2023 Form 10-K for a discussion."),
            (2, "Unrelated page"),
            (3, "Unrelated page"),
        ]
    )
    result = assign_sections(blocks, {1: "1", 2: "2", 3: "3"}, form="10-Q")
    assert [a.how for a in result] == ["heading"] * 4


def test_financial_statements_under_item_15() -> None:
    # PayPal style: "Item 8 ... listed in Part IV, Item 15(a)(1)".
    blocks = _blocks(
        [
            (1, "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA"),
            (1, "The consolidated financial statements listed in Part IV, Item 15(a)(1)."),
            (1, "ITEM 15. EXHIBITS, FINANCIAL STATEMENT SCHEDULES"),
            (2, "CONSOLIDATED BALANCE SHEETS"),
        ]
    )
    result = assign_sections(blocks, {}, form="10-K")
    assert result[3].kind == SectionKind.FINANCIAL_STATEMENTS
    assert result[3].how == "cross_reference"


def test_quarterly_items_without_headings_use_the_table_of_contents() -> None:
    # JPMorgan 10-Q style: Part I items appear only in the table of contents.
    toc = "\n".join(
        [
            "Part I \u2013 Financial information | Page",
            "Item 1. | Financial Statements.",
            "Consolidated statements of income | 4",
            "Notes to consolidated financial statements | 5",
            "Item 2. | Management's Discussion and Analysis. | 2",
            "Item 3. | Quantitative and Qualitative Disclosures About Market Risk. | 6",
            "Part II \u2013 Other information",
            "Item 1A. | Risk Factors. | 6",
        ]
    )
    blocks = _blocks(
        [
            (1, toc, "table"),
            (2, "Financial highlights"),
            (3, "More MD&A"),
            (4, "Consolidated statements of income"),
            (5, "Notes"),
            (6, "Item 3. Quantitative and Qualitative Disclosures About Market Risk."),
            (6, "Refer to the Market Risk Management section."),
            (6, "Part II \u2013 Other Information"),
            (6, "Item 1A. Risk Factors."),
            (6, "Refer to the 2023 Form 10-K."),
        ]
    )
    labels = {p: str(p) for p in range(1, 7)}
    result = assign_sections(blocks, labels, form="10-Q")
    assert [a.kind for a in result[1:5]] == [
        SectionKind.MDNA,
        SectionKind.MDNA,
        SectionKind.FINANCIAL_STATEMENTS,
        SectionKind.FINANCIAL_STATEMENTS,
    ]
    assert result[1].item == "Part I, Item 2" and result[1].how == "table_of_contents"
    assert result[3].item == "Part I, Item 1"
    assert result[5].kind == SectionKind.MARKET_RISK and result[5].how == "heading"
    assert result[8].kind == SectionKind.RISK_FACTORS and result[8].item == "Part II, Item 1A"
    assert result[0].how is None  # the table of contents itself


def test_quarterly_part_comes_from_the_item_title() -> None:
    # Bank of America 10-Q style: the MD&A heading follows a "Part II" TOC line.
    blocks = _blocks(
        [
            (1, "Part II. Other Information"),
            (1, "Item 2. Management's Discussion and Analysis of Financial Condition"),
            (2, "MD&A body"),
            (3, "Part I. Financial Information"),
            (3, "Item 1. Financial Statements"),
            (3, "Statements"),
            (4, "Part II. Other Information"),
            (4, "Item 1. Legal Proceedings"),
        ]
    )
    result = assign_sections(blocks, {}, form="10-Q")
    assert result[1].item == "Part I, Item 2" and result[1].kind == SectionKind.MDNA
    assert result[4].item == "Part I, Item 1"
    assert result[7].item == "Part II, Item 1" and result[7].kind == SectionKind.LEGAL
