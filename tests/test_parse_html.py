from __future__ import annotations

from pathlib import Path

from filings_agent.parse import PARSER_VERSION, parse_document
from filings_agent.parse.html import find_page_labels, html_to_blocks, page_label
from filings_agent.parse.models import SectionKind

FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.htm"


def test_xbrl_and_hidden_content_never_reach_the_text() -> None:
    parsed = parse_document(
        FIXTURE.read_bytes(), accession_number="a", cik=1, form="10-K", document_url="u"
    )
    text = "\n".join(b.text for b in parsed.blocks)
    assert "HIDDEN-XBRL-FACT" not in text  # ix:header facts
    assert "Hidden draft paragraph" not in text  # display:none
    assert "reviewer note" not in text  # HTML comment
    assert "ix:" not in text and "us-gaap" not in text
    # Tagged facts keep their displayed text, without the tags.
    assert "Net income was 456 million in 2023." in text
    assert "Our business is exposed to credit, market and liquidity risk." in text


def test_tables_render_one_row_per_line_with_fragments_glued() -> None:
    parsed = parse_document(
        FIXTURE.read_bytes(), accession_number="a", cik=1, form="10-K", document_url="u"
    )
    cash_flow = next(b for b in parsed.blocks if "operating activities" in b.text)
    assert cash_flow.kind == "table"
    assert "Net cash (used in) provided by operating activities | (250) | 310" in cash_flow.text
    highlights = next(b for b in parsed.blocks if "Net interest margin" in b.text)
    assert "Net interest margin | 2.71% | 2.50%" in highlights.text
    assert "Net interest income | $ 800 | $ 700" in highlights.text


def test_pages_labels_and_anchors() -> None:
    parsed = parse_document(
        FIXTURE.read_bytes(), accession_number="a", cik=1, form="10-K", document_url="https://x/d"
    )
    assert parsed.parser_version == PARSER_VERSION
    item8 = next(b for b in parsed.blocks if b.text.startswith("Item 8."))
    assert (item8.page, item8.page_label, item8.anchor) == (4, "4", "item8")
    assert parsed.citation_url(item8) == "https://x/d#item8"
    # Page footers are recognised as furniture and dropped from the text.
    assert not any(b.text.startswith("Example Bank |") for b in parsed.blocks)
    # Block ids are positional and resolvable.
    assert all(parsed.block(b.id) == b for b in parsed.blocks)
    assert parsed.block("b99999") is None


def test_sections_by_heading() -> None:
    parsed = parse_document(
        FIXTURE.read_bytes(), accession_number="a", cik=1, form="10-K", document_url="u"
    )
    by_kind = {s.kind: s for s in parsed.sections}
    assert by_kind[SectionKind.RISK_FACTORS].item == "Item 1A"
    assert by_kind[SectionKind.MDNA].item == "Item 7"
    assert by_kind[SectionKind.FINANCIAL_STATEMENTS].item == "Item 8"
    assert all(s.how == "heading" for s in parsed.sections)
    statements = parsed.section_blocks(SectionKind.FINANCIAL_STATEMENTS)
    assert any("Total assets" in b.text for b in statements)
    # The table of contents (which mentions every item) stays outside all sections.
    toc = next(b for b in parsed.blocks if b.text.startswith("Part I | Page"))
    assert toc.section == SectionKind.OTHER and toc.item is None
    # Item 9 is recognised as a boundary but is not a section we classify.
    assert parsed.blocks[-1].section == SectionKind.OTHER and parsed.blocks[-1].item == "Item 9"


def test_page_label_recognises_common_footers() -> None:
    assert page_label("35") == "35"
    assert page_label("JPMorgan Chase & Co./2023 Form 10-K | 47") == "47"
    assert page_label("Bank of America 86") == "86"
    assert page_label("Net income increased 12% to $4.5 billion in 2023") is None
    assert page_label("A long paragraph " * 10 + "12") is None


def test_stray_numbers_are_not_page_labels() -> None:
    html = b"".join(
        [
            b"<html><body>",
            b"<p>Body</p><p>Note 3</p><hr style='page-break-after:always'/>",
            b"<p>Body</p><p>Note 9</p><hr style='page-break-after:always'/>",
            b"<p>Body</p><p>12</p><hr style='page-break-after:always'/>",
            b"<p>Body</p><p>13</p>",
            b"</body></html>",
        ]
    )
    blocks = html_to_blocks(html)
    labels, furniture = find_page_labels(blocks)
    # "Note 3"/"Note 9" don't advance with the pages; "12"/"13" do.
    assert labels == {3: "12", 4: "13"}
    assert {blocks[i].text for i in furniture} == {"12", "13"}
