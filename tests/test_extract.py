from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from filings_agent.extract.citations import citation_valid, quote_in_block, shows_number
from filings_agent.extract.context import select_context
from filings_agent.extract.extractor import (
    FilingInfo,
    build_prompt,
    extract_filing,
    extractor_id,
)
from filings_agent.extract.models import ExtractedFigure, FinancialsExtraction, Metric
from filings_agent.llm import LLMError
from filings_agent.parse import ParsedFiling, parse_document
from tests.conftest import FakeLLM

FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.htm"
INFO = FilingInfo(
    accession_number="0000000001-24-000001",
    company_name="Example Bank Corp",
    ticker="EXB",
    form="10-K",
    report_date=date(2023, 12, 31),
)
FY = (date(2023, 1, 1), date(2023, 12, 31))


@pytest.fixture
def parsed() -> ParsedFiling:
    return parse_document(
        FIXTURE.read_bytes(),
        accession_number=INFO.accession_number,
        cik=1,
        form="10-K",
        document_url="https://www.sec.gov/doc.htm",
    )


def _block_with(parsed: ParsedFiling, text: str) -> str:
    return next(b.id for b in parsed.blocks if text in b.text and b.kind == "table")


def fig(
    value: float,
    block_id: str,
    quote: str,
    *,
    scale: str = "millions",
    period: tuple[date | None, date] = FY,
) -> ExtractedFigure:
    return ExtractedFigure.model_validate(
        {
            "value": value,
            "scale": scale,
            "period_start": period[0],
            "period_end": period[1],
            "block_id": block_id,
            "quote": quote,
        }
    )


def model_answer(parsed: ParsedFiling) -> FinancialsExtraction:
    income = _block_with(parsed, "Total net revenue")
    balance = _block_with(parsed, "Total assets")
    cash = _block_with(parsed, "operating activities")
    instant = (None, date(2023, 12, 31))
    return FinancialsExtraction(
        revenue=fig(1234, income, "Total net revenue | 1,234 | 1,100"),
        net_income=fig(456, income, "Net income | $ 456 | $ 400"),
        diluted_eps=fig(4.33, income, "Diluted earnings per share | $ 4.33", scale="units"),
        total_assets=fig(10_000, balance, "Total assets | $ 10,000", period=instant),
        # Wrong: the quote doesn't show the reported number.
        total_equity=fig(1100, balance, "Total stockholders\u2019 equity | 1,000", period=instant),
        operating_cash_flow=fig(
            -250, cash, "Net cash (used in) provided by operating activities | (250)"
        ),
        net_interest_income=fig(800, income, "Net interest income | $ 800"),
        # Cites a block the model was never shown.
        net_interest_margin=fig(2.71, "b00001", "Net interest margin | 2.71%", scale="units"),
        cet1_ratio=None,
    )


def test_context_contains_statements_and_ratio_disclosures(parsed: ParsedFiling) -> None:
    context = select_context(parsed)
    text = "\n".join(b.text for b in context.blocks)
    for needle in ("Total net revenue", "Total assets", "operating activities", "CET1"):
        assert needle in text
    assert "Consolidated statements of income" in text  # the statement's lead-in title
    assert "Our business is exposed" not in text  # risk factors are irrelevant here
    ids = [b.id for b in context.blocks]
    assert ids == sorted(ids), "blocks stay in document order"
    assert not context.truncated


def test_context_respects_the_budget(parsed: ParsedFiling) -> None:
    small = select_context(parsed, max_chars=400)
    assert small.truncated and small.chars <= 400


def test_prompt_states_periods_and_blocks(parsed: ParsedFiling) -> None:
    prompt = build_prompt(INFO, select_context(parsed))
    assert "Example Bank Corp (EXB), Form 10-K for the period ended 2023-12-31" in prompt
    assert "the fiscal year ended 2023-12-31" in prompt
    assert '<block id="b00012" section="Financial Statements" page="4" type="table">' in prompt
    quarterly = build_prompt(
        FilingInfo("a", "Example", "EXB", "10-Q", date(2024, 6, 30)), select_context(parsed)
    )
    assert "three months (fiscal quarter) ended 2024-06-30" in quarterly
    assert "year to date ended 2024-06-30" in quarterly


def test_extraction_normalises_and_checks_citations(parsed: ParsedFiling) -> None:
    llm = FakeLLM(model_answer(parsed))
    record = extract_filing(INFO, parsed, llm)
    assert record.extractor == extractor_id(llm)
    assert (record.input_tokens, record.output_tokens, record.model) == (
        1000,
        200,
        "claude-opus-5-5",
    )
    assert record.cost_usd == pytest.approx((1000 * 4 + 200 * 20) / 1e6)

    revenue = record.figure(Metric.REVENUE)
    assert revenue is not None
    assert revenue.value == 1_234_000_000 and revenue.unit == "USD"
    assert revenue.citation_valid
    assert revenue.section == "financial_statements" and revenue.item == "Item 8"
    assert revenue.page_label == "4"
    assert revenue.url == "https://www.sec.gov/doc.htm#item8"

    eps = record.figure(Metric.DILUTED_EPS)
    assert eps is not None and eps.value == 4.33 and eps.unit == "USD/share"
    cash = record.figure(Metric.OPERATING_CASH_FLOW)
    assert cash is not None and cash.value == -250_000_000 and cash.citation_valid
    assets = record.figure(Metric.TOTAL_ASSETS)
    assert assets is not None and assets.period_start is None

    equity = record.figure(Metric.TOTAL_EQUITY)
    assert equity is not None and not equity.citation_valid
    nim = record.figure(Metric.NET_INTEREST_MARGIN)
    assert nim is not None and not nim.citation_valid and nim.url is None
    assert record.figure(Metric.CET1_RATIO) is None


def test_extraction_propagates_llm_errors(parsed: ParsedFiling) -> None:
    with pytest.raises(LLMError):
        extract_filing(INFO, parsed, FakeLLM(LLMError("overloaded")))


def test_citation_checks() -> None:
    row = "Net cash (used in) provided by operating activities | (250) | 310"
    assert quote_in_block("net cash (used in) provided by operating activities | (250)", row)
    assert quote_in_block("activities |(250)", row), "separator spacing is normalised"
    assert quote_in_block("| Net cash (used in) provided by operating activities | (250) |", row)
    assert not quote_in_block("", row)
    assert shows_number("(1,234)", -1234)
    assert shows_number("2.71%", 2.71)
    assert not shows_number("1,234", 123)
    assert citation_valid(
        "Total stockholders\u2019 equity | 1,000", "Total stockholders' equity | 1,000 | 900", 1000
    )
    assert not citation_valid("Total assets | 1", None, 1)
