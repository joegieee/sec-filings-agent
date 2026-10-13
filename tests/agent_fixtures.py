"""Small, hand-made figures and filing text for agent and Q&A eval tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from filings_agent.agent.financials import Fact, FinancialsIndex, period_label
from filings_agent.agent.retrieval import SearchIndex
from filings_agent.extract.models import METRICS, Metric
from filings_agent.parse.models import Block, ParsedFiling, SectionKind
from filings_agent.storage import StoredFiling

NAMES = {"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corp", "PYPL": "PayPal Holdings"}

# (ticker, metric, form, report date, value)
FIGURES = [
    ("JPM", Metric.NET_INCOME, "10-K", date(2023, 12, 31), 49_552e6),
    ("JPM", Metric.NET_INCOME, "10-K", date(2024, 12, 31), 58_471e6),
    ("JPM", Metric.REVENUE, "10-K", date(2024, 12, 31), 177_556e6),
    ("JPM", Metric.NET_INCOME, "10-Q", date(2025, 9, 30), 14_393e6),
    ("BAC", Metric.NET_INCOME, "10-K", date(2024, 12, 31), 27_132e6),
    ("BAC", Metric.REVENUE, "10-K", date(2024, 12, 31), 101_887e6),
    ("BAC", Metric.NET_INTEREST_INCOME, "10-K", date(2024, 12, 31), 56_060e6),
    ("PYPL", Metric.NET_INCOME, "10-K", date(2024, 12, 31), 4_147e6),
    ("PYPL", Metric.DILUTED_EPS, "10-K", date(2024, 12, 31), 3.99),
    ("PYPL", Metric.OPERATING_CASH_FLOW, "10-Q", date(2025, 9, 30), 4_032e6),
]


def fact(ticker: str, metric: Metric, form: str, end: date, value: float) -> Fact:
    period = METRICS[metric].period
    start = None
    if period != "instant":
        year_start = form == "10-K" or period == "ytd"
        start = date(end.year, 1, 1) if year_start else date(end.year, end.month - 2, 1)
    return Fact(
        ticker=ticker,
        company=NAMES[ticker],
        metric=metric,
        period=period_label(form, end, metric),
        period_start=start,
        period_end=end,
        value=value,
        unit=METRICS[metric].unit,
        form=form,
        accession_number=f"{ticker}-{form}-{end.isoformat()}",
        filing_date=date(end.year + 1, 2, 15),
        url=f"https://www.sec.gov/{ticker}/{end.isoformat()}.htm",
        source="xbrl",
        evidence="XBRL us-gaap:Test",
    )


def financials() -> FinancialsIndex:
    return FinancialsIndex([fact(*row) for row in FIGURES], NAMES)


def stored_filing(ticker: str, form: str, end: date) -> StoredFiling:
    return StoredFiling(
        accession_number=f"{ticker}-{form}-{end.isoformat()}",
        cik=1,
        ticker=ticker,
        company_name=NAMES[ticker],
        form=form,
        filing_date=date(end.year + 1, 2, 15),
        report_date=end,
        url=f"https://www.sec.gov/{ticker}/{end.isoformat()}.htm",
        raw_path=None,
        sha256="x",
        size_bytes=1,
    )


PASSAGES = {
    "BAC": [
        (
            SectionKind.MDNA,
            "Net interest yield on a fully taxable-equivalent basis decreased 13 basis points "
            "to 1.95 percent for 2024, driven by higher deposit costs.",
        ),
        (SectionKind.RISK_FACTORS, "Changes in interest rates could reduce our net income."),
    ],
    "JPM": [
        (
            SectionKind.MDNA,
            "CET1 capital was $275.5 billion, and the Standardized and Advanced CET1 ratios "
            "were 15.7% and 15.8%, respectively.",
        ),
    ],
}


def parsed_filing(ticker: str, end: date) -> ParsedFiling:
    blocks = [
        Block(
            id=f"b{i:05d}",
            kind="text",
            text=text,
            page=i + 1,
            page_label=str(10 + i),
            anchor=f"a{i}",
            section=section,
        )
        for i, (section, text) in enumerate(PASSAGES[ticker])
    ]
    return ParsedFiling(
        parser_version=1,
        accession_number=f"{ticker}-10-K-{end.isoformat()}",
        cik=1,
        form="10-K",
        source_sha256=f"sha-{ticker}",
        document_url=f"https://www.sec.gov/{ticker}/{end.isoformat()}.htm",
        blocks=blocks,
        sections=[],
    )


def search_index(path: Path) -> SearchIndex:
    index = SearchIndex(path)
    end = date(2024, 12, 31)
    for ticker in PASSAGES:
        index.add_filing(stored_filing(ticker, "10-K", end), parsed_filing(ticker, end))
    return index
