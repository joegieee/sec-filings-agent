from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from filings_agent.evals.extraction import evaluate, score_filing
from filings_agent.evals.ground_truth import CompanyFacts
from filings_agent.evals.report import to_markdown
from filings_agent.extract.models import METRICS, ExtractionRecord, Figure, Metric
from filings_agent.storage import StoredFiling

ACC = "0000000001-24-000001"
OTHER_ACC = "0000000001-23-000009"


def _fact(val: float, end: str, start: str | None = None, accn: str = ACC) -> dict[str, Any]:
    fact: dict[str, Any] = {"val": val, "end": end, "accn": accn, "form": "10-Q"}
    if start:
        fact["start"] = start
    return fact


def _facts(**concepts: list[dict[str, Any]]) -> CompanyFacts:
    gaap = {
        name: {"units": {"USD/shares" if "PerShare" in name else "USD": facts}}
        for name, facts in concepts.items()
    }
    return CompanyFacts({"facts": {"us-gaap": gaap}})


QUARTER_FACTS = _facts(
    Revenues=[
        _fact(25_000e6, "2024-06-30", "2024-04-01"),  # Q2
        _fact(50_000e6, "2024-06-30", "2024-01-01"),  # six months
        _fact(24_000e6, "2023-06-30", "2023-04-01"),  # prior-year comparative
    ],
    RevenuesNetOfInterestExpense=[],
    NetIncomeLoss=[_fact(7_000e6, "2024-06-30", "2024-04-01", accn=OTHER_ACC)],
    EarningsPerShareDiluted=[_fact(0.83, "2024-06-30", "2024-04-01")],
    Assets=[_fact(3_258_000e6, "2024-06-30"), _fact(3_180_000e6, "2023-12-31")],
    NetCashProvidedByUsedInOperatingActivities=[
        _fact(-5e9, "2024-06-30", "2024-01-01"),
        _fact(-1e9, "2024-03-31", "2024-01-01"),
    ],
)


def test_ground_truth_picks_the_primary_period_from_this_filing() -> None:
    def expected(metric: Metric) -> float | None:
        e = QUARTER_FACTS.expected(
            metric, accession=ACC, form="10-Q", report_date=date(2024, 6, 30)
        )
        return e.value if e else None

    assert expected(Metric.REVENUE) == 25_000e6  # the quarter, not six months
    assert expected(Metric.TOTAL_ASSETS) == 3_258_000e6  # period end, not the prior year-end
    assert expected(Metric.OPERATING_CASH_FLOW) == -5e9  # year to date
    assert expected(Metric.DILUTED_EPS) == 0.83
    assert expected(Metric.NET_INCOME) is None  # only tagged in another filing
    assert expected(Metric.CET1_RATIO) is None  # no standard XBRL concept


def test_ground_truth_annual_and_concept_order() -> None:
    facts = _facts(
        RevenuesNetOfInterestExpense=[_fact(158e9, "2023-12-31", "2023-01-01")],
        Revenues=[_fact(239e9, "2023-12-31", "2023-01-01")],
        NetIncomeLoss=[
            _fact(49e9, "2023-12-31", "2023-01-01"),
            _fact(13e9, "2023-12-31", "2023-10-01"),
        ],
    )
    revenue = facts.expected(
        Metric.REVENUE, accession=ACC, form="10-K", report_date=date(2023, 12, 31)
    )
    assert revenue is not None and revenue.value == 158e9
    assert revenue.concept == "RevenuesNetOfInterestExpense"
    income = facts.expected(
        Metric.NET_INCOME, accession=ACC, form="10-K", report_date=date(2023, 12, 31)
    )
    assert income is not None and income.value == 49e9  # the fiscal year, not Q4


def _filing(ticker: str = "BAC") -> StoredFiling:
    return StoredFiling(
        accession_number=ACC,
        cik=1,
        ticker=ticker,
        company_name="Example",
        form="10-Q",
        filing_date=date(2024, 7, 30),
        report_date=date(2024, 6, 30),
        url="https://x",
        raw_path=Path("x"),
        sha256=None,
        size_bytes=None,
    )


def _figure(
    metric: Metric, value: float, *, valid: bool = True, start: str | None = "2024-04-01"
) -> Figure:
    return Figure(
        metric=metric,
        value=value,
        unit=METRICS[metric].unit,
        reported_value=value,
        scale="units",
        period_start=None if METRICS[metric].period == "instant" else start,
        period_end=date(2024, 6, 30),
        block_id="b00001",
        quote=str(value),
        citation_valid=valid,
    )


def _record(*figures: Figure, error: str | None = None) -> ExtractionRecord:
    return ExtractionRecord(
        accession_number=ACC,
        extractor="fake",
        model="claude-opus-5-5",
        created_at=datetime(2024, 8, 1, tzinfo=UTC),
        input_tokens=1000,
        output_tokens=100,
        cost_usd=0.006,
        latency_s=2.0,
        error=error,
        figures=list(figures),
    )


def test_scoring_outcomes() -> None:
    record = _record(
        _figure(Metric.REVENUE, 25_010e6),  # within 0.5%
        _figure(Metric.DILUTED_EPS, 0.85),  # off by 2 cents
        _figure(Metric.TOTAL_ASSETS, 3_258_000e6, valid=False),
        _figure(Metric.NET_INTEREST_MARGIN, 1.95),  # unscored: no ground truth
        # operating cash flow missing
    )
    cells = {c.metric: c for c in score_filing(_filing(), record, QUARTER_FACTS, is_bank=True)}
    assert cells[Metric.REVENUE].outcome == "correct" and not cells[Metric.REVENUE].exact
    assert cells[Metric.REVENUE].period_ok
    assert cells[Metric.DILUTED_EPS].outcome == "wrong"
    assert cells[Metric.TOTAL_ASSETS].outcome == "correct" and cells[Metric.TOTAL_ASSETS].exact
    assert cells[Metric.OPERATING_CASH_FLOW].outcome == "missed"
    assert cells[Metric.NET_INTEREST_MARGIN].outcome == "unscored"
    assert cells[Metric.CET1_RATIO].outcome == "unscored"


def test_bank_metrics_must_be_null_for_non_banks() -> None:
    record = _record(_figure(Metric.NET_INTEREST_MARGIN, 2.0))
    cells = {
        c.metric: c for c in score_filing(_filing("PYPL"), record, QUARTER_FACTS, is_bank=False)
    }
    assert cells[Metric.NET_INTEREST_MARGIN].outcome == "hallucinated"
    assert cells[Metric.CET1_RATIO].outcome == "correct_null"
    assert cells[Metric.NET_INTEREST_INCOME].outcome == "correct_null"


def test_report_aggregates_and_renders() -> None:
    good = _record(
        _figure(Metric.REVENUE, 25_000e6),
        _figure(Metric.TOTAL_ASSETS, 3_258_000e6, valid=False),
    )
    failed_filing = _filing()
    failed_filing = StoredFiling(**{**failed_filing.__dict__, "accession_number": OTHER_ACC})
    report = evaluate(
        [_filing(), failed_filing],
        {ACC: good, OTHER_ACC: _record(error="timeout")},
        {1: QUARTER_FACTS},
        banks={"BAC"},
        extractor="fake",
    )
    revenue = next(m for m in report.metrics if m.metric == Metric.REVENUE)
    assert (revenue.scored, revenue.correct, revenue.accuracy) == (1, 1, 1.0)
    assert report.overall.scored == 4  # revenue, EPS, assets, cash flow have ground truth
    assert report.overall.correct == 2
    assert report.overall.citation_validity == pytest.approx(0.5)
    assert report.failed == [OTHER_ACC]
    assert report.cost.filings == 1 and report.cost.per_filing(report.cost.cost_usd) == 0.006

    markdown = to_markdown(report)
    assert "| Revenue | 100.0% | 1/1 |" in markdown
    assert "| All metrics | 50.0% | 2/4 |" in markdown
    assert "avg 1,000 input / 100 output tokens" in markdown
