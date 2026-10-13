"""Score extracted figures against XBRL ground truth.

For every (filing, metric) cell:

* the filing tags the metric in XBRL -> the extraction is ``correct`` (within
  tolerance), ``wrong`` or ``missed`` (null);
* the metric is bank-only and the company isn't a bank -> the right answer is
  null: ``correct_null`` or ``hallucinated``;
* otherwise there is no ground truth -> ``unscored`` (citations still checked).

Citation validity is reported for every non-null figure: the cited block must
be one the model was shown, and the quote must appear in it and show the number.
Context recall (is the expected number printed in the blocks the model saw?)
tells retrieval misses apart from model mistakes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Literal

from pydantic import BaseModel

from filings_agent.evals.context import appears_in
from filings_agent.evals.ground_truth import CompanyFacts
from filings_agent.extract.models import METRICS, ExtractionRecord, Metric
from filings_agent.storage import StoredFiling

Outcome = Literal["correct", "wrong", "missed", "correct_null", "hallucinated", "unscored"]

DEFAULT_REL_TOL = 0.005  # 0.5%
EPS_ABS_TOL = 0.005  # half a cent


class Cell(BaseModel):
    ticker: str
    accession_number: str
    form: str
    report_date: date
    metric: Metric
    outcome: Outcome
    exact: bool = False
    expected: float | None = None
    concept: str | None = None
    predicted: float | None = None
    period_ok: bool | None = None
    in_context: bool | None = None
    citation_valid: bool | None = None
    quote: str | None = None


class MetricScore(BaseModel):
    metric: Metric | None  # None for the all-metrics total
    scored: int = 0
    correct: int = 0
    exact: int = 0
    wrong: int = 0
    missed: int = 0
    in_context: int = 0
    expected_null: int = 0
    correct_null: int = 0
    extracted: int = 0
    citations_valid: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.scored if self.scored else None

    @property
    def exact_rate(self) -> float | None:
        return self.exact / self.scored if self.scored else None

    @property
    def context_recall(self) -> float | None:
        return self.in_context / self.scored if self.scored else None

    @property
    def null_accuracy(self) -> float | None:
        return self.correct_null / self.expected_null if self.expected_null else None

    @property
    def citation_validity(self) -> float | None:
        return self.citations_valid / self.extracted if self.extracted else None


class CostSummary(BaseModel):
    filings: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    latency_s: float = 0.0
    cached: int = 0

    def per_filing(self, value: float | None) -> float | None:
        return value / self.filings if value is not None and self.filings else None


class ExtractionEvalReport(BaseModel):
    extractor: str
    rel_tol: float
    filings: int
    failed: list[str]
    unscored_filings: list[str]
    metrics: list[MetricScore]
    overall: MetricScore
    cost: CostSummary
    cells: list[Cell]


def _matches(metric: Metric, predicted: float, expected: float, rel_tol: float) -> bool:
    if METRICS[metric].unit == "USD/share":
        return abs(predicted - expected) <= EPS_ABS_TOL + 1e-9
    return math.isclose(predicted, expected, rel_tol=rel_tol, abs_tol=1e-9)


def _exact(metric: Metric, predicted: float, expected: float) -> bool:
    if METRICS[metric].unit == "USD/share":
        return round(predicted, 2) == round(expected, 2)
    return math.isclose(predicted, expected, rel_tol=1e-9, abs_tol=0.5)


def score_filing(
    filing: StoredFiling,
    record: ExtractionRecord,
    facts: CompanyFacts | None,
    *,
    is_bank: bool,
    rel_tol: float = DEFAULT_REL_TOL,
    context_numbers: set[float] | None = None,
) -> list[Cell]:
    if filing.report_date is None:
        raise ValueError(f"{filing.accession_number} has no report date")
    cells = []
    for metric in Metric:
        spec = METRICS[metric]
        figure = record.figure(metric)
        expected = (
            facts.expected(
                metric,
                accession=filing.accession_number,
                form=filing.form,
                report_date=filing.report_date,
            )
            if facts is not None
            else None
        )
        cell = Cell(
            ticker=filing.ticker,
            accession_number=filing.accession_number,
            form=filing.form,
            report_date=filing.report_date,
            metric=metric,
            outcome="unscored",
            predicted=figure.value if figure else None,
            citation_valid=figure.citation_valid if figure else None,
            quote=figure.quote if figure else None,
        )
        if expected is not None:
            cell.expected, cell.concept = expected.value, expected.concept
            if context_numbers is not None:
                cell.in_context = appears_in(metric, expected.value, context_numbers)
            if figure is None:
                cell.outcome = "missed"
            else:
                ok = _matches(metric, figure.value, expected.value, rel_tol)
                cell.outcome = "correct" if ok else "wrong"
                cell.exact = ok and _exact(metric, figure.value, expected.value)
                cell.period_ok = figure.period_end == expected.period_end and (
                    spec.period == "instant" or figure.period_start == expected.period_start
                )
        elif spec.bank_only and not is_bank:
            cell.outcome = "correct_null" if figure is None else "hallucinated"
        cells.append(cell)
    return cells


def _aggregate(metric: Metric | None, cells: Sequence[Cell]) -> MetricScore:
    score = MetricScore(metric=metric)
    for c in cells:
        if c.outcome in ("correct", "wrong", "missed"):
            score.scored += 1
            score.in_context += int(bool(c.in_context))
        if c.outcome == "correct":
            score.correct += 1
            score.exact += int(c.exact)
        elif c.outcome == "wrong":
            score.wrong += 1
        elif c.outcome == "missed":
            score.missed += 1
        elif c.outcome in ("correct_null", "hallucinated"):
            score.expected_null += 1
            score.correct_null += int(c.outcome == "correct_null")
        if c.predicted is not None:
            score.extracted += 1
            score.citations_valid += int(bool(c.citation_valid))
    return score


def evaluate(
    filings: Sequence[StoredFiling],
    records: Mapping[str, ExtractionRecord | None],
    facts: Mapping[int, CompanyFacts | None],
    *,
    banks: set[str],
    extractor: str,
    rel_tol: float = DEFAULT_REL_TOL,
    context_numbers: Mapping[str, set[float]] | None = None,
) -> ExtractionEvalReport:
    cells: list[Cell] = []
    failed: list[str] = []
    unscored: list[str] = []
    cost = CostSummary()
    for filing in filings:
        record = records.get(filing.accession_number)
        if record is None or record.error is not None or filing.report_date is None:
            failed.append(filing.accession_number)
            continue
        filing_cells = score_filing(
            filing,
            record,
            facts.get(filing.cik),
            is_bank=filing.ticker in banks,
            rel_tol=rel_tol,
            context_numbers=(context_numbers or {}).get(filing.accession_number),
        )
        if all(c.outcome == "unscored" for c in filing_cells):
            unscored.append(filing.accession_number)
        cells.extend(filing_cells)
        cost.filings += 1
        cost.input_tokens += record.input_tokens
        cost.output_tokens += record.output_tokens
        cost.latency_s += record.latency_s or 0.0
        cost.cached += int(record.cached)
        if record.cost_usd is None or cost.cost_usd is None:
            cost.cost_usd = None
        else:
            cost.cost_usd += record.cost_usd

    metrics = [_aggregate(m, [c for c in cells if c.metric == m]) for m in Metric]
    overall = _aggregate(None, cells)
    return ExtractionEvalReport(
        extractor=extractor,
        rel_tol=rel_tol,
        filings=len(filings),
        failed=failed,
        unscored_filings=unscored,
        metrics=metrics,
        overall=overall,
        cost=cost,
        cells=cells,
    )
