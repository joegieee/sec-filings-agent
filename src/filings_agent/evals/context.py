"""Check whether XBRL numeric magnitudes occur in selected extraction text.

This needs no LLM. Matches allow common scales and ignore sign; they do not
establish that the number belongs to the right metric, column or period.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from filings_agent.evals.ground_truth import CompanyFacts
from filings_agent.extract.citations import numbers_in
from filings_agent.extract.context import Context
from filings_agent.extract.models import METRICS, Metric
from filings_agent.storage import StoredFiling

CHARS_PER_TOKEN = 4  # rough average for English prose and number tables


def printed_numbers(context: Context) -> set[float]:
    return {round(n, 4) for block in context.blocks for n in numbers_in(block.text)}


def appears_in(metric: Metric, expected: float, numbers: set[float]) -> bool:
    """Whether the expected value is printed in the context, at any common scale."""
    value = abs(expected)
    if METRICS[metric].unit != "USD":
        return round(value, 4) in numbers
    return any(round(value / scale, 4) in numbers for scale in (1, 1e3, 1e6, 1e9))


class RecallScore(BaseModel):
    metric: Metric | None
    found: int = 0
    total: int = 0

    @property
    def recall(self) -> float | None:
        return self.found / self.total if self.total else None


class ContextEvalReport(BaseModel):
    filings: int
    avg_chars: float
    max_chars: int
    truncated: int
    metrics: list[RecallScore]
    overall: RecallScore
    misses: list[str]  # "TICKER FORM PERIOD metric"

    @property
    def avg_tokens(self) -> float:
        return self.avg_chars / CHARS_PER_TOKEN


def evaluate_context(
    filings: Sequence[StoredFiling],
    contexts: Mapping[str, Context],
    facts: Mapping[int, CompanyFacts | None],
) -> ContextEvalReport:
    scores = {m: RecallScore(metric=m) for m in Metric}
    misses: list[str] = []
    sizes: list[int] = []
    truncated = 0
    for filing in filings:
        context = contexts.get(filing.accession_number)
        company_facts = facts.get(filing.cik)
        if context is None or company_facts is None or filing.report_date is None:
            continue
        sizes.append(context.chars)
        truncated += int(context.truncated)
        numbers = printed_numbers(context)
        for metric in Metric:
            expected = company_facts.expected(
                metric,
                accession=filing.accession_number,
                form=filing.form,
                report_date=filing.report_date,
            )
            if expected is None:
                continue
            scores[metric].total += 1
            if appears_in(metric, expected.value, numbers):
                scores[metric].found += 1
            else:
                misses.append(f"{filing.ticker} {filing.form} {filing.report_date} {metric.value}")
    overall = RecallScore(
        metric=None,
        found=sum(s.found for s in scores.values()),
        total=sum(s.total for s in scores.values()),
    )
    return ContextEvalReport(
        filings=len(sizes),
        avg_chars=sum(sizes) / len(sizes) if sizes else 0.0,
        max_chars=max(sizes, default=0),
        truncated=truncated,
        metrics=[s for s in scores.values() if s.total],
        overall=overall,
        misses=misses,
    )
