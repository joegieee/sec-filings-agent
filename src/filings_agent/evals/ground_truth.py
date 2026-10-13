"""Expected figures per filing, from EDGAR's XBRL company facts.

Company facts list every value a company tagged, with the filing (``accn``) it
came from and the period it covers. For a filing and a metric we take the fact
from *that* filing whose period matches the metric's primary period (the quarter,
year to date, fiscal year or period-end date). Concepts are tried in order,
because filers tag the same line differently: JPMorgan reports total net revenue
as ``RevenuesNetOfInterestExpense``, Bank of America and PayPal as ``Revenues``.

Only standard taxonomies appear in company facts, so company-specific tags
(net interest margin, CET1 ratio) have no ground truth here; those metrics are
scored for citation validity only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from filings_agent.extract.models import METRICS, Metric

CONCEPTS: dict[Metric, tuple[str, ...]] = {
    Metric.REVENUE: ("RevenuesNetOfInterestExpense", "Revenues"),
    Metric.NET_INCOME: ("NetIncomeLoss",),
    Metric.DILUTED_EPS: ("EarningsPerShareDiluted",),
    Metric.TOTAL_ASSETS: ("Assets",),
    Metric.TOTAL_EQUITY: (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    Metric.OPERATING_CASH_FLOW: (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    Metric.NET_INTEREST_INCOME: ("InterestIncomeExpenseNet",),
    Metric.NET_INTEREST_MARGIN: (),
    Metric.CET1_RATIO: (),
}
_UNITS = ("USD", "USD/shares")

# Duration windows (days) for "a quarter" and "a fiscal year".
_QUARTER = range(80, 101)
_YEAR = range(350, 381)


@dataclass(frozen=True)
class Expected:
    value: float
    concept: str
    period_start: date | None
    period_end: date


class CompanyFacts:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._gaap: Mapping[str, Any] = payload.get("facts", {}).get("us-gaap", {})

    def _facts(self, concept: str, accession: str) -> list[dict[str, Any]]:
        units = self._gaap.get(concept, {}).get("units", {})
        return [f for unit in _UNITS for f in units.get(unit, []) if f.get("accn") == accession]

    def expected(
        self, metric: Metric, *, accession: str, form: str, report_date: date
    ) -> Expected | None:
        spec = METRICS[metric]
        annual = form.upper().startswith("10-K")
        for concept in CONCEPTS[metric]:
            candidates: list[tuple[int, dict[str, Any]]] = []
            for fact in self._facts(concept, accession):
                if date.fromisoformat(fact["end"]) != report_date:
                    continue
                start = fact.get("start")
                if spec.period == "instant":
                    if start is None:
                        candidates.append((0, fact))
                    continue
                if start is None:
                    continue
                days = (report_date - date.fromisoformat(start)).days
                if annual or spec.period == "ytd":
                    # Fiscal year for a 10-K; for a 10-Q's cash flows, the longest
                    # window ending at the period end is the year to date.
                    if (annual and days in _YEAR) or (not annual and days < _YEAR.start):
                        candidates.append((days, fact))
                elif days in _QUARTER:
                    candidates.append((days, fact))
            if candidates:
                _, fact = max(candidates, key=lambda c: c[0])
                start = fact.get("start")
                return Expected(
                    value=float(fact["val"]),
                    concept=concept,
                    period_start=date.fromisoformat(start) if start else None,
                    period_end=report_date,
                )
        return None
