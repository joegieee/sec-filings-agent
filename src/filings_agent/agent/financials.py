"""The agent's structured data: one cited figure per (company, metric, period).

Each filing contributes the figures for its primary period. They come from the
LLM extraction when one is stored (cited by a verbatim quote), and otherwise from
the filing's own XBRL tags (cited by the XBRL concept). ``source`` says which, so
an answer never hides where a number came from.

Periods are labelled the way analysts say them: ``FY2024`` for a 10-K, ``Q3 2025``
for a 10-Q's quarter (and its period-end balance sheet), and ``9M 2025`` for a
10-Q's year-to-date cash flows. Quarter labels are calendar quarters of the period
end, which matches fiscal quarters for the December year-end companies in the set.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

from filings_agent.evals.ground_truth import CompanyFacts
from filings_agent.extract.models import METRICS, ExtractionRecord, Metric, Unit
from filings_agent.storage import StoredFiling

Source = Literal["extracted", "xbrl"]
SourceMode = Literal["auto", "extracted", "xbrl"]


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    company: str
    metric: Metric
    period: str  # FY2024, Q3 2025, 9M 2025
    period_start: date | None
    period_end: date
    value: float
    unit: Unit
    form: str
    accession_number: str
    filing_date: date
    url: str
    source: Source
    evidence: str  # the quoted line, or the XBRL concept

    @property
    def is_annual(self) -> bool:
        return self.period.startswith("FY")


def period_label(form: str, report_date: date, metric: Metric) -> str:
    if form.upper().startswith("10-K"):
        return f"FY{report_date.year}"
    quarter = (report_date.month - 1) // 3 + 1
    if METRICS[metric].period == "ytd" and quarter > 1:
        return f"{3 * quarter}M {report_date.year}"
    return f"Q{quarter} {report_date.year}"


_FY = re.compile(r"^(?:FY|FISCAL\s*(?:YEAR)?)?\s*'?(\d{4})$")
_Q = re.compile(r"^Q([1-4])\s*[-/ ]?\s*(?:FY)?\s*(\d{4})$")
_Q_REV = re.compile(r"^(\d{4})\s*[-/ ]?\s*Q([1-4])$")
_YTD = re.compile(r"^([369])M\s*[-/ ]?\s*(\d{4})$")
LATEST = {"LATEST": "latest", "LATEST ANNUAL": "latest_annual", "LATEST QUARTER": "latest_quarter"}


def normalize_period(text: str) -> str | None:
    """``"fy 2024"`` -> ``"FY2024"``, ``"2025-Q3"`` -> ``"Q3 2025"``; None if unrecognised."""
    t = " ".join(text.strip().upper().replace("_", " ").split())
    if t in LATEST:
        return LATEST[t]
    if m := _FY.match(t):
        return f"FY{m.group(1)}"
    if m := _Q.match(t):
        return f"Q{m.group(1)} {m.group(2)}"
    if m := _Q_REV.match(t):
        return f"Q{m.group(2)} {m.group(1)}"
    if m := _YTD.match(t):
        return f"Q1 {m.group(2)}" if m.group(1) == "3" else f"{m.group(1)}M {m.group(2)}"
    return None


@dataclass
class Lookup:
    facts: list[Fact] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # human-readable reasons


class FinancialsIndex:
    def __init__(self, facts: Iterable[Fact], companies: Mapping[str, str]) -> None:
        self.companies = dict(companies)  # ticker -> name
        self._by_key: dict[tuple[str, Metric], list[Fact]] = {}
        for fact in sorted(facts, key=lambda f: (f.period_end, f.is_annual)):
            self._by_key.setdefault((fact.ticker, fact.metric), []).append(fact)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_key.values())

    def facts(self, ticker: str, metric: Metric) -> list[Fact]:
        return list(self._by_key.get((ticker.upper(), metric), []))

    def periods(self) -> list[str]:
        """Every period label with at least one figure, oldest first."""
        seen: dict[str, date] = {}
        for facts in self._by_key.values():
            for f in facts:
                seen.setdefault(f.period, f.period_end)
        return sorted(seen, key=lambda p: (seen[p], p.startswith("FY")))

    def lookup(self, ticker: str, metric: Metric, periods: Sequence[str]) -> Lookup:
        ticker = ticker.upper()
        result = Lookup()
        if ticker not in self.companies:
            known = ", ".join(sorted(self.companies))
            result.missing.append(f"{ticker} is not in the filings corpus (covered: {known})")
            return result
        available = self.facts(ticker, metric)
        label = METRICS[metric].label
        if not available:
            reason = (
                "is a bank-only metric"
                if METRICS[metric].bank_only
                else "has no figure in the stored filings"
            )
            result.missing.append(f"{label} for {ticker}: {reason}")
            return result
        for raw in periods or ["latest"]:
            wanted = normalize_period(raw)
            match: list[Fact]
            if wanted is None:
                result.missing.append(f"unrecognised period {raw!r} (use FY2024, Q3 2025, 9M 2025)")
                continue
            if wanted == "latest":
                match = available[-1:]
            elif wanted == "latest_annual":
                match = [f for f in available if f.is_annual][-1:]
            elif wanted == "latest_quarter":
                match = [f for f in available if not f.is_annual][-1:]
            else:
                match = [f for f in available if f.period == wanted]
            if not match:
                have = ", ".join(f.period for f in available)
                hint = " (no 10-Q covers Q4: use the FY figure)" if wanted.startswith("Q4") else ""
                result.missing.append(
                    f"{label} for {ticker} in {wanted}: not in the filings{hint}; available: {have}"
                )
            for f in match:
                if f not in result.facts:
                    result.facts.append(f)
        return result


def _from_extraction(
    filing: StoredFiling, record: ExtractionRecord, report_date: date
) -> dict[Metric, Fact]:
    out = {}
    for fig in record.figures:
        if not fig.citation_valid:
            continue  # an uncited figure never reaches an answer
        out[fig.metric] = Fact(
            ticker=filing.ticker,
            company=filing.company_name,
            metric=fig.metric,
            period=period_label(filing.form, report_date, fig.metric),
            period_start=fig.period_start,
            period_end=fig.period_end,
            value=fig.value,
            unit=fig.unit,
            form=filing.form,
            accession_number=filing.accession_number,
            filing_date=filing.filing_date,
            url=fig.url or filing.url,
            source="extracted",
            evidence=fig.quote,
        )
    return out


def _from_xbrl(filing: StoredFiling, facts: CompanyFacts, report_date: date) -> dict[Metric, Fact]:
    out = {}
    for metric in Metric:
        expected = facts.expected(
            metric, accession=filing.accession_number, form=filing.form, report_date=report_date
        )
        if expected is None:
            continue
        out[metric] = Fact(
            ticker=filing.ticker,
            company=filing.company_name,
            metric=metric,
            period=period_label(filing.form, report_date, metric),
            period_start=expected.period_start,
            period_end=expected.period_end,
            value=expected.value,
            unit=METRICS[metric].unit,
            form=filing.form,
            accession_number=filing.accession_number,
            filing_date=filing.filing_date,
            url=filing.url,
            source="xbrl",
            evidence=f"XBRL us-gaap:{expected.concept}",
        )
    return out


def build_financials(
    filings: Sequence[StoredFiling],
    *,
    company_facts: Mapping[int, CompanyFacts | None],
    extractions: Mapping[str, ExtractionRecord | None] | None = None,
    mode: SourceMode = "auto",
    names: Mapping[str, str] | None = None,
) -> FinancialsIndex:
    """Figures for every filing. ``auto`` prefers a stored extraction, metric by metric.

    ``names`` (ticker -> display name) overrides the names EDGAR registered.
    """
    names = names or {}
    facts: list[Fact] = []
    for filing in filings:
        if filing.report_date is None:
            continue
        chosen: dict[Metric, Fact] = {}
        if mode in ("auto", "xbrl") and (cf := company_facts.get(filing.cik)) is not None:
            chosen.update(_from_xbrl(filing, cf, filing.report_date))
        record = (extractions or {}).get(filing.accession_number)
        if mode in ("auto", "extracted") and record is not None and record.error is None:
            chosen.update(_from_extraction(filing, record, filing.report_date))
        name = names.get(filing.ticker, filing.company_name)
        facts.extend(f.model_copy(update={"company": name}) for f in chosen.values())
    companies = {f.ticker: names.get(f.ticker, f.company_name) for f in filings}
    return FinancialsIndex(facts, companies)
