"""The agent's tools, and the registry of every source they hand the model.

Each figure, passage and calculation a tool returns gets a short id (``F1``,
``S1``, ``C1``) that the model cites. Results are rendered as plain text in a
fixed order, so the same question replays the same prompts (and hits the
response cache).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from filings_agent.agent.calculator import CalcError, evaluate
from filings_agent.agent.financials import Fact, FinancialsIndex, normalize_period
from filings_agent.agent.models import SourceRef, ToolCall
from filings_agent.agent.retrieval import Passage, SearchIndex
from filings_agent.extract.models import METRICS, Unit
from filings_agent.parse.models import SECTION_TITLES

PASSAGE_CHARS = 700
SEARCH_RESULTS = 4
MAX_CALLS_PER_STEP = 8


def format_value(value: float, unit: Unit) -> str:
    if unit == "percent":
        return f"{value:.2f}%"
    if unit == "USD/share":
        return f"${value:,.2f} per share"
    sign = "-" if value < 0 else ""
    millions = abs(value) / 1e6
    text = f"{sign}${millions:,.0f} million"
    if abs(value) >= 1e9:
        text += f" ({sign}${abs(value) / 1e9:,.2f} billion)"
    return text


def format_number(value: float) -> str:
    if abs(value) >= 1e6:
        return f"{value:,.0f}"
    return f"{value:,.6f}".rstrip("0").rstrip(".")


class Sources:
    """Every source shown to the model in one run, by id."""

    def __init__(self) -> None:
        self.by_id: dict[str, SourceRef] = {}
        self._keys: dict[tuple[object, ...], str] = {}
        self._counts = {"F": 0, "S": 0, "C": 0}

    def _add(self, prefix: str, key: tuple[object, ...], build: Callable[[str], SourceRef]) -> str:
        if key in self._keys:
            return self._keys[key]
        self._counts[prefix] += 1
        sid = f"{prefix}{self._counts[prefix]}"
        self.by_id[sid] = build(sid)
        self._keys[key] = sid
        return sid

    def fact(self, f: Fact) -> SourceRef:
        def build(sid: str) -> SourceRef:
            span = (
                f"{f.period_start} to {f.period_end}" if f.period_start else f"as of {f.period_end}"
            )
            origin = f'quoted "{f.evidence}"' if f.source == "extracted" else f"tagged {f.evidence}"
            text = (
                f"[{sid}] {f.company} ({f.ticker}) {METRICS[f.metric].label.lower()}, {f.period} "
                f"({span}): {format_value(f.value, f.unit)} = {format_number(f.value)} {f.unit}. "
                f"From its {f.form} filed {f.filing_date}, {origin}."
            )
            return SourceRef(
                id=sid,
                kind="fact",
                text=text,
                ticker=f.ticker,
                metric=f.metric,
                period=f.period,
                value=f.value,
                url=f.url,
                evidence=f.evidence,
                source=f.source,
            )

        return self.by_id[self._add("F", (f.accession_number, f.metric), build)]

    def passage(self, p: Passage) -> SourceRef:
        def build(sid: str) -> SourceRef:
            where = f"{p.ticker} {p.form} {p.period}, {SECTION_TITLES[p.section]}"
            if p.page_label:
                where += f", page {p.page_label}"
            body = p.text if len(p.text) <= PASSAGE_CHARS else p.text[:PASSAGE_CHARS] + " ..."
            return SourceRef(
                id=sid,
                kind="passage",
                text=f'[{sid}] {where}: "{body}"',
                ticker=p.ticker,
                period=p.period,
                url=p.url,
                evidence=body,
            )

        return self.by_id[self._add("S", (p.accession_number, p.block_id), build)]

    def calculation(self, expression: str, value: float, inputs: list[str]) -> SourceRef:
        def build(sid: str) -> SourceRef:
            return SourceRef(
                id=sid,
                kind="calculation",
                text=f"[{sid}] {expression} = {format_number(value)}",
                value=value,
                evidence=expression,
                inputs=inputs,
            )

        return self.by_id[self._add("C", (expression,), build)]

    def variables(self) -> dict[str, float]:
        return {sid: s.value for sid, s in self.by_id.items() if s.value is not None}


_SOURCE_ID = re.compile(r"\b[FSC]\d+\b")


def referenced_ids(expression: str, known: set[str]) -> list[str]:
    return [t for t in dict.fromkeys(_SOURCE_ID.findall(expression)) if t in known]


class Toolbox:
    def __init__(self, financials: FinancialsIndex, search: SearchIndex | None) -> None:
        self.financials = financials
        self.search = search

    def run(self, call: ToolCall, sources: Sources) -> str:
        try:
            if call.tool == "get_financials":
                return self._get_financials(call, sources)
            if call.tool == "compare":
                return self._compare(call, sources)
            if call.tool == "search_filings":
                return self._search(call, sources)
            return self._calculate(call, sources)
        except ToolError as exc:
            return f"error: {exc}"

    def _get_financials(self, call: ToolCall, sources: Sources) -> str:
        company, metric = _require(call.company, "company"), _require(call.metric, "metric")
        found = self.financials.lookup(company, metric, call.periods or ["latest"])
        lines = [sources.fact(f).text for f in found.facts]
        lines += [f"not found: {m}" for m in found.missing]
        return "\n".join(lines) or "no figures"

    def _compare(self, call: ToolCall, sources: Sources) -> str:
        companies = call.companies or ([call.company] if call.company else [])
        if len(companies) < 2:
            raise ToolError("compare needs at least two companies")
        metric = _require(call.metric, "metric")
        period = _require(call.period, "period")
        lines: list[str] = []
        ranked: list[tuple[float, str]] = []
        for company in companies:
            found = self.financials.lookup(company, metric, [period])
            for f in found.facts:
                ref = sources.fact(f)
                lines.append(ref.text)
                ranked.append((f.value, f"{f.ticker} {format_value(f.value, f.unit)} [{ref.id}]"))
            lines += [f"not found: {m}" for m in found.missing]
        if len(ranked) >= 2:
            order = " > ".join(label for _, label in sorted(ranked, key=lambda r: -r[0]))
            lines.append(f"ranking by {METRICS[metric].label.lower()}, highest first: {order}")
        return "\n".join(lines)

    def _search(self, call: ToolCall, sources: Sources) -> str:
        if self.search is None:
            raise ToolError("the search index has not been built (run filings-agent index)")
        query = _require(call.query, "query")
        period = None
        if call.period:
            period = normalize_period(call.period)
            if period is None or period.startswith("latest"):
                raise ToolError(f"unrecognised period {call.period!r} (use FY2024 or Q3 2025)")
            if period[0].isdigit():  # 9M 2025: passages are labelled by the filing's quarter
                period = f"Q{int(period[0]) // 3} {period[-4:]}"
        tickers = [call.company.upper()] if call.company else None
        if tickers and tickers[0] not in self.financials.companies:
            known = ", ".join(sorted(self.financials.companies))
            raise ToolError(f"{tickers[0]} is not in the filings corpus (covered: {known})")
        hits = self.search.search(
            query, tickers=tickers, section=call.section, period=period, limit=SEARCH_RESULTS
        )
        if not hits:
            return "no matching passages"
        return "\n".join(sources.passage(p).text for p in hits)

    def _calculate(self, call: ToolCall, sources: Sources) -> str:
        expression = _require(call.expression, "expression")
        variables = sources.variables()
        try:
            value = evaluate(expression, variables)
        except CalcError as exc:
            raise ToolError(str(exc)) from exc
        ref = sources.calculation(expression, value, referenced_ids(expression, set(variables)))
        return ref.text


class ToolError(Exception):
    pass


def _require[T](value: T | None, name: str) -> T:
    if value is None or value == "":
        raise ToolError(f"missing argument {name!r}")
    return value
