"""Q&A eval: does the agent answer analyst questions correctly, cite well, and abstain?

Numeric reference answers are computed from XBRL company facts (via a
``FinancialsIndex`` built from XBRL only), so no number is typed in by hand.
Scoring is mechanical, with no LLM judge:

* value / change / ratio: some number in the answer matches the reference at the
  precision it is written (and within 5%);
* compare: the answer names the right company and gives its value;
* text: every keyword group appears, and a passage from that company is cited;
* unanswerable: the agent abstained.

Citation precision is the share of cited sources that are about the question's
companies (and, for figures, its metrics). Answers the guardrail replaced count as
abstentions.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from filings_agent.agent.financials import FinancialsIndex
from filings_agent.agent.grounding import mentions
from filings_agent.agent.models import AgentAnswer, SourceRef
from filings_agent.extract.models import METRICS, Metric
from filings_agent.llm.base import LLMConfigError

QuestionKind = Literal["value", "change", "ratio", "compare", "text", "unanswerable"]
MAX_REL_ERROR = 0.05


class Question(BaseModel):
    id: str
    question: str
    kind: QuestionKind
    ticker: str | None = None
    tickers: list[str] = Field(default_factory=list)
    metric: Metric | None = None
    denominator: Metric | None = None
    period: str | None = None
    period_to: str | None = None
    pick: Literal["highest", "lowest"] = "highest"
    keywords: list[list[str]] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def _fields_for_kind(self) -> Question:
        need: dict[str, tuple[str, ...]] = {
            "value": ("ticker", "metric", "period"),
            "change": ("ticker", "metric", "period", "period_to"),
            "ratio": ("ticker", "metric", "denominator", "period"),
            "compare": ("tickers", "metric", "period"),
            "text": ("ticker", "keywords"),
            "unanswerable": (),
        }
        missing = [f for f in need[self.kind] if not getattr(self, f)]
        if missing:
            raise ValueError(f"{self.id}: a {self.kind} question needs {', '.join(missing)}")
        return self

    @property
    def answerable(self) -> bool:
        return self.kind != "unanswerable"

    @property
    def all_tickers(self) -> list[str]:
        return self.tickers or ([self.ticker] if self.ticker else [])

    @property
    def metrics(self) -> set[Metric]:
        return {m for m in (self.metric, self.denominator) if m is not None}


def load_questions(path: Path) -> list[Question]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    questions = [Question.model_validate(q) for q in data.get("questions", [])]
    ids = [q.id for q in questions]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate question ids: {sorted(dupes)}")
    return questions


class Reference(BaseModel):
    value: float | None = None  # the number the answer must contain
    unit: Literal["USD", "USD/share", "percent"] | None = None
    winner: str | None = None  # compare: the ticker the answer must name
    description: str


def _one(index: FinancialsIndex, ticker: str, metric: Metric, period: str) -> float:
    found = index.lookup(ticker, metric, [period])
    if len(found.facts) != 1:
        raise LookupError(f"no XBRL figure for {ticker} {metric.value} {period}")
    return found.facts[0].value


def reference(q: Question, index: FinancialsIndex) -> Reference | None:
    """The reference answer from XBRL figures; None for text and unanswerable questions."""
    if q.kind in ("text", "unanswerable"):
        return None
    assert q.metric is not None and q.period is not None
    label = METRICS[q.metric].label.lower()
    if q.kind == "value":
        assert q.ticker
        v = _one(index, q.ticker, q.metric, q.period)
        return Reference(
            value=v, unit=METRICS[q.metric].unit, description=f"{q.ticker} {label} {q.period}"
        )
    if q.kind == "change":
        assert q.ticker and q.period_to
        old = _one(index, q.ticker, q.metric, q.period)
        new = _one(index, q.ticker, q.metric, q.period_to)
        return Reference(
            value=(new - old) / abs(old) * 100,
            unit="percent",
            description=f"{q.ticker} {label} change {q.period} to {q.period_to}",
        )
    if q.kind == "ratio":
        assert q.ticker and q.denominator
        num = _one(index, q.ticker, q.metric, q.period)
        den = _one(index, q.ticker, q.denominator, q.period)
        return Reference(
            value=num / den * 100,
            unit="percent",
            description=f"{q.ticker} {label} / {METRICS[q.denominator].label.lower()} {q.period}",
        )
    values = {t: _one(index, t, q.metric, q.period) for t in q.tickers}
    pick: Callable[..., str] = max if q.pick == "highest" else min
    winner = pick(values, key=lambda t: values[t])
    return Reference(
        value=values[winner],
        unit=METRICS[q.metric].unit,
        winner=winner,
        description=f"{q.pick} {label} {q.period}: {winner}",
    )


def mentions_value(text: str, value: float, unit: str | None) -> bool:
    """Whether ``text`` states ``value`` at the precision it is written (within 5%)."""
    for m in mentions(text):
        if m.scale:
            scales = [m.scale]
        else:
            scales = [1.0, 1e3, 1e6, 1e9] if unit == "USD" else [1.0]
        for scale in scales:
            close = math.isclose(abs(m.literal) * scale, abs(value), rel_tol=MAX_REL_ERROR)
            if close and m.matches(value, scales=[scale]):
                return True
    return False


Outcome = Literal["correct", "wrong", "false_abstention", "correct_abstention", "answered"]


class QAResult(BaseModel):
    id: str
    kind: QuestionKind
    question: str
    outcome: Outcome | Literal["error", "not_run"]
    answer: str | None = None
    abstained: bool | None = None
    guardrail: str | None = None
    reference: str | None = None
    expected: float | None = None
    cited: int = 0
    cited_relevant: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    cached: bool = False
    error: str | None = None
    citations: list[str] = Field(default_factory=list)


def _relevant(ref: SourceRef, q: Question, by_id: dict[str, SourceRef]) -> bool:
    tickers = set(q.all_tickers)
    if ref.kind == "fact":
        return ref.ticker in tickers and (not q.metrics or ref.metric in q.metrics)
    if ref.kind == "passage":
        return ref.ticker in tickers
    inputs = [by_id[i] for i in ref.inputs if i in by_id]
    return bool(inputs) and all(_relevant(i, q, by_id) for i in inputs)


_NAME_END = re.compile(r"\s+(?:Corp\b|Corporation\b|Inc\b|Holdings\b|&|Co\b)|,", re.I)
_GENERIC = {"bank", "the", "first", "national"}


def names_company(text: str, ticker: str, company: str) -> bool:
    """Whether ``text`` names the company: by ticker, short name, or distinctive first word."""
    lowered = text.lower()
    short = _NAME_END.split(company, maxsplit=1)[0].strip().lower()
    first = short.split()[0] if short else ""
    return (
        re.search(rf"\b{re.escape(ticker)}\b", text) is not None
        or (bool(short) and short in lowered)
        or (len(first) > 3 and first not in _GENERIC and first in lowered)
    )


def score(
    q: Question, answer: AgentAnswer, ref: Reference | None, companies: dict[str, str]
) -> QAResult:
    by_id = {s.id: s for s in answer.sources}
    relevant = [c for c in answer.citations if _relevant(c, q, by_id)]
    result = QAResult(
        id=q.id,
        kind=q.kind,
        question=q.question,
        outcome="answered",
        answer=answer.text,
        abstained=answer.abstained,
        guardrail=answer.guardrail,
        reference=ref.description if ref else None,
        expected=ref.value if ref else None,
        cited=0 if answer.abstained else len(answer.citations),
        cited_relevant=0 if answer.abstained else len(relevant),
        llm_calls=answer.llm_calls,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=answer.cost_usd,
        latency_s=round(answer.latency_s, 3),
        cached=answer.cached,
        error=answer.error,
        citations=[c.id for c in answer.citations],
    )
    if answer.error is not None:
        result.outcome = "error"  # the model call failed: not the agent's judgement
        return result
    if not q.answerable:
        result.outcome = "correct_abstention" if answer.abstained else "wrong"
        return result
    if answer.abstained:
        result.outcome = "false_abstention"
        return result
    text = answer.text
    if q.kind == "text":
        assert q.ticker
        found = all(any(k.lower() in text.lower() for k in group) for group in q.keywords)
        cites_passage = any(c.kind == "passage" and c.ticker == q.ticker for c in relevant)
        ok = found and cites_passage
    else:
        assert ref is not None and ref.value is not None
        ok = mentions_value(text, ref.value, ref.unit)
        if ref.winner is not None:
            ok = ok and names_company(text, ref.winner, companies.get(ref.winner, ref.winner))
    result.outcome = "correct" if ok else "wrong"
    return result


class QASummary(BaseModel):
    agent: str
    questions: int
    run: int
    answerable: int
    correct: int
    false_abstentions: int
    unanswerable: int
    correct_abstentions: int
    cited: int
    cited_relevant: int
    guardrail_triggers: int
    errors: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    by_kind: dict[str, tuple[int, int]]  # kind -> (correct, total)
    results: list[QAResult]

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.answerable if self.answerable else None

    @property
    def abstention_accuracy(self) -> float | None:
        return self.correct_abstentions / self.unanswerable if self.unanswerable else None

    @property
    def false_abstention_rate(self) -> float | None:
        return self.false_abstentions / self.answerable if self.answerable else None

    @property
    def citation_precision(self) -> float | None:
        return self.cited_relevant / self.cited if self.cited else None

    def per_question(self, total: float | None) -> float | None:
        return total / self.run if total is not None and self.run else None


def summarize(agent: str, questions: int, results: Sequence[QAResult]) -> QASummary:
    done = [r for r in results if r.outcome not in ("not_run", "error")]
    answerable = [r for r in done if r.kind != "unanswerable"]
    unanswerable = [r for r in done if r.kind == "unanswerable"]
    costs = [r.cost_usd for r in done]
    by_kind: dict[str, tuple[int, int]] = {}
    for r in done:
        ok, total = by_kind.get(r.kind, (0, 0))
        by_kind[r.kind] = (ok + int(r.outcome in ("correct", "correct_abstention")), total + 1)
    return QASummary(
        agent=agent,
        questions=questions,
        run=len(done),
        answerable=len(answerable),
        correct=sum(r.outcome == "correct" for r in answerable),
        false_abstentions=sum(r.outcome == "false_abstention" for r in answerable),
        unanswerable=len(unanswerable),
        correct_abstentions=sum(r.outcome == "correct_abstention" for r in unanswerable),
        cited=sum(r.cited for r in done),
        cited_relevant=sum(r.cited_relevant for r in done),
        guardrail_triggers=sum(r.guardrail is not None for r in done),
        errors=sum(r.outcome == "error" for r in results),
        llm_calls=sum(r.llm_calls for r in done),
        input_tokens=sum(r.input_tokens for r in done),
        output_tokens=sum(r.output_tokens for r in done),
        cost_usd=None if any(c is None for c in costs) else sum(c for c in costs if c),
        latency_s=sum(r.latency_s for r in done),
        by_kind=by_kind,
        results=list(results),
    )


def run_qa(
    questions: Sequence[Question],
    ask: Callable[[str], AgentAnswer],
    truth: FinancialsIndex,
    *,
    agent: str,
) -> QASummary:
    """Ask every question and score it. A spent daily budget (or another setup error)
    stops the run; the questions not reached are reported as ``not_run``."""
    results: list[QAResult] = []
    stopped: str | None = None
    for q in questions:
        if stopped is not None:
            results.append(
                QAResult(
                    id=q.id, kind=q.kind, question=q.question, outcome="not_run", error=stopped
                )
            )
            continue
        ref = reference(q, truth)
        try:
            answer = ask(q.question)
        except LLMConfigError as exc:
            stopped = str(exc)
            results.append(
                QAResult(
                    id=q.id, kind=q.kind, question=q.question, outcome="not_run", error=stopped
                )
            )
            continue
        results.append(score(q, answer, ref, truth.companies))
    return summarize(agent, len(questions), results)
