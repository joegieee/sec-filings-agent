"""Render eval reports as terminal tables or Markdown (for the README)."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from filings_agent.evals.context import ContextEvalReport, RecallScore
from filings_agent.evals.extraction import ExtractionEvalReport, MetricScore
from filings_agent.evals.qa import QASummary
from filings_agent.extract.models import METRICS


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fraction(num: int, den: int) -> str:
    return f"{num}/{den}" if den else "-"


def _rows(report: ExtractionEvalReport) -> list[list[str]]:
    def row(label: str, s: MetricScore) -> list[str]:
        return [
            label,
            _pct(s.accuracy),
            _fraction(s.correct, s.scored),
            _pct(s.exact_rate),
            str(s.missed) if s.scored else "-",
            _pct(s.context_recall),
            _pct(s.null_accuracy),
            _pct(s.citation_validity),
        ]

    rows = [row(METRICS[s.metric].label, s) for s in report.metrics if s.metric is not None]
    rows.append(row("All metrics", report.overall))
    return rows


_HEADERS = [
    "Metric",
    "Accuracy (±{tol})",
    "Correct",
    "Exact",
    "Missed",
    "In context",
    "Null accuracy",
    "Citations valid",
]


def _headers(report: ExtractionEvalReport) -> list[str]:
    return [h.format(tol=f"{report.rel_tol:.1%}") for h in _HEADERS]


def _cost_line(report: ExtractionEvalReport) -> str:
    c = report.cost
    cost = c.per_filing(c.cost_usd)
    parts = [
        f"{c.filings} filings extracted",
        f"avg {c.per_filing(c.input_tokens) or 0:,.0f} input / "
        f"{c.per_filing(c.output_tokens) or 0:,.0f} output tokens",
        f"avg ${cost:.4f}" if cost is not None else "cost n/a (unpriced model)",
        f"avg {c.per_filing(c.latency_s) or 0:.1f}s per filing",
    ]
    if c.cached:
        parts.append(f"{c.cached} served from cache (original usage shown)")
    return "; ".join(parts)


def print_report(report: ExtractionEvalReport, console: Console) -> None:
    table = Table(title=f"Extraction eval vs XBRL ({report.extractor})")
    for h in _headers(report):
        table.add_column(h, justify="left" if h == "Metric" else "right")
    for r in _rows(report):
        table.add_row(*r)
    console.print(table)
    console.print(_cost_line(report))
    if report.failed:
        console.print(
            f"[red]{len(report.failed)} filings without a usable extraction:[/red] "
            + ", ".join(report.failed)
        )
    if report.unscored_filings:
        console.print(
            "Not yet in XBRL company facts (unscored): " + ", ".join(report.unscored_filings)
        )


def to_markdown(report: ExtractionEvalReport) -> str:
    headers = _headers(report)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
        *("| " + " | ".join(r) + " |" for r in _rows(report)),
        "",
        _cost_line(report) + ".",
    ]
    return "\n".join(lines)


def _context_rows(report: ContextEvalReport) -> list[list[str]]:
    def row(label: str, s: RecallScore) -> list[str]:
        return [label, _pct(s.recall), _fraction(s.found, s.total)]

    rows = [row(METRICS[s.metric].label, s) for s in report.metrics if s.metric is not None]
    rows.append(row("All metrics", report.overall))
    return rows


def _context_line(report: ContextEvalReport) -> str:
    return (
        f"{report.filings} filings; context avg {report.avg_chars:,.0f} chars "
        f"(~{report.avg_tokens:,.0f} tokens), max {report.max_chars:,}; "
        f"{report.truncated} truncated to the budget"
    )


def print_context_report(report: ContextEvalReport, console: Console) -> None:
    table = Table(title="Context recall: ground-truth figures present in the extraction context")
    for h in ("Metric", "Recall", "Found"):
        table.add_column(h, justify="left" if h == "Metric" else "right")
    for r in _context_rows(report):
        table.add_row(*r)
    console.print(table)
    console.print(_context_line(report))
    for miss in report.misses:
        console.print(f"  missing: {miss}")


def context_markdown(report: ContextEvalReport) -> str:
    lines = [
        "| Metric | Recall | Found |",
        "|---|---:|---:|",
        *("| " + " | ".join(r) + " |" for r in _context_rows(report)),
        "",
        _context_line(report) + ".",
    ]
    return "\n".join(lines)


# --- Q&A eval ---------------------------------------------------------------------

_KIND_LABELS = {
    "value": "Single figure",
    "change": "Change over time",
    "ratio": "Ratio",
    "compare": "Comparison",
    "text": "From filing text",
    "quoted": "From a quoted figure",
    "unanswerable": "Unanswerable (must abstain)",
}


def _qa_rows(s: QASummary) -> list[list[str]]:
    rows = [
        ["Answer accuracy (answerable)", _pct(s.accuracy), _fraction(s.correct, s.answerable)],
        [
            "Abstention accuracy (unanswerable)",
            _pct(s.abstention_accuracy),
            _fraction(s.correct_abstentions, s.unanswerable),
        ],
        [
            "False abstentions (answerable)",
            _pct(s.false_abstention_rate),
            _fraction(s.false_abstentions, s.answerable),
        ],
        ["Citation precision", _pct(s.citation_precision), _fraction(s.cited_relevant, s.cited)],
    ]
    for kind, label in _KIND_LABELS.items():
        if kind in s.by_kind:
            ok, total = s.by_kind[kind]
            rows.append([f"  {label}", _pct(ok / total if total else None), f"{ok}/{total}"])
    return rows


def _qa_cost_line(s: QASummary) -> str:
    cost = s.per_question(s.cost_usd)
    parts = [
        f"{s.run}/{s.questions} questions run",
        f"avg {s.per_question(s.llm_calls) or 0:.1f} LLM calls",
        f"{s.per_question(s.input_tokens) or 0:,.0f} input / "
        f"{s.per_question(s.output_tokens) or 0:,.0f} output tokens",
        f"avg ${cost:.4f}" if cost is not None else "cost n/a (unpriced model)",
        f"avg {s.per_question(s.latency_s) or 0:.1f}s per question",
        f"{s.guardrail_triggers} guardrail abstentions",
    ]
    if s.errors:
        parts.append(f"{s.errors} failed calls")
    return "; ".join(parts)


def print_qa_report(s: QASummary, console: Console) -> None:
    table = Table(title=f"Q&A eval ({s.agent})")
    for h in ("Measure", "Score", "Count"):
        table.add_column(h, justify="left" if h == "Measure" else "right")
    for row in _qa_rows(s):
        table.add_row(*row)
    console.print(table)
    console.print(_qa_cost_line(s))
    for r in s.results:
        if r.outcome in ("wrong", "false_abstention", "error"):
            console.print(
                f"  [red]{r.outcome}[/red] {r.id}: {r.answer or r.error}"
                + (
                    f" (expected {r.reference}: {r.expected:,.4g})"
                    if r.expected is not None
                    else ""
                )
            )
    not_run = [r for r in s.results if r.outcome == "not_run"]
    if not_run:
        console.print(f"[yellow]{len(not_run)} questions not run: {not_run[0].error}[/yellow]")


def qa_markdown(s: QASummary) -> str:
    lines = [
        "| Measure | Score | Count |",
        "|---|---:|---:|",
        *("| " + " | ".join(r) + " |" for r in _qa_rows(s)),
        "",
        _qa_cost_line(s) + ".",
    ]
    return "\n".join(lines)
