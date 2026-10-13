"""Command-line entry point: ``filings-agent``."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table

from filings_agent.agent.agent import DEFAULT_MAX_STEPS, Agent
from filings_agent.agent.financials import SourceMode, build_financials
from filings_agent.agent.models import AgentAnswer
from filings_agent.agent.retrieval import SearchIndex
from filings_agent.agent.tools import Toolbox
from filings_agent.config import Company, CompanyUniverse, LLMProvider, Settings
from filings_agent.edgar import EdgarClient
from filings_agent.evals.context import evaluate_context, printed_numbers
from filings_agent.evals.extraction import DEFAULT_REL_TOL, evaluate
from filings_agent.evals.ground_truth import CompanyFacts
from filings_agent.evals.qa import load_questions, run_qa
from filings_agent.evals.report import (
    context_markdown,
    print_context_report,
    print_qa_report,
    print_report,
    qa_markdown,
    to_markdown,
)
from filings_agent.extract.context import Context, select_context
from filings_agent.extract.extractor import extractor_id
from filings_agent.extract.models import METRICS, ExtractionRecord
from filings_agent.fsutil import atomic_write
from filings_agent.http import HttpClient, RateLimiter, RetryPolicy
from filings_agent.ingest import ingest
from filings_agent.llm import LLMConfigError, StructuredLLM
from filings_agent.llm.anthropic import Effort
from filings_agent.llm.factory import build_llm
from filings_agent.log import configure_logging, new_run_id
from filings_agent.parse.models import SectionKind
from filings_agent.pipeline import ExtractReport, extract_filings, load_or_parse, parse_filings
from filings_agent.storage import Store, StoredFiling

app = typer.Typer(no_args_is_help=True, help="Research agent over SEC filings.")
eval_app = typer.Typer(no_args_is_help=True, help="Run evaluations and print their results.")
app.add_typer(eval_app, name="eval")
console = Console()


TickersOption = Annotated[str | None, typer.Option(help="Comma-separated tickers, e.g. JPM,BAC")]
SetOption = Annotated[
    str | None, typer.Option("--set", help="Named set from companies.toml, e.g. core3")
]
FormOption = Annotated[str | None, typer.Option(help="Only this form, e.g. 10-K")]
ProviderOption = Annotated[
    LLMProvider | None, typer.Option(help="Override FRA_LLM_PROVIDER (gemini, ollama, ...)")
]


def _settings() -> Settings:
    try:
        return Settings()  # populated from env / .env
    except ValidationError as exc:
        console.print(f"[red]Invalid configuration:[/red]\n{exc}")
        raise typer.Exit(2) from exc


def _companies(settings: Settings, tickers: str | None, company_set: str | None) -> list[Company]:
    universe = CompanyUniverse.from_toml(settings.companies_file)
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()] if tickers else None
    try:
        return universe.select(ticker_list, company_set)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(2) from exc


def _stored_filings(
    store: Store, companies: list[Company], form: str | None, limit: int | None = None
) -> list[StoredFiling]:
    filings = [
        f
        for f in store.list_filings(ciks=[c.cik for c in companies])
        if f.raw_path is not None and (form is None or f.form == form.upper())
    ]
    if not filings:
        console.print("No downloaded filings match. Run [bold]filings-agent ingest[/bold] first.")
        raise typer.Exit(1)
    return filings[:limit] if limit else filings


def _llm(
    settings: Settings, provider: LLMProvider | None, model: str | None, effort: Effort | None
) -> StructuredLLM:
    try:
        return build_llm(settings, provider=provider, model=model, effort=effort)
    except LLMConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


@app.command("ingest")
def ingest_cmd(
    tickers: TickersOption = None,
    company_set: SetOption = None,
    since: Annotated[
        str | None, typer.Option(help="Earliest filing date (YYYY-MM-DD); default: lookback")
    ] = None,
    facts: Annotated[bool, typer.Option(help="Also fetch XBRL company facts")] = True,
    refresh_facts: Annotated[bool, typer.Option(help="Re-download company facts")] = False,
) -> None:
    """Download 10-K/10-Q filings (and XBRL facts) for the selected companies."""
    settings = _settings()
    configure_logging()
    run_id = new_run_id()

    companies = _companies(settings, tickers, company_set)
    start = (
        date.fromisoformat(since)
        if since
        else date.today() - timedelta(days=365 * settings.lookback_years)
    )

    limiter = RateLimiter(settings.sec_max_requests_per_second)
    retry = RetryPolicy(max_retries=settings.http_max_retries)
    with (
        HttpClient(
            user_agent=settings.sec_user_agent,
            rate_limiter=limiter,
            timeout=settings.http_timeout_seconds,
            retry=retry,
        ) as http,
        Store(settings.data_dir) as store,
    ):
        report = ingest(
            companies,
            edgar=EdgarClient(http),
            store=store,
            forms=settings.forms,
            since=start,
            fetch_facts=facts,
            refresh_facts=refresh_facts,
        )

    table = Table(title=f"Ingest run {run_id} (since {start})")
    for col in ("Ticker", "Discovered", "Downloaded", "Skipped", "Failed", "Status"):
        table.add_column(col)
    for r in report.results:
        status = "[green]ok[/green]" if r.ok else f"[red]{r.error or 'partial'}[/red]"
        table.add_row(
            r.ticker,
            str(r.discovered),
            str(r.downloaded),
            str(r.skipped),
            str(len(r.failed)),
            status,
        )
    console.print(table)
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show what is stored locally, per company and form."""
    settings = _settings()
    with Store(settings.data_dir) as store:
        filings = store.list_filings()
    if not filings:
        console.print("No filings stored yet. Run [bold]filings-agent ingest --set core3[/bold].")
        return
    counts = Counter((f.ticker, f.form) for f in filings)
    stored = Counter((f.ticker, f.form) for f in filings if f.raw_path is not None)
    latest: dict[tuple[str, str], date] = {}
    for f in filings:
        key = (f.ticker, f.form)
        latest[key] = max(latest.get(key, f.filing_date), f.filing_date)
    table = Table(title=f"Stored filings ({settings.data_dir})")
    for col in ("Ticker", "Form", "Listed", "Downloaded", "Latest filing"):
        table.add_column(col)
    for key in sorted(counts):
        table.add_row(*key, str(counts[key]), str(stored[key]), latest[key].isoformat())
    console.print(table)


_HOW = {"cross_reference": "xref", "table_of_contents": "toc"}


def _section_cell(chars: int, how: set[str]) -> str:
    """e.g. "358k xref": size, and how it was located if not by its own heading."""
    if not chars:
        return "-"
    notes = " ".join(sorted(_HOW[h] for h in how if h in _HOW))
    size = f"{chars / 1000:,.0f}k" if chars >= 1000 else "<1k"
    return f"{size} {notes}".strip()


@app.command("parse")
def parse_cmd(
    tickers: TickersOption = None,
    company_set: SetOption = None,
    form: FormOption = None,
    force: Annotated[
        bool, typer.Option(help="Re-parse even if an up-to-date parse exists")
    ] = False,
) -> None:
    """Split stored filings into section-tagged text blocks (Items 1A, 7, 7A, 8...)."""
    settings = _settings()
    configure_logging()
    run_id = new_run_id()
    companies = _companies(settings, tickers, company_set)
    with Store(settings.data_dir) as store:
        outcomes = parse_filings(store, _stored_filings(store, companies, form), force=force)

    table = Table(
        title=f"Parse run {run_id}: characters per section",
        caption="xref: located via a cross-reference; toc: via the table of contents",
    )
    for col in (
        "Ticker",
        "Form",
        "Period",
        "Blocks",
        "Risk factors",
        "MD&A",
        "Fin. statements",
        "Status",
    ):
        table.add_column(col)
    kinds = (SectionKind.RISK_FACTORS, SectionKind.MDNA, SectionKind.FINANCIAL_STATEMENTS)
    for o in outcomes:
        f = o.filing
        period = f.report_date.isoformat() if f.report_date else "?"
        if o.parsed is None:
            table.add_row(f.ticker, f.form, period, "-", "-", "-", "-", f"[red]{o.error}[/red]")
            continue
        cells = []
        for kind in kinds:
            sections = [s for s in o.parsed.sections if s.kind == kind]
            cells.append(_section_cell(sum(s.chars for s in sections), {s.how for s in sections}))
        status = "cached" if o.skipped else "[green]parsed[/green]"
        table.add_row(f.ticker, f.form, period, str(len(o.parsed.blocks)), *cells, status)
    console.print(table)
    if any(o.parsed is None for o in outcomes):
        raise typer.Exit(1)


def _print_extractions(report: ExtractReport, filings: list[StoredFiling]) -> None:
    by_accession = {r.accession_number: r for r in report.records}
    errors = dict(report.failed)
    table = Table(title=f"Extractions ({report.extractor})")
    for col in (
        "Ticker",
        "Form",
        "Period",
        "Figures",
        "Citations valid",
        "Tokens in/out",
        "Cost",
        "Seconds",
        "Status",
    ):
        table.add_column(col)
    for f in filings:
        period = f.report_date.isoformat() if f.report_date else "?"
        record = by_accession.get(f.accession_number)
        if record is None:
            error = errors.get(f.accession_number, "not run")
            table.add_row(f.ticker, f.form, period, *["-"] * 5, f"[red]{error}[/red]")
            continue
        valid = sum(fig.citation_valid for fig in record.figures)
        cost = f"${record.cost_usd:.4f}" if record.cost_usd is not None else "n/a"
        status = "cached" if record.cached else "[green]ok[/green]"
        table.add_row(
            f.ticker,
            f.form,
            period,
            f"{len(record.figures)}/{len(METRICS)}",
            f"{valid}/{len(record.figures)}",
            f"{record.input_tokens:,}/{record.output_tokens:,}",
            cost,
            f"{record.latency_s or 0:.1f}",
            status,
        )
    console.print(table)


def _run_extraction(
    store: Store, filings: list[StoredFiling], llm: StructuredLLM, *, force: bool, max_chars: int
) -> ExtractReport:
    try:
        return extract_filings(store, filings, llm, force=force, max_context_chars=max_chars)
    except LLMConfigError as exc:
        console.print(f"[red]Cannot run extraction: {exc}[/red]")
        raise typer.Exit(2) from exc


@app.command("extract")
def extract_cmd(
    tickers: TickersOption = None,
    company_set: SetOption = None,
    form: FormOption = None,
    limit: Annotated[
        int | None, typer.Option(help="At most this many filings (newest first)")
    ] = None,
    provider: ProviderOption = None,
    model: Annotated[str | None, typer.Option(help="Override FRA_LLM_MODEL")] = None,
    effort: Annotated[Effort | None, typer.Option(help="Override FRA_LLM_EFFORT")] = None,
    force: Annotated[bool, typer.Option(help="Re-extract even if a stored result exists")] = False,
) -> None:
    """Extract key financial figures, each with a citation, from stored filings."""
    settings = _settings()
    configure_logging()
    new_run_id()
    companies = _companies(settings, tickers, company_set)
    llm = _llm(settings, provider, model, effort)
    with Store(settings.data_dir) as store:
        filings = _stored_filings(store, companies, form, limit)
        report = _run_extraction(
            store, filings, llm, force=force, max_chars=settings.extraction_max_context_chars
        )
    _print_extractions(report, filings)
    if not report.ok:
        raise typer.Exit(1)


def _company_facts(store: Store, companies: list[Company]) -> dict[int, CompanyFacts | None]:
    facts: dict[int, CompanyFacts | None] = {}
    for company in companies:
        path = store.company_facts_path(company.cik)
        facts[company.cik] = CompanyFacts(json.loads(path.read_bytes())) if path.is_file() else None
        if facts[company.cik] is None:
            console.print(
                f"[yellow]No XBRL company facts for {company.ticker}; "
                "its figures are unscored (run ingest with --facts).[/yellow]"
            )
    return facts


def _contexts(store: Store, filings: list[StoredFiling], max_chars: int) -> dict[str, Context]:
    contexts = {}
    for filing in filings:
        parsed = load_or_parse(store, filing).parsed
        if parsed is not None:
            contexts[filing.accession_number] = select_context(parsed, max_chars=max_chars)
    return contexts


def _save_report(settings: Settings, name: str, run_id: str, report: BaseModel) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = settings.data_dir / "evals" / f"{name}-{stamp}-{run_id}.json"
    atomic_write(out, report.model_dump_json(indent=2).encode())
    console.print(f"Full results: {out}")


MarkdownOption = Annotated[bool, typer.Option(help="Also print the results as a Markdown table")]


@eval_app.command("context")
def eval_context_cmd(
    tickers: TickersOption = None,
    company_set: SetOption = None,
    form: FormOption = None,
    markdown: MarkdownOption = False,
) -> None:
    """Check that ground-truth figures reach the extraction context (no LLM calls)."""
    settings = _settings()
    configure_logging()
    run_id = new_run_id()
    companies = _companies(settings, tickers, company_set)
    with Store(settings.data_dir) as store:
        filings = _stored_filings(store, companies, form)
        contexts = _contexts(store, filings, settings.extraction_max_context_chars)
        facts = _company_facts(store, companies)
    result = evaluate_context(filings, contexts, facts)
    print_context_report(result, console)
    _save_report(settings, "context", run_id, result)
    if markdown:
        typer.echo(context_markdown(result))


@eval_app.command("extraction")
def eval_extraction_cmd(
    tickers: TickersOption = None,
    company_set: SetOption = None,
    form: FormOption = None,
    provider: ProviderOption = None,
    model: Annotated[str | None, typer.Option(help="Override FRA_LLM_MODEL")] = None,
    effort: Annotated[Effort | None, typer.Option(help="Override FRA_LLM_EFFORT")] = None,
    tolerance: Annotated[
        float, typer.Option(help="Relative tolerance for a correct value")
    ] = DEFAULT_REL_TOL,
    markdown: MarkdownOption = False,
) -> None:
    """Extract (reusing stored results) and score figures against XBRL company facts."""
    settings = _settings()
    configure_logging()
    run_id = new_run_id()
    companies = _companies(settings, tickers, company_set)
    llm = _llm(settings, provider, model, effort)
    max_chars = settings.extraction_max_context_chars
    with Store(settings.data_dir) as store:
        filings = _stored_filings(store, companies, form)
        report = _run_extraction(store, filings, llm, force=False, max_chars=max_chars)
        contexts = _contexts(store, filings, max_chars)
        facts = _company_facts(store, companies)
    records: dict[str, ExtractionRecord | None] = {r.accession_number: r for r in report.records}
    result = evaluate(
        filings,
        records,
        facts,
        banks={c.ticker for c in companies if c.sector == "bank"},
        extractor=report.extractor,
        rel_tol=tolerance,
        context_numbers={acc: printed_numbers(ctx) for acc, ctx in contexts.items()},
    )
    print_report(result, console)
    _save_report(settings, "extraction", run_id, result)
    if markdown:
        typer.echo(to_markdown(result))
    if result.failed:
        raise typer.Exit(1)


# --- agent ----------------------------------------------------------------------------


def _search_index_path(settings: Settings) -> Path:
    return settings.data_dir / "index" / "search.db"


@app.command("index")
def index_cmd(
    tickers: TickersOption = None,
    company_set: SetOption = None,
    force: Annotated[bool, typer.Option(help="Re-index even up-to-date filings")] = False,
) -> None:
    """Build the keyword search index over parsed filing text (no LLM calls)."""
    settings = _settings()
    configure_logging()
    new_run_id()
    companies = _companies(settings, tickers, company_set)
    indexed = skipped = 0
    failed: list[str] = []
    with Store(settings.data_dir) as store, SearchIndex(_search_index_path(settings)) as index:
        for filing in _stored_filings(store, companies, None):
            outcome = load_or_parse(store, filing)
            if outcome.parsed is None or filing.report_date is None:
                failed.append(f"{filing.ticker} {filing.accession_number}: {outcome.error}")
                continue
            if not force and index.is_current(outcome.parsed):
                skipped += 1
                continue
            index.add_filing(filing, outcome.parsed)
            indexed += 1
        total = index.count()
    console.print(
        f"Indexed {indexed} filings ({skipped} already current); "
        f"{total:,} passages in {_search_index_path(settings)}"
    )
    for line in failed:
        console.print(f"[red]failed:[/red] {line}")
    if failed:
        raise typer.Exit(1)


FinancialsOption = Annotated[
    SourceMode,
    typer.Option(
        help="Where get_financials reads figures: stored extractions, falling back to XBRL "
        "(auto), or only one of them"
    ),
]
StepsOption = Annotated[int, typer.Option(min=1, help="Most LLM calls per question")]


def _build_agent(
    settings: Settings,
    store: Store,
    companies: list[Company],
    llm: StructuredLLM,
    *,
    financials: SourceMode,
    max_steps: int,
) -> tuple[Agent, SearchIndex | None]:
    filings = _stored_filings(store, companies, None)
    facts = _company_facts(store, companies)
    extractor = extractor_id(llm)
    extractions = {
        f.accession_number: store.get_extraction(f.accession_number, extractor) for f in filings
    }
    index_path = _search_index_path(settings)
    search = SearchIndex(index_path) if index_path.is_file() else None
    if search is None or search.count() == 0:
        console.print("[yellow]No search index yet; run filings-agent index.[/yellow]")
    figures = build_financials(
        filings,
        company_facts=facts,
        extractions=extractions,
        mode=financials,
        names={c.ticker: c.name for c in companies},
    )
    return Agent(llm, Toolbox(figures, search), max_steps=max_steps), search


def _print_answer(answer: AgentAnswer, *, trace: bool) -> None:
    style = "yellow" if answer.abstained else "green"
    console.print(f"[bold {style}]{answer.text}[/bold {style}]")
    if answer.guardrail:
        console.print(f"[yellow]Guardrail: {answer.guardrail}[/yellow]")
        if answer.draft:
            console.print(f"Model's draft: {answer.draft}")
    if answer.citations:
        table = Table(title="Sources")
        for col in ("Id", "Source", "Link"):
            table.add_column(col)
        for c in answer.citations:
            table.add_row(c.id, c.text.split("] ", 1)[-1], c.url or "")
        console.print(table)
    if trace:
        for step in answer.steps:
            console.print(f"[bold]Step {step.step}[/bold]: {step.reasoning}")
            if step.tool_output:
                console.print(step.tool_output, markup=False)
    cost = f"${answer.cost_usd:.4f}" if answer.cost_usd is not None else "cost n/a"
    console.print(
        f"{answer.llm_calls} LLM calls, {answer.input_tokens:,} input / "
        f"{answer.output_tokens:,} output tokens, {cost}, {answer.latency_s:.1f}s"
        + (" (from cache)" if answer.cached else "")
        + f"; query {answer.query_id}"
    )


@app.command("ask")
def ask_cmd(
    question: Annotated[str, typer.Argument(help="An analyst question, in quotes")],
    tickers: TickersOption = None,
    company_set: SetOption = "core3",
    provider: ProviderOption = None,
    model: Annotated[str | None, typer.Option(help="Override FRA_LLM_MODEL")] = None,
    effort: Annotated[Effort | None, typer.Option(help="Override FRA_LLM_EFFORT")] = None,
    financials: FinancialsOption = "auto",
    max_steps: StepsOption = DEFAULT_MAX_STEPS,
    trace: Annotated[bool, typer.Option(help="Show each step's tool calls and results")] = False,
) -> None:
    """Answer a question from the filings, with a citation for every figure."""
    settings = _settings()
    configure_logging()
    new_run_id()
    companies = _companies(settings, tickers, company_set)
    llm = _llm(settings, provider, model, effort)
    with Store(settings.data_dir) as store:
        agent, search = _build_agent(
            settings, store, companies, llm, financials=financials, max_steps=max_steps
        )
        try:
            answer = agent.ask(question)
        except LLMConfigError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        finally:
            if search is not None:
                search.close()
    _print_answer(answer, trace=trace)
    out = settings.data_dir / "traces" / f"{answer.query_id}.json"
    atomic_write(out, answer.model_dump_json(indent=2).encode())
    console.print(f"Trace: {out}")


@eval_app.command("qa")
def eval_qa_cmd(
    questions_file: Annotated[Path, typer.Option("--questions", help="Question set (TOML)")] = Path(
        "evals/qa_core3.toml"
    ),
    ids: Annotated[
        str | None, typer.Option(help="Only these question ids (comma-separated)")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="At most this many questions")] = None,
    company_set: SetOption = "core3",
    provider: ProviderOption = None,
    model: Annotated[str | None, typer.Option(help="Override FRA_LLM_MODEL")] = None,
    effort: Annotated[Effort | None, typer.Option(help="Override FRA_LLM_EFFORT")] = None,
    financials: FinancialsOption = "auto",
    max_steps: StepsOption = DEFAULT_MAX_STEPS,
    markdown: MarkdownOption = False,
) -> None:
    """Ask the agent the Q&A set and score answers, citations and abstentions.

    Reference answers come from XBRL. Responses are cached, so a run stopped by the
    daily request budget continues where it left off the next day.
    """
    settings = _settings()
    configure_logging()
    run_id = new_run_id()
    questions = load_questions(questions_file)
    if ids:
        wanted = {i.strip() for i in ids.split(",")}
        questions = [q for q in questions if q.id in wanted]
    questions = questions[:limit] if limit else questions
    companies = _companies(settings, None, company_set)
    llm = _llm(settings, provider, model, effort)
    with Store(settings.data_dir) as store:
        agent, search = _build_agent(
            settings, store, companies, llm, financials=financials, max_steps=max_steps
        )
        truth = build_financials(
            _stored_filings(store, companies, None),
            company_facts=_company_facts(store, companies),
            mode="xbrl",
            names={c.ticker: c.name for c in companies},
        )
        try:
            summary = run_qa(questions, agent.ask, truth, agent=agent.id)
        finally:
            if search is not None:
                search.close()
    print_qa_report(summary, console)
    _save_report(settings, "qa", run_id, summary)
    if markdown:
        typer.echo(qa_markdown(summary))
    if summary.run < summary.questions:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
