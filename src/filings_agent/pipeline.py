"""Parse and extract stored filings, idempotently. One failing filing never stops a run."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import structlog
from pydantic import ValidationError

from filings_agent.extract.context import DEFAULT_MAX_CHARS
from filings_agent.extract.extractor import FilingInfo, extract_filing, extractor_id
from filings_agent.extract.models import ExtractionRecord
from filings_agent.llm.base import LLMConfigError, LLMError, StructuredLLM
from filings_agent.parse import PARSER_VERSION, ParsedFiling, parse_document
from filings_agent.storage import Store, StoredFiling

log = structlog.get_logger(__name__)


@dataclass
class ParseOutcome:
    filing: StoredFiling
    parsed: ParsedFiling | None = None
    skipped: bool = False
    error: str | None = None


def load_or_parse(store: Store, filing: StoredFiling, *, force: bool = False) -> ParseOutcome:
    """Parse a stored document, reusing an up-to-date parse from disk when there is one."""
    if filing.raw_path is None or not filing.raw_path.is_file():
        return ParseOutcome(filing, error="document not downloaded; run ingest")
    if not force:
        try:
            existing = store.load_parsed(filing)
        except ValidationError:
            existing = None  # written by an incompatible version: redo it
        if (
            existing is not None
            and existing.parser_version == PARSER_VERSION
            and existing.source_sha256 == filing.sha256
        ):
            return ParseOutcome(filing, existing, skipped=True)
    content = filing.raw_path.read_bytes()
    try:
        parsed = parse_document(
            content,
            accession_number=filing.accession_number,
            cik=filing.cik,
            form=filing.form,
            document_url=filing.url,
        )
    except Exception as exc:  # a malformed document must not abort the run
        log.error("parse.failed", accession=filing.accession_number, error=repr(exc))
        return ParseOutcome(filing, error=f"{type(exc).__name__}: {exc}")
    store.save_parsed(filing, parsed)
    log.info(
        "parse.saved",
        accession=filing.accession_number,
        blocks=len(parsed.blocks),
        sections=[s.kind.value for s in parsed.sections],
    )
    return ParseOutcome(filing, parsed)


def parse_filings(
    store: Store, filings: Sequence[StoredFiling], *, force: bool = False
) -> list[ParseOutcome]:
    return [load_or_parse(store, f, force=force) for f in filings]


@dataclass
class ExtractReport:
    extractor: str
    records: list[ExtractionRecord] = field(default_factory=list)
    reused: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (accession, error)

    @property
    def ok(self) -> bool:
        return not self.failed


def extract_filings(
    store: Store,
    filings: Sequence[StoredFiling],
    llm: StructuredLLM,
    *,
    force: bool = False,
    max_context_chars: int = DEFAULT_MAX_CHARS,
) -> ExtractReport:
    """Extract figures for each filing; reuse stored results for this extractor unless forced.

    A per-filing failure is recorded and the run continues; ``LLMConfigError``
    (e.g. missing credentials) propagates, since every call would fail.
    """
    report = ExtractReport(extractor=extractor_id(llm))
    for filing in filings:
        structlog.contextvars.bind_contextvars(accession=filing.accession_number)
        try:
            existing = store.get_extraction(filing.accession_number, report.extractor)
            if existing is not None and existing.error is None and not force:
                report.records.append(existing)
                report.reused += 1
                continue
            record = _extract_one(store, filing, llm, report.extractor, max_context_chars)
            store.save_extraction(record)
            if record.error is not None:
                report.failed.append((filing.accession_number, record.error))
            else:
                report.records.append(record)
        finally:
            structlog.contextvars.unbind_contextvars("accession")
    return report


def _extract_one(
    store: Store,
    filing: StoredFiling,
    llm: StructuredLLM,
    extractor: str,
    max_context_chars: int,
) -> ExtractionRecord:
    def failure(error: str) -> ExtractionRecord:
        log.error("extract.failed", error=error)
        return ExtractionRecord.failed(filing.accession_number, extractor, error)

    if filing.report_date is None:
        return failure("filing has no report date")
    outcome = load_or_parse(store, filing)
    if outcome.parsed is None:
        return failure(outcome.error or "parse failed")
    info = FilingInfo(
        accession_number=filing.accession_number,
        company_name=filing.company_name,
        ticker=filing.ticker,
        form=filing.form,
        report_date=filing.report_date,
    )
    try:
        record = extract_filing(info, outcome.parsed, llm, max_context_chars=max_context_chars)
    except LLMConfigError:
        raise  # every filing would fail the same way: stop the run
    except LLMError as exc:
        return failure(str(exc))
    log.info(
        "extract.done",
        model=record.model,
        figures=len(record.figures),
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cost_usd=record.cost_usd,
        latency_s=record.latency_s,
        cached=record.cached,
    )
    return record
