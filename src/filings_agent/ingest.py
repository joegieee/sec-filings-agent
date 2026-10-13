"""Ingestion: discover 10-K/10-Q filings on EDGAR and store them locally, idempotently."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

import structlog

from filings_agent.config import Company
from filings_agent.edgar import EdgarClient
from filings_agent.http import HttpError
from filings_agent.storage import Store

log = structlog.get_logger(__name__)


class CompanyMismatchError(Exception):
    """The configured CIK belongs to a company that doesn't carry the configured ticker."""


@dataclass
class CompanyResult:
    ticker: str
    discovered: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    facts_saved: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.failed


@dataclass
class IngestReport:
    results: list[CompanyResult]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)


def ingest(
    companies: Sequence[Company],
    *,
    edgar: EdgarClient,
    store: Store,
    forms: Iterable[str],
    since: date,
    fetch_facts: bool = True,
    refresh_facts: bool = False,
) -> IngestReport:
    """Ingest filings for each company. A failure in one company never stops the others."""
    forms = tuple(forms)
    results = []
    for company in companies:
        structlog.contextvars.bind_contextvars(ticker=company.ticker)
        try:
            results.append(
                _ingest_company(
                    company,
                    edgar=edgar,
                    store=store,
                    forms=forms,
                    since=since,
                    fetch_facts=fetch_facts,
                    refresh_facts=refresh_facts,
                )
            )
        except (HttpError, CompanyMismatchError) as exc:
            log.error("ingest.company_failed", error=str(exc))
            results.append(CompanyResult(ticker=company.ticker, error=str(exc)))
        finally:
            structlog.contextvars.unbind_contextvars("ticker")
    return IngestReport(results)


def _ingest_company(
    company: Company,
    *,
    edgar: EdgarClient,
    store: Store,
    forms: tuple[str, ...],
    since: date,
    fetch_facts: bool,
    refresh_facts: bool,
) -> CompanyResult:
    result = CompanyResult(ticker=company.ticker)
    submissions = edgar.get_submissions(company.cik)
    if company.ticker not in submissions.tickers:
        raise CompanyMismatchError(
            f"CIK {company.cik} is {submissions.name!r} with tickers {submissions.tickers}, "
            f"not {company.ticker}; fix config/companies.toml"
        )
    store.upsert_company(cik=company.cik, ticker=company.ticker, name=submissions.name)

    filings = edgar.list_filings(submissions, forms=forms, since=since)
    result.discovered = len(filings)
    log.info("ingest.discovered", count=len(filings), forms=forms, since=since.isoformat())

    for filing in filings:
        store.upsert_filing(filing)
        if store.has_document(filing):
            result.skipped += 1
            continue
        try:
            content = edgar.download_document(filing)
        except HttpError as exc:
            log.error("ingest.document_failed", accession=filing.accession_number, error=str(exc))
            result.failed.append(filing.accession_number)
            continue
        store.save_document(filing, content)
        result.downloaded += 1
        log.info(
            "ingest.document_saved",
            accession=filing.accession_number,
            form=filing.form,
            filing_date=filing.filing_date.isoformat(),
            bytes=len(content),
        )

    if fetch_facts and (refresh_facts or not store.company_facts_path(company.cik).is_file()):
        store.save_company_facts(company.cik, edgar.get_company_facts(company.cik))
        result.facts_saved = True

    log.info(
        "ingest.company_done",
        downloaded=result.downloaded,
        skipped=result.skipped,
        failed=len(result.failed),
    )
    return result
