"""Thin client over the official SEC EDGAR endpoints."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from filings_agent.edgar.models import CompanySubmissions, FilingMeta, parse_filing_table
from filings_agent.http import HttpClient

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


class EdgarClient:
    def __init__(self, http: HttpClient) -> None:
        self._http = http

    def get_submissions(self, cik: int) -> CompanySubmissions:
        return CompanySubmissions.from_json(self._http.get(SUBMISSIONS_URL.format(cik=cik)).json())

    def list_filings(
        self,
        submissions: CompanySubmissions,
        *,
        forms: Iterable[str],
        since: date,
    ) -> list[FilingMeta]:
        """Filings of the given forms filed on or after ``since``, newest first.

        ``recent`` only holds the latest ~1000 filings. Large banks file thousands
        of prospectus supplements a year, so older pages are fetched whenever
        they overlap the requested window.
        """
        wanted = set(forms)
        candidates = list(submissions.recent)
        for page in submissions.older_pages:
            if page.filing_to >= since:
                payload = self._http.get(SUBMISSIONS_PAGE_URL.format(name=page.name)).json()
                candidates.extend(parse_filing_table(submissions.cik, payload))

        by_accession: dict[str, FilingMeta] = {}
        for f in candidates:
            if f.form in wanted and f.filing_date >= since:
                by_accession.setdefault(f.accession_number, f)
        return sorted(by_accession.values(), key=lambda f: f.filing_date, reverse=True)

    def download_document(self, filing: FilingMeta) -> bytes:
        return self._http.get(filing.document_url).content

    def get_company_facts(self, cik: int) -> bytes:
        """Raw XBRL company-facts JSON; the ground truth for extraction evals."""
        return self._http.get(COMPANY_FACTS_URL.format(cik=cik)).content
