"""Typed views over EDGAR's submissions JSON."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


class FilingMeta(BaseModel):
    """One filing as listed in EDGAR submissions."""

    model_config = ConfigDict(frozen=True)

    cik: int
    accession_number: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    form: str
    filing_date: date
    report_date: date | None
    primary_document: str
    primary_doc_description: str | None = None
    is_xbrl: bool = False
    is_inline_xbrl: bool = False

    @field_validator("report_date", "primary_doc_description", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def accession_nodash(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def document_url(self) -> str:
        return f"{ARCHIVES_BASE}/{self.cik}/{self.accession_nodash}/{self.primary_document}"


class SubmissionsPage(BaseModel):
    """Pointer to an older page of filings (``filings.files`` in the submissions JSON)."""

    name: str
    filing_count: int = Field(alias="filingCount")
    filing_from: date = Field(alias="filingFrom")
    filing_to: date = Field(alias="filingTo")


class CompanySubmissions(BaseModel):
    cik: int
    name: str
    tickers: list[str]
    recent: list[FilingMeta]
    older_pages: list[SubmissionsPage]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> CompanySubmissions:
        cik = int(payload["cik"])
        filings = payload["filings"]
        return cls(
            cik=cik,
            name=payload["name"],
            tickers=[t.upper() for t in payload.get("tickers", [])],
            recent=parse_filing_table(cik, filings["recent"]),
            older_pages=[SubmissionsPage.model_validate(p) for p in filings.get("files", [])],
        )


# EDGAR returns filings as parallel arrays (column-oriented); map them to our fields.
_COLUMNS = {
    "accessionNumber": "accession_number",
    "form": "form",
    "filingDate": "filing_date",
    "reportDate": "report_date",
    "primaryDocument": "primary_document",
    "primaryDocDescription": "primary_doc_description",
    "isXBRL": "is_xbrl",
    "isInlineXBRL": "is_inline_xbrl",
}
_REQUIRED = ("accessionNumber", "form", "filingDate", "primaryDocument")


def parse_filing_table(cik: int, table: Mapping[str, Sequence[Any]]) -> list[FilingMeta]:
    missing = [c for c in _REQUIRED if c not in table]
    if missing:
        raise ValueError(f"submissions table missing columns: {missing}")
    n = len(table["accessionNumber"])
    lengths = {c: len(table[c]) for c in _COLUMNS if c in table}
    if any(length != n for length in lengths.values()):
        raise ValueError(f"submissions table has ragged columns: {lengths}")
    rows: list[FilingMeta] = []
    for i in range(n):
        fields: dict[str, Any] = {"cik": cik}
        for src, dst in _COLUMNS.items():
            if src in table:
                fields[dst] = table[src][i]
        fields.setdefault("report_date", None)
        rows.append(FilingMeta.model_validate(fields))
    return rows
