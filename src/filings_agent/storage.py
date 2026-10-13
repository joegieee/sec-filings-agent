"""Local storage: SQLite for metadata, content-addressed raw files on disk.

Layout under ``data_dir``::

    filings.db                                  metadata, extractions and extracted figures
    raw/<cik>/<accession>/<primary_document>    filing documents, as served by EDGAR
    raw/<cik>/companyfacts.json                 XBRL company facts (eval ground truth)
    parsed/<cik>/<accession>.json               section-tagged text blocks (see ``parse``)

All writes are idempotent: re-ingesting the same filing updates its metadata in
place and never duplicates rows or files.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from filings_agent.edgar.models import FilingMeta
from filings_agent.extract.models import ExtractionRecord, Figure
from filings_agent.fsutil import atomic_write
from filings_agent.parse.models import ParsedFiling

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    cik         INTEGER PRIMARY KEY,
    ticker      TEXT NOT NULL,
    name        TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS filings (
    accession_number         TEXT PRIMARY KEY,
    cik                      INTEGER NOT NULL REFERENCES companies(cik),
    form                     TEXT NOT NULL,
    filing_date              TEXT NOT NULL,
    report_date              TEXT,
    primary_document         TEXT NOT NULL,
    primary_doc_description  TEXT,
    is_xbrl                  INTEGER NOT NULL,
    is_inline_xbrl           INTEGER NOT NULL,
    url                      TEXT NOT NULL,
    raw_path                 TEXT,
    sha256                   TEXT,
    size_bytes               INTEGER,
    fetched_at               TEXT
);

CREATE INDEX IF NOT EXISTS filings_by_company ON filings (cik, form, filing_date);

-- v2: one row per (filing, extractor); a failed attempt keeps its error so a re-run retries it.
CREATE TABLE IF NOT EXISTS extractions (
    accession_number  TEXT NOT NULL REFERENCES filings(accession_number),
    extractor         TEXT NOT NULL,
    model             TEXT,
    created_at        TEXT NOT NULL,
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL,
    latency_s         REAL,
    cached            INTEGER NOT NULL DEFAULT 0,
    error             TEXT,
    PRIMARY KEY (accession_number, extractor)
);

-- v2: extracted figures, one per metric found; a metric the filing doesn't report has no row.
CREATE TABLE IF NOT EXISTS figures (
    accession_number  TEXT NOT NULL,
    extractor         TEXT NOT NULL,
    metric            TEXT NOT NULL,
    value             REAL NOT NULL,
    unit              TEXT NOT NULL,
    reported_value    REAL NOT NULL,
    scale             TEXT NOT NULL,
    period_start      TEXT,
    period_end        TEXT NOT NULL,
    block_id          TEXT NOT NULL,
    quote             TEXT NOT NULL,
    section           TEXT,
    item              TEXT,
    page_label        TEXT,
    url               TEXT,
    citation_valid    INTEGER NOT NULL,
    PRIMARY KEY (accession_number, extractor, metric),
    FOREIGN KEY (accession_number, extractor)
        REFERENCES extractions (accession_number, extractor) ON DELETE CASCADE
);
"""


@dataclass(frozen=True)
class StoredFiling:
    accession_number: str
    cik: int
    ticker: str
    company_name: str
    form: str
    filing_date: date
    report_date: date | None
    url: str
    raw_path: Path | None
    sha256: str | None
    size_bytes: int | None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.parsed_dir = data_dir / "parsed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(data_dir / "filings.db")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        try:
            self._migrate()
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _migrate(self) -> None:
        (version,) = self._conn.execute("PRAGMA user_version").fetchone()
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema v{version} is newer than this code (v{SCHEMA_VERSION})"
            )
        with self._tx() as cur:
            cur.executescript(_SCHEMA)
            cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # --- writes -----------------------------------------------------------

    def upsert_company(self, *, cik: int, ticker: str, name: str) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO companies (cik, ticker, name, updated_at) VALUES (?, ?, ?, ?)
                ON CONFLICT (cik) DO UPDATE SET
                    ticker = excluded.ticker, name = excluded.name, updated_at = excluded.updated_at
                """,
                (cik, ticker, name, _now()),
            )

    def upsert_filing(self, filing: FilingMeta) -> None:
        """Insert or refresh filing metadata. Never touches the stored document columns."""
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO filings (
                    accession_number, cik, form, filing_date, report_date, primary_document,
                    primary_doc_description, is_xbrl, is_inline_xbrl, url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (accession_number) DO UPDATE SET
                    form = excluded.form,
                    filing_date = excluded.filing_date,
                    report_date = excluded.report_date,
                    primary_document = excluded.primary_document,
                    primary_doc_description = excluded.primary_doc_description,
                    is_xbrl = excluded.is_xbrl,
                    is_inline_xbrl = excluded.is_inline_xbrl,
                    url = excluded.url
                """,
                (
                    filing.accession_number,
                    filing.cik,
                    filing.form,
                    filing.filing_date.isoformat(),
                    filing.report_date.isoformat() if filing.report_date else None,
                    filing.primary_document,
                    filing.primary_doc_description,
                    int(filing.is_xbrl),
                    int(filing.is_inline_xbrl),
                    filing.document_url,
                ),
            )

    def document_path(self, filing: FilingMeta) -> Path:
        return self.raw_dir / str(filing.cik) / filing.accession_number / filing.primary_document

    def has_document(self, filing: FilingMeta) -> bool:
        """True if the document is stored and the file on disk still matches its recorded size."""
        row = self._conn.execute(
            "SELECT raw_path, size_bytes FROM filings WHERE accession_number = ?",
            (filing.accession_number,),
        ).fetchone()
        if row is None or row["raw_path"] is None:
            return False
        path = self.data_dir / str(row["raw_path"])
        expected_size = int(row["size_bytes"])
        return path.is_file() and path.stat().st_size == expected_size

    def save_document(self, filing: FilingMeta, content: bytes) -> Path:
        path = self.document_path(filing)
        atomic_write(path, content)
        with self._tx() as cur:
            cur.execute(
                """
                UPDATE filings SET raw_path = ?, sha256 = ?, size_bytes = ?, fetched_at = ?
                WHERE accession_number = ?
                """,
                (
                    path.relative_to(self.data_dir).as_posix(),
                    hashlib.sha256(content).hexdigest(),
                    len(content),
                    _now(),
                    filing.accession_number,
                ),
            )
            if cur.rowcount != 1:
                raise LookupError(f"no metadata row for {filing.accession_number}; upsert first")
        return path

    def company_facts_path(self, cik: int) -> Path:
        return self.raw_dir / str(cik) / "companyfacts.json"

    def save_company_facts(self, cik: int, content: bytes) -> Path:
        path = self.company_facts_path(cik)
        atomic_write(path, content)
        return path

    def parsed_path(self, filing: StoredFiling) -> Path:
        return self.parsed_dir / str(filing.cik) / f"{filing.accession_number}.json"

    def save_parsed(self, filing: StoredFiling, parsed: ParsedFiling) -> Path:
        path = self.parsed_path(filing)
        atomic_write(path, parsed.model_dump_json().encode())
        return path

    def save_extraction(self, record: ExtractionRecord) -> None:
        """Replace any earlier result for the same filing and extractor."""
        with self._tx() as cur:
            cur.execute(
                "DELETE FROM figures WHERE accession_number = ? AND extractor = ?",
                (record.accession_number, record.extractor),
            )
            cur.execute(
                """
                INSERT OR REPLACE INTO extractions (
                    accession_number, extractor, model, created_at, input_tokens,
                    output_tokens, cost_usd, latency_s, cached, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.accession_number,
                    record.extractor,
                    record.model,
                    record.created_at.isoformat(timespec="seconds"),
                    record.input_tokens,
                    record.output_tokens,
                    record.cost_usd,
                    record.latency_s,
                    int(record.cached),
                    record.error,
                ),
            )
            cur.executemany(
                """
                INSERT INTO figures (
                    accession_number, extractor, metric, value, unit, reported_value, scale,
                    period_start, period_end, block_id, quote, section, item, page_label, url,
                    citation_valid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.accession_number,
                        record.extractor,
                        f.metric,
                        f.value,
                        f.unit,
                        f.reported_value,
                        f.scale,
                        f.period_start.isoformat() if f.period_start else None,
                        f.period_end.isoformat(),
                        f.block_id,
                        f.quote,
                        f.section,
                        f.item,
                        f.page_label,
                        f.url,
                        int(f.citation_valid),
                    )
                    for f in record.figures
                ],
            )

    # --- reads ------------------------------------------------------------

    def list_filings(
        self, *, cik: int | None = None, ciks: Iterable[int] | None = None
    ) -> list[StoredFiling]:
        query = """
            SELECT f.*, c.ticker, c.name AS company_name FROM filings f JOIN companies c USING (cik)
            {where} ORDER BY c.ticker, f.filing_date DESC
        """
        wanted = [cik] if cik is not None else list(ciks) if ciks is not None else None
        params: tuple[object, ...] = ()
        where = ""
        if wanted is not None:
            where = f"WHERE f.cik IN ({', '.join('?' * len(wanted))})"
            params = tuple(wanted)
        rows = self._conn.execute(query.format(where=where), params).fetchall()
        return [
            StoredFiling(
                accession_number=r["accession_number"],
                cik=r["cik"],
                ticker=r["ticker"],
                company_name=r["company_name"],
                form=r["form"],
                filing_date=date.fromisoformat(r["filing_date"]),
                report_date=date.fromisoformat(r["report_date"]) if r["report_date"] else None,
                url=r["url"],
                raw_path=self.data_dir / r["raw_path"] if r["raw_path"] else None,
                sha256=r["sha256"],
                size_bytes=r["size_bytes"],
            )
            for r in rows
        ]

    def load_parsed(self, filing: StoredFiling) -> ParsedFiling | None:
        path = self.parsed_path(filing)
        if not path.is_file():
            return None
        return ParsedFiling.model_validate_json(path.read_bytes())

    def get_extraction(self, accession_number: str, extractor: str) -> ExtractionRecord | None:
        row = self._conn.execute(
            "SELECT * FROM extractions WHERE accession_number = ? AND extractor = ?",
            (accession_number, extractor),
        ).fetchone()
        if row is None:
            return None
        figures = self._conn.execute(
            "SELECT * FROM figures WHERE accession_number = ? AND extractor = ? ORDER BY metric",
            (accession_number, extractor),
        ).fetchall()
        return ExtractionRecord(
            accession_number=row["accession_number"],
            extractor=row["extractor"],
            model=row["model"],
            created_at=datetime.fromisoformat(row["created_at"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cost_usd=row["cost_usd"],
            latency_s=row["latency_s"],
            cached=bool(row["cached"]),
            error=row["error"],
            figures=[
                Figure(
                    metric=f["metric"],
                    value=f["value"],
                    unit=f["unit"],
                    reported_value=f["reported_value"],
                    scale=f["scale"],
                    period_start=f["period_start"],
                    period_end=f["period_end"],
                    block_id=f["block_id"],
                    quote=f["quote"],
                    section=f["section"],
                    item=f["item"],
                    page_label=f["page_label"],
                    url=f["url"],
                    citation_valid=bool(f["citation_valid"]),
                )
                for f in figures
            ],
        )
