from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from filings_agent.edgar import FilingMeta
from filings_agent.extract.models import ExtractionRecord, Figure, Metric
from filings_agent.storage import Store, StoredFiling


def _filing(accession: str = "0000019617-24-000100", form: str = "10-K") -> FilingMeta:
    return FilingMeta(
        cik=19617,
        accession_number=accession,
        form=form,
        filing_date=date(2024, 2, 16),
        report_date=date(2023, 12, 31),
        primary_document="jpm-20231231.htm",
    )


def test_upsert_is_idempotent(store: Store) -> None:
    store.upsert_company(cik=19617, ticker="JPM", name="JPMorgan")
    store.upsert_company(cik=19617, ticker="JPM", name="JPMorgan Chase")
    for _ in range(3):
        store.upsert_filing(_filing())
    filings = store.list_filings()
    assert len(filings) == 1
    assert filings[0].ticker == "JPM"


def test_save_document_records_hash_and_survives_metadata_refresh(store: Store) -> None:
    store.upsert_company(cik=19617, ticker="JPM", name="JPMorgan")
    filing = _filing()
    store.upsert_filing(filing)
    assert not store.has_document(filing)

    path = store.save_document(filing, b"<html>10-K</html>")
    assert path.read_bytes() == b"<html>10-K</html>"
    assert path == store.data_dir / "raw/19617/0000019617-24-000100/jpm-20231231.htm"
    assert store.has_document(filing)

    store.upsert_filing(filing)  # a re-run must not clear the stored document
    (stored,) = store.list_filings()
    assert stored.raw_path == path
    assert stored.size_bytes == len(b"<html>10-K</html>")
    assert stored.sha256 is not None and len(stored.sha256) == 64
    assert store.has_document(filing)


def test_has_document_detects_missing_or_truncated_file(store: Store) -> None:
    store.upsert_company(cik=19617, ticker="JPM", name="JPMorgan")
    filing = _filing()
    store.upsert_filing(filing)
    path = store.save_document(filing, b"full content")
    path.write_bytes(b"trunc")
    assert not store.has_document(filing)
    path.unlink()
    assert not store.has_document(filing)


def test_save_document_requires_metadata_row(store: Store) -> None:
    with pytest.raises(LookupError):
        store.save_document(_filing(), b"x")


def test_no_temp_files_left_behind(store: Store) -> None:
    store.upsert_company(cik=19617, ticker="JPM", name="JPMorgan")
    filing = _filing()
    store.upsert_filing(filing)
    path = store.save_document(filing, b"content")
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_reopen_preserves_data_and_rejects_newer_schema(tmp_path: Path) -> None:
    data = tmp_path / "data"
    with Store(data) as s:
        s.upsert_company(cik=19617, ticker="JPM", name="JPMorgan")
        s.upsert_filing(_filing())
    with Store(data) as s:
        assert len(s.list_filings()) == 1

    conn = sqlite3.connect(data / "filings.db")
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(RuntimeError, match="newer"):
        Store(data)


def stored_filing(store: Store, content: bytes) -> StoredFiling:
    """Register a downloaded 10-K for Example Bank and return its stored view."""
    store.upsert_company(cik=1, ticker="EXB", name="Example Bank Corp")
    filing = FilingMeta(
        cik=1,
        accession_number="0000000001-24-000001",
        form="10-K",
        filing_date=date(2024, 2, 15),
        report_date=date(2023, 12, 31),
        primary_document="exb-20231231.htm",
    )
    store.upsert_filing(filing)
    store.save_document(filing, content)
    (stored,) = store.list_filings(cik=1)
    return stored


def test_extraction_round_trip_and_replace(store: Store) -> None:
    stored = stored_filing(store, b"<html></html>")
    figure = Figure(
        metric=Metric.NET_INCOME,
        value=456e6,
        unit="USD",
        reported_value=456,
        scale="millions",
        period_start=date(2023, 1, 1),
        period_end=date(2023, 12, 31),
        block_id="b00012",
        quote="Net income | $ 456",
        section="financial_statements",
        item="Item 8",
        page_label="4",
        url="https://x#item8",
        citation_valid=True,
    )
    record = ExtractionRecord(
        accession_number=stored.accession_number,
        extractor="fake|v1",
        model="m",
        created_at=datetime(2024, 3, 1, tzinfo=UTC),
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.01,
        latency_s=1.0,
        figures=[figure],
    )
    store.save_extraction(record)
    assert store.get_extraction(stored.accession_number, "fake|v1") == record
    assert store.get_extraction(stored.accession_number, "other") is None

    store.save_extraction(ExtractionRecord.failed(stored.accession_number, "fake|v1", "boom"))
    again = store.get_extraction(stored.accession_number, "fake|v1")
    assert again is not None and again.error == "boom" and again.figures == []


def test_v1_database_is_migrated(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    conn = sqlite3.connect(data / "filings.db")
    conn.executescript(
        "CREATE TABLE companies (cik INTEGER PRIMARY KEY, ticker TEXT NOT NULL,"
        " name TEXT NOT NULL, updated_at TEXT NOT NULL); PRAGMA user_version = 1;"
    )
    conn.close()
    with Store(data) as s:
        tables = {r[0] for r in s._conn.execute("SELECT name FROM sqlite_master")}
        assert {"extractions", "figures"} <= tables
