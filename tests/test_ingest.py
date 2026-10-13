from __future__ import annotations

from datetime import date

import httpx

from filings_agent.config import Company
from filings_agent.edgar import EdgarClient
from filings_agent.ingest import ingest
from filings_agent.storage import Store
from tests.conftest import FakeEdgar, filing_row

JPM = Company(ticker="JPM", cik=19617, name="JPMorgan Chase & Co.", sector="bank")
PYPL = Company(ticker="PYPL", cik=1633917, name="PayPal Holdings Inc.", sector="fintech")
SINCE = date(2023, 1, 1)


def _setup(fake: FakeEdgar) -> None:
    fake.add_company(
        JPM.cik,
        "JPM",
        "JPMORGAN CHASE & CO",
        recent=[
            filing_row("0000019617-24-000300", "10-Q", "2024-05-01", "2024-03-31"),
            filing_row("0000019617-24-000250", "8-K", "2024-04-12"),
            filing_row("0000019617-24-000100", "10-K", "2024-02-16", "2023-12-31"),
        ],
    )
    fake.add_company(
        PYPL.cik,
        "PYPL",
        "PayPal Holdings, Inc.",
        recent=[filing_row("0001633917-24-000010", "10-K", "2024-02-02", "2023-12-31")],
    )


def _doc_calls(fake: FakeEdgar) -> int:
    return sum(n for url, n in fake.calls.items() if "/Archives/" in url)


def test_ingest_downloads_then_is_idempotent(
    fake_edgar: FakeEdgar, edgar: EdgarClient, store: Store
) -> None:
    _setup(fake_edgar)

    first = ingest([JPM, PYPL], edgar=edgar, store=store, forms=["10-K", "10-Q"], since=SINCE)
    assert first.ok
    assert [(r.ticker, r.discovered, r.downloaded, r.skipped) for r in first.results] == [
        ("JPM", 2, 2, 0),
        ("PYPL", 1, 1, 0),
    ]
    assert all(r.facts_saved for r in first.results)
    assert _doc_calls(fake_edgar) == 3

    second = ingest([JPM, PYPL], edgar=edgar, store=store, forms=["10-K", "10-Q"], since=SINCE)
    assert second.ok
    assert [(r.downloaded, r.skipped, r.facts_saved) for r in second.results] == [
        (0, 2, False),
        (0, 1, False),
    ]
    assert _doc_calls(fake_edgar) == 3, "re-run must not re-download documents"
    assert len(store.list_filings()) == 3, "re-run must not duplicate rows"


def test_ingest_redownloads_a_deleted_document(
    fake_edgar: FakeEdgar, edgar: EdgarClient, store: Store
) -> None:
    _setup(fake_edgar)
    ingest([JPM], edgar=edgar, store=store, forms=["10-K"], since=SINCE)
    (stored,) = store.list_filings()
    assert stored.raw_path is not None
    stored.raw_path.unlink()

    report = ingest([JPM], edgar=edgar, store=store, forms=["10-K"], since=SINCE)
    assert report.results[0].downloaded == 1
    assert stored.raw_path.is_file()


def test_one_bad_document_does_not_stop_the_run(
    fake_edgar: FakeEdgar, edgar: EdgarClient, store: Store
) -> None:
    _setup(fake_edgar)
    bad = "https://www.sec.gov/Archives/edgar/data/19617/000001961724000300/doc-0000019617-24-000300.htm"
    fake_edgar.routes[bad] = httpx.Response(404)

    report = ingest([JPM, PYPL], edgar=edgar, store=store, forms=["10-K", "10-Q"], since=SINCE)
    jpm, pypl = report.results
    assert jpm.failed == ["0000019617-24-000300"]
    assert jpm.downloaded == 1
    assert pypl.ok
    assert not report.ok

    # Once the document becomes available a re-run picks up just the missing one.
    del fake_edgar.routes[bad]
    fake_edgar.add_bytes(bad, b"<html>now available</html>")
    retry = ingest([JPM], edgar=edgar, store=store, forms=["10-K", "10-Q"], since=SINCE)
    assert (retry.results[0].downloaded, retry.results[0].skipped) == (1, 1)


def test_cik_ticker_mismatch_fails_that_company_only(
    fake_edgar: FakeEdgar, edgar: EdgarClient, store: Store
) -> None:
    _setup(fake_edgar)
    wrong = Company(ticker="BAC", cik=JPM.cik, name="Bank of America", sector="bank")
    report = ingest([wrong, PYPL], edgar=edgar, store=store, forms=["10-K"], since=SINCE)
    bad, good = report.results
    assert bad.error is not None and "not BAC" in bad.error
    assert good.ok
    assert {f.ticker for f in store.list_filings()} == {"PYPL"}


def test_unreachable_company_is_reported(edgar: EdgarClient, store: Store) -> None:
    report = ingest([JPM], edgar=edgar, store=store, forms=["10-K"], since=SINCE)
    assert report.results[0].error is not None
    assert "404" in report.results[0].error


def test_facts_can_be_skipped_or_refreshed(
    fake_edgar: FakeEdgar, edgar: EdgarClient, store: Store
) -> None:
    _setup(fake_edgar)
    facts_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json"
    ingest([JPM], edgar=edgar, store=store, forms=["10-K"], since=SINCE, fetch_facts=False)
    assert fake_edgar.calls[facts_url] == 0
    ingest([JPM], edgar=edgar, store=store, forms=["10-K"], since=SINCE)
    ingest([JPM], edgar=edgar, store=store, forms=["10-K"], since=SINCE, refresh_facts=True)
    assert fake_edgar.calls[facts_url] == 2
    assert store.company_facts_path(JPM.cik).is_file()
