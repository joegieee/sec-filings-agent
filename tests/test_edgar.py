from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from filings_agent.edgar import EdgarClient, FilingMeta
from filings_agent.edgar.models import parse_filing_table
from tests.conftest import FakeEdgar, filing_row, to_table

CIK = 19617


def test_parse_filing_table_maps_columns_and_blanks() -> None:
    rows = [
        filing_row("0000019617-24-000100", "10-K", "2024-02-16", "2023-12-31", "jpm-20231231.htm"),
        filing_row("0000019617-24-000200", "8-K", "2024-03-01"),
    ]
    filings = parse_filing_table(CIK, to_table(rows))
    assert filings[0].form == "10-K"
    assert filings[0].report_date == date(2023, 12, 31)
    assert filings[0].is_inline_xbrl is True
    assert filings[1].report_date is None
    assert filings[0].document_url == (
        "https://www.sec.gov/Archives/edgar/data/19617/000001961724000100/jpm-20231231.htm"
    )


def test_parse_filing_table_rejects_ragged_and_missing_columns() -> None:
    table = to_table([filing_row("0000019617-24-000100", "10-K", "2024-02-16")])
    table["form"].append("10-Q")
    with pytest.raises(ValueError, match="ragged"):
        parse_filing_table(CIK, table)
    del table["form"]
    with pytest.raises(ValueError, match="missing columns"):
        parse_filing_table(CIK, table)


def test_accession_number_format_is_validated() -> None:
    with pytest.raises(ValidationError):
        FilingMeta(
            cik=1,
            accession_number="bad",
            form="10-K",
            filing_date=date(2024, 1, 1),
            report_date=None,
            primary_document="x.htm",
        )


def test_list_filings_filters_forms_and_dates(fake_edgar: FakeEdgar, edgar: EdgarClient) -> None:
    fake_edgar.add_company(
        CIK,
        "JPM",
        "JPMORGAN CHASE & CO",
        recent=[
            filing_row("0000019617-24-000300", "10-Q", "2024-05-01", "2024-03-31"),
            filing_row("0000019617-24-000250", "424B2", "2024-04-10"),
            filing_row("0000019617-24-000100", "10-K", "2024-02-16", "2023-12-31"),
            filing_row("0000019617-24-000090", "10-K/A", "2024-02-10", "2022-12-31"),
            filing_row("0000019617-20-000100", "10-K", "2020-02-25", "2019-12-31"),
        ],
    )
    subs = edgar.get_submissions(CIK)
    assert subs.tickers == ["JPM"]
    filings = edgar.list_filings(subs, forms=["10-K", "10-Q"], since=date(2023, 1, 1))
    assert [f.accession_number for f in filings] == [
        "0000019617-24-000300",
        "0000019617-24-000100",
    ]


def test_list_filings_pages_only_overlapping_older_files(
    fake_edgar: FakeEdgar, edgar: EdgarClient
) -> None:
    fake_edgar.add_company(
        CIK,
        "JPM",
        "JPMORGAN CHASE & CO",
        recent=[filing_row("0000019617-24-000300", "10-Q", "2024-05-01", "2024-03-31")],
        older_pages={
            "CIK0000019617-submissions-001.json": (
                "2023-01-01",
                "2024-04-30",
                [
                    filing_row("0000019617-23-000500", "10-Q", "2023-11-01", "2023-09-30"),
                    # Same accession also present in recent: must be de-duplicated.
                    filing_row("0000019617-24-000300", "10-Q", "2024-05-01", "2024-03-31"),
                ],
            ),
            "CIK0000019617-submissions-002.json": (
                "2015-01-01",
                "2018-12-31",
                [filing_row("0000019617-16-000100", "10-K", "2016-02-23", "2015-12-31")],
            ),
        },
    )
    subs = edgar.get_submissions(CIK)
    filings = edgar.list_filings(subs, forms=["10-K", "10-Q"], since=date(2021, 10, 1))
    assert [f.accession_number for f in filings] == [
        "0000019617-24-000300",
        "0000019617-23-000500",
    ]
    assert fake_edgar.calls["https://data.sec.gov/submissions/CIK0000019617-submissions-001.json"]
    assert not fake_edgar.calls[
        "https://data.sec.gov/submissions/CIK0000019617-submissions-002.json"
    ]
