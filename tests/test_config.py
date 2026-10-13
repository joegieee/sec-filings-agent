from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from filings_agent.config import Company, CompanyUniverse, Settings

REPO_COMPANIES = Path(__file__).parent.parent / "config" / "companies.toml"


def test_settings_require_contact_email_in_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRA_SEC_USER_AGENT", "my-bot")
    with pytest.raises(ValidationError, match="contact email"):
        Settings(_env_file=None)


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FRA_SEC_USER_AGENT", "bot me@example.com")
    monkeypatch.setenv("FRA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FRA_SEC_MAX_REQUESTS_PER_SECOND", "8")
    s = Settings(_env_file=None)
    assert s.data_dir == tmp_path
    assert s.sec_max_requests_per_second == 8
    assert s.forms == ("10-K", "10-Q")


def test_settings_reject_rate_above_sec_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRA_SEC_USER_AGENT", "bot me@example.com")
    monkeypatch.setenv("FRA_SEC_MAX_REQUESTS_PER_SECOND", "11")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_repo_company_universe_is_valid() -> None:
    universe = CompanyUniverse.from_toml(REPO_COMPANIES)
    assert len(universe.companies) == 20
    assert [c.ticker for c in universe.select(set_name="core3")] == ["JPM", "BAC", "PYPL"]


def _universe() -> CompanyUniverse:
    return CompanyUniverse(
        companies=[
            Company(ticker="aaa", cik=1, name="A", sector="bank"),
            Company(ticker="BBB", cik=2, name="B", sector="fintech"),
            Company(ticker="CCC", cik=3, name="C", sector="payments"),
        ],
        sets={"pair": ["CCC", "AAA"]},
    )


def test_select_combines_set_and_tickers_in_config_order() -> None:
    u = _universe()
    assert [c.ticker for c in u.select(["bbb"], "pair")] == ["AAA", "BBB", "CCC"]
    assert [c.ticker for c in u.select()] == ["AAA", "BBB", "CCC"]


def test_select_rejects_unknown() -> None:
    u = _universe()
    with pytest.raises(KeyError, match="ZZZ"):
        u.select(["ZZZ"])
    with pytest.raises(KeyError, match="nope"):
        u.select(set_name="nope")


def test_universe_rejects_duplicates_and_dangling_sets() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        CompanyUniverse(
            companies=[
                Company(ticker="A", cik=1, name="A", sector="bank"),
                Company(ticker="A", cik=2, name="A2", sector="bank"),
            ]
        )
    with pytest.raises(ValidationError, match="unknown tickers"):
        CompanyUniverse(
            companies=[Company(ticker="A", cik=1, name="A", sector="bank")],
            sets={"s": ["B"]},
        )
