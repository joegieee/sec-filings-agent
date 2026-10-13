from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel

from filings_agent.llm import CachedLLM, LLMBudgetError, LLMConfigError, LLMError
from filings_agent.llm.budget import BudgetedLLM, RequestLedger
from tests.conftest import FakeLLM


class Answer(BaseModel):
    value: float


def _budgeted(tmp_path: Path, inner: FakeLLM, limit: int, day: list[date]) -> BudgetedLLM:
    ledger = RequestLedger(tmp_path / "llm_requests.json")
    return BudgetedLLM(inner, provider="gemini", limit=limit, ledger=ledger, today=lambda: day[0])


def test_budget_stops_at_the_daily_limit_and_resets_next_day(tmp_path: Path) -> None:
    day = [date(2026, 10, 23)]
    inner = FakeLLM(Answer(value=1))
    llm = _budgeted(tmp_path, inner, 2, day)
    llm.generate(system="s", prompt="a", schema=Answer)
    llm.generate(system="s", prompt="b", schema=Answer)
    assert llm.remaining() == 0
    with pytest.raises(LLMBudgetError, match=r"2/2 on 2026-10-23"):
        llm.generate(system="s", prompt="c", schema=Answer)
    assert len(inner.calls) == 2
    day[0] = date(2026, 10, 24)
    assert llm.remaining() == 2
    llm.generate(system="s", prompt="c", schema=Answer)


def test_budget_is_shared_through_the_ledger_and_counts_failures(tmp_path: Path) -> None:
    day = [date(2026, 10, 23)]
    first = _budgeted(tmp_path, FakeLLM(LLMError("boom")), 2, day)
    with pytest.raises(LLMError):
        first.generate(system="s", prompt="p", schema=Answer)
    second = _budgeted(tmp_path, FakeLLM(Answer(value=1)), 2, day)
    assert second.remaining() == 1, "a failed request still used quota, in another process too"


def test_cache_hits_do_not_spend_budget(tmp_path: Path) -> None:
    day = [date(2026, 10, 23)]
    llm = CachedLLM(_budgeted(tmp_path, FakeLLM(Answer(value=1)), 1, day), tmp_path / "cache")
    llm.generate(system="s", prompt="p", schema=Answer)
    assert llm.generate(system="s", prompt="p", schema=Answer).cached
    with pytest.raises(LLMBudgetError):
        llm.generate(system="s", prompt="other", schema=Answer)


def test_budget_error_stops_runs_like_a_config_error() -> None:
    assert issubclass(LLMBudgetError, LLMConfigError)


def test_ledger_tolerates_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "llm_requests.json"
    path.write_text("{not json")
    ledger = RequestLedger(path)
    assert ledger.used("gemini", date(2026, 1, 1)) == 0
    assert ledger.add("gemini", date(2026, 1, 1)) == 1
