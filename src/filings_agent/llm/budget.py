"""A persistent daily cap on generate calls, inside the response cache.

The day resets at midnight Pacific. Each generate counts once before sending;
SDK retries and output repairs can send additional requests. This is a local
budget, not an exact count of requests billed or counted by a provider.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel

from filings_agent.fsutil import atomic_write
from filings_agent.llm.base import LLMConfigError, LLMResult, StructuredLLM

try:
    QUOTA_TZ: ZoneInfo | None = ZoneInfo("America/Los_Angeles")
except ZoneInfoNotFoundError:  # no tz database (e.g. Windows without tzdata)
    QUOTA_TZ = None


class LLMBudgetError(LLMConfigError):
    """Today's request budget is spent; every further call would be refused."""


def quota_day() -> date:
    """The current quota day: Google resets free-tier quotas at midnight Pacific."""
    return datetime.now(QUOTA_TZ).date()


class RequestLedger:
    """Requests sent per (provider, day), persisted as JSON so separate runs add up."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, dict[str, int]]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_bytes())
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def used(self, provider: str, day: date) -> int:
        return int(self._load().get(provider, {}).get(day.isoformat(), 0))

    def add(self, provider: str, day: date) -> int:
        data = self._load()
        days = data.setdefault(provider, {})
        days[day.isoformat()] = int(days.get(day.isoformat(), 0)) + 1
        atomic_write(self.path, json.dumps(data, indent=2, sort_keys=True).encode())
        return days[day.isoformat()]


class BudgetedLLM:
    """Refuses calls once ``limit`` requests have been sent today for ``provider``."""

    def __init__(
        self,
        inner: StructuredLLM,
        *,
        provider: str,
        limit: int,
        ledger: RequestLedger,
        today: Callable[[], date] = quota_day,
    ) -> None:
        self._inner = inner
        self._provider = provider
        self._limit = limit
        self._ledger = ledger
        self._today = today

    @property
    def id(self) -> str:
        return self._inner.id

    def remaining(self) -> int:
        return max(0, self._limit - self._ledger.used(self._provider, self._today()))

    def generate[T: BaseModel](self, *, system: str, prompt: str, schema: type[T]) -> LLMResult[T]:
        day = self._today()
        used = self._ledger.used(self._provider, day)
        if used >= self._limit:
            raise LLMBudgetError(
                f"daily {self._provider} request budget spent ({used}/{self._limit} on {day}); "
                "cached results are kept, so re-run tomorrow to continue "
                "(FRA_LLM_DAILY_LIMIT sets the cap)"
            )
        # Count before sending: a request that fails still uses quota.
        self._ledger.add(self._provider, day)
        return self._inner.generate(system=system, prompt=prompt, schema=schema)
