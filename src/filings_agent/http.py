"""SEC HTTP requests with rate limiting, timeouts and retry backoff.

LLM adapters use their SDKs or their own HTTP clients.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
import structlog

log = structlog.get_logger(__name__)

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class HttpError(Exception):
    """Raised when a request fails permanently or exhausts its retries."""

    def __init__(self, url: str, message: str, status_code: int | None = None) -> None:
        super().__init__(f"{message} ({url})")
        self.url = url
        self.status_code = status_code


class RateLimiter:
    """Thread-safe minimum-interval limiter (at most ``rate`` acquisitions per second)."""

    def __init__(
        self,
        rate_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self._interval = 1.0 / rate_per_second
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        # Reserve a slot under the lock, then sleep outside it so other threads
        # can reserve the following slots concurrently.
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        wait = slot - now
        if wait > 0:
            self._sleep(wait)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    retry_statuses: frozenset[int] = field(default=RETRYABLE_STATUSES)

    def delay(self, attempt: int, retry_after: str | None = None) -> float:
        """Backoff before retry number ``attempt + 1``; honours ``Retry-After`` when given."""
        if retry_after:
            parsed = _parse_retry_after(retry_after)
            if parsed is not None:
                return min(parsed, self.max_delay)
        exp = self.base_delay * (2**attempt)
        return float(min(self.max_delay, exp * (0.5 + random.random() / 2)))


def _parse_retry_after(value: str) -> float | None:
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


class HttpClient:
    def __init__(
        self,
        *,
        user_agent: str,
        rate_limiter: RateLimiter,
        timeout: float = 30.0,
        retry: RetryPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._limiter = rate_limiter
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get(self, url: str) -> httpx.Response:
        last_problem = "no attempts made"
        last_status: int | None = None
        for attempt in range(self._retry.max_retries + 1):
            self._limiter.acquire()
            started = time.perf_counter()
            retry_after: str | None = None
            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "http.transport_error",
                    url=url,
                    attempt=attempt,
                    error=last_problem,
                    elapsed_ms=_ms_since(started),
                )
            else:
                status = response.status_code
                log.debug(
                    "http.response",
                    url=url,
                    status=status,
                    attempt=attempt,
                    bytes=len(response.content),
                    elapsed_ms=_ms_since(started),
                )
                if status < 400:
                    return response
                if status not in self._retry.retry_statuses:
                    hint = " (check FRA_SEC_USER_AGENT)" if status == 403 else ""
                    raise HttpError(url, f"HTTP {status}{hint}", status_code=status)
                last_problem, last_status = f"HTTP {status}", status
                retry_after = response.headers.get("Retry-After")
                log.warning("http.retryable_status", url=url, status=status, attempt=attempt)

            if attempt < self._retry.max_retries:
                self._sleep(self._retry.delay(attempt, retry_after))

        raise HttpError(
            url,
            f"gave up after {self._retry.max_retries + 1} attempts: {last_problem}",
            status_code=last_status,
        )


def _ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
