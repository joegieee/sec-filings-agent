from __future__ import annotations

import httpx
import pytest

from filings_agent.http import HttpClient, HttpError, RateLimiter, RetryPolicy
from tests.conftest import USER_AGENT, FakeEdgar

URL = "https://data.sec.gov/x.json"


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_requests() -> None:
    clock = FakeClock()
    limiter = RateLimiter(5, clock=clock, sleep=clock.sleep)
    for _ in range(4):
        limiter.acquire()
    assert clock.sleeps == pytest.approx([0.2, 0.2, 0.2])
    assert clock.now == pytest.approx(100.6)


def test_rate_limiter_does_not_sleep_after_idle_gap() -> None:
    clock = FakeClock()
    limiter = RateLimiter(5, clock=clock, sleep=clock.sleep)
    limiter.acquire()
    clock.now += 10
    limiter.acquire()
    assert clock.sleeps == []


def _client(fake: FakeEdgar, sleeps: list[float], max_retries: int = 3) -> HttpClient:
    return HttpClient(
        user_agent=USER_AGENT,
        rate_limiter=RateLimiter(1000, sleep=lambda _: None),
        retry=RetryPolicy(max_retries=max_retries, base_delay=1.0, max_delay=8.0),
        transport=httpx.MockTransport(fake.handler),
        sleep=sleeps.append,
    )


def test_sends_user_agent(fake_edgar: FakeEdgar) -> None:
    fake_edgar.add_bytes(URL, b"{}")
    _client(fake_edgar, []).get(URL)
    assert fake_edgar.user_agents == {USER_AGENT}


def test_retries_retryable_status_then_succeeds(fake_edgar: FakeEdgar) -> None:
    fake_edgar.add_sequence(
        URL, [httpx.Response(503), httpx.Response(429), httpx.Response(200, content=b"ok")]
    )
    sleeps: list[float] = []
    assert _client(fake_edgar, sleeps).get(URL).content == b"ok"
    assert fake_edgar.calls[URL] == 3
    assert len(sleeps) == 2
    # Exponential backoff with jitter: attempt 0 in [0.5, 1], attempt 1 in [1, 2].
    assert 0.5 <= sleeps[0] <= 1.0
    assert 1.0 <= sleeps[1] <= 2.0


def test_honours_retry_after_header(fake_edgar: FakeEdgar) -> None:
    fake_edgar.add_sequence(
        URL,
        [httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(200, content=b"ok")],
    )
    sleeps: list[float] = []
    _client(fake_edgar, sleeps).get(URL)
    assert sleeps == [3.0]


def test_gives_up_after_max_retries(fake_edgar: FakeEdgar) -> None:
    fake_edgar.add_sequence(URL, [httpx.Response(500)])
    sleeps: list[float] = []
    with pytest.raises(HttpError, match="gave up after 3 attempts") as excinfo:
        _client(fake_edgar, sleeps, max_retries=2).get(URL)
    assert excinfo.value.status_code == 500
    assert fake_edgar.calls[URL] == 3
    assert len(sleeps) == 2


def test_does_not_retry_client_errors(fake_edgar: FakeEdgar) -> None:
    sleeps: list[float] = []
    with pytest.raises(HttpError, match="HTTP 404") as excinfo:
        _client(fake_edgar, sleeps).get("https://data.sec.gov/missing.json")
    assert excinfo.value.status_code == 404
    assert sleeps == []


def test_403_hints_at_user_agent(fake_edgar: FakeEdgar) -> None:
    fake_edgar.add_sequence(URL, [httpx.Response(403)])
    with pytest.raises(HttpError, match="FRA_SEC_USER_AGENT"):
        _client(fake_edgar, []).get(URL)


def test_retries_transport_errors() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, content=b"ok")

    client = HttpClient(
        user_agent=USER_AGENT,
        rate_limiter=RateLimiter(1000, sleep=lambda _: None),
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    assert client.get(URL).content == b"ok"
    assert attempts == 2


def test_retry_policy_delay_is_capped() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=5.0)
    assert policy.delay(10) <= 5.0
    assert policy.delay(0, retry_after="120") == 5.0
    assert policy.delay(0, retry_after="not-a-date") <= 1.0
