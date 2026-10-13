"""Shared fixtures. Unit tests never touch the network: HTTP goes through a fake EDGAR."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import structlog
from pydantic import BaseModel

from filings_agent.edgar import EdgarClient
from filings_agent.http import HttpClient, RateLimiter, RetryPolicy
from filings_agent.llm.base import LLMResult, Usage
from filings_agent.llm.pricing import cost_usd
from filings_agent.storage import Store

USER_AGENT = "filings-agent-tests test@example.com"


def filing_row(
    accession: str,
    form: str,
    filing_date: str,
    report_date: str = "",
    primary_document: str | None = None,
) -> dict[str, Any]:
    return {
        "accessionNumber": accession,
        "form": form,
        "filingDate": filing_date,
        "reportDate": report_date,
        "primaryDocument": primary_document or f"doc-{accession}.htm",
        "primaryDocDescription": form,
        "isXBRL": 1,
        "isInlineXBRL": 1,
    }


def to_table(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert row dicts to EDGAR's column-oriented layout."""
    keys = rows[0].keys() if rows else filing_row("0000000000-00-000000", "x", "2000-01-01")
    return {k: [r[k] for r in rows] for k in keys}


@dataclass
class FakeEdgar:
    """In-memory stand-in for data.sec.gov and www.sec.gov."""

    routes: dict[str, httpx.Response | list[httpx.Response]] = field(default_factory=dict)
    calls: Counter[str] = field(default_factory=Counter)
    user_agents: set[str] = field(default_factory=set)

    def add_json(self, url: str, payload: object) -> None:
        self.routes[url] = httpx.Response(200, content=json.dumps(payload).encode())

    def add_bytes(self, url: str, content: bytes) -> None:
        self.routes[url] = httpx.Response(200, content=content)

    def add_sequence(self, url: str, responses: list[httpx.Response]) -> None:
        self.routes[url] = list(responses)

    def add_company(
        self,
        cik: int,
        ticker: str,
        name: str,
        recent: list[dict[str, Any]],
        older_pages: dict[str, tuple[str, str, list[dict[str, Any]]]] | None = None,
    ) -> None:
        """Register submissions (+ optional older pages), every document and company facts."""
        files = []
        all_rows = list(recent)
        for page_name, (frm, to, rows) in (older_pages or {}).items():
            files.append(
                {"name": page_name, "filingCount": len(rows), "filingFrom": frm, "filingTo": to}
            )
            self.add_json(f"https://data.sec.gov/submissions/{page_name}", to_table(rows))
            all_rows.extend(rows)
        self.add_json(
            f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
            {
                "cik": str(cik),
                "name": name,
                "tickers": [ticker],
                "filings": {"recent": to_table(recent), "files": files},
            },
        )
        for row in all_rows:
            acc = row["accessionNumber"].replace("-", "")
            self.add_bytes(
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{row['primaryDocument']}",
                f"<html>{ticker} {row['form']} {row['filingDate']}</html>".encode(),
            )
        self.add_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
            {"cik": cik, "entityName": name, "facts": {}},
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls[url] += 1
        self.user_agents.add(request.headers.get("User-Agent", ""))
        route = self.routes.get(url)
        if route is None:
            return httpx.Response(404)
        if isinstance(route, list):
            return route.pop(0) if len(route) > 1 else route[0]
        return route


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    # The CLI configures structlog against the stream CliRunner swaps in; undo that.
    yield
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


@pytest.fixture
def fake_edgar() -> FakeEdgar:
    return FakeEdgar()


@pytest.fixture
def http(fake_edgar: FakeEdgar) -> Iterator[HttpClient]:
    client = HttpClient(
        user_agent=USER_AGENT,
        rate_limiter=RateLimiter(1000, sleep=lambda _: None),
        retry=RetryPolicy(max_retries=3, base_delay=0.01),
        transport=httpx.MockTransport(fake_edgar.handler),
        sleep=lambda _: None,
    )
    yield client
    client.close()


@pytest.fixture
def edgar(http: HttpClient) -> EdgarClient:
    return EdgarClient(http)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    s = Store(tmp_path / "data")
    yield s
    s.close()


class FakeLLM:
    """Returns canned outputs (one per call, last one repeated) and records the prompts."""

    def __init__(self, *outputs: BaseModel | Exception, model: str = "claude-opus-5-5") -> None:
        self.outputs = list(outputs)
        self.model = model
        self.calls: list[dict[str, str]] = []

    @property
    def id(self) -> str:
        return f"fake:{self.model}"

    def generate[T: BaseModel](self, *, system: str, prompt: str, schema: type[T]) -> LLMResult[T]:
        self.calls.append({"system": system, "prompt": prompt})
        output = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(output, Exception):
            raise output
        usage = Usage(input_tokens=1000, output_tokens=200)
        return LLMResult(
            output=schema.model_validate(output.model_dump()),
            usage=usage,
            model=self.model,
            latency_s=1.5,
            cost_usd=cost_usd(self.model, usage),
        )
