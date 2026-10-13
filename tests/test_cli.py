from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from filings_agent import cli
from filings_agent.agent.models import AgentStep, FinalAnswer, ToolCall
from filings_agent.extract.models import FinancialsExtraction, Metric
from filings_agent.http import HttpClient
from filings_agent.llm import LLMBudgetError, LLMConfigError, LLMError
from filings_agent.parse import parse_document
from tests.conftest import FakeEdgar, FakeLLM, filing_row

REPO_COMPANIES = Path(__file__).parent.parent / "config" / "companies.toml"
runner = CliRunner()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_edgar: FakeEdgar) -> Path:
    data = tmp_path / "data"
    monkeypatch.setenv("FRA_SEC_USER_AGENT", "tests test@example.com")
    monkeypatch.setenv("FRA_DATA_DIR", str(data))
    monkeypatch.setenv("FRA_COMPANIES_FILE", str(REPO_COMPANIES))
    monkeypatch.chdir(tmp_path)  # keep any developer .env out of the test

    def fake_http(**kwargs: Any) -> HttpClient:
        kwargs["transport"] = httpx.MockTransport(fake_edgar.handler)
        kwargs["sleep"] = lambda _: None
        return HttpClient(**kwargs)

    monkeypatch.setattr(cli, "HttpClient", fake_http)
    fake_edgar.add_company(
        19617,
        "JPM",
        "JPMORGAN CHASE & CO",
        recent=[filing_row("0000019617-24-000100", "10-K", "2024-02-16", "2023-12-31")],
    )
    return data


def test_ingest_then_status(env: Path) -> None:
    result = runner.invoke(cli.app, ["ingest", "--tickers", "JPM", "--since", "2023-01-01"])
    assert result.exit_code == 0, result.output
    assert "JPM" in result.output
    assert (env / "raw/19617/0000019617-24-000100/doc-0000019617-24-000100.htm").is_file()

    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "10-K" in result.output
    assert "2024-02-16" in result.output


def test_ingest_exit_code_reflects_failures(env: Path) -> None:
    # BAC is not registered in the fake EDGAR, so its submissions request 404s.
    result = runner.invoke(cli.app, ["ingest", "--tickers", "JPM,BAC", "--since", "2023-01-01"])
    assert result.exit_code == 1


def test_unknown_ticker_and_bad_config(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert runner.invoke(cli.app, ["ingest", "--tickers", "NOPE"]).exit_code == 2
    monkeypatch.setenv("FRA_SEC_USER_AGENT", "no-email")
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 2
    assert "contact email" in result.output


def test_status_when_empty(env: Path) -> None:
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "No filings stored yet" in result.output


FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.htm"
DOC_URL = (
    "https://www.sec.gov/Archives/edgar/data/19617/000001961724000100/doc-0000019617-24-000100.htm"
)


@pytest.fixture
def pipeline_env(env: Path, fake_edgar: FakeEdgar, monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """A real-looking 10-K behind the fake EDGAR, XBRL facts for it, and a fake LLM."""
    fake_edgar.add_bytes(DOC_URL, FIXTURE.read_bytes())
    fy = {"start": "2023-01-01", "end": "2023-12-31", "accn": "0000019617-24-000100"}
    fake_edgar.add_json(
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json",
        {
            "facts": {
                "us-gaap": {
                    "Revenues": {"units": {"USD": [{**fy, "val": 1_234_000_000}]}},
                    "NetIncomeLoss": {"units": {"USD": [{**fy, "val": 456_000_000}]}},
                }
            }
        },
    )
    parsed = parse_document(
        FIXTURE.read_bytes(), accession_number="x", cik=19617, form="10-K", document_url="u"
    )
    income = next(b.id for b in parsed.blocks if "Total net revenue" in b.text)
    figure = {
        "scale": "millions",
        "period_start": "2023-01-01",
        "period_end": "2023-12-31",
        "block_id": income,
    }
    answer = FinancialsExtraction.model_validate(
        {
            **{m: None for m in FinancialsExtraction.model_fields},
            "revenue": {**figure, "value": 1234, "quote": "Total net revenue | 1,234"},
            "net_income": {**figure, "value": 450, "quote": "Net income | $ 456"},
        }
    )
    llm = FakeLLM(answer)
    monkeypatch.setattr(cli, "_llm", lambda *args: llm)
    result = runner.invoke(cli.app, ["ingest", "--tickers", "JPM", "--since", "2023-01-01"])
    assert result.exit_code == 0, result.output
    return llm


def test_parse_extract_eval(pipeline_env: FakeLLM, env: Path) -> None:
    result = runner.invoke(cli.app, ["parse", "--tickers", "JPM"])
    assert result.exit_code == 0, result.output
    assert "parsed" in result.output
    assert (env / "parsed/19617/0000019617-24-000100.json").is_file()
    assert "cached" in runner.invoke(cli.app, ["parse", "--tickers", "JPM"]).output

    result = runner.invoke(cli.app, ["extract", "--tickers", "JPM"])
    assert result.exit_code == 0, result.output
    assert "2/9" in result.output  # figures found
    assert "1/2" in result.output  # citations valid (net income quote shows 456, not 450)
    assert len(pipeline_env.calls) == 1

    result = runner.invoke(cli.app, ["eval", "extraction", "--tickers", "JPM", "--markdown"])
    assert result.exit_code == 0, result.output
    assert len(pipeline_env.calls) == 1, "eval reuses the stored extraction"
    assert "| Revenue | 100.0% | 1/1 |" in result.output
    assert "| Net income | 0.0% | 0/1 |" in result.output
    (report,) = (env / "evals").glob("extraction-*.json")
    assert '"outcome": "wrong"' in report.read_text()


def test_eval_exit_code_on_failed_extraction(
    pipeline_env: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_llm", lambda *args: FakeLLM(LLMError("no credits")))
    result = runner.invoke(cli.app, ["eval", "extraction", "--tickers", "JPM"])
    assert result.exit_code == 1
    assert "without a usable extraction" in result.output


def test_missing_credentials_stop_the_run(
    pipeline_env: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_llm", lambda *args: FakeLLM(LLMConfigError("set ANTHROPIC_API_KEY")))
    result = runner.invoke(cli.app, ["extract", "--tickers", "JPM"])
    assert result.exit_code == 2
    assert "set ANTHROPIC_API_KEY" in result.output


def test_eval_context_needs_no_llm(pipeline_env: FakeLLM, env: Path) -> None:
    result = runner.invoke(cli.app, ["eval", "context", "--tickers", "JPM", "--markdown"])
    assert result.exit_code == 0, result.output
    assert "| All metrics | 100.0% | 2/2 |" in result.output
    assert not pipeline_env.calls


def test_index_ask_and_eval_qa(
    pipeline_env: FakeLLM, env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = runner.invoke(cli.app, ["index", "--tickers", "JPM"])
    assert result.exit_code == 0, result.output
    assert "Indexed 1 filings" in result.output
    assert "already current" in runner.invoke(cli.app, ["index", "--tickers", "JPM"]).output

    lookup = AgentStep(
        reasoning="look it up",
        tool_calls=[
            ToolCall(
                tool="get_financials", company="JPM", metric=Metric.REVENUE, periods=["FY2023"]
            )
        ],
        answer=None,
    )
    answer = AgentStep(
        reasoning="done",
        tool_calls=[],
        answer=FinalAnswer(
            text="JPMorgan's revenue was $1,234 million in FY2023.",
            citations=["F1"],
            abstained=False,
        ),
    )
    llm = FakeLLM(lookup, answer)
    monkeypatch.setattr(cli, "_llm", lambda *args: llm)
    result = runner.invoke(
        cli.app, ["ask", "JPM revenue in 2023?", "--tickers", "JPM", "--financials", "xbrl"]
    )
    assert result.exit_code == 0, result.output
    assert "$1,234 million" in result.output
    assert "us-gaap:Revenues" in result.output  # the citation
    assert "2 LLM calls" in result.output
    assert len(list((env / "traces").glob("*.json"))) == 1

    questions = tmp_path / "qa.toml"
    questions.write_text(
        '[[questions]]\nid = "rev"\nquestion = "JPM revenue in 2023?"\nkind = "value"\n'
        'ticker = "JPM"\nmetric = "revenue"\nperiod = "FY2023"\n'
    )
    monkeypatch.setattr(cli, "_llm", lambda *args: FakeLLM(lookup, answer))
    result = runner.invoke(cli.app, ["eval", "qa", "--questions", str(questions), "--markdown"])
    assert result.exit_code == 0, result.output
    assert "| Answer accuracy (answerable) | 100.0% | 1/1 |" in result.output
    assert "| Citation precision | 100.0% | 1/1 |" in result.output

    spent = FakeLLM(LLMBudgetError("daily gemini request budget spent"))
    monkeypatch.setattr(cli, "_llm", lambda *args: spent)
    result = runner.invoke(cli.app, ["eval", "qa", "--questions", str(questions)])
    assert result.exit_code == 1
    assert "1 questions not run" in result.output
