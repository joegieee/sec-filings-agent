from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from filings_agent.agent.agent import Agent
from filings_agent.agent.calculator import CalcError, evaluate
from filings_agent.agent.financials import build_financials, normalize_period, period_label
from filings_agent.agent.grounding import mentions, render, ungrounded
from filings_agent.agent.models import AgentStep, FinalAnswer, ToolCall
from filings_agent.agent.retrieval import SearchIndex, chunk, fts_queries
from filings_agent.agent.tools import Sources, Toolbox, format_value
from filings_agent.evals.ground_truth import CompanyFacts
from filings_agent.extract.models import ExtractionRecord, Figure, Metric
from filings_agent.llm import LLMConfigError, LLMError
from filings_agent.parse.models import SectionKind
from tests.agent_fixtures import financials, parsed_filing, search_index, stored_filing
from tests.conftest import FakeLLM


@pytest.fixture
def index(tmp_path: Path) -> Iterator[SearchIndex]:
    ix = search_index(tmp_path / "search.db")
    yield ix
    ix.close()


@pytest.fixture
def toolbox(index: SearchIndex) -> Toolbox:
    return Toolbox(financials(), index)


def step(*calls: ToolCall, answer: FinalAnswer | None = None) -> AgentStep:
    return AgentStep(reasoning="r", tool_calls=list(calls), answer=answer)


def final(text: str, citations: list[str], abstained: bool = False) -> AgentStep:
    return step(answer=FinalAnswer(text=text, citations=citations, abstained=abstained))


GET_JPM_NI = ToolCall(
    tool="get_financials", company="JPM", metric=Metric.NET_INCOME, periods=["FY2023", "FY2024"]
)


# --- calculator ----------------------------------------------------------------------


def test_calculator_evaluates_arithmetic_over_source_ids() -> None:
    v = {"F1": 49_552e6, "F2": 58_471e6}
    assert evaluate("(F2 - F1) / F1 * 100", v) == pytest.approx(17.999, rel=1e-4)
    assert evaluate("pct_change(F1, F2)", v) == pytest.approx(17.999, rel=1e-4)
    assert evaluate("cagr(100, 121, 2)") == pytest.approx(10.0)
    assert evaluate("avg(1, 2, 3) + max(1, 5) - abs(-2) + round(2.6)") == 8.0
    assert evaluate("-2 ** 2") == -4.0


@pytest.mark.parametrize(
    "expression",
    [
        "F9 + 1",
        "__import__('os').system('true')",
        "(1).real",
        "1 / 0",
        "2 ** 1000",
        "True + 1",
        "x if 1 else 2",
        "(-8) ** 0.5",
        "1 +",
        "pct_change(0, 5)",
    ],
)
def test_calculator_rejects_unsafe_or_invalid_input(expression: str) -> None:
    with pytest.raises(CalcError):
        evaluate(expression, {"F1": 1.0})


# --- periods and figures ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("FY2024", "FY2024"),
        ("fy 2024", "FY2024"),
        ("2024", "FY2024"),
        ("fiscal year 2024", "FY2024"),
        ("Q3 2025", "Q3 2025"),
        ("q3-2025", "Q3 2025"),
        ("2025Q3", "Q3 2025"),
        ("9M 2025", "9M 2025"),
        ("3M 2025", "Q1 2025"),
        ("latest annual", "latest_annual"),
        ("Latest", "latest"),
        ("last year", None),
    ],
)
def test_normalize_period(text: str, expected: str | None) -> None:
    assert normalize_period(text) == expected


def test_period_labels() -> None:
    assert period_label("10-K", date(2024, 12, 31), Metric.NET_INCOME) == "FY2024"
    assert period_label("10-Q", date(2025, 9, 30), Metric.NET_INCOME) == "Q3 2025"
    assert period_label("10-Q", date(2025, 9, 30), Metric.TOTAL_ASSETS) == "Q3 2025"
    assert period_label("10-Q", date(2025, 9, 30), Metric.OPERATING_CASH_FLOW) == "9M 2025"
    assert period_label("10-Q", date(2025, 3, 31), Metric.OPERATING_CASH_FLOW) == "Q1 2025"


def test_lookup_explains_what_is_missing() -> None:
    fin = financials()
    found = fin.lookup("jpm", Metric.NET_INCOME, ["FY2024", "latest", "Q4 2024", "FY2019", "soon"])
    assert [f.period for f in found.facts] == ["FY2024", "Q3 2025"]
    assert any("no 10-Q covers Q4" in m for m in found.missing)
    assert any("FY2019" in m and "available: FY2023" in m for m in found.missing)
    assert any("unrecognised period 'soon'" in m for m in found.missing)
    assert "not in the filings corpus" in fin.lookup("V", Metric.REVENUE, ["FY2024"]).missing[0]
    assert "bank-only" in fin.lookup("PYPL", Metric.CET1_RATIO, ["FY2024"]).missing[0]
    assert fin.lookup("JPM", Metric.NET_INCOME, ["latest annual"]).facts[0].period == "FY2024"


def test_build_financials_prefers_cited_extractions_over_xbrl() -> None:
    filing = stored_filing("JPM", "10-K", date(2024, 12, 31))
    facts = CompanyFacts(
        {
            "facts": {
                "us-gaap": {
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "val": 58_471e6,
                                    "start": "2024-01-01",
                                    "end": "2024-12-31",
                                    "accn": filing.accession_number,
                                }
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {"val": 4e12, "end": "2024-12-31", "accn": filing.accession_number}
                            ]
                        }
                    },
                }
            }
        }
    )

    def figure(metric: Metric, value: float, valid: bool) -> Figure:
        return Figure(
            metric=metric,
            value=value,
            unit="USD",
            reported_value=value / 1e6,
            scale="millions",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            block_id="b00001",
            quote=f"Net income | {value / 1e6:,.0f}",
            citation_valid=valid,
        )

    record = ExtractionRecord(
        accession_number=filing.accession_number,
        extractor="x",
        model="m",
        created_at=datetime.now(UTC),
        figures=[
            figure(Metric.NET_INCOME, 58_400e6, True),
            figure(Metric.TOTAL_ASSETS, 1.0, False),  # uncited: never used
        ],
    )
    auto = build_financials(
        [filing],
        company_facts={1: facts},
        extractions={filing.accession_number: record},
        names={"JPM": "JPMorgan"},
    )
    ni = auto.facts("JPM", Metric.NET_INCOME)[0]
    assert (ni.value, ni.source, ni.company) == (58_400e6, "extracted", "JPMorgan")
    assert auto.facts("JPM", Metric.TOTAL_ASSETS)[0].source == "xbrl"
    xbrl = build_financials(
        [filing],
        company_facts={1: facts},
        extractions={filing.accession_number: record},
        mode="xbrl",
    )
    assert xbrl.facts("JPM", Metric.NET_INCOME)[0].value == 58_471e6
    extracted = build_financials(
        [filing],
        company_facts={1: facts},
        extractions={filing.accession_number: record},
        mode="extracted",
    )
    assert extracted.facts("JPM", Metric.TOTAL_ASSETS) == []


# --- retrieval ---------------------------------------------------------------------------


def test_search_filters_and_cites(index: SearchIndex) -> None:
    hits = index.search("net interest yield", tickers=["BAC"])
    assert hits and hits[0].ticker == "BAC" and "1.95 percent" in hits[0].text
    assert hits[0].period == "FY2024" and hits[0].page_label == "10"
    assert hits[0].url.endswith("#a0")
    assert index.search("interest rates", section=SectionKind.RISK_FACTORS)[0].section == (
        SectionKind.RISK_FACTORS
    )
    assert index.search("CET1", tickers=["BAC"]) == []
    assert index.search("CET1", period="Q3 2025") == []
    assert index.search("the and of") == []


def test_reindexing_is_idempotent(index: SearchIndex) -> None:
    before = index.count()
    parsed = parsed_filing("BAC", date(2024, 12, 31))
    assert index.is_current(parsed)
    index.add_filing(stored_filing("BAC", "10-K", date(2024, 12, 31)), parsed)
    assert index.count() == before


def test_fts_queries_quote_terms() -> None:
    assert fts_queries('"cet1*" OR') == ['"cet1"']
    queries = fts_queries('net "interest" margin; DROP TABLE')
    assert queries[0].startswith('NEAR("net" "interest" "margin" "drop" "table"')
    assert queries[1] == '"net" AND "interest" AND "margin" AND "drop" AND "table"'
    assert fts_queries("the of and") == []


def test_chunking_groups_paragraphs_and_keeps_tables_alone() -> None:
    parsed = parsed_filing("BAC", date(2024, 12, 31))
    passages = list(chunk(parsed))
    assert len(passages) == 2  # different sections never share a passage


# --- grounding -------------------------------------------------------------------------


def test_mentions_skip_dates_years_and_counts() -> None:
    text = (
        "On December 31, 2024 (2024-12-31) the 3 banks reported $58.5 billion, 18%, "
        "$19.75 per share and 12.4bn in FY2024 Q3 and 9M 2025 under Form 10-K, not $9M."
    )
    raw = [m.raw for m in mentions(text)]
    assert raw == ["$58.5 billion", "18%", "$19.75", "12.4bn", "$9M"]


def test_render_evaluates_placeholders() -> None:
    out = render(
        "up {=pct_change(F1, F2):.1f}% to {=F2 / 1e9:,.1f} billion",
        {
            "F1": 49_552e6,
            "F2": 58_471e6,
        },
    )
    assert out.text == "up 18.0% to 58.5 billion"
    assert out.used == ["F1", "F2"] and not out.errors
    bad = render("{=F9} and {=F1:!r}", {"F1": 1.0})
    assert len(bad.errors) == 2


def test_ungrounded_numbers_are_reported() -> None:
    values = [58_471e6, 49_552e6, 19.75]
    assert (
        ungrounded("Net income was $58.5 billion ($58,471 million), EPS $19.75.", values, []) == []
    )
    assert ungrounded("It was $60 billion.", values, []) == ["$60 billion"]
    assert (
        ungrounded(
            "Yield fell 13 bps to 1.95 percent.", [], ["decreased 13 basis points to 1.95 percent"]
        )
        == []
    )
    assert ungrounded("Up {=pct_change(F1, F2):.1f}%.", values, []) == []


def test_format_value() -> None:
    assert format_value(58_471e6, "USD") == "$58,471 million ($58.47 billion)"
    assert format_value(-512e6, "USD") == "-$512 million"
    assert format_value(4.33, "USD/share") == "$4.33 per share"
    assert format_value(2.711, "percent") == "2.71%"


# --- tools -----------------------------------------------------------------------------


def test_tools_register_sources_once(toolbox: Toolbox) -> None:
    sources = Sources()
    first = toolbox.run(GET_JPM_NI, sources)
    assert "[F1]" in first and "[F2]" in first and "FY2024" in first
    again = toolbox.run(GET_JPM_NI, sources)
    assert again == first, "the same figure keeps its id"
    calc = toolbox.run(ToolCall(tool="calculate", expression="F2 - F1"), sources)
    assert calc == "[C1] F2 - F1 = 8,919,000,000"
    assert sources.by_id["C1"].inputs == ["F2", "F1"]


def test_compare_ranks_in_code(toolbox: Toolbox) -> None:
    out = toolbox.run(
        ToolCall(
            tool="compare",
            companies=["PYPL", "JPM", "BAC"],
            metric=Metric.NET_INCOME,
            period="FY2024",
        ),
        Sources(),
    )
    assert "highest first: JPM" in out and out.index("BAC $27,132") < out.index("PYPL $4,147")


def test_tool_errors_are_returned_not_raised(toolbox: Toolbox) -> None:
    sources = Sources()
    assert toolbox.run(ToolCall(tool="calculate", expression="F1 +"), sources).startswith("error")
    assert "missing argument 'metric'" in toolbox.run(
        ToolCall(tool="get_financials", company="JPM"), sources
    )
    assert "not in the filings corpus" in toolbox.run(
        ToolCall(tool="search_filings", company="V", query="revenue"), sources
    )
    assert "at least two" in toolbox.run(
        ToolCall(tool="compare", companies=["JPM"], metric=Metric.REVENUE, period="FY2024"),
        sources,
    )
    assert "[S1]" in toolbox.run(
        ToolCall(tool="search_filings", query="CET1 ratios", period="FY2024"), sources
    )
    no_index = Toolbox(financials(), None)
    assert "index" in no_index.run(ToolCall(tool="search_filings", query="x"), sources)


# --- the agent loop ------------------------------------------------------------------------


def test_agent_answers_with_computed_numbers_and_citations(toolbox: Toolbox) -> None:
    llm = FakeLLM(
        step(GET_JPM_NI),
        final(
            "JPMorgan's net income rose {=pct_change(F1, F2):.1f}% to $58.5 billion in FY2024.",
            ["F2"],
        ),
    )
    answer = Agent(llm, toolbox).ask("How did JPM's net income change in 2024?")
    assert answer.text == "JPMorgan's net income rose 18.0% to $58.5 billion in FY2024."
    assert not answer.abstained and answer.guardrail is None
    assert [c.id for c in answer.citations] == ["F2", "F1"], "placeholder inputs are cited too"
    assert answer.llm_calls == 2 and answer.input_tokens == 2000
    assert "[F1]" in llm.calls[1]["prompt"] and "step 2 of 4" in llm.calls[1]["prompt"]


def test_guardrail_abstains_on_ungrounded_numbers(toolbox: Toolbox) -> None:
    llm = FakeLLM(step(GET_JPM_NI), final("Net income was $61 billion in FY2024.", ["F2"]))
    answer = Agent(llm, toolbox).ask("q")
    assert answer.abstained and answer.ungrounded_numbers == ["$61 billion"]
    assert answer.draft == "Net income was $61 billion in FY2024."
    assert "numbers not found" in (answer.guardrail or "")


def test_guardrail_requires_a_citation(toolbox: Toolbox) -> None:
    llm = FakeLLM(step(GET_JPM_NI), final("Net income was $58.5 billion.", ["X9"]))
    answer = Agent(llm, toolbox).ask("q")
    assert answer.abstained and answer.guardrail == "the answer cites no source"


def test_guardrail_catches_bad_placeholders(toolbox: Toolbox) -> None:
    llm = FakeLLM(step(GET_JPM_NI), final("Up {=F1 / 0:.1f}%.", ["F1"]))
    assert "calculation failed" in (Agent(llm, toolbox).ask("q").guardrail or "")


def test_model_can_abstain(toolbox: Toolbox) -> None:
    llm = FakeLLM(final("Visa is not in the corpus.", [], abstained=True))
    answer = Agent(llm, toolbox).ask("Visa's revenue?")
    assert answer.abstained and answer.guardrail is None and answer.llm_calls == 1


def test_last_step_must_answer(toolbox: Toolbox) -> None:
    llm = FakeLLM(
        step(
            GET_JPM_NI, answer=FinalAnswer(text="$58.5 billion", citations=["F2"], abstained=False)
        )
    )
    answer = Agent(llm, toolbox, max_steps=2).ask("q")
    assert answer.llm_calls == 2
    assert "the last one" in llm.calls[1]["prompt"]
    assert answer.text == "$58.5 billion" and not answer.abstained
    assert answer.steps[1].tool_calls == [], "tool calls on the last step are not run"


def test_running_out_of_steps_abstains(toolbox: Toolbox) -> None:
    answer = Agent(FakeLLM(step(GET_JPM_NI)), toolbox, max_steps=2).ask("q")
    assert answer.abstained and answer.guardrail == "ran out of steps"


def test_llm_failures(toolbox: Toolbox) -> None:
    failed = Agent(FakeLLM(LLMError("boom")), toolbox).ask("q")
    assert failed.abstained and failed.error == "boom" and failed.llm_calls == 0
    with pytest.raises(LLMConfigError):
        Agent(FakeLLM(LLMConfigError("no key")), toolbox).ask("q")
    with pytest.raises(ValueError):
        Agent(FakeLLM(LLMError("x")), toolbox, max_steps=0)
