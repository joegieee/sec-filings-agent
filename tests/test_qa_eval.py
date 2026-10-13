from __future__ import annotations

from pathlib import Path

import pytest

from filings_agent.agent.agent import Agent
from filings_agent.agent.models import AgentAnswer, AgentStep, FinalAnswer, ToolCall
from filings_agent.agent.tools import Toolbox
from filings_agent.evals.qa import (
    Question,
    load_questions,
    mentions_value,
    names_company,
    reference,
    run_qa,
)
from filings_agent.evals.report import qa_markdown
from filings_agent.extract.models import Metric
from filings_agent.llm import LLMBudgetError
from tests.agent_fixtures import financials
from tests.conftest import FakeLLM

REPO_QUESTIONS = Path(__file__).parent.parent / "evals" / "qa_core3.toml"


def q(**fields: object) -> Question:
    return Question.model_validate({"id": "x", "question": "?", **fields})


def test_repo_question_set_is_valid() -> None:
    questions = load_questions(REPO_QUESTIONS)
    assert len(questions) >= 20
    assert sum(not x.answerable for x in questions) >= 8


def test_questions_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs ticker, metric"):
        q(kind="value", period="FY2024")
    path = tmp_path / "q.toml"
    path.write_text(
        '[[questions]]\nid = "a"\nquestion = "?"\nkind = "unanswerable"\n'
        '[[questions]]\nid = "a"\nquestion = "?"\nkind = "unanswerable"\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(path)


def test_references_come_from_the_figures() -> None:
    fin = financials()
    value = reference(q(kind="value", ticker="JPM", metric="net_income", period="FY2024"), fin)
    assert value is not None and value.value == 58_471e6
    change = reference(
        q(kind="change", ticker="JPM", metric="net_income", period="FY2023", period_to="FY2024"),
        fin,
    )
    assert change is not None and change.value == pytest.approx(17.999, rel=1e-4)
    ratio = reference(
        q(
            kind="ratio",
            ticker="BAC",
            metric="net_interest_income",
            denominator="revenue",
            period="FY2024",
        ),
        fin,
    )
    assert ratio is not None and ratio.value == pytest.approx(55.02, rel=1e-3)
    lowest = reference(
        q(
            kind="compare",
            tickers=["JPM", "BAC", "PYPL"],
            metric="net_income",
            period="FY2024",
            pick="lowest",
        ),
        fin,
    )
    assert lowest is not None and (lowest.winner, lowest.value) == ("PYPL", 4_147e6)
    assert reference(q(kind="unanswerable"), fin) is None
    with pytest.raises(LookupError):
        reference(q(kind="value", ticker="JPM", metric="revenue", period="FY2019"), fin)


@pytest.mark.parametrize(
    ("text", "value", "unit", "ok"),
    [
        ("$58.5 billion", 58_471e6, "USD", True),
        ("$58,471 million", 58_471e6, "USD", True),
        ("58,471", 58_471e6, "USD", True),
        ("$58 billion", 58_471e6, "USD", True),
        ("$60 billion", 58_471e6, "USD", False),
        ("$0.1 trillion", 58_471e6, "USD", False),
        ("rose 18.0%", 17.999, "percent", True),
        ("fell 18%", -17.999, "percent", True),
        ("rose 17.5%", 17.999, "percent", False),
        ("EPS of $3.99", 3.99, "USD/share", True),
    ],
)
def test_mentions_value(text: str, value: float, unit: str, ok: bool) -> None:
    assert mentions_value(text, value, unit) is ok


def test_names_company() -> None:
    assert names_company("JPMorgan led", "JPM", "JPMorgan Chase & Co.")
    assert names_company("BAC led", "BAC", "Bank of America Corp")
    assert names_company("Bank of America led", "BAC", "BANK OF AMERICA CORP /DE/")
    assert not names_company("the other bank led", "BAC", "Bank of America Corp")


def _ask(*steps: AgentStep) -> Agent:
    return Agent(FakeLLM(*steps), Toolbox(financials(), None))


def _answer(text: str, cites: list[str], abstained: bool = False) -> AgentStep:
    return AgentStep(
        reasoning="r",
        tool_calls=[],
        answer=FinalAnswer(text=text, citations=cites, abstained=abstained),
    )


def _get(ticker: str, metric: Metric, *periods: str) -> AgentStep:
    call = ToolCall(tool="get_financials", company=ticker, metric=metric, periods=list(periods))
    return AgentStep(reasoning="r", tool_calls=[call], answer=None)


def test_run_scores_answers_citations_and_abstentions() -> None:
    questions = [
        q(id="v", kind="value", ticker="JPM", metric="net_income", period="FY2024"),
        q(id="w", kind="value", ticker="PYPL", metric="diluted_eps", period="FY2024"),
        q(id="u", kind="unanswerable"),
        q(id="u2", kind="unanswerable"),
        q(id="c", kind="compare", tickers=["JPM", "BAC"], metric="net_income", period="FY2024"),
    ]
    answers = iter(
        [
            # v: correct, cites the right figure plus an irrelevant one
            _ask(
                _get("JPM", Metric.NET_INCOME, "FY2024"),
                AgentStep(
                    reasoning="r",
                    tool_calls=[
                        ToolCall(
                            tool="get_financials",
                            company="BAC",
                            metric=Metric.NET_INCOME,
                            periods=["FY2024"],
                        )
                    ],
                    answer=None,
                ),
                _answer("JPMorgan earned $58.5 billion in FY2024.", ["F1", "F2"]),
            ),
            # w: abstains although it could answer
            _ask(_answer("I can't tell.", [], abstained=True)),
            # u: abstains correctly
            _ask(_answer("Not in the corpus.", [], abstained=True)),
            # u2: answers anyway (wrong)
            _ask(_get("JPM", Metric.NET_INCOME, "FY2024"), _answer("$58.5 billion", ["F1"])),
            # c: names the winner and its value
            _ask(
                AgentStep(
                    reasoning="r",
                    tool_calls=[
                        ToolCall(
                            tool="compare",
                            companies=["JPM", "BAC"],
                            metric=Metric.NET_INCOME,
                            period="FY2024",
                        )
                    ],
                    answer=None,
                ),
                _answer("JPMorgan, with $58.5 billion.", ["F1"]),
            ),
        ]
    )
    summary = run_qa(questions, lambda text: next(answers).ask(text), financials(), agent="t")
    outcomes = {r.id: r.outcome for r in summary.results}
    assert outcomes == {
        "v": "correct",
        "w": "false_abstention",
        "u": "correct_abstention",
        "u2": "wrong",
        "c": "correct",
    }
    assert summary.by_kind == {"value": (1, 2), "unanswerable": (1, 2), "compare": (1, 1)}
    assert summary.correct == 2 and summary.answerable == 3
    assert summary.abstention_accuracy == 0.5
    assert summary.citation_precision is not None and summary.citation_precision < 1
    assert "Answer accuracy" in qa_markdown(summary)


def test_run_stops_when_the_budget_is_spent() -> None:
    questions = [q(id=f"u{i}", kind="unanswerable") for i in range(3)]
    calls = 0

    def ask(text: str) -> AgentAnswer:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise LLMBudgetError("daily gemini request budget spent")
        return _ask(_answer("No.", [], abstained=True)).ask(text)

    summary = run_qa(questions, ask, financials(), agent="t")
    assert [r.outcome for r in summary.results] == ["correct_abstention", "not_run", "not_run"]
    assert summary.run == 1 and "budget" in (summary.results[2].error or "")
