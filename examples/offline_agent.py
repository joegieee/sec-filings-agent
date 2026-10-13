"""Run the agent with invented figures and scripted replies, without network or storage."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import structlog
from pydantic import BaseModel

from filings_agent.agent.agent import Agent
from filings_agent.agent.financials import Fact, FinancialsIndex
from filings_agent.agent.models import AgentStep, FinalAnswer, ToolCall
from filings_agent.agent.tools import Toolbox
from filings_agent.extract.models import Metric
from filings_agent.llm.base import LLMResult, Usage


class ScriptedLLM:
    id = "scripted-example"

    def __init__(self, replies: list[AgentStep]) -> None:
        self.replies: Iterator[AgentStep] = iter(replies)

    def generate[T: BaseModel](self, *, system: str, prompt: str, schema: type[T]) -> LLMResult[T]:
        return LLMResult(
            output=schema.model_validate(next(self.replies).model_dump()),
            usage=Usage(),
            model=self.id,
            latency_s=0,
            cost_usd=0,
        )


def main() -> None:
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(50))
    figures = [
        Fact(
            ticker="EXB",
            company="Example Bank",
            metric=Metric.NET_INCOME,
            period=f"FY{year}",
            period_start=date(year, 1, 1),
            period_end=date(year, 12, 31),
            value=millions * 1e6,
            unit="USD",
            form="10-K",
            accession_number=f"synthetic-{year}",
            filing_date=date(year + 1, 2, 1),
            url=f"https://example.invalid/EXB/{year}#income",
            source="extracted",
            evidence=f"Net income | {millions} (USD millions; invented for this example)",
        )
        for year, millions in [(2023, 100), (2024, 120)]
    ]
    lookup = AgentStep(
        reasoning="Read both annual figures.",
        tool_calls=[
            ToolCall(
                tool="get_financials",
                company="EXB",
                metric=Metric.NET_INCOME,
                periods=["FY2023", "FY2024"],
            )
        ],
        answer=None,
    )
    replies = [
        lookup,
        AgentStep(
            reasoning="Use the two figures to calculate the change.",
            tool_calls=[],
            answer=FinalAnswer(
                text="Example Bank's net income rose {=pct_change(F1, F2):.1f}% "
                "to $120 million in FY2024.",
                citations=["F2"],
                abstained=False,
            ),
        ),
        lookup,
        AgentStep(
            reasoning="This deliberately wrong draft should be rejected.",
            tool_calls=[],
            answer=FinalAnswer(
                text="Net income was $999 million.", citations=["F2"], abstained=False
            ),
        ),
    ]
    agent = Agent(
        ScriptedLLM(replies), Toolbox(FinancialsIndex(figures, {"EXB": "Example Bank"}), None)
    )
    print("Synthetic example: invented figures and scripted replies; no model calls.")
    answer = agent.ask("How did EXB net income change from 2023 to 2024?")
    print(answer.text)
    print("Sources:", ", ".join(source.id for source in answer.citations))
    print(answer.steps[0].tool_output)
    rejected = agent.ask("What was EXB net income in FY2024?")
    print("Abstained:", rejected.abstained)
    print("Reason:", rejected.guardrail)


if __name__ == "__main__":
    main()
