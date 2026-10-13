"""Structured model steps, source references and per-question traces."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from filings_agent.extract.models import Metric
from filings_agent.parse.models import SectionKind

ToolName = Literal["get_financials", "search_filings", "compare", "calculate"]
SourceKind = Literal["fact", "passage", "calculation"]


# --- what the model returns ------------------------------------------------------


class ToolCall(BaseModel):
    """One tool call. Only the arguments the tool takes are set; the rest are null."""

    tool: ToolName
    company: str | None = Field(
        default=None, description="Ticker, for get_financials and search_filings (e.g. JPM)."
    )
    companies: list[str] | None = Field(default=None, description="Tickers, for compare.")
    metric: Metric | None = Field(default=None, description="For get_financials and compare.")
    periods: list[str] | None = Field(
        default=None,
        description="For get_financials: e.g. FY2024, Q3 2025, 9M 2025, latest, latest annual.",
    )
    period: str | None = Field(
        default=None, description="For compare (required) and search_filings (optional)."
    )
    query: str | None = Field(
        default=None, description="For search_filings: a few specific keywords."
    )
    section: SectionKind | None = Field(
        default=None, description="For search_filings (optional): limit to one section."
    )
    expression: str | None = Field(
        default=None,
        description="For calculate: arithmetic over numbers and source ids, e.g. (F2 - F1) / F1.",
    )


class FinalAnswer(BaseModel):
    text: str = Field(
        description="The answer in one to three sentences. Write computed numbers as "
        "{=EXPR:FMT} placeholders over source ids, e.g. {=pct_change(F1, F2):.1f}%."
    )
    citations: list[str] = Field(description="Ids of every source the answer relies on.")
    abstained: bool = Field(
        description="True when the filings can't support an answer (or it asks for advice)."
    )


class AgentStep(BaseModel):
    reasoning: str = Field(description="One or two sentences: what you need next, and why.")
    tool_calls: list[ToolCall] = Field(
        description="Tools to run now (all you need, in one step); empty when answering."
    )
    answer: FinalAnswer | None = Field(description="Your final answer, once you have the sources.")


# --- what a run produces -----------------------------------------------------------


class SourceRef(BaseModel):
    """Something the agent saw and can cite: a figure, a passage, or a calculation."""

    id: str
    kind: SourceKind
    text: str  # as shown to the model
    ticker: str | None = None
    metric: Metric | None = None
    period: str | None = None
    value: float | None = None
    url: str | None = None
    evidence: str | None = None  # quote or XBRL concept (facts), passage text, or expression
    source: str | None = None  # facts: "extracted" or "xbrl"
    inputs: list[str] = Field(default_factory=list)  # calculations: the ids they used


class StepTrace(BaseModel):
    step: int
    reasoning: str
    tool_calls: list[ToolCall]
    tool_output: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    llm_latency_s: float
    tool_latency_s: float
    cached: bool


class AgentAnswer(BaseModel):
    query_id: str
    question: str
    model: str | None
    text: str
    abstained: bool
    citations: list[SourceRef]
    draft: str | None = None  # the model's own text, before placeholders and guardrails
    guardrail: str | None = None  # why the guardrail replaced the answer, if it did
    ungrounded_numbers: list[str] = Field(default_factory=list)
    error: str | None = None
    steps: list[StepTrace] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)

    @property
    def llm_calls(self) -> int:
        return len(self.steps)

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def cost_usd(self) -> float | None:
        costs = [s.cost_usd for s in self.steps]
        return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)

    @property
    def latency_s(self) -> float:
        return sum(s.llm_latency_s + s.tool_latency_s for s in self.steps)

    @property
    def cached(self) -> bool:
        return bool(self.steps) and all(s.cached for s in self.steps)
