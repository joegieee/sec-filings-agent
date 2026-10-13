"""Run structured model steps, dispatch tools, then check the final draft.

Each step requests tools or supplies an answer. Calculated placeholders are
rendered in code; other numbers go through the heuristic checks in grounding.
"""

from __future__ import annotations

import time
import uuid

import structlog

from filings_agent.agent.financials import FinancialsIndex
from filings_agent.agent.grounding import render, ungrounded
from filings_agent.agent.models import AgentAnswer, AgentStep, FinalAnswer, StepTrace, ToolCall
from filings_agent.agent.tools import MAX_CALLS_PER_STEP, Sources, Toolbox
from filings_agent.extract.models import METRICS
from filings_agent.llm.base import LLMConfigError, LLMError, StructuredLLM
from filings_agent.parse.models import SectionKind

log = structlog.get_logger(__name__)

# Bump when the prompts or the step schema change (part of the agent id, like the
# extractor's version, so eval results from different versions never mix).
AGENT_VERSION = 1
DEFAULT_MAX_STEPS = 4

SYSTEM_PROMPT = """\
You are a financial research assistant. You answer analyst questions about \
companies using only their SEC filings, through these tools:

- get_financials(company, metric, periods): reported figures for one company. \
periods is a list such as ["FY2023", "FY2024"], ["Q3 2025"], ["9M 2025"] or \
["latest annual"]. Annual figures are FYyyyy (from the 10-K); quarters are \
"Qn yyyy" (from 10-Qs; there is no Q4 10-Q, use the FY figure); cash flows in \
10-Qs are year to date ("6M 2025", "9M 2025").
- compare(companies, metric, period): one metric for several companies in one \
period, ranked by code.
- search_filings(query, company, section, period): passages of filing text, for \
anything that isn't a listed metric (explanations, risks, ratios such as net \
interest margin or CET1). Use a few specific keywords, e.g. "net interest \
margin", not a whole question. company, section and period are optional.
- calculate(expression): arithmetic over numbers and source ids, e.g. \
"(F2 - F1) / F1 * 100". Functions: pct_change(old, new), cagr(first, last, \
years), avg, sum, min, max, abs, round.

Each tool result has an id: F1 (figure), S1 (passage), C1 (calculation).

How to work:
- Ask for everything you need in one step when you can (several tool calls at \
once); you have few steps.
- Never do arithmetic yourself. In the answer text, write computed numbers as \
placeholders over source ids, which are evaluated for you: \
{=pct_change(F1, F2):.1f}% or {=(F2 - F1) / 1e9:,.1f} billion or \
{=F1 / F2 * 100:.1f}%. Figures are in base units (USD, not millions). You may \
also copy a figure as printed in a source, rounded, e.g. "$58.5 billion".
- Every number in your answer must come from a source. Don't write page \
numbers or source ids in the text; list the ids you used in citations.
- Abstain (abstained = true, and say briefly what is missing) when the sources \
don't support an answer: a company not in the corpus, a period not covered, a \
metric not reported, or a forecast. Never guess, and never use outside knowledge \
for figures.
- Don't give investment advice or buy/sell/hold views, and don't give market \
data such as share prices: abstain and say you report and compare figures from \
filings.
- Answer in one to three sentences, naming the period and units.
"""


class Agent:
    def __init__(
        self,
        llm: StructuredLLM,
        toolbox: Toolbox,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps

    @property
    def id(self) -> str:
        return f"{self.llm.id}|agent-v{AGENT_VERSION}|steps={self.max_steps}"

    def ask(self, question: str) -> AgentAnswer:
        query_id = uuid.uuid4().hex[:12]
        structlog.contextvars.bind_contextvars(query_id=query_id)
        try:
            return self._ask(question, query_id)
        finally:
            structlog.contextvars.unbind_contextvars("query_id")

    def _ask(self, question: str, query_id: str) -> AgentAnswer:
        sources = Sources()
        steps: list[StepTrace] = []
        transcript: list[str] = []
        final: FinalAnswer | None = None
        model: str | None = None
        error: str | None = None
        for step in range(1, self.max_steps + 1):
            last = step == self.max_steps
            prompt = self._prompt(question, transcript, step, last)
            try:
                result = self.llm.generate(system=SYSTEM_PROMPT, prompt=prompt, schema=AgentStep)
            except LLMConfigError:
                raise  # every question would fail the same way
            except LLMError as exc:
                error = str(exc)
                log.error("agent.llm_failed", step=step, error=error)
                break
            model = result.model
            out = result.output
            calls = [] if last else out.tool_calls[:MAX_CALLS_PER_STEP]
            started = time.perf_counter()
            outputs = [self.toolbox.run(call, sources) for call in calls]
            tool_latency = time.perf_counter() - started
            tool_output = "\n\n".join(
                f"{_describe(call)}\n{text}" for call, text in zip(calls, outputs, strict=True)
            )
            steps.append(
                StepTrace(
                    step=step,
                    reasoning=out.reasoning,
                    tool_calls=calls,
                    tool_output=tool_output,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    cost_usd=result.cost_usd,
                    llm_latency_s=round(result.latency_s, 3),
                    tool_latency_s=round(tool_latency, 4),
                    cached=result.cached,
                )
            )
            log.info(
                "agent.step",
                step=step,
                tools=[c.tool for c in calls],
                answered=out.answer is not None and not calls,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cost_usd=result.cost_usd,
                latency_s=round(result.latency_s, 3),
                cached=result.cached,
            )
            if not calls:
                final = out.answer
                break
            transcript.append(f"Step {step}: {out.reasoning}\n\n{tool_output}")
        answer = self._finish(question, query_id, model, final, sources, error)
        answer.steps = steps
        log.info(
            "agent.answer",
            abstained=answer.abstained,
            guardrail=answer.guardrail,
            llm_calls=answer.llm_calls,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            cost_usd=answer.cost_usd,
            latency_s=round(answer.latency_s, 3),
        )
        return answer

    def _prompt(self, question: str, transcript: list[str], step: int, last: bool) -> str:
        fin: FinancialsIndex = self.toolbox.financials
        companies = "\n".join(f"- {t}: {name}" for t, name in sorted(fin.companies.items()))
        metrics = "\n".join(
            f"- {s.metric.value} ({s.unit}{', banks only' if s.bank_only else ''})"
            for s in METRICS.values()
        )
        sections = ", ".join(k.value for k in SectionKind)
        work = "\n\n".join(transcript) if transcript else "(nothing yet)"
        if last:
            instruction = (
                f"This is step {step} of {self.max_steps}, the last one: give your final "
                "answer now from the sources above (or abstain). Tool calls will not run."
            )
        else:
            instruction = (
                f"This is step {step} of {self.max_steps}. Call the tools you need, or give "
                "your final answer with tool_calls empty."
            )
        return (
            f"Question: {question}\n\n"
            f"Companies in the corpus (tickers):\n{companies}\n\n"
            f"Metrics for get_financials and compare:\n{metrics}\n\n"
            f"Periods with figures: {', '.join(fin.periods())}\n"
            f"Sections for search_filings: {sections}\n\n"
            f"Work so far:\n\n{work}\n\n{instruction}"
        )

    def _finish(
        self,
        question: str,
        query_id: str,
        model: str | None,
        final: FinalAnswer | None,
        sources: Sources,
        error: str | None,
    ) -> AgentAnswer:
        def abstain(
            reason: str, draft: str | None = None, loose: list[str] | None = None
        ) -> AgentAnswer:
            return AgentAnswer(
                query_id=query_id,
                question=question,
                model=model,
                text="I can't give a sourced answer to this from the filings.",
                abstained=True,
                citations=[],
                draft=draft,
                guardrail=reason,
                error=error,
                sources=list(sources.by_id.values()),
                ungrounded_numbers=loose or [],
            )

        if final is None:
            return abstain("the model call failed" if error else "ran out of steps")
        rendered = render(final.text, sources.variables())
        cited = [c.strip().strip("[]") for c in final.citations]
        ids = [i for i in dict.fromkeys([*cited, *rendered.used]) if i in sources.by_id]
        if final.abstained:
            return AgentAnswer(
                query_id=query_id,
                question=question,
                model=model,
                text=rendered.text,
                abstained=True,
                citations=[sources.by_id[i] for i in ids],
                draft=final.text,
                sources=list(sources.by_id.values()),
            )
        if rendered.errors:
            return abstain("calculation failed: " + "; ".join(rendered.errors), final.text)
        values = [s.value for s in sources.by_id.values() if s.value is not None]
        passages = [s.evidence or "" for s in sources.by_id.values() if s.kind == "passage"]
        loose = ungrounded(final.text, values + rendered.values, passages)
        if loose:
            return abstain(
                "numbers not found in any source: " + ", ".join(loose),
                final.text,
                loose,
            )
        if not ids:
            return abstain("the answer cites no source", final.text)
        return AgentAnswer(
            query_id=query_id,
            question=question,
            model=model,
            text=rendered.text,
            abstained=False,
            citations=[sources.by_id[i] for i in ids],
            draft=final.text,
            sources=list(sources.by_id.values()),
        )


def _describe(call: ToolCall) -> str:
    args = {
        k: v
        for k, v in call.model_dump(mode="json", exclude={"tool"}).items()
        if v not in (None, [], "")
    }
    rendered = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{call.tool}({rendered}):"
