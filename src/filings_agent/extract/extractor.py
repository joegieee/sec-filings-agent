"""Extract key financial figures from one parsed filing with an LLM, each with a citation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from filings_agent.extract.citations import citation_valid
from filings_agent.extract.context import DEFAULT_MAX_CHARS, Context, select_context
from filings_agent.extract.models import (
    METRICS,
    SCALE_FACTORS,
    ExtractedFigure,
    ExtractionRecord,
    Figure,
    FinancialsExtraction,
    Metric,
)
from filings_agent.llm.base import LLMResult, StructuredLLM
from filings_agent.parse import PARSER_VERSION
from filings_agent.parse.models import SECTION_TITLES, Block, ParsedFiling

# Bump when the prompt, schema or context selection changes: it is part of the
# extractor id, so results (and cached responses) from different versions never mix.
EXTRACT_VERSION = 1

SYSTEM_PROMPT = """\
You extract key financial figures from SEC filings (10-K and 10-Q). You are given \
excerpts of one filing as blocks, each with an id. For every requested metric, \
report the company-wide (consolidated) figure for the filing's primary reporting \
period and cite the block you read it from.

Rules:
- Only report numbers printed in the excerpts. Never calculate, convert, annualise, \
or infer a figure: no adding, subtracting or rescaling. If a metric is not printed \
in the excerpts for the requested period, return null for it.
- Use consolidated company figures, never a business segment's, a subsidiary bank's \
or a non-GAAP ("adjusted", "managed", "fully taxable-equivalent") variant when the \
reported GAAP figure is given.
- Columns matter: tables often show several periods side by side (current vs prior \
year, three months vs year to date). Take the column for the requested period.
- value: the number exactly as printed, without applying the table's scale; \
parentheses or a minus sign mean negative.
- scale: the scale stated for the table or sentence ("in millions" -> millions; \
"$13.7 billion" -> billions). Per-share amounts and percentages use "units".
- quote: copy a short excerpt verbatim from the cited block (the table row or the \
phrase) that contains the printed number.
"""


@dataclass(frozen=True)
class FilingInfo:
    accession_number: str
    company_name: str
    ticker: str
    form: str
    report_date: date


def extractor_id(llm: StructuredLLM) -> str:
    return f"{llm.id}|extract-v{EXTRACT_VERSION}|parse-v{PARSER_VERSION}"


def _periods(form: str, report_date: date) -> dict[str, str]:
    end = report_date.isoformat()
    if form.upper().startswith("10-K"):
        return {
            "flow": f"the fiscal year ended {end}",
            "ytd": f"the fiscal year ended {end}",
            "instant": f"as of {end}",
        }
    return {
        "flow": f"the three months (fiscal quarter) ended {end}, not year to date",
        "ytd": f"year to date ended {end} (the cash flow statement's only current period)",
        "instant": f"as of {end}",
    }


def render_block(block: Block) -> str:
    attrs = f'id="{block.id}" section="{SECTION_TITLES[block.section]}"'
    if block.page_label:
        attrs += f' page="{block.page_label}"'
    return f'<block {attrs} type="{block.kind}">\n{block.text}\n</block>'


def build_prompt(info: FilingInfo, context: Context) -> str:
    periods = _periods(info.form, info.report_date)
    metric_lines = "\n".join(
        f"- {spec.metric.value} ({spec.unit}, for {periods[spec.period]}): {spec.definition}"
        for spec in METRICS.values()
    )
    blocks = "\n\n".join(render_block(b) for b in context.blocks)
    return (
        f"Filing: {info.company_name} ({info.ticker}), Form {info.form} for the period "
        f"ended {info.report_date.isoformat()}.\n\n"
        f"Metrics to extract:\n{metric_lines}\n\n"
        f"Excerpts:\n\n{blocks}"
    )


def _figure(
    metric: Metric, raw: ExtractedFigure, parsed: ParsedFiling, allowed: set[str]
) -> Figure:
    spec = METRICS[metric]
    block = parsed.block(raw.block_id) if raw.block_id in allowed else None
    scale = "units" if spec.unit != "USD" else raw.scale
    return Figure(
        metric=metric,
        value=raw.value * SCALE_FACTORS[scale],
        unit=spec.unit,
        reported_value=raw.value,
        scale=scale,
        period_start=None if spec.period == "instant" else raw.period_start,
        period_end=raw.period_end,
        block_id=raw.block_id,
        quote=raw.quote,
        section=block.section.value if block else None,
        item=block.item if block else None,
        page_label=block.page_label if block else None,
        url=parsed.citation_url(block) if block else None,
        citation_valid=citation_valid(raw.quote, block.text if block else None, raw.value),
    )


def to_record(
    info: FilingInfo,
    extractor: str,
    result: LLMResult[FinancialsExtraction],
    parsed: ParsedFiling,
    context: Context,
) -> ExtractionRecord:
    allowed = {b.id for b in context.blocks}
    figures = [
        _figure(metric, raw, parsed, allowed)
        for metric, raw in result.output.figures().items()
        if raw is not None
    ]
    return ExtractionRecord(
        accession_number=info.accession_number,
        extractor=extractor,
        model=result.model,
        created_at=datetime.now(UTC),
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cost_usd=result.cost_usd,
        latency_s=round(result.latency_s, 3),
        cached=result.cached,
        figures=figures,
    )


def extract_filing(
    info: FilingInfo,
    parsed: ParsedFiling,
    llm: StructuredLLM,
    *,
    max_context_chars: int = DEFAULT_MAX_CHARS,
) -> ExtractionRecord:
    """Run one extraction. Raises ``LLMError`` if the model call fails."""
    context = select_context(parsed, max_chars=max_context_chars)
    result = llm.generate(
        system=SYSTEM_PROMPT, prompt=build_prompt(info, context), schema=FinancialsExtraction
    )
    return to_record(info, extractor_id(llm), result, parsed, context)
