"""Metric definitions, the LLM's output schema, and the stored result of an extraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Unit = Literal["USD", "USD/share", "percent"]
# Which period a metric is reported for:
#   flow     -> income-statement style: the quarter (10-Q) or fiscal year (10-K)
#   ytd      -> cash-flow statement: year to date (10-Q) or fiscal year (10-K)
#   instant  -> balance-sheet style: as of the period end
PeriodType = Literal["flow", "ytd", "instant"]
Scale = Literal["units", "thousands", "millions", "billions"]

SCALE_FACTORS: dict[str, float] = {
    "units": 1.0,
    "thousands": 1e3,
    "millions": 1e6,
    "billions": 1e9,
}


class Metric(StrEnum):
    REVENUE = "revenue"
    NET_INCOME = "net_income"
    DILUTED_EPS = "diluted_eps"
    TOTAL_ASSETS = "total_assets"
    TOTAL_EQUITY = "total_equity"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    NET_INTEREST_INCOME = "net_interest_income"
    NET_INTEREST_MARGIN = "net_interest_margin"
    CET1_RATIO = "cet1_ratio"


@dataclass(frozen=True)
class MetricSpec:
    metric: Metric
    label: str
    definition: str
    unit: Unit
    period: PeriodType
    bank_only: bool = False


METRICS: dict[Metric, MetricSpec] = {
    s.metric: s
    for s in (
        MetricSpec(
            Metric.REVENUE,
            "Revenue",
            "Total revenue as reported on the income statement. For banks this is total net "
            "revenue, i.e. net of interest expense (e.g. 'Total net revenue', 'Total revenue, "
            "net of interest expense').",
            "USD",
            "flow",
        ),
        MetricSpec(
            Metric.NET_INCOME,
            "Net income",
            "Net income (loss) attributable to the company, from the income statement. Not "
            "net income applicable to common stockholders.",
            "USD",
            "flow",
        ),
        MetricSpec(
            Metric.DILUTED_EPS,
            "Diluted EPS",
            "Diluted earnings (loss) per common share, from the income statement.",
            "USD/share",
            "flow",
        ),
        MetricSpec(
            Metric.TOTAL_ASSETS,
            "Total assets",
            "Total assets, from the balance sheet.",
            "USD",
            "instant",
        ),
        MetricSpec(
            Metric.TOTAL_EQUITY,
            "Total equity",
            "Total stockholders' (shareholders') equity attributable to the company, from the "
            "balance sheet.",
            "USD",
            "instant",
        ),
        MetricSpec(
            Metric.OPERATING_CASH_FLOW,
            "Operating cash flow",
            "Net cash provided by (used in) operating activities, from the cash flow "
            "statement. Negative when cash was used.",
            "USD",
            "ytd",
        ),
        MetricSpec(
            Metric.NET_INTEREST_INCOME,
            "Net interest income",
            "Net interest income (interest income minus interest expense), from the income "
            "statement. Banks only; null for companies that don't report it as a line item.",
            "USD",
            "flow",
            bank_only=True,
        ),
        MetricSpec(
            Metric.NET_INTEREST_MARGIN,
            "Net interest margin",
            "Net interest margin (or net yield on interest-earning assets) for the period, as a "
            "percentage, as stated by the company (reported basis, not fully taxable-equivalent "
            "if both are given). Banks only.",
            "percent",
            "flow",
            bank_only=True,
        ),
        MetricSpec(
            Metric.CET1_RATIO,
            "CET1 ratio",
            "Common Equity Tier 1 (CET1) capital ratio of the company at period end, as a "
            "percentage. If both Standardized and Advanced approaches are reported, use the "
            "Standardized ratio. Banks only.",
            "percent",
            "instant",
            bank_only=True,
        ),
    )
}


# --- what the LLM returns ---------------------------------------------------------


class ExtractedFigure(BaseModel):
    """One figure as the model read it from the filing, before normalisation."""

    value: float = Field(
        description="The number exactly as printed, without applying the table's scale. "
        "Parentheses or a minus sign mean negative: '(1,234)' -> -1234. '4.33' -> 4.33. "
        "'2.71%' -> 2.71."
    )
    scale: Scale = Field(
        description="Scale the printed number is stated in, e.g. 'millions' for a table "
        "headed '(in millions)'. Per-share amounts and percentages are 'units'."
    )
    period_start: date | None = Field(
        description="First day of the period the figure covers; null for point-in-time "
        "figures such as total assets or a capital ratio."
    )
    period_end: date = Field(description="Last day of the period, or the as-of date.")
    block_id: str = Field(description="id of the block the number is taken from.")
    quote: str = Field(
        description="Verbatim excerpt of that block, at most about 200 characters, containing "
        "the printed number and its label (for a table, the row as written, including the "
        "' | ' cell separators)."
    )


class FinancialsExtraction(BaseModel):
    """The model's answer for one filing: each metric, or null when the excerpts lack it."""

    revenue: ExtractedFigure | None
    net_income: ExtractedFigure | None
    diluted_eps: ExtractedFigure | None
    total_assets: ExtractedFigure | None
    total_equity: ExtractedFigure | None
    operating_cash_flow: ExtractedFigure | None
    net_interest_income: ExtractedFigure | None
    net_interest_margin: ExtractedFigure | None
    cet1_ratio: ExtractedFigure | None

    def figures(self) -> dict[Metric, ExtractedFigure | None]:
        return {m: getattr(self, m.value) for m in Metric}


# --- what we store --------------------------------------------------------------


class Figure(BaseModel):
    """A normalised, cited figure: ``value`` is in base units (USD, USD/share or percent)."""

    model_config = ConfigDict(frozen=True)

    metric: Metric
    value: float
    unit: Unit
    reported_value: float
    scale: Scale
    period_start: date | None
    period_end: date
    block_id: str
    quote: str
    section: str | None = None
    item: str | None = None
    page_label: str | None = None
    url: str | None = None
    citation_valid: bool


class ExtractionRecord(BaseModel):
    accession_number: str
    extractor: str
    model: str | None
    created_at: datetime
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float | None = None
    cached: bool = False
    error: str | None = None
    figures: list[Figure] = Field(default_factory=list)

    @classmethod
    def failed(cls, accession_number: str, extractor: str, error: str) -> ExtractionRecord:
        return cls(
            accession_number=accession_number,
            extractor=extractor,
            model=None,
            created_at=datetime.now(UTC),
            error=error,
        )

    def figure(self, metric: Metric) -> Figure | None:
        return next((f for f in self.figures if f.metric == metric), None)
