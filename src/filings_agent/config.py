"""Runtime settings (from env / .env) and the company universe (from TOML)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

LLMProvider = Literal["gemini", "ollama", "openai_compatible", "anthropic"]


class Settings(BaseSettings):
    """Process-wide settings. Every field can be set with an ``FRA_``-prefixed env var."""

    model_config = SettingsConfigDict(env_prefix="FRA_", env_file=".env", extra="ignore")

    # SEC fair-access policy requires a descriptive User-Agent with contact details.
    sec_user_agent: str
    data_dir: Path = Path("data")
    companies_file: Path = Path("config/companies.toml")
    # SEC allows up to 10 req/s; stay well under it by default.
    sec_max_requests_per_second: float = Field(default=5.0, gt=0, le=10)
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    http_max_retries: int = Field(default=5, ge=0)
    forms: tuple[str, ...] = ("10-K", "10-Q")
    lookback_years: int = Field(default=3, ge=1)

    # LLM (extraction). FRA_LLM_PROVIDER picks the backend; each reads its key from the
    # variable its own SDK uses. Gemini's key (Google AI Studio) has a free tier, and
    # Ollama runs open models locally with no key at all.
    llm_provider: LLMProvider = "gemini"
    llm_model: str | None = None  # default depends on the provider
    llm_base_url: str | None = None  # required for openai_compatible
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "FRA_GEMINI_API_KEY"),
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "FRA_ANTHROPIC_API_KEY")
    )
    llm_api_key: SecretStr | None = None  # FRA_LLM_API_KEY, for openai_compatible
    # Reasoning effort; unset means the provider's default ("high" for Claude).
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # USD per million tokens, for cost reporting where list prices aren't built in.
    llm_price_input: float | None = Field(default=None, ge=0)
    llm_price_output: float | None = Field(default=None, ge=0)
    llm_max_tokens: int = Field(default=16_000, gt=0)
    # Unset means the provider default: 300 s and 4 retries; 900 s and 1 retry for
    # Gemini (slow free-tier calls, small daily quota); 900 s for Ollama.
    llm_timeout_seconds: float | None = Field(default=None, gt=0)
    llm_max_retries: int | None = Field(default=None, ge=0)
    # Claude only: server-side refusal fallback (re-runs a declined request elsewhere).
    llm_fallbacks: bool = True
    llm_cache: bool = True
    # Real (uncached) requests allowed per day, counted in data/llm_requests.json and
    # reset at midnight Pacific. Unset: 20 for Gemini (its free tier), no cap otherwise.
    llm_daily_limit: int | None = Field(default=None, ge=0)
    ollama_num_ctx: int = Field(default=16_384, ge=2048)
    # Thinking makes local models much slower; null leaves the model's own default.
    ollama_think: bool | None = False
    extraction_max_context_chars: int = Field(default=80_000, gt=0)

    @field_validator("gemini_api_key", "anthropic_api_key", "llm_api_key")
    @classmethod
    def _blank_key_is_unset(cls, value: SecretStr | None) -> SecretStr | None:
        return value if value is not None and value.get_secret_value().strip() else None

    @model_validator(mode="after")
    def _prices_come_in_pairs(self) -> Settings:
        if (self.llm_price_input is None) != (self.llm_price_output is None):
            raise ValueError("set both FRA_LLM_PRICE_INPUT and FRA_LLM_PRICE_OUTPUT, or neither")
        return self

    @field_validator("sec_user_agent")
    @classmethod
    def _user_agent_has_contact_email(cls, value: str) -> str:
        if not _EMAIL_RE.search(value):
            raise ValueError(
                "FRA_SEC_USER_AGENT must include a contact email, "
                "e.g. 'filings-agent you@example.com' (SEC fair-access policy)"
            )
        return value.strip()


class Company(BaseModel):
    ticker: str
    cik: int = Field(gt=0)
    name: str
    sector: Literal["bank", "fintech", "payments"]

    @field_validator("ticker")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()


class CompanyUniverse(BaseModel):
    companies: list[Company]
    sets: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_consistency(self) -> CompanyUniverse:
        tickers = [c.ticker for c in self.companies]
        dupes = {t for t in tickers if tickers.count(t) > 1}
        if dupes:
            raise ValueError(f"duplicate tickers in company universe: {sorted(dupes)}")
        known = set(tickers)
        for set_name, members in self.sets.items():
            unknown = {m.upper() for m in members} - known
            if unknown:
                raise ValueError(f"set {set_name!r} references unknown tickers: {sorted(unknown)}")
        return self

    @classmethod
    def from_toml(cls, path: Path) -> CompanyUniverse:
        with path.open("rb") as fh:
            return cls.model_validate(tomllib.load(fh))

    def select(
        self, tickers: list[str] | None = None, set_name: str | None = None
    ) -> list[Company]:
        """Resolve tickers and/or a named set to companies, preserving config order."""
        wanted: set[str] = set()
        if set_name is not None:
            if set_name not in self.sets:
                raise KeyError(f"unknown company set {set_name!r}; known: {sorted(self.sets)}")
            wanted.update(t.upper() for t in self.sets[set_name])
        if tickers:
            wanted.update(t.upper() for t in tickers)
        if not wanted:
            return list(self.companies)
        by_ticker = {c.ticker: c for c in self.companies}
        unknown = wanted - by_ticker.keys()
        if unknown:
            raise KeyError(f"unknown tickers: {sorted(unknown)}")
        return [c for c in self.companies if c.ticker in wanted]
