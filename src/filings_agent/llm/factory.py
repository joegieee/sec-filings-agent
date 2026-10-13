"""Build the configured LLM provider (wrapped in the response cache)."""

from __future__ import annotations

from pydantic import SecretStr

from filings_agent.config import LLMProvider, Settings
from filings_agent.llm.anthropic import DEFAULT_MODEL as CLAUDE_MODEL
from filings_agent.llm.anthropic import AnthropicLLM, Effort
from filings_agent.llm.base import LLMConfigError, StructuredLLM
from filings_agent.llm.budget import BudgetedLLM, RequestLedger
from filings_agent.llm.cache import CachedLLM
from filings_agent.llm.ollama import DEFAULT_BASE_URL as OLLAMA_URL
from filings_agent.llm.ollama import DEFAULT_MODEL as OLLAMA_MODEL
from filings_agent.llm.ollama import OllamaLLM
from filings_agent.llm.openai_compat import GEMINI_BASE_URL, OpenAICompatibleLLM, ReasoningEffort

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FREE_DAILY_REQUESTS = 20
# OpenAI-style APIs take low/medium/high; Claude's two higher levels map onto "high".
_REASONING: dict[str, ReasoningEffort] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value is not None else None


def build_llm(
    settings: Settings,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    effort: Effort | None = None,
) -> StructuredLLM:
    """The provider from settings (or the overrides), with its daily request cap if any,
    cached unless FRA_LLM_CACHE=false (cache hits never count against the cap)."""
    provider = provider or settings.llm_provider
    model = model or settings.llm_model
    effort = effort or settings.llm_effort
    prices = (
        (settings.llm_price_input, settings.llm_price_output)
        if settings.llm_price_input is not None and settings.llm_price_output is not None
        else None
    )
    timeout = settings.llm_timeout_seconds
    retries = settings.llm_max_retries
    reasoning = _REASONING[effort] if effort is not None else None

    llm: StructuredLLM
    if provider == "gemini":
        llm = OpenAICompatibleLLM(
            provider="gemini",
            model=model or GEMINI_MODEL,
            base_url=settings.llm_base_url or GEMINI_BASE_URL,
            api_key=_secret(settings.gemini_api_key),
            key_name="GEMINI_API_KEY",
            max_tokens=settings.llm_max_tokens,
            reasoning_effort=reasoning,
            prices=prices,
            # Free-tier Gemini calls can take several minutes, and every retry of a
            # timed-out call spends one of a small daily request quota.
            timeout_s=timeout or 900.0,
            max_retries=retries if retries is not None else 1,
        )
    elif provider == "openai_compatible":
        if not settings.llm_base_url or not model:
            raise LLMConfigError(
                "openai_compatible needs FRA_LLM_BASE_URL and FRA_LLM_MODEL "
                "(e.g. https://openrouter.ai/api/v1 and a model id)"
            )
        llm = OpenAICompatibleLLM(
            provider="openai_compatible",
            model=model,
            base_url=settings.llm_base_url,
            api_key=_secret(settings.llm_api_key),
            key_name="FRA_LLM_API_KEY",
            max_tokens=settings.llm_max_tokens,
            reasoning_effort=reasoning,
            prices=prices,
            timeout_s=timeout or 300.0,
            max_retries=retries if retries is not None else 4,
        )
    elif provider == "ollama":
        llm = OllamaLLM(
            model=model or OLLAMA_MODEL,
            base_url=settings.llm_base_url or OLLAMA_URL,
            num_ctx=settings.ollama_num_ctx,
            think=settings.ollama_think,
            timeout_s=timeout or 900.0,
        )
    else:
        llm = AnthropicLLM(
            model=model or CLAUDE_MODEL,
            effort=effort or "high",
            max_tokens=settings.llm_max_tokens,
            api_key=_secret(settings.anthropic_api_key),
            timeout_s=timeout or 300.0,
            max_retries=retries if retries is not None else 4,
            fallbacks=settings.llm_fallbacks,
        )
    limit = settings.llm_daily_limit
    if limit is None and provider == "gemini":
        limit = GEMINI_FREE_DAILY_REQUESTS
    if limit is not None:
        ledger = RequestLedger(settings.data_dir / "llm_requests.json")
        llm = BudgetedLLM(llm, provider=provider, limit=limit, ledger=ledger)
    if settings.llm_cache:
        return CachedLLM(llm, settings.data_dir / "cache" / "llm")
    return llm
