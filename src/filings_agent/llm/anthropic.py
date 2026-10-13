"""Claude via the official Anthropic SDK, using structured outputs (JSON schema)."""

from __future__ import annotations

import time
from typing import Literal

import anthropic
from anthropic.types.beta import BetaJSONOutputFormatParam, BetaOutputConfigParam
from pydantic import BaseModel, ValidationError

from filings_agent.llm.base import LLMConfigError, LLMError, LLMRefusalError, LLMResult, Usage
from filings_agent.llm.pricing import cost_usd

DEFAULT_MODEL = "claude-opus-5-5"
# Server-side fallback: if the model declines a request, the API re-runs it on a
# substitute model within the same call ("default" routing by refusal category).
FALLBACK_BETA = "server-side-fallback-2026-07-01"
NO_CREDENTIALS = "no Anthropic API credentials: set ANTHROPIC_API_KEY (in the environment or .env)"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AnthropicLLM:
    """Structured-output calls to Claude.

    The SDK retries 408/409/429/5xx and connection errors with exponential
    backoff (``max_retries``); anything still failing surfaces as ``LLMError``.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        effort: Effort = "high",
        max_tokens: int = 16_000,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        max_retries: int = 4,
        fallbacks: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self._client = client
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    def _get_client(self) -> anthropic.Anthropic:
        # Created on first use, so runs served entirely from cache need no credentials.
        if self._client is None:
            try:
                self._client = anthropic.Anthropic(
                    api_key=self._api_key, timeout=self._timeout_s, max_retries=self._max_retries
                )
            except anthropic.AnthropicError as exc:
                raise LLMConfigError(f"{NO_CREDENTIALS} ({exc})") from exc
        return self._client

    @property
    def id(self) -> str:
        return f"anthropic:{self.model}:effort={self.effort}"

    def generate[T: BaseModel](self, *, system: str, prompt: str, schema: type[T]) -> LLMResult[T]:
        client = self._get_client()
        output_config: BetaOutputConfigParam = {
            "effort": self.effort,
            "format": BetaJSONOutputFormatParam(
                type="json_schema", schema=anthropic.transform_schema(schema)
            ),
        }
        started = time.perf_counter()
        try:
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config=output_config,
                betas=[FALLBACK_BETA] if self.fallbacks else anthropic.omit,
                fallbacks="default" if self.fallbacks else anthropic.omit,
            )
        except anthropic.CredentialsError as exc:
            raise LLMConfigError(f"{NO_CREDENTIALS} ({exc})") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMConfigError(f"Anthropic API rejected the credentials: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMConfigError(f"unknown model {self.model!r}: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Anthropic API unreachable: {exc}") from exc
        except TypeError as exc:
            # The SDK raises TypeError when it finds no credentials at all.
            if "authentication" not in str(exc):
                raise
            raise LLMConfigError(NO_CREDENTIALS) from exc
        latency = time.perf_counter() - started

        # Check why generation stopped before trusting the output: a refusal or a
        # max_tokens cut-off leaves no (or truncated) JSON.
        if response.stop_reason == "refusal":
            raise LLMRefusalError(f"{response.model} declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"response truncated at max_tokens={self.max_tokens}")
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            output = schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(f"structured output failed validation: {exc}") from exc
        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_input_tokens=u.cache_read_input_tokens or 0,
            cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
        )
        return LLMResult(
            output=output,
            usage=usage,
            model=response.model,
            latency_s=latency,
            cost_usd=cost_usd(response.model, usage),
        )
