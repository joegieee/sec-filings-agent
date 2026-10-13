"""Any OpenAI-compatible chat-completions API: Gemini, OpenRouter, DeepSeek, Groq...

Structured output is requested as a strict JSON schema. A backend that rejects
``json_schema`` falls back to plain JSON mode with the schema in the prompt; that
switch is remembered for the rest of the run. Either way the reply is validated
with the Pydantic model, and an invalid reply gets one repair attempt.
"""

from __future__ import annotations

import json
import time
from typing import Literal

import openai
import structlog
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONObject, ResponseFormatJSONSchema
from pydantic import BaseModel, ValidationError

from filings_agent.llm.base import (
    LLMConfigError,
    LLMError,
    LLMRefusalError,
    LLMResult,
    Usage,
    json_text,
)
from filings_agent.llm.pricing import cost_usd
from filings_agent.llm.schema import inline_json_schema

log = structlog.get_logger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
ReasoningEffort = Literal["low", "medium", "high"]


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        key_name: str,
        max_tokens: int = 16_000,
        reasoning_effort: ReasoningEffort | None = None,
        prices: tuple[float, float] | None = None,
        timeout_s: float = 300.0,
        max_retries: int = 4,
        client: openai.OpenAI | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.prices = prices
        self._base_url = base_url
        self._api_key = api_key
        self._key_name = key_name
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client = client
        self._json_schema_supported = True

    @property
    def id(self) -> str:
        effort = f":effort={self.reasoning_effort}" if self.reasoning_effort else ""
        return f"{self.provider}:{self.model}{effort}"

    def _get_client(self) -> openai.OpenAI:
        # Created on first use, so runs served entirely from cache need no key.
        if self._client is None:
            if not self._api_key:
                raise LLMConfigError(f"no API key for {self.provider}: set {self._key_name}")
            self._client = openai.OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout_s,
                max_retries=self._max_retries,
            )
        return self._client

    def generate[T: BaseModel](self, *, system: str, prompt: str, schema: type[T]) -> LLMResult[T]:
        client = self._get_client()
        json_schema = inline_json_schema(schema)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        usage = Usage()
        started = time.perf_counter()
        for attempt in range(2):
            content, served_by, call_usage = self._complete(client, messages, json_schema)
            usage = Usage(
                input_tokens=usage.input_tokens + call_usage.input_tokens,
                output_tokens=usage.output_tokens + call_usage.output_tokens,
            )
            try:
                output = schema.model_validate_json(json_text(content))
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMError(f"structured output failed validation: {exc}") from exc
                log.warning("llm.invalid_output_retry", provider=self.provider, errors=exc.errors())
                messages += [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "That reply does not match the required JSON schema: "
                        f"{exc}. Reply again with only the corrected JSON object.",
                    },
                ]
                continue
            return LLMResult(
                output=output,
                usage=usage,
                model=served_by,
                latency_s=time.perf_counter() - started,
                cost_usd=cost_usd(served_by, usage, self.prices),
            )
        raise AssertionError("unreachable")

    def _complete(
        self,
        client: openai.OpenAI,
        messages: list[ChatCompletionMessageParam],
        json_schema: dict[str, object],
    ) -> tuple[str, str, Usage]:
        while True:
            if self._json_schema_supported:
                response_format: ResponseFormatJSONSchema | ResponseFormatJSONObject = {
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": json_schema, "strict": True},
                }
                sent = messages
            else:
                response_format = {"type": "json_object"}
                schema_note = json.dumps(json_schema)
                first = messages[0]
                sent = [
                    {
                        "role": "system",
                        "content": f"{first['content']}\n\nReply with a JSON object that "
                        f"matches this JSON schema:\n{schema_note}",
                    },
                    *messages[1:],
                ]
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=sent,
                    response_format=response_format,
                    max_completion_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort or openai.omit,
                )
            except openai.BadRequestError as exc:
                if self._json_schema_supported and _about_response_format(exc.message):
                    # Some backends don't accept json_schema; JSON mode works nearly everywhere.
                    log.warning("llm.json_schema_rejected", provider=self.provider, error=str(exc))
                    self._json_schema_supported = False
                    continue
                raise LLMError(f"{self.provider} rejected the request: {exc.message}") from exc
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                raise LLMConfigError(
                    f"{self.provider} rejected the API key ({self._key_name}): {exc.message}"
                ) from exc
            except openai.NotFoundError as exc:
                raise LLMConfigError(
                    f"{self.provider} has no model {self.model!r} (set FRA_LLM_MODEL): "
                    f"{exc.message}"
                ) from exc
            except openai.APIStatusError as exc:
                raise LLMError(f"{self.provider} error {exc.status_code}: {exc.message}") from exc
            except openai.APIConnectionError as exc:
                raise LLMError(f"{self.provider} unreachable: {exc}") from exc
            break

        if not response.choices:
            raise LLMError(f"{self.provider} returned no choices")
        choice = response.choices[0]
        if choice.finish_reason == "content_filter" or choice.message.refusal:
            raise LLMRefusalError(f"{self.model} declined the request")
        if choice.finish_reason == "length":
            raise LLMError(f"response truncated at max_tokens={self.max_tokens}")
        u = response.usage
        usage = Usage(
            input_tokens=u.prompt_tokens if u else 0,
            output_tokens=u.completion_tokens if u else 0,
        )
        return choice.message.content or "", response.model or self.model, usage


def _about_response_format(message: str) -> bool:
    text = message.lower()
    return any(word in text for word in ("response_format", "json_schema", "schema", "json"))
