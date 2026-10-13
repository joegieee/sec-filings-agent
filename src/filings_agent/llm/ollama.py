"""Local open models via Ollama's native API (no key, no cost, nothing leaves the machine).

The native ``/api/chat`` endpoint is used rather than Ollama's OpenAI-compatible
one because it lets us set the context window. Ollama's default window can be
smaller than an extraction prompt, and it truncates silently, so ``num_ctx`` is
set explicitly. A prompt that may not fit is refused before it is sent (from a
conservative size estimate), and the reported prompt size is checked afterwards.

The schema goes to Ollama as a grammar, so dates are kept constrained and strings
capped; an invalid reply still gets one repair attempt, as with the other providers.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from filings_agent.llm.base import LLMConfigError, LLMError, LLMResult, Usage, json_text
from filings_agent.llm.schema import inline_json_schema

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
# Numbers and table separators tokenize densely; real filings measure ~3 chars per token,
# so 2.5 overestimates the prompt and errs towards refusing rather than truncating.
CHARS_PER_TOKEN = 2.5
OUTPUT_RESERVE_TOKENS = 2_048
MAX_STRING_LENGTH = 1_000

log = structlog.get_logger(__name__)


class OllamaLLM:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        num_ctx: int = 16_384,
        think: bool | None = False,
        timeout_s: float = 900.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.think = think
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    @property
    def id(self) -> str:
        think = "" if self.think is None else f":think={str(self.think).lower()}"
        return f"ollama:{self.model}:ctx={self.num_ctx}{think}"

    def generate[T: BaseModel](self, *, system: str, prompt: str, schema: type[T]) -> LLMResult[T]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        json_schema = inline_json_schema(
            schema, keep_formats=True, max_string_length=MAX_STRING_LENGTH
        )
        usage = Usage()
        started = time.perf_counter()
        for attempt in range(2):
            content, served_by, call_usage = self._chat(messages, json_schema)
            usage = Usage(
                input_tokens=usage.input_tokens + call_usage.input_tokens,
                output_tokens=usage.output_tokens + call_usage.output_tokens,
            )
            try:
                output = schema.model_validate_json(json_text(content))
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMError(f"structured output failed validation: {exc}") from exc
                log.warning("llm.invalid_output_retry", provider="ollama", errors=exc.errors())
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
                cost_usd=0.0,
            )
        raise AssertionError("unreachable")

    def _chat(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any]
    ) -> tuple[str, str, Usage]:
        estimate = int(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN)
        if estimate > self.num_ctx - OUTPUT_RESERVE_TOKENS:
            raise LLMError(
                f"prompt (~{estimate} tokens) may not fit the {self.num_ctx}-token context "
                "window with room for the reply; raise FRA_OLLAMA_NUM_CTX"
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "format": json_schema,
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": 0},
        }
        if self.think is not None:
            body["think"] = self.think
        data = self._post(body)

        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        if prompt_tokens >= self.num_ctx:
            raise LLMError(
                f"prompt ({prompt_tokens} tokens) filled the {self.num_ctx}-token context "
                "window and was truncated; raise FRA_OLLAMA_NUM_CTX"
            )
        if data.get("done_reason") == "length":
            raise LLMError("response truncated: the context window ran out while generating")
        content = str((data.get("message") or {}).get("content") or "")
        usage = Usage(input_tokens=prompt_tokens, output_tokens=int(data.get("eval_count") or 0))
        return content, str(data.get("model") or self.model), usage

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/api/chat"
        try:
            response = self._client.post(url, json=body)
        except httpx.ConnectError as exc:
            raise LLMConfigError(
                f"Ollama is not reachable at {self._base_url}: start it with `ollama serve`"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        if response.status_code == 404:
            raise LLMConfigError(
                f"Ollama has no model {self.model!r}: run `ollama pull {self.model}`"
            )
        if response.status_code == 400 and "think" in body and "think" in response.text.lower():
            # Models without a thinking mode can reject the flag; retry without it.
            return self._post({k: v for k, v in body.items() if k != "think"})
        if response.status_code >= 400:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:300]}")
        data: dict[str, Any] = response.json()
        return data
