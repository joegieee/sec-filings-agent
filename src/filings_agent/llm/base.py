"""Provider-neutral interface for "prompt in, validated Pydantic object out" calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


class LLMError(Exception):
    """A call failed permanently (after the client's own retries)."""


class LLMRefusalError(LLMError):
    """The model declined to answer."""


class LLMConfigError(LLMError):
    """Every call will fail until the setup is fixed (no credentials, unknown model...)."""


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    output: T
    usage: Usage
    model: str  # the model that actually served the request
    latency_s: float
    cost_usd: float | None = None  # None when the model's price is unknown
    cached: bool = False


class StructuredLLM(Protocol):
    @property
    def id(self) -> str:
        """Everything that changes the output (provider, model, settings); used as a cache key."""
        ...

    def generate[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T]
    ) -> LLMResult[T]: ...


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def json_text(content: str) -> str:
    """The JSON in a reply, without reasoning blocks or Markdown fences around it."""
    return _FENCE.sub("", _THINK.sub("", content).strip())
