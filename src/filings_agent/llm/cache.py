"""Cache validated responses by provider ID, prompts and output schema.

Hits make no provider call but retain the original usage, cost and latency for
comparison with uncached runs. Changed keys and failed calls require new requests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from filings_agent.fsutil import atomic_write
from filings_agent.llm.base import LLMResult, StructuredLLM, Usage


class _Entry(BaseModel):
    output: dict[str, object]
    usage: Usage
    model: str
    latency_s: float
    cost_usd: float | None = None


class CachedLLM:
    def __init__(self, inner: StructuredLLM, cache_dir: Path) -> None:
        self._inner = inner
        self._dir = cache_dir

    @property
    def id(self) -> str:
        return self._inner.id

    def key(self, *, system: str, prompt: str, schema: type[BaseModel]) -> str:
        material = json.dumps(
            {
                "llm": self._inner.id,
                "system": system,
                "prompt": prompt,
                "schema": schema.model_json_schema(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def generate[T: BaseModel](self, *, system: str, prompt: str, schema: type[T]) -> LLMResult[T]:
        path = self._path(self.key(system=system, prompt=prompt, schema=schema))
        if path.is_file():
            try:
                entry = _Entry.model_validate_json(path.read_bytes())
                return LLMResult(
                    output=schema.model_validate(entry.output),
                    usage=entry.usage,
                    model=entry.model,
                    latency_s=entry.latency_s,
                    cost_usd=entry.cost_usd,
                    cached=True,
                )
            except ValidationError:
                path.unlink()  # stale or corrupt entry: fall through and refresh it
        result = self._inner.generate(system=system, prompt=prompt, schema=schema)
        entry = _Entry(
            output=result.output.model_dump(mode="json"),
            usage=result.usage,
            model=result.model,
            latency_s=result.latency_s,
            cost_usd=result.cost_usd,
        )
        atomic_write(path, entry.model_dump_json().encode())
        return result

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"
