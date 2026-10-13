from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from filings_agent.llm import CachedLLM, LLMConfigError, LLMError, LLMRefusalError, Usage
from filings_agent.llm.anthropic import FALLBACK_BETA, AnthropicLLM
from filings_agent.llm.pricing import cost_usd
from tests.conftest import FakeLLM


class Answer(BaseModel):
    value: float
    label: str | None


def test_cache_serves_repeat_calls_with_original_usage(tmp_path: Path) -> None:
    inner = FakeLLM(Answer(value=1.5, label="x"))
    llm = CachedLLM(inner, tmp_path)
    first = llm.generate(system="s", prompt="p", schema=Answer)
    second = llm.generate(system="s", prompt="p", schema=Answer)
    assert len(inner.calls) == 1
    assert not first.cached and second.cached
    assert second.output == first.output
    assert (second.usage, second.latency_s, second.model) == (
        first.usage,
        first.latency_s,
        first.model,
    )
    llm.generate(system="s", prompt="other prompt", schema=Answer)
    assert len(inner.calls) == 2, "a different prompt is a different cache entry"


def test_cache_recovers_from_a_corrupt_entry(tmp_path: Path) -> None:
    inner = FakeLLM(Answer(value=2.0, label=None))
    llm = CachedLLM(inner, tmp_path)
    key = llm.key(system="s", prompt="p", schema=Answer)
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"output": {"value": "not a number"}}')
    result = llm.generate(system="s", prompt="p", schema=Answer)
    assert not result.cached and result.output.value == 2.0
    assert json.loads(path.read_text())["output"]["value"] == 2.0


def test_cache_does_not_store_failures(tmp_path: Path) -> None:
    inner = FakeLLM(LLMError("boom"), Answer(value=3.0, label=None))
    llm = CachedLLM(inner, tmp_path)
    with pytest.raises(LLMError):
        llm.generate(system="s", prompt="p", schema=Answer)
    assert llm.generate(system="s", prompt="p", schema=Answer).output.value == 3.0


def test_pricing() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(4.0 + 2.0)
    assert cost_usd("some-unpriced-model", usage) is None


def _message(text: str, *, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1200, "output_tokens": 300, "cache_read_input_tokens": 0},
    }


def _anthropic(responses: list[httpx2.Response], seen: list[httpx2.Request]) -> anthropic.Anthropic:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return responses.pop(0)

    return anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def test_anthropic_request_and_response_mapping() -> None:
    seen: list[httpx2.Request] = []
    client = _anthropic(
        [httpx2.Response(200, json=_message('{"value": 4.33, "label": "EPS"}'))], seen
    )
    llm = AnthropicLLM(model="claude-opus-5-5", effort="high", client=client)
    result = llm.generate(system="be exact", prompt="extract", schema=Answer)

    assert result.output == Answer(value=4.33, label="EPS")
    assert result.usage == Usage(input_tokens=1200, output_tokens=300)
    assert result.model == "claude-opus-5-5"
    assert result.cost_usd == pytest.approx((1200 * 4 + 300 * 20) / 1e6)
    (request,) = seen
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == "be exact"
    assert body["messages"] == [{"role": "user", "content": "extract"}]
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert llm.id == "anthropic:claude-opus-5-5:effort=high"


def test_anthropic_without_fallbacks_sends_no_beta() -> None:
    seen: list[httpx2.Request] = []
    client = _anthropic([httpx2.Response(200, json=_message('{"value": 1, "label": null}'))], seen)
    AnthropicLLM(client=client, fallbacks=False).generate(system="s", prompt="p", schema=Answer)
    body = json.loads(seen[0].content)
    assert "fallbacks" not in body
    assert FALLBACK_BETA not in seen[0].headers.get("anthropic-beta", "")


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx2.Response(200, json=_message("", stop_reason="refusal")), LLMRefusalError),
        (httpx2.Response(200, json=_message('{"value": 1', stop_reason="max_tokens")), LLMError),
        (
            httpx2.Response(
                400,
                json={"type": "error", "error": {"type": "invalid_request_error", "message": "x"}},
            ),
            LLMError,
        ),
        (
            httpx2.Response(
                401,
                json={"type": "error", "error": {"type": "authentication_error", "message": "x"}},
            ),
            LLMConfigError,
        ),
        (
            httpx2.Response(
                404, json={"type": "error", "error": {"type": "not_found_error", "message": "x"}}
            ),
            LLMConfigError,
        ),
    ],
)
def test_anthropic_failures_raise_llm_errors(
    response: httpx2.Response, error: type[Exception]
) -> None:
    llm = AnthropicLLM(client=_anthropic([response], []))
    with pytest.raises(error):
        llm.generate(system="s", prompt="p", schema=Answer)


def test_anthropic_client_needs_credentials_only_when_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent")
    llm = AnthropicLLM()  # constructing is free: cached runs need no key
    assert llm.id.startswith("anthropic:")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")  # never reached
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        llm.generate(system="s", prompt="p", schema=Answer)
