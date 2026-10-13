from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import httpx2
import openai
import pytest
from pydantic import BaseModel

from filings_agent.config import Settings
from filings_agent.extract.models import FinancialsExtraction
from filings_agent.llm import CachedLLM, LLMConfigError, LLMError, LLMRefusalError
from filings_agent.llm.budget import BudgetedLLM
from filings_agent.llm.factory import GEMINI_MODEL, build_llm
from filings_agent.llm.ollama import OllamaLLM
from filings_agent.llm.openai_compat import GEMINI_BASE_URL, OpenAICompatibleLLM
from filings_agent.llm.schema import inline_json_schema


class Item(BaseModel):
    when: str
    amount: float | None


class Answer(BaseModel):
    label: str
    item: Item | None


GOOD = '{"label": "x", "item": {"when": "2024", "amount": 1.5}}'


# --- schema ------------------------------------------------------------------------


def test_inline_schema_is_flat_strict_and_complete() -> None:
    schema = inline_json_schema(FinancialsExtraction)
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text and '"title"' not in text
    assert '"format"' not in text

    def objects(node: Any) -> list[dict[str, Any]]:
        found = []
        if isinstance(node, dict):
            if node.get("type") == "object":
                found.append(node)
            for value in node.values():
                found += objects(value)
        elif isinstance(node, list):
            for value in node:
                found += objects(value)
        return found

    for obj in objects(schema):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    revenue = schema["properties"]["revenue"]["anyOf"]
    assert {"type": "null"} in revenue
    figure = next(v for v in revenue if v.get("type") == "object")
    assert "YYYY-MM-DD" in figure["properties"]["period_end"]["description"]


def test_inline_schema_for_grammar_backends_keeps_dates_and_caps_strings() -> None:
    schema = inline_json_schema(FinancialsExtraction, keep_formats=True, max_string_length=500)
    figure = next(v for v in schema["properties"]["revenue"]["anyOf"] if v.get("type") == "object")
    props = figure["properties"]
    assert props["period_end"]["format"] == "date"
    assert "maxLength" not in props["period_end"]
    assert props["quote"]["maxLength"] == 500
    assert "maxLength" not in props["scale"]  # an enum is already bounded


# --- OpenAI-compatible (Gemini) ----------------------------------------------------


def _completion(content: str, *, finish: str = "stop", model: str = GEMINI_MODEL) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
    }


def _error(status: int, message: str) -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"message": message, "code": status}})


def _gemini(
    responses: list[httpx2.Response], seen: list[dict[str, Any]], **kwargs: Any
) -> OpenAICompatibleLLM:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return responses.pop(0)

    client = openai.OpenAI(
        api_key="test",
        base_url=GEMINI_BASE_URL,
        max_retries=0,
        http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    options: dict[str, Any] = {
        "provider": "gemini",
        "model": GEMINI_MODEL,
        "base_url": GEMINI_BASE_URL,
        "api_key": "test",
        "key_name": "GEMINI_API_KEY",
        "client": client,
    }
    return OpenAICompatibleLLM(**{**options, **kwargs})


def test_requests_a_strict_json_schema_and_maps_the_result() -> None:
    seen: list[dict[str, Any]] = []
    llm = _gemini(
        [httpx2.Response(200, json=_completion(GOOD))],
        seen,
        reasoning_effort="low",
        prices=(0.5, 3.0),
    )
    result = llm.generate(system="sys", prompt="go", schema=Answer)
    assert result.output.item is not None and result.output.item.amount == 1.5
    assert (result.usage.input_tokens, result.usage.output_tokens) == (1000, 100)
    assert result.model == GEMINI_MODEL
    assert result.cost_usd == pytest.approx((1000 * 0.5 + 100 * 3.0) / 1e6)
    (body,) = seen
    assert body["model"] == GEMINI_MODEL
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "$ref" not in json.dumps(body["response_format"])
    assert body["reasoning_effort"] == "low"
    assert llm.id == f"gemini:{GEMINI_MODEL}:effort=low"


def test_unpriced_model_has_no_cost() -> None:
    llm = _gemini([httpx2.Response(200, json=_completion(GOOD))], [])
    result = llm.generate(system="s", prompt="p", schema=Answer)
    assert result.cost_usd is None
    assert llm.id == f"gemini:{GEMINI_MODEL}"


def test_falls_back_to_json_mode_when_json_schema_is_rejected() -> None:
    seen: list[dict[str, Any]] = []
    llm = _gemini(
        [
            _error(400, "Invalid response_format: json_schema is not supported"),
            httpx2.Response(200, json=_completion(GOOD)),
            httpx2.Response(200, json=_completion(GOOD)),
        ],
        seen,
    )
    llm.generate(system="sys", prompt="p", schema=Answer)
    llm.generate(system="sys", prompt="p", schema=Answer)
    assert [b["response_format"]["type"] for b in seen] == [
        "json_schema",
        "json_object",
        "json_object",  # remembered for the rest of the run
    ]
    assert '"additionalProperties"' in seen[1]["messages"][0]["content"]


def test_other_bad_requests_are_errors() -> None:
    llm = _gemini([_error(400, "prompt is too long")], [])
    with pytest.raises(LLMError, match="too long"):
        llm.generate(system="s", prompt="p", schema=Answer)


def test_invalid_output_gets_one_repair_attempt() -> None:
    seen: list[dict[str, Any]] = []
    fenced = f"<think>reasoning</think>```json\n{GOOD}\n```"
    llm = _gemini(
        [
            httpx2.Response(200, json=_completion('{"label": "x"}')),
            httpx2.Response(200, json=_completion(fenced)),
        ],
        seen,
    )
    result = llm.generate(system="s", prompt="p", schema=Answer)
    assert result.output.label == "x"
    assert result.usage.input_tokens == 2000, "both attempts are counted"
    assert seen[1]["messages"][-1]["role"] == "user"
    assert "does not match" in seen[1]["messages"][-1]["content"]

    bad = _gemini([httpx2.Response(200, json=_completion("nope"))] * 2, [])
    with pytest.raises(LLMError, match="validation"):
        bad.generate(system="s", prompt="p", schema=Answer)


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (_error(401, "API key not valid"), LLMConfigError),
        (_error(404, "model not found"), LLMConfigError),
        (_error(429, "quota exceeded"), LLMError),
        (httpx2.Response(200, json=_completion(GOOD, finish="length")), LLMError),
        (httpx2.Response(200, json=_completion("", finish="content_filter")), LLMRefusalError),
    ],
)
def test_failures(response: httpx2.Response, error: type[Exception]) -> None:
    with pytest.raises(error):
        _gemini([response], []).generate(system="s", prompt="p", schema=Answer)


def test_missing_key_is_a_config_error() -> None:
    llm = OpenAICompatibleLLM(
        provider="gemini",
        model=GEMINI_MODEL,
        base_url=GEMINI_BASE_URL,
        api_key=None,
        key_name="GEMINI_API_KEY",
    )
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        llm.generate(system="s", prompt="p", schema=Answer)


# --- Ollama ------------------------------------------------------------------------


def _ollama(handler: Any, **kwargs: Any) -> OllamaLLM:
    return OllamaLLM(client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def _chat(content: str, prompt_tokens: int = 3000, done: str = "stop") -> dict[str, Any]:
    return {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done,
        "prompt_eval_count": prompt_tokens,
        "eval_count": 250,
    }


def test_ollama_request_and_result() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_chat(GOOD))

    llm = _ollama(handler, num_ctx=16_384)
    result = llm.generate(system="s", prompt="p", schema=Answer)
    assert result.output.label == "x"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (3000, 250)
    assert result.cost_usd == 0.0
    (body,) = seen
    assert body["options"] == {"num_ctx": 16_384, "temperature": 0}
    assert body["think"] is False and body["stream"] is False
    assert body["format"]["additionalProperties"] is False
    assert body["format"]["properties"]["label"]["maxLength"] == 1_000
    assert llm.id == "ollama:qwen3:8b:ctx=16384:think=false"


def test_ollama_refuses_a_truncated_prompt() -> None:
    llm = _ollama(lambda r: httpx.Response(200, json=_chat(GOOD, prompt_tokens=4096)), num_ctx=4096)
    with pytest.raises(LLMError, match="FRA_OLLAMA_NUM_CTX"):
        llm.generate(system="s", prompt="p", schema=Answer)


def test_ollama_refuses_a_prompt_too_big_to_fit_before_sending() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be sent")

    llm = _ollama(handler, num_ctx=4096)
    with pytest.raises(LLMError, match="may not fit"):
        llm.generate(system="s", prompt="x" * 10_000, schema=Answer)


def test_ollama_repairs_invalid_output_once() -> None:
    seen: list[dict[str, Any]] = []
    replies = iter(['{"label": "x"}', GOOD])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_chat(next(replies)))

    result = _ollama(handler).generate(system="s", prompt="p", schema=Answer)
    assert result.output.label == "x"
    assert result.usage.input_tokens == 6000
    assert [m["role"] for m in seen[1]["messages"]] == ["system", "user", "assistant", "user"]

    always_bad = _ollama(lambda r: httpx.Response(200, json=_chat('{"label": "x"}')))
    with pytest.raises(LLMError, match="failed validation"):
        always_bad.generate(system="s", prompt="p", schema=Answer)


def test_ollama_setup_errors() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMConfigError, match="ollama serve"):
        _ollama(refuse).generate(system="s", prompt="p", schema=Answer)
    missing = _ollama(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMConfigError, match="ollama pull qwen3:8b"):
        missing.generate(system="s", prompt="p", schema=Answer)


def test_ollama_retries_without_think_for_models_that_lack_it() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "model does not support thinking"})
        return httpx.Response(200, json=_chat(GOOD))

    _ollama(handler).generate(system="s", prompt="p", schema=Answer)
    assert ["think" in b for b in seen] == [True, False]


# --- factory -----------------------------------------------------------------------


def _settings(tmp_path: Path, **env: Any) -> Settings:
    return Settings(
        _env_file=None,
        sec_user_agent="t t@example.com",
        data_dir=tmp_path,
        **env,
    )


def test_factory_defaults_to_gemini_and_caches(tmp_path: Path) -> None:
    llm = build_llm(_settings(tmp_path))
    assert isinstance(llm, CachedLLM)
    assert llm.id == f"gemini:{GEMINI_MODEL}"


def test_gemini_defaults_protect_the_free_tier_quota(tmp_path: Path) -> None:
    capped = build_llm(_settings(tmp_path, llm_cache=False))
    # Gemini's free tier is 20 requests a day, so it gets a daily cap by default.
    assert isinstance(capped, BudgetedLLM) and capped.remaining() == 20
    llm = capped._inner
    assert isinstance(llm, OpenAICompatibleLLM)
    # A long timeout and a single retry: retrying slow calls burns the daily quota.
    assert (llm._timeout_s, llm._max_retries) == (900.0, 1)
    tuned = build_llm(_settings(tmp_path, llm_cache=False, llm_max_retries=3, llm_daily_limit=2))
    assert isinstance(tuned, BudgetedLLM) and tuned.remaining() == 2
    assert isinstance(tuned._inner, OpenAICompatibleLLM) and tuned._inner._max_retries == 3
    assert not isinstance(
        build_llm(_settings(tmp_path, llm_cache=False), provider="ollama"), BudgetedLLM
    )


def test_factory_providers(tmp_path: Path) -> None:
    settings = _settings(tmp_path, llm_cache=False)
    assert build_llm(settings, provider="ollama").id == "ollama:qwen3:8b:ctx=16384:think=false"
    assert build_llm(settings, provider="anthropic").id == "anthropic:claude-opus-5-5:effort=high"
    assert build_llm(settings, model="gemini-x", effort="max").id == "gemini:gemini-x:effort=high"
    with pytest.raises(LLMConfigError, match="FRA_LLM_BASE_URL"):
        build_llm(settings, provider="openai_compatible")
    routed = _settings(
        tmp_path, llm_cache=False, llm_base_url="https://openrouter.ai/api/v1", llm_model="m"
    )
    assert build_llm(routed, provider="openai_compatible").id == "openai_compatible:m"


def test_settings_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "  ")
    assert _settings(tmp_path).gemini_api_key is None
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    key = _settings(tmp_path).gemini_api_key
    assert key is not None and key.get_secret_value() == "abc"
    with pytest.raises(ValueError, match="FRA_LLM_PRICE_OUTPUT"):
        _settings(tmp_path, llm_price_input=0.3)
