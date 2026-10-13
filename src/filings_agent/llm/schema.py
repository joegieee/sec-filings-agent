"""JSON Schema for providers with partial JSON Schema support (Gemini, Ollama...).

Pydantic emits ``$defs``/``$ref``, titles and string formats; constrained-decoding
backends support these unevenly. This produces a flat, self-contained schema:
references inlined, every property required (nullable ones as ``anyOf [..., null]``),
no extra properties, and string formats moved into the description. The output is
still validated with the Pydantic model, so nothing is lost.

Grammar-based backends (Ollama, llama.cpp) can enforce more than that: with
``keep_formats`` dates stay constrained by the grammar, and ``max_string_length``
caps free-text strings so a looping model stops instead of filling the context.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

_DROP = frozenset({"title", "default", "$defs"})


def inline_json_schema(
    model: type[BaseModel],
    *,
    keep_formats: bool = False,
    max_string_length: int | None = None,
) -> dict[str, Any]:
    schema = model.model_json_schema()
    defs: dict[str, Any] = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return walk({**defs[name], **extra})
        out = {k: walk(v) for k, v in node.items() if k not in _DROP}
        fmt = out.get("format") if keep_formats else out.pop("format", None)
        if fmt is not None:
            hint = "YYYY-MM-DD" if fmt == "date" else fmt
            out["description"] = f"{out.get('description', '')} Format: {hint}.".strip()
        if (
            max_string_length is not None
            and out.get("type") == "string"
            and fmt is None
            and "enum" not in out
        ):
            out["maxLength"] = max_string_length
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    result: dict[str, Any] = walk(schema)
    return result
