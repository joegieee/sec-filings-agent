"""Safe arithmetic for the agent, so the model never does maths itself.

Expressions are parsed with ``ast`` and only numbers, fact references (``F1``),
the four operators, powers and a few named functions are allowed; nothing is
ever passed to ``eval``.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _pct_change(old: float, new: float) -> float:
    return (new - old) / abs(old) * 100


def _cagr(first: float, last: float, years: float) -> float:
    growth: float = (last / first) ** (1 / years)
    return (growth - 1) * 100


FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *xs: math.fsum(xs),
    "avg": lambda *xs: math.fsum(xs) / len(xs),
    "pct_change": _pct_change,
    "cagr": _cagr,
}
MAX_EXPONENT = 100


class CalcError(ValueError):
    """The expression is invalid, references an unknown fact, or can't be evaluated."""


def evaluate(expression: str, variables: Mapping[str, float] | None = None) -> float:
    variables = variables or {}
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"not a valid expression: {expression!r}") from exc

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            if isinstance(node.value, bool):
                raise CalcError(f"unsupported value in {expression!r}")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise CalcError(f"unknown reference {node.id!r}")
            return float(variables[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise CalcError("exponent too large")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and not node.keywords
        ):
            return float(FUNCTIONS[node.func.id](*(walk(a) for a in node.args)))
        raise CalcError(f"unsupported syntax in {expression!r}: {ast.dump(node)[:60]}")

    try:
        result = walk(tree.body)
    except ZeroDivisionError as exc:
        raise CalcError(f"division by zero in {expression!r}") from exc
    except (TypeError, OverflowError, ValueError) as exc:
        if isinstance(exc, CalcError):
            raise
        raise CalcError(f"cannot evaluate {expression!r}: {exc}") from exc
    if isinstance(result, complex) or not math.isfinite(result):
        raise CalcError(f"{expression!r} has no finite real value")
    return result
