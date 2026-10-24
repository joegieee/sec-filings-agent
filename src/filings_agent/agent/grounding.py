"""Render arithmetic placeholders and flag numbers absent from retrieved sources.

Matching uses numeric magnitudes at the written precision, with scale alternatives
for bare numbers. It does not verify sign, units, period or claim-to-citation
alignment. Dates, period labels, years and small plain counts are skipped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from filings_agent.agent.calculator import CalcError, evaluate
from filings_agent.extract.citations import numbers_in

_PLACEHOLDER = re.compile(r"\{=\s*([^{}]+?)\s*(?::\s*([^{}:]*))?\}")
_FORMAT_SPEC = re.compile(r"^,?(?:\.\d)?[f%]?$")

_SCALES = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
    "t": 1e12,
}
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATES = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?",
    re.IGNORECASE,
)
# Year-to-date labels such as "9M 2025", as the tools write periods; "$9M" stays a number.
_PERIODS = re.compile(r"(?<![\w$.])(?:3|6|9|12)M\s*\d{4}\b", re.IGNORECASE)
_NUMBER = re.compile(
    r"(?<![\w.])(?P<dollar>\$\s?)?(?P<sign>-|\N{MINUS SIGN})?(?P<int>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<frac>\.\d+)?(?:\s?(?P<pct>%|percent\b)|\s?(?P<scale>thousand|million|billion|trillion"
    r"|mn|bn|tn|[kmbt]\b))?(?![\w.]*\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Mention:
    """A number as written in text: its literal, decimals shown, and scale."""

    raw: str
    literal: float
    decimals: int
    scale: float | None  # None when no scale word was written
    percent: bool
    dollar: bool

    def matches(self, value: float, *, scales: Iterable[float] | None = None) -> bool:
        """Whether ``value`` rounds to this mention at the precision it was written."""
        half = 0.5 * 10 ** (-self.decimals)
        for scale in scales or ([self.scale] if self.scale else [1.0]):
            if abs(abs(value) / scale - abs(self.literal)) <= half * (1 + 1e-9) + 1e-12:
                return True
        return False


def mentions(text: str) -> list[Mention]:
    """Numbers written in ``text``, skipping dates, period labels, years and small counts."""
    found = []
    for m in _NUMBER.finditer(_PERIODS.sub(" ", _DATES.sub(" ", text))):
        digits = m.group("int").replace(",", "")
        frac = m.group("frac") or ""
        literal = float(digits + frac)
        scale_word = (m.group("scale") or "").lower()
        scale = _SCALES.get(scale_word)
        pct, dollar = bool(m.group("pct")), bool(m.group("dollar"))
        plain = not (pct or dollar or scale or frac)
        if plain and (1900 <= literal <= 2100 or literal <= 10):
            continue  # a year, or a count such as "3 banks"
        if m.group("sign"):
            literal = -literal
        found.append(
            Mention(
                raw=m.group(0).strip(),
                literal=literal,
                decimals=len(frac) - 1 if frac else 0,
                scale=scale,
                percent=pct,
                dollar=dollar,
            )
        )
    return found


@dataclass
class Rendered:
    text: str
    used: list[str] = field(default_factory=list)  # source ids the placeholders read
    values: list[float] = field(default_factory=list)  # what they evaluated to
    errors: list[str] = field(default_factory=list)


def render(draft: str, variables: Mapping[str, float]) -> Rendered:
    """Evaluate ``{=EXPR:FMT}`` placeholders against source values."""
    out = Rendered(text="")
    ids = re.compile(r"\b[FC]\d+\b")

    def substitute(m: re.Match[str]) -> str:
        expression, spec = m.group(1), (m.group(2) or "").strip()
        try:
            value = evaluate(expression, variables)
        except CalcError as exc:
            out.errors.append(str(exc))
            return m.group(0)
        if spec and not _FORMAT_SPEC.match(spec):
            out.errors.append(f"unsupported format {spec!r} in {m.group(0)}")
            spec = ""
        out.used.extend(i for i in ids.findall(expression) if i not in out.used)
        out.values.append(value)
        return format(value, spec or ",.2f")

    out.text = _PLACEHOLDER.sub(substitute, draft)
    return out


def ungrounded(draft: str, values: Iterable[float], passages: Iterable[str]) -> list[str]:
    """Numbers the model wrote itself (outside placeholders) that no source supports."""
    known = list(values)
    printed = [n for text in passages for n in numbers_in(text)]
    missing = []
    for mention in mentions(_PLACEHOLDER.sub(" ", draft)):
        scales = None if mention.scale else (1.0, 1e3, 1e6, 1e9)
        if any(mention.matches(v, scales=scales) for v in known):
            continue
        if any(mention.matches(n, scales=(1.0,)) for n in printed):
            continue
        missing.append(mention.raw)
    return missing
