"""Inline-XBRL / HTML filing documents to plain-text blocks.

The LLM must see the filing as a reader would, not the XBRL tagging behind it
(otherwise the extraction eval against XBRL would be circular). So:

* the hidden ``ix:header`` (all hidden facts and contexts) and anything styled
  ``display:none`` is dropped;
* ``ix:*`` elements are unwrapped: only their displayed text survives;
* tables become one block each, one line per row, cells joined with `` | ``.

Page breaks are counted so each block knows its physical page, and each page's
printed number is recovered from its header or footer where possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import html
from lxml.html import HtmlElement

_SKIP_TAGS = frozenset({"script", "style", "head", "title", "ix:header"})
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "br", "hr",
        "section", "article", "body", "center", "blockquote", "pre", "dl", "dt", "dd",
        "tr", "td", "th", "caption",
    }
)  # fmt: skip
_WS = re.compile(r"\s+")
# Cells that are only currency or bracket fragments: Workiva splits "$ 1,234" and
# "(56 )" into separate cells, which reads badly. Glue them back onto the number.
_LEADING_FRAGMENT = re.compile(r"^(?:\$|€|£|\(|\$\s*\()$")
_TRAILING_FRAGMENT = re.compile(r"^(?:\)|%|\)%|%\)|bps)$")


@dataclass
class RawBlock:
    kind: str  # "text" | "table"
    text: str
    page: int
    anchor: str | None


@dataclass
class _Walker:
    blocks: list[RawBlock] = field(default_factory=list)
    page: int = 1
    anchor: str | None = None
    _buf: list[str] = field(default_factory=list)
    _buf_anchor: str | None = None

    def add_text(self, text: str | None) -> None:
        if not text:
            return
        if not self._buf and text.strip():
            self._buf_anchor = self.anchor
        if self._buf or text.strip():
            self._buf.append(text)

    def flush(self) -> None:
        text = normalize("".join(self._buf))
        self._buf.clear()
        if text:
            self.blocks.append(RawBlock("text", text, self.page, self._buf_anchor))

    def walk(self, el: HtmlElement) -> None:
        tag = el.tag
        if not isinstance(tag, str) or tag in _SKIP_TAGS or _hidden(el):
            return
        style = (el.get("style") or "").lower().replace(" ", "")
        if "page-break-before:always" in style:
            self.flush()
            self.page += 1
        element_id = el.get("id")
        is_block = tag in _BLOCK_TAGS
        if is_block:
            self.flush()
        if element_id and not tag.startswith("ix:"):
            self.anchor = element_id
        if tag == "table":
            self.flush()
            self._table(el)
        else:
            self.add_text(el.text)
            for child in el:
                self.walk(child)
                self.add_text(child.tail)
        if is_block:
            self.flush()
        if "page-break-after:always" in style:
            self.flush()
            self.page += 1

    def _table(self, table: HtmlElement) -> None:
        anchor = self.anchor
        rows: list[list[str]] = []
        for tr in table.iter("tr"):
            if _hidden(tr):
                continue
            cells = [_text_of(td) for td in tr if td.tag in ("td", "th") and not _hidden(td)]
            cells = _merge_fragments([c for c in cells if c])
            if cells:
                rows.append(cells)
            for el in tr.iter():
                el_id = el.get("id") if isinstance(el.tag, str) else None
                if el_id and not str(el.tag).startswith("ix:"):
                    self.anchor = el_id
        if not rows:
            return
        if len(rows) == 1 and len(rows[0]) <= 2:
            # Layout tables (a footnote marker beside its text, a page footer) read as prose.
            self.blocks.append(RawBlock("text", " ".join(rows[0]), self.page, anchor))
            return
        text = "\n".join(" | ".join(row) for row in rows)
        self.blocks.append(RawBlock("table", text, self.page, anchor))


def normalize(text: str) -> str:
    return _WS.sub(" ", text.replace("\xa0", " ").replace("​", "")).strip()


def _hidden(el: HtmlElement) -> bool:
    style = el.get("style")
    return bool(style) and "display:none" in style.lower().replace(" ", "")


def _text_of(el: HtmlElement) -> str:
    parts: list[str] = []

    def collect(node: HtmlElement) -> None:
        if not isinstance(node.tag, str) or node.tag in _SKIP_TAGS or _hidden(node):
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            collect(child)
            if child.tail:
                parts.append(child.tail)
        if node.tag in ("p", "div", "br"):
            parts.append(" ")

    collect(el)
    return normalize("".join(parts))


def _merge_fragments(cells: list[str]) -> list[str]:
    merged: list[str] = []
    pending = ""
    for cell in cells:
        if _LEADING_FRAGMENT.match(cell):
            pending = f"{pending}{cell}"
            continue
        if pending:
            cell = f"{pending} {cell}" if pending.endswith("$") else f"{pending}{cell}"
            pending = ""
        if merged and _TRAILING_FRAGMENT.match(cell):
            merged[-1] += cell
        else:
            merged.append(cell)
    if pending:
        merged.append(pending)
    return merged


def html_to_blocks(content: bytes) -> list[RawBlock]:
    """Flatten an HTML filing into text and table blocks, in document order."""
    root = html.fromstring(content)
    walker = _Walker()
    walker.walk(root)
    walker.flush()
    return walker.blocks


# --- printed page numbers -------------------------------------------------------

_YEAR_OR_FORM = re.compile(r"\b(?:19|20)\d{2}\b|\b10-[KQ]\b|\bForm\b", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"(?<![\w.,$])(\d{1,3})(?![\w.,%])")


def page_label(text: str) -> str | None:
    """The printed page number in a page header/footer block, e.g. ``"Bank of America 86"``."""
    if len(text) > 80:
        return None
    numbers = _PAGE_NUMBER.findall(_YEAR_OR_FORM.sub(" ", text))
    return numbers[0] if len(numbers) == 1 else None


def find_page_labels(blocks: list[RawBlock]) -> tuple[dict[int, str], set[int]]:
    """Map physical page -> printed page number, from each page's first/last blocks.

    Returns the labels and the indexes of the blocks they were read from (page
    furniture, which callers drop from the text).
    """
    by_page: dict[int, list[int]] = {}
    for i, block in enumerate(blocks):
        by_page.setdefault(block.page, []).append(i)
    candidates: dict[int, list[tuple[int, int]]] = {}  # page -> [(block index, number)]
    for page, indexes in by_page.items():
        # Footers are more reliable than headers, so check the page's tail first.
        for i in dict.fromkeys([*indexes[::-1][:2], *indexes[:2]]):
            label = page_label(blocks[i].text)
            if label is not None:
                candidates.setdefault(page, []).append((i, int(label)))

    def consistent(page: int, number: int) -> bool:
        # A real page number advances in step with its neighbours; a stray
        # "Note 3" heading does not.
        return any(
            n - number == other - page
            for other in range(page - 3, page + 4)
            if other != page
            for _, n in candidates.get(other, [])
        )

    labels: dict[int, str] = {}
    furniture: set[int] = set()
    for page, found in candidates.items():
        for i, number in found:
            if consistent(page, number):
                labels[page] = str(number)
                furniture.add(i)
                break
    return labels, furniture
