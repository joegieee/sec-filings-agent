"""Keyword retrieval over filing text: BM25 ranking with SQLite FTS5.

Parsed blocks are grouped into passages (consecutive paragraphs of one section,
up to ``MAX_PASSAGE_CHARS``; a table is its own passage) and indexed with the
Porter stemmer. Every passage keeps its filing, section, page and first block id,
so a search hit is a citation. This needs no model and no network; vector and
hybrid search are planned experiments, to be measured against this baseline.

The index lives in its own SQLite file and is rebuilt per filing, so re-indexing
is idempotent.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from filings_agent.agent.financials import period_label
from filings_agent.extract.models import Metric
from filings_agent.parse.models import Block, ParsedFiling, SectionKind
from filings_agent.storage import StoredFiling

# Bump when chunking changes, so stale indexes are rebuilt.
INDEX_VERSION = 1
MAX_PASSAGE_CHARS = 1_500
MAX_TABLE_CHARS = 4_000
MIN_BLOCK_CHARS = 40  # page furniture and stray labels aren't worth indexing
NEAR_DISTANCE = 6  # extra tokens allowed between query terms in the strictest pass

_SCHEMA = """
CREATE TABLE IF NOT EXISTS indexed (
    accession_number  TEXT PRIMARY KEY,
    source_sha256     TEXT NOT NULL,
    index_version     INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(
    text,
    accession_number UNINDEXED, ticker UNINDEXED, form UNINDEXED, period UNINDEXED,
    report_date UNINDEXED, section UNINDEXED, item UNINDEXED, page_label UNINDEXED,
    block_id UNINDEXED, url UNINDEXED,
    tokenize = 'porter unicode61'
);
"""

_WORD = re.compile(r"[A-Za-z0-9]+(?:[.'][A-Za-z0-9]+)*")
# Words too common in filings to help ranking.
_STOP = frozenset(
    (  # noqa: SIM905
        "a an and are as at be by did do does for from had has have how in is it its of on or "
        "the to was were what when which who why with company company's "
        # Question words that say what to do with the text rather than what it contains.
        "change changed changes driver drivers drove explain reason reasons trend"
    ).split()
)


@dataclass(frozen=True)
class Passage:
    text: str
    accession_number: str
    ticker: str
    form: str
    period: str
    report_date: date
    section: SectionKind
    item: str | None
    page_label: str | None
    block_id: str
    url: str
    score: float = 0.0


def filing_period(form: str, report_date: date) -> str:
    """The filing's own period label: FY2024 for a 10-K, Q3 2025 for a 10-Q."""
    return period_label(form, report_date, Metric.REVENUE)


def chunk(parsed: ParsedFiling) -> Iterator[tuple[str, Block]]:
    """(passage text, first block) pairs, in document order."""
    buffer: list[Block] = []

    def flush() -> Iterator[tuple[str, Block]]:
        if buffer:
            yield "\n".join(b.text for b in buffer), buffer[0]
            buffer.clear()

    for block in parsed.blocks:
        if len(block.text) < MIN_BLOCK_CHARS:
            continue
        if block.kind == "table":
            yield from flush()
            yield block.text[:MAX_TABLE_CHARS], block
            continue
        if buffer and (
            block.section != buffer[0].section
            or sum(len(b.text) for b in buffer) + len(block.text) > MAX_PASSAGE_CHARS
        ):
            yield from flush()
        buffer.append(block)
    yield from flush()


def _terms(text: str) -> list[str]:
    words = [w.lower() for w in _WORD.findall(text)]
    return list(dict.fromkeys(w for w in words if w not in _STOP))


def fts_queries(text: str) -> list[str]:
    """Stricter to looser FTS5 queries for free text: all terms close together, all
    terms anywhere, then any term. Terms are quoted, so no FTS5 syntax gets through."""
    terms = [f'"{t}"' for t in _terms(text)]
    if not terms:
        return []
    if len(terms) == 1:
        return terms
    return [
        f"NEAR({' '.join(terms)}, {NEAR_DISTANCE + len(terms)})",
        " AND ".join(terms),
        " OR ".join(terms),
    ]


class SearchIndex:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SearchIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def is_current(self, parsed: ParsedFiling) -> bool:
        row = self._conn.execute(
            "SELECT source_sha256, index_version FROM indexed WHERE accession_number = ?",
            (parsed.accession_number,),
        ).fetchone()
        return (
            row is not None
            and row["source_sha256"] == parsed.source_sha256
            and row["index_version"] == INDEX_VERSION
        )

    def add_filing(self, filing: StoredFiling, parsed: ParsedFiling) -> int:
        """(Re)index one filing; returns the number of passages."""
        if filing.report_date is None:
            raise ValueError(f"{filing.accession_number} has no report date")
        period = filing_period(filing.form, filing.report_date)
        rows = [
            (
                text,
                filing.accession_number,
                filing.ticker,
                filing.form,
                period,
                filing.report_date.isoformat(),
                first.section.value,
                first.item,
                first.page_label,
                first.id,
                parsed.citation_url(first),
            )
            for text, first in chunk(parsed)
        ]
        with self._conn:
            self._conn.execute(
                "DELETE FROM passages WHERE accession_number = ?", (filing.accession_number,)
            )
            self._conn.executemany(
                "INSERT INTO passages (text, accession_number, ticker, form, period, report_date,"
                " section, item, page_label, block_id, url) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO indexed VALUES (?, ?, ?)",
                (filing.accession_number, parsed.source_sha256, INDEX_VERSION),
            )
        return len(rows)

    def count(self) -> int:
        (n,) = self._conn.execute("SELECT count(*) FROM passages").fetchone()
        return int(n)

    def search(
        self,
        query: str,
        *,
        tickers: Sequence[str] | None = None,
        section: SectionKind | None = None,
        period: str | None = None,
        limit: int = 5,
    ) -> list[Passage]:
        """Best passages for ``query``: strict matches first, looser ones fill the rest."""
        hits: dict[int, Passage] = {}
        for match in fts_queries(query):
            if len(hits) >= limit:
                break
            for rowid, passage in self._search(match, tickers, section, period, limit):
                if len(hits) < limit:
                    hits.setdefault(rowid, passage)
        return list(hits.values())

    def _search(
        self,
        match: str,
        tickers: Sequence[str] | None,
        section: SectionKind | None,
        period: str | None,
        limit: int,
    ) -> list[tuple[int, Passage]]:
        where = ["passages MATCH ?"]
        params: list[object] = [match]
        if tickers:
            where.append(f"ticker IN ({', '.join('?' * len(tickers))})")
            params.extend(t.upper() for t in tickers)
        if section is not None:
            where.append("section = ?")
            params.append(section.value)
        if period is not None:
            where.append("period = ?")
            params.append(period)
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT rowid, *, bm25(passages) AS score FROM passages WHERE {' AND '.join(where)} "
            "ORDER BY score, report_date DESC, block_id LIMIT ?",
            params,
        ).fetchall()
        return [
            (
                r["rowid"],
                Passage(
                    text=r["text"],
                    accession_number=r["accession_number"],
                    ticker=r["ticker"],
                    form=r["form"],
                    period=r["period"],
                    report_date=date.fromisoformat(r["report_date"]),
                    section=SectionKind(r["section"]),
                    item=r["item"],
                    page_label=r["page_label"],
                    block_id=r["block_id"],
                    url=r["url"],
                    score=-float(r["score"]),  # FTS5's bm25() is lower-is-better
                ),
            )
            for r in rows
        ]
