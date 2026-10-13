from __future__ import annotations

from pathlib import Path

import pytest

from filings_agent.extract.models import FinancialsExtraction
from filings_agent.llm import LLMConfigError, LLMError
from filings_agent.parse import PARSER_VERSION
from filings_agent.pipeline import extract_filings, parse_filings
from filings_agent.storage import Store, StoredFiling
from tests.conftest import FakeLLM
from tests.test_storage import stored_filing

FIXTURE = Path(__file__).parent / "fixtures" / "sample_10k.htm"
EMPTY = FinancialsExtraction.model_validate({m: None for m in FinancialsExtraction.model_fields})


def _filing(store: Store) -> StoredFiling:
    return stored_filing(store, FIXTURE.read_bytes())


def test_parse_is_idempotent_and_versioned(store: Store) -> None:
    filing = _filing(store)
    (first,) = parse_filings(store, [filing])
    assert first.parsed is not None and not first.skipped
    (second,) = parse_filings(store, [filing])
    assert second.skipped and second.parsed == first.parsed
    assert second.parsed.parser_version == PARSER_VERSION
    (forced,) = parse_filings(store, [filing], force=True)
    assert not forced.skipped


def test_parse_redoes_a_stale_or_corrupt_parse(store: Store) -> None:
    filing = _filing(store)
    parse_filings(store, [filing])
    store.parsed_path(filing).write_text('{"parser_version": 0}')
    (outcome,) = parse_filings(store, [filing])
    assert outcome.parsed is not None and not outcome.skipped


def test_parse_reports_missing_documents(store: Store) -> None:
    filing = _filing(store)
    assert filing.raw_path is not None
    filing.raw_path.unlink()
    (outcome,) = parse_filings(store, [filing])
    assert outcome.parsed is None and outcome.error is not None


def test_extract_stores_results_and_reuses_them(store: Store) -> None:
    filing = _filing(store)
    llm = FakeLLM(EMPTY)
    first = extract_filings(store, [filing], llm)
    assert first.ok and first.reused == 0 and len(llm.calls) == 1
    second = extract_filings(store, [filing], llm)
    assert second.reused == 1 and len(llm.calls) == 1
    extract_filings(store, [filing], llm, force=True)
    assert len(llm.calls) == 2


def test_extract_failure_is_recorded_and_retried(store: Store) -> None:
    filing = _filing(store)
    failing = FakeLLM(LLMError("rate limited"))
    report = extract_filings(store, [filing], failing)
    assert not report.ok and report.failed == [(filing.accession_number, "rate limited")]
    stored = store.get_extraction(filing.accession_number, report.extractor)
    assert stored is not None and stored.error == "rate limited"

    working = FakeLLM(EMPTY, model=failing.model)
    retry = extract_filings(store, [filing], working)
    assert retry.ok and len(working.calls) == 1, "a failed attempt is never reused"


def test_extract_stops_on_configuration_errors(store: Store) -> None:
    filing = _filing(store)
    with pytest.raises(LLMConfigError):
        extract_filings(store, [filing], FakeLLM(LLMConfigError("no key")))
    assert store.get_extraction(filing.accession_number, "fake:claude-opus-5-5") is None
