# sec-filings-agent

Download SEC 10-Ks and 10-Qs, extract financial figures, and ask questions against the
stored filings. Answers include filing links and evidence; extraction results can be
compared with EDGAR's XBRL company facts.

The starter corpus is JPMorgan, Bank of America and PayPal (`core3`). Two banks and a
fintech exercise an important distinction: a missing bank ratio is expected for PayPal,
but worth investigating for JPMorgan. The company config has 20 names; the parsing work
and evaluation questions so far focus on these three.

This is a CLI project. Ingestion, parsing, extraction, keyword search and the agent loop
are implemented. Live extraction and Q&A accuracy results are still pending.

## Run it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
cp .env.example .env
```

Set `FRA_SEC_USER_AGENT` in `.env` to include your contact email. For the default Gemini
backend, also set `GEMINI_API_KEY`. To use a local model instead, see [Providers](#providers).

```bash
# Download the last three years of filings and XBRL facts from SEC.
uv run filings-agent ingest --set core3
uv run filings-agent status

# Parse and index locally. Neither command calls an LLM.
uv run filings-agent parse --set core3
uv run filings-agent index --set core3

# Calls the configured LLM; saves the answer and tool trace under data/traces/.
uv run filings-agent ask "How did JPMorgan's net income change from 2023 to 2024?" --trace
```

`ask` can use XBRL figures before any LLM extraction has been run. With the default
`--financials auto`, it prefers stored extractions with valid quotes and falls back to
XBRL per metric. Use `--financials xbrl` or `--financials extracted` to choose explicitly.
The extraction selected is the one matching the current model and extractor version.

To extract figures from filing text:

```bash
# Calls the LLM for one annual filing. --tickers and --form narrow the selection.
uv run filings-agent extract --tickers JPM --form 10-K --limit 1
```

Raw HTML, parsed blocks, SQLite metadata and LLM responses live under `data/` (ignored by
Git). Ingestion skips documents whose stored size still matches. Extraction reuses
successful stored results unless `--force` is set. Response-cache hits avoid LLM calls
when the provider ID, prompts and schema match; changed inputs and failed calls may
require new requests. Cached reports retain the original token usage, cost and latency.

## Try the agent without a key

```bash
uv run python examples/offline_agent.py
```

This uses invented Example Bank figures and scripted model replies. It runs the actual
tool lookup, placeholder calculation and answer checks, then shows an unsupported-number
abstention. It makes no network requests and writes no data. It demonstrates the plumbing,
not a model's ability to choose tools or answer questions.

```text
Synthetic example: invented figures and scripted replies; no model calls.
Example Bank's net income rose 20.0% to $120 million in FY2024.
Sources: F2, F1
...
Abstained: True
Reason: numbers not found in any source: $999 million
```

The script prints the tool output and source evidence as well.

## Decisions that matter

Financial periods need separate treatment. Income-statement figures in a 10-Q are for
the quarter; operating cash flow is year to date; balance-sheet figures are at period
end. Labels such as `FY2024`, `Q3 2025` and `9M 2025` preserve that distinction. Quarter
labels currently assume a December fiscal year end. There is no Q4 10-Q.

Section parsing has several fallbacks because the source layouts differ. The cases
covered include PayPal's Item 8 pointer to Item 15, bank item headings embedded in tables,
and JPMorgan's page references into its annual report. The parser tries body headings,
internal page references, then table-of-contents ranges. See
[`parse/sections.py`](src/filings_agent/parse/sections.py) and its
[tests](tests/test_sections.py) before changing those rules.

Search uses SQLite FTS5 with BM25 ranking and Porter stemming. It is a baseline that runs
locally without an embedding model. Tables stay separate from paragraph passages, and
hits retain a filing link and block ID.

The agent has four steps by default and can request up to eight tools per step. Tools
look up figures, compare companies, search passages and evaluate arithmetic. Computed
answers can use placeholders such as `{=pct_change(F1, F2):.1f}%`, evaluated in Python.
The model is prompted to abstain on unsupported questions, forecasts and investment advice.

## Checks and their limits

A citation check means the quote appears in a block the model received and contains the
reported number's magnitude. It does **not** establish that the quote supports the metric,
sign, units or period. Answer checks likewise use numeric matching across retrieved
sources, rather than checking every claim against its cited source. Years and small
plain counts are skipped. These checks catch some unsupported numbers; they do not prove
an answer is correct.

A few other limitations affect how to read the results:

- Extraction scoring records period mismatches separately; the accuracy score currently
  depends on the value match. A correct value with a wrong period can still score correct.
- Stored extractions are keyed by filing and extractor ID. Reusing them does not check
  changes to the raw document or the context-size setting; use `extract --force` after
  those changes.
- Q&A scoring matches numbers and keywords. Its citation precision measures company and
  metric relevance, rather than whether the citation supports the full claim.

## Evaluation

```bash
# Local numeric-presence check; no LLM.
uv run filings-agent eval context --set core3 --markdown

# Calls the LLM as needed; successful stored extractions are reused.
uv run filings-agent eval extraction --set core3 --markdown
uv run filings-agent eval qa --markdown
```

Extraction is scored against XBRL facts from the same accession and reporting period.
The report includes value accuracy (0.5% tolerance; half a cent for EPS), exact matches,
misses, null handling, quote validity and usage. NIM and CET1 lack reference values in
company facts, so their values are not scored there.

“Context recall” checks whether the expected numeric magnitude occurs anywhere in the
selected text at a common scale. It does not check the metric label or table column.
The earlier README reported 234/234 over a 36-filing corpus; the underlying run artifact
is not checked in, so that result is unverified here. Live extraction and Q&A results
have not been published.

The [26 Q&A questions](evals/qa_core3.toml) cover figures, changes, ratios, comparisons,
filing text and abstentions. Numeric references are computed from XBRL; text questions
use keyword groups. Because the agent can also read XBRL, this evaluates tool use and
answer handling separately from extraction. JSON reports are written to `data/evals/`.

## Providers

Set `FRA_LLM_PROVIDER` in `.env`. `--provider`, `--model` and `--effort` override it per run.
Other settings are listed in [`.env.example`](.env.example) and
[`Settings`](src/filings_agent/config.py).

| Backend | Configuration |
|---|---|
| `gemini` (default) | `GEMINI_API_KEY`; default model ID is `gemini-3.8-flash`. |
| `ollama` | Local server; default model is `qwen3:8b`. |
| `openai_compatible` | `FRA_LLM_BASE_URL`, `FRA_LLM_MODEL`, `FRA_LLM_API_KEY`. |
| `anthropic` | `ANTHROPIC_API_KEY`; default model ID is `claude-opus-5-5`. |

These are configured defaults, not a guarantee of model availability or a particular
provider plan. Gemini gets a local daily cap of 20 `generate` calls by default, reset at
midnight Pacific. Set `FRA_LLM_DAILY_LIMIT` to your budget. Repairs and SDK retries can
send additional requests, so this counter is not an exact provider-quota ledger.

The OpenAI-compatible adapter falls back from strict schema to JSON mode when the backend
rejects the format. It and Ollama allow one repair attempt for invalid output. Anthropic
raises an error on validation failure. All three validate replies with Pydantic.

For Ollama, pull the model and start the server, then set `FRA_LLM_PROVIDER=ollama`:

```bash
ollama pull qwen3:8b
ollama serve
```

The native Ollama adapter sets the context window (`FRA_OLLAMA_NUM_CTX`, default 16384),
estimates whether the prompt fits and checks the reported token count. Use this adapter
instead of pointing `openai_compatible` at Ollama, where those controls are absent.
Memory needs depend on the model and context size.

Cost reports use the built-in price table for known Claude models, configured
`FRA_LLM_PRICE_INPUT` / `FRA_LLM_PRICE_OUTPUT` for other hosted models, and zero API cost
for Ollama. An unknown price is reported as unavailable. Check the configured prices
before relying on a cost estimate.

## Development

```bash
uv run pytest            # fake HTTP and model responses; no live-service acceptance
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Start with [`cli.py`](src/filings_agent/cli.py) for command wiring,
[`pipeline.py`](src/filings_agent/pipeline.py) for parse/extract runs, or
[`agent.py`](src/filings_agent/agent/agent.py) for the question loop.
The next useful milestone is a reproducible live evaluation before expanding retrieval
or adding a web interface.
