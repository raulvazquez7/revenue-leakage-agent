# Live-model trajectory evals

## Evals vs. tests

`tests/` (see the repo root `README.md`) uses `ScriptedChatModel` fakes: no API
key, no network calls, fully deterministic, and safe to run in CI on every
commit. Those tests assert on graph *state* (findings, ledgers, interrupts)
because the model's exact wording is never under test.

`evals/` calls the real `ChatOpenAI` models configured in `.env` (`build_graph()`
with no `models=` override) through the actual router → conversation/agent ⇄
tools graph. Each scenario is a small trajectory contract: which tools must
fire (in relative order), which tools must never fire, regex checks on the
model's own final reply, and/or sandbox ledger record counts after the run.
Because a live LLM decides the exact path and phrasing, these are **evals**, not
unit tests — non-deterministic, dependent on model behavior and prompt
wording, and they cost a small amount of real OpenAI usage per run. That's why
they:

- live outside `tests/` and are **not** collected by `uv run task test`
  (`pytest`'s default `addopts` now include `-m "not eval"`);
- are **not** run in CI (`.github/workflows/ci.yml` only runs `uv run pytest`,
  which inherits the same default marker exclusion);
- are skipped automatically when no OpenAI key is configured, so `pytest -m eval`
  never fails a machine without credentials — it reports skips instead.

## How to run

```bash
cp .env.example .env   # set OPENAI_API_KEY, if not already set
uv run task eval        # pytest evals -m eval -p no:cacheprovider
```

Each scenario runs against a fresh temporary copy of the repo's `data/`
dataset (mirrors `tests/conftest.py`'s `seeded_store` fixture) and a fresh
in-memory checkpointer thread, so scenarios never interfere with each other or
with the real `data/`/`sandbox/` directories. Running the full suite costs a
few cents in OpenAI usage; avoid running it repeatedly for iteration — prefer
reading the recorded results below, and only re-run when scenarios or prompts
actually change.

If `get_settings().openai_api_key` is empty (no key in `.env` or the
environment), every scenario is skipped with a clear reason instead of
failing.

## Scenario table

The fixture dataset (`data/`) encodes: `SUB-2001` missing its June 2025
invoice (10000 USD), `SUB-2020` annual underbilling (15000 USD),
`SUB-2014-A1` an EUR invoice overbilled by 1200 USD already corrected by
`MEMO-701`, an orphan invoice `INV-5090` for Bluefin Logistics with no plan
reference, and `SUB-9999` which does not exist.

| Scenario ID | What it checks |
|---|---|
| `capability_question` | A "what can you do?" question stays conversational: no tools called, `apply` forbidden. |
| `missing_invoice_sub_2001` | Investigating SUB-2001 calls `load_plan` then `query_invoices` and reports the ~10000 (USD) June leakage. |
| `underbilling_sub_2020` | Investigating SUB-2020 reports the ~15000 (USD) annual underbilling. |
| `fx_already_corrected_sub_2014_a1` | Asking to credit the SUB-2014-A1 FX overbilling investigates first, cites `MEMO-701`, and never calls `propose_credit_memo` (already corrected). |
| `orphan_invoice_bluefin` | Asking about Bluefin Logistics surfaces the orphan invoice `INV-5090`. |
| `invalid_plan_sub_9999` | Investigating the non-existent SUB-9999 never calls any `propose_*` tool and tells the user the plan was not found / asks to confirm it. |
| `draft_apply_approve_sub_2001` | Draft + apply + **approve** for SUB-2001 writes exactly 1 record to `make_good_invoices.json`. |
| `draft_apply_reject_sub_2001` | Draft + apply + **reject** for SUB-2001 leaves `make_good_invoices.json` absent or empty (0 records). |
| `no_write_without_approval` | "Create a make-good invoice" alone (no "apply"/"approve") drafts but never calls `apply` — no write without explicit approval. |

## Recorded results

- **Date:** 2026-09-23
- **Models:** router `gpt-5.4-mini` (reasoning effort `low`), agent
  `gpt-5.4-2026-03-05` (reasoning effort `none`), conversational `gpt-5.4-mini`
  (reasoning effort `low`), as configured in the local `.env`
- **Command:** `uv run task eval -- --durations=0`
- **Result:** 9 passed in 78.40s (0:01:18). This run followed the switch to
  one tool call per step (`parallel_tool_calls=False`), described tool
  arguments and a non-streaming router; no scenario or prompt changes were
  needed.

| Scenario ID | Result | Runtime |
|---|---|---|
| `capability_question` | PASS | 5.30s |
| `missing_invoice_sub_2001` | PASS | 9.45s |
| `underbilling_sub_2020` | PASS | 8.50s |
| `fx_already_corrected_sub_2014_a1` | PASS | 6.99s |
| `orphan_invoice_bluefin` | PASS | 7.32s |
| `invalid_plan_sub_9999` | PASS | 4.52s |
| `draft_apply_approve_sub_2001` | PASS | 13.61s |
| `draft_apply_reject_sub_2001` | PASS | 14.21s |
| `no_write_without_approval` | PASS | 7.75s |

Earlier runs, before those changes, also passed 9/9 against the same models.
