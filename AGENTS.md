# AGENTS.md

Guidance for AI coding assistants (Claude Code, Cursor, Codex, ...) working in
this repository.

## Project

A stateful LangGraph agent that investigates revenue leakage between billing
plans and invoices and applies corrections to a local sandbox only after human
approval. See `README.md` for the overview and `docs/architecture.md` for the
state schema, tools, approval flow and persistence.

## Layout

- `src/revenue_leakage_agent/`: `graph.py`, `state.py`, `nodes/`, `tools.py`,
  `domain/` (deterministic billing math and models), `store.py`, `context.py`,
  `history.py`, `persistence.py`, `tracing.py`, `llm.py`, `config.py`,
  `prompts/`, `interfaces/` (CLI, Streamlit, streaming helpers, FastAPI).
- `tests/`: `unit/` and `integration/`; `fakes.py` holds `ScriptedChatModel`.
- `evals/`: live-model trajectory evals (opt-in, not in CI).
- `data/`: read-only synthetic dataset. `sandbox/` is generated and git-ignored.

## Ground rules

- Money, dates, IDs, FX conversion and sandbox writes are deterministic Python.
  The LLM routes, selects tools and explains; it never produces financial facts.
- Every sandbox mutation must go through a LangGraph `interrupt()` approval.
- Keep prompts in `src/revenue_leakage_agent/prompts/`; keep runtime state out of static prompts.
- Pass dependencies through `AgentContext` (runtime context), not globals.
- Prefer small, typed, pure functions; Pyright runs in strict mode.
- Tests use scripted models and assert on state, never on model wording. They
  run on code-default settings: `tests/conftest.py` ignores `.env`.
- The agent runs one tool call per step (only `messages` has a reducer), so a
  tool may write state keys without coordinating with other tools.

## Commands

```bash
uv sync --all-groups   # install
uv run task check      # ruff format --check, ruff check, pyright
uv run task test       # pytest (unit + integration, excludes evals)
uv run task precommit  # pre-commit hooks on all files
uv run task eval       # live-model evals (needs OPENAI_API_KEY, costs money)
uv run task ui         # also: task cli [--verbose], task api, task studio
```

Run `uv run task check && uv run task test` before proposing a change.
