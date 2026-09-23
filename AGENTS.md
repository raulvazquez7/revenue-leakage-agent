# AGENTS.md

Guidance for AI coding assistants (Claude Code, Cursor, Codex, ...) working in
this repository.

## Project

A stateful LangGraph agent that investigates revenue leakage between billing
plans and invoices and applies corrections to a local sandbox only after human
approval. See `README.md` for the architecture.

## Ground rules

- Money, dates, IDs, FX conversion and sandbox writes are deterministic Python.
  The LLM routes, selects tools and explains; it never produces financial facts.
- Every sandbox mutation must go through a LangGraph `interrupt()` approval.
- Keep prompts in `prompts/`; keep runtime state out of static prompts.
- Prefer small, typed, pure functions; Pyright runs in strict mode.

## Commands

```bash
uv sync --all-groups   # install
uv run task check      # ruff format --check, ruff check, pyright
uv run task test       # pytest
```

Run `uv run task check && uv run task test` before proposing a change.
