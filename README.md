# Revenue Leakage Agent

A stateful [LangGraph](https://github.com/langchain-ai/langgraph) agent that acts
as a "financial detective": it compares billing plans with issued invoices,
explains revenue leakage with verifiable evidence, drafts corrective actions and
applies them to a local sandbox only after explicit human approval.

> **Side project.** Built over a weekend to explore LangChain and LangGraph
> patterns (routing, tool calling, human-in-the-loop, deterministic tools). It is
> a learning project, not production software.

## Quickstart

```bash
uv sync --all-groups
cp .env.example .env   # set OPENAI_API_KEY
uv run task ui       # Streamlit UI
uv run task cli      # terminal chat (same as: uv run revenue-leakage-agent)
uv run task studio   # LangGraph Studio via `langgraph dev` (isolated uvx env)
uv run task api      # HTTP API on :8000 (OpenAPI docs at /docs)
```

## Docker

```bash
cp .env.example .env        # set OPENAI_API_KEY
docker compose up --build   # UI on :8501, API on :8000
docker compose down -v      # stop and drop the sandbox/checkpoint volume
```

Both services share a named volume for sandbox ledgers and SQLite checkpoints.

## Quality checks

```bash
uv run task check
uv run task test
```

## License

[MIT](LICENSE)
