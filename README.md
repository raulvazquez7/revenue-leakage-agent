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
uv run task studio   # LangGraph Studio via `langgraph dev` (studio group)
```

## Quality checks

```bash
uv run task check
uv run task test
```

## License

[MIT](LICENSE)
