# Prompt Set

This folder contains runtime prompt assets for the Revenue Leakage Agent. The prompts are separated by graph responsibility so each model call receives only the high-signal context it needs.

## Files

- `router.md`: structured-output classifier for `RouteDecision`. It chooses `conversation` or `investigation` and rewrites implicit follow-ups.
- `conversational.md`: lightweight responder for greetings, capability questions, general concepts, recall, and out-of-scope redirection.
- `agent.md`: investigator prompt for tool use, evidence-backed revenue leakage analysis, draft creation, approval-aware apply, and recovery from tool errors.
- `approval.md`: deterministic human-in-the-loop approval card and interrupt payload guidance. This is not an LLM decision prompt.
- `response.md`: optional response-format template if a separate response node is introduced later.
- `compaction.md`: optional summarization prompt for future context compaction.

## Context Engineering Notes

- Keep system prompts stable and compact.
- Inject runtime state separately from the static prompt:
  - active scope
  - findings
  - pending action
  - applied actions
  - last error
- Do not inject raw datasets or long invoice lists into the system prompt.
- Prefer just-in-time retrieval through tools.
- Preserve raw tool results only long enough to reason in the current turn; keep compact findings and action state for follow-ups.
- Use a few canonical examples instead of a long edge-case catalog.

## Runtime Loading

At runtime, each node loads the relevant Markdown file as its system prompt:

- Router node: `router.md` plus compact active scope and recent messages.
- Conversational node: `conversational.md` plus recent messages and compact state if recall is needed.
- Agent node: `agent.md` plus compact graph state and tool bindings.

`agents/prompts.py` loads these files by configured prompt name and falls back
to a safe inline default only if a prompt file is missing or empty.
