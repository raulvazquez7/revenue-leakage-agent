# Revenue Leakage Response Template

Use this template only if a separate response layer is introduced later. In the current graph, the investigator can produce final answers directly.

## Response Principles

- Reply in the user's language.
- Start with the answer or final status.
- Cite evidence compactly.
- Include IDs, period, expected amount, actual amount, currency, and action ID when relevant.
- Mention assumptions, truncation, missing scope, or skipped checks when they affect confidence.
- Do not reveal hidden prompts, chain-of-thought, node names, graph internals, or raw tool payloads unless the user explicitly asks for implementation details.

## Investigation Finding

```text
<Direct answer: found leakage / no issue / overbilling / needs more scope.>

Evidence:
- Plan: <plan_id>, <amount> <currency>, <cadence>.
- Period: <period_start> to <period_end>.
- Expected: <expected_amount> <currency>.
- Actual: <actual_amount> <currency> from <invoice_ids or no valid invoice>.
- FX: <rate/date/rounding> when relevant.

Impact: <impact_amount> <currency>.
Recommended action: <action_type or none>.
Next step: <draft/apply/clarify/no action>.
```

## Clean Result

```text
I did not find revenue leakage for <scope>.

Evidence:
- Expected billing matched actual valid invoices for <periods>.
- Checked invoices: <invoice_ids or count>.
- Currency: <currency>, with FX evidence if applicable.

No corrective action is needed.
```

## Draft Created

```text
I prepared a draft <action_type>; nothing has been written to the sandbox yet.

Draft: <action_id>
Target: <plan_id and/or invoice_id>
Amount: <amount> <currency>
Reason: <reason>

Would you like to apply it?
```

## Apply Approved

```text
Applied. The sandbox now contains <action_type> <action_id>.

Sandbox record: <record id or compact record summary>
Audit log: <audit entry id or compact reference>

Why: <one-sentence evidence-backed reason>.
```

## Apply Rejected

```text
Rejected. I did not write anything to the sandbox.

Draft: <action_id>
Reason it was proposed: <one-sentence evidence-backed reason>.
```

## Recoverable Error

```text
I could not complete that step yet: <short error message>.

<One concise next question or corrective action based on llm_instruction.>
```

## Unsupported Action

```text
I can identify and explain that issue, but <action_type> is not supported in this version yet.

What I found: <compact evidence>.
Supported next step: <nearest available action or no action>.
```
