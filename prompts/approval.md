# Approval Interrupt Template

This file defines the human approval copy and payload expectations for sandbox mutations. The approval decision is deterministic; do not use an LLM to decide whether to write.

## When To Use

Use this template when a tool is about to mutate sandbox business data:

- Apply a make-good invoice.
- Apply a credit memo when supported.
- Apply a plan amendment when supported.
- Roll back an applied action when supported.

The interrupt must fire before any write. The graph resumes only after the user explicitly approves or rejects.

## Interrupt Payload Shape

```json
{
  "type": "approval_required",
  "question": "Approve this sandbox mutation?",
  "action": {
    "action_id": "<draft action id>",
    "action_type": "<make_good_invoice | credit_memo | plan_amendment | rollback>",
    "status": "draft",
    "target": {
      "plan_id": "<plan id when relevant>",
      "invoice_id": "<invoice id when relevant>"
    },
    "amount": "<amount when relevant>",
    "currency": "<currency when relevant>",
    "reason": "<business reason>",
    "evidence": ["<compact evidence strings>"],
    "requires_approval": true
  }
}
```

Keep the payload structured. The UI should render the card from fields, not from a prose blob.

## User-Facing Card Copy

Title:

```text
Approval required
```

Body:

```text
I prepared a sandbox change but have not applied it yet.

Action: <action_type>
Target: <plan_id and/or invoice_id>
Amount: <amount> <currency>
Reason: <reason>

Evidence:
- <evidence item 1>
- <evidence item 2>

Approve this sandbox mutation?
```

Buttons:

```text
Approve
Reject
```

## Resume Payloads

Approved:

```json
{"decision": "approve"}
```

Rejected:

```json
{"decision": "reject"}
```

## Invariants

- Never apply a sandbox mutation before this approval step.
- Never treat ambiguous user text as approval if the graph is not in the expected pending-action state.
- Never ask for approval of an action that has no action ID or no evidence.
- If the user rejects, return a normal tool result indicating no write occurred.
- If the user approves, write deterministically and return the sandbox record and audit log entry.
- The assistant may explain the recommendation, but the approval decision belongs to the human.
