from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable, Sequence
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.prebuilt.tool_node import TOOL_CALL_ERROR_TEMPLATE, ToolCallRequest
from langgraph.types import Command, interrupt
from pydantic import Field

from revenue_leakage_agent.context import AgentContext, resolve_store
from revenue_leakage_agent.domain.billing import (
    apply_invoice_filters,
    compare_plan_to_invoices,
    convert_amount,
    related_credit_memos,
    summarize_invoices,
)
from revenue_leakage_agent.domain.models import (
    ACTION_DRAFT_ADAPTER,
    CreditMemoDraft,
    Currency,
    InvoiceFilters,
    MakeGoodInvoiceDraft,
    PlanAmendmentDraft,
    ToolError,
)
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.store import JsonStore

# Model-visible argument descriptions. ``runtime`` is injected by ToolNode and
# never appears in the schema sent to the model.
PlanId = Annotated[str, Field(description="Billing plan ID, for example SUB-2001.")]
Reason = Annotated[
    str, Field(min_length=1, description="Business reason shown to the approver.")
]


@tool
def load_plan(
    plan_id: PlanId,
    runtime: ToolRuntime[AgentContext, AgentState],
) -> Command[Any]:
    """Load a billing plan by ID and update the active investigation scope."""

    store = resolve_store(runtime.context)
    normalized_plan_id = plan_id.strip().upper()
    plan = store.get_plan(normalized_plan_id)
    if plan is None:
        return _error_command(
            runtime,
            ToolError(
                error_code="PLAN_NOT_FOUND",
                message=f"No billing plan exists for {normalized_plan_id}.",
                llm_instruction="Ask the user to confirm the plan ID.",
            ),
        )

    payload = plan.model_dump(mode="json")
    return _state_command(
        runtime=runtime,
        payload={"ok": True, "plan": payload},
        active_scope={
            "plan_id": plan.plan_id,
            "customer_name": plan.customer_name,
            "investigation_mode": "plan",
        },
    )


@tool
def query_invoices(
    runtime: ToolRuntime[AgentContext, AgentState],
    plan_id: Annotated[
        str | None,
        Field(
            description="Billing plan ID, for example SUB-2001. When set, the "
            "result includes the expected-vs-actual plan comparison."
        ),
    ] = None,
    customer_name: Annotated[
        str | None, Field(description="Exact customer name, e.g. Bluefin Logistics.")
    ] = None,
    start_date: Annotated[
        date | None,
        Field(description="Earliest invoice issue_date to include (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        date | None,
        Field(description="Latest invoice issue_date to include (YYYY-MM-DD)."),
    ] = None,
    status: Annotated[
        str | None, Field(description="Invoice status, e.g. paid or open.")
    ] = None,
    currency: Annotated[
        str | None, Field(description="Invoice currency: USD, EUR or GBP.")
    ] = None,
    limit: Annotated[
        int, Field(ge=1, le=100, description="Maximum invoices to return.")
    ] = 25,
) -> Command[Any]:
    """Query invoices and include plan comparison when plan_id is provided.

    Date filters apply to the invoice ``issue_date``. Invoices with an empty
    ``plan_id`` are orphan payments with no contract reference.
    """

    filters = InvoiceFilters(
        plan_id=plan_id.strip().upper() if plan_id else None,
        customer_name=customer_name,
        start_date=start_date,
        end_date=end_date,
        status=status,
        currency=_validated_currency(currency),
        limit=limit,
    )
    store = resolve_store(runtime.context)
    all_invoices = store.load_invoices()
    invoices, truncated = apply_invoice_filters(all_invoices, filters)
    summary = summarize_invoices(invoices)
    all_credit_memos = store.load_credit_memos()
    credit_memos = related_credit_memos(
        credit_memos=all_credit_memos,
        invoices=invoices,
        plan_id=filters.plan_id,
    )

    comparison: dict[str, Any] | None = None
    findings: list[dict[str, Any]] = []
    if filters.plan_id:
        plan = store.get_plan(filters.plan_id)
        if plan is None:
            return _error_command(
                runtime,
                ToolError(
                    error_code="PLAN_NOT_FOUND",
                    message=f"No billing plan exists for {filters.plan_id}.",
                    llm_instruction="Ask the user to confirm the plan ID.",
                ),
            )
        comparison = compare_plan_to_invoices(
            plan=plan,
            invoices=all_invoices,
            exchange_rates=store.load_exchange_rates(),
            credit_memos=all_credit_memos,
            filters=filters,
        )
        findings = comparison["findings"]

    payload = {
        "ok": True,
        "filters_applied": filters.model_dump(mode="json"),
        "count": len(invoices),
        "truncated": truncated,
        "invoices": [invoice.model_dump(mode="json") for invoice in invoices],
        **summary,
        "related_credit_memos": credit_memos,
        "plan_comparison": comparison,
    }
    return _state_command(
        runtime=runtime,
        payload=payload,
        findings=findings,
    )


@tool
def fx_convert(
    amount: Annotated[Decimal, Field(gt=0, description="Amount to convert.")],
    from_ccy: Annotated[str, Field(description="Source currency code, e.g. EUR.")],
    to_ccy: Annotated[str, Field(description="Target currency code, e.g. USD.")],
    on_date: Annotated[
        date, Field(description="Date of the FX rate to use (YYYY-MM-DD).")
    ],
    runtime: ToolRuntime[AgentContext, AgentState],
) -> dict[str, Any]:
    """Convert an amount using the exchange-rate dataset."""

    return convert_amount(
        amount=amount,
        from_ccy=from_ccy,
        to_ccy=to_ccy,
        on_date=on_date,
        exchange_rates=resolve_store(runtime.context).load_exchange_rates(),
    )


@tool
def propose_make_good_invoice(
    plan_id: PlanId,
    amount: Annotated[
        Decimal,
        Field(gt=0, description="Revenue to recover, in the plan currency."),
    ],
    reason: Reason,
    runtime: ToolRuntime[AgentContext, AgentState],
) -> Command[Any]:
    """Create a draft make-good invoice without writing to sandbox ledgers."""

    store = resolve_store(runtime.context)
    normalized_plan_id = plan_id.strip().upper()
    plan = store.get_plan(normalized_plan_id)
    if plan is None:
        return _error_command(
            runtime,
            ToolError(
                error_code="PLAN_NOT_FOUND",
                message=f"No billing plan exists for {normalized_plan_id}.",
                llm_instruction="Ask the user to confirm the plan ID before drafting.",
            ),
        )

    evidence = _matching_finding_evidence(
        findings=runtime.state.get("findings", []),
        plan_id=normalized_plan_id,
        amount=amount,
    )
    draft = MakeGoodInvoiceDraft(
        action_id=f"DRAFT-MG-{uuid4().hex[:8].upper()}",
        plan_id=plan.plan_id,
        amount=amount,
        currency=plan.currency,
        reason=reason,
        evidence=evidence,
    )
    return _draft_command(runtime=runtime, draft=draft.model_dump(mode="json"))


@tool
def propose_credit_memo(
    invoice_id: Annotated[
        str, Field(description="Overbilled invoice ID, for example INV-5022.")
    ],
    amount: Annotated[
        Decimal,
        Field(gt=0, description="Amount to credit back, in the plan currency."),
    ],
    reason: Reason,
    runtime: ToolRuntime[AgentContext, AgentState],
) -> Command[Any]:
    """Create a draft credit memo to correct overbilling on a specific invoice.

    The credit is denominated in the plan currency (the comparison currency),
    falling back to the invoice currency for orphan invoices. Nothing is written
    to the sandbox until the human approves the pending action.
    """

    store = resolve_store(runtime.context)
    normalized_invoice_id = invoice_id.strip().upper()
    invoice = store.get_invoice(normalized_invoice_id)
    if invoice is None:
        return _error_command(
            runtime,
            ToolError(
                error_code="INVOICE_NOT_FOUND",
                message=f"No invoice exists for {normalized_invoice_id}.",
                llm_instruction="Ask the user to confirm the invoice ID.",
            ),
        )

    plan = store.get_plan(invoice.plan_id) if invoice.plan_id else None
    currency: Currency = plan.currency if plan is not None else invoice.currency
    evidence = _matching_finding_evidence_by_invoice(
        findings=runtime.state.get("findings", []),
        invoice_id=normalized_invoice_id,
        amount=amount,
    )
    draft = CreditMemoDraft(
        action_id=f"DRAFT-CM-{uuid4().hex[:8].upper()}",
        invoice_id=invoice.invoice_id,
        plan_id=invoice.plan_id,
        amount=amount,
        currency=currency,
        reason=reason,
        evidence=evidence,
    )
    return _draft_command(runtime=runtime, draft=draft.model_dump(mode="json"))


@tool
def propose_plan_amendment(
    plan_id: PlanId,
    change_set: Annotated[
        dict[str, Any],
        Field(description='Plan fields to change, e.g. {"total_value": 96000}.'),
    ],
    reason: Reason,
    runtime: ToolRuntime[AgentContext, AgentState],
) -> Command[Any]:
    """Create a draft plan amendment when the plan no longer matches the deal.

    ``change_set`` holds the proposed plan fields to change (for example
    ``total_value``, ``cadence``, or ``entitlements``). Nothing is written to the
    sandbox until the human approves the pending action.
    """

    store = resolve_store(runtime.context)
    normalized_plan_id = plan_id.strip().upper()
    plan = store.get_plan(normalized_plan_id)
    if plan is None:
        return _error_command(
            runtime,
            ToolError(
                error_code="PLAN_NOT_FOUND",
                message=f"No billing plan exists for {normalized_plan_id}.",
                llm_instruction="Ask the user to confirm the plan ID before drafting.",
            ),
        )
    if not change_set:
        return _error_command(
            runtime,
            ToolError(
                error_code="EMPTY_CHANGE_SET",
                message="A plan amendment needs at least one field to change.",
                llm_instruction="Ask the user which plan fields should change.",
            ),
        )

    evidence = [
        f"Plan {plan.plan_id} currently: total_value {plan.total_value} "
        f"{plan.currency}, cadence {plan.cadence}.",
        f"Proposed change_set: {change_set}.",
    ]
    draft = PlanAmendmentDraft(
        action_id=f"DRAFT-PA-{uuid4().hex[:8].upper()}",
        plan_id=plan.plan_id,
        change_set=change_set,
        reason=reason,
        evidence=evidence,
    )
    return _draft_command(runtime=runtime, draft=draft.model_dump(mode="json"))


@tool
def apply(
    runtime: ToolRuntime[AgentContext, AgentState],
    action_id: Annotated[
        str | None,
        Field(description="Draft action ID. Omit to apply the current pending draft."),
    ] = None,
) -> Command[Any]:
    """Apply the pending action only after a human approval interrupt."""

    pending_action = runtime.state.get("pending_action")
    if pending_action is None:
        return _error_command(
            runtime,
            ToolError(
                error_code="ACTION_NOT_FOUND",
                message="There is no pending draft action to apply.",
                llm_instruction="Ask the user which action should be drafted first.",
            ),
        )

    draft = ACTION_DRAFT_ADAPTER.validate_python(pending_action)
    if action_id is not None and action_id != draft.action_id:
        return _error_command(
            runtime,
            ToolError(
                error_code="ACTION_NOT_FOUND",
                message=f"Pending draft is {draft.action_id}, not {action_id}.",
                llm_instruction="Use the pending action ID or ask the user to choose.",
            ),
        )

    store = resolve_store(runtime.context)
    if store.get_action_already_applied(draft.action_id):
        return _error_command(
            runtime,
            ToolError(
                error_code="ACTION_ALREADY_APPLIED",
                message=f"Action {draft.action_id} was already applied.",
                recoverable=False,
                llm_instruction="Tell the user the action was already applied.",
            ),
        )

    decision = interrupt(
        {
            "type": "approval_required",
            "question": "Approve this sandbox mutation?",
            "action": draft.model_dump(mode="json"),
        }
    )
    if not _is_approved(decision):
        rejected = {
            "action_id": draft.action_id,
            "action_type": draft.action_type,
            "status": "rejected",
            "sandbox_file": None,
            "sandbox_record_id": None,
            "audit_log_entry": {"decision": decision},
        }
        return _state_command(
            runtime=runtime,
            payload={"ok": True, "status": "rejected", "action_id": draft.action_id},
            pending_action=None,
            applied_actions=[*runtime.state.get("applied_actions", []), rejected],
        )

    draft_payload = draft.model_dump(mode="json")
    record, sandbox_file, record_id = _write_sandbox_record(store, draft_payload)
    audit_entry = store.append_audit_log(
        {
            "event_type": "action_applied",
            "action": draft_payload,
            "sandbox_record": record,
        }
    )
    applied = {
        "action_id": draft.action_id,
        "action_type": draft.action_type,
        "status": "applied",
        "sandbox_file": sandbox_file,
        "sandbox_record_id": record_id,
        "audit_log_entry": audit_entry,
    }
    return _state_command(
        runtime=runtime,
        payload={
            "ok": True,
            "status": "applied",
            "action_type": draft.action_type,
            "sandbox_record": record,
            "audit_log_entry": audit_entry,
        },
        pending_action=None,
        applied_actions=[*runtime.state.get("applied_actions", []), applied],
    )


@tool
def rollback(
    runtime: ToolRuntime[AgentContext, AgentState],
    action_id: Annotated[
        str | None,
        Field(description="Applied action ID. Omit to roll back the most recent one."),
    ] = None,
) -> Command[Any]:
    """Undo an applied sandbox action after a human approval interrupt.

    Rolls back the most recent applied action, or a specific one by ``action_id``.
    """

    applied_actions = runtime.state.get("applied_actions", [])
    target = _find_rollback_target(applied_actions, action_id)
    if target is None:
        return _error_command(
            runtime,
            ToolError(
                error_code="ACTION_NOT_FOUND",
                message="There is no applied action available to roll back.",
                llm_instruction="Tell the user there is nothing to roll back.",
            ),
        )

    decision = interrupt(
        {
            "type": "approval_required",
            "question": "Approve rolling back this applied action?",
            "action": {
                "action_id": target["action_id"],
                "action_type": target["action_type"],
                "status": "rollback_requested",
                "sandbox_file": target.get("sandbox_file"),
                "sandbox_record_id": target.get("sandbox_record_id"),
            },
        }
    )
    if not _is_approved(decision):
        return _state_command(
            runtime=runtime,
            payload={
                "ok": True,
                "status": "rollback_rejected",
                "action_id": target["action_id"],
            },
        )

    store = resolve_store(runtime.context)
    removed = store.remove_sandbox_record(target["action_type"], target["action_id"])
    if removed is None:
        return _error_command(
            runtime,
            ToolError(
                error_code="SANDBOX_RECORD_NOT_FOUND",
                message=f"No sandbox record found for {target['action_id']}.",
                recoverable=False,
                llm_instruction="Tell the user the record was already removed.",
            ),
        )
    audit_entry = store.append_audit_log(
        {
            "event_type": "action_rolled_back",
            "action_id": target["action_id"],
            "action_type": target["action_type"],
            "removed_record": removed,
        }
    )
    updated_actions = [
        {**action, "status": "rolled_back"}
        if action["action_id"] == target["action_id"]
        else action
        for action in applied_actions
    ]
    return _state_command(
        runtime=runtime,
        payload={
            "ok": True,
            "status": "rolled_back",
            "action_id": target["action_id"],
            "removed_record": removed,
            "audit_log_entry": audit_entry,
        },
        applied_actions=updated_actions,
    )


def get_tools() -> list[BaseTool]:
    return [
        load_plan,
        query_invoices,
        fx_convert,
        propose_make_good_invoice,
        propose_credit_memo,
        propose_plan_amendment,
        apply,
        rollback,
    ]


def build_tool_node(tools: Sequence[BaseTool] | None = None) -> ToolNode:
    """The graph's ToolNode: one call per step, tool errors become ToolMessages."""

    return ToolNode(
        list(tools) if tools is not None else get_tools(),
        handle_tool_errors=tool_error_message,
        wrap_tool_call=one_tool_call_per_step,
    )


def one_tool_call_per_step(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """``ToolNode`` wrapper: run only the first tool call of an AIMessage.

    Non-message state keys (``active_scope``, ``findings``, ``pending_action``,
    ...) take one write per step, so two state-writing calls in one AIMessage
    would raise ``InvalidUpdateError`` and leave the thread unreadable. The
    agent binds tools with ``parallel_tool_calls=False``; this guards models
    that ignore it by answering every extra call with an error ToolMessage, so
    each call still has a result and the model can retry them one at a time.
    """

    call_ids = _last_tool_call_ids(request.state)
    if len(call_ids) <= 1 or request.tool_call["id"] == call_ids[0]:
        return execute(request)
    return ToolMessage(
        content=json.dumps(
            {
                "ok": False,
                "error": {
                    "error_code": "PARALLEL_TOOL_CALL",
                    "message": "Only one tool call runs per step; this one was "
                    "skipped.",
                    "llm_instruction": "Call one tool at a time; repeat this "
                    "call on its own if it is still needed.",
                },
            }
        ),
        tool_call_id=str(request.tool_call["id"]),
        name=request.tool_call["name"],
        status="error",
    )


def tool_error_message(error: Exception) -> str:
    """``ToolNode`` error handler: the default message, but interrupts propagate.

    Equivalent to ``handle_tool_errors=True`` except that ``GraphBubbleUp``
    (raised by ``interrupt()`` in ``apply``/``rollback``) is re-raised. With a
    ``wrap_tool_call`` wrapper, ToolNode 1.x routes any exception escaping the
    wrapper through this handler, and ``True`` would turn the approval
    interrupt into an error ToolMessage.
    """

    if isinstance(error, GraphBubbleUp):
        raise error
    return TOOL_CALL_ERROR_TEMPLATE.format(error=repr(error))


def _last_tool_call_ids(state: Any) -> list[str]:
    messages: list[Any] = (
        cast(dict[str, Any], state).get("messages", [])
        if isinstance(state, dict)
        else []
    )
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return [str(call["id"]) for call in message.tool_calls]
    return []


def _state_command(
    *,
    runtime: ToolRuntime[AgentContext, AgentState],
    payload: dict[str, Any],
    **updates: Any,
) -> Command[Any]:
    return cast(
        Command[Any],
        Command(
            update={
                **updates,
                "messages": [
                    ToolMessage(
                        content=json.dumps(payload, default=str),
                        tool_call_id=_tool_call_id(runtime),
                    )
                ],
            }
        ),
    )


def _error_command(
    runtime: ToolRuntime[AgentContext, AgentState], error: ToolError
) -> Command[Any]:
    payload = {"ok": False, "error": error.model_dump(mode="json")}
    return _state_command(
        runtime=runtime,
        payload=payload,
        last_error=error.model_dump(mode="json"),
    )


def _draft_command(
    *, runtime: ToolRuntime[AgentContext, AgentState], draft: dict[str, Any]
) -> Command[Any]:
    payload = {
        "ok": True,
        "draft": draft,
        "message": "Draft created. Explicit human approval is required before apply.",
    }
    return _state_command(
        runtime=runtime,
        payload=payload,
        pending_action=draft,
    )


def _tool_call_id(runtime: ToolRuntime[AgentContext, AgentState]) -> str:
    """ToolNode always supplies the call ID; fail loudly if invoked without one."""

    if runtime.tool_call_id is None:
        raise ValueError("Tool was invoked without a tool_call_id.")
    return runtime.tool_call_id


def _write_sandbox_record(
    store: JsonStore,
    draft: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    """Persist an approved draft to its ledger and return (record, file, record_id)."""

    action_type = draft["action_type"]
    if action_type == "make_good_invoice":
        record = store.append_make_good_invoice(draft)
        return record, "sandbox/make_good_invoices.json", record["invoice_id"]
    if action_type == "credit_memo":
        record = store.append_credit_memo(draft)
        return record, "sandbox/credit_memos.json", record["memo_id"]
    if action_type == "plan_amendment":
        record = store.append_plan_amendment(draft)
        return record, "sandbox/plan_amendments.json", record["amendment_id"]
    raise ValueError(f"Unknown action type: {action_type}")


def _find_rollback_target(
    applied_actions: list[dict[str, Any]],
    action_id: str | None,
) -> dict[str, Any] | None:
    live = [action for action in applied_actions if action.get("status") == "applied"]
    if action_id is not None:
        return next(
            (action for action in live if action.get("action_id") == action_id),
            None,
        )
    return live[-1] if live else None


def _matching_finding_evidence(
    findings: list[dict[str, Any]],
    plan_id: str,
    amount: Decimal,
) -> list[str]:
    for finding in findings:
        if finding.get("plan_id") != plan_id:
            continue
        if Decimal(str(finding.get("amount", "0"))) == amount:
            evidence = finding.get("evidence", [])
            return [str(item) for item in evidence]
    return [f"Make-good amount requested for plan {plan_id}: {amount}."]


def _matching_finding_evidence_by_invoice(
    findings: list[dict[str, Any]],
    invoice_id: str,
    amount: Decimal,
) -> list[str]:
    for finding in findings:
        if invoice_id not in [str(i) for i in finding.get("invoice_ids", [])]:
            continue
        if Decimal(str(finding.get("amount", "0"))) == amount:
            evidence = finding.get("evidence", [])
            return [str(item) for item in evidence]
    return [f"Credit memo requested for invoice {invoice_id}: {amount}."]


def _is_approved(decision: Any) -> bool:
    if isinstance(decision, str):
        return _normalize_decision(decision) in {"approve", "approved", "yes", "si"}
    if isinstance(decision, dict):
        decision_map = cast(dict[str, Any], decision)
        value = _normalize_decision(str(decision_map.get("decision", "")))
        return value in {"approve", "approved", "yes", "si"}
    return False


def _validated_currency(value: str | None) -> Currency | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if normalized not in {"USD", "EUR", "GBP"}:
        raise ValueError(f"Unsupported currency filter: {value}")
    return cast(Currency, normalized)


def _normalize_decision(value: str) -> str:
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .lower()
    )
