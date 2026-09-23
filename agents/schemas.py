from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

Currency = Literal["USD", "EUR", "GBP"]
BillingCadence = Literal["Monthly", "Quarterly", "Annual"]
FindingType = Literal[
    "missing_invoice",
    "underbilling",
    "overbilling",
    "fx_mismatch",
    "plan_mismatch",
    "ok",
]
FindingStatus = Literal[
    "ok",
    "leakage",
    "overbilled",
    "plan_mismatch",
    "already_corrected",
]
RecommendedAction = Literal[
    "none",
    "make_good_invoice",
    "credit_memo",
    "plan_amendment",
]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Plan(BaseModel):
    """Contracted billing plan. Amounts are contract totals, not per period."""

    model_config = ConfigDict(extra="allow")

    plan_id: str = Field(min_length=1)
    customer_name: str = Field(min_length=1)
    total_value: Decimal = Field(gt=0)
    currency: Currency
    cadence: BillingCadence
    start_date: date
    entitlements: list[str] = Field(default_factory=list)
    amends: str | None = Field(
        default=None,
        description="Plan ID this plan supersedes, when it is an amendment.",
    )
    notes: str | None = None


class Invoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    invoice_id: str = Field(min_length=1)
    # Orphan invoices carry an empty plan_id (no contract reference).
    plan_id: str = Field(default="")
    customer_name: str = Field(min_length=1)
    issue_date: date
    due_date: date | None = None
    amount_invoiced: Decimal
    currency: Currency
    status: str = Field(min_length=1)
    description: str | None = None


class CreditMemo(BaseModel):
    model_config = ConfigDict(extra="allow")

    memo_id: str = Field(min_length=1)
    invoice_id: str = Field(min_length=1)
    plan_id: str = Field(default="")
    amount: Decimal = Field(gt=0)
    currency: Currency
    issue_date: date
    reason: str = Field(min_length=1)


class ExchangeRate(StrictBaseModel):
    date: date
    from_currency: Currency
    to_currency: Currency
    rate: Decimal = Field(gt=0)


class InvestigationScope(StrictBaseModel):
    plan_id: str | None = None
    customer_name: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    investigation_mode: Literal["plan", "customer", "period", "broad_scan"] = "plan"
    resolved_question: str | None = None


class Finding(StrictBaseModel):
    finding_id: str
    plan_id: str
    invoice_ids: list[str] = Field(default_factory=list)
    period_start: date
    period_end: date
    type: FindingType
    status: FindingStatus
    amount: Decimal = Field(ge=0)
    currency: Currency
    evidence: list[str] = Field(default_factory=list)
    recommended_action: RecommendedAction = "none"


ActionType = Literal["make_good_invoice", "credit_memo", "plan_amendment"]


class MakeGoodInvoiceDraft(StrictBaseModel):
    """Draft new invoice to recover missed or underbilled revenue."""

    action_id: str
    action_type: Literal["make_good_invoice"] = "make_good_invoice"
    status: Literal["draft"] = "draft"
    plan_id: str = Field(min_length=1)
    amount: Decimal = Field(gt=0)
    currency: Currency
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    requires_approval: Literal[True] = True


class CreditMemoDraft(StrictBaseModel):
    """Draft negative invoice (credit) to correct overbilling on an invoice."""

    action_id: str
    action_type: Literal["credit_memo"] = "credit_memo"
    status: Literal["draft"] = "draft"
    invoice_id: str = Field(min_length=1)
    plan_id: str = Field(default="")
    amount: Decimal = Field(gt=0)
    currency: Currency
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    requires_approval: Literal[True] = True


class PlanAmendmentDraft(StrictBaseModel):
    """Draft update to a billing plan when the plan no longer reflects the deal."""

    action_id: str
    action_type: Literal["plan_amendment"] = "plan_amendment"
    status: Literal["draft"] = "draft"
    plan_id: str = Field(min_length=1)
    change_set: dict[str, Any] = Field(
        description="Proposed plan fields to change, e.g. total_value, cadence.",
    )
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    requires_approval: Literal[True] = True


ActionDraft = Annotated[
    MakeGoodInvoiceDraft | CreditMemoDraft | PlanAmendmentDraft,
    Field(discriminator="action_type"),
]

ACTION_DRAFT_ADAPTER: TypeAdapter[
    MakeGoodInvoiceDraft | CreditMemoDraft | PlanAmendmentDraft
] = TypeAdapter(ActionDraft)


class AppliedAction(StrictBaseModel):
    action_id: str
    action_type: ActionType
    status: Literal["applied", "rejected", "rolled_back"]
    sandbox_file: str | None = None
    sandbox_record_id: str | None = None
    audit_log_entry: dict[str, Any] = Field(default_factory=dict)


class ToolError(StrictBaseModel):
    error_code: str
    message: str
    recoverable: bool = True
    llm_instruction: str


class InvoiceFilters(StrictBaseModel):
    plan_id: str | None = None
    customer_name: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    status: str | None = None
    currency: Currency | None = None
    limit: int = Field(default=25, ge=1, le=100)


class RouteDecision(StrictBaseModel):
    route: Literal["conversation", "investigation"]
    intent: Literal[
        "chit_chat",
        "capability_question",
        "investigation",
        "out_of_scope",
    ]
    reason: str = Field(min_length=1)
    resolved_question: str | None = None
