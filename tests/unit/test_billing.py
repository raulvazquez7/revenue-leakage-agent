from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from revenue_leakage_agent.domain.billing import (
    compare_plan_to_invoices,
    convert_amount,
)
from revenue_leakage_agent.domain.models import (
    ACTION_DRAFT_ADAPTER,
    CreditMemo,
    CreditMemoDraft,
    ExchangeRate,
    Invoice,
    MakeGoodInvoiceDraft,
    Plan,
    PlanAmendmentDraft,
)

EUR_USD_2025_08_12 = ExchangeRate(
    date=date(2025, 8, 12),
    from_currency="EUR",
    to_currency="USD",
    rate=Decimal("1.12"),
)


def test_detects_missing_interior_month_for_monthly_plan() -> None:
    """SUB-2001 bills 10000/month; June has no invoice while surrounding
    months do, so it is a real interior gap of missed revenue."""

    plan = _plan("SUB-2001", total_value=Decimal("120000"), cadence="Monthly")
    invoices = [
        _invoice("INV-5005", "SUB-2001", "2025-05-03", Decimal("10000")),
        _invoice("INV-5006", "SUB-2001", "2025-07-03", Decimal("10000")),
    ]

    comparison = _compare(plan, invoices)
    findings = comparison["findings"]

    assert len(findings) == 1
    assert findings[0]["type"] == "missing_invoice"
    assert findings[0]["amount"] == "10000.00"
    assert findings[0]["recommended_action"] == "make_good_invoice"
    assert findings[0]["period_start"] == "2025-06-01"


def test_does_not_flag_missing_period_before_first_invoice() -> None:
    """A quarterly plan whose only invoice lands in a later quarter must not
    fabricate a missing invoice for the pre-billing quarter."""

    plan = _plan("SUB-9002", total_value=Decimal("90000"), cadence="Quarterly")
    invoices = [_invoice("INV-9002", "SUB-9002", "2025-06-10", Decimal("20000"))]

    comparison = _compare(plan, invoices)
    findings = comparison["findings"]

    assert all(f["type"] != "missing_invoice" for f in findings)
    # 22500 expected per quarter vs 20000 billed -> 2500 underbilling.
    assert len(findings) == 1
    assert findings[0]["type"] == "underbilling"
    assert findings[0]["amount"] == "2500.00"


def test_detects_annual_underbilling() -> None:
    """SUB-2020 has an annual contract of 150000 but was billed 135000,
    a 15000 make-good."""

    plan = _plan(
        "SUB-2020",
        total_value=Decimal("150000"),
        cadence="Annual",
        start_date=date(2025, 2, 1),
    )
    invoices = [_invoice("INV-5041", "SUB-2020", "2025-02-06", Decimal("135000"))]

    comparison = _compare(plan, invoices)
    findings = comparison["findings"]

    assert len(findings) == 1
    assert findings[0]["type"] == "underbilling"
    assert findings[0]["amount"] == "15000.00"
    assert findings[0]["recommended_action"] == "make_good_invoice"


def test_fx_overbilling_marked_already_corrected_by_credit_memo() -> None:
    """SUB-2014-A1 was billed 22500 EUR (=25200 USD) against a 24000 USD
    target; credit memo MEMO-701 already corrects the 1200 USD overbilling."""

    plan = _plan(
        "SUB-2014-A1",
        total_value=Decimal("96000"),
        cadence="Quarterly",
        start_date=date(2025, 8, 1),
    )
    invoices = [
        Invoice(
            invoice_id="INV-5022",
            plan_id="SUB-2014-A1",
            customer_name="Bluefin Logistics",
            issue_date=date(2025, 8, 12),
            due_date=date(2025, 9, 11),
            amount_invoiced=Decimal("22500"),
            currency="EUR",
            status="paid",
            description="Quarterly Route Optimizer subscription",
        )
    ]
    memo = CreditMemo(
        memo_id="MEMO-701",
        plan_id="SUB-2014-A1",
        invoice_id="INV-5022",
        amount=Decimal("1200"),
        currency="USD",
        issue_date=date(2025, 8, 25),
        reason="FX overbilling adjustment (EUR to USD)",
    )

    comparison = _compare(
        plan,
        invoices,
        credit_memos=[memo],
        exchange_rates=[EUR_USD_2025_08_12],
    )
    findings = comparison["findings"]

    assert len(findings) == 1
    assert findings[0]["type"] == "fx_mismatch"
    assert findings[0]["status"] == "already_corrected"
    assert findings[0]["amount"] == "1200.00"
    assert findings[0]["recommended_action"] == "none"


def test_month_end_start_does_not_drift_to_the_28th() -> None:
    """Each period starts at start_date + k months, not previous start + 1."""

    plan = _plan("SUB-EOM", Decimal("120000"), "Monthly", start_date=date(2025, 1, 31))
    invoices = [
        _invoice(f"INV-{day}", "SUB-EOM", day, Decimal("10000"))
        for day in ("2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30")
    ]

    comparison = _compare(plan, invoices)

    assert [row["period_start"] for row in comparison["periods"]] == [
        "2025-01-31",
        "2025-02-28",
        "2025-03-31",
        "2025-04-30",
    ]
    assert [row["period_end"] for row in comparison["periods"]] == [
        "2025-02-27",
        "2025-03-30",
        "2025-04-29",
        "2025-05-30",
    ]
    assert comparison["findings"] == []


def test_leap_day_annual_start_returns_to_feb_29_in_leap_years() -> None:
    plan = _plan("SUB-LEAP", Decimal("1000"), "Annual", start_date=date(2024, 2, 29))
    starts = ["2024-02-29", "2025-02-28", "2026-02-28", "2027-02-28", "2028-02-29"]
    invoices = [
        _invoice(f"INV-{day}", "SUB-LEAP", day, Decimal("1000")) for day in starts
    ]

    comparison = _compare(plan, invoices)

    assert [row["period_start"] for row in comparison["periods"]] == starts
    assert comparison["findings"] == []


def test_one_credit_memo_cannot_correct_two_overbilled_periods() -> None:
    """A memo covers only the period holding the invoice it references."""

    plan = _plan("SUB-OVER", Decimal("40000"), "Quarterly")
    invoices = [
        _invoice("INV-Q1", "SUB-OVER", "2025-01-10", Decimal("11200")),
        _invoice("INV-Q2", "SUB-OVER", "2025-04-10", Decimal("11200")),
    ]
    memo = CreditMemo(
        memo_id="MEMO-Q1",
        plan_id="SUB-OVER",
        invoice_id="INV-Q1",
        amount=Decimal("1200"),
        currency="USD",
        issue_date=date(2025, 1, 20),
        reason="Q1 overbilling",
    )

    findings = _compare(plan, invoices, credit_memos=[memo])["findings"]

    assert [(f["invoice_ids"], f["status"]) for f in findings] == [
        (["INV-Q1"], "already_corrected"),
        (["INV-Q2"], "overbilled"),
    ]


def test_converts_fx_with_documented_rounding_policy() -> None:
    result = convert_amount(
        amount=Decimal("22500"),
        from_ccy="EUR",
        to_ccy="USD",
        on_date=date(2025, 8, 12),
        exchange_rates=[EUR_USD_2025_08_12],
    )

    assert result["converted_amount"] == "25200.00"
    assert result["rounding_policy"] == "round half up to 2 decimal places"


def test_action_draft_adapter_discriminates_by_action_type() -> None:
    """The pending-action payload round-trips to the correct draft model."""

    payloads = [
        {
            "action_id": "DRAFT-MG-1",
            "action_type": "make_good_invoice",
            "plan_id": "SUB-2001",
            "amount": "10000",
            "currency": "USD",
            "reason": "Missing June.",
        },
        {
            "action_id": "DRAFT-CM-1",
            "action_type": "credit_memo",
            "invoice_id": "INV-5022",
            "amount": "1200",
            "currency": "USD",
            "reason": "FX overbilling.",
        },
        {
            "action_id": "DRAFT-PA-1",
            "action_type": "plan_amendment",
            "plan_id": "SUB-2014",
            "change_set": {"total_value": 96000},
            "reason": "Superseded contract.",
        },
    ]

    drafts = [ACTION_DRAFT_ADAPTER.validate_python(p) for p in payloads]

    assert isinstance(drafts[0], MakeGoodInvoiceDraft)
    assert isinstance(drafts[1], CreditMemoDraft)
    assert isinstance(drafts[2], PlanAmendmentDraft)


def _compare(
    plan: Plan,
    invoices: list[Invoice],
    *,
    credit_memos: list[CreditMemo] | None = None,
    exchange_rates: list[ExchangeRate] | None = None,
) -> dict[str, Any]:
    return compare_plan_to_invoices(
        plan=plan,
        invoices=invoices,
        exchange_rates=exchange_rates or [],
        credit_memos=credit_memos or [],
    )


def _plan(
    plan_id: str,
    total_value: Decimal,
    cadence: str,
    start_date: date | None = None,
) -> Plan:
    return Plan(
        plan_id=plan_id,
        customer_name="Test Customer",
        total_value=total_value,
        currency="USD",
        cadence=cadence,  # type: ignore[arg-type]
        start_date=start_date or date(2025, 1, 1),
        entitlements=["Analytics"],
    )


def _invoice(
    invoice_id: str,
    plan_id: str,
    issue_date: str,
    amount: Decimal,
) -> Invoice:
    issued = date.fromisoformat(issue_date)
    return Invoice(
        invoice_id=invoice_id,
        plan_id=plan_id,
        customer_name="Test Customer",
        issue_date=issued,
        due_date=issued,
        amount_invoiced=amount,
        currency="USD",
        status="paid",
        description="Test invoice",
    )
