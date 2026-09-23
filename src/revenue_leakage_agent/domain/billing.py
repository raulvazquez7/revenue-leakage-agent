"""Deterministic expected-vs-actual billing calculations.

This module is intentionally not model-backed. It compares plans, invoices,
periods, currencies, and FX rates so the LLM can explain verified evidence
instead of inventing financial facts.

The dataset expresses a plan as a contract ``total_value`` plus a
``cadence``; there is no per-period amount and invoices carry no billing
period. We therefore derive the expected per-period amount as
``total_value / periods_per_year`` and reconstruct periods from the plan
``start_date``, mapping each invoice to a period by its ``issue_date``.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from revenue_leakage_agent.domain.models import (
    CreditMemo,
    ExchangeRate,
    Finding,
    Invoice,
    InvoiceFilters,
    Plan,
)

MONEY_QUANT = Decimal("0.01")

PERIODS_PER_YEAR: dict[str, int] = {"Monthly": 12, "Quarterly": 4, "Annual": 1}
CADENCE_MONTHS: dict[str, int] = {"Monthly": 1, "Quarterly": 3, "Annual": 12}


def apply_invoice_filters(
    invoices: list[Invoice],
    filters: InvoiceFilters,
) -> tuple[list[Invoice], bool]:
    filtered = [
        invoice
        for invoice in invoices
        if _matches_filters(invoice=invoice, filters=filters)
    ]
    truncated = len(filtered) > filters.limit
    return filtered[: filters.limit], truncated


def summarize_invoices(invoices: list[Invoice]) -> dict[str, Any]:
    totals: dict[str, Decimal] = {}
    status_counts: dict[str, int] = {}
    for invoice in invoices:
        totals[invoice.currency] = (
            totals.get(invoice.currency, Decimal("0")) + invoice.amount_invoiced
        )
        status_counts[invoice.status] = status_counts.get(invoice.status, 0) + 1

    return {
        "totals_by_currency": {
            currency: _money(amount) for currency, amount in sorted(totals.items())
        },
        "status_counts": dict(sorted(status_counts.items())),
    }


def related_credit_memos(
    credit_memos: list[CreditMemo],
    invoices: list[Invoice],
    plan_id: str | None,
) -> list[dict[str, Any]]:
    invoice_ids = {invoice.invoice_id for invoice in invoices}
    related = [
        memo
        for memo in credit_memos
        if memo.invoice_id in invoice_ids or (plan_id and memo.plan_id == plan_id)
    ]
    return [memo.model_dump(mode="json") for memo in related]


def compare_plan_to_invoices(
    plan: Plan,
    invoices: list[Invoice],
    exchange_rates: list[ExchangeRate],
    credit_memos: list[CreditMemo],
) -> dict[str, Any]:
    """Compare every invoice of ``plan`` against its expected billing periods.

    The comparison always covers the whole plan: ``query_invoices`` filters
    only narrow the invoice list it returns, not the periods checked here.
    """

    plan_invoices = [inv for inv in invoices if inv.plan_id == plan.plan_id]
    expected = _money(plan.total_value / PERIODS_PER_YEAR[plan.cadence])
    periods = _expected_periods(plan=plan, invoices=plan_invoices)

    assignments = [
        [inv for inv in plan_invoices if start <= inv.issue_date < end]
        for start, end in periods
    ]
    nonempty_indices = [i for i, group in enumerate(assignments) if group]
    first_billed = nonempty_indices[0] if nonempty_indices else None
    last_billed = nonempty_indices[-1] if nonempty_indices else None

    period_rows: list[dict[str, Any]] = []
    findings: list[Finding] = []

    for index, ((period_start, period_end), period_invoices) in enumerate(
        zip(periods, assignments, strict=True)
    ):
        display_end = _period_display_end(period_end)
        interior = (
            first_billed is not None
            and last_billed is not None
            and first_billed <= index <= last_billed
        )
        actual, fx_involved, evidence = _actual_amount_in_plan_currency(
            invoices=period_invoices,
            plan=plan,
            exchange_rates=exchange_rates,
        )
        delta = _money(expected - actual)
        status = _period_status(
            delta=delta,
            has_invoices=bool(period_invoices),
            interior=interior,
        )

        period_rows.append(
            {
                "period_start": period_start.isoformat(),
                "period_end": display_end.isoformat(),
                "expected_amount": str(expected),
                "actual_amount": str(actual),
                "delta": str(delta),
                "currency": plan.currency,
                "invoice_ids": [inv.invoice_id for inv in period_invoices],
                "fx_involved": fx_involved,
                "status": status,
                "evidence": evidence,
            }
        )

        finding = _build_finding(
            plan=plan,
            period_start=period_start,
            display_end=display_end,
            period_invoices=period_invoices,
            expected=expected,
            actual=actual,
            delta=delta,
            status=status,
            fx_involved=fx_involved,
            evidence=evidence,
            credit_memos=credit_memos,
            exchange_rates=exchange_rates,
        )
        if finding is not None:
            findings.append(finding)

    return {
        "plan_id": plan.plan_id,
        "currency": plan.currency,
        "cadence": plan.cadence,
        "expected_per_period": str(expected),
        "amends": plan.amends,
        "periods": period_rows,
        "findings": [finding.model_dump(mode="json") for finding in findings],
    }


def convert_amount(
    amount: Decimal,
    from_ccy: str,
    to_ccy: str,
    on_date: date,
    exchange_rates: list[ExchangeRate],
) -> dict[str, Any]:
    if from_ccy == to_ccy:
        return {
            "original_amount": str(_money(amount)),
            "from_ccy": from_ccy,
            "to_ccy": to_ccy,
            "rate": "1",
            "converted_amount": str(_money(amount)),
            "on_date": on_date.isoformat(),
            "rounding_policy": "round half up to 2 decimal places",
        }

    rate = next(
        (
            exchange_rate
            for exchange_rate in exchange_rates
            if exchange_rate.from_currency == from_ccy
            and exchange_rate.to_currency == to_ccy
            and exchange_rate.date == on_date
        ),
        None,
    )
    if rate is None:
        raise ValueError(
            f"No FX rate for {from_ccy}->{to_ccy} on {on_date.isoformat()}"
        )

    converted = _money(amount * rate.rate)
    return {
        "original_amount": str(_money(amount)),
        "from_ccy": from_ccy,
        "to_ccy": to_ccy,
        "rate": str(rate.rate),
        "converted_amount": str(converted),
        "on_date": on_date.isoformat(),
        "rounding_policy": "round half up to 2 decimal places",
    }


def _matches_filters(invoice: Invoice, filters: InvoiceFilters) -> bool:
    return (
        (filters.plan_id is None or invoice.plan_id == filters.plan_id.strip().upper())
        and (
            filters.customer_name is None
            or invoice.customer_name == filters.customer_name
        )
        and (filters.start_date is None or invoice.issue_date >= filters.start_date)
        and (filters.end_date is None or invoice.issue_date <= filters.end_date)
        and (filters.status is None or invoice.status == filters.status)
        and (filters.currency is None or invoice.currency == filters.currency)
    )


def _expected_periods(
    plan: Plan,
    invoices: list[Invoice],
) -> list[tuple[date, date]]:
    """Reconstruct [start, next_start) periods from the plan start to the last invoice.

    Returns empty when the plan has no invoices, because there is no observed
    horizon to bound period generation and we do not fabricate obligations.
    """

    if not invoices:
        return []

    months = CADENCE_MONTHS[plan.cadence]
    horizon = max(invoice.issue_date for invoice in invoices)

    # Period k starts at start_date + k * months (clamped to month end), so a
    # Jan-31 start gives Feb-28, Mar-31, Apr-30 rather than drifting to the 28th.
    periods: list[tuple[date, date]] = []
    k = 0
    start = plan.start_date
    while start <= horizon:
        next_start = _add_months(plan.start_date, (k + 1) * months)
        periods.append((start, next_start))
        k += 1
        start = next_start
    return periods


def _actual_amount_in_plan_currency(
    invoices: list[Invoice],
    plan: Plan,
    exchange_rates: list[ExchangeRate],
) -> tuple[Decimal, bool, list[str]]:
    total = Decimal("0")
    fx_involved = False
    evidence: list[str] = []
    for invoice in invoices:
        if invoice.currency == plan.currency:
            amount = _money(invoice.amount_invoiced)
            evidence.append(
                f"{invoice.invoice_id}: {amount} {plan.currency} ({invoice.status})."
            )
        else:
            fx_involved = True
            conversion = convert_amount(
                amount=invoice.amount_invoiced,
                from_ccy=invoice.currency,
                to_ccy=plan.currency,
                on_date=invoice.issue_date,
                exchange_rates=exchange_rates,
            )
            amount = Decimal(conversion["converted_amount"])
            evidence.append(
                f"{invoice.invoice_id}: {invoice.amount_invoiced} {invoice.currency} x "
                f"FX rate {conversion['rate']} on {conversion['on_date']} = "
                f"{amount} {plan.currency} ({conversion['rounding_policy']})."
            )
        total += amount
    return _money(total), fx_involved, evidence


def _build_finding(
    plan: Plan,
    period_start: date,
    display_end: date,
    period_invoices: list[Invoice],
    expected: Decimal,
    actual: Decimal,
    delta: Decimal,
    status: str,
    fx_involved: bool,
    evidence: list[str],
    credit_memos: list[CreditMemo],
    exchange_rates: list[ExchangeRate],
) -> Finding | None:
    if status == "ok":
        return None

    invoice_ids = [inv.invoice_id for inv in period_invoices]
    period_label = f"{period_start.isoformat()} to {display_end.isoformat()}"

    if status == "missing_invoice":
        return Finding(
            finding_id=f"F-{plan.plan_id}-{period_start.isoformat()}-missing",
            plan_id=plan.plan_id,
            invoice_ids=[],
            period_start=period_start,
            period_end=display_end,
            type="missing_invoice",
            status="leakage",
            amount=expected,
            currency=plan.currency,
            evidence=[
                f"Expected {expected} {plan.currency} for {period_label}.",
                "No invoice was issued for that billing period.",
            ],
            recommended_action="make_good_invoice",
        )

    finding_type = (
        "fx_mismatch"
        if fx_involved
        else ("underbilling" if delta > 0 else "overbilling")
    )
    base_evidence = [
        f"Expected {expected} {plan.currency}; actual was {actual} {plan.currency} "
        f"for {period_label}.",
        *evidence,
    ]

    if delta > 0:  # underbilling
        return Finding(
            finding_id=f"F-{plan.plan_id}-{period_start.isoformat()}-under",
            plan_id=plan.plan_id,
            invoice_ids=invoice_ids,
            period_start=period_start,
            period_end=display_end,
            type=finding_type,
            status="leakage",
            amount=delta,
            currency=plan.currency,
            evidence=base_evidence,
            recommended_action="make_good_invoice",
        )

    # overbilling
    over = delta.copy_abs()
    covered = _credit_memo_coverage(
        credit_memos=credit_memos,
        invoice_ids=invoice_ids,
        plan=plan,
        overbilled=over,
        exchange_rates=exchange_rates,
    )
    if covered is not None:
        return Finding(
            finding_id=f"F-{plan.plan_id}-{period_start.isoformat()}-corrected",
            plan_id=plan.plan_id,
            invoice_ids=invoice_ids,
            period_start=period_start,
            period_end=display_end,
            type=finding_type,
            status="already_corrected",
            amount=over,
            currency=plan.currency,
            evidence=[*base_evidence, covered],
            recommended_action="none",
        )

    return Finding(
        finding_id=f"F-{plan.plan_id}-{period_start.isoformat()}-over",
        plan_id=plan.plan_id,
        invoice_ids=invoice_ids,
        period_start=period_start,
        period_end=display_end,
        type=finding_type,
        status="overbilled",
        amount=over,
        currency=plan.currency,
        evidence=base_evidence,
        recommended_action="credit_memo",
    )


def _credit_memo_coverage(
    credit_memos: list[CreditMemo],
    invoice_ids: list[str],
    plan: Plan,
    overbilled: Decimal,
    exchange_rates: list[ExchangeRate],
) -> str | None:
    """Return an evidence string when existing credit memos cover the overbilling.

    A memo counts only for the period holding the invoice it references, so
    one memo can't correct several overbilled periods of the same plan.
    ``CreditMemo.invoice_id`` is required, so there are no plan-level memos.
    """

    invoice_id_set = set(invoice_ids)
    total_credit = Decimal("0")
    matched: list[str] = []
    for memo in credit_memos:
        if memo.invoice_id not in invoice_id_set:
            continue
        if memo.currency == plan.currency:
            amount = _money(memo.amount)
        else:
            conversion = convert_amount(
                amount=memo.amount,
                from_ccy=memo.currency,
                to_ccy=plan.currency,
                on_date=memo.issue_date,
                exchange_rates=exchange_rates,
            )
            amount = Decimal(conversion["converted_amount"])
        total_credit += amount
        matched.append(f"{memo.memo_id} ({amount} {plan.currency})")

    if matched and total_credit >= overbilled:
        return (
            f"Existing credit memo already corrects the overbilling: "
            f"{', '.join(matched)}."
        )
    return None


def _period_status(delta: Decimal, has_invoices: bool, interior: bool) -> str:
    if not has_invoices:
        return "missing_invoice" if interior else "ok"
    if delta == Decimal("0.00"):
        return "ok"
    return "underbilling" if delta > 0 else "overbilling"


def _period_display_end(next_start: date) -> date:
    return _add_days(next_start, -1)


def _add_days(value: date, days: int) -> date:
    return date.fromordinal(value.toordinal() + days)


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, monthrange(year, month)[1])
    return date(year, month, day)


def _money(amount: Decimal) -> Decimal:
    return amount.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)
