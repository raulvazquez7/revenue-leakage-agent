"""Golden-dataset regression test for the deterministic billing comparison.

Locks in the expected findings for every plan shipped in the repo's sample
``data/`` directory, plus the orphan-invoice behaviour of ``query_invoices``.
No model or graph is involved: this exercises the pure domain layer that the
tools call, so a change to ``compare_plan_to_invoices`` that alters real
findings is caught immediately.
"""

from __future__ import annotations

from revenue_leakage_agent.domain.billing import (
    apply_invoice_filters,
    compare_plan_to_invoices,
)
from revenue_leakage_agent.domain.models import InvoiceFilters
from revenue_leakage_agent.store import JsonStore

EXPECTED_FINDINGS: dict[str, list[tuple[str, str, str]]] = {
    "SUB-2001": [("missing_invoice", "10000.00", "leakage")],
    "SUB-2014": [],
    "SUB-2014-A1": [("fx_mismatch", "1200.00", "already_corrected")],
    "SUB-2020": [("underbilling", "15000.00", "leakage")],
    "SUB-2033": [],
}


def test_compare_plan_to_invoices_matches_golden_findings_for_every_plan(
    seeded_store: JsonStore,
) -> None:
    plans = seeded_store.load_plans()
    invoices = seeded_store.load_invoices()
    exchange_rates = seeded_store.load_exchange_rates()
    credit_memos = seeded_store.load_credit_memos()

    assert {plan.plan_id for plan in plans} == set(EXPECTED_FINDINGS)

    for plan in plans:
        comparison = compare_plan_to_invoices(
            plan=plan,
            invoices=invoices,
            exchange_rates=exchange_rates,
            credit_memos=credit_memos,
        )
        actual = [
            (finding["type"], finding["amount"], finding["status"])
            for finding in comparison["findings"]
        ]
        assert actual == EXPECTED_FINDINGS[plan.plan_id], plan.plan_id


def test_query_invoices_includes_orphan_invoice_for_bluefin_logistics(
    seeded_store: JsonStore,
) -> None:
    invoices = seeded_store.load_invoices()
    filters = InvoiceFilters(customer_name="Bluefin Logistics")

    matched, truncated = apply_invoice_filters(invoices, filters)

    assert not truncated
    orphan = next(inv for inv in matched if inv.invoice_id == "INV-5090")
    assert orphan.plan_id == ""
