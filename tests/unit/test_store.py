from __future__ import annotations

from decimal import Decimal

from revenue_leakage_agent.domain.models import (
    CreditMemoDraft,
    MakeGoodInvoiceDraft,
    PlanAmendmentDraft,
)
from revenue_leakage_agent.store import JsonStore


def test_json_store_applies_make_good_and_audit_log(store: JsonStore) -> None:
    draft = MakeGoodInvoiceDraft(
        action_id="DRAFT-MG-TEST",
        plan_id="SUB-2001",
        amount=Decimal("10000"),
        currency="USD",
        reason="Missing June 2025 billing.",
        evidence=["Expected 10000 USD; no invoice was found."],
    )

    invoice_record = store.append_make_good_invoice(draft.model_dump(mode="json"))
    audit_entry = store.append_audit_log(
        {"event_type": "action_applied", "action": draft.model_dump(mode="json")}
    )

    assert invoice_record["invoice_id"] == "INV-MG-0001"
    assert invoice_record["source_action_id"] == "DRAFT-MG-TEST"
    assert audit_entry["audit_id"] == "AUD-0001"
    assert store.get_action_already_applied("DRAFT-MG-TEST")


def test_json_store_applies_credit_memo(store: JsonStore) -> None:
    draft = CreditMemoDraft(
        action_id="DRAFT-CM-TEST",
        invoice_id="INV-5022",
        plan_id="SUB-2014-A1",
        amount=Decimal("1200"),
        currency="USD",
        reason="FX overbilling correction.",
        evidence=["25200 USD billed vs 24000 USD expected."],
    )

    record = store.append_credit_memo(draft.model_dump(mode="json"))

    assert record["memo_id"] == "CM-0001"
    assert record["invoice_id"] == "INV-5022"
    assert record["amount"] == "1200"
    assert store.get_action_already_applied("DRAFT-CM-TEST")


def test_json_store_applies_plan_amendment(store: JsonStore) -> None:
    draft = PlanAmendmentDraft(
        action_id="DRAFT-PA-TEST",
        plan_id="SUB-2014",
        change_set={"total_value": 96000, "cadence": "Quarterly"},
        reason="Superseded by SUB-2014-A1 from 2025-08-01.",
        evidence=["Amendment increases quarterly target to 24000 USD."],
    )

    record = store.append_plan_amendment(draft.model_dump(mode="json"))

    assert record["amendment_id"] == "AMD-0001"
    assert record["plan_id"] == "SUB-2014"
    assert record["change_set"]["total_value"] == 96000
    assert store.get_action_already_applied("DRAFT-PA-TEST")


def test_rollback_removes_sandbox_record_and_reopens_idempotency(
    store: JsonStore,
) -> None:
    """Rolling back an applied action deletes its ledger record so the same
    action can be proposed and applied again."""

    draft = CreditMemoDraft(
        action_id="DRAFT-CM-ROLL",
        invoice_id="INV-5022",
        plan_id="SUB-2014-A1",
        amount=Decimal("1200"),
        currency="USD",
        reason="FX overbilling correction.",
    )
    store.append_credit_memo(draft.model_dump(mode="json"))
    assert store.get_action_already_applied("DRAFT-CM-ROLL")

    removed = store.remove_sandbox_record("credit_memo", "DRAFT-CM-ROLL")

    assert removed is not None
    assert removed["source_action_id"] == "DRAFT-CM-ROLL"
    assert not store.get_action_already_applied("DRAFT-CM-ROLL")
    assert store.remove_sandbox_record("credit_memo", "DRAFT-CM-ROLL") is None


def test_reset_sandbox_removes_ledgers_and_audit_log_only(store: JsonStore) -> None:
    draft = CreditMemoDraft(
        action_id="DRAFT-CM-RESET",
        invoice_id="INV-5022",
        amount=Decimal("1200"),
        currency="USD",
        reason="FX overbilling correction.",
    )
    store.append_credit_memo(draft.model_dump(mode="json"))
    store.append_audit_log({"event_type": "action_applied"})
    store.data_dir.mkdir(parents=True)
    dataset_file = store.data_dir / "credit_memos.json"
    dataset_file.write_text("[]\n", encoding="utf-8")
    unrelated = store.sandbox_dir / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    removed = store.reset_sandbox()

    assert sorted(path.name for path in removed) == [
        "audit_log.json",
        "credit_memos.json",
    ]
    assert store.load_audit_log() == []
    assert not store.get_action_already_applied("DRAFT-CM-RESET")
    assert dataset_file.exists()
    assert unrelated.exists()
    assert store.reset_sandbox() == []


def test_load_audit_log_returns_entries_in_order(store: JsonStore) -> None:
    assert store.load_audit_log() == []

    store.append_audit_log({"event_type": "action_applied"})
    store.append_audit_log({"event_type": "action_rolled_back"})

    entries = store.load_audit_log()
    assert [entry["audit_id"] for entry in entries] == ["AUD-0001", "AUD-0002"]
    assert entries[1]["event_type"] == "action_rolled_back"
