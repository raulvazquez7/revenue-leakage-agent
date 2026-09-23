from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel
from settings import AppSettings, get_settings

from agents.schemas import CreditMemo, ExchangeRate, Invoice, Plan

ModelT = TypeVar("ModelT", bound=BaseModel)

# Writable sandbox ledgers keyed by action type. Every applied record carries a
# ``source_action_id`` so idempotency checks and rollbacks can locate it.
SANDBOX_LEDGERS: dict[str, str] = {
    "make_good_invoice": "make_good_invoices.json",
    "credit_memo": "credit_memos.json",
    "plan_amendment": "plan_amendments.json",
}


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, date | datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class JsonStore:
    """Small JSON adapter around the read-only dataset and sandbox ledgers."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.data_dir = self.settings.data_dir
        self.sandbox_dir = self.settings.sandbox_dir

    def load_plans(self) -> list[Plan]:
        return self._read_models(self.data_dir / "billing_plans.json", Plan)

    def load_invoices(self) -> list[Invoice]:
        return self._read_models(self.data_dir / "invoices.json", Invoice)

    def load_credit_memos(self) -> list[CreditMemo]:
        return self._read_models(self.data_dir / "credit_memos.json", CreditMemo)

    def load_exchange_rates(self) -> list[ExchangeRate]:
        return self._read_models(self.data_dir / "exchange_rates.json", ExchangeRate)

    def get_plan(self, plan_id: str) -> Plan | None:
        normalized = plan_id.strip().upper()
        records = self._read_json_list(self.data_dir / "billing_plans.json")
        match = next(
            (
                record
                for record in records
                if str(record.get("plan_id", "")).strip().upper() == normalized
            ),
            None,
        )
        return Plan.model_validate(match) if match is not None else None

    def get_invoice(self, invoice_id: str) -> Invoice | None:
        normalized = invoice_id.strip().upper()
        records = self._read_json_list(self.data_dir / "invoices.json")
        match = next(
            (
                record
                for record in records
                if str(record.get("invoice_id", "")).strip().upper() == normalized
            ),
            None,
        )
        return Invoice.model_validate(match) if match is not None else None

    def get_action_already_applied(self, action_id: str) -> bool:
        """True when a live sandbox record exists for the action (survives rollback)."""

        for ledger in SANDBOX_LEDGERS.values():
            records = self._read_json_list(self.sandbox_dir / ledger)
            if any(record.get("source_action_id") == action_id for record in records):
                return True
        return False

    def append_make_good_invoice(self, draft: dict[str, Any]) -> dict[str, Any]:
        path = self._ledger_path("make_good_invoice")
        records = self._read_json_list(path)
        record = {
            "invoice_id": f"INV-MG-{len(records) + 1:04d}",
            "source_action_id": draft["action_id"],
            "plan_id": draft["plan_id"],
            "invoice_date": datetime.now(UTC).date().isoformat(),
            "amount": str(draft["amount"]),
            "currency": draft["currency"],
            "status": "issued",
            "reason": draft["reason"],
            "evidence": draft.get("evidence", []),
        }
        records.append(record)
        self._write_json(path, records)
        return record

    def append_credit_memo(self, draft: dict[str, Any]) -> dict[str, Any]:
        path = self._ledger_path("credit_memo")
        records = self._read_json_list(path)
        record = {
            "memo_id": f"CM-{len(records) + 1:04d}",
            "source_action_id": draft["action_id"],
            "invoice_id": draft["invoice_id"],
            "plan_id": draft.get("plan_id", ""),
            "issue_date": datetime.now(UTC).date().isoformat(),
            "amount": str(draft["amount"]),
            "currency": draft["currency"],
            "status": "issued",
            "reason": draft["reason"],
            "evidence": draft.get("evidence", []),
        }
        records.append(record)
        self._write_json(path, records)
        return record

    def append_plan_amendment(self, draft: dict[str, Any]) -> dict[str, Any]:
        path = self._ledger_path("plan_amendment")
        records = self._read_json_list(path)
        record = {
            "amendment_id": f"AMD-{len(records) + 1:04d}",
            "source_action_id": draft["action_id"],
            "plan_id": draft["plan_id"],
            "effective_date": datetime.now(UTC).date().isoformat(),
            "change_set": draft["change_set"],
            "status": "applied",
            "reason": draft["reason"],
            "evidence": draft.get("evidence", []),
        }
        records.append(record)
        self._write_json(path, records)
        return record

    def remove_sandbox_record(
        self,
        action_type: str,
        action_id: str,
    ) -> dict[str, Any] | None:
        """Delete the sandbox record for an applied action (used by rollback)."""

        path = self._ledger_path(action_type)
        records = self._read_json_list(path)
        removed = next(
            (r for r in records if r.get("source_action_id") == action_id),
            None,
        )
        if removed is None:
            return None
        remaining = [r for r in records if r.get("source_action_id") != action_id]
        self._write_json(path, remaining)
        return removed

    def append_audit_log(self, entry: dict[str, Any]) -> dict[str, Any]:
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        path = self.sandbox_dir / "audit_log.json"
        records = self._read_json_list(path)
        audit_entry = {
            "audit_id": f"AUD-{len(records) + 1:04d}",
            "created_at": datetime.now(UTC).isoformat(),
            **entry,
        }
        records.append(audit_entry)
        self._write_json(path, records)
        return audit_entry

    def _ledger_path(self, action_type: str) -> Path:
        ledger = SANDBOX_LEDGERS.get(action_type)
        if ledger is None:
            raise ValueError(f"Unknown sandbox action type: {action_type}")
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        return self.sandbox_dir / ledger

    def _read_models(self, path: Path, model: type[ModelT]) -> list[ModelT]:
        payload = self._read_json_list(path)
        return [model.model_validate(item) for item in payload]

    def _read_json_list(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        payload_raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload_raw, list):
            raise ValueError(f"Expected a JSON list in {path}")
        payload = cast(list[Any], payload_raw)
        return [self._ensure_mapping(item, path) for item in payload]

    def _write_json(self, path: Path, payload: Any) -> None:
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)

    @staticmethod
    def _ensure_mapping(item: Any, path: Path) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError(f"Expected object entries in {path}")
        return cast(dict[str, Any], item)
