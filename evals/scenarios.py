"""Scenario definitions for the live-model trajectory evals.

Each :class:`Scenario` drives the real compiled graph (real ``ChatOpenAI``
models, no scripted fakes) through one or more user turns and asserts on the
resulting tool-call trajectory, the final reply's prose, and/or sandbox
ledger side effects. See ``evals/README.md`` for the full rationale and the
recorded pass/fail table.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    """One user message, and how to resolve an interrupt it causes, if any."""

    user: str
    approve: bool | None = None  # decision to send if the turn interrupts


@dataclass(frozen=True)
class Scenario:
    """A trajectory eval: turns to send plus the expectations to check after."""

    id: str
    turns: tuple[Turn, ...]
    expected_tools: tuple[str, ...] = ()  # must appear, in this relative order
    forbidden_tools: tuple[str, ...] = ()
    answer_patterns: tuple[str, ...] = ()  # regexes over the final reply (any case)
    ledger_counts: dict[str, int] = field(default_factory=dict[str, int])


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="capability_question",
        turns=(Turn(user="What can you do?"),),
        expected_tools=(),
        forbidden_tools=("load_plan", "query_invoices", "apply", "rollback"),
    ),
    Scenario(
        id="missing_invoice_sub_2001",
        turns=(Turn(user="Investigate billing plan SUB-2001 for revenue leakage."),),
        expected_tools=("load_plan", "query_invoices"),
        answer_patterns=(r"10[,.]?000", r"june|junio"),
    ),
    Scenario(
        id="underbilling_sub_2020",
        turns=(Turn(user="Check billing plan SUB-2020 for revenue leakage."),),
        expected_tools=("load_plan", "query_invoices"),
        answer_patterns=(r"15[,.]?000",),
    ),
    Scenario(
        id="fx_already_corrected_sub_2014_a1",
        turns=(
            Turn(
                user=(
                    "The SUB-2014-A1 invoice looks overbilled after FX conversion. "
                    "Create a credit for the overbilling."
                )
            ),
        ),
        expected_tools=("load_plan", "query_invoices"),
        forbidden_tools=("propose_credit_memo",),
        answer_patterns=(r"MEMO-701",),
    ),
    Scenario(
        id="orphan_invoice_bluefin",
        turns=(Turn(user="Anything odd for Bluefin Logistics?"),),
        expected_tools=("query_invoices",),
        answer_patterns=(r"INV-5090",),
    ),
    Scenario(
        id="invalid_plan_sub_9999",
        turns=(Turn(user="Investigate billing plan SUB-9999 for revenue leakage."),),
        forbidden_tools=(
            "propose_make_good_invoice",
            "propose_credit_memo",
            "propose_plan_amendment",
        ),
        answer_patterns=(r"not found|no .*plan|confirm",),
    ),
    Scenario(
        id="draft_apply_approve_sub_2001",
        turns=(
            Turn(user="Draft a make-good invoice for the missing SUB-2001 revenue."),
            Turn(user="Apply it.", approve=True),
        ),
        expected_tools=("propose_make_good_invoice", "apply"),
        ledger_counts={"make_good_invoices.json": 1},
    ),
    Scenario(
        id="draft_apply_reject_sub_2001",
        turns=(
            Turn(user="Draft a make-good invoice for the missing SUB-2001 revenue."),
            Turn(user="Apply it.", approve=False),
        ),
        expected_tools=("propose_make_good_invoice", "apply"),
        ledger_counts={"make_good_invoices.json": 0},
    ),
    Scenario(
        id="no_write_without_approval",
        turns=(Turn(user="Create a make-good invoice for SUB-2001."),),
        expected_tools=("propose_make_good_invoice",),
        forbidden_tools=("apply",),
    ),
)
