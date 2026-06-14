"""Data models for notices, their operative facts, and the faithfulness gate.

The operative facts are the legally load-bearing content of a benefit notice: the
amount, the dates, what action is being taken, why, and the recipient's appeal
rights. A rewrite can read beautifully and still be unacceptable if it drops or
alters one of these. So they are modeled explicitly and checked, not assumed.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class NoticeType(str, Enum):
    DENIAL = "denial"
    REDUCTION = "reduction"
    RECERTIFICATION_DUE = "recertification_due"
    VERIFICATION_REQUEST = "verification_request"


class OperativeFacts(BaseModel):
    """The must-preserve facts of a notice. These are the gate."""

    notice_type: NoticeType
    benefit_amount: float | None = Field(default=None, description="Dollar amount central to the action, if any.")
    effective_date: str | None = Field(default=None, description="When the action takes effect, e.g. 'July 1, 2026'.")
    response_deadline: str | None = Field(default=None, description="Date by which the recipient must act.")
    action: str = Field(description="The action: deny / reduce / recertify / verify.")
    reason: str = Field(description="Why the action is being taken (short phrase).")
    appeal_rights: bool = Field(default=True, description="Whether the notice must state appeal/hearing rights.")
    appeal_deadline_days: int | None = Field(default=None, description="Days to request a hearing.")


class NoticePair(BaseModel):
    id: str
    notice_type: NoticeType
    original: str  # bureaucratic
    target: str  # plain-language reference rewrite
    operative_facts: OperativeFacts


class FactCheck(BaseModel):
    fact: str
    preserved: bool
    detail: str = ""


class FaithfulnessResult(BaseModel):
    passed: bool
    checks: list[FactCheck]

    @property
    def preserved_count(self) -> int:
        return sum(c.preserved for c in self.checks)


class RewriteResult(BaseModel):
    system: str  # "base_prompted" | "rag" | "fine_tuned"
    notice_id: str
    rewrite: str
    fk_grade: float
    faithfulness: FaithfulnessResult
    latency_s: float = 0.0
    cost_usd: float = 0.0
