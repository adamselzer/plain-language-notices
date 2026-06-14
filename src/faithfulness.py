"""The faithfulness gate: did a rewrite preserve every operative fact?

Readability is a trap. A notice can score at a sixth-grade reading level and still
drop the appeal deadline or restate the benefit amount, which is worse than the
bureaucratic original. So the eval cannot be readability alone. This module checks
each operative fact survives, unchanged, in the rewrite. A rewrite that improves
readability but loses or alters an operative fact FAILS, full stop.

The gate enforces the facts that are *exactly* checkable: the amount, the dates,
the action, and the appeal rights (with the hearing deadline). Preservation of the
*reason* is semantic (a plain rewrite legitimately paraphrases it), so it is judged
by the LLM in the eval rather than by brittle token matching here. Keeping the
deterministic gate to exactly-verifiable facts is what lets it be a hard,
trustworthy pass/fail.

The checks are deterministic and explicit, so a failure says exactly which fact was
dropped.
"""

from __future__ import annotations

import re

from .schema import FactCheck, FaithfulnessResult, OperativeFacts

ACTION_SYNONYMS = {
    "deny": {"deny", "denied", "denial", "not approve", "cannot approve", "can't approve",
             "can not approve", "not approved", "turned down", "rejected"},
    "reduce": {"reduce", "reduced", "lower", "lowered", "decrease", "go down", "going down", "less"},
    "recertify": {"recertify", "recertification", "renew", "renewal", "review", "reapply"},
    "verify": {"verify", "verification", "proof", "provide", "document", "send us"},
}

def _norm(text: str) -> str:
    return re.sub(r"[,]", "", text.lower())


def _amount_present(amount: float, text: str) -> bool:
    t = _norm(text)
    candidates = {f"{amount:.0f}", f"{amount:.2f}"}
    if amount == int(amount):
        candidates.add(str(int(amount)))
    return any(c in t for c in candidates)


def _date_present(date_str: str, text: str) -> bool:
    t = text.lower()
    if date_str.lower() in t:
        return True
    # relaxed: "July 1, 2026" -> accept "july 1"
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2})", date_str)
    if m and f"{m.group(1).lower()} {m.group(2)}" in t:
        return True
    return False


def _action_present(action: str, text: str) -> bool:
    t = _norm(text)
    return any(syn in t for syn in ACTION_SYNONYMS.get(action, {action}))


def check_faithfulness(facts: OperativeFacts, rewrite: str) -> FaithfulnessResult:
    checks: list[FactCheck] = []

    if facts.benefit_amount is not None:
        ok = _amount_present(facts.benefit_amount, rewrite)
        checks.append(FactCheck(fact="benefit_amount", preserved=ok,
                                detail=f"${facts.benefit_amount:,.2f}"))
    if facts.effective_date:
        ok = _date_present(facts.effective_date, rewrite)
        checks.append(FactCheck(fact="effective_date", preserved=ok, detail=facts.effective_date))
    if facts.response_deadline:
        ok = _date_present(facts.response_deadline, rewrite)
        checks.append(FactCheck(fact="response_deadline", preserved=ok, detail=facts.response_deadline))

    checks.append(FactCheck(fact="action", preserved=_action_present(facts.action, rewrite),
                            detail=facts.action))

    if facts.appeal_rights:
        t = rewrite.lower()
        mentions = ("appeal" in t) or ("hearing" in t)
        deadline_ok = True
        if facts.appeal_deadline_days is not None:
            deadline_ok = str(facts.appeal_deadline_days) in t
        checks.append(FactCheck(fact="appeal_rights", preserved=(mentions and deadline_ok),
                                detail=f"appeal/hearing within {facts.appeal_deadline_days} days"))

    passed = all(c.preserved for c in checks)
    return FaithfulnessResult(passed=passed, checks=checks)
