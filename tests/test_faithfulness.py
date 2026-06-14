"""Tests for the faithfulness gate and readability scoring.

The gate is the core artifact, so these are thorough: each operative fact, when
dropped or altered, must fail; faithful paraphrases must pass.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.faithfulness import check_faithfulness
from src.readability import flesch_kincaid_grade
from src.schema import NoticePair, NoticeType, OperativeFacts

DATA = Path(__file__).resolve().parent.parent / "data"


def _reduction_facts():
    return OperativeFacts(
        notice_type=NoticeType.REDUCTION, benefit_amount=187, effective_date="July 1, 2026",
        action="reduce", reason="your income is above the limit", appeal_rights=True,
        appeal_deadline_days=90,
    )


GOOD = ("Your food assistance will go down to $187 a month starting July 1, 2026. This is "
        "because your income is above the limit. You can ask for a hearing within 90 days.")


def test_faithful_rewrite_passes():
    assert check_faithfulness(_reduction_facts(), GOOD).passed


def test_dropped_amount_fails():
    text = GOOD.replace("$187", "a new amount")
    r = check_faithfulness(_reduction_facts(), text)
    assert not r.passed
    assert any(c.fact == "benefit_amount" and not c.preserved for c in r.checks)


def test_altered_amount_fails():
    text = GOOD.replace("$187", "$210")
    assert not check_faithfulness(_reduction_facts(), text).passed


def test_dropped_date_fails():
    text = GOOD.replace("starting July 1, 2026", "soon")
    r = check_faithfulness(_reduction_facts(), text)
    assert any(c.fact == "effective_date" and not c.preserved for c in r.checks)
    assert not r.passed


def test_dropped_appeal_deadline_fails():
    text = GOOD.replace("within 90 days", "")
    assert not check_faithfulness(_reduction_facts(), text).passed


def test_missing_appeal_rights_fails():
    text = "Your food assistance will go down to $187 a month starting July 1, 2026."
    assert not check_faithfulness(_reduction_facts(), text).passed


def test_action_synonym_accepted():
    # "go down" is an accepted synonym for the reduce action
    assert any(c.fact == "action" and c.preserved
               for c in check_faithfulness(_reduction_facts(), GOOD).checks)


def test_paraphrased_reason_still_passes():
    # reason is NOT a blocking deterministic check (it is semantic); paraphrase is fine
    text = GOOD.replace("your income is above the limit", "you earn too much to qualify")
    assert check_faithfulness(_reduction_facts(), text).passed


def test_non_adverse_notice_skips_appeal_check():
    facts = OperativeFacts(notice_type=NoticeType.VERIFICATION_REQUEST, response_deadline="May 5, 2026",
                           action="verify", reason="your pay", appeal_rights=False)
    text = "We need proof of your pay. Please send it by May 5, 2026."
    assert check_faithfulness(facts, text).passed


# --- readability ------------------------------------------------------------


def test_readability_orders_correctly():
    bureaucratic = ("Pursuant to the aforementioned determination, your benefit allotment shall be "
                    "reduced effective the subsequent month in accordance with applicable regulations.")
    plain = "Your help will go down next month."
    assert flesch_kincaid_grade(bureaucratic) > flesch_kincaid_grade(plain)


# --- data integrity ---------------------------------------------------------


def test_all_targets_pass_gate_and_hit_reading_level():
    pairs = []
    for f in ("pairs.jsonl", "holdout.jsonl"):
        pairs += [NoticePair(**json.loads(l)) for l in (DATA / f).read_text().splitlines() if l.strip()]
    assert len(pairs) >= 40
    assert all(check_faithfulness(p.operative_facts, p.target).passed for p in pairs)
    # most targets should be well below college reading level
    assert sum(flesch_kincaid_grade(p.target) <= 8 for p in pairs) / len(pairs) >= 0.85
