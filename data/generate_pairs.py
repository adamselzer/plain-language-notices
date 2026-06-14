"""Generate synthetic (bureaucratic notice -> plain rewrite) pairs.

All notices are synthetic. No real notices or PII are used. Each pair carries its
operative facts (amount, dates, action, reason, appeal rights), so the faithfulness
gate has ground truth to check any rewrite against. The plain "target" rewrites are
written to a sixth-grade-ish reading level and, by construction, preserve every
operative fact, so they double as the gold references and as a sanity check on the
gate.

Writes data/pairs.jsonl (train) and data/holdout.jsonl (eval).

Run:  python data/generate_pairs.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema import NoticePair, NoticeType, OperativeFacts

DATA = Path(__file__).resolve().parent
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
APPEAL_DAYS = 90

ADVERSE_REASONS = [
    ("your household income exceeds the applicable income limit", "your income is above the limit"),
    ("a failure to provide requested verification within the prescribed period", "we did not get the documents we asked for"),
    ("a reported change in your household composition", "the number of people in your home changed"),
    ("the countable earned income reported for your household", "the pay you reported"),
]
VERIFY_ITEMS = [
    ("your current earned income", "your pay"),
    ("your residence and address", "where you live"),
    ("the members of your household", "who lives with you"),
]


def _date(rng: random.Random) -> str:
    return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, 2026"


def _reduction(rng, i) -> NoticePair:
    amt = rng.choice([112, 187, 245, 309, 421])
    eff = _date(rng)
    formal, plain = rng.choice(ADVERSE_REASONS)
    facts = OperativeFacts(notice_type=NoticeType.REDUCTION, benefit_amount=amt, effective_date=eff,
                           action="reduce", reason=formal, appeal_rights=True, appeal_deadline_days=APPEAL_DAYS)
    original = (
        "Pursuant to a review of your eligibility, this notice serves to inform you that your "
        f"Food Assistance Program benefit allotment shall be reduced to ${amt}.00 per month, "
        f"effective {eff}, said reduction being predicated upon {formal}. Should you wish to "
        f"contest this determination, you are entitled to request an administrative hearing within "
        f"{APPEAL_DAYS} days of the date of this notice."
    )
    target = (
        f"Your food assistance will go down to ${amt} a month starting {eff}. This is because "
        f"{plain}. If you think this is wrong, you can ask for a hearing within {APPEAL_DAYS} days."
    )
    return NoticePair(id=f"reduction-{i:03d}", notice_type=NoticeType.REDUCTION,
                      original=original, target=target, operative_facts=facts)


def _denial(rng, i) -> NoticePair:
    formal, plain = rng.choice(ADVERSE_REASONS)
    facts = OperativeFacts(notice_type=NoticeType.DENIAL, action="deny", reason=formal,
                           appeal_rights=True, appeal_deadline_days=APPEAL_DAYS)
    original = (
        "This correspondence constitutes formal notification that your application for Food "
        "Assistance Program benefits has been denied, said denial being predicated upon "
        f"{formal}. You are hereby advised of your right to request an administrative hearing "
        f"within {APPEAL_DAYS} days of the date of this notice."
    )
    target = (
        f"We can't approve your food assistance application. This is because {plain}. If you "
        f"disagree, you can ask for a hearing within {APPEAL_DAYS} days."
    )
    return NoticePair(id=f"denial-{i:03d}", notice_type=NoticeType.DENIAL,
                      original=original, target=target, operative_facts=facts)


def _recert(rng, i) -> NoticePair:
    deadline = _date(rng)
    facts = OperativeFacts(notice_type=NoticeType.RECERTIFICATION_DUE, response_deadline=deadline,
                           action="recertify", reason="your certification period is ending",
                           appeal_rights=False)
    original = (
        "Be advised that the certification period governing your receipt of Food Assistance "
        "Program benefits is set to expire. You are required to complete the redetermination "
        f"process on or before {deadline} in order to avoid an interruption in your benefits."
    )
    target = (
        f"Your food assistance is ending soon because your benefit period is ending. To keep "
        f"getting it, finish your renewal by {deadline}."
    )
    return NoticePair(id=f"recert-{i:03d}", notice_type=NoticeType.RECERTIFICATION_DUE,
                      original=original, target=target, operative_facts=facts)


def _verify(rng, i) -> NoticePair:
    deadline = _date(rng)
    formal, plain = rng.choice(VERIFY_ITEMS)
    facts = OperativeFacts(notice_type=NoticeType.VERIFICATION_REQUEST, response_deadline=deadline,
                           action="verify", reason=formal, appeal_rights=False)
    original = (
        f"In order to continue processing your case, you must furnish verification of {formal} "
        f"no later than {deadline}. Failure to provide the requested documentation by said date "
        "may result in denial or closure of your case."
    )
    target = (
        f"We need proof of {plain} to keep your case open. Please send it by {deadline}. If we "
        f"do not get it, your benefits could stop."
    )
    return NoticePair(id=f"verify-{i:03d}", notice_type=NoticeType.VERIFICATION_REQUEST,
                      original=original, target=target, operative_facts=facts)


def build(n_per_type: int = 15) -> list[NoticePair]:
    rng = random.Random(13)
    pairs: list[NoticePair] = []
    for i in range(n_per_type):
        pairs.append(_reduction(rng, i))
        pairs.append(_denial(rng, i))
        pairs.append(_recert(rng, i))
        pairs.append(_verify(rng, i))
    rng.shuffle(pairs)
    return pairs


def main() -> None:
    pairs = build()
    holdout = pairs[: len(pairs) // 4]
    train = pairs[len(pairs) // 4 :]
    (DATA / "pairs.jsonl").write_text("\n".join(p.model_dump_json() for p in train))
    (DATA / "holdout.jsonl").write_text("\n".join(p.model_dump_json() for p in holdout))
    print(f"Wrote {len(train)} train pairs and {len(holdout)} holdout pairs "
          f"across {len(set(p.notice_type for p in pairs))} notice types.")


if __name__ == "__main__":
    main()
