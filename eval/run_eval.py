"""Four-way evaluation: reference / base-prompted / RAG / fine-tuned.

The headline pairs readability with a faithfulness gate. A rewrite that reads
beautifully but drops an operative fact is disqualified regardless of its reading
level, because in this domain that is the dangerous failure. The eval reports, per
system:

  - target-grade hit rate : fraction of rewrites at or below the target reading grade
  - mean FK grade         : average Flesch-Kincaid grade level
  - faithfulness pass rate : fraction preserving ALL operative facts (the gate)
  - mean latency / cost    : where the small fine-tune wins at scale

Systems run depend on what is available:
  - reference   : the gold plain targets (offline, always)
  - base_prompted, rag : need ANTHROPIC_API_KEY
  - fine_tuned  : needs a trained adapter under adapters/ (Colab/Kaggle GPU)

Run:  python eval/run_eval.py --offline           # reference only, no key
      python eval/run_eval.py                       # + base_prompted + rag (needs key)
      python eval/run_eval.py --with-finetuned      # + fine-tuned (needs adapter)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_GRADE, estimate_cost, has_llm
from src.faithfulness import check_faithfulness
from src.readability import flesch_kincaid_grade
from src.rewrite import DEFAULT_ADAPTER, FineTunedRewriter
from src.schema import NoticePair, RewriteResult

HOLDOUT = Path(__file__).resolve().parent.parent / "data" / "holdout.jsonl"
REPORT = Path(__file__).with_name("report.md")


def _load() -> list[NoticePair]:
    return [NoticePair(**json.loads(l)) for l in HOLDOUT.read_text().splitlines() if l.strip()]


def _score(system: str, pair: NoticePair, rewrite: str, latency: float, cost: float) -> RewriteResult:
    return RewriteResult(
        system=system, notice_id=pair.id, rewrite=rewrite,
        fk_grade=flesch_kincaid_grade(rewrite),
        faithfulness=check_faithfulness(pair.operative_facts, rewrite),
        latency_s=latency, cost_usd=cost,
    )


def evaluate_system(system: str, pairs: list[NoticePair]) -> list[RewriteResult]:
    results = []
    client = None
    rewriter = None
    if system in {"base_prompted", "rag"}:
        from src import baselines
        client = baselines._client()
    if system == "fine_tuned":
        rewriter = FineTunedRewriter()

    for p in pairs:
        if system == "reference":
            results.append(_score(system, p, p.target, 0.0, 0.0))
        elif system == "base_prompted":
            from src.baselines import base_prompted_rewrite
            text, dt, ti, to = base_prompted_rewrite(p.original, client)
            results.append(_score(system, p, text, dt, estimate_cost(ti, to)))
        elif system == "rag":
            from src.baselines import rag_rewrite
            text, dt, ti, to = rag_rewrite(p.original, client)
            results.append(_score(system, p, text, dt, estimate_cost(ti, to)))
        elif system == "fine_tuned":
            text, dt = rewriter.rewrite(p.original)
            results.append(_score(system, p, text, dt, 0.0))
    return results


def aggregate(system: str, results: list[RewriteResult]) -> dict:
    n = len(results)
    return {
        "system": system,
        "n": n,
        "target_grade_hit_rate": round(sum(r.fk_grade <= TARGET_GRADE for r in results) / n, 3),
        "mean_fk_grade": round(sum(r.fk_grade for r in results) / n, 2),
        "faithfulness_pass_rate": round(sum(r.faithfulness.passed for r in results) / n, 3),
        "mean_latency_s": round(sum(r.latency_s for r in results) / n, 3),
        "mean_cost_usd": round(sum(r.cost_usd for r in results) / n, 6),
    }


def main(argv: list[str]) -> int:
    pairs = _load()
    if "--sample" in argv:
        pairs = pairs[: int(argv[argv.index("--sample") + 1])]

    systems = ["reference"]
    offline = "--offline" in argv
    if not offline and has_llm():
        systems += ["base_prompted", "rag"]
    if "--with-finetuned" in argv and DEFAULT_ADAPTER.exists():
        systems.append("fine_tuned")

    rows = [aggregate(s, evaluate_system(s, pairs)) for s in systems]
    report = render(rows, len(pairs), systems)
    print(report)
    REPORT.write_text(report)
    return 0


def render(rows: list[dict], n: int, systems: list[str]) -> str:
    lines = [
        "# Eval report — plain-language-notices (four-way comparison)",
        "",
        f"Holdout notices: {n}. Target reading level: grade <= {TARGET_GRADE:.0f}.",
        f"Systems evaluated: {', '.join(systems)}.",
        "",
        "| System | Target-grade hit | Mean FK grade | Faithfulness pass | Mean latency | Mean cost |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['system']} | {r['target_grade_hit_rate']:.0%} | {r['mean_fk_grade']} | "
            f"{r['faithfulness_pass_rate']:.0%} | {r['mean_latency_s']}s | ${r['mean_cost_usd']:.5f} |"
        )
    lines += [
        "",
        "Faithfulness is the gate: a rewrite that drops or alters an operative fact "
        "(amount, date, action, appeal rights) fails regardless of how well it reads. "
        "The `reference` row is the gold plain targets and sets the ceiling. With a key, "
        "`base_prompted` and `rag` fill in; with a trained adapter, `fine_tuned` shows the "
        "cost/latency case for a small self-hosted model at notice-sending scale.",
        "",
    ]
    if systems == ["reference"]:
        lines.append("_Run without `--offline` (and with a key / adapter) to populate the model rows._")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
