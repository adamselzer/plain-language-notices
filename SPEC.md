# Project 3 — `plain-language-notices`

**Pattern:** Fine-tuning
**One line:** A small open model, fine-tuned to rewrite bureaucratic benefit notices into plain
language at a target reading level — **while provably preserving the legally operative facts.**

This is Project Re:Form as an FDE artifact. It's the textbook case for fine-tuning over prompting:
reliable format and tone at scale, where prompting only gets you ~80% of the way. But the real signal
is *judgment* — knowing when to fine-tune, and building an eval that catches the failure mode unique
to this task: a notice that's readable but legally wrong.

---

## Scope

**In scope**
- A dataset of (original bureaucratic notice → plain-language rewrite) pairs.
- A parameter-efficient (**LoRA / QLoRA**) fine-tune of a small open base model on a free Colab/Kaggle
  GPU.
- An eval that compares the fine-tune against three baselines: base model prompted, a RAG approach,
  and the fine-tune.
- A **faithfulness gate**: a check that the rewrite preserved the operative facts — the benefit
  amount, the deadline, the appeal rights, the reason for the action.
- A thin front-end: paste a notice → get the plain version, a diff, and a "preserved facts" checklist.

**Explicitly out of scope**
- No training on real notices containing real PII. Notices are synthetic or fully de-identified.

---

## Why fine-tune here (document this decision explicitly — it's half the grade)

Make the case in the README, with the eval to back it:

- **Format and tone consistency.** A bank-grade compliance voice, or a sixth-grade reading level,
  enforced on every notice — prompting drifts; a fine-tune holds.
- **Cost and latency at scale.** State agencies send millions of notices. A small fine-tuned model
  doing this per-notice is far cheaper than calling a frontier model every time. Put a number on it.
- **The honest counterpoint.** Also say where you *wouldn't* fine-tune — e.g., policy Q&A belongs in
  RAG (Project 2), not a fine-tune, because the policy changes and you want citations. Showing you
  know the boundary is the point.

---

## The hard part (this is what makes it senior)

**Readability is a trap.** A notice can score beautifully on reading level and still drop the appeal
deadline or misstate the benefit amount — which would be worse than the original. So the eval cannot
be readability alone. You need a **meaning-preservation check** on the operative facts:

- Extract the operative facts from the original (amount, dates, action, reason, appeal rights).
- Verify each survives, unchanged, in the rewrite.
- A rewrite that improves readability but loses or alters an operative fact **fails**, full stop.

Flagging this caveat unprompted is exactly the kind of thing that signals seniority in the interview.

---

## Architecture / repo structure

```
plain-language-notices/
├── README.md
├── data/
│   ├── generate_pairs.py        # synthetic (original → plain) notice pairs
│   ├── pairs.jsonl              # training pairs
│   ├── holdout.jsonl            # held-out eval set
│   └── operative_facts.py       # extracts the must-preserve facts per notice
├── train/
│   ├── finetune_lora.ipynb      # Colab-runnable LoRA/QLoRA fine-tune
│   └── config.yaml              # reproducible run config
├── src/
│   ├── baselines.py             # base-prompted and RAG baselines
│   ├── rewrite.py               # inference wrapper for the fine-tuned model
│   └── faithfulness.py          # operative-fact preservation check
├── app/                         # paste-a-notice UI: rewrite + diff + facts checklist
├── eval/
│   ├── run_eval.py              # 4-way comparison across quality/faithfulness/cost/latency
│   └── report.md
└── pyproject.toml
```

## Tech stack

- **Hugging Face PEFT** for LoRA/QLoRA adapters; **TRL** for the SFT loop.
- **Unsloth** for 2x-faster, low-memory training with ready-to-run free Colab notebooks — fork one.
  (**Axolotl** is the YAML-config alternative if you want fully reproducible config-driven runs.)
- A small open base model (an 8B-class instruct model is plenty; the value is in judgment, not GPU
  budget).
- Readability scoring (Flesch-Kincaid grade level or similar).

---

## Synthetic data

`generate_pairs.py` builds notice pairs: take templated bureaucratic notices (denial, reduction,
recertification-due, verification-request) stuffed with jargon and passive voice, and pair each with a
plain-language target written to a fixed reading level and voice. Keep `operative_facts.py` in sync so
every pair carries its must-preserve facts. Hold out a clean slice for eval. README: synthetic only,
no real notices.

---

## Evaluation (the differentiator)

`eval/run_eval.py` runs the **four-way comparison** on the holdout set and reports, for each of
{base-prompted, RAG, fine-tuned}:

- **Readability** (target grade level — hit rate and average).
- **Faithfulness / operative-fact preservation** — the gate. % of rewrites that preserved *all*
  operative facts. A rewrite that fails this is disqualified regardless of readability.
- **Cost per notice** and **latency per notice** — this is where the fine-tune wins at scale; show
  the math.
- A small **human-judgment** sample (you, rating tone/clarity on 20 examples) to sanity-check the
  automated scores.

The headline an interviewer wants: *"For policy Q&A I'd use RAG, not fine-tuning — but for rewriting
millions of notices, a small fine-tune is cheaper, more consistent, and here's the cost-per-notice
number that proves it. And here's the faithfulness gate that stops it from ever shipping a readable
notice that's legally wrong."*

---

## README framing

Connect it to the real-world problem (incomprehensible government notices drive failed renewals and
wrongful churn). State the fine-tune-vs-alternatives decision up front. Then the four-way eval table,
with the faithfulness gate explained as the thing that makes this safe to deploy.

## Interview one-liner

> "Readability alone is a trap — a notice can read beautifully and still drop the appeal deadline. So
> my eval pairs reading level with a faithfulness gate on the operative facts, and I benchmarked the
> fine-tune against prompting and RAG on quality, cost, and latency. The judgment is the artifact."

## Build order for Claude Code

1. `generate_pairs.py` + `operative_facts.py` + holdout split.
2. `baselines.py` (base-prompted + RAG) and `faithfulness.py` — so you can measure before training.
3. The LoRA fine-tune notebook + config.
4. `run_eval.py` four-way comparison + `report.md`.
5. Paste-a-notice front-end with diff + preserved-facts checklist.
