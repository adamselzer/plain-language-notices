# plain-language-notices

A small fine-tuned model that rewrites bureaucratic benefit notices into plain
language **while provably preserving the legally operative facts**: the amount,
the dates, the action, the reason, and the appeal rights.

This is the fine-tuning project in a four-part portfolio on AI in the public
benefits safety net. The real artifact here is judgment: knowing when to fine-tune
versus prompt versus retrieve, and building an eval that catches the failure mode
unique to this task: a notice that reads beautifully and is legally wrong.

> Incomprehensible government notices drive failed renewals and wrongful churn:
> people lose benefits they qualify for because they could not understand a letter.
> Rewriting notices in plain language helps, but only if the rewrite keeps every
> operative fact. A readable notice that drops the appeal deadline is worse than the
> jargon it replaced. So readability alone is never the metric.

## When to fine-tune (and when not), stated up front

- **Fine-tune here.** Rewriting notices is a fixed-format, fixed-tone task done at
  enormous volume (states send millions of notices). A small fine-tuned model holds
  a consistent sixth-grade voice that prompting drifts away from, and at per-notice
  scale it is far cheaper and lower-latency than calling a frontier model every time.
- **Do not fine-tune for policy Q&A.** That belongs in RAG (the sibling
  `policy-manual-rag` project), because the policy changes and you need citations,
  not weights that memorize a snapshot. Knowing this boundary is half the point.

## The hard part: readability is a trap

A rewrite can score at a sixth-grade level and still drop the appeal deadline or
misstate the benefit amount. So the eval pairs a readability score with a
**faithfulness gate** on the operative facts:

- The operative facts (amount, dates, action, appeal rights) are checked
  **deterministically and exactly** in `src/faithfulness.py`. A rewrite that drops
  or alters one **fails**, regardless of how well it reads.
- Preservation of the *reason* is semantic (a plain rewrite legitimately paraphrases
  it), so it is judged by the LLM in the eval rather than by brittle token matching.

That split (exact facts gated deterministically, meaning judged by an LLM) is the
design choice that makes the gate both strict and fair.

## Architecture

```
plain-language-notices/
├── data/
│   ├── generate_pairs.py     # synthetic (bureaucratic -> plain) notice pairs + operative facts
│   ├── pairs.jsonl           # training pairs
│   └── holdout.jsonl         # held-out eval pairs
├── src/
│   ├── schema.py             # Notice, OperativeFacts, faithfulness result models
│   ├── readability.py        # self-contained Flesch-Kincaid (no NLTK download)
│   ├── faithfulness.py       # THE GATE: exact operative-fact preservation
│   ├── baselines.py          # base-prompted and RAG baselines (Claude)
│   └── rewrite.py            # fine-tuned-model inference wrapper (loads the LoRA adapter)
├── train/
│   ├── finetune_lora.ipynb   # Colab/Kaggle QLoRA fine-tune (Unsloth + TRL)
│   └── config.yaml           # reproducible run config
├── eval/
│   └── run_eval.py           # four-way comparison: reference / base-prompted / RAG / fine-tuned
├── app/app.py                # paste-a-notice UI: rewrite + diff + preserved-facts checklist
└── tests/
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python data/generate_pairs.py        # synthetic pairs (no PII)
python eval/run_eval.py --offline    # reference (gold) readability + faithfulness, no key
pytest                               # 11 tests (gate + readability + data integrity)

# Train the adapter on a free Colab/Kaggle GPU:
#   open train/finetune_lora.ipynb, run it, download adapters/plain-notices-lora/

# Full four-way comparison (needs a key for the baselines, an adapter for fine-tuned):
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python eval/run_eval.py --with-finetuned
streamlit run app/app.py
```

The LoRA/QLoRA fine-tune runs on a free Colab/Kaggle GPU by design; the heavy
training libraries are not laptop dependencies, so the data pipeline, the gate, the
eval harness, and the tests all run locally with no GPU.

## Synthetic data only, never real PII

`data/generate_pairs.py` builds notice pairs across four types (denial, reduction,
recertification-due, verification-request). Each bureaucratic original is paired
with a plain target written to a low reading level, and each carries its operative
facts so the gate has ground truth. Synthetic only; no real notices are used.

## Evaluation

The four-way harness reports, per system, the target-grade hit rate, mean reading
grade, **faithfulness pass rate (the gate)**, mean latency, and mean cost.

Results on the 15 held-out notices (reference = gold targets; base-prompted and
RAG via Claude; fine-tuned pending the Colab run):

| System | Target-grade hit | Mean FK grade | Faithfulness pass | Cost/notice |
|---|---|---|---|---|
| reference (gold targets) | 93% | 4.3 | 100% | n/a |
| base-prompted | 20% | 8.5 | 100% | $0.0018 |
| RAG (plain-language guide in context) | 100% | 4.2 | 100% | $0.0019 |
| fine-tuned | _train via `train/finetune_lora.ipynb`_ | | | |

What this shows, and the case for fine-tuning: **plain prompting hits the target
reading level only 20% of the time** (mean grade 8.5, above the sixth-to-eighth
grade target); putting the plain-language style guide in context (RAG) fixes
readability completely (100%). The fine-tune's job is to match that readability
*without* carrying the guide in every prompt, at lower cost and latency across
millions of notices. That is the row the Colab training fills in.

Faithfulness is 100% across all systems here because the synthetic notices are
simple and the models are strong; the gate does its work on harder notices where a
model drops or alters an operative fact. A small human-rated sample (tone/clarity on
~20 examples) sanity-checks the automated scores.

## What I'd do differently at production scale

- **A semantic faithfulness judge in the gate.** The deterministic gate covers the
  exactly-checkable facts; production would add a calibrated LLM judge for reason
  and tone, validated against human ratings.
- **Real notice distributions.** The synthetic notices are templated; production
  notices vary by program, county, and era, and the fine-tune should train on a
  de-identified sample of the real ones.
- **A larger human-rated set** to anchor the automated readability and faithfulness
  scores, and per-notice-type breakdowns.
- **Quantized deployment** (the adapter merged and served at low cost) to realize
  the per-notice cost advantage the eval projects.
