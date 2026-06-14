# Portfolio notes — plain-language-notices

A plain-language account of what this project is and the judgment behind it.

## Pattern

**Fine-tuning.** A parameter-efficient (LoRA/QLoRA) fine-tune of a small open model
on a free GPU, with a domain-aware eval.

## Concept demonstrated

Fine-tuning judgment and domain-aware eval design: choosing the right tool (and
saying where I would not fine-tune), and designing a metric that catches the
dangerous failure a readability score misses. The artifact is not the adapter, it
is the judgment around it.

## Why it matters in this domain

Unreadable benefit notices cause real harm: people lose coverage they qualify for
because they could not parse a letter. Plain-language rewriting helps at the scale
of millions of notices, which is exactly where a small, consistent, cheap fine-tune
beats calling a frontier model per notice. But a rewrite that improves readability
while dropping the appeal deadline or changing the amount is worse than the
original. So the project is built around a faithfulness gate, not a readability
score.

## Key design decisions and tradeoffs

1. **Fine-tune for this, RAG for policy Q&A.** Documented up front: notice
   rewriting is fixed-format, fixed-tone, high-volume, which a fine-tune holds
   better and cheaper than prompting; policy Q&A belongs in RAG because policy
   changes and needs citations. *Rejected:* fine-tuning everything, or prompting
   everything. Naming the boundary is the senior signal.

2. **A deterministic faithfulness gate on the exactly-checkable facts.** Amount,
   dates, action, and appeal rights are verified by exact matching; a drop or change
   fails outright. *Rejected:* judging faithfulness entirely with an LLM. The
   legally operative facts are exactly checkable, and a deterministic gate is a
   harder, more trustworthy pass/fail than a model's opinion.

3. **Reason preservation judged semantically, not by tokens.** A plain rewrite
   paraphrases the reason, so token overlap would produce false failures. *Rejected:*
   forcing the reason through the deterministic gate (an earlier version did, and it
   failed faithful targets). Reason goes to the LLM judge; the gate stays exact.

4. **A self-contained readability metric.** Flesch-Kincaid implemented directly
   rather than via a library that downloads an NLTK corpus at runtime. *Rejected:*
   the textstat/NLTK path, which fails in locked-down environments. Reproducibility
   beats a marginally more accurate syllable count.

5. **Training kept off the laptop.** The QLoRA fine-tune runs on a free Colab/Kaggle
   GPU; the heavy libraries are not repo dependencies. *Rejected:* listing
   torch/unsloth as install requirements. The data pipeline, gate, eval, and tests
   must run on any machine; only the training step needs a GPU, and it lives in a
   notebook.

## How it's evaluated

A four-way comparison (reference / base-prompted / RAG / fine-tuned) on the holdout
set, reporting target-grade hit rate, mean reading grade, the faithfulness pass
rate, and mean latency and cost. The gate is the headline: a rewrite that fails it
is disqualified regardless of readability. The offline reference run (gold targets:
93% target-grade, 100% faithfulness) validates the harness and the gate without a
key; the model rows fill in with a key and a trained adapter, where the fine-tune's
cost-per-notice advantage at scale is the argument.

These metrics map to the stake: readability is the benefit, faithfulness is the
safety constraint, and cost-per-notice is why a small fine-tune is the right tool
for this job and not a frontier-model API call.

## What I'd do differently at production scale

- Add a calibrated LLM judge for reason and tone alongside the deterministic gate.
- Train on a de-identified sample of real notices across programs and counties.
- Build a larger human-rated set to anchor the automated scores.
- Merge and quantize the adapter for low-cost serving to realize the projected
  per-notice savings.
