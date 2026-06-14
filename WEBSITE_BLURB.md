# Website blurb — plain-language-notices

Drop-in copy for a portfolio page. A short paragraph plus highlight bullets.

---

## plain-language-notices

A small fine-tuned model that rewrites bureaucratic benefit notices into plain
language while preserving every legally operative fact. Unreadable government
notices cause real harm: people lose coverage they qualify for because they could
not parse a letter. Rewriting helps at the scale of millions of notices, which is
where a small, consistent, cheap fine-tune beats calling a frontier model each time.
But the real work is the eval. Readability alone is a trap, because a notice can
read beautifully and still drop the appeal deadline, so I pair a reading-level score
with a faithfulness gate that checks the amount, the dates, the action, and the
appeal rights survive exactly. A rewrite that reads well but loses a fact fails. It
is the fine-tuning project in a four-part portfolio on AI in the safety net, and it
states plainly where I would use RAG instead.

**Highlights**

- A LoRA/QLoRA fine-tune of a small open model (Colab/Kaggle GPU) for consistent,
  cheap, plain-language rewriting at notice-sending scale.
- A deterministic faithfulness gate on the operative facts: a readable rewrite that
  drops the appeal deadline or changes the amount is disqualified.
- A four-way eval (reference, base-prompted, RAG, fine-tuned) across readability,
  faithfulness, cost, and latency; the gold reference targets hit a fifth-grade
  reading level at 100% faithfulness.
- An explicit fine-tune-vs-prompt-vs-RAG decision, including where I would not
  fine-tune (policy Q&A belongs in RAG, with citations).
- Synthetic notices only; no real notices or PII.

---

*Voice note: drafted in the plain, finding-first house style from your other
writing (no em dashes, evidence over decoration). Worth a pass against the live
aselzer.com voice before publishing.*
