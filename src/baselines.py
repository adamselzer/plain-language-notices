"""Baselines to compare the fine-tune against: base-prompted and RAG.

The portfolio point is judgment: showing where a fine-tune beats prompting and
where it does not. These two baselines are what the fine-tune is measured against.

  - base_prompted: a capable frontier model (Claude) asked, zero-shot, to rewrite a
    notice in plain language while preserving the operative facts.
  - rag: the same, but with a plain-language style guide and a worked example
    retrieved into the context (a stand-in for retrieval over a style corpus).

Both call Claude and need ANTHROPIC_API_KEY. The fine-tuned model (src/rewrite.py)
needs no API at inference, which is the cost argument the eval quantifies.
"""

from __future__ import annotations

import time

from .config import ANSWER_MODEL, anthropic_key

PLAIN_GUIDE = """\
Plain-language rules for benefit notices:
- Write at about a sixth-grade reading level. Short sentences. Common words.
- Lead with what is happening to the reader and what they must do.
- Keep every operative fact exactly: dollar amounts, dates, the action, the reason,
  and appeal/hearing rights with the deadline.
- No legal jargon ("pursuant to", "predicated upon", "said determination")."""

EXAMPLE = (
    "Bureaucratic: 'This correspondence constitutes formal notification that your "
    "application has been denied, predicated upon excess income. You may request a "
    "hearing within 90 days.'\n"
    "Plain: 'We can't approve your food assistance because your income is above the "
    "limit. If you disagree, you can ask for a hearing within 90 days.'"
)

SYSTEM = (
    "You rewrite government benefit notices into plain language. Preserve every "
    "operative fact exactly (amounts, dates, the action, the reason, and appeal/"
    "hearing rights with the deadline). Output only the rewritten notice."
)


def _client():
    import anthropic

    key = anthropic_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; add it to .env for the baselines.")
    return anthropic.Anthropic(api_key=key)


def _call(client, prompt: str) -> tuple[str, float, int, int]:
    t0 = time.time()
    msg = client.messages.create(
        model=ANSWER_MODEL, max_tokens=400, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    dt = time.time() - t0
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    return text, dt, msg.usage.input_tokens, msg.usage.output_tokens


def base_prompted_rewrite(notice: str, client=None) -> tuple[str, float, int, int]:
    client = client or _client()
    return _call(client, f"Rewrite this benefit notice in plain language:\n\n{notice}")


def rag_rewrite(notice: str, client=None) -> tuple[str, float, int, int]:
    client = client or _client()
    prompt = (
        f"{PLAIN_GUIDE}\n\nExample:\n{EXAMPLE}\n\n"
        f"Now rewrite this benefit notice in plain language:\n\n{notice}"
    )
    return _call(client, prompt)
