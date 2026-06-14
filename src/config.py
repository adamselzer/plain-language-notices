"""Configuration: key loading, model id, target reading level, and price constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

ANSWER_MODEL = os.environ.get("NOTICES_MODEL", "claude-sonnet-4-6")

# Target reading level for a "plain" notice (Flesch-Kincaid grade at or below).
TARGET_GRADE = float(os.environ.get("TARGET_GRADE", "8.0"))

# Illustrative prices ($ per token) for the cost-per-notice estimate. These are
# order-of-magnitude figures for a frontier model; the point of the metric is the
# ratio to a small self-hosted fine-tune (whose marginal API cost is ~$0), not the
# exact dollar value. Override via env if needed.
PRICE_INPUT = float(os.environ.get("PRICE_INPUT_PER_TOKEN", "0.000003"))
PRICE_OUTPUT = float(os.environ.get("PRICE_OUTPUT_PER_TOKEN", "0.000015"))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(input_tokens * PRICE_INPUT + output_tokens * PRICE_OUTPUT, 6)


def anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip() or None


def has_llm() -> bool:
    return anthropic_key() is not None
