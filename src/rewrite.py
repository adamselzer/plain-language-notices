"""Inference wrapper for the fine-tuned model.

The LoRA adapter is trained on a free Colab/Kaggle GPU (train/finetune_lora.ipynb)
and its small artifact saved under adapters/. This wrapper loads the base model
plus the adapter and rewrites a notice. It runs wherever a GPU (or patient CPU) and
the training-time libraries are available; it is intentionally lazy-imported so the
rest of the repo installs and runs without torch/transformers/peft.

The cost argument the eval makes: at inference this calls no external API, so the
marginal cost per notice is effectively zero compared with a frontier-model call.
"""

from __future__ import annotations

import time
from pathlib import Path

DEFAULT_ADAPTER = Path(__file__).resolve().parent.parent / "adapters" / "plain-notices-lora"

SYSTEM = (
    "You rewrite government benefit notices into plain language at a sixth-grade "
    "reading level, preserving every operative fact (amounts, dates, the action, the "
    "reason, and appeal/hearing rights with the deadline)."
)


class FineTunedRewriter:
    """Lazily loads the base model + LoRA adapter and rewrites notices."""

    def __init__(self, adapter_path: str | Path = DEFAULT_ADAPTER, base_model: str | None = None):
        self.adapter_path = Path(adapter_path)
        self.base_model = base_model
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        if not self.adapter_path.exists():
            raise FileNotFoundError(
                f"No adapter at {self.adapter_path}. Train it with train/finetune_lora.ipynb "
                "(Colab/Kaggle GPU) and place the adapter here."
            )
        # Lazy heavy imports so the package installs without these on a laptop.
        import torch  # noqa: F401
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, pipeline

        model = AutoPeftModelForCausalLM.from_pretrained(self.adapter_path, device_map="auto")
        tok = AutoTokenizer.from_pretrained(self.adapter_path)
        self._pipe = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=400)

    def rewrite(self, notice: str) -> tuple[str, float]:
        self._load()
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Rewrite this benefit notice in plain language:\n\n{notice}"}]
        t0 = time.time()
        out = self._pipe(messages)[0]["generated_text"]
        text = out[-1]["content"] if isinstance(out, list) else str(out)
        return text.strip(), time.time() - t0
