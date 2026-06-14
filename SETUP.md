# Setup notes

## GitHub

Created and pushed with the `gh` CLI as `adamselzer`:

```
https://github.com/adamselzer/plain-language-notices
```

Push further changes with `git add -A && git commit && git push`.

## What runs where

- **Laptop, no GPU, no key:** data generation, the faithfulness gate, readability,
  the offline eval (`python eval/run_eval.py --offline`), and the tests.
- **With `ANTHROPIC_API_KEY`** (in a gitignored `.env`): the base-prompted and RAG
  baselines and their rows in the four-way eval.
- **Free Colab/Kaggle GPU:** the LoRA fine-tune (`train/finetune_lora.ipynb`).
  Download the resulting `adapters/plain-notices-lora/` into the repo to enable the
  `fine_tuned` row (`python eval/run_eval.py --with-finetuned`) and the app's
  fine-tuned mode.

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env   # .env is gitignored
```

Commit identity is set per-repo to `Adam Selzer <hello@aselzer.com>`.

## Regenerating data

```bash
python data/generate_pairs.py   # writes data/pairs.jsonl and data/holdout.jsonl
```
