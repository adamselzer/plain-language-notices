# Eval report — plain-language-notices (four-way comparison)

Holdout notices: 15. Target reading level: grade <= 8.
Systems evaluated: reference, base_prompted, rag.

| System | Target-grade hit | Mean FK grade | Faithfulness pass | Mean latency | Mean cost |
|---|---|---|---|---|---|
| reference | 93% | 4.26 | 100% | 0.0s | $0.00000 |
| base_prompted | 20% | 8.48 | 100% | 8.255s | $0.00176 |
| rag | 100% | 4.16 | 100% | 11.951s | $0.00189 |

Faithfulness is the gate: a rewrite that drops or alters an operative fact (amount, date, action, appeal rights) fails regardless of how well it reads. The `reference` row is the gold plain targets and sets the ceiling. With a key, `base_prompted` and `rag` fill in; with a trained adapter, `fine_tuned` shows the cost/latency case for a small self-hosted model at notice-sending scale.
