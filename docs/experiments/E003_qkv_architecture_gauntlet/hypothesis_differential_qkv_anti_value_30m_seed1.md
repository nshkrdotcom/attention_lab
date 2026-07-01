CLAIM:
Explicit positive and negative/suppressive QKV branches may expose separable subtractive write behavior while preserving enough early learning to remain study-worthy.

KILL_CONDITION:
Gauntlet screens show NaN/Inf, non-descending loss, missing checkpoint, or `differential_qkv_activity` failure.

MECHANISM_PROOF:
`attention_diagnostics.jsonl` must show positive `pos_output_norm`, `neg_output_norm`, `branch_output_delta`, and finite positive `diff_lambda`.

NEAREST_BORING_EXPLANATION:
Any early improvement could be parameter count or optimization noise rather than useful anti-value structure.

CONTROL_THAT_RULES_IT_OUT:
Compare against `standard_refactor_control_30m_seed1` at matched gauntlet rung and fixed data/training contract.
