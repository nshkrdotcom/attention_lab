CLAIM:
An explicit scope/operator stream plus receiver-side gate may make modifier and content interactions more diagnosable than standard Q/K/V alone.

KILL_CONDITION:
Gauntlet screens show NaN/Inf, non-descending loss, missing checkpoint, or `scope_gated_qkv_activity` failure.

MECHANISM_PROOF:
`attention_diagnostics.jsonl` must show positive `scope_output_norm`, `content_output_norm`, `scope_content_interaction_norm`, and a finite unsaturated `gate_mean`.

NEAREST_BORING_EXPLANATION:
Any early improvement could be extra projection capacity or a generic gated residual effect rather than scope/operator separation.

CONTROL_THAT_RULES_IT_OUT:
Compare against `standard_refactor_control_30m_seed1` at matched gauntlet rung and fixed data/training contract.
