# Hypothesis: multi_qkv_static_3track_global_30m_seed1

CLAIM:
The static globally shared 3-track Q/K/V bank tests whether the first-build banked architecture trains stably under a fixed depth-only cycle.

KILL_CONDITION:
The run fails training, emits NaN/Inf, fails checkpoint reload, fails manifest verification, fails final verify_run, lacks valid attention diagnostics, lacks global-bank evidence, or has degenerate track gradients.

MECHANISM_PROOF:
`attention_diagnostics.jsonl` must show `uses_global_bank=true`, `track_count=3`, `route_formula=layer_idx % track_count`, scalar active tracks, nonzero per-track gradients over the run, and all three tracks used across layers.

NEAREST_BORING_EXPLANATION:
Any behavior may be due to global parameter sharing or static cyclic tying rather than train-time or position-time rotation.

CONTROL_THAT_RULES_IT_OUT:
standard_refactor_control_30m_seed1.
