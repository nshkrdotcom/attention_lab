# Hypothesis: multi_qkv_position_rotation_3track_global_30m_seed1

CLAIM:
The position-rotation run tests whether an inference-persistent token-position clock can route a globally shared Q/K/V bank by `(layer_idx + position) % 3` during both training and eval/generation.

KILL_CONDITION:
The run fails training, emits NaN/Inf, fails checkpoint reload, fails manifest verification, fails final verify_run, lacks valid attention diagnostics, implements scalar routing instead of per-position routing, breaks causal masking, or fails destructive route tests.

MECHANISM_PROOF:
`attention_diagnostics.jsonl` must show `position_routing_enabled=true`, `active_track_index=null`, multi-track nonzero `active_track_counts` within a sequence, nonzero gradients for all tracks, and route formula `(layer_idx + position) % track_count`.

NEAREST_BORING_EXPLANATION:
Any improvement may be due to extra computation, effective ensemble-like projection selection, parameter-count differences, or static global bank effects rather than inference-persistent clocking.

CONTROL_THAT_RULES_IT_OUT:
multi_qkv_static_3track_global_30m_seed1.
