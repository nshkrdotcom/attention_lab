# Hypothesis: multi_qkv_train_rotation_3track_global_30m_seed1

CLAIM:
The train-time rotation run tests whether forcing each globally shared Q/K/V track to occupy different depth phases during training improves validation behavior or mechanism robustness beyond the static global cycle, even after deployment freezes back to the static layout.

KILL_CONDITION:
The run fails training, emits NaN/Inf, fails checkpoint reload, fails manifest verification, fails final verify_run, lacks valid attention diagnostics, fails to show changing active tracks across training steps, uses step rotation during eval/generation, or fails destructive route tests.

MECHANISM_PROOF:
`attention_diagnostics.jsonl` must show train rows using `(layer_idx + step) % 3`, eval/generate rows using `layer_idx % 3`, `eval_freeze_mode=true`, non-null training steps, nonzero track gradients, and all three tracks active over training.

NEAREST_BORING_EXPLANATION:
Any improvement may be due to global 3-track sharing, static cyclic tying, parameter-count differences, optimizer noise, or run variance rather than train-time phase exposure.

CONTROL_THAT_RULES_IT_OUT:
multi_qkv_static_3track_global_30m_seed1.
