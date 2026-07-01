CLAIM:
The CP-bilinear rank-8 score branch tests whether extra low-rank score capacity changes validation loss or sample efficiency versus standard attention.

KILL_CONDITION:
The run fails training, emits NaN/Inf, has inactive CP gradients, fails checkpoint reload, fails manifest verification, or does not pass final verify_run.

MECHANISM_PROOF:
`attention_diagnostics.jsonl` must show nonzero CP activity, including `cp_gradient_norm > 1e-6`, before loss differences are interpreted.

NEAREST_BORING_EXPLANATION:
Any improvement may come from extra low-rank score parameters rather than value-conditioned trilinear structure.

CONTROL_THAT_RULES_IT_OUT:
standard_30m_seed1.
