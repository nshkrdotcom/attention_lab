CLAIM:
The CP-trilinear rank-8 value-conditioned score branch tests whether value-conditioned score augmentation improves next-token validation loss or sample efficiency beyond standard attention and the bilinear low-rank control.

KILL_CONDITION:
The run fails training, emits NaN/Inf, has inactive CP gradients, fails checkpoint reload, fails manifest verification, is no better than the CP-bilinear control, or does not pass final verify_run.

MECHANISM_PROOF:
`attention_diagnostics.jsonl` must show nonzero CP activity, including `cp_gradient_norm > 1e-6`, and lambda/score diagnostics must indicate the branch is used.

NEAREST_BORING_EXPLANATION:
Any improvement may be explained by extra low-rank score capacity, random seed noise, throughput differences, or verifier/eval drift rather than value-conditioned trilinear structure.

CONTROL_THAT_RULES_IT_OUT:
standard_30m_seed1 and cp_bilinear_r8_30m_seed1.
