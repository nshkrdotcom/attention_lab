CLAIM:
This run checks whether the experiment-control standard path reproduces the standard-attention behavior under the E001 fixed training contract.

KILL_CONDITION:
The run fails training, checkpoint reload, manifest verification, eval_loss, bounded HellaSwag, summarize_run, or final verify_run.

MECHANISM_PROOF:
Standard attention has no non-standard mechanism activity requirement; validity is established by manifest checks, full training completion, eval_loss reload, summarize_run, and verify_run.

NEAREST_BORING_EXPLANATION:
Any difference from `standard_30m_seed1` may indicate path, config, or orchestration drift rather than a meaningful model behavior difference.

CONTROL_THAT_RULES_IT_OUT:
standard_30m_seed1.
