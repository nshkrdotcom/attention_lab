# Hypothesis: standard_refactor_control_30m_seed1

CLAIM:
This run checks whether shared-path changes for step, schedule mode, position IDs, diagnostics, and Multi-QKV compatibility preserve standard-attention behavior under the E002 fixed training contract.

KILL_CONDITION:
The run fails training, checkpoint reload, manifest verification, eval_loss, bounded HellaSwag, summarize_run, final verify_run, or diverges materially from the known standard baseline without an explained intentional change.

MECHANISM_PROOF:
Standard attention has no Multi-QKV mechanism. Validity is established by unchanged standard model tests, checkpoint reload, eval_loss, summarize_run, and verify_run.

NEAREST_BORING_EXPLANATION:
Any downstream Multi-QKV candidate difference may be caused by shared training/eval plumbing changes rather than architecture.

CONTROL_THAT_RULES_IT_OUT:
This standard refactor control is the plumbing control for E002. It must pass before interpreting A/B/C.
