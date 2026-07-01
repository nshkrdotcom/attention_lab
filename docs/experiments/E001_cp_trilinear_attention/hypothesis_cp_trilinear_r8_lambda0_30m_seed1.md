CLAIM:
The CP-trilinear lambda-zero run checks that the CP branch wiring can be present while contributing zero score augmentation under the E001 fixed training contract.

KILL_CONDITION:
The run diverges materially from standard behavior without an explained wiring reason, fails training, fails checkpoint reload, fails manifest verification, or does not pass final verify_run.

MECHANISM_PROOF:
The CP branch is present but fixed at zero contribution; diagnostics should show the branch does not affect scores while the standard path remains valid.

NEAREST_BORING_EXPLANATION:
Any candidate improvement may be due to implementation/wiring artifacts rather than active CP-trilinear score augmentation.

CONTROL_THAT_RULES_IT_OUT:
standard_30m_seed1 and cp_trilinear_r8_30m_seed1.
