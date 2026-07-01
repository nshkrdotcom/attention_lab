CLAIM:
This run establishes the standard-attention comparison point under the E001 fixed training contract.

KILL_CONDITION:
The run fails training, checkpoint reload, manifest verification, eval_loss, bounded HellaSwag, summarize_run, or final verify_run.

MECHANISM_PROOF:
Standard attention has no non-standard mechanism activity requirement; validity is established by manifest checks, full training completion, eval_loss reload, summarize_run, and verify_run.

NEAREST_BORING_EXPLANATION:
Any later candidate difference may be due to data order, optimizer settings, token budget, or evaluation drift rather than architecture.

CONTROL_THAT_RULES_IT_OUT:
This standard run is the control; candidates must use the same manifest, seed, optimizer, token budget, and verifier pipeline.
