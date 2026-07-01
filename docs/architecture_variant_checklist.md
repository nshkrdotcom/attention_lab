# Architecture Variant Checklist

Use this checklist for every new attention architecture. Baseline files must not be
edited unless the experiment explicitly includes a standard-refactor control.

For queued or overnight runs, also follow:

```text
docs/guides/experiment_queue_discipline_checklist.md
```

For E002 Multi-QKV Shift Register work, also follow the stricter implementation docset:

```text
docs/implementation/0901_multiqkv_shift_register/
```

1. Add the architecture module under `src/attention_lab/models/attention/`.
2. Add a registry entry in `src/attention_lab/models/attention/registry.py`.
3. Add config validation support for the new `attention_type`.
4. Add unit tests for shape, causal masking, and gradient flow.
5. Add a lambda=0 or equivalent identity/wiring control when applicable.
6. Add parameter-count inspection coverage.
7. Add mechanism diagnostics.
8. Add experiment configs under `configs/experiments/<EXPERIMENT_ID>/`.
9. Add or update the experiment plan.
10. Train the candidate under the same manifest and token budget.
11. Run `eval_loss.py`, `eval_generate.py`, and bounded `eval_hellaswag.py`.
12. Run `summarize_run.py`.
13. Run `verify_run.py` with `--expect-data-manifest`.
14. Run `compare_runs.py`.
15. Update the experiment report.
16. State the claim boundary and negative-result interpretation.
17. Write `docs/experiments/<EXPERIMENT_ID>/hypothesis_<run_name>.md` before any
    full queue run.
18. Confirm mechanism diagnostics prove the non-standard path is active before
    interpreting loss.
19. Confirm the nearest boring explanation has a queued or completed control.
20. Set `queue.requires_run` for non-standard FULL candidates, or explicitly document
    `queue.skip_control_check: true`.
21. Leave `queue.full_run_approved: false` until the operator intentionally approves
    the FULL run with `attn-queue approve`.
22. Keep `queue.allow_overwrite_existing_run_dir: false` unless overwriting a known
    disposable diagnostic run.
23. Export the queue report with `attn-queue export-report --experiment <ID>`.
24. Add a decision note with `attn-queue morning-note` after overnight review.

The standard-attention baseline path must remain intact throughout the experiment.
