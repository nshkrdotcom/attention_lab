# E002 Initial Results Template

Status: template only. Do not fill metrics by hand.

## Runs

| Run | Completed | Verified | Diagnostics | Destructive Test | Evidence Level |
| --- | --- | --- | --- | --- | --- |
| `standard_refactor_control_30m_seed1` | no | no | n/a | n/a | not_run |
| `multi_qkv_static_3track_global_30m_seed1` | no | no | no | no | not_run |
| `multi_qkv_train_rotation_3track_global_30m_seed1` | no | no | no | no | not_run |
| `multi_qkv_position_rotation_3track_global_30m_seed1` | no | no | no | no | not_run |

## Metrics

Populate only from generated artifacts.

Required sources:

```text
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/metrics.jsonl
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/run_summary.json
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/val_loss.json
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/hellaswag.json
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/attention_diagnostics.jsonl
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/qkv_track_destructive_test.json
```

## Interpretation

Do not interpret until all required artifacts exist.
