# E001 CP Trilinear Attention Results

Status: template. Full 3000-step runs are manual and must not be reported unless the
run scripts complete and final verification passes.

## Claim Boundary

State the exact local claim supported by the run matrix. Do not claim general
attention superiority.

## Run Matrix

| Variant | Config | Run Directory | Status |
| --- | --- | --- | --- |
| Standard | `standard_30m_seed1.yaml` | `runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1` | not_run |
| Standard refactor control | `standard_refactor_control_30m_seed1.yaml` | `runs/experiments/E001_cp_trilinear_attention/standard_refactor_control_30m_seed1` | not_run |
| CP bilinear r8 | `cp_bilinear_r8_30m_seed1.yaml` | `runs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1` | not_run |
| CP trilinear r8 | `cp_trilinear_r8_30m_seed1.yaml` | `runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1` | not_run |
| CP trilinear r8 lambda0 | `cp_trilinear_r8_lambda0_30m_seed1.yaml` | `runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_lambda0_30m_seed1` | not_run |

## Comparison Summary

Link to `comparison.json` after runs complete.

## Diagnostics

Summarize `evals/attention_diagnostics.jsonl` for CP candidate runs.

## Negative Result Interpretation

Record whether a negative result appears to be optimization, capacity, implementation,
or mechanism-related.
