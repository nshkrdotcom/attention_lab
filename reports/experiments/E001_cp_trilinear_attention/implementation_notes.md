# E001 Implementation Notes

Status: implementation complete; full 3000-step runs prepared but not executed in
this pass.

## Architecture Definitions

`cp_bilinear` adds a low-rank bilinear score branch to standard causal attention:

```text
extra_score_ij = sum_r q_low_i,r * k_low_j,r / sqrt(R)
scores = standard_scores + lambda * extra_scores
```

`cp_trilinear` adds a value-conditioned low-rank CP score branch:

```text
extra_score_ij = sum_r q_low_i,r * k_low_j,r * v_low_j,r / sqrt(R)
scores = standard_scores + lambda * extra_scores
```

The canonical E001 attention types are `cp_bilinear` and `cp_trilinear`. The older
`trilinear_cp` placeholder remains intentionally unimplemented.

## Lambda Controls

Trainable candidate configs use:

```yaml
cp_lambda_init: 0.0
cp_lambda_trainable: true
cp_lambda_fixed: false
```

The lambda-zero wiring control uses:

```yaml
cp_lambda_init: 0.0
cp_lambda_trainable: false
cp_lambda_fixed: true
```

## Parameter Counts

All counts are from `uv run scripts/inspect_model_config.py`.

| Config | Attention | Non-positional params | Including positional | Delta vs standard | Percent delta |
| --- | --- | ---: | ---: | ---: | ---: |
| `standard_30m_seed1.yaml` | `standard` | 29938560 | 30331776 | 0 | 0.0000% |
| `cp_bilinear_r8_30m_seed1.yaml` | `cp_bilinear` | 30159750 | 30552966 | 221190 | 0.7388% |
| `cp_trilinear_r8_30m_seed1.yaml` | `cp_trilinear` | 30270342 | 30663558 | 331782 | 1.1082% |
| `cp_trilinear_r8_lambda0_30m_seed1.yaml` | `cp_trilinear` | 30270336 | 30663552 | 331776 | 1.1082% |

The six-parameter difference between trainable and fixed trilinear lambda configs is
one scalar lambda per layer.

## Diagnostics

CP variants emit JSONL diagnostics when `diagnostics.attention_diagnostics_every` is
set. E001 full configs use:

```yaml
diagnostics:
  attention_diagnostics_every: 250
```

Diagnostic path:

```text
runs/experiments/E001_cp_trilinear_attention/<run_name>/evals/attention_diagnostics.jsonl
```

Diagnostic rows include lambda value, CP and standard score statistics, attention
entropy, CP parameter norm, and CP gradient norm.

## Manual Full-Run Scripts

Prepared but not executed in this pass:

```text
scripts/experiments/E001_cp_trilinear_attention/run_full_standard_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_bilinear_r8_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_trilinear_r8_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_trilinear_r8_lambda0_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_all_full.sh
scripts/experiments/E001_cp_trilinear_attention/compare_full_runs.sh
```

## Implementation Checks

Final QC results are recorded in `results.md`. Full-run scientific results are absent
until the manual scripts are executed and verified.
