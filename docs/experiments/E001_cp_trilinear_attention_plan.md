# E001 CP Trilinear Attention Plan

Status: implementation prepared. CP-bilinear and CP-trilinear attention modules are
implemented behind the registry, but E001 exploration is screen-first. Full 3000-step
runs are promotion-stage artifacts only and are not claimed until their run scripts
complete and verify.

## Hypothesis

A CP-style attention score augmentation may improve validation loss or training
efficiency relative to the standard-attention 30M baseline under this repository's
local GPT/FineWeb-Edu setup.

## Non-Claim Boundaries

This experiment may only support a local finding for small GPT pretraining on the
manifested FineWeb-Edu 100M token shard. It must not claim broad superiority of a new
attention mechanism, general downstream gains, or scaling behavior beyond the measured
configs.

## Architecture Variants

- `standard_30m_seed1`: runnable standard-attention comparison run.
- `standard_refactor_control_30m_seed1`: runnable standard-attention control for any
  future standard-path refactor.
- `cp_bilinear_r8_30m_seed1`: low-rank bilinear additive score-branch control.
- `cp_trilinear_r8_30m_seed1`: value-conditioned low-rank trilinear additive
  score-branch candidate.
- `cp_trilinear_r8_lambda0_30m_seed1`: wiring/identity-style control with fixed
  zero branch contribution.

## Controls

The standard baseline and standard refactor control must run with the same data,
tokenizer, batch construction, optimizer, LR schedule, seed, and eval cadence as the
candidate configs. If the implementation changes shared model code, the standard
refactor control must be rerun before comparing candidates.

## Fixed Baseline Contract

All direct comparisons must hold fixed:

- Dataset manifest: `data/fineweb_edu_100m/manifest.json`
- Data root: `data/fineweb_edu_100m`
- Tokenizer: `gpt2`
- Vocab size: `50304`
- Model shape: block size `1024`, layers `6`, heads `6`, embedding size `384`
- Batch: `B=4`, `T=1024`, total batch size `262144`
- Steps: `3000`
- Optimizer and LR schedule from `baseline_30m_fineweb100m.yaml`
- Validation, checkpoint, sample, and HellaSwag commands

## Screen-First Run Matrix

| Config | Status | Run Directory |
| --- | --- | --- |
| `standard_30m_seed1.yaml` | screen first; full only if approved | `runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1` |
| `standard_refactor_control_30m_seed1.yaml` | screen first when shared code changes | `runs/experiments/E001_cp_trilinear_attention/standard_refactor_control_30m_seed1` |
| `cp_bilinear_r8_30m_seed1.yaml` | screen first; requires active CP diagnostics | `runs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1` |
| `cp_trilinear_r8_30m_seed1.yaml` | screen first; requires active CP diagnostics | `runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1` |
| `cp_trilinear_r8_lambda0_30m_seed1.yaml` | manual promoted control only | `runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_lambda0_30m_seed1` |

Default operator path:

```bash
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run attn-queue add configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
uv run attn-queue add configs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1.yaml
uv run attn-queue add configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1.yaml
uv run attn-queue start
uv run attn-queue promotion-report cp_trilinear_r8_30m_seed1
```

Approve at most selected full runs after promotion report review:

```bash
uv run attn-queue approve cp_trilinear_r8_30m_seed1
```

## Success Criteria

- Candidate run completes to step 3000.
- `verify_run.py` passes with `--expect-data-manifest`.
- Checkpoint reload `eval_loss.py` matches the run's final validation loss within
  normal evaluation tolerance.
- Candidate comparison artifact reports final/best validation loss, perplexity,
  throughput, and VRAM against the matched standard baseline.
- Any claimed improvement is larger than the noise observed in standard/control runs.

## Kill Criteria

- Candidate repeatedly produces NaN/Inf loss.
- Candidate cannot complete a 20-step sanity run.
- Candidate violates the fixed data manifest or comparison contract.
- Candidate has severe throughput/VRAM regression without a compensating validation
  signal.
- Diagnostic outputs show degenerate or unused CP parameters.

## Required Diagnostics

CP candidate runs must emit attention diagnostics to:

```text
runs/experiments/E001_cp_trilinear_attention/<run_name>/evals/attention_diagnostics.jsonl
```

The schema is documented at:

```text
reports/schema/attention_diagnostics.schema.json
```

## Required Artifacts

- Run directory with config, manifest, metrics, checkpoints, samples, evals, and summary.
- `evals/val_loss.json`
- `evals/hellaswag.json`
- `evals/run_summary.json`
- `evals/attention_diagnostics.jsonl` for CP candidates.
- `reports/experiments/E001_cp_trilinear_attention/comparison.json`
- Completed experiment result report.

The comparison script requires full-run summaries, eval artifacts, checkpoints, final `verify_run.py` success, and CP diagnostics with `cp_gradient_norm` above the shared queue activity threshold for CP candidates.
Lambda0 is skipped unless its manual promoted full-run artifacts exist.

## Manual Promotion-Stage Full-Run Commands

```bash
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
scripts/experiments/E001_cp_trilinear_attention/compare_full_runs.sh
```

Individual full-run scripts verify data manifests, train, verify, run evals, summarize,
and verify again. They are intentionally not executed by implementation-agent passes.
They require `ATTENTION_LAB_I_UNDERSTAND_THIS_IS_A_PROMOTED_FULL_RUN=1` and still do not replace queue promotion approval.
`run_all_full.sh` refuses to launch the matrix; it is not the exploration path.
Lambda0 is a manual promoted control only after active CP evidence justifies running it.
