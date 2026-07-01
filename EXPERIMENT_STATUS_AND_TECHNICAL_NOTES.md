# Experiment Status And Technical Notes

This document is the current dynamic status page for Attention Lab experiments. It is intentionally both an experiment-status ledger and a technical interpretation note. Keep README.md and AGENTS.md short by pointing to this file instead of duplicating fast-changing run state.

Last updated from pasted local run artifacts: 2026-07-01 UTC.

## Current Bottom Line

E001 now has partial verified-looking full-run artifacts for three 3000-step runs:

| Run | Status from provided artifacts | Final val loss | Final ppl | Median tokens/sec | Peak VRAM allocated | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `standard_30m_seed1` | completed to step 3000 | 4.076830863952637 | 58.958326507329176 | 109427.64995743861 | 3240.92431640625 MB | Main standard-attention E001 baseline. |
| `cp_bilinear_r8_30m_seed1` | completed to step 3000 | 4.086292743682861 | 59.51883063145288 | 33625.54559587996 | 4443.17626953125 MB | Active CP branch; worse loss than standard and much slower. |
| `cp_trilinear_r8_30m_seed1` | completed to step 3000 | 4.062333106994629 | 58.10972926077219 | 16761.225878491987 | 6772.76513671875 MB | Best loss among the three, but very slow and high VRAM. |
| `cp_trilinear_r8_lambda0_30m_seed1` | incomplete / failed / interrupted | unknown | unknown | unknown | unknown | Directory exists but has empty checkpoints/evals/samples in the shown tree. |
| `standard_refactor_control_30m_seed1` | not observed as completed | unknown | unknown | unknown | unknown | Config exists; no completed run artifact was provided. |

The old E001 report embedded in the pasted file still says the 3000-step full runs were not executed. That is now stale relative to the run summaries included later in the same pasted artifact. Treat this file as the reconciliation layer: older generated reports may lag local run state until regenerated.

## Evidence Boundary

The current evidence is enough to say that E001 has real 3000-step artifacts for `standard_30m_seed1`, `cp_bilinear_r8_30m_seed1`, and `cp_trilinear_r8_30m_seed1`.

Do not yet claim a final scientific result until the repository reports have been regenerated from the actual run directories and the following commands pass from the current codebase root:

```bash
uv run scripts/verify_run.py \
  --run_dir runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-eval-loss \
  --expect-hellaswag \
  --expect-data-manifest

uv run scripts/verify_run.py \
  --run_dir runs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-eval-loss \
  --expect-hellaswag \
  --expect-data-manifest

uv run scripts/verify_run.py \
  --run_dir runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-eval-loss \
  --expect-hellaswag \
  --expect-data-manifest

scripts/experiments/E001_cp_trilinear_attention/compare_full_runs.sh
uv run attn-queue export-report --experiment E001_cp_trilinear_attention
```

After that, update:

```text
reports/experiments/E001_cp_trilinear_attention/results.md
reports/experiments/E001_cp_trilinear_attention/run_index.json
reports/experiments/E001_cp_trilinear_attention/run_index.md
```

## E001 Technical Interpretation

### Loss

At face value, `cp_trilinear_r8_30m_seed1` has the best final validation loss:

```text
standard final_val_loss      = 4.076830863952637
cp_bilinear final_val_loss   = 4.086292743682861
cp_trilinear final_val_loss  = 4.062333106994629
```

Relative to standard, CP-trilinear improves final validation loss by about `0.01450`, while CP-bilinear is worse than standard by about `0.00946`. This is directionally interesting because the value-conditioned trilinear branch beats both the plain standard baseline and the CP-bilinear low-rank score-capacity control in this one seed.

This is not yet enough for a broad claim. It is a local, single-seed result on the FineWeb-Edu 100M setup. The correct claim shape is:

```text
In one local ~30M GPT / FineWeb-Edu 100M E001 run, CP-trilinear reached a lower final validation loss than both standard attention and CP-bilinear, but at severe throughput and VRAM cost. This warrants replication and control completion, not an architecture superiority claim.
```

### Throughput and VRAM

The performance cost is large:

```text
standard median_tokens_per_sec      = 109427.65
cp_bilinear median_tokens_per_sec   = 33625.55
cp_trilinear median_tokens_per_sec  = 16761.23

standard peak_vram_allocated_mb     = 3240.92
cp_bilinear peak_vram_allocated_mb  = 4443.18
cp_trilinear peak_vram_allocated_mb = 6772.77
```

Approximate ratios:

| Run | Speed vs standard | VRAM vs standard |
| --- | ---: | ---: |
| CP-bilinear | 0.307x | 1.371x |
| CP-trilinear | 0.153x | 2.090x |

So even if the CP-trilinear loss signal survives replication, the current implementation is not an efficiency result. It is an architecture/mechanism probe result at best.

### Mechanism Activity

The CP branches are not dead. The pasted diagnostics show nonzero `cp_gradient_norm` and nonzero CP-score statistics for both CP-bilinear and CP-trilinear across training. At step 3000, CP-bilinear has nonzero `cp_gradient_norm` in all six layers. CP-trilinear also has nonzero `cp_gradient_norm` in all six layers at step 3000.

That supports the narrow statement that the branches were active and trainable. It does not by itself prove that the branch caused the loss difference.

### CP-Bilinear Control

The CP-bilinear run is important because it controls for extra low-rank score capacity without value-conditioned trilinear structure. Since CP-bilinear is worse than standard while CP-trilinear is better than standard in the current summaries, the first-order result is not simply “adding a CP score branch helps.” The interesting hypothesis is narrower:

```text
The value-conditioned trilinear score branch may be doing something different from the bilinear low-rank score-capacity control.
```

That hypothesis still needs replication and the lambda-zero control.

### Lambda-Zero Control

`cp_trilinear_r8_lambda0_30m_seed1` is not interpretable yet. The tree shown by the user indicates the directory exists but checkpoints/evals/samples are empty. Treat it as incomplete, failed, or interrupted until metrics and verifier output say otherwise.

This run matters because it distinguishes “the trilinear code path exists” from “the trilinear branch contributes nonzero score augmentation.” Without it, some implementation/wiring explanations remain open.

### Standard Refactor Control

`standard_refactor_control_30m_seed1` is also not interpretable from the provided artifacts. If shared code changed while E001 was implemented, the standard refactor control should be run or explicitly waived with rationale before making strong claims.

## E002 Status

E002 currently has at least one partial directory:

```text
runs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1
```

The shown tree indicates it has metadata files, metrics files, empty checkpoints, empty evals, and empty samples. Treat it as incomplete / failed / interrupted until run summaries and verifier output exist.

E002 should not be interpreted until the canonical first-build runs complete:

```text
standard_refactor_control_30m_seed1
multi_qkv_static_3track_global_30m_seed1
multi_qkv_train_rotation_3track_global_30m_seed1
multi_qkv_position_rotation_3track_global_30m_seed1
```

For E002, validation loss is not interpretable without `attention_diagnostics.jsonl` and the destructive route test output:

```text
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/attention_diagnostics.jsonl
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/qkv_track_destructive_test.json
```

## What To Do Next

1. Regenerate E001 comparison artifacts from actual run directories.
2. Verify all completed E001 runs with `verify_run.py` and manifest checks.
3. Decide whether to rerun or mark incomplete:
   - `cp_trilinear_r8_lambda0_30m_seed1`
   - `standard_refactor_control_30m_seed1`
4. Update E001 results and run index after verification.
5. Only then decide whether E001 merits replication seeds or a tighter diagnostic follow-up.
6. Do not proceed to E002 scientific interpretation until E002 has complete run summaries, diagnostics, destructive tests, and verifier passes.

## Suggested Report Regeneration Command

From the repo root:

```bash
scripts/experiments/E001_cp_trilinear_attention/compare_full_runs.sh
uv run attn-queue export-report --experiment E001_cp_trilinear_attention
cat \
  reports/experiments/E001_cp_trilinear_attention/results.md \
  reports/experiments/E001_cp_trilinear_attention/run_index.md \
  reports/experiments/E001_cp_trilinear_attention/comparison*.json
```

## Status Labels Used Here

- `completed to step 3000`: a run summary exists showing `max_step: 3000`.
- `verified`: only use this after `verify_run.py` passes with the relevant flags.
- `incomplete / failed / interrupted`: a run directory exists but required checkpoints, evals, summaries, or samples are missing.
- `not observed as completed`: no completed run summary was provided.
- `stale report`: a checked-in or generated report contradicts newer local artifacts.

