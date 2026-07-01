# 0901 Rebuild 04: Diagnostics and Mechanism Checks

## Purpose

This document defines the diagnostics, mechanism checks, destructive tests, schemas, and evidence gates for the first-build
Multi-QKV Shift Register experiment. Loss alone is not sufficient evidence. The run must prove:

```text
- the global Q/K/V bank was used
- the intended tracks were selected
- gradients reached the selected tracks
- inactive tracks remained inactive under hard scalar routing
- position routing used multiple tracks within a sequence
- B froze its train-time clock during eval/generation
- destructive route perturbations changed model behavior measurably
```

The first-build variants are:

```text
A: multi_qkv_static_3track_global
B: multi_qkv_train_rotation_3track_global
C: multi_qkv_position_rotation_3track_global
```

## Diagnostic Output Locations

Training-time attention diagnostics are written through the existing diagnostics cadence to:

```text
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/attention_diagnostics.jsonl
```

Destructive/off-route diagnostics are written to:

```text
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/qkv_track_destructive_test.json
```

Comparison artifacts must read both files. Diagnostic artifacts must not exist only in stdout.

## Diagnostic Frequency

Use `diagnostics.attention_diagnostics_every`.

```yaml
diagnostics:
  attention_diagnostics_every: 250
```

Tiny integration tests may set:

```yaml
diagnostics:
  attention_diagnostics_every: 1
```

Do not emit every-step diagnostics in full runs unless explicitly configured.

## Required JSONL Fields

Every Multi-QKV row must include:

```json
{
  "schema_version": 1,
  "experiment_id": "E002_multitrack_qkv_shift_register",
  "run_name": "multi_qkv_static_3track_global_30m_seed1",
  "attention_type": "multi_qkv_static_3track_global",
  "route_formula": "layer_idx % track_count",
  "uses_global_bank": true,
  "track_count": 3,
  "layer_idx": 0,
  "layer": 0,
  "step": 250,
  "last_forward_step": 250,
  "schedule_mode": "train",
  "active_track_index": 0,
  "active_track_counts": {"0": 1024, "1": 0, "2": 0},
  "track_gradient_norm": 0.014,
  "per_track_gradient_norm": {"0": 0.014, "1": 0.0, "2": 0.0},
  "per_track_qkv_weight_norm": {"0": 11.92, "1": 12.04, "2": 11.88},
  "position_routing_enabled": false,
  "eval_freeze_mode": false
}
```

For position routing, `active_track_index` is `null` and `active_track_counts` is mandatory:

```json
{
  "attention_type": "multi_qkv_position_rotation_3track_global",
  "route_formula": "(layer_idx + position) % track_count",
  "active_track_index": null,
  "active_track_counts": {"0": 342, "1": 341, "2": 341},
  "track_gradient_norm": null,
  "per_track_gradient_norm": {"0": 0.011, "1": 0.010, "2": 0.012},
  "position_routing_enabled": true,
  "eval_freeze_mode": false
}
```

Field meanings:

- `uses_global_bank` must be `true` for A/B/C.
- `track_count` must be `3` for canonical A/B/C.
- `layer_idx` and `layer` may both be emitted and must agree.
- `last_forward_step` is the step actually passed into attention. B train rows must not have `null`.
- `schedule_mode` must be `train`, `eval`, or `generate`.
- `active_track_counts` must sum to a positive token count.
- `track_gradient_norm` is the active scalar track gradient for A/B and may be `null` for C.
- `per_track_gradient_norm` and `per_track_qkv_weight_norm` are keyed by string track IDs.
- `position_routing_enabled` is `false` for A/B and `true` for C.
- `eval_freeze_mode` is `true` only for B.

Optional fields such as `track_entropy`, `per_track_output_norm`, `track_output_delta`, throughput, and VRAM are useful but
are not required for first acceptance unless the schema and tests are updated.

## Schema

The schema lives at:

```text
reports/schema/attention_diagnostics.schema.json
```

It must accept both existing CP diagnostics and Multi-QKV diagnostics. Schema validation checks structure only. Scientific
acceptance is handled by mechanism checks.

## Mechanism Check: qkv_track_activity

The queue mechanism check lives in:

```text
src/attention_lab/queue/mechanism_checks.py
```

It reads `attention_diagnostics.jsonl` and considers Multi-QKV rows whose `attention_type` starts with `multi_qkv_` or
legacy QKV rows whose type starts with `qkv_shift_`.

Common pass criteria:

```text
[ ] at least one diagnostic row exists
[ ] every inspected row has uses_global_bank == true
[ ] track_count == 3
[ ] per_track_qkv_weight_norm has keys "0", "1", "2"
[ ] at least one track has nonzero gradient in at least one row
[ ] active_track_counts exists and sums to a positive token count
```

A-specific criteria:

```text
[ ] route_formula == "layer_idx % track_count"
[ ] position_routing_enabled == false
[ ] eval_freeze_mode == false
[ ] active_track_index is in {0,1,2}
[ ] active_track_counts places all tokens on active_track_index
[ ] all three tracks appear across layers when n_layer >= 3
```

B-specific criteria:

```text
[ ] route_formula describes layer_idx + step during train and layer_idx during eval/generate
[ ] position_routing_enabled == false
[ ] eval_freeze_mode == true
[ ] train rows have non-null last_forward_step
[ ] train rows show active tracks changing across steps when multiple steps are present
[ ] eval/generate rows, if present, use layer_idx % track_count
```

C-specific criteria:

```text
[ ] route_formula == "(layer_idx + position) % track_count"
[ ] position_routing_enabled == true
[ ] eval_freeze_mode == false
[ ] active_track_index is null
[ ] active_track_counts has more than one nonzero track for seq_len >= 3
[ ] per_track_gradient_norm shows nonzero gradients for all tracks in at least one row
```

The check result exposes `passed`, `reason`, and `details` in addition to the existing `active` and `note` fields.

## Destructive / Off-Route Test

The destructive test script is:

```text
scripts/qkv_track_destructive_test.py
```

Required usage after a manual full run:

```bash
uv run scripts/qkv_track_destructive_test.py \
  --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml \
  --checkpoint runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  --out runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/evals/qkv_track_destructive_test.json \
  --num-batches 4
```

Supported perturbations:

```text
force_track
rotate_tracks
zero_selected
```

The script uses a temporary `MultiQKVDebugRouteOverride` context manager. Normal train/eval code must never set route
overrides.

Output JSON includes:

```json
{
  "schema_version": 1,
  "experiment_id": "E002_multitrack_qkv_shift_register",
  "run_name": "multi_qkv_static_3track_global_30m_seed1",
  "checkpoint": "runs/.../checkpoints/ckpt_last.pt",
  "config": "configs/.../multi_qkv_static_3track_global_30m_seed1.yaml",
  "num_batches": 4,
  "perturbations": [
    {
      "name": "rotate_tracks",
      "natural_loss": 5.8123,
      "perturbed_loss": 5.9344,
      "loss_delta": 0.1221,
      "mean_abs_logit_delta": 0.031,
      "max_abs_logit_delta": 0.44
    }
  ],
  "destructive_test_passed": true
}
```

The destructive gate passes only when losses are finite and at least one perturbation changes loss or logits above the
predeclared tiny thresholds.

## Evidence Levels

Use these report labels:

```text
not_implemented
implemented_not_run
screened_mechanism_active
full_run_verified
candidate_evidence
insufficient_evidence
killed
```

The current implementation pass may only claim `implemented_not_run`. Full-run and candidate-evidence labels require manual
3000-step runs and verified artifacts.

## Full-Run Interpretation Gate

Do not interpret A/B/C loss until these files exist for each run:

```text
evals/run_summary.json
evals/val_loss.json
evals/hellaswag.json
evals/attention_diagnostics.jsonl
evals/qkv_track_destructive_test.json
```

If diagnostics or destructive output is missing or degenerate:

```text
evidence_level = insufficient_evidence
```

even if validation loss appears improved.
