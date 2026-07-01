# E002 Multi-QKV Shift Register Plan

## Hypothesis

Deterministic hard-switched routing over a globally shared three-track bundled Q/K/V bank may change local small-GPT
learning behavior beyond standard attention and beyond static global-bank capacity alone.

This experiment does not claim broad transformer superiority, reasoning improvement, or scaling behavior.

## Source Idea Boundary

The first-build source idea has two parts:

```text
1. Keep a pool of same-role Q/K/V projection matrices.
2. Select one bundled Q/K/V triple by a fixed deterministic schedule.
```

The hypothesis is not "more parameters." The hypothesis is that deterministic rotation pressures shared banked Q/K/V weights
toward depth- or phase-portable subroutines.

## Non-Goals

Do not implement or interpret these as first-build evidence:

```text
typed content/control/binding streams
learned routing
softmix routing
warmup or annealed routing
LoRA deltas
stochastic schedules
decoupled/coprime Q/K/V clocks
layer-local bank ablations
```

Existing old skeleton configs remain `status: experimental_unimplemented` future variants.

## Canonical Initial Run Matrix

| Run | Attention Type | Role |
| --- | --- | --- |
| `standard_refactor_control_30m_seed1` | `standard` | Shared-path standard control |
| `multi_qkv_static_3track_global_30m_seed1` | `multi_qkv_static_3track_global` | A: static global 3-track cycle |
| `multi_qkv_train_rotation_3track_global_30m_seed1` | `multi_qkv_train_rotation_3track_global` | B: train-time rotation, eval freeze |
| `multi_qkv_position_rotation_3track_global_30m_seed1` | `multi_qkv_position_rotation_3track_global` | C: position-clock routing at train/eval/generate |

B and C are primarily interpreted relative to A/static-global, not merely standard attention.

`standard_30m_seed1.yaml` remains runnable as a legacy/noncanonical comparison config. It is not one of the four
canonical first-build configs. Validation reports:

```text
canonical_first_build_config_count = 4
legacy_or_auxiliary_runnable_config_count = 1
runnable_config_count = 5
unimplemented_config_count = 6
```

## Architecture Contract

All Multi-QKV first-build variants use one globally shared bank:

```text
Q_bank[0..2], K_bank[0..2], V_bank[0..2]
```

The repo implementation may use packed projections:

```text
c_attn_bank[0..2]: Linear(n_embd, 3*n_embd)
```

Q/K/V are bundled:

```text
Q = Q_bank[active_track]
K = K_bank[active_track]
V = V_bank[active_track]
```

The active track is deterministic, hard-switched, not learned, and not content-dependent.

`c_proj` remains layer-local in the first build.

## A/B/C Formulas

```text
track_count = 3
```

A, static global:

```text
active_track(layer_idx, step, pos, mode) = layer_idx mod 3
```

B, train rotation:

```text
training: active_track(layer_idx, step, pos, mode="train") = (layer_idx + step) mod 3
eval:     active_track(layer_idx, step, pos, mode in {"eval", "generate"}) = layer_idx mod 3
```

B must raise if training step is missing. The training-step clock must not be active during eval or generation.

C, position rotation:

```text
active_track(layer_idx, step, pos, mode) = (layer_idx + pos) mod 3
```

C routes per position. It is invalid if it collapses to scalar layer routing.

Current generation does not use a KV cache. For `multi_qkv_position_rotation_3track_global`, position IDs during
generation are recomputed for the full cropped context window passed to `GPT.forward`. Therefore the current
implementation uses window-relative positions during generation. If incremental KV-cache generation is added later, E002
C must define and test whether routing uses absolute generated-token positions or window-relative positions before the
KV-cache path is enabled.

## Fixed Training Contract

Direct comparisons use the same FineWeb-Edu 100M manifest, GPT-2 tokenizer, train/val shards, optimizer, LR schedule, seed,
batch construction, eval cadence, checkpoint cadence, and verifier/eval/summarize commands.

Core shape and budget:

```text
data_root: data/fineweb_edu_100m
tokenizer: gpt2
vocab_size: 50304
train_tokens: 100000000
val_tokens: 4000000
block_size: 1024
n_layer: 6
n_head: 6
n_embd: 384
B: 4
T: 1024
total_batch_size: 262144
max_steps: 3000
```

Because the Q/K/V bank is global rather than per-layer, Multi-QKV variants have fewer parameters than standard attention.
Parameter counts must be reported explicitly.

## Diagnostics

Multi-QKV variants emit diagnostics to:

```text
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/attention_diagnostics.jsonl
```

Required fields include:

```text
schema_version
experiment_id
run_name
attention_type
route_formula
uses_global_bank
track_count
layer_idx
layer
step
last_forward_step
schedule_mode
active_track_index
active_track_counts
track_gradient_norm
per_track_gradient_norm
per_track_qkv_weight_norm
position_routing_enabled
eval_freeze_mode
```

The queue mechanism check is:

```yaml
queue:
  mechanism_check: qkv_track_activity
  allow_missing_diagnostics: false
```

`qkv_track_activity` must reject missing, empty, all-zero, scalar-only C, or missing-step B diagnostics.

Destructive/off-route diagnostics are written by:

```text
scripts/qkv_track_destructive_test.py
```

to:

```text
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/qkv_track_destructive_test.json
```

## Evidence Levels

Use these labels:

```text
not_implemented
implemented_not_run
screened_mechanism_active
full_run_verified
candidate_evidence
insufficient_evidence
killed
```

Validation loss is not interpretable without mechanism diagnostics and destructive-test output. A Multi-QKV run with missing
or degenerate diagnostics is `insufficient_evidence` even if loss appears improved.

## Manual Full-Run Boundary

Implementation QC does not include full 3000-step E002 runs. Full runs are manual/operator work from a frozen source state.

Do not claim completion until actual artifacts pass:

```text
train
verify_run
eval_loss
eval_generate
eval_hellaswag
summarize_run
final verify_run
qkv_track_activity
qkv_track_destructive_test.py
```

## Manual Command Sequence

Run from codebase root:

```bash
git status --short
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register

scripts/experiments/E002_multitrack_qkv_shift_register/run_full_standard_refactor_control.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_static_global.sh

uv run scripts/qkv_track_destructive_test.py \
  --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml \
  --checkpoint runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  --out runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/evals/qkv_track_destructive_test.json \
  --num-batches 4

scripts/experiments/E002_multitrack_qkv_shift_register/run_full_train_rotation_global.sh

uv run scripts/qkv_track_destructive_test.py \
  --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml \
  --checkpoint runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  --out runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1/evals/qkv_track_destructive_test.json \
  --num-batches 4

scripts/experiments/E002_multitrack_qkv_shift_register/run_full_position_rotation_global.sh

uv run scripts/qkv_track_destructive_test.py \
  --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml \
  --checkpoint runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  --out runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1/evals/qkv_track_destructive_test.json \
  --num-batches 4

scripts/experiments/E002_multitrack_qkv_shift_register/compare_initial_full_runs.sh
```

## Comparison Requirements

Comparison output must include:

```text
final validation loss
best validation loss
perplexity
tokens/sec
VRAM
total/trainable/attention projection parameters
global qkv bank parameters
HellaSwag result if available
mechanism_check_passed
destructive_test_passed
evidence_level
```

## Success Criteria

```text
[ ] Candidate configs instantiate and validate.
[ ] Standard control remains unchanged.
[ ] Multi-QKV bank identity is global across blocks.
[ ] Mechanism diagnostics show nonzero selected-track activity.
[ ] C diagnostics show per-position routing.
[ ] B diagnostics show train-step routing and eval freeze.
[ ] Full-run comparisons pass manifest-aware verify/eval/summarize checks after manual execution.
[ ] B/C are interpreted primarily relative to A/static-global.
```

## Kill Criteria

```text
- Standard refactor control differs from standard behavior without an explained intentional change.
- A cannot train stably.
- A diagnostics do not prove global bank routing.
- B silently uses step=0 when training step is missing.
- B uses step rotation during eval/generation.
- C implements scalar layer routing instead of per-position routing.
- C breaks causal masking.
- Any candidate cannot reload checkpoint.
- Any candidate fails manifest verification.
- Any candidate emits missing or degenerate diagnostics.
- Destructive/off-route test shows no measurable output/logit/loss effect from changing track routes.
```

## Result Claim Boundary

At implementation completion, the only valid claim is:

```text
The initial E002 global Multi-QKV A/B/C experiments are implemented, tested, documented, and ready for manual full-run execution.
```
