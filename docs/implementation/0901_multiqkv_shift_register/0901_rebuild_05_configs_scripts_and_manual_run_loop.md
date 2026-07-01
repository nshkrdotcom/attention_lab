# 0901 Rebuild 05: Configs, Scripts, and Manual Run Loop

## Purpose

This document defines the canonical E002 config files, full-run scripts, comparison script, and manual human/operator run loop
for the first-build Multi-QKV Shift Register experiment.

This document does not claim results. Full 3000-step training runs are manual/operator work.

## Absolute Boundary

The implementation prepares:

```text
configs
hypothesis docs
manual run scripts
report templates
comparison tooling
destructive-test tooling
validation hooks
```

The implementation must not create or fake:

```text
completed run directories
checkpoint files
val_loss.json
hellaswag.json
run_summary.json
comparison JSON with fake metrics
qkv_track_destructive_test.json without a checkpoint/data forward pass
result claims
```

## Canonical Config Files

The only first-build canonical configs are:

```text
configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

Existing old skeleton names may remain only as `status: experimental_unimplemented` future variants.

`standard_30m_seed1.yaml` may remain runnable as a legacy/noncanonical comparison config. It is not one of the four
canonical first-build configs. The expected E002 validation counts are:

```text
canonical_first_build_config_count = 4
legacy_or_auxiliary_runnable_config_count = 1
runnable_config_count = 5
unimplemented_config_count = 6
```

## Shared Config Contract

All four canonical configs use the E002 30M-ish training contract:

```yaml
data:
  data_root: data/fineweb_edu_100m
  tokenizer: gpt2
  vocab_size: 50304
  train_tokens: 100000000
  val_tokens: 4000000

train:
  device: cuda
  dtype: bfloat16
  compile: false
  eval_at_start: true
  B: 4
  T: 1024
  total_batch_size: 262144
  max_steps: 3000
  grad_clip: 1.0
  weight_decay: 0.1
  learning_rate: 0.0006
  min_lr: 0.00006
  warmup_steps: 100
  val_every: 250
  val_steps: 20
  save_every: 1000
  log_every: 10

sample:
  sample_every: 1000
  prompt: "The history of mathematics"
  num_samples: 4
  max_new_tokens: 96
  top_k: 50
  temperature: 1.0
  seed: 42
```

Multi-QKV configs add:

```yaml
diagnostics:
  attention_diagnostics_every: 250

queue:
  mechanism_check: qkv_track_activity
  allow_missing_diagnostics: false
```

## Canonical Variants

Standard refactor control:

```yaml
model:
  attention_type: standard
queue:
  family: multitrack_qkv_shift_register
  full_run_approved: false
  allow_overwrite_existing_run_dir: false
```

A: static global cycle:

```yaml
model:
  attention_type: multi_qkv_static_3track_global
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_mod
queue:
  requires_run: standard_refactor_control_30m_seed1
```

Formula:

```text
active_track(layer_idx, step, position, mode) = layer_idx mod 3
```

B: train-time rotation with eval freeze:

```yaml
model:
  attention_type: multi_qkv_train_rotation_3track_global
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_plus_step_train_layer_eval
queue:
  requires_run: multi_qkv_static_3track_global_30m_seed1
```

Training formula:

```text
active_track(layer_idx, step, position, mode="train") = (layer_idx + step) mod 3
```

Eval/generation formula:

```text
active_track(layer_idx, step, position, mode in {"eval", "generate"}) = layer_idx mod 3
```

C: position rotation:

```yaml
model:
  attention_type: multi_qkv_position_rotation_3track_global
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_plus_position
queue:
  requires_run: multi_qkv_static_3track_global_30m_seed1
```

Formula:

```text
active_track(layer_idx, step, position, mode) = (layer_idx + position) mod 3
```

Current generation does not use a KV cache. For `multi_qkv_position_rotation_3track_global`, position IDs during
generation are recomputed for the full cropped context window passed to `GPT.forward`. Therefore the current
implementation uses window-relative positions during generation. If incremental KV-cache generation is added later, E002
C must define and test whether routing uses absolute generated-token positions or window-relative positions before the
KV-cache path is enabled.

## Hypothesis Docs

Required hypothesis docs:

```text
docs/experiments/E002_multitrack_qkv_shift_register/hypothesis_standard_refactor_control_30m_seed1.md
docs/experiments/E002_multitrack_qkv_shift_register/hypothesis_multi_qkv_static_3track_global_30m_seed1.md
docs/experiments/E002_multitrack_qkv_shift_register/hypothesis_multi_qkv_train_rotation_3track_global_30m_seed1.md
docs/experiments/E002_multitrack_qkv_shift_register/hypothesis_multi_qkv_position_rotation_3track_global_30m_seed1.md
```

Each uses exactly these evidence sections:

```text
CLAIM:
KILL_CONDITION:
MECHANISM_PROOF:
NEAREST_BORING_EXPLANATION:
CONTROL_THAT_RULES_IT_OUT:
```

For B and C, `CONTROL_THAT_RULES_IT_OUT` is `multi_qkv_static_3track_global_30m_seed1`, not merely standard attention.

## Full-Run Scripts

Executable scripts:

```text
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_standard_refactor_control.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_static_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_train_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_position_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_all_full_initial.sh
scripts/experiments/E002_multitrack_qkv_shift_register/compare_initial_full_runs.sh
```

The per-run scripts:

```text
1. cd to the repo root
2. define CONFIG and RUN_DIR
3. refuse existing run artifacts
4. verify data
5. train with the config
6. verify the run after training
7. run eval_loss
8. run eval_generate
9. run bounded eval_hellaswag
10. summarize the run
11. verify final artifacts
12. print the destructive-test command for Multi-QKV runs
```

The repo's checkpoint path is:

```text
<RUN_DIR>/checkpoints/ckpt_last.pt
```

not `<RUN_DIR>/ckpt.pt`.

The wrappers use the repo's real CLIs:

```bash
uv run scripts/train.py --config "$CONFIG" --overwrite
uv run scripts/verify_run.py --run_dir "$RUN_DIR" --expect-complete-training --expect-sample --expect-data-manifest
uv run scripts/eval_loss.py --checkpoint "$RUN_DIR/checkpoints/ckpt_last.pt" --data_root data/fineweb_edu_100m
uv run scripts/eval_generate.py --checkpoint "$RUN_DIR/checkpoints/ckpt_last.pt" --prompt "The history of mathematics"
uv run scripts/eval_hellaswag.py --checkpoint "$RUN_DIR/checkpoints/ckpt_last.pt" --max_examples 100
uv run scripts/summarize_run.py --run_dir "$RUN_DIR"
uv run scripts/verify_run.py --run_dir "$RUN_DIR" --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag --expect-data-manifest
```

## Comparison Script

`compare_initial_full_runs.sh` requires, before comparison:

```text
evals/run_summary.json
evals/val_loss.json
evals/hellaswag.json
evals/attention_diagnostics.jsonl for Multi-QKV runs
evals/qkv_track_destructive_test.json for Multi-QKV runs
```

It writes pairwise comparisons:

```text
reports/experiments/E002_multitrack_qkv_shift_register/comparison_static_global_vs_standard_refactor.json
reports/experiments/E002_multitrack_qkv_shift_register/comparison_train_rotation_global_vs_static_global.json
reports/experiments/E002_multitrack_qkv_shift_register/comparison_position_rotation_global_vs_static_global.json
```

Comparison rows include:

```text
final_val_loss
best_val_loss
final_val_perplexity
median_tokens_per_sec
peak_vram_allocated_mb
parameters_including_positional
trainable_parameters
global_qkv_bank_parameters
attention_projection_parameters
hellaswag_acc
mechanism_check_passed
destructive_test_passed
evidence_level
```

## Report Files

Required report files:

```text
reports/experiments/E002_multitrack_qkv_shift_register/README.md
reports/experiments/E002_multitrack_qkv_shift_register/results.md
reports/experiments/E002_multitrack_qkv_shift_register/results_template.md
reports/experiments/E002_multitrack_qkv_shift_register/initial_comparison_template.json
```

`results.md` must not claim full-run outcomes until generated artifacts exist.

## Human Manual Run Loop

Run from codebase root.

### Step 0: Preflight

```bash
pwd
git status --short
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
```

Stop if CUDA, data verification, experiment validation, or source-state checks fail.

### Step 1: Inspect Configs

```bash
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

Record parameter counts before interpreting comparisons.

### Step 2: Run Standard Refactor Control

```bash
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_standard_refactor_control.sh
```

Stop if this fails.

### Step 3: Run A Static Global

```bash
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_static_global.sh
```

Then:

```bash
uv run scripts/qkv_track_destructive_test.py \
  --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml \
  --checkpoint runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  --out runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/evals/qkv_track_destructive_test.json \
  --num-batches 4
```

Stop if A fails; B and C are not interpretable without A.

### Step 4: Run B Train Rotation

```bash
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_train_rotation_global.sh
```

Then run the destructive test with B's config/checkpoint.

### Step 5: Run C Position Rotation

```bash
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_position_rotation_global.sh
```

Then run the destructive test with C's config/checkpoint. C may be slower because the first correct implementation computes
all three Q/K/V projections and selects per position.

### Step 6: Compare

```bash
scripts/experiments/E002_multitrack_qkv_shift_register/compare_initial_full_runs.sh
```

### Step 7: Update Results From Artifacts

Only after comparison artifacts exist, update:

```text
reports/experiments/E002_multitrack_qkv_shift_register/results.md
```

Populate metrics only from generated JSON artifacts. Every claim must point to an artifact path.

## Stop/Continue Decision Loop

After each run ask:

```text
1. Did train complete?
2. Did verify_run pass?
3. Did eval_loss pass?
4. Did eval_generate pass?
5. Did eval_hellaswag pass or fail in a documented nonblocking way?
6. Did summarize_run pass?
7. Did attention diagnostics exist?
8. Did qkv_track_activity pass?
9. Did destructive test pass?
10. Is the run interpretable?
```

If diagnostics fail but loss looks good, set `evidence_level = insufficient_evidence`.

## Queue Compatibility

Manual scripts are primary for initial E002. Configs remain queue-compatible through:

```yaml
queue:
  family: multitrack_qkv_shift_register
  full_run_approved: false
  allow_overwrite_existing_run_dir: false
  requires_run: ...
  mechanism_check: qkv_track_activity
  allow_missing_diagnostics: false
```

Do not bypass queue approval and clobber-protection semantics if using the queue.

## Implementation-Agent QC

Run:

```bash
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

Do not run:

```bash
scripts/experiments/E002_multitrack_qkv_shift_register/run_all_full_initial.sh
```

during implementation QC.

## Operator Handoff Commands

```bash
# From codebase root
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

## Handoff To Next Doc

The final document is the copy-paste implementation prompt and final acceptance checklist.
