# Attention Lab

Attention Lab is a local GPT pretraining harness for controlled attention-architecture experiments.

The purpose is **mechanistic interpretability research on novel attention architectures**, not production efficiency, chat fine-tuning, API evals, or distributed frontier-scale training. The core question is whether deliberately nonstandard attention variants can expose more separable, local, stable, or causally controllable mechanisms than standard transformer attention.

The harness is designed to support:

```text
architecture variant -> real local training -> checkpointed artifacts
-> diagnostics -> mechanism backfill -> post-hoc probes
-> matched controls -> cautious interpretation
```

Efficiency still matters as a constraint: variants must be trainable, stable, and comparable enough to produce mechanisms worth interpreting. But efficiency is not the research objective.

## What this repo is

Attention Lab is:

* a single-GPU local GPT-style pretraining harness;
* a controlled attention-architecture experiment framework;
* a screen-first experiment runner for cheap candidate filtering;
* a checkpoint/reports/probes system for post-hoc mechanism investigation;
* a place to test whether architecture can change feature separation, causal locality, routing/content separation, operator-like behavior, or superposition structure.

Attention Lab is not:

* a chat fine-tuning stack;
* an API evaluation framework;
* a general distributed pretraining platform;
* a production-efficient transformer proposal;
* a benchmark-chasing leaderboard project.

## Screen-first experiment workflow

Attention Lab is screen-first for architecture exploration.
Full 3000-step runs are promotion artifacts, not default exploration.
A screen report is not a scientific result.

## Current evidence boundary

README.md explains how to use the harness. It is **not** the source of truth for fast-changing experiment state.

Before interpreting experiment results, read:

```text
EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md
```

That file is the dynamic reconciliation layer for completed, partial, stale, screen-only, checkpoint-backed, and not-yet-verified artifacts.

If these disagree:

```text
README.md
reports/
runs/
queue state
generated gauntlet reports
mechanism backfill inventories
```

then reconcile the state in `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md`, regenerate reports, and only then interpret results.

A few hard rules:

* A config existing is not evidence.
* A queue row existing is not evidence.
* A screen report is not a scientific result.
* A checkpoint means post-hoc recomputation is possible, not that the mechanism claim is true.
* A full run is not evidence until train, eval, summarize, and verify pass on real artifacts.
* Mechanistic claims require component-level probes, interventions, matched controls, and clear falsifiers.

## Current high-level status

README.md is not the live experiment ledger.

For current E001-E004 state, use:

- EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md
- reports/mechanisms/backfill/
- reports/mechanisms/probes/
- reports/mechanisms/cross_experiment_candidate_report.md

Keep this README focused on stable setup, workflow, and evidence-boundary rules.

## Repository map

```text
configs/                         Training and experiment configs
configs/experiments/             Registered architecture experiments
configs/mechanisms/              Prompt files and mechanism-probe helper configs
data/                            Local tokenized datasets and manifests
  fineweb_edu_100m/              Default dataset location; .npy shards are not committed
docs/                            Contracts, plans, implementation notes, and guides
reports/                         Human-readable and JSON reports
reports/experiments/             Experiment reports, gauntlet reports, run indexes
reports/mechanisms/              Backfill inventories, probe artifacts, mechanism reports
runs/                            Training outputs; ignored except .gitkeep
runs/screen/                     Gauntlet screen/rung artifacts
scripts/                         CLI wrappers for setup, train, eval, verify, queue, probes
src/attention_lab/               Library code
src/attention_lab/mechanisms/    Hook registry, capture, interventions, patching, backfill, probes
tests/                           Unit and integration tests
AGENTS.md                        Rules for coding agents working in this repo
```

Generated files are mostly ignored. In particular:

```text
data/**/*.npy
runs/*
queue runtime state
hellaswag/
wandb/
```

are not part of the committed repository unless intentionally added as derived reports or small metadata.

## Requirements

* Python 3.11 or 3.12.
* `uv` for dependency management.
* NVIDIA CUDA-capable GPU for real training.
* Enough disk space for token shards, checkpoints, run artifacts, HellaSwag cache, and optional larger datasets.

The project declares CUDA PyTorch wheels through the `pytorch-cu128` index on Linux.

The package exposes console/script entry points including:

```text
attention-lab-train
attention-lab-eval-loss
attention-lab-eval-generate
attention-lab-eval-hellaswag
attention-lab-verify-run
attention-lab-summarize-run
attention-lab-queue
attn-queue
```

Most commands in this README use `uv run scripts/...` because that keeps the workflow explicit and easy to audit.

## First-day setup

From the repository root:

```bash
uv sync
uv run scripts/verify_cuda.py
```

`verify_cuda.py` prints the installed Torch version, CUDA availability, CUDA version, device name, and BF16 support. It exits with an error if CUDA is unavailable.

Prepare the default FineWeb-Edu 100M train / 4M validation dataset:

```bash
scripts/prepare_fineweb_edu_100m.sh
```

That wrapper runs:

```bash
uv run scripts/prepare_fineweb_edu.py \
  --out_dir data/fineweb_edu_100m \
  --train_tokens 100000000 \
  --val_tokens 4000000

uv run scripts/write_data_manifest.py \
  --data_root data/fineweb_edu_100m \
  --out data/fineweb_edu_100m/manifest.json

uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
```

Run the 20-step sanity config before any real training:

```bash
uv run scripts/inspect_model_config.py \
  --config configs/baseline_15m_fineweb100m_sanity.yaml

uv run scripts/train.py \
  --config configs/baseline_15m_fineweb100m_sanity.yaml \
  --overwrite

uv run scripts/verify_run.py \
  --run_dir runs/baseline_15m_fineweb100m_sanity_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-data-manifest
```

The sanity config verifies dependencies, data loading, checkpointing, sampling, and run verification before spending GPU time on a real run.

## Dataset setup and recovery

The committed repo includes:

```text
data/fineweb_edu_100m/manifest.json
```

but does not include `.npy` token shards. Tokenized datasets are local artifacts and are ignored by Git.

A complete default dataset directory should look like:

```text
data/fineweb_edu_100m/
  manifest.json
  edufineweb_val_000000.npy
  edufineweb_train_000001.npy
```

The expected default manifest describes:

```text
dataset: HuggingFaceFW/fineweb-edu
dataset_config: sample-10BT
split: train
tokenizer: gpt2
validation tokens: 4,000,000
training tokens: 100,000,000
shard dtype: uint16
```

### If the dataset is missing

Run:

```bash
scripts/prepare_fineweb_edu_100m.sh
```

This streams `HuggingFaceFW/fineweb-edu`, tokenizes with GPT-2 BPE, writes `uint16` NumPy shards, writes a manifest, and verifies hashes.

### If the `.npy` shards exist but the manifest is missing

Rebuild and verify the manifest:

```bash
uv run scripts/write_data_manifest.py \
  --data_root data/fineweb_edu_100m \
  --out data/fineweb_edu_100m/manifest.json

uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
```

### If shards were manually downloaded or copied

Place them under the data root expected by the config. For the default 100M setup:

```text
data/fineweb_edu_100m/
```

Then write a fresh manifest and verify it:

```bash
uv run scripts/write_data_manifest.py \
  --data_root data/fineweb_edu_100m \
  --out data/fineweb_edu_100m/manifest.json

uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
```

Do not trust copied data until `verify_data.py --verify_hashes` passes against the manifest used by the training run.

### Larger dataset configs

Some configs expect larger local datasets that are not committed:

```text
configs/baseline_70m_fineweb300m.yaml  -> data/fineweb_edu_300m
configs/baseline_125m_fineweb1b.yaml   -> data/fineweb_edu_1b
configs/baseline_124m_fineweb1b.yaml   -> historical alias for the 125M config
```

Prepare larger datasets with the same script, changing `--out_dir` and `--train_tokens`:

```bash
uv run scripts/prepare_fineweb_edu.py \
  --out_dir data/fineweb_edu_300m \
  --train_tokens 300000000 \
  --val_tokens 4000000

uv run scripts/write_data_manifest.py \
  --data_root data/fineweb_edu_300m \
  --out data/fineweb_edu_300m/manifest.json

uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_300m \
  --manifest data/fineweb_edu_300m/manifest.json \
  --verify_hashes
```

For 1B tokens:

```bash
uv run scripts/prepare_fineweb_edu.py \
  --out_dir data/fineweb_edu_1b \
  --train_tokens 1000000000 \
  --val_tokens 4000000
```

then write and verify the manifest the same way.

## Baseline configs

Recommended baseline configs:

```text
configs/baseline_15m_fineweb100m_sanity.yaml  20-step sanity run
configs/baseline_16m_fineweb100m.yaml         smaller true ~16M tier
configs/baseline_30m_fineweb100m.yaml         canonical local 30M baseline
configs/baseline_70m_fineweb300m.yaml         larger local baseline template
configs/baseline_125m_fineweb1b.yaml          125M-ish template; needs 1B-token data
```

`configs/baseline_15m_fineweb100m.yaml` is a historical name. The model shape is the same as the canonical 30M config: 6 layers, 6 heads, embedding size 384, block size 1024. Prefer:

```text
configs/baseline_30m_fineweb100m.yaml
```

for new 30M baseline runs.

The completed historical baseline run, if present locally, is:

```text
runs/baseline_15m_fineweb100m_seed1
```

Its README-recorded summary was:

```text
final_val_loss: 4.081209182739258
best_val_loss: 4.081209182739258
final_val_perplexity: 59.2170307875361
median_tokens_per_sec: 107022.7422894312
peak_vram_allocated_mb: 3240.92431640625
bounded_hellaswag_accuracy_norm: 0.34
```

That run processed:

```text
3000 * 262144 = 786432000
```

token positions. This is multiple passes over the 100M-token training shard, not a unique 786M-token corpus.

## Running a full baseline

Use the canonical 30M config for new baseline work:

```bash
uv run scripts/train.py \
  --config configs/baseline_30m_fineweb100m.yaml \
  --overwrite
```

Verify the run:

```bash
uv run scripts/verify_run.py \
  --run_dir runs/baseline_30m_fineweb100m_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-data-manifest
```

Run evaluations and summarize:

```bash
uv run scripts/eval_loss.py \
  --checkpoint runs/baseline_30m_fineweb100m_seed1/checkpoints/ckpt_last.pt

uv run scripts/eval_generate.py \
  --checkpoint runs/baseline_30m_fineweb100m_seed1/checkpoints/ckpt_last.pt \
  --prompt "The history of mathematics"

uv run scripts/eval_hellaswag.py \
  --checkpoint runs/baseline_30m_fineweb100m_seed1/checkpoints/ckpt_last.pt \
  --max_examples 100

uv run scripts/summarize_run.py \
  --run_dir runs/baseline_30m_fineweb100m_seed1

uv run scripts/verify_run.py \
  --run_dir runs/baseline_30m_fineweb100m_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-eval-loss \
  --expect-hellaswag \
  --expect-data-manifest
```

## Experiment organization

Experiments are registered in:

```text
docs/experiments/experiments.yaml
```

Each experiment follows this layout:

```text
configs/experiments/<EXPERIMENT_ID>/       Configs for all runs in the experiment
docs/experiments/<EXPERIMENT_ID>_plan.md   Hypothesis, controls, gates, and commands
reports/experiments/<EXPERIMENT_ID>/       Comparison reports, run indexes, gauntlet reports
runs/experiments/<EXPERIMENT_ID>/          Generated full-run artifacts
runs/screen/                               Gauntlet screen/rung artifacts
```

List and validate experiments:

```bash
uv run scripts/list_experiments.py
uv run scripts/list_experiments.py --id E001_cp_trilinear_attention

uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
uv run scripts/validate_experiment.py --id E003_qkv_architecture_gauntlet
uv run scripts/validate_experiment.py --id E004_operator_binding_qkv_gauntlet
```

## Attention implementations

Attention modules live under:

```text
src/attention_lab/models/attention/
```

The model selects an implementation through config:

```yaml
model:
  attention_type: standard
```

Implemented canonical types include:

```text
standard
cp_bilinear
cp_trilinear
multi_qkv_static_3track_global
multi_qkv_train_rotation_3track_global
multi_qkv_position_rotation_3track_global
differential_qkv_anti_value
scope_gated_qkv
operator_valued_attention
q3k3v3_role_routed_attention
dynamic_value_query_conditioned_attention
```

The historical `trilinear_cp` placeholder is intentionally unimplemented. Use:

```text
cp_trilinear
```

for E001.

When adding an attention variant:

1. Read `docs/architecture_experiment_contract.md`.
2. Read `docs/architecture_variant_checklist.md`.
3. Add the implementation under `src/attention_lab/models/attention/`.
4. Register it through the attention registry.
5. Add config validation support.
6. Add tests for shape, causal masking, gradient flow, parameter count, and diagnostics.
7. Add hook-site specs for mechanism capture where appropriate.
8. Keep data, seed, optimizer, learning-rate schedule, token budget, batch construction, checkpoint cadence, and eval cadence fixed for direct comparisons.
9. Add or update an experiment plan before interpreting results.

Do not rewrite the trainer just to test an architecture variant.

## Experiment E001: CP trilinear attention

Experiment ID:

```text
E001_cp_trilinear_attention
```

Plan:

```text
docs/experiments/E001_cp_trilinear_attention_plan.md
```

Canonical configs:

```text
configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
configs/experiments/E001_cp_trilinear_attention/standard_refactor_control_30m_seed1.yaml
configs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1.yaml
configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1.yaml
configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_lambda0_30m_seed1.yaml
```

Current local state, as summarized in the dynamic status file:

```text
standard_30m_seed1                  checkpoint available
cp_bilinear_r8_30m_seed1             checkpoint available
cp_trilinear_r8_30m_seed1            checkpoint available
cp_trilinear_r8_lambda0_30m_seed1    checkpoint unavailable / incomplete
standard_refactor_control_30m_seed1  checkpoint unavailable for E001
```

Validate and inspect:

```bash
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention

uv run scripts/inspect_model_config.py \
  --config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml

uv run scripts/inspect_model_config.py \
  --config configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1.yaml \
  --baseline-config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
```

Manual promotion-stage full-run scripts:

```bash
scripts/experiments/E001_cp_trilinear_attention/run_full_standard_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_bilinear_r8_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_trilinear_r8_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_trilinear_r8_lambda0_30m.sh
```

These scripts are for frozen, promoted full runs only. They are not the default exploration path.

Individual full-run scripts refuse direct execution unless this is set:

```bash
ATTENTION_LAB_I_UNDERSTAND_THIS_IS_A_PROMOTED_FULL_RUN=1
```

That acknowledgement is not a substitute for scientific justification.

Compare after required summaries exist:

```bash
scripts/experiments/E001_cp_trilinear_attention/compare_full_runs.sh
```

CP diagnostics are written to:

```text
runs/experiments/E001_cp_trilinear_attention/<run_name>/evals/attention_diagnostics.jsonl
```

The diagnostics schema is:

```text
reports/schema/attention_diagnostics.schema.json
```

### E001 interpretation boundary

Earlier local summaries showed CP-trilinear reaching lower final validation loss than standard and CP-bilinear in one seed, but at severe throughput and VRAM cost.

The correct claim shape is:

```text
In one local ~30M GPT / FineWeb-Edu 100M E001 run,
CP-trilinear reached a lower final validation loss than both standard attention
and CP-bilinear, but at severe throughput and VRAM cost.
This warrants replication and mechanism follow-up, not an architecture superiority claim.
```

## Experiment E002: Multi-QKV shift/register

Experiment ID:

```text
E002_multitrack_qkv_shift_register
```

Plan:

```text
docs/experiments/E002_multitrack_qkv_shift_register_plan.md
```

Canonical first-build configs:

```text
configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

Current local state is reconciled in `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md`. In this working copy, the canonical E002 standard/candidate checkpoints are present locally. These are local reconciliation facts for generated artifacts; a fresh clone may not contain the checkpoint files.

```text
standard_refactor_control_30m_seed1                  checkpoint_available_at_last_local_reconciliation: true
multi_qkv_static_3track_global_30m_seed1             checkpoint_available_at_last_local_reconciliation: true
multi_qkv_train_rotation_3track_global_30m_seed1     checkpoint_available_at_last_local_reconciliation: true
multi_qkv_position_rotation_3track_global_30m_seed1  checkpoint_available_at_last_local_reconciliation: true
```

Validate:

```bash
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
```

Manual promotion-stage full-run scripts:

```bash
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_standard_refactor_control.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_static_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_train_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_position_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/compare_initial_full_runs.sh
```

These are for approved full runs after promotion.

Individual full-run scripts require:

```bash
ATTENTION_LAB_I_UNDERSTAND_THIS_IS_A_PROMOTED_FULL_RUN=1
```

Multi-QKV diagnostics are written to:

```text
runs/experiments/E002_multitrack_qkv_shift_register/<run_name>/evals/attention_diagnostics.jsonl
```

Run the destructive route test after a candidate checkpoint exists:

```bash
uv run scripts/qkv_track_destructive_test.py \
  --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml \
  --checkpoint runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  --out runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/evals/qkv_track_destructive_test.json \
  --num-batches 4
```

Old E002 skeleton configs with:

```text
status: experimental_unimplemented
```

are placeholders for future work, not first-build evidence.

### E002 route-index semantics

A real bug was fixed in the E002 position-rotation mechanism probe path.

Correct semantics:

```text
selected_track is a discrete route/index diagnostic site.
track_q, track_k, track_v, and track_out are continuous intervention sites.
```

Do not treat `selected_track` as an ordinary floating-point activation for zero/scale interventions. Capture it for diagnostics. Apply continuous interventions to track tensors.

Use this pattern:

```bash
uv run scripts/run_mechanism_probe.py \
  --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml \
  --checkpoint runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  --prompts-file configs/mechanisms/quick_probe_prompts.txt \
  --sites selected_track,track_q,track_k,track_v,track_out \
  --intervention-sites track_q,track_k,track_v,track_out \
  --interventions zero,scale \
  --layer 0 \
  --scale 0.0 \
  --output-dir reports/mechanisms/probes/E002_position_rotation_quick
```

### E002 interpretation boundary

E002 is ready for route-specialization mechanism work, but not final scientific claims.

The next question is:

```text
Do the global QKV tracks specialize into distinguishable routing/content roles,
and do route/track interventions produce predictable localized effects?
```

## Experiment E003: QKV architecture gauntlet

Experiment ID:

```text
E003_qkv_architecture_gauntlet
```

Plan:

```text
docs/experiments/E003_qkv_architecture_gauntlet_plan.md
```

E003 is an interpretability-oriented QKV architecture gauntlet, not an efficiency experiment.

Implemented variants:

```text
differential_qkv_anti_value  positive and negative/suppressive QKV branches
scope_gated_qkv              content stream plus explicit scope/operator stream and receiver-side gate
standard_refactor_control    matched standard-attention control
```

The gauntlet uses staged screen rungs:

```text
rung020 -> rung150 -> rung500
```

Current local state is reconciled in `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md`. In this working copy, the E003 screen checkpoints listed here are present locally. These are local reconciliation facts for generated artifacts; a fresh clone may not contain the checkpoint files. A checkpoint makes recomputation possible; it does not prove a mechanism claim.

```text
differential_qkv_anti_value_30m_seed1_rung020       checkpoint_available_at_last_local_reconciliation: true
differential_qkv_anti_value_30m_seed1_rung150       checkpoint_available_at_last_local_reconciliation: true
differential_qkv_anti_value_30m_seed1_rung500       checkpoint_available_at_last_local_reconciliation: true

scope_gated_qkv_30m_seed1_rung020                   checkpoint_available_at_last_local_reconciliation: true
scope_gated_qkv_30m_seed1_rung150                   checkpoint_available_at_last_local_reconciliation: true
scope_gated_qkv_30m_seed1_rung500                   checkpoint_available_at_last_local_reconciliation: true

standard_refactor_control_30m_seed1_rung020         checkpoint_available_at_last_local_reconciliation: true
standard_refactor_control_30m_seed1_rung150         checkpoint_available_at_last_local_reconciliation: true
standard_refactor_control_30m_seed1_rung500         checkpoint_available_at_last_local_reconciliation: true
```

Canonical current post-hoc probes:

```text
reports/mechanisms/probes/E003_differential_rung500_inventory_path/
reports/mechanisms/probes/E003_scope_gated_rung500_inventory_path/
```

Validate and run the gauntlet:

```bash
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
uv run scripts/validate_experiment.py --id E003_qkv_architecture_gauntlet
scripts/experiments/E003_qkv_architecture_gauntlet/run_gauntlet.sh
uv run attn-queue gauntlet-report --experiment E003_qkv_architecture_gauntlet
```

Inspect the plan without launching training:

```bash
uv run attn-queue gauntlet-plan \
  --experiment E003_qkv_architecture_gauntlet \
  --policy configs/experiments/E003_qkv_architecture_gauntlet/gauntlet_policy.yaml
```

Run one safe gauntlet action:

```bash
uv run attn-queue gauntlet-run \
  --experiment E003_qkv_architecture_gauntlet \
  --policy configs/experiments/E003_qkv_architecture_gauntlet/gauntlet_policy.yaml \
  --once
```

Do not pass `--allow-full` unless intentionally approving long full runs.

### E003 interpretation boundary

E003 has committed rung/probe/Tier-1 report artifacts, but local checkpoint-backed recomputation requires the `runs/screen` checkpoints to exist in the working copy.

It does not prove that differential or scope-gated streams form semantically clean mechanisms.

The next question is:

```text
Do branch_delta, positive/negative streams, scope_out, gate,
and content_scope_product provide more local causal handles than standard attention?
```

## Experiment E004: Operator-binding QKV gauntlet

Experiment ID:

```text
E004_operator_binding_qkv_gauntlet
```

Plan:

```text
docs/experiments/E004_operator_binding_qkv_gauntlet_plan.md
```

E004 builds on the E003 gauntlet infrastructure. It is not an efficiency experiment and not a model-improvement claim. It asks whether higher-risk QKV decompositions can survive early pretraining while producing nondegenerate, inspectable mechanisms.

Implemented variants:

```text
operator_valued_attention                    routed add/suppress/gate/transform/bind write modes
q3k3v3_role_routed_attention                 content/operator/binding role streams
dynamic_value_query_conditioned_attention    receiver-conditioned value read-mode gate
standard_refactor_control                    matched standard-attention control
```

The gauntlet uses staged screen rungs:

```text
rung020 -> rung150 -> rung500
```

Current local state is reconciled in `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md`. In this working copy, the E004 screen checkpoints listed here are present locally. These are local reconciliation facts for generated artifacts; a fresh clone may not contain the checkpoint files. A checkpoint makes recomputation possible; it does not prove a mechanism claim.

```text
operator_valued_attention_30m_seed2_rung020                    checkpoint_available_at_last_local_reconciliation: true
operator_valued_attention_30m_seed2_rung150                    checkpoint_available_at_last_local_reconciliation: true
operator_valued_attention_30m_seed2_rung500                    checkpoint_available_at_last_local_reconciliation: true

dynamic_value_query_conditioned_attention_30m_seed2_rung020    checkpoint_available_at_last_local_reconciliation: true
dynamic_value_query_conditioned_attention_30m_seed2_rung150    checkpoint_available_at_last_local_reconciliation: true
dynamic_value_query_conditioned_attention_30m_seed2_rung500    checkpoint_available_at_last_local_reconciliation: true

q3k3v3_role_routed_attention_30m_seed2_rung020                 checkpoint_available_at_last_local_reconciliation: true

standard_refactor_control_30m_seed2_rung020                    checkpoint_available_at_last_local_reconciliation: true
standard_refactor_control_30m_seed2_rung150                    checkpoint_available_at_last_local_reconciliation: true
standard_refactor_control_30m_seed2_rung500                    checkpoint_available_at_last_local_reconciliation: true
```

Canonical current E004 operator-valued probe:

```text
reports/mechanisms/probes/E004_operator_valued_rung500_inventory_path/
```

Historical partial E004 operator-valued probe:

```text
reports/mechanisms/probes/E004_operator_valued_rung500/
```

Both point to the same checkpoint. The `_inventory_path` probe is richer and should be treated as canonical.

Validate and run the gauntlet:

```bash
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
uv run scripts/validate_experiment.py --id E004_operator_binding_qkv_gauntlet
scripts/experiments/E004_operator_binding_qkv_gauntlet/run_gauntlet.sh
uv run attn-queue gauntlet-report --experiment E004_operator_binding_qkv_gauntlet
```

Inspect the plan without launching training:

```bash
uv run attn-queue gauntlet-plan \
  --experiment E004_operator_binding_qkv_gauntlet \
  --policy configs/experiments/E004_operator_binding_qkv_gauntlet/gauntlet_policy.yaml
```

Run one safe gauntlet action:

```bash
uv run attn-queue gauntlet-run \
  --experiment E004_operator_binding_qkv_gauntlet \
  --policy configs/experiments/E004_operator_binding_qkv_gauntlet/gauntlet_policy.yaml \
  --once
```

Do not pass `--allow-full` unless intentionally approving long full runs.

Reports are written under:

```text
reports/experiments/E004_operator_binding_qkv_gauntlet/
```

### E004 interpretation boundary

E004 operator-valued has committed rung/probe/Tier-1 report artifacts, but local checkpoint-backed recomputation requires the `runs/screen` checkpoints to exist in the working copy.

It does not prove that operator probabilities or operator-specific outputs are semantically clean mechanisms.

The next question is:

```text
Do operator_probs, add/suppress/gate/transform/bind outputs,
and combined operator output provide more local causal handles
than standard attention or E003 stream decompositions?
```

## Screen-first experiment workflow

The normal workflow is:

1. Verify CUDA and data.
2. Validate experiment configs.
3. Run cheap screens or gauntlet rungs.
4. Generate and inspect promotion reports.
5. Promote only selected candidates.
6. Run full jobs only when explicitly approved.
7. Verify full artifacts.
8. Regenerate reports.
9. Run mechanism backfill and post-hoc probes.
10. Interpret only after matched controls and intervention evidence exist.

Concrete E002 queue example:

```bash
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register

uv run attn-queue add configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run attn-queue add configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
uv run attn-queue add configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
uv run attn-queue add configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml

uv run attn-queue start
uv run attn-queue status
uv run attn-queue promotion-report multi_qkv_static_3track_global_30m_seed1
uv run attn-queue approve multi_qkv_static_3track_global_30m_seed1
```

`SANITY` rows are non-evidence holding/smoke-intent rows, not executed evidence screens. Use them when recording that a config is not ready for screening yet:

```bash
uv run attn-queue add --stage SANITY configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
uv run attn-queue advance-to-screen multi_qkv_static_3track_global_30m_seed1
```

The approver checks the promotion report before setting a row executable as `FULL`. Directly setting `full_run_approved` is blocked in the ledger API.

Non-standard candidates need non-degenerate diagnostics. `queue.allow_missing_diagnostics: true` records an exception and yields `needs_investigation`, not a clean promotion.

Each screen preserves:

```text
screen_config.yaml
resolved_config.yaml
metrics.jsonl
queue_screen.log
checkpoints/ckpt_last.pt
evals/attention_diagnostics.jsonl when applicable
promotion_report.json
```

The canonical promotion report is also written under `reports/`.

## Experiment queue

The queue layer is a serial, single-GPU operator tool. It does not decide what is scientifically worth running.

It can:

* screen candidate configs;
* write promotion reports;
* record state in SQLite;
* block unsafe full runs;
* execute approved runs;
* export run indexes.

Add configs and inspect queue state:

```bash
uv run attn-queue add configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
uv run attn-queue status
uv run attn-queue ls
uv run attn-queue show standard_30m_seed1
uv run attn-queue note standard_30m_seed1 "SHOWS: pending screen"
```

Generate promotion reports and approve or block full runs:

```bash
uv run attn-queue promotion-report standard_30m_seed1
uv run attn-queue approve standard_30m_seed1
uv run attn-queue unapprove standard_30m_seed1
```

Start or stop the daemon:

```bash
uv run attn-queue start
uv run attn-queue stop
```

Review experiment readiness and results:

```bash
uv run attn-queue doctor --experiment E001_cp_trilinear_attention
uv run attn-queue doctor --experiment E002_multitrack_qkv_shift_register
uv run attn-queue doctor --experiment E003_qkv_architecture_gauntlet
uv run attn-queue doctor --experiment E004_operator_binding_qkv_gauntlet

uv run attn-queue leaderboard --min-stage FULL --sort loss
uv run attn-queue leaderboard --sort speed

uv run attn-queue export-report --experiment E001_cp_trilinear_attention

uv run attn-queue morning-note --experiment E001_cp_trilinear_attention \
  --shows "..." \
  --not-shows "..." \
  --next "..."
```

Gauntlet commands:

```bash
uv run attn-queue gauntlet-plan \
  --experiment E003_qkv_architecture_gauntlet \
  --policy configs/experiments/E003_qkv_architecture_gauntlet/gauntlet_policy.yaml

uv run attn-queue gauntlet-run \
  --experiment E003_qkv_architecture_gauntlet \
  --policy configs/experiments/E003_qkv_architecture_gauntlet/gauntlet_policy.yaml \
  --once

uv run attn-queue gauntlet-report \
  --experiment E003_qkv_architecture_gauntlet

uv run attn-queue gauntlet-plan \
  --experiment E004_operator_binding_qkv_gauntlet \
  --policy configs/experiments/E004_operator_binding_qkv_gauntlet/gauntlet_policy.yaml

uv run attn-queue gauntlet-run \
  --experiment E004_operator_binding_qkv_gauntlet \
  --policy configs/experiments/E004_operator_binding_qkv_gauntlet/gauntlet_policy.yaml \
  --once

uv run attn-queue gauntlet-report \
  --experiment E004_operator_binding_qkv_gauntlet
```

Queue safety rules:

* Full runs require a clean promotion report and explicit approval.
* Do not pass `--allow-full` casually.
* `SANITY` rows are non-evidence holding rows.
* Existing run directories are protected unless `queue.allow_overwrite_existing_run_dir: true` is explicit.
* Non-standard full runs require a passed control through `queue.requires_run` unless an explicit skip is documented.
* Non-standard screen promotion requires non-degenerate mechanism diagnostics.
* Missing diagnostics exceptions cannot cleanly promote.
* The screener lowers or injects diagnostics cadence for non-standard attention where needed.
* Screen length is configurable through `queue.screen_steps`, `queue.screen_val_every`, `queue.screen_save_every`, and `queue.screen_diagnostics_every`.

Supported mechanism checks include:

```text
cp_gradient_norm
qkv_track_activity
differential_qkv_activity
scope_gated_qkv_activity
operator_valued_activity
q3k3v3_role_activity
dynamic_value_activity
```

Read the detailed queue guide before using the queue for real full runs:

```text
docs/guides/experiment_queue_discipline_checklist.md
```

## Comparing runs

For individual comparisons:

```bash
uv run scripts/compare_runs.py \
  --experiment E001_cp_trilinear_attention \
  --baseline runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1 \
  --candidate runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1 \
  --json-out reports/experiments/E001_cp_trilinear_attention/comparison_cp_trilinear_r8_vs_standard.json
```

Comparison artifacts read:

```text
evals/run_summary.json
```

and report:

```text
loss
perplexity
throughput
VRAM
experiment metadata
candidate/control deltas and ratios
```

For Multi-QKV candidates, comparison output also includes mechanism and destructive-test fields when relevant artifacts exist.

## Mechanism investigation

Attention Lab includes a native mechanism investigation layer under:

```text
src/attention_lab/mechanisms/
```

It provides:

* deterministic hook-site registry;
* `ActivationCache` records;
* activation capture during real forward passes;
* zero, mean-ablation, scale, replace, and cache-patching interventions;
* E001–E004 mechanism backfill inventories;
* generated cross-experiment mechanism candidate reports;
* post-hoc probe CLI;
* strict missing/deployed hook-site checks.

This layer is native to the local GPT and attention modules. TransformerLens compatibility is an adapter goal, not a prerequisite.

### Backfill inventories

Backfill inventories are deterministic derived artifacts. They record the git commit used as the source state and mark paths as repo-root-relative. They intentionally do not include a timestamp, so repeated generation from the same tree is comparable.

Generate E001–E004 inventories:

```bash
uv run scripts/backfill_mechanism_inventory.py \
  --experiments E004 E003 E002 E001 \
  --repo-root . \
  --output-root reports/mechanisms/backfill

uv run scripts/compare_mechanism_candidates.py \
  --backfill-root reports/mechanisms/backfill \
  --output reports/mechanisms/cross_experiment_candidate_report.md
```

Backfill evidence levels:

```text
artifact_summary       metadata recoverable from existing configs/reports/summaries/diagnostics
checkpoint_recompute   checkpoint exists, so small-batch activations/interventions can be recomputed
not_available          checkpoint or required evidence is absent
```

Missing historical activations remain missing. Backfill reports list unavailable items explicitly as:

```text
missing
not_recorded
checkpoint_unavailable
```

### Checkpoint availability sanity check

Use this to ensure backfill availability is grounded in the actual filesystem:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

root = Path(".")
bad = []

for inv_path in sorted(Path("reports/mechanisms/backfill").glob("*/inventory.json")):
    inv = json.loads(inv_path.read_text())
    for row in inv["candidates"]:
        cp = row.get("checkpoint_path")
        status = row.get("checkpoint_status")
        run_name = row.get("run_name")

        if status == "available":
            if not cp:
                bad.append((inv_path, run_name, "available_but_no_checkpoint_path", cp))
            elif not (root / cp).exists():
                bad.append((inv_path, run_name, "available_but_path_missing", cp))

        if status != "available" and cp:
            bad.append((inv_path, run_name, "unavailable_but_checkpoint_path_present", cp))

if bad:
    print("BAD CHECKPOINT STATUS:")
    for item in bad:
        print("\t".join(map(str, item)))
    raise SystemExit(1)

print("OK: every available checkpoint path exists on disk")
PY
```

List all current checkpoints:

```bash
find runs -path '*/checkpoints/ckpt_last.pt' -print | sort
```

List backfill-available checkpoint rows:

```bash
jq -r '
  .candidates[]
  | select(.checkpoint_status=="available")
  | [.experiment_id, .run_name, .checkpoint_path]
  | @tsv
' reports/mechanisms/backfill/*/inventory.json | sort
```

### Post-hoc mechanism probes

Run a small post-hoc probe only when a checkpoint exists.

E004 operator-valued rung500 canonical example:

```bash
uv run scripts/run_mechanism_probe.py \
  --config configs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2_rung500.yaml \
  --checkpoint runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt \
  --prompts-file configs/mechanisms/quick_probe_prompts.txt \
  --sites operator_probs,operator_add_out,operator_suppress_out,operator_gate_out,operator_transform_out,operator_bind_out,operator_combined_out \
  --interventions zero,scale \
  --layer 0 \
  --scale 0.0 \
  --output-dir reports/mechanisms/probes/E004_operator_valued_rung500_inventory_path \
  --device cuda
```

E003 differential rung500:

```bash
uv run scripts/run_mechanism_probe.py \
  --config configs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1_rung500.yaml \
  --checkpoint runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt \
  --prompts-file configs/mechanisms/quick_probe_prompts.txt \
  --sites pos_q,pos_k,pos_v,neg_q,neg_k,neg_v,pos_out,neg_out,branch_delta,lambda \
  --interventions zero,scale \
  --layer 0 \
  --scale 0.0 \
  --output-dir reports/mechanisms/probes/E003_differential_rung500_inventory_path \
  --device cuda
```

E003 scope-gated rung500:

```bash
uv run scripts/run_mechanism_probe.py \
  --config configs/experiments/E003_qkv_architecture_gauntlet/scope_gated_qkv_30m_seed1_rung500.yaml \
  --checkpoint runs/screen/scope_gated_qkv_30m_seed1_rung500_bb3de557aae8/checkpoints/ckpt_last.pt \
  --prompts-file configs/mechanisms/quick_probe_prompts.txt \
  --sites content_out,scope_out,gate,content_scope_product,gated_content \
  --interventions zero,scale \
  --layer 0 \
  --scale 0.0 \
  --output-dir reports/mechanisms/probes/E003_scope_gated_rung500_inventory_path \
  --device cuda
```

`run_mechanism_probe.py` reads the tokenizer from the config, currently supports `gpt2`, validates token IDs against the configured vocabulary size, and writes tokenizer metadata into activation and probe summaries.

Unsupported tokenizers fail explicitly.

The probe CLI supports:

```text
zero
mean_ablate
scale
replace
patch_from_cache
```

Shape-compatible tensor replacement:

```bash
uv run scripts/run_mechanism_probe.py \
  --config <config.yaml> \
  --checkpoint <ckpt_last.pt> \
  --prompt "The history of mathematics" \
  --sites attn_out \
  --interventions replace \
  --replacement-tensor <tensor.pt> \
  --output-dir reports/mechanisms/probes/<probe_name>
```

Cache patching for selected batch/token positions:

```bash
uv run scripts/run_mechanism_probe.py \
  --config <config.yaml> \
  --checkpoint <ckpt_last.pt> \
  --prompt "The history of mathematics" \
  --sites attn_out \
  --interventions patch_from_cache \
  --source-cache <activation_cache_with_tensors.pt> \
  --source-site attn_out \
  --batch-indices 0 \
  --token-indices 3,4,5 \
  --output-dir reports/mechanisms/probes/<probe_name>
```

The CLI fails before model execution if:

* `scale` lacks `--scale`;
* `replace` lacks both `--replacement-tensor` and `--source-cache`;
* `patch_from_cache` lacks `--source-cache`.

Use `--sites` for capture sites and `--intervention-sites` when only some captured sites should be edited.

This matters for discrete route/index sites such as Multi-QKV `selected_track`.

### Tier-1 mechanism probe suite

The Tier-1 suite is the statistically controlled E003/E004 follow-up path. It adds trained linear probes, grouped train/test splitting, shuffled-label nulls, random-site nulls, matched controls, bootstrap CIs, FDR-BH correction, target-vs-decoy specificity gates, optional patch/restoration metrics, alignment-to-control metrics, and mechanism-probe claim gates.

Read first:

```text
docs/mechanism_probe_framework.md
```

Committed Tier-1 inputs:

```text
docs/mechanisms/hypotheses/E003_differential_negation_tier1.yaml
docs/mechanisms/hypotheses/E004_operator_valued_negation_tier1.yaml
configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml
configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml
```

Regenerate or validate task suites with `scripts/generate_tier1_mechanism_tasks.py`. The committed suites include GPT-2 single-token target/foil metadata plus clean/corrupt answer positions and explicit patch-token alignment for restoration.
They also include `metadata.content_sha256`; the fingerprint checks file integrity. Confirmatory suite execution and validate-only also regenerate built-in Tier-1 suites from metadata to reject tampered suites whose hash was recomputed. Future task generators need equivalent regeneration validation before their suites can support confirmatory claims.

Executable Tier-1 presets:

```text
E003 differential -> standard_refactor_control_30m_seed1_rung500
E004 operator-valued -> standard_refactor_control_30m_seed2_rung500
```

E003 exploratory cheap scan:

```bash
uv run scripts/run_mechanism_probe_suite.py \
  --experiment-id E003_qkv_architecture_gauntlet \
  --candidate differential \
  --checkpoint runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt \
  --task-file configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml \
  --output-dir reports/mechanisms/probes/E003_differential_tier1_probe_only_inventory_path \
  --exploratory \
  --probe-only \
  --feature-pooling mean_sequence \
  --control-mode matched \
  --min-n 50 \
  --bootstrap-samples 1000 \
  --fdr-alpha 0.05 \
  --seed 1
```

E003 confirmatory full run:

```bash
uv run scripts/run_mechanism_probe_suite.py \
  --experiment-id E003_qkv_architecture_gauntlet \
  --candidate differential \
  --checkpoint runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt \
  --task-file configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml \
  --hypothesis-doc docs/mechanisms/hypotheses/E003_differential_negation_tier1.yaml \
  --output-dir reports/mechanisms/probes/E003_differential_tier1_confirmatory_inventory_path \
  --feature-pooling patch_positions_mean \
  --control-mode matched \
  --min-n 50 \
  --bootstrap-samples 1000 \
  --fdr-alpha 0.05 \
  --seed 1
```

E004 exploratory cheap scan:

```bash
uv run scripts/run_mechanism_probe_suite.py \
  --experiment-id E004_operator_binding_qkv_gauntlet \
  --candidate operator_valued \
  --checkpoint runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt \
  --task-file configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml \
  --output-dir reports/mechanisms/probes/E004_operator_valued_tier1_probe_only_inventory_path \
  --exploratory \
  --probe-only \
  --feature-pooling mean_sequence \
  --control-mode matched \
  --min-n 50 \
  --bootstrap-samples 1000 \
  --fdr-alpha 0.05 \
  --seed 2
```

E004 confirmatory full run:

```bash
uv run scripts/run_mechanism_probe_suite.py \
  --experiment-id E004_operator_binding_qkv_gauntlet \
  --candidate operator_valued \
  --checkpoint runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt \
  --task-file configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml \
  --hypothesis-doc docs/mechanisms/hypotheses/E004_operator_valued_negation_tier1.yaml \
  --output-dir reports/mechanisms/probes/E004_operator_valued_tier1_confirmatory_inventory_path \
  --feature-pooling patch_positions_mean \
  --control-mode matched \
  --min-n 50 \
  --bootstrap-samples 1000 \
  --fdr-alpha 0.05 \
  --seed 2
```

Outputs:

```text
metrics.json
claim_gates.json
summary.md
```

Validate generated suite artifacts with:

```bash
uv run scripts/summarize_mechanism_probe_suite.py \
  --output-dir reports/mechanisms/probes/<suite-output-dir> \
  --validate
```

Preflight local checkpoint availability without fabricating artifacts:

```bash
uv run scripts/verify_tier1_mechanism_probe_suite.py --preflight-only
```

`candidate_mechanism_evidence` is mechanism-probe scoped and means single-seed, checkpoint-backed, statistically controlled evidence. It is not replication and not a global experiment status.
`exploratory_probe_signal` is a capped exploratory status, not a passed confirmatory claim gate. `claim_gates.json` records `exploratory_signal`, `controlled_probe_gate_passed`, `candidate_mechanism_gate_passed`, `highest_status`, `status_kind`, and the compatibility `claim_gate_passed` field for each cell. Treat `claim_gate_passed` as an alias for `candidate_mechanism_gate_passed`; a controlled-probe pass has its own boolean and does not imply causal mechanism evidence.

Confirmatory runs use strict preset site resolution. Unknown confirmatory `--sites` values fail before model execution; exploratory unknown sites require explicit `--site-spec-file` metadata and remain noncanonical. E004 `operator_probs` is a low-dimensional probability site and may lack a matched-dimensional random-site null. That is reported as a per-cell feasibility limit, not a run-wide implementation failure. Noncanonical control overrides and missing-control diagnostic runs are recorded but cap claims below evidence statuses.
For patch/restoration, E004 `operator_probs` is capture/probe-only until a validated probability-site intervention exists; full-width operator output sites remain continuous patch candidates.
The random-site null pool is the complete preset-declared Tier-1 null family and is recorded in `metrics.json` preflight metadata; it is not an unrestricted hook sweep. Actual selection still requires matched dimensionality and compatible tensor kind from captured tensors.

### Strict capture mode

Capture-all can be made strict through:

```python
capture_activations(..., require_declared_sites=True)
```

Strict mode reports declared-but-unemitted hook sites separately from explicitly requested missing sites. Disabled optional branches and unsupported declared sites are visible without faking tensors.

## Output artifacts

A completed run directory usually contains:

```text
config.yaml
config_source.txt
data_manifest.json
data_manifest.sha256
environment.txt
git_commit.txt
metrics.csv
metrics.jsonl
checkpoints/ckpt_last.pt
checkpoints/ckpt_step_*.pt
samples/*.txt
evals/val_loss.json
evals/hellaswag.json
evals/run_summary.json
evals/attention_diagnostics.jsonl      for non-standard attention where applicable
evals/qkv_track_destructive_test.json  for Multi-QKV route tests where applicable
```

A screen/rung directory under `runs/screen` usually contains:

```text
screen_config.yaml
resolved_config.yaml
metrics.jsonl
queue_screen.log
checkpoints/ckpt_last.pt
evals/attention_diagnostics.jsonl when applicable
promotion_report.json
```

A mechanism probe directory usually contains:

```text
activation_summary.json
intervention_summary.json
probe_report.md
```

A run is not evidence until the relevant train, eval, summarize, verify, and mechanism-probe commands have actually passed.

## Testing and quality checks

Before committing source or documentation changes, run the normal local QC set:

```bash
uv sync
uv run ruff check .
uv run pytest
```

Validate registered experiments:

```bash
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
uv run scripts/validate_experiment.py --id E003_qkv_architecture_gauntlet
uv run scripts/validate_experiment.py --id E004_operator_binding_qkv_gauntlet
```

Verify default data:

```bash
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
```

Run queue doctor checks:

```bash
uv run attn-queue doctor --experiment E001_cp_trilinear_attention
uv run attn-queue doctor --experiment E002_multitrack_qkv_shift_register
uv run attn-queue doctor --experiment E003_qkv_architecture_gauntlet
uv run attn-queue doctor --experiment E004_operator_binding_qkv_gauntlet
```

Targeted tests used during the latest route-index/probe fix included:

```bash
uv run pytest tests/test_mechanism_backfill.py
uv run pytest tests/test_mechanism_probe_cli.py
uv run pytest tests/test_mechanism_capture_multi_qkv.py
uv run pytest tests/test_attention_multi_qkv_global.py
```

Use targeted tests while developing, but do not treat targeted tests as a substitute for the full QC set before committing.

Latest recorded QC in the dynamic status note:

```text
See EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md for the latest reconciled QC record.
Do not reuse older "Tier-1 verifier blocked by missing checkpoint" notes without rechecking `runs/screen`.
```

## Known limitations

* E001 and E002 completed local full runs are still local single-seed evidence unless replicated.
* E003/E004 gauntlet screen reports are advancement evidence, not full-run scientific results.
* E003/E004 rung checkpoints make post-hoc recomputation possible; they do not prove semantic mechanism roles.
* The historical `trilinear_cp` attention type remains unimplemented; use `cp_trilinear`.
* E002 softmix, warmup routing, LoRA deltas, learned routing, stochastic routing, coprime clocks, and typed streams remain unimplemented future work.
* `torch.compile` is intentionally unsupported for baseline QC.
* DDP code may exist, but the tested path is single-GPU local training.
* HellaSwag is optional bounded evaluation support, not the primary metric.
* HF export is currently an honest stub, so `lm-evaluation-harness` is deferred.
* The queue doctor is a readiness check only; it does not launch training.
* Missing historical activations cannot be reconstructed unless tensors were saved or a checkpoint can recompute them.
* Checkpoint availability means post-hoc probing is possible, not that the architecture hypothesis is supported.
* Current quick probes are activation/intervention plumbing. The Tier-1 mechanism probe suite is the statistically controlled path for E003/E004 mechanism claims.

## Current next work

The E001-E004 backfill / quick-probe artifact phase is complete. The Tier-1 mechanism-probe framework now exists for staged E003/E004 follow-up.

Initial executable Tier-1 targets:

- E003 differential_qkv_anti_value_30m_seed1_rung500
- E004 operator_valued_attention_30m_seed2_rung500

Other E001-E004 follow-ups remain useful, but they should not displace restoring/rerunning the initial Tier-1 suite targets when checkpoint-backed recomputation is needed.

## First-day checklist

Use this sequence for a new machine or a new clone:

```bash
uv sync
uv run scripts/verify_cuda.py
scripts/prepare_fineweb_edu_100m.sh
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
uv run scripts/inspect_model_config.py \
  --config configs/baseline_15m_fineweb100m_sanity.yaml
uv run scripts/train.py \
  --config configs/baseline_15m_fineweb100m_sanity.yaml \
  --overwrite
uv run scripts/verify_run.py \
  --run_dir runs/baseline_15m_fineweb100m_sanity_seed1 \
  --expect-complete-training \
  --expect-sample \
  --expect-data-manifest
uv run ruff check .
uv run pytest
```

After that passes, move to the canonical 30M baseline or a registered experiment config.
