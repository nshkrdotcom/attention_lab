# Attention Lab

Attention Lab is a local GPT pretraining harness for controlled attention-architecture experiments. It is screen-first for architecture exploration: run cheap screens, inspect stability and mechanism activity, and promote only selected candidates to manifest-checked full runs against standard-attention controls.

The repository is intentionally not a chat fine-tuning stack, API evaluation framework, or general distributed pretraining platform. Its purpose is local architecture research with reproducible data, configs, checkpoints, evaluation artifacts, and reports.

## What you can do with this repo

- Prepare a FineWeb-Edu token dataset for local pretraining.
- Train standard GPT-style causal language model baselines from scratch.
- Swap attention implementations through config rather than rewriting the trainer.
- Screen architecture candidates before spending full-run GPU time.
- Run fixed-budget promoted architecture experiments against matched controls.
- Resume and verify checkpoints.
- Evaluate validation loss, perplexity, generation samples, bounded HellaSwag, throughput, and VRAM.
- Record experiment outputs under reproducible run and report directories.
- Use the queue layer to screen candidate configs before committing GPU time to full runs.

## Current experiment status

See [`EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md`](EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md) before interpreting any E001/E002 result. That file is the dynamic status and technical discussion layer for completed, partial, stale, and not-yet-verified experiment artifacts.

## Evidence boundary

README.md explains how to use the harness. It is not the source of truth for live experiment state. If README.md, generated reports, queue indexes, and local run directories disagree, reconcile them in [`EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md`](EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md), then regenerate the reports.

Attention Lab is screen-first for architecture exploration. Most variants should die in short screens. Full 3000-step runs are promotion artifacts, not default exploration. Queue readiness is not scientific evidence. A run script existing is not approval to run it.

A screen report is not a scientific result. A full run is not evidence until train, eval, summarize, and final verify pass on actual artifacts.

## Repository map

```text
configs/                         Training and experiment configs
configs/experiments/             Registered architecture experiments
data/                            Local tokenized datasets and manifests
  fineweb_edu_100m/              Default dataset location; .npy shards are not committed
docs/                            Contracts, plans, implementation notes, and guides
reports/                         Human-readable and JSON reports; most generated reports are ignored
runs/                            Training outputs; ignored except .gitkeep
scripts/                         CLI wrappers for setup, train, eval, verify, queue, and experiments
src/attention_lab/               Library code
tests/                           Unit and integration tests
AGENTS.md                        Rules for coding agents working in this repo
```

Generated files are mostly ignored. In particular, `data/**/*.npy`, `runs/*`, queue runtime state, `hellaswag/`, and `wandb/` are not part of the committed repository.

## Requirements

- Python 3.11 or 3.12.
- `uv` for dependency management.
- NVIDIA CUDA-capable GPU for real training.
- Enough disk space for token shards, checkpoints, run artifacts, HellaSwag cache, and optional larger datasets.

The project declares CUDA PyTorch wheels through the `pytorch-cu128` index on Linux, and the package scripts include `attention-lab-train`, `attention-lab-eval-loss`, `attention-lab-eval-generate`, `attention-lab-eval-hellaswag`, `attention-lab-verify-run`, `attention-lab-summarize-run`, `attention-lab-queue`, and `attn-queue`.

## Fast start

From the repository root:

```bash
uv sync
uv run scripts/verify_cuda.py
```

`verify_cuda.py` prints the installed Torch version, CUDA availability, CUDA version, device name, and BF16 support, and exits with an error if CUDA is unavailable.

Then prepare the default FineWeb-Edu 100M train / 4M validation dataset:

```bash
scripts/prepare_fineweb_edu_100m.sh
```

That wrapper runs the full dataset path:

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

After setup, validate the baseline config and run a short sanity training job:

```bash
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
```

Use the sanity config first. It runs only 20 steps and verifies that dependencies, data loading, checkpointing, sampling, and run verification all work before you spend time on a real run.

## Dataset setup and recovery

The committed repo includes `data/fineweb_edu_100m/manifest.json` but does not include the `.npy` token shards. This is intentional: tokenized datasets are local artifacts and are ignored by Git.

A complete default dataset directory should look like this after preparation:

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

### If you manually downloaded or copied shards

Place them under the data root that the config expects. For the default 100M config, use:

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

Do not trust copied data until `verify_data.py --verify_hashes` passes against the manifest that will be used by the training run.

### Larger dataset configs

Some configs expect larger local datasets that are not committed and may not be prepared yet:

```text
configs/baseline_70m_fineweb300m.yaml  -> data/fineweb_edu_300m
configs/baseline_125m_fineweb1b.yaml   -> data/fineweb_edu_1b
configs/baseline_124m_fineweb1b.yaml   -> historical alias for the 125M config
```

Prepare those datasets with the same script, changing `--out_dir` and `--train_tokens`:

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

For 1B tokens, use `--out_dir data/fineweb_edu_1b --train_tokens 1000000000 --val_tokens 4000000`.

## Baseline configs

Recommended baseline configs:

```text
configs/baseline_15m_fineweb100m_sanity.yaml  20-step smoke/sanity run
configs/baseline_16m_fineweb100m.yaml         smaller true ~16M tier
configs/baseline_30m_fineweb100m.yaml         canonical local 30M baseline
configs/baseline_70m_fineweb300m.yaml         larger local baseline template
configs/baseline_125m_fineweb1b.yaml          125M-ish template; needs 1B-token data
```

`configs/baseline_15m_fineweb100m.yaml` is a historical name. The model shape is the same as the canonical 30M config: 6 layers, 6 heads, embedding size 384, block size 1024. Prefer `configs/baseline_30m_fineweb100m.yaml` for new 30M runs.

The completed historical baseline run, if present locally, is:

```text
runs/baseline_15m_fineweb100m_seed1
```

Its README-recorded summary is:

```text
final_val_loss: 4.081209182739258
best_val_loss: 4.081209182739258
final_val_perplexity: 59.2170307875361
median_tokens_per_sec: 107022.7422894312
peak_vram_allocated_mb: 3240.92431640625
bounded_hellaswag_accuracy_norm: 0.34
```

That run processed `3000 * 262144 = 786432000` token positions. This is multiple passes over the 100M-token training shard, not a unique 786M-token corpus.

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
reports/experiments/<EXPERIMENT_ID>/       Comparison reports and run indexes
runs/experiments/<EXPERIMENT_ID>/          Generated run artifacts
```

List and validate experiments:

```bash
uv run scripts/list_experiments.py
uv run scripts/list_experiments.py --id E001_cp_trilinear_attention
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
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
```

The historical `trilinear_cp` placeholder is intentionally unimplemented. Use `cp_trilinear` for E001.

When adding an attention variant:

1. Read `docs/architecture_experiment_contract.md`.
2. Read `docs/architecture_variant_checklist.md`.
3. Add the implementation under `src/attention_lab/models/attention/`.
4. Register it through the attention registry.
5. Add config validation support.
6. Add tests for shape, causal masking, gradient flow, parameter count, and diagnostics.
7. Keep data, seed, optimizer, learning-rate schedule, token budget, batch construction, checkpoint cadence, and eval cadence fixed for direct comparisons.
8. Add or update an experiment plan before interpreting results.

Do not rewrite the trainer just to test an architecture variant.

## E001: CP trilinear attention

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

These scripts are for frozen, promoted full runs only. They are not the default exploration path. The all-full matrix launcher refuses to run by design; use the queue screen-first workflow and approve selected full runs from promotion reports.

Screen the active CP path first:

```bash
uv run attn-queue add configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
uv run attn-queue add configs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1.yaml
uv run attn-queue add configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1.yaml
uv run attn-queue start
uv run attn-queue promotion-report cp_trilinear_r8_30m_seed1
```

Compare after the required summaries exist:

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

## E002: Multi-QKV shift register

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

These are for approved full runs after promotion. The default E002 path is to screen the standard refactor control and the static/train/position Multi-QKV variants, generate promotion reports, approve at most selected full runs, and compare only after verified full-run plus destructive-test artifacts exist.

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

Old E002 skeleton configs with `status: experimental_unimplemented` are placeholders for future work, not first-build evidence.

## Screen-first experiment workflow

1. Verify CUDA and data.
2. Validate experiment configs.
3. Add candidate configs to the queue.
4. Run screens.
5. Generate and review promotion reports.
6. Approve at most selected full runs.
7. Run promoted full runs.
8. Verify full artifacts.
9. Compare only verified full runs.

Concrete E002 example:

```bash
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
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

The approver checks the promotion report before setting a row executable as `FULL`. Non-standard candidates need non-degenerate diagnostics unless an explicit config exception records why diagnostics are absent. Multi-QKV reports include screen destructive-test evidence when feasible, or a recorded screen-level infeasibility reason.

## Experiment queue

The queue layer is a serial, single-GPU operator tool. It does not decide what is scientifically worth running. It screens candidate configs, writes promotion reports, records state in SQLite, blocks unsafe full runs, executes approved runs, and exports run indexes.

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
uv run attn-queue leaderboard --min-stage FULL --sort loss
uv run attn-queue leaderboard --sort speed
uv run attn-queue export-report --experiment E001_cp_trilinear_attention
uv run attn-queue morning-note --experiment E001_cp_trilinear_attention \
  --shows "..." \
  --not-shows "..." \
  --next "..."
```

Queue safety rules:

- Full runs require a clean promotion report and explicit approval.
- Existing run directories are protected unless `queue.allow_overwrite_existing_run_dir: true` is explicit.
- Non-standard full runs require a passed control through `queue.requires_run` unless an explicit skip is documented.
- Non-standard screen promotion requires mechanism diagnostics unless explicitly allowed.
- The 150-step screener lowers or injects `diagnostics.attention_diagnostics_every: 50` for non-standard attention so mechanism diagnostics can exist during screening.
- Screen length is configurable through `queue.screen_steps`, `queue.screen_val_every`, `queue.screen_save_every`, and `queue.screen_diagnostics_every`.
- Supported mechanism checks include `cp_gradient_norm` and `qkv_track_activity`.

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

Comparison artifacts read `evals/run_summary.json`, add experiment metadata, require candidate runs to live under the experiment run directory, and report loss, perplexity, throughput, and VRAM deltas and ratios when numeric fields are available.

For Multi-QKV candidates, comparison output also includes mechanism and destructive-test fields when the relevant artifacts exist.

## Testing and quality checks

Before committing source or documentation changes, run the normal local QC set:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
uv run attn-queue doctor --experiment E001_cp_trilinear_attention
uv run attn-queue doctor --experiment E002_multitrack_qkv_shift_register
```

Use targeted tests while developing, but do not treat targeted tests as a substitute for the full QC set before committing.

## Output artifacts

A completed run directory usually contains:

```text
config.yaml
metrics.jsonl
checkpoints/ckpt_last.pt
samples/*.txt or sample output files
evals/val_loss.json
evals/hellaswag.json
evals/run_summary.json
evals/attention_diagnostics.jsonl      for non-standard attention where applicable
evals/qkv_track_destructive_test.json  for Multi-QKV route tests where applicable
```

A run is not evidence until the relevant train, eval, summarize, and final verify commands have actually passed.

## Known limitations

- Full 3000-step E001 and E002 runs may be prepared without being executed locally; do not claim results unless verified artifacts exist.
- The historical `trilinear_cp` attention type remains unimplemented; use `cp_trilinear`.
- E002 softmix, warmup routing, LoRA deltas, learned routing, stochastic routing, coprime clocks, and typed streams remain unimplemented future work.
- `torch.compile` is intentionally unsupported for baseline QC.
- DDP code may exist, but the tested path is single-GPU local training.
- HellaSwag is optional bounded evaluation support, not the primary metric.
- HF export is currently an honest stub, so `lm-evaluation-harness` is deferred.
- The queue doctor is a readiness check only; it does not launch training.

## First-day checklist

Use this sequence for a new machine or a new clone:

```bash
uv sync
uv run scripts/verify_cuda.py
scripts/prepare_fineweb_edu_100m.sh
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
uv run scripts/inspect_model_config.py --config configs/baseline_15m_fineweb100m_sanity.yaml
uv run scripts/train.py --config configs/baseline_15m_fineweb100m_sanity.yaml --overwrite
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_sanity_seed1 --expect-complete-training --expect-sample --expect-data-manifest
uv run pytest
uv run ruff check .
```

After that passes, move to the canonical 30M baseline or a registered experiment config.
