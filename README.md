# Attention Lab

Attention Lab is a small GPT pretraining harness for local attention architecture
experiments. The standard-attention baseline is the control path; new mechanisms live
behind the attention registry and must be evaluated through manifest-checked,
verifier-checked runs.

## 1. Baseline Setup

Use `uv` only:

```bash
uv sync
uv run scripts/verify_cuda.py
```

Prepare and verify FineWeb-Edu 100M/4M shards:

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

## 2. Verified Baseline Result

The completed historical baseline run is:

```text
runs/baseline_15m_fineweb100m_seed1
```

The name is historical. The actual model size is about 30M parameters:

```text
parameters_excluding_positional: 29938560
parameters_including_positional: 30331776
```

Use the accurate-size config for new baseline runs:

```text
configs/baseline_30m_fineweb100m.yaml
```

Completed baseline result:

```text
final_val_loss: 4.081209182739258
best_val_loss: 4.081209182739258
final_val_perplexity: 59.2170307875361
median_tokens_per_sec: 107022.7422894312
peak_vram_allocated_mb: 3240.92431640625
bounded_hellaswag_accuracy_norm: 0.34
```

The run processed `3000 * 262144 = 786432000` token positions, meaning multiple
passes over the 100M-token train shard, not a unique 786M-token corpus.

## 3. Experiment Organization

Experiments are registered in:

```text
docs/experiments/experiments.yaml
```

The first architecture experiment is:

```text
E001_cp_trilinear_attention
```

Directory convention:

```text
configs/experiments/<EXPERIMENT_ID>/
docs/experiments/<EXPERIMENT_ID>_plan.md
runs/experiments/<EXPERIMENT_ID>/
reports/experiments/<EXPERIMENT_ID>/
```

List and validate experiments:

```bash
uv run scripts/list_experiments.py
uv run scripts/list_experiments.py --id E001_cp_trilinear_attention
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
```

## 4. Architecture Modules

Attention implementations live under:

```text
src/attention_lab/models/attention/
  standard.py
  cp_bilinear.py
  cp_trilinear.py
  multi_qkv_common.py
  multi_qkv_static.py
  multi_qkv_train_rotation.py
  multi_qkv_position_rotation.py
  trilinear_cp.py
  registry.py
```

Compatibility shims preserve old import paths:

```text
src/attention_lab/models/attention_standard.py
src/attention_lab/models/attention_trilinear_cp.py
src/attention_lab/models/attention_registry.py
```

Implemented canonical attention types:

```yaml
model:
  attention_type: standard
```

```yaml
model:
  attention_type: cp_bilinear
  cp_rank: 8
  cp_lambda_init: 0.0
  cp_lambda_trainable: true
  cp_lambda_fixed: false
```

```yaml
model:
  attention_type: cp_trilinear
  cp_rank: 8
  cp_lambda_init: 0.0
  cp_lambda_trainable: true
  cp_lambda_fixed: false
```

The historical `trilinear_cp` placeholder remains intentionally unimplemented. Use
`cp_trilinear` for E001.

E002 first-build Multi-QKV attention types are implemented as globally shared,
hard-switched bundled Q/K/V banks:

```yaml
model:
  attention_type: multi_qkv_static_3track_global
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_mod
```

```yaml
model:
  attention_type: multi_qkv_train_rotation_3track_global
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_plus_step_train_layer_eval
```

```yaml
model:
  attention_type: multi_qkv_position_rotation_3track_global
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_plus_position
```

The 0901 implementation contract is documented in:

```text
docs/implementation/0901_multiqkv_shift_register/
```

## 5. Add A New Attention Variant

Follow:

```text
docs/architecture_variant_checklist.md
docs/architecture_experiment_contract.md
docs/pre_experiment_cleanup_checklist.md
docs/guides/experiment_queue_discipline_checklist.md
```

Rules:

- Do not edit baseline files unless the experiment explicitly includes a
  standard-refactor control.
- Add new modules under `src/attention_lab/models/attention/`.
- Add registry and config validation support.
- Add tests for shape, causal masking, gradient flow, parameter count, and diagnostics.
- Keep dataset manifest, token budget, optimizer, LR schedule, seed, and eval cadence
  fixed for direct comparisons.
- Before queueing future full runs, write a hypothesis document and pass the queue
  discipline checklist in `docs/guides/experiment_queue_discipline_checklist.md`.

## 6. Validate E001

```bash
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/inspect_model_config.py \
  --config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
uv run scripts/inspect_model_config.py \
  --config configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1.yaml \
  --baseline-config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
```

The validator checks config parseability, unique run directories, experiment run-dir
containment, fixed baseline fields, dataset manifest, and local historical baseline
summary when present.

## 7. Manual Full-Run Execution

The E001 3000-step full runs are intentionally prepared but not executed by the
implementation pass. Run them manually when ready:

```bash
scripts/experiments/E001_cp_trilinear_attention/run_full_standard_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_bilinear_r8_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_trilinear_r8_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_trilinear_r8_lambda0_30m.sh
```

Or run the full ordered set, stopping on first failure:

```bash
scripts/experiments/E001_cp_trilinear_attention/run_all_full.sh
```

Each full-run script performs:

```text
verify_data with manifest hashes
train.py --overwrite
verify_run with complete-training/sample/data-manifest checks
eval_loss.py
eval_generate.py
eval_hellaswag.py --max_examples 100
summarize_run.py
final verify_run with eval-loss/HellaSwag/data-manifest checks
```

After all required summaries exist, compare the completed full runs:

```bash
scripts/experiments/E001_cp_trilinear_attention/compare_full_runs.sh
```

That script writes comparison JSON files under:

```text
reports/experiments/E001_cp_trilinear_attention/
```

## 8. Compare Runs Manually

For individual comparisons:

```bash
uv run scripts/compare_runs.py \
  --experiment E001_cp_trilinear_attention \
  --baseline runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1 \
  --candidate runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1 \
  --json-out reports/experiments/E001_cp_trilinear_attention/comparison_cp_trilinear_r8_vs_standard.json
```

Comparisons read `evals/run_summary.json`, add experiment metadata, require candidate
runs to live under the experiment run directory, and report deltas/ratios for loss,
perplexity, throughput, and VRAM when numeric values are available.

CP runs emit diagnostics to:

```text
runs/experiments/E001_cp_trilinear_attention/<run_name>/evals/attention_diagnostics.jsonl
```

Schema:

```text
reports/schema/attention_diagnostics.schema.json
```

## 9. Experiment Queue

The queue layer is a thin serial orchestration layer over the existing harness. It does
not decide what to run; it screens, runs, verifies, logs, and records the configs you
add.

Initialize by adding configs:

```bash
uv run attn-queue add configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
uv run attn-queue status
uv run attn-queue ls
uv run attn-queue show standard_30m_seed1
uv run attn-queue note standard_30m_seed1 "SHOWS: pending screen"
```

Full runs require explicit approval after screening:

```bash
uv run attn-queue approve standard_30m_seed1
uv run attn-queue unapprove standard_30m_seed1
```

The approval state is stored in the SQLite ledger. Configs may set
`queue.full_run_approved: false` or `true` as their initial enqueue state, but the
operator CLI is the normal way to approve or unapprove a queued row.

Launch or stop the daemon:

```bash
uv run attn-queue start
uv run attn-queue stop
```

The daemon is single-GPU and serial. It scans `queue/inbox/`, screens configs, then runs
eligible full configs through the same manifest-checked train/eval/verify pipeline used
by the manual full-run scripts.

Queue safety rules:

- FULL rows do not execute until `queue.full_run_approved` is true in the ledger.
- Existing run artifacts are not overwritten unless
  `queue.allow_overwrite_existing_run_dir: true` is explicit.
- Non-standard FULL rows need `queue.requires_run` pointing at a passed control, unless
  `queue.skip_control_check: true` is explicitly set and recorded as a warning.
- Non-standard screen promotion requires mechanism diagnostics unless
  `queue.allow_missing_diagnostics: true` is explicit.
- The 150-step screener injects `diagnostics.attention_diagnostics_every: 50` for
  non-standard attention, or lowers an existing larger cadence to 50, so mechanism
  diagnostics can exist during screening.
- `queue.mechanism_check` currently supports `cp_gradient_norm` and
  `qkv_track_activity`.

Operator review commands:

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

`export-report` writes `run_index.json` and `run_index.md` under the experiment report
directory, including approval, overwrite, control, and mechanism-check fields.
`morning-note` appends to `decision_log.md`.

E001 hypothesis templates already exist under:

```text
docs/experiments/E001_cp_trilinear_attention/hypothesis_<run_name>.md
```

They are gates for later approval, not evidence that full runs completed.

## 10. E002 Multi-QKV First Build

The E002 first-build architecture code and canonical configs are implemented, but no
full 3000-step E002 result has been run or claimed in this implementation pass:

```text
E002_multitrack_qkv_shift_register
```

Validate it with:

```bash
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
```

Canonical runnable first-build configs:

```text
configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

Manual full-run scripts:

```bash
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_standard_refactor_control.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_static_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_train_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_position_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/compare_initial_full_runs.sh
```

Old E002 skeleton configs remain `status: experimental_unimplemented` and are not first-build evidence.

Before queueing long runs, follow:

```text
docs/guides/experiment_queue_discipline_checklist.md
```

## 11. Known Limitations

- Full 3000-step E001 runs are prepared but not executed in the implementation pass.
- Full 3000-step E002 runs are prepared but not executed in the implementation pass.
- E002 first-build A/B/C code exists, but no verified E002 full-run result exists yet.
- E002 softmix, warmup routing, LoRA deltas, learned routing, stochastic routing,
  coprime clocks, and typed streams remain unimplemented future work.
- Queue `doctor` is a readiness check only; it does not launch training.
- The historical `trilinear_cp` attention type remains unimplemented; use canonical
  `cp_trilinear`.
- `torch.compile` is intentionally unsupported for baseline QC.
- DDP exists but is not the tested path.
- OpenAI Evals is not used for this training baseline.
- HellaSwag is optional bounded smoke/eval support, not the primary metric.
- HF export is still an honest stub, so `lm-evaluation-harness` remains deferred.
