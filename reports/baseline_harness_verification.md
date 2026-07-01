# Baseline Harness Verification

Date: 2026-06-29

Git commit at sanity verification start: `f71e790381aa7982db51663639d342a8b09edb65`

Git commit at full baseline run: `0760b275d46a5c920d79761609b59600d602f6f8`

## Machine And Environment

- Platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- Python: `3.12.2`
- PyTorch: `2.11.0+cu128`
- CUDA available: `True`
- CUDA version: `12.8`
- GPU: `NVIDIA GeForce RTX 5060 Ti`
- bf16 supported: `True`

## Commands Run

```bash
uv sync
uv run pytest
uv run ruff check .
uv run scripts/verify_cuda.py
uv run scripts/prepare_fineweb_edu.py --out_dir data/fineweb_edu_100m --train_tokens 100000000 --val_tokens 4000000
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m
uv run scripts/train.py --config configs/baseline_15m_fineweb100m_sanity.yaml --overwrite
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_sanity_seed1 --expect-complete-training --expect-sample
uv run scripts/eval_loss.py --checkpoint runs/baseline_15m_fineweb100m_sanity_seed1/checkpoints/ckpt_last.pt --data_root data/fineweb_edu_100m
uv run scripts/eval_generate.py --checkpoint runs/baseline_15m_fineweb100m_sanity_seed1/checkpoints/ckpt_last.pt --prompt "The history of mathematics"
uv run scripts/eval_hellaswag.py --checkpoint runs/baseline_15m_fineweb100m_sanity_seed1/checkpoints/ckpt_last.pt --max_examples 100
uv run scripts/summarize_run.py --run_dir runs/baseline_15m_fineweb100m_sanity_seed1
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_sanity_seed1 --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag
```

The first data-prep attempt wrote both shards but exited with code `134` during Python
native-extension shutdown. `scripts/prepare_fineweb_edu.py` was patched to exit
explicitly after successful writes, and the same 100M/4M command was rerun
successfully with exit code `0`.

## CUDA Result

```text
torch: 2.11.0+cu128
cuda available: True
cuda version: 12.8
device: NVIDIA GeForce RTX 5060 Ti
bf16 supported: True
```

## FineWeb-Edu Data

Prepared and verified:

```text
data/fineweb_edu_100m/edufineweb_train_000001.npy (100000000,) uint16 0 50256
data/fineweb_edu_100m/edufineweb_val_000000.npy (4000000,) uint16 0 50256
```

Manifest written during pre-experiment hardening:

```text
data/fineweb_edu_100m/manifest.json
manifest sha256: 3302a779a89ee9f77a0c5717a963dd2744b5ee89dfef56b8c0d098cb61718f17
train shard sha256: 7bc89b5e75a6eba3e471c5434b03e98dd3be6aaa8ce043a9aae564bf51e25893
val shard sha256: efb01e4b8dad9ce4aa906ca8afbb36bd0329d4135e00741556eb4a70689f784c
```

## Test Results

```text
70 passed in 6.07s
```

## Ruff Result

```text
All checks passed!
```

Final QC after the full baseline documentation updates:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_seed1 --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag --expect-data-manifest
uv run scripts/summarize_run.py --run_dir runs/baseline_15m_fineweb100m_seed1
uv run scripts/inspect_model_config.py --config configs/baseline_30m_fineweb100m.yaml
```

All final QC commands passed.

## Sanity Run Result

Run directory:

```text
runs/baseline_15m_fineweb100m_sanity_seed1
```

Training completed for `20` steps. Summary:

```json
{
  "max_step": 20,
  "train_event_count": 20,
  "val_event_count": 3,
  "initial_val_loss": 10.908893585205078,
  "final_val_loss": 8.215901374816895,
  "best_val_loss": 8.215901374816895,
  "initial_val_perplexity": 54660.334833045774,
  "final_val_perplexity": 3699.3091863806258,
  "median_tokens_per_sec": 104956.08126630174,
  "peak_vram_mb": 3240.92431640625,
  "checkpoint_count": 1
}
```

Run verification with complete-training, sample, eval-loss, and HellaSwag expectations
passed.

## Eval Loss Result

```json
{
  "checkpoint": "runs/baseline_15m_fineweb100m_sanity_seed1/checkpoints/ckpt_last.pt",
  "split": "val",
  "steps": 5,
  "loss": 8.215901374816895,
  "perplexity": 3699.3091863806258
}
```

## Generation Result

Generation ran successfully from `ckpt_last.pt` with prompt:

```text
The history of mathematics
```

The sample was text-like but low quality and repetitive, which is expected after only
20 training steps.

## HellaSwag Bounded Result

```json
{
  "checkpoint": "runs/baseline_15m_fineweb100m_sanity_seed1/checkpoints/ckpt_last.pt",
  "split": "val",
  "num_total": 100,
  "num_correct_norm": 27,
  "accuracy_norm": 0.27
}
```

## Full Baseline Run

The full standard-attention baseline run completed on 2026-06-29. The config was not
changed from `configs/baseline_15m_fineweb100m.yaml`.

```bash
uv run scripts/train.py --config configs/baseline_15m_fineweb100m.yaml --overwrite
```

Post-run commands:

```bash
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_seed1 --expect-complete-training --expect-sample
uv run scripts/eval_loss.py --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt --data_root data/fineweb_edu_100m
uv run scripts/eval_generate.py --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt --prompt "The history of mathematics"
uv run scripts/eval_hellaswag.py --checkpoint runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt --max_examples 100
uv run scripts/summarize_run.py --run_dir runs/baseline_15m_fineweb100m_seed1
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_seed1 --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag --expect-data-manifest
```

Training completed:

```json
{
  "run_dir": "runs/baseline_15m_fineweb100m_seed1",
  "max_step": 3000,
  "train_event_count": 301,
  "val_event_count": 13,
  "initial_val_loss": 10.910149574279785,
  "final_val_loss": 4.081209182739258,
  "best_val_loss": 4.081209182739258,
  "initial_val_perplexity": 54729.03074804456,
  "final_val_perplexity": 59.2170307875361,
  "median_tokens_per_sec": 107022.7422894312,
  "peak_vram_mb": 3240.92431640625,
  "checkpoint_count": 3
}
```

The first metric timestamp was `2026-06-29T21:50:10+00:00`; the final checkpoint
event timestamp was `2026-06-29T23:58:11+00:00`, for about 2h08m wall-clock runtime.
PyTorch peak allocated VRAM was 3240.92 MB. A concurrent `nvidia-smi` sample during
training reported about 12 GB device memory in use.

Full-run verifier result:

```json
{
  "run_dir": "runs/baseline_15m_fineweb100m_seed1",
  "max_step": 3000,
  "train_event_count": 301,
  "val_event_count": 13,
  "checkpoint_event_count": 3,
  "data_manifest": true,
  "ok": true
}
```

Full-run eval loss result:

```json
{
  "checkpoint": "runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt",
  "split": "val",
  "steps": 20,
  "loss": 4.081209182739258,
  "perplexity": 59.2170307875361,
  "manifest_check": {
    "status": "matched",
    "data_manifest_sha256": "3302a779a89ee9f77a0c5717a963dd2744b5ee89dfef56b8c0d098cb61718f17"
  }
}
```

Full-run generation result:

```text
Prompt: The history of mathematics
Output file: runs/baseline_15m_fineweb100m_seed1/samples/sample_step_last.txt
```

The generated samples were text-like and topical, but still low-quality as expected
for this small baseline and token budget.

Full-run bounded HellaSwag result:

```json
{
  "checkpoint": "runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt",
  "split": "val",
  "num_total": 100,
  "num_correct_norm": 34,
  "accuracy_norm": 0.34,
  "data_path": "hellaswag/hellaswag_val.jsonl",
  "data_url": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
  "data_sha256": "0aa3b88843990f3f10a97b9575c94d7b71fb2205240ba04ae4884d9e9c992588"
}
```

## Pre-Experiment Hardening Result

Completed on 2026-06-30 before starting novel attention work.

Added accurate-size naming:

```text
configs/baseline_15m_fineweb100m.yaml   historical completed run name
configs/baseline_30m_fineweb100m.yaml   accurate-size alias for new runs
```

Added standard-attention config ladder:

```text
configs/baseline_16m_fineweb100m.yaml
configs/baseline_30m_fineweb100m.yaml
configs/baseline_70m_fineweb300m.yaml
configs/baseline_125m_fineweb1b.yaml
```

Inspected model sizes:

```text
16M tier: 16025856 excluding positional, 16288000 including positional
30M tier: 29938560 excluding positional, 30331776 including positional
70M tier: 69810688 excluding positional, 70334976 including positional
125M tier: 123587328 excluding positional, 124373760 including positional
```

Added data-manifest workflow:

```bash
uv run scripts/write_data_manifest.py --data_root data/fineweb_edu_100m --out data/fineweb_edu_100m/manifest.json
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
```

Added runtime-memory fields for new training rows:

```text
peak_vram_allocated_mb
peak_vram_reserved_mb
current_vram_allocated_mb
current_vram_reserved_mb
nvidia_smi_memory_mb
```

The historical run only has `peak_vram_mb`; summarization maps that to
`peak_vram_allocated_mb` for backward compatibility.

Added resume hardening:

```text
--overwrite and --resume are mutually exclusive
resume checks model config compatibility
resume checks B/T/total_batch_size compatibility
resume checks optimizer-relevant fields
resume checks data manifests when both run and data root have one
resume writes resume_from.txt
resume appends a resume event to metrics
```

Added comparison and export surfaces:

```text
docs/architecture_experiment_contract.md
docs/upstream_borrowing_audit.md
reports/schema/run_summary.schema.json
scripts/compare_runs.py
scripts/export_hf.py
```

`scripts/export_hf.py` is an honest nonzero stub. `lm-evaluation-harness` remains
deferred until HF export is implemented and verified.

## Pre-Experiment Cleanup Fix Pass

Completed before starting novel attention work:

```text
docs/pre_experiment_cleanup_checklist.md
```

Resolved comparison-safety gaps:

```text
verify_run.py supports --expect-data-manifest
checkpoints store data_manifest and data_manifest_sha256
eval_loss.py rejects manifest mismatches unless explicitly overridden
resume validates checkpoint-embedded manifest provenance
config validation rejects unknown run/data/train/sample keys
baseline_125m_fineweb1b.yaml is canonical; baseline_124m is a historical alias
scripts/run_full_30m_baseline.sh uses accurate-size naming
HellaSwag eval JSON records data_path, data_url, and data_sha256
```

## Final Architecture-Experiment Preparation

Completed before implementing CP-bilinear or CP-trilinear attention.

Attention module package organization:

```text
src/attention_lab/models/attention/
  __init__.py
  standard.py
  trilinear_cp.py
  registry.py
```

Compatibility shims remain at the previous import paths:

```text
src/attention_lab/models/attention_standard.py
src/attention_lab/models/attention_trilinear_cp.py
src/attention_lab/models/attention_registry.py
```

Experiment convention and E001 paths:

```text
experiment manifest: docs/experiments/experiments.yaml
E001 plan: docs/experiments/E001_cp_trilinear_attention_plan.md
E001 config dir: configs/experiments/E001_cp_trilinear_attention
E001 report dir: reports/experiments/E001_cp_trilinear_attention
E001 run dir: runs/experiments/E001_cp_trilinear_attention
```

Validation and comparison tooling:

```text
scripts/list_experiments.py
scripts/validate_experiment.py
scripts/compare_runs.py --experiment E001_cp_trilinear_attention
```

Future diagnostic and implementation guidance:

```text
reports/schema/attention_diagnostics.schema.json
docs/architecture_variant_checklist.md
```

Final QC commands:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run scripts/list_experiments.py
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/inspect_model_config.py --config configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
uv run scripts/verify_run.py --run_dir runs/baseline_15m_fineweb100m_seed1 --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag --expect-data-manifest
uv run scripts/summarize_run.py --run_dir runs/baseline_15m_fineweb100m_seed1
```

Final QC results:

```text
pytest: 70 passed in 6.07s
ruff: All checks passed!
list_experiments: E001_cp_trilinear_attention listed with status planned
validate_experiment: ok=True, config_count=5, runnable_config_count=2, unimplemented_config_count=3
inspect_model_config: 29938560 excluding positional, 30331776 including positional
verify_data: manifest verified
verify_run historical baseline: ok=True, data_manifest=True
summarize_run historical baseline: final_val_loss=4.081209182739258
```

This section records the preparation state before CP implementation. The later E001
implementation pass below supersedes the statement that CP configs were skeletons.

## E001 CP Attention Implementation Pass

Implemented after the architecture-experiment preparation pass:

```text
src/attention_lab/models/attention/cp_bilinear.py
src/attention_lab/models/attention/cp_trilinear.py
src/attention_lab/models/attention/cp_common.py
```

Canonical runnable E001 attention types:

```text
cp_bilinear
cp_trilinear
```

The historical `trilinear_cp` placeholder remains intentionally unimplemented and is
not used by E001 configs.

E001 parameter counts:

```text
standard_30m_seed1:                 29938560 excluding positional, 30331776 including positional
cp_bilinear_r8_30m_seed1:           30159750 excluding positional, delta +221190 (+0.7388%)
cp_trilinear_r8_30m_seed1:          30270342 excluding positional, delta +331782 (+1.1082%)
cp_trilinear_r8_lambda0_30m_seed1:  30270336 excluding positional, delta +331776 (+1.1082%)
```

Added implementation and manual execution artifacts:

```text
reports/experiments/E001_cp_trilinear_attention/implementation_notes.md
reports/experiments/E001_cp_trilinear_attention/results.md
reports/experiments/E001_cp_trilinear_attention/run_index.md
reports/experiments/E001_cp_trilinear_attention/run_index.json
scripts/experiments/E001_cp_trilinear_attention/run_full_standard_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_bilinear_r8_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_trilinear_r8_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_full_cp_trilinear_r8_lambda0_30m.sh
scripts/experiments/E001_cp_trilinear_attention/run_all_full.sh
scripts/experiments/E001_cp_trilinear_attention/compare_full_runs.sh
```

The full 3000-step E001 runs were intentionally not executed in this pass. Manual
execution is required before any scientific comparison claim.

E001 implementation QC results:

```text
pytest: 92 passed in 7.92s
ruff: All checks passed!
validate_experiment: ok=True, config_count=5, runnable_config_count=5, unimplemented_config_count=0
verify_data: manifest verified for data/fineweb_edu_100m/manifest.json
historical baseline verify_run: ok=True, data_manifest=True
script syntax checks: passed
```

## Known Limitations

- Full 3000-step E001 CP comparison runs are prepared but not executed in this pass.
- The historical `trilinear_cp` attention type remains unimplemented; E001 uses
  canonical `cp_trilinear`.
- `torch.compile` is rejected by config validation for baseline QC.
- DDP is present but not part of the tested baseline completion path.
- OpenAI Evals is not used for this stage.
- `lm-evaluation-harness` is deferred until HF export exists.

## Queue Hardening Completion Pass

This pass hardens the experiment queue for unattended overnight use. It does not add
new attention architectures and does not launch any full 3000-step experiment runs.

Implemented queue safety changes:

```text
full-run approval gate: queue.full_run_approved plus attn-queue approve/unapprove
run-dir clobber protection: queue.allow_overwrite_existing_run_dir defaults false
screener nonzero-exit semantics: all nonzero exits fail, even after partial progress
mechanism-check registry: cp_gradient_norm and qkv_track_activity
control dependency: non-standard FULL rows require queue.requires_run or explicit skip
leaderboard filtering/sorting: --min-stage and --sort
state transitions: processed configs leave queue/inbox
full-run artifact validation: summary/eval/checkpoint/HellaSwag required before PASSED
run-index export: attn-queue export-report
decision log: attn-queue morning-note
E002 skeleton: E002_multitrack_qkv_shift_register
screen diagnostics cadence: non-standard screens force diagnostics every 50 steps
doctor command: attn-queue doctor --experiment <ID>
queue report safety fields: approval, overwrite, requires_run, mechanism_check
E001 hypothesis templates: docs/experiments/E001_cp_trilinear_attention/hypothesis_*.md
queue dry-run test: tests/test_queue_end_to_end.py
```

No scientific claims are added by this queue pass. E001 and E002 full-run evidence must
come from actual train/eval/summarize/verify artifacts.

## E002 Initial Global Multi-QKV Implementation Preparation

This pass implements the first-build E002 architecture family without launching full
3000-step experiments.

Implemented attention types:

```text
multi_qkv_static_3track_global
multi_qkv_train_rotation_3track_global
multi_qkv_position_rotation_3track_global
```

Prepared canonical E002 configs:

```text
standard_refactor_control_30m_seed1
multi_qkv_static_3track_global_30m_seed1
multi_qkv_train_rotation_3track_global_30m_seed1
multi_qkv_position_rotation_3track_global_30m_seed1
```

Parameter counts from `inspect_model_config.py`:

```text
standard_refactor_control_30m_seed1:              29938560 excluding positional, 30331776 including positional, 3538944 attention projection
multi_qkv_static_3track_global_30m_seed1:         28611456 excluding positional, delta -1327104 (-4.4328%), 2211840 attention projection
multi_qkv_train_rotation_3track_global_30m_seed1: 28611456 excluding positional, delta -1327104 (-4.4328%), 2211840 attention projection
multi_qkv_position_rotation_3track_global_30m_seed1: 28611456 excluding positional, delta -1327104 (-4.4328%), 2211840 attention projection
```

The negative parameter delta is expected because Q/K/V banks are globally shared
across layers. B/C comparisons should primarily use the static global-bank variant as
the nearest boring-explanation control.

Prepared docs and scripts:

```text
docs/implementation/0901_multiqkv_shift_register/
docs/experiments/E002_multitrack_qkv_shift_register/
scripts/experiments/E002_multitrack_qkv_shift_register/
scripts/qkv_track_destructive_test.py
reports/experiments/E002_multitrack_qkv_shift_register/
```

No E002 full-run result, eval result, HellaSwag result, or comparison JSON is claimed
until the manual scripts are run and verified.

E002 implementation QC:

```text
uv sync: passed
uv run ruff check .: All checks passed!
uv run pytest: 162 passed, 1 skipped
uv run pytest --run-integration: 163 passed
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register: ok=True, config_count=11, runnable_config_count=5, unimplemented_config_count=6
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes: manifest verified
uv run scripts/verify_data.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml --verify_hashes: manifest verified
uv run attn-queue doctor --experiment E002_multitrack_qkv_shift_register: passed
```
