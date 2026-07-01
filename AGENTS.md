# AGENTS.md

This file defines the operating rules for coding agents working in Attention Lab.

Attention Lab is a local small-GPT pretraining harness for controlled attention-architecture experiments. The standard-attention GPT path is the control. New mechanisms must be selected through config and the attention registry, trained against matched controls, and interpreted only through verified artifacts.

## Agent priorities

1. Preserve the baseline harness.
2. Preserve dataset manifest discipline.
3. Preserve run verification and checkpoint integrity.
4. Make architecture variants modular and testable.
5. Keep claims tied to actual artifacts.
6. Keep onboarding and operator documentation accurate.

Do not optimize for looking complete. Optimize for a new user being able to reproduce setup, data verification, sanity training, full training, evaluation, and comparison without hidden context.

## Start here before editing

Read these documents before changing code:

```text
README.md
docs/architecture_experiment_contract.md
docs/architecture_variant_checklist.md
docs/pre_experiment_cleanup_checklist.md
docs/guides/experiment_queue_discipline_checklist.md
docs/experiments/experiments.yaml
```

For experiment-specific work, also read the relevant plan:

```text
docs/experiments/E001_cp_trilinear_attention_plan.md
docs/experiments/E002_multitrack_qkv_shift_register_plan.md
```

For Multi-QKV implementation work, read:

```text
docs/implementation/0901_multiqkv_shift_register/
```

## Repository boundaries

Use the existing harness. Do not replace it with a new trainer, a new config system, or a new experiment framework unless explicitly asked.

Important directories:

```text
configs/                         Baseline and experiment configs
configs/experiments/             Registered experiment configs
data/                            Local datasets and manifests
src/attention_lab/models/attention/  Attention implementations
src/attention_lab/training/      Training, config, checkpointing, verification
src/attention_lab/evals/         Evaluation code
src/attention_lab/queue/         Queue and run orchestration layer
docs/                            Plans, contracts, guides
reports/                         Reports and schemas
runs/                            Generated training outputs
tests/                           Test suite
```

Generated runtime artifacts are not source. Do not commit `.npy` token shards, checkpoints, run directories, queue databases, HellaSwag cache files, W&B directories, or transient logs unless an explicit report artifact is meant to be versioned.

## Dependency and environment rules

Use `uv` only for dependency operations and commands:

```bash
uv sync
uv run <command>
```

Do not introduce a parallel `pip install`, Conda, Poetry, or shell-specific environment workflow.

The normal first environment check is:

```bash
uv run scripts/verify_cuda.py
```

CUDA is required for real training. CPU-only behavior may be useful for unit tests but is not evidence for training results.

## Dataset rules

The default dataset root is:

```text
data/fineweb_edu_100m
```

The default prepared dataset is FineWeb-Edu with GPT-2 tokenization:

```text
train tokens: 100,000,000
validation tokens: 4,000,000
shard dtype: uint16
manifest: data/fineweb_edu_100m/manifest.json
```

The `.npy` shards are intentionally ignored by Git. A fresh clone may have a manifest but no token shards.

If data is missing, use:

```bash
scripts/prepare_fineweb_edu_100m.sh
```

If shards were manually copied or downloaded, always rebuild and verify the manifest before training:

```bash
uv run scripts/write_data_manifest.py \
  --data_root data/fineweb_edu_100m \
  --out data/fineweb_edu_100m/manifest.json

uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
```

Do not waive manifest checks to make a run pass. A run with a data-manifest mismatch is not acceptable evidence for an architecture comparison.

## Baseline rules

The standard-attention path is the control path.

Do not weaken or bypass:

```text
config validation
data manifest checks
checkpoint save/load/resume
train/eval metrics
verify_run.py
run summaries
comparison report contracts
```

Do not edit standard attention or shared training code casually. If a change to shared code is required, the experiment must include and run a standard-refactor control before candidate differences are interpreted.

Prefer the accurately named baseline config for new 30M runs:

```text
configs/baseline_30m_fineweb100m.yaml
```

Remember that `configs/baseline_15m_fineweb100m.yaml` is a historical name for the same 30M-ish shape. The true smaller tier is:

```text
configs/baseline_16m_fineweb100m.yaml
```

Use the sanity config only to test the pipeline:

```text
configs/baseline_15m_fineweb100m_sanity.yaml
```

A sanity run is not architecture evidence.

## Architecture implementation rules

New attention modules go here:

```text
src/attention_lab/models/attention/
```

Architecture selection must happen through config:

```yaml
model:
  attention_type: <registered_attention_type>
```

When adding or changing an attention mechanism:

1. Add the module under `src/attention_lab/models/attention/`.
2. Register it in the attention registry.
3. Extend config validation for any new model keys.
4. Add tests for construction, forward shape, causal masking, gradient flow, parameter count, and diagnostics.
5. Keep trainer changes minimal and mechanism-agnostic.
6. Keep the standard-attention path passing unchanged tests.
7. Add or update experiment configs under `configs/experiments/<EXPERIMENT_ID>/`.
8. Add or update hypothesis and plan docs under `docs/experiments/`.

Do not implement new mechanisms by adding conditionals throughout the trainer when the registry/module boundary can handle the change.

## Experiment rules

Experiment configs belong under:

```text
configs/experiments/<EXPERIMENT_ID>/
```

Experiment reports belong under:

```text
reports/experiments/<EXPERIMENT_ID>/
```

Experiment run artifacts belong under:

```text
runs/experiments/<EXPERIMENT_ID>/
```

Every direct comparison must hold fixed:

```text
dataset manifest
data root
tokenizer
model scale unless parameter-count difference is the explicit variable
seed
batch construction
optimizer
learning-rate schedule
training token budget
evaluation cadence
checkpoint cadence
verification path
```

Do not interpret validation-loss differences unless the matched control and candidate both passed the required train/eval/summarize/verify pipeline.

## Evidence and claim rules

Full-run evidence must come from actual artifacts produced by real commands.

A run is not complete merely because a config exists, a script exists, or a report template exists. Do not claim a full run completed unless the relevant final `verify_run.py` command passed with the required flags.

Acceptable evidence artifacts normally include:

```text
config.yaml
metrics.jsonl
checkpoints/ckpt_last.pt
evals/val_loss.json
evals/hellaswag.json when required
evals/run_summary.json
evals/attention_diagnostics.jsonl for non-standard mechanisms where required
evals/qkv_track_destructive_test.json for Multi-QKV route evidence where required
```

Use these statuses honestly:

```text
planned
implemented_not_run
screened_mechanism_active
full_run_verified
candidate_evidence
insufficient_evidence
killed
```

Do not turn `implemented_not_run` into `candidate_evidence`. Do not call missing diagnostics a pass. Do not handwrite fake run artifacts.

## E001 rules

E001 is:

```text
E001_cp_trilinear_attention
```

The canonical CP attention type is:

```text
cp_trilinear
```

The historical `trilinear_cp` placeholder remains intentionally unimplemented. Do not use it for E001 evidence.

CP candidates must emit attention diagnostics where required. CP mechanism checks should establish nonzero CP activity, including meaningful CP gradient diagnostics, before loss differences are interpreted.

## E002 rules

E002 is:

```text
E002_multitrack_qkv_shift_register
```

The canonical first-build run matrix is:

```text
standard_refactor_control_30m_seed1
multi_qkv_static_3track_global_30m_seed1
multi_qkv_train_rotation_3track_global_30m_seed1
multi_qkv_position_rotation_3track_global_30m_seed1
```

Do not treat old skeleton configs marked `status: experimental_unimplemented` as runnable evidence.

The first-build Multi-QKV interpretation is limited to globally shared, hard-switched bundled Q/K/V banks. Do not add learned routing, softmix routing, stochastic clocks, warmup schedules, LoRA deltas, typed streams, or coprime Q/K/V clocks unless explicitly requested as a new experiment.

Multi-QKV candidates require mechanism diagnostics. Route behavior must be demonstrated by diagnostics and, after checkpointed runs exist, destructive route tests.

## Queue rules

The queue is a thin serial orchestration layer over the existing harness. It must not replace the training, verification, eval, or reporting contracts.

Read before queue work:

```text
docs/guides/experiment_queue_discipline_checklist.md
```

Queue safety requirements:

- Full runs require explicit approval through the ledger, normally via `uv run attn-queue approve <run>`.
- Existing run directories are protected by default.
- Do not set `queue.allow_overwrite_existing_run_dir: true` casually.
- Non-standard full runs require a passed `queue.requires_run` control unless `queue.skip_control_check: true` is explicitly documented.
- Non-standard screen promotion requires mechanism diagnostics unless `queue.allow_missing_diagnostics: true` is explicitly documented.
- The queue daemon is single-GPU and serial. Do not make it concurrent without a new design and tests.
- The queue doctor is a readiness check; it does not launch training.

Useful commands:

```bash
uv run attn-queue status
uv run attn-queue ls
uv run attn-queue show <run_id_or_name>
uv run attn-queue approve <run_id_or_name>
uv run attn-queue unapprove <run_id_or_name>
uv run attn-queue doctor --experiment <EXPERIMENT_ID>
uv run attn-queue leaderboard --min-stage FULL --sort loss
uv run attn-queue export-report --experiment <EXPERIMENT_ID>
uv run attn-queue morning-note --experiment <EXPERIMENT_ID> --shows "..." --not-shows "..." --next "..."
```

## Documentation rules

Keep README.md focused on onboarding and operator workflows. It should answer:

- What is this repo?
- What is committed and what must be generated locally?
- How do I install dependencies?
- How do I verify CUDA?
- How do I prepare or recover datasets?
- How do I run the first sanity job?
- How do I run and verify a real baseline?
- How are experiments organized?
- How do I queue, compare, and interpret runs?

Keep AGENTS.md focused on constraints for agents. It should not be a duplicate tutorial, but it must contain enough operational rules to prevent unsafe edits and false claims.

When code behavior changes, update docs in the same change set.

## Required QC before committing

Normally run:

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

For targeted implementation work, run the relevant targeted tests first, then the full QC set before commit.

If data is unavailable in the current environment, state that clearly and still run all non-data QC that is possible. Do not pretend data verification passed.

## Prohibited shortcuts

Do not:

- Commit generated `.npy` shards or checkpoints.
- Handwrite metrics, summaries, or eval artifacts to simulate a run.
- Remove manifest verification because data is inconvenient.
- Use a candidate run as its own control.
- Compare runs with different data manifests as architecture evidence.
- Treat queue readiness as training evidence.
- Treat a 20-step or 150-step screen as a full run.
- Add broad framework abstractions without a concrete experiment need.
- Hide an architecture change inside shared trainer plumbing.
- Leave README, AGENTS, plans, or reports stale after changing commands or behavior.

## Commit hygiene

Before presenting work as done:

1. Show what files changed.
2. Show what commands were run.
3. Show which checks passed or failed.
4. State honestly whether data-dependent checks were skipped because data was unavailable.
5. State whether any full runs were actually executed.
6. Do not claim scientific results unless verified artifacts support them.


## Dynamic Experiment Status

Before summarizing, updating, or claiming anything about E001, E002, or later experiments, read:

```text
EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md
