# AGENTS.md

This file defines the operating rules for coding agents working in Attention Lab.

Attention Lab is a local GPT pretraining harness for controlled attention-architecture experiments. The standard-attention GPT path is the control. New mechanisms must be selected through config and the attention registry, trained or screened against matched controls, and interpreted only through verified artifacts.

The project exists for mechanistic interpretability research on novel architectures, not for production-efficient transformer design. Efficiency matters only insofar as a variant must be trainable, stable, and comparable enough to produce mechanisms worth interpreting.

## Agent priorities

1. Preserve the baseline harness.
2. Preserve dataset manifest discipline.
3. Preserve checkpoint, resume, and verification integrity.
4. Keep architecture variants modular, registered, and testable.
5. Keep mechanism evidence tied to real artifacts.
6. Keep scientific claims narrow, falsifiable, and artifact-backed.
7. Keep operator documentation current.
8. Avoid fake progress.

Do not optimize for looking complete. Optimize for a new user being able to reproduce setup, data verification, sanity training, experiment execution, checkpoint recomputation, evaluation, mechanism backfill, post-hoc probes, and comparison without hidden context.

## Project framing

Attention Lab is:

```text
local GPT pretraining harness
controlled attention-architecture experiment framework
screen-first candidate filter
checkpoint-backed mechanism investigation substrate
```

Attention Lab is not:

```text
chat fine-tuning stack
API eval framework
general distributed pretraining framework
production-efficient transformer proposal
benchmark-chasing leaderboard project
```

The research frame is:

```text
Use deliberately nonstandard attention architectures as interpretability instruments.
Test whether architectural decomposition changes feature separation, causal locality,
routing/content separation, operator-like behavior, or superposition structure.
```

The correct claim shape is usually:

```text
This architecture produced checkpoint-backed artifacts that make a mechanism question probeable.
```

not:

```text
This architecture is better.
```

## Start here before editing

Read these documents before changing code:

```text
README.md
EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md
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
docs/experiments/E003_qkv_architecture_gauntlet_plan.md
docs/experiments/E004_operator_binding_qkv_gauntlet_plan.md
```

For Multi-QKV implementation work, read:

```text
docs/implementation/0901_multiqkv_shift_register/
```

Before summarizing, updating, or claiming anything about experiment status, read:

```text
EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md
reports/mechanisms/cross_experiment_candidate_report.md
```

If README, reports, generated inventories, queue state, and local run directories disagree, reconcile the discrepancy in `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md` before making claims.

## Repository boundaries

Use the existing harness.

Do not rewrite the trainer, replace the config system, introduce a parallel experiment framework, or add broad abstractions unless explicitly requested.

Important directories:

```text
configs/                              Baseline and experiment configs
configs/experiments/                  Registered experiment configs
configs/mechanisms/                   Mechanism-probe helper configs and prompts
data/                                 Local datasets and manifests
src/attention_lab/models/attention/   Attention implementations
src/attention_lab/training/           Training, config, checkpointing, verification
src/attention_lab/evals/              Evaluation code
src/attention_lab/queue/              Queue and run orchestration layer
src/attention_lab/mechanisms/         Hook sites, capture, interventions, probes, backfill
docs/                                 Plans, contracts, guides
reports/                              Reports, schemas, backfill inventories, probe outputs
runs/                                 Generated training outputs
runs/screen/                          Gauntlet screen/rung artifacts
tests/                                Test suite
```

Generated runtime artifacts are not source. Do not commit `.npy` token shards, checkpoints, full run directories, queue databases, HellaSwag cache files, W&B directories, or transient logs unless an explicit small report artifact is meant to be versioned.

Versioned report artifacts must be clearly derived from real commands and should not contain hand-written fake metrics.

## Dependency and environment rules

Use `uv` only for dependency operations and commands:

```bash
uv sync
uv run <command>
```

Do not introduce parallel dependency workflows:

```text
pip install
Conda
Poetry
manual virtualenv setup
shell-specific environment hacks
```

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

Do not waive manifest checks to make a run pass.

A run with a data-manifest mismatch is not acceptable evidence for architecture comparison.

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
mechanism backfill consistency checks
```

Do not edit standard attention or shared training code casually.

If a change to shared code is required, the experiment must include and run an appropriate standard or standard-refactor control before candidate differences are interpreted.

Prefer the accurately named baseline config for new 30M runs:

```text
configs/baseline_30m_fineweb100m.yaml
```

`configs/baseline_15m_fineweb100m.yaml` is a historical name for the same 30M-ish shape.

The true smaller tier is:

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
5. Add hook-site specs for mechanism capture when the architecture exposes new internal components.
6. Keep trainer changes minimal and mechanism-agnostic.
7. Keep the standard-attention path passing unchanged tests.
8. Add or update experiment configs under `configs/experiments/<EXPERIMENT_ID>/`.
9. Add or update hypothesis and plan docs under `docs/experiments/`.
10. Update README and experiment status docs if commands, artifacts, or interpretation boundaries change.

Do not implement new mechanisms by adding conditionals throughout the trainer when the registry/module boundary can handle the change.

## Mechanism investigation rules

Mechanism code lives under:

```text
src/attention_lab/mechanisms/
```

This layer exists to support real post-hoc investigation from checkpoints. It should not fake or infer missing tensors.

Mechanism artifacts include:

```text
reports/mechanisms/backfill/
reports/mechanisms/cross_experiment_candidate_report.md
reports/mechanisms/probes/<probe_name>/
```

Use mechanism backfill to identify what is actually probeable:

```bash
uv run scripts/backfill_mechanism_inventory.py \
  --experiments E004 E003 E002 E001 \
  --repo-root . \
  --output-root reports/mechanisms/backfill

uv run scripts/compare_mechanism_candidates.py \
  --backfill-root reports/mechanisms/backfill \
  --output reports/mechanisms/cross_experiment_candidate_report.md
```

Backfill evidence levels mean:

```text
artifact_summary       metadata recoverable from existing configs/reports/summaries/diagnostics
checkpoint_recompute   checkpoint exists, so activations/interventions can be recomputed
not_available          checkpoint or required evidence is absent
```

Checkpoint availability means only this:

```text
post-hoc recomputation is possible
```

It does not mean the mechanism hypothesis is supported.

### Backfill filesystem grounding

Backfill must only mark a checkpoint as available if the checkpoint exists in the current working copy.

Use this consistency check after regenerating inventories:

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

Do not let promotion JSON alone imply local checkpoint availability.

### Probe rules

Run post-hoc probes only against existing checkpoints.

A mechanism probe directory normally contains:

```text
activation_summary.json
intervention_summary.json
probe_report.md
```

Use `--sites` for capture sites and `--intervention-sites` when only some captured sites should be edited.

Discrete route/index sites are not ordinary continuous activations. For Multi-QKV:

```text
selected_track = discrete diagnostic route/index site
track_q        = continuous intervention site
track_k        = continuous intervention site
track_v        = continuous intervention site
track_out      = continuous intervention site
```

Do not apply zero/scale interventions to `selected_track`. Capture it, but intervene only on continuous track tensors unless a future validated route-replacement intervention is explicitly implemented.

Correct E002 position-rotation pattern:

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

For E004 operator-valued rung500, treat this as canonical if present:

```text
reports/mechanisms/probes/E004_operator_valued_rung500_inventory_path/
```

The older probe folder:

```text
reports/mechanisms/probes/E004_operator_valued_rung500/
```

may be retained as historical/partial evidence if it points to the same checkpoint, but the `_inventory_path` probe supersedes it for current analysis.

### Tier-1 mechanism probe suite rules

The Tier-1 mechanism probe suite is implemented under:

```text
src/attention_lab/mechanisms/
scripts/run_mechanism_probe_suite.py
scripts/summarize_mechanism_probe_suite.py
scripts/verify_tier1_mechanism_probe_suite.py
docs/mechanism_probe_framework.md
```

It is scoped to statistically controlled post-hoc mechanism checks for promoted E003/E004 candidates, not generic probing infrastructure.

Committed Tier-1 hypothesis docs live under:

```text
docs/mechanisms/hypotheses/E003_differential_negation_tier1.yaml
docs/mechanisms/hypotheses/E004_operator_valued_negation_tier1.yaml
```

Committed deterministic Tier-1 task suites live under:

```text
configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml
configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml
```

Regenerate or validate them with:

```text
scripts/generate_tier1_mechanism_tasks.py
```

Confirmatory Tier-1 task suites must carry deterministic provenance plus `metadata.content_sha256`. The fingerprint is an integrity check, not standalone provenance. Confirmatory suite execution and the validate-only path for the built-in Tier-1 generator regenerate from metadata and reject files whose records do not match deterministic generator output. Future generators need an equivalent regeneration validator before supporting confirmatory claims.

Executable Tier-1 presets are:

```text
E003 differential -> standard_refactor_control_30m_seed1_rung500
E004 operator-valued -> standard_refactor_control_30m_seed2_rung500
```

Do not pair E004 against the E003 seed1 control.

The mechanism-probe claim ladder is scoped to probe outputs:

```text
insufficient_evidence
exploratory_probe_signal
controlled_probe_signal
candidate_mechanism_evidence
```

Do not confuse `candidate_mechanism_evidence` with the broader project status vocabulary. It means single-seed, checkpoint-backed, statistically controlled mechanism evidence, not replication.

`exploratory_probe_signal` is an exploratory status, not a passed confirmatory claim gate. Machine-readable outputs include `exploratory_signal`, `controlled_probe_gate_passed`, `candidate_mechanism_gate_passed`, `highest_status`, `status_kind`, and the compatibility `claim_gate_passed` field. Treat `claim_gate_passed` as an alias for `candidate_mechanism_gate_passed`; controlled-probe evidence has its own boolean and does not imply causal mechanism evidence.

Confirmatory Tier-1 runs require:

```text
docs/mechanisms/hypotheses/<name>.yaml
minimum 50 contrast pairs per family
grouped split discipline
shuffled-label null
random-site null when feasible for that site-layer cell
matched control evidence
bootstrap CIs
FDR-BH over every computed site x layer x task_family x metric cell
target-vs-decoy specificity
valid patch/restoration and mediation metrics for candidate_mechanism_evidence
task-aligned feature pooling for candidate_mechanism_evidence
valid clean/corrupt restoration token alignment metadata for full patching
```

Non-exploratory `--probe-only` is not confirmatory. Cheap probe-only staging must use `--exploratory --probe-only` and cannot reach `candidate_mechanism_evidence`.

Random-site null unavailability caps only the affected `(site x layer)` cell, not the whole run. Missing matched controls, noncanonical/seed-mismatched controls, missing decoys, invalid denominators, and exploratory/probe-only mode must cap claims honestly.

Random-site null pools are the complete preset-declared Tier-1 null family, not an unrestricted hook sweep and not proof that every plausible null site was tested. The suite must still inspect actual captured shapes and tensor kinds before selecting a random-site null.

Confirmatory `--sites` values must be declared in the Tier-1 preset. Unknown exploratory sites require explicit `--site-spec-file` metadata and remain noncanonical. Do not invent tensor kinds, control sites, or full-layer comparators for unknown sites.

Full-run restoration must patch only validated aligned token positions. Do not patch whole clean sequence caches into corrupted prompts when token lengths differ. Mean-sequence feature pooling is exploratory/diagnostic for Tier-1; confirmatory candidate evidence requires `answer_position` or `patch_positions_mean` pooling.

For E004 Tier-1, `operator_probs` is a low-dimensional probability site and is capture/probe-only for patching until a validated probability-site intervention exists. Do not report continuous patch/restoration or mediation as valid for `operator_probs`; full-width operator output sites remain patchable when their other gates are valid.

Use `scripts/verify_tier1_mechanism_probe_suite.py --preflight-only` to check local checkpoint availability and input validity without fabricating artifacts. Use `scripts/summarize_mechanism_probe_suite.py --validate` to validate produced `metrics.json`, `claim_gates.json`, and `summary.md`.

Do not make Tier-2/Tier-3 presets executable, do not build SAE purity infrastructure in Tier-1, and do not handwrite fake `metrics.json`, `claim_gates.json`, or `summary.md` artifacts.

## Experiment rules

Experiment configs belong under:

```text
configs/experiments/<EXPERIMENT_ID>/
```

Experiment reports belong under:

```text
reports/experiments/<EXPERIMENT_ID>/
```

Full experiment run artifacts belong under:

```text
runs/experiments/<EXPERIMENT_ID>/
```

Gauntlet screen/rung artifacts may belong under:

```text
runs/screen/
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

Do not interpret mechanism differences unless the matched control and candidate have comparable capture/probe artifacts.

## Evidence and claim rules

Full-run evidence must come from actual artifacts produced by real commands.

A run is not complete merely because a config exists, a script exists, or a report template exists.

Do not claim a full run completed unless the relevant final `verify_run.py` command passed with the required flags.

Acceptable evidence artifacts normally include:

```text
config.yaml
config_source.txt
data_manifest.json
data_manifest.sha256
environment.txt
git_commit.txt
metrics.jsonl
checkpoints/ckpt_last.pt
evals/val_loss.json
evals/hellaswag.json when required
evals/run_summary.json
evals/attention_diagnostics.jsonl for non-standard mechanisms where required
evals/qkv_track_destructive_test.json for Multi-QKV route evidence where required
```

Acceptable mechanism evidence artifacts include:

```text
reports/mechanisms/backfill/<EXPERIMENT_ID>/inventory.json
reports/mechanisms/backfill/<EXPERIMENT_ID>/inventory.md
reports/mechanisms/backfill/<EXPERIMENT_ID>/candidate_matrix.csv
reports/mechanisms/backfill/<EXPERIMENT_ID>/missing_artifacts.md
reports/mechanisms/cross_experiment_candidate_report.md
reports/mechanisms/probes/<probe_name>/activation_summary.json
reports/mechanisms/probes/<probe_name>/intervention_summary.json
reports/mechanisms/probes/<probe_name>/probe_report.md
```

Use statuses honestly:

```text
planned
implemented_not_run
screened_mechanism_active
checkpoint_recompute
artifact_summary
full_run_verified
candidate_evidence
insufficient_evidence
diagnostic_rescue
profiling_redesign
killed
not_available
```

Do not turn `implemented_not_run` into `candidate_evidence`.

Do not call missing diagnostics a pass.

Do not handwrite fake run artifacts.

Do not handwrite fake mechanism artifacts.

## E001 rules

E001 is:

```text
E001_cp_trilinear_attention
```

The canonical CP attention type is:

```text
cp_trilinear
```

The historical placeholder remains intentionally unimplemented:

```text
trilinear_cp
```

Do not use `trilinear_cp` for E001 evidence.

Current local evidence summarized in `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md` says E001 has checkpoint-backed completed local full runs for:

```text
standard_30m_seed1
cp_bilinear_r8_30m_seed1
cp_trilinear_r8_30m_seed1
```

and unavailable/incomplete status for:

```text
cp_trilinear_r8_lambda0_30m_seed1
standard_refactor_control_30m_seed1
```

CP candidates must emit attention diagnostics where required.

CP mechanism checks should establish nonzero CP activity, including meaningful CP gradient diagnostics, before loss differences are interpreted.

The current E001 claim boundary is:

```text
single-seed local signal
not architecture superiority
requires replication, lambda-zero control, and mechanism follow-up
```

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

Current local evidence summarized in `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md` says these four have local checkpoints under:

```text
runs/experiments/E002_multitrack_qkv_shift_register/
```

Do not treat old skeleton configs marked:

```text
status: experimental_unimplemented
```

as runnable evidence.

The first-build Multi-QKV interpretation is limited to globally shared, hard-switched bundled Q/K/V banks.

Do not add these unless explicitly requested as a new experiment:

```text
learned routing
softmix routing
stochastic clocks
warmup schedules
LoRA deltas
typed streams
coprime Q/K/V clocks
```

Multi-QKV candidates require mechanism diagnostics.

Route behavior must be demonstrated by diagnostics and, after checkpointed runs exist, destructive route tests or post-hoc route/track intervention probes.

Preserve the route-index semantics:

```text
selected_track is discrete/capture-only
track_q, track_k, track_v, track_out are continuous/intervention-capable
```

## E003 rules

E003 is:

```text
E003_qkv_architecture_gauntlet
```

The first implemented E003 attention types are:

```text
differential_qkv_anti_value
scope_gated_qkv
standard_refactor_control
```

E003 is not an efficiency experiment.

Interpret it only as a screen-first architecture gauntlet for feature separation and mechanistic legibility.

The gauntlet may advance candidates through:

```text
rung020
rung150
rung500
```

only from structured promotion reports, metrics, checkpoints, and nondegenerate diagnostics.

Current local evidence summarized in `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md` says E003 rung checkpoints exist under `runs/screen` for differential, scope-gated, and standard controls.

Current canonical E003 post-hoc probes include:

```text
reports/mechanisms/probes/E003_differential_rung500_inventory_path/
reports/mechanisms/probes/E003_scope_gated_rung500_inventory_path/
```

Do not treat generated rung configs, queue readiness, or gauntlet report existence as full-run evidence.

Full 3000-step E003 runs still require clean promotion evidence and approval through the queue full-run gate.

Required E003 mechanism checks:

```text
differential_qkv_activity
scope_gated_qkv_activity
```

Current E003 mechanism question:

```text
Do branch_delta, positive/negative streams, scope_out, gate, and content_scope_product
provide more local causal handles than standard attention?
```

## E004 rules

E004 is:

```text
E004_operator_binding_qkv_gauntlet
```

The first implemented E004 attention types are:

```text
operator_valued_attention
q3k3v3_role_routed_attention
dynamic_value_query_conditioned_attention
standard_refactor_control
```

E004 is not an efficiency experiment and not a model-improvement claim.

Interpret it only as a screen-first gauntlet for operator-like write modes, role-routed Q/K/V streams, and dynamic value read-mode probes.

The gauntlet may advance candidates through:

```text
rung020
rung150
rung500
```

only from structured promotion reports, metrics, checkpoints, and nondegenerate diagnostics.

Current local evidence summarized in `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md` says E004 rung checkpoints exist under `runs/screen` for:

```text
operator_valued_attention
dynamic_value_query_conditioned_attention
q3k3v3_role_routed_attention rung020
standard_refactor_control
```

Current canonical E004 operator-valued probe:

```text
reports/mechanisms/probes/E004_operator_valued_rung500_inventory_path/
```

Do not treat generated rung configs, queue readiness, or gauntlet report existence as full-run evidence.

Full 3000-step E004 runs still require clean promotion evidence and approval through the queue full-run gate.

Required E004 mechanism checks:

```text
operator_valued_activity
q3k3v3_role_activity
dynamic_value_activity
```

Current E004 interpretation boundaries:

```text
operator_valued_attention_30m_seed2_rung500 -> strongest current E004 mechanism candidate
dynamic_value_query_conditioned_attention_30m_seed2_rung500 -> diagnostic rescue
q3k3v3_role_routed_attention_30m_seed2_rung020 -> profiling/redesign
```

Current E004 mechanism question:

```text
Do operator_probs, add/suppress/gate/transform/bind outputs, and combined operator output
provide more local causal handles than standard attention or E003 stream decompositions?
```

## Queue rules

The queue is a thin serial orchestration layer over the existing harness. It must not replace training, verification, eval, or reporting contracts.

Read before queue work:

```text
docs/guides/experiment_queue_discipline_checklist.md
```

Queue safety requirements:

* Full runs require a clean promotion report plus explicit approval through `uv run attn-queue approve <run>`.
* Do not convert screen candidates into full runs without promotion reports.
* Do not approve full runs by setting `full_run_approved` directly.
* Do not recommend `run_all_full.sh` or `run_all_full_initial.sh` for exploration.
* Existing run directories are protected by default.
* Do not set `queue.allow_overwrite_existing_run_dir: true` casually.
* Non-standard full runs require a passed `queue.requires_run` control unless `queue.skip_control_check: true` is explicitly documented.
* Non-standard screen promotion requires nondegenerate mechanism diagnostics.
* `queue.allow_missing_diagnostics: true` is an auditable exception that produces `needs_investigation`, not a clean approval.
* Do not interpret validation loss without required diagnostics.
* Do not treat queue readiness or script existence as evidence.
* The queue daemon is single-GPU and serial.
* Do not make the queue concurrent without a new design and tests.
* The queue doctor is a readiness check; it does not launch training.

Useful commands:

```bash
uv run attn-queue status
uv run attn-queue ls
uv run attn-queue show <run_id_or_name>
uv run attn-queue promotion-report <run_id_or_name>
uv run attn-queue approve <run_id_or_name>
uv run attn-queue unapprove <run_id_or_name>
uv run attn-queue doctor --experiment <EXPERIMENT_ID>
uv run attn-queue leaderboard --min-stage FULL --sort loss
uv run attn-queue export-report --experiment <EXPERIMENT_ID>
uv run attn-queue morning-note --experiment <EXPERIMENT_ID> --shows "..." --not-shows "..." --next "..."
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

Do not pass `--allow-full` unless intentionally approving long full runs.

## Documentation rules

Keep `README.md` focused on onboarding and operator workflows. It should answer:

```text
What is this repo?
What is committed and what must be generated locally?
How do I install dependencies?
How do I verify CUDA?
How do I prepare or recover datasets?
How do I run the first sanity job?
How do I run and verify a real baseline?
How are experiments organized?
How do I queue, compare, probe, and interpret runs?
```

Keep `AGENTS.md` focused on constraints for coding agents. It should prevent unsafe edits, false claims, bad experiment hygiene, and stale documentation.

Keep `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md` focused on dynamic experiment state and interpretation boundaries.

When code behavior changes, update docs in the same change set.

When experiment artifacts change, update status and generated reports in the same change set.

## Required QC before committing

Normally run:

```bash
uv sync
uv run ruff check .
uv run pytest
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
uv run scripts/validate_experiment.py --id E003_qkv_architecture_gauntlet
uv run scripts/validate_experiment.py --id E004_operator_binding_qkv_gauntlet
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
uv run attn-queue doctor --experiment E001_cp_trilinear_attention
uv run attn-queue doctor --experiment E002_multitrack_qkv_shift_register
uv run attn-queue doctor --experiment E003_qkv_architecture_gauntlet
uv run attn-queue doctor --experiment E004_operator_binding_qkv_gauntlet
```

For targeted implementation work, run the relevant targeted tests first, then the full QC set before commit.

Examples of targeted tests for mechanism/probe work:

```bash
uv run pytest tests/test_mechanism_backfill.py
uv run pytest tests/test_mechanism_probe_cli.py
uv run pytest tests/test_mechanism_capture_multi_qkv.py
uv run pytest tests/test_attention_multi_qkv_global.py
```

If data is unavailable in the current environment, state that clearly and still run all non-data QC that is possible.

Do not pretend data verification passed.

## Prohibited shortcuts

Do not:

* Commit generated `.npy` shards or checkpoints.
* Handwrite metrics, summaries, eval artifacts, or mechanism artifacts to simulate a run.
* Remove manifest verification because data is inconvenient.
* Use a candidate run as its own control.
* Compare runs with different data manifests as architecture evidence.
* Treat queue readiness as training evidence.
* Treat a 20-step or 150-step screen as a full run.
* Treat checkpoint availability as mechanism evidence.
* Treat activation presence as causal evidence.
* Treat missing diagnostics as a pass.
* Add broad framework abstractions without a concrete experiment need.
* Hide an architecture change inside shared trainer plumbing.
* Leave README, AGENTS, plans, status notes, or reports stale after changing commands or behavior.
* Claim scientific results without verified artifacts and controls.
* Run long/full jobs casually.
* Pass `--allow-full` casually.
* Set overwrite flags casually.

## Commit hygiene

Before presenting work as done:

1. Show what files changed.
2. Show what commands were run.
3. Show which checks passed or failed.
4. State whether data-dependent checks were skipped because data was unavailable.
5. State whether any training, screen, full run, or post-hoc probe was actually executed.
6. State which artifact paths were generated or modified.
7. State whether generated reports were regenerated from current artifacts.
8. State whether any old artifacts are retained as historical/partial evidence.
9. Do not claim scientific results unless verified artifacts support them.

## Current status anchor

The latest known committed status anchor is:

```text
8aa6add8b77a2c376c2194801a46d7e572dbd0ce
```

That commit is `Update mechanism backfill inventory commit hashes`. Treat later local state as unknown until inspected.

Always re-check local state before claiming current status:

```bash
find runs -path '*/checkpoints/ckpt_last.pt' -print | sort

jq -r '
  .candidates[]
  | select(.checkpoint_status=="available")
  | [.experiment_id, .run_name, .checkpoint_path]
  | @tsv
' reports/mechanisms/backfill/*/inventory.json | sort

find reports/mechanisms/probes -maxdepth 2 -type f | sort

sed -n '1,260p' reports/mechanisms/cross_experiment_candidate_report.md
```
