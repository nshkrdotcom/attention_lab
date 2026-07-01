# Experiment Queue And Discipline Checklist

This guide turns the QKV brainstorm notes into an implementation and operating
checklist for Attention Lab. It is intentionally detailed because the queue system is
useful only if it preserves experiment meaning, provenance, and comparison discipline.

This document is not evidence that any queued run completed. It is the work plan and
acceptance checklist for implementing and operating the queue layer.

## How To Read The Checkboxes

The top `Implementation Status` section is the completion log for the queue system
itself. The phase checklists below are reusable acceptance and operating gates. They
are intentionally not all checked globally because many items must be re-evaluated for
each future experiment, each queued config, or each manual full-run freeze.

In short:

- Checked items near the top mean the system feature exists.
- Unchecked phase items mean "do this or verify this when using that process."
- Per-run science gates, such as hypothesis docs, one-batch overfit, destructive
  mechanism tests, controls, and morning notes, remain operator responsibilities.

## Implementation Status

Completed in this implementation pass:

- [x] Queue package added under `src/attention_lab/queue/`.
- [x] Runtime queue directories defined and `queue/inbox/.gitkeep` tracked.
- [x] Queue runtime artifacts ignored in `.gitignore`.
- [x] SQLite ledger implemented with direct `sqlite3`.
- [x] Inbox scanning and config deduplication implemented.
- [x] Strict optional `queue:` config metadata validation added.
- [x] Stage-1 screener command/config override and verdict logic implemented.
- [x] Mechanism-active check from `attention_diagnostics.jsonl` implemented.
- [x] Full-run pipeline executor implemented around existing scripts.
- [x] Full-run source-state capture writes `git_state.txt`.
- [x] Watchdog daemon loop implemented for serial screen/full execution.
- [x] CLI implemented with `attn-queue` and `attention-lab-queue` entrypoints.
- [x] ASCII leaderboard implemented.
- [x] Queue daemon script added at `scripts/queue_daemon.sh`.
- [x] Unit tests added for ledger, ingestion, screener, runner, watchdog, CLI, and
  leaderboard.
- [x] FULL-run approval gate implemented through `queue.full_run_approved` and
  `attn-queue approve` / `attn-queue unapprove`.
- [x] Run-directory clobber protection implemented through
  `queue.allow_overwrite_existing_run_dir`.
- [x] Screener nonzero-exit and missing-metrics semantics hardened.
- [x] Mechanism checks moved into `src/attention_lab/queue/mechanism_checks.py` with
  CP and future QKV-track check families.
- [x] Non-standard FULL runs require `queue.requires_run` or explicit
  `queue.skip_control_check`.
- [x] `leaderboard --min-stage` and `leaderboard --sort` implemented.
- [x] Processed queue configs move out of `queue/inbox/`.
- [x] Full-run PASSED state requires summary, eval-loss, HellaSwag, checkpoint, and
  final verifier artifacts.
- [x] Queue run-index export added via `attn-queue export-report`.
- [x] Decision-log support added via `attn-queue morning-note`.
- [x] E002 multitrack QKV shift-register skeleton added without architecture code.
- [x] Screen diagnostics cadence forced to 50 steps for non-standard 150-step screens.
- [x] Read-only readiness check added via `attn-queue doctor`.
- [x] Queue run-index exports include approval, overwrite, control, and
  mechanism-check fields.
- [x] E001 hypothesis templates added without approving any run.
- [x] End-to-end fake queue dry-run test added without launching training.
- [x] No full training runs executed by this implementation pass.

Still operator/manual by design:

- [ ] Running long full experiments from a frozen external working copy.
- [ ] Writing run-specific hypothesis documents before queueing FULL runs.
- [ ] Performing one-batch overfit and destructive mechanism-active tests when they
  are not represented as automated configs.
- [ ] Reading the leaderboard every morning and writing interpretation notes.
- [ ] Updating experiment reports with actual full-run results after verified runs.

## Source Documents Covered

External source docs read on 2026-06-29/2026-06-30:

- `0700_Experiment_Queue_System_arch.md`
- `0701_considerations.md`
- `0702_discipline_layer.md`

Requirements covered here:

- Thin orchestration layer above the existing harness.
- SQLite-backed run ledger.
- Queue filesystem layout.
- Config ingestion by content hash.
- Stage-1 150-step screen with kill criteria.
- Mechanism-active checks from attention diagnostics.
- Full-run executor wrapping existing scripts.
- Watchdog daemon with graceful shutdown.
- CLI entrypoint and command set.
- ASCII leaderboard.
- Run naming and deduplication.
- E002-and-later experiment config convention.
- Morning workflow and note-taking discipline.
- Explicit non-goals.
- Hypothesis gate before full runs.
- Fast-run ladder before serious runs.
- Failure taxonomy.
- Combinatorial discipline.
- Source-freeze/evidence boundary.
- TDD and QC expectations.

## Scope Boundary

This guide prepares the queue and discipline layer. It does not replace the existing
training harness and does not weaken any existing checks.

The queue must call the current scripts:

```text
scripts/verify_data.py
scripts/train.py
scripts/verify_run.py
scripts/eval_loss.py
scripts/eval_generate.py
scripts/eval_hellaswag.py
scripts/summarize_run.py
scripts/compare_runs.py
```

The queue must not rewrite training, checkpointing, config validation, manifest checks,
run verification, or existing experiment reports.

## No Full Runs In This Working Copy

For this documentation and checklist pass:

- Do not launch long training runs from this working copy.
- Do not create fake run summaries.
- Do not handwrite full-run eval artifacts.
- Do not mark queue-driven runs as complete unless the actual queue/full-run commands
  have completed and `verify_run.py` has passed.
- Manual source-freeze and long-run execution are handled outside this working copy.

## Phase 00 - Source Freeze And Evidence Boundary

Purpose: prevent experiment evidence from depending on a moving source tree.

Checklist:

- [ ] Decide whether the run will be launched from a clean commit, a git worktree, or a
  commit-stamped external run copy.
- [ ] Confirm the run directory records the exact git commit.
- [ ] Confirm the run directory records `git diff HEAD` at full-run start.
- [ ] Confirm queue logs warn if a full run starts with a non-empty diff.
- [ ] Confirm the warning is preserved in `queue_runner.log` and visible from the
  leaderboard or `attn-queue show`.
- [ ] Confirm source edits during long runs happen in a different working copy.
- [ ] Confirm the run artifact directory contains config, commit, diff, environment,
  data manifest, metrics, checkpoint, samples, evals, and summary.
- [ ] Confirm no documentation claims exceed verified artifacts.

Manual Full-Run Freeze Policy:

- [ ] User performs full-run freeze outside this working copy.
- [ ] Agent may prepare scripts, configs, docs, and tests.
- [ ] Agent must not claim full-run results unless the user provides artifacts or the
  agent actually runs the full command sequence.
- [ ] Any report created before full runs must say `not_run` or `prepared_not_run`.

TDD checks:

- [ ] Unit test for full-run runner source-state capture.
- [ ] Unit test for dirty-tree warning classification.
- [ ] Unit test for missing `git_state.txt` causing verifier/report warning if queue
  artifacts are later verified.

## Phase 01 - Queue Filesystem And Git Hygiene

Purpose: create the queue surface without committing transient runtime state.

Target layout:

```text
attention_lab/
  queue/
    __init__.py
    ledger.py
    runner.py
    screener.py
    leaderboard.py
    watchdog.py
    cli.py

  queue/inbox/.gitkeep
  scripts/queue_daemon.sh
  data/queue.db
```

Runtime directories:

```text
queue/inbox/
queue/active/
queue/full_pending/
queue/done/
queue/failed/
data/queue.db
data/queue.pid
```

Checklist:

- [ ] Add `queue/inbox/.gitkeep`.
- [ ] Add `queue/active/`, `queue/full_pending/`, `queue/done/`, and
  `queue/failed/` to `.gitignore`.
- [ ] Add `data/queue.db` to `.gitignore`.
- [ ] Add `data/queue.pid` to `.gitignore`.
- [ ] Add queue package under the existing source package namespace.
- [ ] Add `scripts/queue_daemon.sh`.
- [ ] Add a console entrypoint in `pyproject.toml`.
- [ ] Ensure the package remains importable through `uv run`.
- [ ] Ensure no checkpoints, logs, or queue DB files are committed.

TDD checks:

- [ ] Test queue directories are created by initialization helpers.
- [ ] Test inbox path is stable and repo-relative.
- [ ] Test `.gitignore` contains runtime queue paths.
- [ ] Test console entrypoint resolves.

## Phase 02 - SQLite Ledger

Purpose: make queue state durable, queryable, crash-resistant, and dependency-free.

Required table: `runs`.

Required columns:

```text
id
config_path
config_name
run_dir
attention_type
stage
status
failure_class
enqueued_at
started_at
finished_at
step_reached
final_val_loss
best_val_loss
final_ppl
median_tokens_per_sec
peak_vram_allocated_mb
hellaswag_acc
ablation_logit_delta
mechanism_active
full_run_approved
allow_overwrite_existing_run_dir
notes
```

Required stages:

```text
SCREEN
FULL
```

Required statuses:

```text
PENDING
RUNNING
PASSED
FAILED
KILLED
```

Required failure classes:

```text
NAN
FLAT_LOSS
DEAD_GRAD
COMPILE_ERROR
OOM
SLOW
VERIFY_FAIL
RUN_DIR_EXISTS
UNKNOWN
```

Checklist:

- [ ] Use standard-library `sqlite3`, no ORM.
- [ ] Enable WAL mode.
- [ ] Create schema idempotently.
- [ ] Use content hash as stable run id: `sha256(config_file_contents)[:12]`.
- [ ] Deduplicate identical configs by content hash.
- [ ] Store config filename stem as `config_name`.
- [ ] Store resolved `run.out_dir`.
- [ ] Store `model.attention_type`.
- [ ] Store screen and full-run metrics after summaries exist.
- [ ] Store full-run approval and run-dir overwrite policy.
- [ ] Store human-editable `notes`.
- [ ] Support special baseline calibration row `id='__baseline__'`.
- [ ] Provide CRUD helpers for enqueue, start, pass, fail, kill, requeue, note, list.
- [ ] Treat interrupted `RUNNING` rows with null `finished_at` as restartable.

TDD checks:

- [ ] In-memory SQLite schema creation.
- [ ] WAL pragma is applied for file-backed DB.
- [ ] Insert pending screen row.
- [ ] Duplicate content hash is no-op.
- [ ] List by stage/status.
- [ ] Update notes.
- [ ] Requeue failed/killed full run clears failure class.
- [ ] Baseline calibration row can be read and updated.
- [ ] Interrupted running row reset behavior is deterministic.

## Phase 03 - Config Ingestion

Purpose: make adding configs safe and boring.

Checklist:

- [ ] `scan_inbox()` lists all `.yaml` files in `queue/inbox/`.
- [ ] Each config is validated with existing `load_config()`.
- [ ] Invalid configs are logged and skipped; watchdog must not crash.
- [ ] Valid configs are inserted as `stage=SCREEN`, `status=PENDING`.
- [ ] Existing content hashes are skipped.
- [ ] `attention_type`, `run_dir`, and `config_name` are stored.
- [ ] Config copy/move semantics are explicit.
- [ ] `attn-queue add` copies one or more configs into inbox and validates first.
- [ ] Ingested configs should retain original content for provenance.

TDD checks:

- [ ] Valid config in inbox inserts one row.
- [ ] Invalid config does not insert and logs validation error.
- [ ] Adding the same config twice inserts one row.
- [ ] Changed seed changes content hash and inserts a new row.
- [ ] Missing run/data/model/train sections are rejected by existing validation.

## Phase 04 - Stage-1 Screener

Purpose: avoid spending overnight GPU time on configs that cannot run, cannot learn,
or have inactive mechanisms.

Screen behavior:

- [ ] Create temporary screen run dir under `runs/screen/<config_name>_<id>/`.
- [ ] Run existing `train.py` with overrides:
  - [ ] `max_steps=150`
  - [ ] `val_every=50`
  - [ ] `save_every=150`
- [ ] For non-standard attention, force `diagnostics.attention_diagnostics_every <= 50`
  so mechanism diagnostics can be emitted during the screen.
- [ ] Capture stdout/stderr.
- [ ] Apply kill criteria in order.
- [ ] Write verdict to ledger.
- [ ] Delete screen run directory unless `--keep-screens` is set.

Kill criteria in order:

- [ ] `OOM`: CUDA OOM appears in stderr.
- [ ] `NAN`: any NaN or Inf in train loss, validation loss, or stderr within 150
  steps.
- [ ] `COMPILE_ERROR`: train exits nonzero before step 10 or before metrics exist.
- [ ] `UNKNOWN`: train exits nonzero after step 10.
- [ ] `VERIFY_FAIL`: metrics are missing after an otherwise successful subprocess.
- [ ] `UNKNOWN`: final expected screen step is not reached.
- [ ] `FLAT_LOSS`: step-150 val loss is greater than step-10 val loss times `0.97`.
- [ ] `DEAD_GRAD`: mechanism-active check fails.
- [ ] `SLOW`: median tokens/sec below `baseline_tokens_per_sec * 0.30`.
- [ ] Otherwise mark `PASSED` and promote to `FULL`.

Mechanism-active check:

- [ ] For `standard`, mechanism-active is not required.
- [ ] For non-standard attention, read `evals/attention_diagnostics.jsonl` when present.
- [ ] For CP variants, mark `mechanism_active=1` if any row has
  `cp_gradient_norm > 1e-6`.
- [ ] For future QKV-track variants, use `track_gradient_norm`,
  `per_track_gradient_norm`, `branch_off_logit_delta`, or `track_output_delta`.
- [ ] Mark `mechanism_active=0` and `DEAD_GRAD` when diagnostics prove dead activity.
- [ ] Mark `mechanism_active=null` when diagnostics are missing.
- [ ] Block non-standard FULL promotion on missing diagnostics unless
  `queue.allow_missing_diagnostics: true` is explicit.

Baseline calibration:

- [ ] Maintain `baseline_screen_tokens_per_sec` in row `id='__baseline__'`.
- [ ] Update calibration when a `standard` attention config passes screening.
- [ ] Skip SLOW check if no baseline calibration exists.

TDD checks:

- [ ] Nonzero subprocess before step 10 classifies `COMPILE_ERROR`.
- [ ] CUDA OOM stderr classifies `OOM`.
- [ ] NaN val loss classifies `NAN`.
- [ ] Flat loss threshold classifies `FLAT_LOSS`.
- [ ] Low speed with baseline classifies `SLOW`.
- [ ] Missing baseline skips `SLOW`.
- [ ] CP diagnostics with positive gradient mark mechanism active.
- [ ] CP diagnostics with zero gradients mark dead grad.
- [ ] QKV-track diagnostics with positive track activity mark mechanism active.
- [ ] Missing diagnostics blocks non-standard FULL promotion by default.
- [ ] `--keep-screens` preserves screen run dir.

## Phase 05 - Full Run Executor

Purpose: execute one complete manifest-checked training/eval/summary pipeline.

Full Runner Pipeline Contract:

```text
uv run scripts/verify_data.py --data_root <data_root> --manifest <manifest_path> --verify_hashes
uv run scripts/train.py --config <config_path> --overwrite
uv run scripts/verify_run.py --run_dir <run_dir> --expect-complete-training --expect-sample --expect-data-manifest
uv run scripts/eval_loss.py --checkpoint <run_dir>/checkpoints/ckpt_last.pt --data_root <data_root>
uv run scripts/eval_generate.py --checkpoint <run_dir>/checkpoints/ckpt_last.pt --prompt "The history of mathematics"
uv run scripts/eval_hellaswag.py --checkpoint <run_dir>/checkpoints/ckpt_last.pt --max_examples 100
uv run scripts/summarize_run.py --run_dir <run_dir>
uv run scripts/verify_run.py --run_dir <run_dir> --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag --expect-data-manifest
```

Checklist:

- [ ] Resolve `data_root` and manifest from config/data manifest convention.
- [ ] Build `ckpt_last.pt` path from run directory.
- [ ] Refuse existing run artifacts unless
  `queue.allow_overwrite_existing_run_dir: true` is explicit.
- [ ] Stream stdout/stderr to `<run_dir>/queue_runner.log`.
- [ ] Tee runner logs to watchdog stdout.
- [ ] Stop on first nonzero exit.
- [ ] Classify failures from exit code/stderr.
- [ ] On success, read `evals/run_summary.json`.
- [ ] Require `evals/val_loss.json`, `evals/hellaswag.json`, `checkpoints/ckpt_last.pt`,
  and final `verify_run.py` success before marking PASSED.
- [ ] Require summary fields: `max_step`, `final_val_loss`, `best_val_loss`,
  `final_val_perplexity`, and `median_tokens_per_sec`.
- [ ] Write final numeric fields to ledger.
- [ ] Write HellaSwag accuracy when available.
- [ ] Record `VERIFY_FAIL` for failed final verification.
- [ ] Preserve failure logs.

TDD checks:

- [ ] Runner builds exact subprocess command list.
- [ ] Runner stops after first failed step.
- [ ] Runner writes queue log.
- [ ] Runner ingests run summary on success.
- [ ] Runner classifies OOM from stderr.
- [ ] Runner classifies unknown nonzero failure as `UNKNOWN`.
- [ ] Runner classifies failed final verifier as `VERIFY_FAIL`.
- [ ] Runner classifies existing artifacts as `RUN_DIR_EXISTS`.
- [ ] Runner refuses success if required artifacts or summary fields are missing.

## Phase 06 - Watchdog Daemon

Purpose: keep one GPU busy with FIFO work while preserving restartability.

Main loop:

- [ ] Scan inbox.
- [ ] Run all pending screens in FIFO order.
- [ ] Run the next pending full run in FIFO order.
- [ ] Sleep when no work exists.
- [ ] No threading.
- [ ] No async.
- [ ] One GPU.

Graceful shutdown:

- [ ] Write PID to `data/queue.pid`.
- [ ] SIGTERM sets a shutdown flag.
- [ ] Current step is allowed to finish.
- [ ] Daemon exits between steps.
- [ ] On restart, unfinished `RUNNING` rows with null `finished_at` are reset to `PENDING`.

Checklist:

- [ ] Do not run two experiments concurrently.
- [ ] Do not skip screening for inbox configs.
- [ ] Do not execute FULL rows until `full_run_approved=1`.
- [ ] Do not execute non-standard FULL rows without a passed `queue.requires_run`
  control unless `queue.skip_control_check: true` is explicit and warned.
- [ ] Do not promote missing-hypothesis full runs unless explicitly overridden.
- [ ] Print enough status for overnight logs.
- [ ] Keep loop interval configurable; default 60 seconds.

TDD checks:

- [ ] Watchdog calls screeners before full runner.
- [ ] Watchdog blocks unapproved FULL rows.
- [ ] Watchdog blocks non-standard FULL rows missing controls.
- [ ] Watchdog picks oldest pending full row.
- [ ] Watchdog sleeps when no work exists.
- [ ] SIGTERM flag stops loop after current step.
- [ ] PID file is written and removed or marked stale safely.

## Phase 07 - Queue CLI

Purpose: provide a terminal-only operator interface.

Entrypoint:

```text
attention-lab-queue = "attention_lab.queue.cli:main"
```

Command checklist:

- [ ] `attn-queue status`
  - [ ] Prints leaderboard.
- [ ] `attn-queue add <config_path> [<config_path>...]`
  - [ ] Validates configs.
  - [ ] Copies configs to inbox.
  - [ ] Inserts ledger rows or no-ops on duplicates.
- [ ] `attn-queue ls [--stage SCREEN|FULL] [--status PENDING|RUNNING|PASSED|FAILED|KILLED]`
  - [ ] Lists matching rows.
- [ ] `attn-queue show <run_id_or_name>`
  - [ ] Prints full ledger row.
  - [ ] Prints last 20 lines of `queue_runner.log` when available.
- [ ] `attn-queue note <run_id_or_name> "<text>"`
  - [ ] Updates notes field.
- [ ] `attn-queue kill <run_id_or_name>`
  - [ ] Sets pending run to `KILLED`.
  - [ ] Sends SIGTERM to watchdog for running run.
- [ ] `attn-queue requeue <run_id_or_name>`
  - [ ] Resets failed/killed full run to pending.
  - [ ] Clears failure class.
  - [ ] Does not rescreen.
- [ ] `attn-queue start`
  - [ ] Launches `scripts/queue_daemon.sh`.
- [ ] `attn-queue stop`
  - [ ] Sends SIGTERM to PID in `data/queue.pid`.
- [ ] `attn-queue leaderboard [--min-stage SCREEN|FULL] [--sort loss|ppl|speed]`
  - [ ] Prints filtered/sorted leaderboard.
- [ ] `attn-queue doctor --experiment <EXPERIMENT_ID>`
  - [ ] Prints read-only `OK`, `WARN`, and `FAIL` readiness checks.
  - [ ] Exits nonzero only on `FAIL`.
- [ ] `attn-queue approve <run_id_or_name>`
  - [ ] Allows an otherwise ready FULL run to execute.
- [ ] `attn-queue unapprove <run_id_or_name>`
  - [ ] Blocks a FULL run from executing.
- [ ] `attn-queue export-report --experiment <EXPERIMENT_ID>`
  - [ ] Writes queue `run_index.json` and `run_index.md`.
- [ ] `attn-queue morning-note --experiment <EXPERIMENT_ID> --shows ... --not-shows ... --next ...`
  - [ ] Appends to `decision_log.md`.

TDD checks:

- [ ] CLI parser dispatches every command.
- [ ] Invalid command returns nonzero.
- [ ] `add` validates before copying.
- [ ] `show` handles missing log gracefully.
- [ ] `kill` rejects unknown rows.
- [ ] `requeue` rejects passed rows.
- [ ] `approve` and `unapprove` update the ledger approval flag.
- [ ] `export-report` filters rows by experiment.
- [ ] `morning-note` rejects empty fields.

## Phase 08 - Leaderboard

Purpose: make the morning state readable without a web UI.

Checklist:

- [ ] ASCII-only output.
- [ ] Fixed-width columns.
- [ ] Header includes timestamp.
- [ ] Header includes running run and step when discoverable.
- [ ] Sections:
  - [ ] `RUNNING / RECENT`
  - [ ] `PENDING`
  - [ ] `KILLED / FAILED`
- [ ] Columns:
  - [ ] run name
  - [ ] attention type
  - [ ] stage
  - [ ] status
  - [ ] loss
  - [ ] perplexity
  - [ ] tokens/sec
  - [ ] VRAM
  - [ ] hot signal / mechanism_active
  - [ ] notes
- [ ] Use `---` for missing numeric values.
- [ ] Show `mechanism_active` as `1`, `0`, or `-`.
- [ ] Truncate long run names.
- [ ] Count current step from `metrics.jsonl` when possible.
- [ ] Print add-command hint.

TDD checks:

- [ ] Empty leaderboard renders.
- [ ] Running row renders current step.
- [ ] Passed row renders loss and ppl.
- [ ] Missing summary renders dashes.
- [ ] Mechanism-active values render as expected.
- [ ] Long names truncate deterministically.

## Phase 09 - Discipline Gate

Purpose: prevent meaningless configs from entering full-run execution.

Before any FULL config enters the queue, this file must exist:

```text
docs/experiments/<EXPERIMENT_ID>/hypothesis_<run_name>.md
```

Hypothesis Document Template:

```text
CLAIM:
<specific falsifiable statement>

KILL_CONDITION:
<specific result that kills this form of the idea>

MECHANISM_PROOF:
<exact diagnostic proving the mechanism is active>

NEAREST_BORING_EXPLANATION:
<simplest positive-result explanation that is not the hypothesis>

CONTROL_THAT_RULES_IT_OUT:
<completed or queued control run>
```

Checklist:

- [ ] The hypothesis doc exists before FULL promotion.
- [ ] All five fields are present.
- [ ] No field is empty.
- [ ] Claim is falsifiable.
- [ ] Kill condition is numeric or artifact-specific.
- [ ] Mechanism proof names exact diagnostics and thresholds.
- [ ] Boring explanation is explicit.
- [ ] Control run exists, is queued before candidate, or is completed.
- [ ] Queue blocks full promotion when the doc is missing.
- [ ] Optional `--skip-hypothesis-check` exists but is logged and visible.

TDD checks:

- [ ] Missing hypothesis doc blocks full run.
- [ ] Empty field blocks full run.
- [ ] Valid hypothesis doc allows full run.
- [ ] Skip flag allows run but records warning.
- [ ] Config-level `queue.hypothesis_doc` path is respected when added.

## Phase 10 - Required Fast-Run Ladder

Purpose: catch implementation bugs before expensive runs.

Every novel attention type must pass:

- [ ] Step 1: compile and shape/import check.
- [ ] Step 2: 150-step screen.
- [ ] Step 3: one-batch overfit test.
- [ ] Step 4: mechanism-active destructive test.
- [ ] Step 5: static/control run queued before candidate.

Step details:

- [ ] Compile/shape check catches import errors, wrong einsum dims, missing buffers,
  and basic forward failures.
- [ ] 150-step screen catches NaN, flat loss, dead gradients, OOM, and too-slow runs.
- [ ] One-batch overfit uses a tiny model and confirms it can drive a fixed batch
  toward near-zero loss.
- [ ] Mechanism-active destructive test disables the novel path and requires logits,
  loss, or activations to change.
- [ ] Static/control queued rule prevents candidate-first interpretation.

TDD checks:

- [ ] Tiny CPU/GPU configs exercise the exact new path.
- [ ] One-batch overfit helper can be run against any attention type.
- [ ] Destructive test reports nonzero delta for active mechanism.
- [ ] Candidate config with unmet `requires_run` does not start.

## Phase 11 - Mechanism Proof

Purpose: avoid trusting loss curves from a standard backbone with a dead side branch.

Required evidence for novel mechanisms:

- [ ] Gradient norms for mechanism-specific parameters.
- [ ] Parameter norms for mechanism-specific parameters.
- [ ] Lambda/effect-size values.
- [ ] Branch-on versus branch-off logit or loss deltas.
- [ ] Attention score statistics where relevant.
- [ ] Attention entropy where relevant.
- [ ] Diagnostic file path in run report.
- [ ] Mechanism-active flag in ledger.

For current CP variants:

- [ ] `attention_diagnostics.jsonl` exists for CP runs.
- [ ] `cp_gradient_norm > 1e-6` in at least one row for a basic active check.
- [ ] `lambda_value` is logged.
- [ ] `cp_score_std` and `standard_score_std` are logged.
- [ ] `cp_to_standard_score_std_ratio` is logged.
- [ ] Lambda-zero control remains distinguishable as a null/wiring control.

TDD checks:

- [ ] Diagnostics collect rows after backward.
- [ ] CP gradient norm is numeric when gradients exist.
- [ ] Fixed lambda has no trainable lambda parameter.
- [ ] Trainable lambda receives gradients.
- [ ] Destructive branch-off delta test can read a checkpoint and produce metrics.

## Phase 12 - Failure Taxonomy

Purpose: classify failures in a way that suggests the next useful action.

Failure classes and response:

- [ ] `NAN`
  - [ ] Treat as implementation/optimization issue first.
  - [ ] Check init scale, normalization, dtype, clipping.
  - [ ] Retry in float32 or with safer init before killing the idea.
- [ ] `FLAT_LOSS`
  - [ ] Treat as ambiguous.
  - [ ] Try lambda warmup or larger init once.
  - [ ] Kill the config after one failed retry, not necessarily the idea.
- [ ] `DEAD_GRAD`
  - [ ] Treat as implementation bug.
  - [ ] Fix gradient path before requeue.
- [ ] `SLOW`
  - [ ] Interpret carefully at 150 steps.
  - [ ] If theoretically FLOP-neutral, run a 200-step timing check before killing.
- [ ] `OOM`
  - [ ] Reduce batch/model or add separate diagnostic config.
  - [ ] Do not silently alter direct-comparison configs.
- [ ] `COMPILE_ERROR`
  - [ ] Fix code.
- [ ] `VERIFY_FAIL`
  - [ ] Read verifier output before interpreting metrics.
- [ ] `UNKNOWN`
  - [ ] Preserve logs and classify manually.

TDD checks:

- [ ] Failure classifier maps known stderr patterns.
- [ ] Each failure class appears in CLI/leaderboard.
- [ ] Notes field can record human interpretation.

## Phase 13 - Combinatorial Discipline

Purpose: prevent the queue from becoming a slot machine.

Rules:

- [ ] One mechanism first, then parameter variation.
- [ ] No more than three configs for one idea family before mechanism_active=1.
- [ ] New parameter variation can be queued only after base version has
  `mechanism_active=1`.
- [ ] Two mechanisms can be combined only after both individually pass with
  `mechanism_active=1`.
- [ ] Control before candidate.
- [ ] Candidate interpretation requires nearest boring explanation to be ruled out.

TDD checks:

- [ ] Queue policy can count family configs.
- [ ] Family over-limit emits warning or blocks, depending configured strictness.
- [ ] Combined-mechanism config requires completed individual mechanism rows.

## Phase 14 - Morning Workflow

Purpose: force interpretation before adding more runs.

Morning checklist:

- [ ] Run `attn-queue status`.
- [ ] Read mechanism-active/hot-signal column before loss.
- [ ] Run `attn-queue show <interesting_run>`.
- [ ] Read last 20 lines of `queue_runner.log`.
- [ ] Write a morning note before adding new configs.
- [ ] Export the run index for the experiment.

Morning note template:

```text
SHOWS: <what this run actually demonstrated>
NOT_SHOWS: <what this run did not demonstrate>
NEXT: <one run that would most change interpretation>
```

Promotion checklist:

- [ ] `mechanism_active = 1`.
- [ ] Loss improves over nearest control, not just standard baseline.
- [ ] At least one boring explanation is ruled out by a completed control.
- [ ] If all three are not true, next run is a control or ablation, not scale-up.

TDD checks:

- [ ] `attn-queue note` stores notes exactly.
- [ ] `attn-queue morning-note` appends decision log entries.
- [ ] `attn-queue export-report` writes JSON and Markdown run indexes.
- [ ] Leaderboard displays note excerpt.
- [ ] Promotion helper refuses missing mechanism/control evidence.

## Phase 15 - E002 And Future Config Convention

Purpose: make future experiment families queue-ready without changing the harness.

For E002 and onward:

```text
configs/experiments/E002_multitrack_qkv_shift_register/
  standard_30m_seed1.yaml
  multi_qkv_static_3track_30m_seed1.yaml
  multi_qkv_train_and_layer_shift_3track_30m_seed1.yaml
  multi_qkv_train_shift_3track_30m_seed1.yaml
  multi_qkv_layer_shift_3track_30m_seed1.yaml
  multi_qkv_softmix_3track_30m_seed1.yaml
  multi_qkv_train_shift_warmup_3track_30m_seed1.yaml
```

Checklist:

- [ ] Experiment has an entry in `docs/experiments/experiments.yaml`.
- [ ] Plan exists before configs are added.
- [ ] Report directory exists.
- [ ] Standard/control configs come before candidates.
- [ ] Each config has unique `run.out_dir`.
- [ ] Each config preserves fixed comparison fields unless explicitly marked
  diagnostic.
- [ ] Each candidate has a hypothesis doc.
- [ ] Each candidate names the nearest control.
- [ ] Queue optional config fields are documented before use.

Potential queue config extension:

```yaml
queue:
  requires_run: standard_30m_seed1
  hypothesis_doc: docs/experiments/E002_multitrack_qkv_shift_register/hypothesis_<run_name>.md
  mechanism_check: qkv_track_activity
  full_run_approved: false
  allow_overwrite_existing_run_dir: false
```

Before adding this field:

- [ ] Extend config validation strictly.
- [ ] Add tests for unknown queue keys.
- [ ] Add tests for `requires_run` gating.
- [ ] Add docs explaining queue fields are orchestration metadata, not model/training
  behavior.

## Phase 16 - TDD And QC Gates

Purpose: ensure queue/discipline work is reviewable and regression-resistant.

Required implementation order:

1. [ ] Write failing tests for the unit or workflow surface.
2. [ ] Implement the smallest change that satisfies those tests.
3. [ ] Run targeted tests.
4. [ ] Run full tests.
5. [ ] Run ruff.
6. [ ] Update docs/reports honestly.
7. [ ] Commit.
8. [ ] Push.

Queue implementation order:

1. [ ] `ledger.py`
2. [ ] `screener.py`
3. [ ] `runner.py`
4. [ ] `watchdog.py`
5. [ ] `leaderboard.py`
6. [ ] `cli.py`
7. [ ] `scripts/queue_daemon.sh`
8. [ ] `pyproject.toml` entrypoint
9. [ ] `.gitignore`
10. [ ] `queue/inbox/.gitkeep`

Minimum QC command set:

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
```

When queue code exists, add:

```bash
uv run attention-lab-queue status
uv run attention-lab-queue ls
uv run attention-lab-queue export-report --experiment E001_cp_trilinear_attention
uv run attention-lab-queue doctor --experiment E001_cp_trilinear_attention
```

When full-run artifacts exist, add:

```bash
uv run scripts/verify_run.py \
  --run_dir <RUN_DIR> \
  --expect-complete-training \
  --expect-sample \
  --expect-eval-loss \
  --expect-hellaswag \
  --expect-data-manifest
```

## Coverage Matrix

| Requirement | Covered By |
| --- | --- |
| Queue filesystem | Phase 01 |
| SQLite ledger | Phase 02 |
| Config ingestion | Phase 03 |
| Stage-1 screen | Phase 04 |
| Mechanism-active check | Phase 04, Phase 11 |
| Baseline speed calibration | Phase 04 |
| Full-run executor | Phase 05 |
| Log capture | Phase 05 |
| Watchdog daemon | Phase 06 |
| CLI commands | Phase 07 |
| Leaderboard | Phase 08 |
| Run naming/deduplication | Phase 02, Phase 03 |
| E002 config convention | Phase 15 |
| Morning workflow | Phase 14 |
| Non-goals | Scope Boundary |
| Implementation order | Phase 16 |
| Hypothesis gate | Phase 09 |
| Required ladder | Phase 10 |
| Mechanism proof | Phase 11 |
| Failure taxonomy | Phase 12 |
| Combinatorial trap | Phase 13 |
| Source freeze | Phase 00 |

## Final Pre-Queue Checklist

Before the first config enters `queue/inbox/`:

- [ ] Queue ledger tests pass.
- [ ] Screener tests pass.
- [ ] Runner command-list tests pass.
- [ ] Watchdog tests pass.
- [ ] CLI tests pass.
- [ ] Leaderboard tests pass.
- [ ] Hypothesis gate tests pass.
- [ ] Existing training harness tests still pass.
- [ ] Existing E001 configs still validate.
- [ ] Data manifest verifies.
- [ ] README links to this guide.
- [ ] Architecture variant checklist links to this guide.
- [ ] Full-run freeze policy is understood and handled outside the active
  implementation working copy.
