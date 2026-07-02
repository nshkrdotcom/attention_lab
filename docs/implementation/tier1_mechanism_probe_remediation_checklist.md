# Tier-1 Mechanism Probe Remediation Checklist

Status: implementation remediation in progress on 2026-07-02.

This checklist tracks the audit feedback against the Tier-1 mechanism-probe suite. It is an implementation/QC artifact, not mechanism evidence.

## Remediation Items

- [x] Re-read operating docs and inspect the existing mechanism package before editing.
  - Read `AGENTS.md`, `README.md`, `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md`, experiment contract/checklists, queue guide, experiment registry, E001-E004 plans, and `docs/mechanism_probe_framework.md`.
  - Inspected `src/attention_lab/mechanisms/`, scripts, tests, task suites, and current checkpoint paths.

- [x] Strengthen deterministic task-suite validation.
  - Added `metadata.content_sha256` over task metadata and records.
  - Confirmatory validation rejects missing or mismatched fingerprints.
  - Built-in Tier-1 validate-only regenerates from metadata and rejects files that do not match deterministic generator output.
  - Regenerated committed E003/E004 Tier-1 task suites with fingerprints.

- [x] Resolve exploratory claim-gate ambiguity.
  - Kept `exploratory_probe_signal` as a capped status.
  - Added `claim_gate_passed` and `status_kind` to cell gate outputs.
  - `exploratory_probe_signal` now remains machine-readable as not a passed confirmatory claim gate.

- [x] Make E004 `operator_probs` capture/probe-only for patching.
  - Marked `operator_probs` non-continuous in the Tier-1 preset.
  - Removed its full-layer comparator and added explicit no-control/no-comparator reasons.
  - Full-width E004 operator output sites remain continuous patch/restoration candidates.

- [x] Make random-site null pool scope explicit.
  - Suite preflight now records the complete preset-declared Tier-1 random-site null family.
  - Selection still inspects actual captured feature shapes and compatible tensor kinds before using a random-site null.

- [x] Update operator documentation.
  - Updated `docs/mechanism_probe_framework.md`.
  - Updated `AGENTS.md`.
  - Updated `README.md`.
  - Reconciled stale checkpoint text in `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md`.

- [x] Regenerate current Tier-1 suite reports from checkpoints.
  - E003/E004 Tier-1 exploratory and confirmatory report directories were regenerated from current local checkpoints.
  - Validated all four regenerated report directories with `scripts/summarize_mechanism_probe_suite.py --validate`.
  - Regenerated E004 confirmatory artifacts mark `operator_probs` patching/mediation invalid because the site is non-continuous.

- [x] Run full QC and record results before commit.
  - `uv sync` passed.
  - `uv run ruff check .` passed.
  - Targeted mechanism tests passed: `52 passed`.
  - Full `uv run pytest` passed: `455 passed, 1 skipped`.
  - E001-E004 `scripts/validate_experiment.py` passed.
  - FineWeb-Edu data verification with `--verify_hashes` passed.
  - E001-E004 `attn-queue doctor` passed.
  - Tier-1 preflight passed with E003/E004 candidate and matched-control checkpoints present.
  - Backfill checkpoint-path consistency check passed.
