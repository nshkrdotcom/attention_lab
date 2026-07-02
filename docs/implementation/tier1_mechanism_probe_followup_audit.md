# Tier-1 Mechanism Probe Follow-Up Audit

Status vocabulary for this document:

```text
implemented
implemented_but_hardened_in_this_followup
still_deferred_with_reason
not_applicable_to_tier1
```

## Requirement Checklist

| Requirement | Status | Notes |
| --- | --- | --- |
| real trained linear probes | implemented | `linear_probe.py` trains real logistic probes; CLI tests exercise actual code paths. |
| grouped splitting | implemented | `grouped_train_test_split` keeps pair/template groups out of leakage splits. |
| AUC | implemented | Probe metrics emit `linear_probe_auc` and AUC contrasts. |
| shuffled-label nulls | implemented | Same probe path retrained on shuffled train labels. |
| random-site nulls | implemented_but_hardened_in_this_followup | Reporting now includes candidate tensor kind and considered sites with rejection reasons. |
| matched controls | implemented_but_hardened_in_this_followup | Preflight records expected/actual control paths; missing confirmatory controls require explicit diagnostic mode. |
| seed-matched control resolution | implemented | E003 resolves seed1 control; E004 resolves seed2 control. |
| control override behavior | implemented_but_hardened_in_this_followup | Noncanonical and forced overrides remain capped and visible in artifacts. |
| bootstrap CIs | implemented | Bootstrap results gate probe, null, specificity, restoration, and mediation metrics. |
| minimum-N enforcement | implemented | Confirmatory runs reject `--min-n` below the 50-pair floor. |
| confirmatory task-suite floor | implemented | Committed E003/E004 suites validate at 50 pairs/family. |
| deterministic task generation | implemented_but_hardened_in_this_followup | Generator now emits restoration alignment metadata. |
| hypothesis doc validation | implemented_but_hardened_in_this_followup | Hypotheses now state full-layer restoration, alignment, and pooling requirements. |
| exploratory-mode claim cap | implemented | Exploratory runs cannot reach confirmatory claims. |
| probe-only staged execution | implemented | `--exploratory --probe-only` skips patching/restoration. |
| full confirmatory execution | implemented_but_hardened_in_this_followup | Confirmatory E003/E004 reran from local checkpoints with hardened metadata and schema. |
| causal patching/restoration metrics | implemented_but_hardened_in_this_followup | Patching now requires validated token alignment and source/target token mappings. |
| mediation_fraction | implemented | Emitted and FDR-gated when valid. |
| alignment-to-control metric | implemented | Probe-direction cosine/absolute alignment emitted when matched control probes exist. |
| FDR-BH scope | implemented_but_hardened_in_this_followup | FDR includes full-layer restoration and emits invalid/unavailable cells separately. |
| decoy/specificity gate | implemented | `target_vs_decoy_specificity` is bootstrapped and FDR-gated. |
| summary artifact generation | implemented_but_hardened_in_this_followup | Summary now reports pooling, restoration alignment, and invalid/unavailable FDR cells. |
| claim gate evaluation | implemented_but_hardened_in_this_followup | Top claim now requires task-aligned pooling, valid restoration alignment, and full-layer restoration FDR pass. |
| Tier-2/Tier-3 non-executability | implemented | Stub presets remain `stub_not_executable`. |
| single-seed caveat | implemented | Docs and summaries keep single-seed/non-replicated caveat. |

## Remaining Limitations

- The current Tier-1 artifacts are single-seed and not replication.
- The current confirmatory E003/E004 reruns remain `insufficient_evidence`.
- E004 `operator_probs` remains a low-dimensional probability site with no matched-dimensional compatible random-site null and no matched-control site.
- The task suite is a deterministic negation-focused Tier-1 suite; broader task families require new pre-registration before use.
- SAE feature purity/polysemanticity analysis remains deferred Stage-2 work.
- Cross-architecture E003/E004 universality and cross-seed CKA/probe-direction replication remain deferred Tier-1.5/Stage-2 work.

## Follow-Up Changes

- Added restoration alignment metadata validation for GPT-2 target/foil tokens, clean/corrupt answer positions, and explicit source/target patch token indices.
- Added source-token to target-token cache patching support without padding, truncation, or whole-sequence fallback.
- Added explicit feature pooling strategies and default confirmatory `patch_positions_mean` pooling.
- Made confirmatory site selection strict; exploratory unknown sites require explicit metadata and remain noncanonical.
- Added preflight reporting and explicit diagnostic mode for missing matched controls.
- Added FDR-BH invalid/unavailable cell reporting and full-layer restoration coverage.
- Added artifact schema validation through `scripts/summarize_mechanism_probe_suite.py --validate`.
- Added `scripts/verify_tier1_mechanism_probe_suite.py` for preflight and checkpoint-gated real execution.
- Regenerated deterministic E003/E004 Tier-1 task suites and reran real E003/E004 probe-only and confirmatory artifacts from local checkpoints.

## Final Run Report

Files changed:

```text
AGENTS.md
EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md
README.md
configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml
configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml
docs/mechanism_probe_framework.md
docs/mechanisms/hypotheses/
reports/mechanisms/probes/E003_differential_tier1_*_inventory_path/
reports/mechanisms/probes/E004_operator_valued_tier1_*_inventory_path/
scripts/run_mechanism_probe_suite.py
scripts/summarize_mechanism_probe_suite.py
scripts/verify_tier1_mechanism_probe_suite.py
src/attention_lab/mechanisms/
tests/test_mechanism_*.py
```

Tests added:

```text
restoration alignment validation and source/target token patching tests
feature pooling tests
strict confirmatory/exploratory site selection tests
missing-control diagnostic preflight tests
random-site considered-site rejection tests
FDR invalid/unavailable cell tests
artifact schema validation tests
committed task-suite validation tests
verification-script preflight test
```

Tests run:

```text
uv run pytest tests/test_mechanism_linear_probe.py
uv run pytest tests/test_mechanism_controls.py
uv run pytest tests/test_mechanism_claim_gates.py
uv run pytest tests/test_mechanism_patching.py
uv run pytest tests/test_mechanism_probe_suite_cli.py
uv run pytest tests/test_mechanism_probe_summary.py
uv run pytest tests/test_mechanism_task_generation_cli.py
uv run ruff check .
uv run scripts/verify_tier1_mechanism_probe_suite.py --preflight-only
uv run scripts/verify_tier1_mechanism_probe_suite.py --device cuda --bootstrap-samples 1000 --batch-size 16
uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E003_differential_tier1_probe_only_inventory_path --validate
uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E003_differential_tier1_confirmatory_inventory_path --validate
uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E004_operator_valued_tier1_probe_only_inventory_path --validate
uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E004_operator_valued_tier1_confirmatory_inventory_path --validate
uv run ruff check .
uv run pytest
uv run pytest --run-integration
uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
uv run scripts/validate_experiment.py --id E003_qkv_architecture_gauntlet
uv run scripts/validate_experiment.py --id E004_operator_binding_qkv_gauntlet
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
uv run attn-queue doctor --experiment E001_cp_trilinear_attention
uv run attn-queue doctor --experiment E002_multitrack_qkv_shift_register
uv run attn-queue doctor --experiment E003_qkv_architecture_gauntlet
uv run attn-queue doctor --experiment E004_operator_binding_qkv_gauntlet
uv sync
```

QC result:

```text
Targeted mechanism tests passed.
ruff check passed.
Full pytest passed: 452 passed, 1 skipped.
pytest --run-integration passed: 453 passed.
E001-E004 validate_experiment commands passed.
FineWeb-Edu data manifest and hashes verified.
E001-E004 attn-queue doctor commands passed.
Documented summary artifact validation passed for all four suite output directories.
uv sync completed without dependency changes.
```

Checkpoint availability:

```text
E003 candidate checkpoint: available
E003 canonical seed1 control checkpoint: available
E004 candidate checkpoint: available
E004 canonical seed2 control checkpoint: available
```

Produced artifacts:

```text
reports/mechanisms/probes/E003_differential_tier1_probe_only_inventory_path/ -> exploratory_probe_signal
reports/mechanisms/probes/E003_differential_tier1_confirmatory_inventory_path/ -> insufficient_evidence
reports/mechanisms/probes/E004_operator_valued_tier1_probe_only_inventory_path/ -> exploratory_probe_signal
reports/mechanisms/probes/E004_operator_valued_tier1_confirmatory_inventory_path/ -> insufficient_evidence
```

No `candidate_mechanism_evidence` claim was produced.
