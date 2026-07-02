# Tier-1 Mechanism Probe Framework

This document defines the Tier-1 mechanism probe suite for promoted E003/E004 candidates.

The narrow Tier-1 question is:

```text
Do the promoted E003/E004 candidate components provide statistically controlled,
checkpoint-backed evidence of task-relevant mechanism behavior beyond matched controls?
```

Tier-1 also records a minimal representational-alignment diagnostic:

```text
Is the candidate exposing a representation aligned with an already-present control
representation, or does it appear to form a different decomposition?
```

This is not a generic probing platform, not SAE infrastructure, not a Tier-2/Tier-3 executor, and not evidence from training.

## What Backfill Proves

Mechanism backfill identifies what can be investigated from existing artifacts. `checkpoint_recompute` means a checkpoint exists and activations can be recomputed. It does not prove a semantic mechanism, a causal role, or architecture superiority.

Quick probes prove the capture/intervention plumbing can run on selected prompts. Tier-1 probes add trained probes, grouped splits, nulls, matched controls, patching/restoration metrics, bootstrap CIs, FDR-BH correction, and claim gates.

Mean activation deltas are descriptive only. They can guide inspection, but no claim gate passes from raw `delta > 0`.

## Executable Tier-1 Presets

Presets live in `src/attention_lab/mechanisms/presets.py`.

Executable Tier-1 candidates:

```text
E003 differential -> standard_refactor_control_30m_seed1_rung500
E004 operator-valued -> standard_refactor_control_30m_seed2_rung500
```

The E004 seed2 control is intentional. Do not pair E004 against the E003 seed1 control.

Tier-2/Tier-3 presets are `stub_not_executable` until future work explicitly promotes them.

## Task Contrasts

Task files are YAML or JSON with:

```text
x_pos      target phenomenon present
x_neg      target phenomenon removed or contrasted
x_para     target phenomenon preserved under paraphrase or marker variation
x_decoy    confound or near-control example
pair_id
template_id
family_id
metadata
```

`pair_id`, `template_id`, and `family_id` are required so splits cannot leak paired or templated examples across train/test boundaries.

Confirmatory suites require deterministic provenance:

```text
generator_name
generator_version
template_set
filler_set
generation_seed
created_at
```

`src/attention_lab/mechanisms/task_generation.py` provides deterministic template/filler generation. It uses a local seeded RNG for filler assignment, writes fixed provenance, and validates GPT-2 single-token restoration labels. Committed confirmatory suites live in:

```text
configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml
configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml
```

Regenerate or validate them with:

```bash
uv run scripts/generate_tier1_mechanism_tasks.py \
  --output configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml \
  --candidate e003_differential \
  --pairs-per-family 50 \
  --seed 1

uv run scripts/generate_tier1_mechanism_tasks.py \
  --output configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml \
  --candidate e004_operator_valued \
  --pairs-per-family 50 \
  --seed 2

uv run scripts/generate_tier1_mechanism_tasks.py \
  --output configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml \
  --candidate e003_differential \
  --pairs-per-family 50 \
  --validate-only
```

Confirmatory runs enforce at least 50 contrast pairs per task family. `--min-n` may not be set below 50 for confirmatory evidence. Exploratory runs may use smaller or hand-authored files, but their claim ladder is capped.

## Hypothesis Docs

Confirmatory runs require:

```bash
--hypothesis-doc docs/mechanisms/hypotheses/<name>.yaml
```

Required YAML fields:

```text
CLAIM
KILL_CONDITION
MECHANISM_PROOF
NEAREST_BORING_EXPLANATION
CONTROL_THAT_RULES_IT_OUT
TARGET_SITES
TASK_CONTRASTS
PRIMARY_METRIC
STATISTICAL_TEST
MIN_N
FDR_SCOPE
EXPECTED_DIRECTION
```

Without `--hypothesis-doc`, the run must use `--exploratory` and cannot make confirmatory claims.

The committed Tier-1 hypothesis docs are:

```text
docs/mechanisms/hypotheses/E003_differential_negation_tier1.yaml
docs/mechanisms/hypotheses/E004_operator_valued_negation_tier1.yaml
```

Malformed confirmatory inputs fail before model/checkpoint loading. This includes a missing or invalid hypothesis doc, a task suite below 50 pairs per family, missing deterministic provenance, missing decoys, and invalid restoration token metadata for full runs.

## Confirmatory Preflight And Sites

Before model loading, the suite records a preflight block in `metrics.json` with candidate/config/checkpoint paths, canonical and actual control paths, task-suite validation, hypothesis validation, selected site metadata, patching metadata validation, and feature-pooling strategy.

Confirmatory `--sites` values must be declared in the Tier-1 preset for that candidate. Unknown confirmatory sites fail before model execution. Exploratory unknown sites are allowed only with explicit `--site-spec-file` metadata containing the site, layer, tensor kind, continuity, matched-control site or no-control reason, and full-layer comparator or no-comparator reason. Such sites are noncanonical and cannot support `candidate_mechanism_evidence`.

Confirmatory runs require a candidate checkpoint path that exists. Missing matched controls fail before execution unless `--allow-diagnostic-with-missing-control` is used; that flag creates capped diagnostic artifacts only and cannot reach `controlled_probe_signal` or `candidate_mechanism_evidence`.

## Probe Metrics

The suite trains real linear probes with grouped train/test splitting. The stricter default groups by `template_id`; all variants from a `pair_id` stay in the same fold. This prevents leakage from shared surface structure across `x_pos`, `x_neg`, `x_para`, and `x_decoy`.

Primary metrics include:

```text
linear_probe_auc
auc_minus_shuffled_auc
auc_minus_random_site_auc
auc_minus_matched_control_auc
target_vs_decoy_specificity
```

Shuffled-label nulls retrain the same probe after shuffling train labels. Random-site nulls retrain the same probe on a randomly selected non-candidate site with matched dimensionality and compatible site kind.

Random-site null selection inspects actual captured feature shapes. It never pads, truncates, projects, or coerces mismatched sites. Missing random-site nulls are feasibility limits for that `(site x layer)` cell, not automatic implementation failures and not run-wide caps.

Low-dimensional E004 `operator_probs` sites are not coerced into d_model-shaped comparisons.

## Feature Pooling

Captured sequence tensors are pooled explicitly and every cell records the strategy:

```text
mean_sequence
final_token
answer_position
patch_positions_mean
```

`auto` resolves to `mean_sequence` for exploratory runs and to `patch_positions_mean` for confirmatory runs. Confirmatory `candidate_mechanism_evidence` requires task-aligned pooling (`answer_position` or `patch_positions_mean`). `mean_sequence` is allowed as exploratory/diagnostic pooling but caps the top claim because it can wash out localized mechanism behavior.

## Matched Controls

Matched controls are resolved from presets unless overridden:

```text
--control-mode matched
--control-checkpoint <path>
--control-config <path>
```

Overrides are recorded in `metrics.json` and `summary.md`. Noncanonical or seed-mismatched controls cap claims below `candidate_mechanism_evidence`, even with `--force-noncanonical-control`.

No mechanism evidence claim can pass without matched control evidence.

## Statistics

The suite computes bootstrap CIs for primary effects and target-vs-decoy specificity. FDR-BH correction is applied over the full run-level comparison family:

```text
every computed site x layer x task_family x metric cell in the run
```

This includes primary probe metrics, shuffled-label contrasts, random-site contrasts, matched-control contrasts, specificity metrics, `component_patch_restoration`, `full_layer_patch_restoration`, and `mediation_fraction` when present.

Do not narrow FDR-BH to only the pre-registered target site or only the primary metric unless no other tested cells were computed.

`metrics.json` separates `fdr_bh.tested_cells` from `fdr_bh.invalid_or_unavailable_cells`. Invalid or unavailable cells record a reason and do not receive meaningful p-values.

Target-vs-decoy specificity uses a bootstrap CI on:

```text
target_effect - decoy_effect
```

The CI must exclude zero in the expected direction under the same FDR-BH family.

## Patching And Restoration

`--probe-only` skips interventions, causal patching, restoration, and mediation metrics.

Full runs compute patching only when task records include explicit GPT-2 single-token restoration metadata:

```text
metadata.target_token_text
metadata.foil_token_text
metadata.target_token_id
metadata.foil_token_id
metadata.clean_answer_position
metadata.corrupted_answer_position
metadata.patch_token_indices
metadata.clean_patch_token_indices
metadata.corrupted_patch_token_indices
metadata.clean_corrupt_token_alignment
```

The committed suites use `" true"` and `" false"` as single GPT-2 next-token labels. Multi-token labels are rejected; the suite never silently uses the first subtoken. Clean and corrupted prompts are patched only at validated aligned positions. If token lengths differ, explicit clean/corrupted patch indices are required. The suite does not patch whole sequence tensors across unaligned clean/corrupt prompts. Missing token metadata, invalid patch alignment, or invalid denominators makes restoration invalid and blocks gates depending on it.

Restoration formula:

```text
restoration_score =
  (patched_logitdiff - corrupted_logitdiff)
  / (clean_logitdiff - corrupted_logitdiff)
```

Mediation formula:

```text
mediation_fraction =
  component_patch_restoration / full_layer_patch_restoration
```

If denominators are too small, metrics are marked invalid. Discrete route/index sites are capture-only unless a future route-replacement intervention is validated.

## Alignment To Control

The suite computes:

```text
probe_direction_cosine_to_control
probe_direction_alignment_abs
```

This compares a candidate probe weight direction to its matched-control probe weight direction when dimensions match.

High alignment suggests an already-present or universal feature may be surfaced differently. Low alignment may indicate a different decomposition worth scrutiny. Low alignment is not representational novelty evidence by itself.

Candidate-to-control alignment does not prove cross-architecture universality.

## Claim Gates

Mechanism-probe status vocabulary:

```text
insufficient_evidence
exploratory_probe_signal
controlled_probe_signal
candidate_mechanism_evidence
```

This vocabulary is scoped to mechanism probes and is distinct from the repository-wide experiment status vocabulary.

`candidate_mechanism_evidence` means single-seed, checkpoint-backed, statistically controlled evidence. It is not replication.

No claim gate can pass without minimum N, grouped split discipline, bootstrap CI/statistical correction, matched control evidence, and non-decoy specificity evidence. Confirmatory evidence also requires the 50-pair task-suite floor and a valid hypothesis doc.

Full confirmatory mode is the only path that can reach `candidate_mechanism_evidence`. Non-exploratory `--probe-only` is rejected; cheap scans must use `--exploratory --probe-only`.

## Commands

Exploratory E003 cheap scan:

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

Confirmatory E003 full suite:

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

Exploratory E004 cheap scan:

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

Confirmatory E004 full suite:

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

Each run writes:

```text
metrics.json
claim_gates.json
summary.md
```

`metrics.json` contains task-suite validation, control canonicality, per-cell probe/null/control/alignment metrics, FDR-BH tested cells, and patching/restoration metrics when run. `claim_gates.json` contains per-cell blockers/caps and the overall mechanism-probe status. `summary.md` is the human-readable limitation report.

Regenerate a summary from existing suite artifacts:

```bash
uv run scripts/summarize_mechanism_probe_suite.py \
  --output-dir reports/mechanisms/probes/<suite-output-dir> \
  --validate
```

Check preflight and run real suites only when local checkpoints exist:

```bash
uv run scripts/verify_tier1_mechanism_probe_suite.py --preflight-only
uv run scripts/verify_tier1_mechanism_probe_suite.py --device cuda
```

## Disallowed Claims

Tier-1 does not allow claims that:

```text
the mechanism is universal
the mechanism is replicated
the architecture has solved negation
the architecture has lower superposition
the candidate is representationally novel
the result proves causal mechanism in general
candidate-to-control alignment proves cross-architecture universality
single-seed mechanism evidence is replication
```

## Deferred Stage-2 SAE Work

SAE feature-purity and polysemanticity analysis is deferred Stage-2 work. It should test whether flagged stream features fire cleanly on the target phenomenon and not on sentiment, frequency, topic, or decoy confounds.

This is not implemented in Tier-1. Build it only after Tier-1 clears `candidate_mechanism_evidence`.

## Deferred Tier-1.5 Universality Work

Current alignment compares each candidate to its own matched control.

Future cross-architecture universality analysis should compare flagged E003 and E004 directions against each other, using probe-direction alignment and/or CKA, then cross-seed replication once replicated checkpoints exist.

## Replication Limitation

Current Tier-1 E003/E004 mechanism results are single-seed unless replicated checkpoints are explicitly present.

A second seed per Tier-1 architecture is the natural next investment after Tier-1 clears gates. Future replication should include cross-seed alignment or CKA-style representational comparison.
