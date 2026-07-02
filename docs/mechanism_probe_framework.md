# Tier-1 Mechanism Probe Framework

This document defines the Tier-1 statistical mechanism-probe suite. It extends the
existing quick capture/intervention probe path; it does not replace training,
backfill, verification, or queue promotion.

The narrow Tier-1 question is:

```text
Do the promoted E003/E004 candidate components provide statistically controlled,
checkpoint-backed evidence of task-relevant mechanism behavior beyond matched controls?
```

Tier-1 can also report a minimal representational-alignment diagnostic:

```text
Is the candidate exposing a representation that appears aligned with an already-present
control representation, or does it appear to form a different representational decomposition?
```

That diagnostic is not a novelty proof.

## What Backfill Proves

Backfill inventories say what is recoverable from current artifacts. A checkpoint with
`checkpoint_recompute` means activations can be recomputed from the local checkpoint.
It does not mean the mechanism hypothesis is true.

Quick probes show that hook sites and simple interventions can run. They are plumbing
and triage artifacts. Mean activation deltas are descriptive only. They are insufficient
because they have no grouped split discipline, no trained readout, no null comparison,
no matched control, no task specificity, and no multiple-comparison correction.

## What The Suite Does

The suite runner is:

```bash
uv run scripts/run_mechanism_probe_suite.py
```

It writes:

```text
metrics.json
claim_gates.json
summary.md
```

The summary can be regenerated with:

```bash
uv run scripts/summarize_mechanism_probe_suite.py --input-dir <suite-output-dir>
```

Tier-1 executable presets are deliberately narrow:

```text
E003_qkv_architecture_gauntlet / differential
E004_operator_binding_qkv_gauntlet / operator_valued
```

Scope-gated, dynamic-value, q3k3v3, and later Tier-2/Tier-3 ideas remain
`stub_not_executable` in this suite until a future pass adds them explicitly.

## Task Contrasts

Task files must contain contrast records with:

```text
x_pos
x_neg
x_para
x_decoy
metadata
pair_id
template_id
family_id
```

`x_pos` contains the target phenomenon. `x_neg` removes or contrasts it. `x_para`
preserves it under a paraphrase or marker variation. `x_decoy` is a confound or
near-control, such as frequency, topic, shape, or adverb control.

`pair_id`, `template_id`, and `family_id` are required because paired or templated
examples must not leak across train/test splits. At minimum, all variants from the
same `pair_id` stay in the same fold. When template leakage is plausible, the split
can group by `template_id`.

Confirmatory task suites require at least 50 contrast pairs per family. A contrast
pair means one grouped unit containing the relevant positive, negative, paraphrase,
and decoy variants for the same template/filler configuration. The committed floor
exists to prevent scientifically empty five-line task files from satisfying gates.

Exploratory runs may use smaller or hand-authored task files, but their claim ladder
is capped.

## Deterministic Generation

`src/attention_lab/mechanisms/task_generation.py` provides deterministic
template/filler generation. A deterministic suite records:

```text
generator_name
generator_version
template_set
filler_set
generation_seed
created_at
```

Confirmatory runs require deterministic provenance or a separately validated committed
suite with equivalent provenance. The loader does not silently drop missing decoys.

## Hypothesis Docs

Confirmatory runs require:

```bash
--hypothesis-doc docs/mechanisms/hypotheses/<hypothesis-name>.yaml
```

The YAML document must contain:

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

Without a hypothesis doc, use `--exploratory`. Exploratory runs cannot make
confirmatory claims.

## Linear Probes

The suite trains real binary linear probes over checkpoint-recomputed activations.
It reports AUC and contrasts such as:

```text
linear_probe_auc
auc_minus_shuffled_auc
auc_minus_random_site_auc
auc_minus_matched_control_auc
```

The shuffled-label null trains the same probe after seed-controlled label shuffling.
The random-site null trains the same probe on a seed-selected non-candidate site with
matched feature dimensionality and compatible tensor kind. The matched-control
comparison trains the same probe on the preset control site when available.

No claim gate can pass from raw activation mean deltas alone.

## Random-Site Nulls

The random-site null is selected from actual captured activation shapes:

```text
same feature dimensionality
compatible tensor kind
not the candidate site
seed-controlled selection
```

The suite never pads, truncates, projects, or coerces incompatible dimensions. If no
valid null exists, `random_site_null_available=false` is emitted with a reason, and
claim gates are capped below controlled evidence. This is a feasibility limit, not
automatically an implementation failure.

This matters for E004: `operator_probs` is low-dimensional probability data, while
operator outputs are `d_model` activations. The suite must not compare those by shape
coercion.

## Matched Controls

Canonical Tier-1 controls are encoded in `src/attention_lab/mechanisms/presets.py`:

```text
E003 differential -> standard_refactor_control_30m_seed1_rung500
E004 operator_valued -> standard_refactor_control_30m_seed2_rung500
```

Use `--control-mode matched` for canonical resolution. The override flags are:

```text
--control-checkpoint
--control-config
--force-noncanonical-control
```

Overrides are recorded in artifacts. Noncanonical or seed-mismatched controls cap
claims below `candidate_mechanism_evidence`, even with the force flag.

No claim gate can pass without matched control evidence.

## Statistics

The suite computes bootstrap confidence intervals and applies Benjamini-Hochberg FDR
correction. The FDR-BH comparison family is the full run-level family:

```text
every computed site x layer x task_family x metric cell in the run
```

This includes primary probe metrics, shuffled-null contrasts, random-site-null
contrasts, matched-control contrasts, target-vs-decoy specificity metrics, and
restoration/mediation metrics when present. Do not narrow FDR-BH to only the
pre-registered target site or only the primary metric unless the run computes no other
tested cells.

The primary statistical gate requires a corrected decision and a bootstrap CI excluding
the null in the expected direction. Target-vs-decoy specificity uses a bootstrap CI on:

```text
target_effect - decoy_effect
```

and is included in the same FDR-BH scope.

## Patching And Restoration

Patching/restoration is skipped in `--probe-only` mode.

For full runs, restoration must use real logit-difference metadata. The exact formula is:

```text
restoration_score =
  (patched_logitdiff - corrupted_logitdiff)
  / (clean_logitdiff - corrupted_logitdiff)
```

The suite also records:

```text
component_patch_restoration
full_layer_patch_restoration
mediation_fraction =
  component_patch_restoration / full_layer_patch_restoration
```

If a denominator is missing, invalid, or too small, the metric is invalid and dependent
gates cannot pass. Discrete route/index sites are not continuous activations and must
not receive continuous patching.

## Alignment To Control

The alignment metric compares the candidate linear-probe direction against its own
matched-control probe direction:

```text
probe_direction_cosine_to_control
probe_direction_alignment_abs
```

If dimensions differ, alignment is unavailable with a clear reason. The suite does not
pad, truncate, project, or coerce directions.

Interpretation:

```text
high alignment -> likely same/universal feature surfaced differently
low alignment  -> possible different decomposition, requiring more scrutiny
```

Low alignment is not representational novelty evidence by itself.

## Claim Gates

The suite uses a mechanism-probe-scoped ladder:

```text
insufficient_evidence
exploratory_probe_signal
controlled_probe_signal
candidate_mechanism_evidence
```

This vocabulary is distinct from the repository-wide experiment status vocabulary.

`exploratory_probe_signal` is allowed for exploratory/probe-only scans that produce
real trained probe metrics. It is not confirmatory evidence.

`controlled_probe_signal` requires trained probe signal, shuffled-label null,
random-site null, matched control, canonical control pairing, minimum N, grouped split
discipline, bootstrap CI/FDR-BH pass, and decoy specificity.

`candidate_mechanism_evidence` additionally requires a full non-probe-only run, valid
causal patch/restoration metrics, valid mediation fraction, hypothesis doc, canonical
matched controls, and confirmatory task-suite floor.

`candidate_mechanism_evidence` means:

```text
single-seed, checkpoint-backed, statistically controlled evidence
```

It does not mean a replicated finding.

## Cheap-First Staging

Exploratory cheap scan:

```bash
uv run scripts/run_mechanism_probe_suite.py \
  --experiment-id E003_qkv_architecture_gauntlet \
  --candidate differential \
  --checkpoint runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt \
  --task-file <task-file> \
  --output-dir reports/mechanisms/probes/E003_differential_probe_only_inventory_path \
  --exploratory \
  --probe-only \
  --control-mode matched \
  --min-n 100 \
  --bootstrap-samples 1000 \
  --fdr-alpha 0.05 \
  --seed 1 \
  --device cuda
```

Confirmatory full E003 pattern:

```bash
uv run scripts/run_mechanism_probe_suite.py \
  --experiment-id E003_qkv_architecture_gauntlet \
  --candidate differential \
  --checkpoint runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt \
  --task-file <task-file> \
  --hypothesis-doc docs/mechanisms/hypotheses/<hypothesis-file>.yaml \
  --output-dir reports/mechanisms/probes/E003_differential_confirmatory_inventory_path \
  --control-mode matched \
  --min-n 100 \
  --bootstrap-samples 1000 \
  --fdr-alpha 0.05 \
  --seed 1 \
  --device cuda
```

Exploratory E004 pattern:

```bash
uv run scripts/run_mechanism_probe_suite.py \
  --experiment-id E004_operator_binding_qkv_gauntlet \
  --candidate operator_valued \
  --checkpoint runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt \
  --task-file <task-file> \
  --output-dir reports/mechanisms/probes/E004_operator_valued_probe_only_inventory_path \
  --exploratory \
  --probe-only \
  --control-mode matched \
  --min-n 100 \
  --bootstrap-samples 1000 \
  --fdr-alpha 0.05 \
  --seed 2 \
  --device cuda
```

The E004 command resolves the seed2 standard-refactor control by default.

## Disallowed Claims

Do not claim any of the following from Tier-1 alone:

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

## Deferred Stage-2

SAE feature-purity and polysemanticity analysis is deferred. Stage-2 should test
whether flagged stream features fire cleanly on the target phenomenon and not on
sentiment, frequency, topic, or decoy confounds. It should only be built after Tier-1
clears `candidate_mechanism_evidence`.

## Deferred Tier-1.5 Universality

The current alignment metric compares a candidate against its own matched control. It
does not answer whether E003 and E004 converge on the same feature direction.

A future cross-architecture universality analysis should compare flagged E003 and E004
directions against each other using probe-direction alignment and/or CKA, then repeat
the comparison across seeds once replicated checkpoints exist.

## Replication Limitation

Current Tier-1 E003/E004 candidates are single-seed unless replicated checkpoints are
explicitly present. A second seed per Tier-1 architecture is the natural next
investment after Tier-1 clears gates. Future replication should include cross-seed
alignment or CKA-style representational comparison.
