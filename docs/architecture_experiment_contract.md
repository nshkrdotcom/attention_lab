# Architecture Experiment Contract

This repository is the substrate for attention architecture experiments. New attention
modules must compare against the standard-attention baseline without changing unrelated
training conditions.

## Fixed Inputs

All architecture variants must use the same:

- Data manifest and shard hashes.
- Tokenizer.
- Train and validation shards.
- Batch construction and data-order policy.
- Token budget and gradient accumulation policy.
- Optimizer and optimizer hyperparameters.
- Learning-rate schedule.
- Seed policy.
- Validation and sample cadence.
- Checkpoint, eval, summary, and run verification scripts.

If a variant needs a different setting, that run is not a direct architecture
comparison. Record it as an ablation or engineering experiment.

## Required Metrics

Every architecture run must report:

- Parameter count and parameter delta versus the matched baseline.
- Final validation loss.
- Best validation loss.
- Final validation perplexity.
- Median tokens/sec.
- Peak allocated and reserved VRAM.
- Wall-clock runtime.
- Checkpoint reload eval loss.
- Bounded HellaSwag result when requested.
- Run verifier result.

The machine-readable run summary format is documented in
`reports/schema/run_summary.schema.json`.

## First CP Experiment Preconditions

Do not start CP attention experiments until these runs exist and verify:

- Standard-attention baseline.
- Standard-attention refactor/control run if the model path changes.
- CP bilinear candidate.
- CP trilinear candidate.

The CP modules are intentionally not implemented in this hardening pass.

## Mechanism Investigation Addendum

Survival screens and full-run comparisons are not mechanism explanations. Mechanism investigation now uses the native substrate under `src/attention_lab/mechanisms/`:

- Hook sites are declared in a registry and include standard GPT sites plus architecture-specific sites for E001-E004 mechanisms.
- Activation capture returns `ActivationCache` records from real forward passes.
- Interventions support zero, mean ablation, replacement, scaling, and cache patching at named sites.
- Backfill inventories distinguish `artifact_summary`, `checkpoint_recompute`, and `not_available`.
- Backfill inventories record deterministic generation provenance: `generated_from_commit` and `repo_root_relative`.
- Positive cross-experiment classifications require usable evidence. Rows with `evidence_level: not_available` remain `not_evaluated` even if their attention type matches a known follow-up family.

Do not claim that a historical run has recoverable activations unless tensors were actually captured or can be recomputed from a checkpoint and a specified prompt/eval batch. Historical run directories remain read-only evidence inputs; derived mechanism artifacts belong under `reports/mechanisms/`.

Post-hoc probes must use the tokenizer declared by the config. The current probe CLI supports GPT-2 tokenization and fails explicitly for unsupported tokenizers or prompt token IDs outside the configured vocabulary. Probe outputs must record tokenizer metadata.

Capture-only instrumentation is part of the live forward path. New hook support must include no-op capture equivalence tests showing that logits are unchanged when capture is enabled and no interventions are applied. If a declared site cannot be emitted for a config or remains runtime-unsupported, `capture_activations(..., require_declared_sites=True)` must report it instead of fabricating tensors.
