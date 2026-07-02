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

Do not claim that a historical run has recoverable activations unless tensors were actually captured or can be recomputed from a checkpoint and a specified prompt/eval batch. Historical run directories remain read-only evidence inputs; derived mechanism artifacts belong under `reports/mechanisms/`.
