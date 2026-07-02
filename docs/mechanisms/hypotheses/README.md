# Mechanism Hypothesis Docs

Tier-1 confirmatory mechanism probe suites require YAML hypothesis docs in this directory.

Required fields:

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

Do not place confirmatory mechanism-probe hypotheses in a second directory or markdown convention.

Committed Tier-1 hypothesis docs:

```text
E003_differential_negation_tier1.yaml
E004_operator_valued_negation_tier1.yaml
```

They pair with deterministic task suites under:

```text
configs/mechanisms/tier1_tasks/
```

Confirmatory task suites must include GPT-2 single-token target/foil metadata plus clean/corrupt
answer positions, patch-token indices, and `clean_corrupt_token_alignment`. If clean and corrupted
prompts differ in token length, explicit clean and corrupted patch indices are required.

Validate the task suites with:

```bash
uv run scripts/generate_tier1_mechanism_tasks.py \
  --output configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml \
  --candidate e003_differential \
  --pairs-per-family 50 \
  --validate-only
```

The hypothesis docs intentionally do not claim replication, universality, architecture superiority,
solved negation, lower superposition, or representational novelty. Noncanonical controls, missing
matched controls, missing random-site nulls for an affected cell, invalid restoration metadata, and
exploratory/probe-only mode cap the mechanism-probe claim ladder. Confirmatory candidate evidence
also requires task-aligned pooling and FDR-BH pass for component restoration, full-layer restoration,
and mediation_fraction.
