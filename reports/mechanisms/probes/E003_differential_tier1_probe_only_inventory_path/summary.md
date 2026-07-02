# Tier-1 Mechanism Probe Suite Summary

This report uses a mechanism-probe-specific claim ladder, distinct from the repository-wide experiment status vocabulary.

## Run
- experiment_id: `E003_qkv_architecture_gauntlet`
- candidate: `differential`
- checkpoint: `runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt`
- task_file: `configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml`
- mode: `exploratory`
- probe_only: `True`
- overall_mechanism_probe_status: `exploratory_probe_signal`

## Control
- expected_control_checkpoint: `runs/screen/standard_refactor_control_30m_seed1_rung500_7752266a764e/checkpoints/ckpt_last.pt`
- actual_control_checkpoint: `runs/screen/standard_refactor_control_30m_seed1_rung500_7752266a764e/checkpoints/ckpt_last.pt`
- canonical_control: `True`
- override_used: `False`
- control_available: `True`
- reason: none

## Task Suite
- deterministic_provenance: `True`
- confirmatory_floor_met: `True`
- restoration_token_metadata_valid: `True`
- pair_counts_by_family: `{'negation_scope': 50}`
- validation_errors: `[]`
- validation_warnings: `[]`

## FDR-BH
- comparison_family: every computed `(site x layer x task_family x metric)` cell in this run, including probe, null, matched-control, specificity, restoration, and mediation metrics when present.
- alpha: `0.05`
- tested_cells: `['branch_delta[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'branch_delta[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'branch_delta[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'branch_delta[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'branch_delta[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'neg_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'neg_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'neg_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'neg_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'neg_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'pos_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'pos_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'pos_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'pos_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'pos_out[0]|family=negation_scope|metric=target_vs_decoy_specificity']`

## Site Results
### `branch_delta[0]|family=negation_scope`
- claim_gate: `exploratory_probe_signal`
- blockers: `[]`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.4444444444444444`
- auc_minus_random_site_auc: `0.0`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available`
- random_site_null_available: `True`
- selected_random_site: `mlp_out[0]`
- random_site_reason: none
- patching_valid: `False`
- patching_reason: probe-only mode
- mediation_fraction_valid: `False`
- mediation_fraction: `None`
- probe_direction_cosine_to_control: `0.024503561156061116`
- alignment_available: `True`

### `neg_out[0]|family=negation_scope`
- claim_gate: `exploratory_probe_signal`
- blockers: `[]`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.6`
- auc_minus_random_site_auc: `0.0`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available`
- random_site_null_available: `True`
- selected_random_site: `resid_mid[0]`
- random_site_reason: none
- patching_valid: `False`
- patching_reason: probe-only mode
- mediation_fraction_valid: `False`
- mediation_fraction: `None`
- probe_direction_cosine_to_control: `-0.06687405850945626`
- alignment_available: `True`

### `pos_out[0]|family=negation_scope`
- claim_gate: `exploratory_probe_signal`
- blockers: `[]`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.47111111111111115`
- auc_minus_random_site_auc: `0.0`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available`
- random_site_null_available: `True`
- selected_random_site: `mlp_out[0]`
- random_site_reason: none
- patching_valid: `False`
- patching_reason: probe-only mode
- mediation_fraction_valid: `False`
- mediation_fraction: `None`
- probe_direction_cosine_to_control: `-0.0023994484502870973`
- alignment_available: `True`

## Limitations
- Exploratory runs and probe-only runs cannot support confirmatory mechanism claims.
- Missing matched controls, missing decoys, missing random-site nulls, invalid statistics, or noncanonical controls cap the affected claim gates.
- Missing random-site nulls are feasibility limits for the affected `(site x layer)` cell, not automatic implementation failures and not a run-wide cap.
- Candidate-to-control alignment is not cross-architecture universality evidence.
- Low alignment is not representational novelty evidence by itself.
- This Tier-1 status is single-seed, checkpoint-backed, statistically controlled evidence when gates pass; it is not a replicated finding.
- This report is not evidence that the mechanism is universal, replicated, solved a task family, lowered superposition, or proves a causal mechanism in general.

Probe-only mode skipped interventions, causal patching, restoration, and mediation metrics.
Exploratory mode capped the claim ladder below confirmatory evidence.
