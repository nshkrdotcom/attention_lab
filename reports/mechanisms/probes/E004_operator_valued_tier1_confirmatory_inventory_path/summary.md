# Tier-1 Mechanism Probe Suite Summary

This report uses a mechanism-probe-specific claim ladder, distinct from the repository-wide experiment status vocabulary.

## Run
- experiment_id: `E004_operator_binding_qkv_gauntlet`
- candidate: `operator_valued`
- checkpoint: `runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt`
- task_file: `configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml`
- mode: `confirmatory`
- probe_only: `False`
- overall_mechanism_probe_status: `insufficient_evidence`

## Control
- expected_control_checkpoint: `runs/screen/standard_refactor_control_30m_seed2_rung500_3cc31db15c20/checkpoints/ckpt_last.pt`
- actual_control_checkpoint: `runs/screen/standard_refactor_control_30m_seed2_rung500_3cc31db15c20/checkpoints/ckpt_last.pt`
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
- tested_cells: `['operator_add_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_add_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_add_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_add_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_add_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_add_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_add_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_bind_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_bind_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_bind_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_bind_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_bind_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_bind_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_bind_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_combined_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_combined_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_combined_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_combined_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_combined_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_combined_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_combined_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_gate_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_gate_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_gate_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_gate_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_gate_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_gate_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_gate_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_probs[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_probs[0]|family=negation_scope|metric=component_patch_restoration', 'operator_probs[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_probs[0]|family=negation_scope|metric=mediation_fraction', 'operator_probs[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_suppress_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_suppress_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_suppress_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_suppress_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_suppress_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_suppress_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_suppress_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_transform_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_transform_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_transform_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_transform_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_transform_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_transform_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_transform_out[0]|family=negation_scope|metric=target_vs_decoy_specificity']`

## Site Results
### `operator_add_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.28`
- auc_minus_random_site_auc: `0.008888888888888835`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `mlp_out[0]`
- random_site_reason: none
- patching_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `-0.28797898661139304`
- probe_direction_cosine_to_control: `-0.053988322463357455`
- alignment_available: `True`

### `operator_bind_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.6977777777777778`
- auc_minus_random_site_auc: `0.0`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `attn_out[0]`
- random_site_reason: none
- patching_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `0.5746968726904086`
- probe_direction_cosine_to_control: `-0.022734384623603884`
- alignment_available: `True`

### `operator_combined_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.5377777777777778`
- auc_minus_random_site_auc: `0.008888888888888835`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `mlp_out[0]`
- random_site_reason: none
- patching_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `1.0`
- probe_direction_cosine_to_control: `-0.07358546085322445`
- alignment_available: `True`

### `operator_gate_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.5111111111111111`
- auc_minus_random_site_auc: `0.0`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `resid_mid[0]`
- random_site_reason: none
- patching_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `0.0531309478949411`
- probe_direction_cosine_to_control: `-0.11214721172182981`
- alignment_available: `True`

### `operator_probs[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- blockers: `['shuffled-label null comparison failed', 'random-site null unavailable for this site-layer cell', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- linear_probe_auc: `0.92`
- auc_minus_shuffled_auc: `-0.03555555555555556`
- auc_minus_random_site_auc: `None`
- auc_minus_matched_control_auc: `None`
- target_vs_decoy_specificity: `-0.008888888888888835`
- random_site_status: `unavailable_no_compatible_matched_dimensionality_site`
- random_site_null_available: `False`
- selected_random_site: `None`
- random_site_reason: no non-candidate random-site null with matched dimensionality and compatible site type; this is a null-feasibility limit, not an implementation failure
- patching_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `0.15205910942072132`
- probe_direction_cosine_to_control: `None`
- alignment_available: `False`

### `operator_suppress_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.6177777777777778`
- auc_minus_random_site_auc: `0.0`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `attn_out[0]`
- random_site_reason: none
- patching_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `0.12395478982335278`
- probe_direction_cosine_to_control: `-0.028658327861959824`
- alignment_available: `True`

### `operator_transform_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.48444444444444446`
- auc_minus_random_site_auc: `0.004444444444444473`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `resid_post[0]`
- random_site_reason: none
- patching_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `0.9812761398860664`
- probe_direction_cosine_to_control: `-0.005041233938798633`
- alignment_available: `True`

## Limitations
- Exploratory runs and probe-only runs cannot support confirmatory mechanism claims.
- Missing matched controls, missing decoys, missing random-site nulls, invalid statistics, or noncanonical controls cap the affected claim gates.
- Missing random-site nulls are feasibility limits for the affected `(site x layer)` cell, not automatic implementation failures and not a run-wide cap.
- Candidate-to-control alignment is not cross-architecture universality evidence.
- Low alignment is not representational novelty evidence by itself.
- This Tier-1 status is single-seed, checkpoint-backed, statistically controlled evidence when gates pass; it is not a replicated finding.
- This report is not evidence that the mechanism is universal, replicated, solved a task family, lowered superposition, or proves a causal mechanism in general.
