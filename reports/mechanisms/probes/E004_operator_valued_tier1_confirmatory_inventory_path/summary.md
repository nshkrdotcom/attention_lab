# Tier-1 Mechanism Probe Suite Summary

This report uses a mechanism-probe-specific claim ladder, distinct from the repository-wide experiment status vocabulary.
An exploratory signal is not a passed confirmatory claim gate.

## Run
- experiment_id: `E004_operator_binding_qkv_gauntlet`
- candidate: `operator_valued`
- checkpoint: `runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt`
- task_file: `configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml`
- mode: `confirmatory`
- probe_only: `False`
- feature_pooling: `patch_positions_mean`
- task_aligned_pooling: `True`
- overall_mechanism_probe_status: `insufficient_evidence`
- overall_gate_aggregation: existential over evaluated cells; true means at least one cell reached the gate, not that every selected site/layer/family passed
- overall_exploratory_signal: `False`
- overall_controlled_probe_gate_passed: `False`
- overall_candidate_mechanism_gate_passed: `False`
- overall_claim_gate_passed: `False`
- claim_gate_passed_semantics: `candidate_mechanism_gate_passed compatibility alias`

## Control
- expected_control_checkpoint: `runs/screen/standard_refactor_control_30m_seed2_rung500_3cc31db15c20/checkpoints/ckpt_last.pt`
- actual_control_checkpoint: `runs/screen/standard_refactor_control_30m_seed2_rung500_3cc31db15c20/checkpoints/ckpt_last.pt`
- canonical_control: `True`
- override_used: `False`
- control_available: `True`
- reason: none

## Task Suite
- deterministic_provenance: `True`
- deterministic_fingerprint_valid: `True`
- deterministic_fingerprint_reason: none
- confirmatory_floor_met: `True`
- restoration_token_metadata_valid: `True`
- pair_counts_by_family: `{'negation_scope': 50}`
- validation_errors: `[]`
- validation_warnings: `[]`

## FDR-BH
- comparison_family: every computed `(site x layer x task_family x metric)` cell in this run, including probe, null, matched-control, specificity, restoration, and mediation metrics when present.
- alpha: `0.05`
- tested_cells: `['operator_add_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_add_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_add_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_add_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_add_out[0]|family=negation_scope|metric=full_layer_patch_restoration', 'operator_add_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_add_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_add_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_bind_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_bind_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_bind_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_bind_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_bind_out[0]|family=negation_scope|metric=full_layer_patch_restoration', 'operator_bind_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_bind_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_bind_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_combined_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_combined_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_combined_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_combined_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_combined_out[0]|family=negation_scope|metric=full_layer_patch_restoration', 'operator_combined_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_combined_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_combined_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_gate_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_gate_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_gate_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_gate_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_gate_out[0]|family=negation_scope|metric=full_layer_patch_restoration', 'operator_gate_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_gate_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_gate_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_probs[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_probs[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_probs[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_suppress_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_suppress_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_suppress_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_suppress_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_suppress_out[0]|family=negation_scope|metric=full_layer_patch_restoration', 'operator_suppress_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_suppress_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_suppress_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'operator_transform_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'operator_transform_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'operator_transform_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'operator_transform_out[0]|family=negation_scope|metric=component_patch_restoration', 'operator_transform_out[0]|family=negation_scope|metric=full_layer_patch_restoration', 'operator_transform_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'operator_transform_out[0]|family=negation_scope|metric=mediation_fraction', 'operator_transform_out[0]|family=negation_scope|metric=target_vs_decoy_specificity']`
- invalid_or_unavailable_cells: `[{'cell_id': 'operator_probs[0]|family=negation_scope', 'metric': 'auc_minus_random_site_auc', 'reason': 'no non-candidate random-site null with matched dimensionality and compatible site type; this is a null-feasibility limit, not an implementation failure', 'site': 'operator_probs[0]', 'family_id': 'negation_scope'}, {'cell_id': 'operator_probs[0]|family=negation_scope', 'metric': 'auc_minus_matched_control_auc', 'reason': 'standard matched control has no operator probability site', 'site': 'operator_probs[0]', 'family_id': 'negation_scope'}, {'cell_id': 'operator_probs[0]|family=negation_scope', 'metric': 'component_patch_restoration', 'reason': 'site metadata marks this site as non-continuous', 'site': 'operator_probs[0]'}, {'cell_id': 'operator_probs[0]|family=negation_scope', 'metric': 'full_layer_patch_restoration', 'reason': 'site metadata marks this site as non-continuous', 'site': 'operator_probs[0]'}, {'cell_id': 'operator_probs[0]|family=negation_scope', 'metric': 'mediation_fraction', 'reason': 'site metadata marks this site as non-continuous', 'site': 'operator_probs[0]'}]`

## Random-Site Null Pool
- scope: `complete preset-declared Tier-1 random-site null family`
- selection_policy: same-layer non-candidate sites declared in the preset; selection still requires actual captured feature dimensionality and compatible tensor kind
- declared_sites: `['attn_out[0]', 'resid_mid[0]', 'mlp_out[0]', 'resid_post[0]']`

## Site Results
### `operator_add_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- exploratory_signal: `False`
- controlled_probe_gate_passed: `False`
- candidate_mechanism_gate_passed: `False`
- highest_status: `insufficient_evidence`
- highest_status_semantics: highest claim ladder threshold reached for this cell
- claim_gate_passed: `False`
- status_kind: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- feature_pooling: `patch_positions_mean`
- task_aligned_pooling: `True`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.3733333333333333`
- auc_minus_random_site_auc: `0.03111111111111109`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `mlp_out[0]`
- random_site_reason: none
- patching_valid: `True`
- restoration_alignment_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `1.4214727428773515`
- probe_direction_cosine_to_control: `0.019558533969486494`
- alignment_available: `True`

### `operator_bind_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- exploratory_signal: `False`
- controlled_probe_gate_passed: `False`
- candidate_mechanism_gate_passed: `False`
- highest_status: `insufficient_evidence`
- highest_status_semantics: highest claim ladder threshold reached for this cell
- claim_gate_passed: `False`
- status_kind: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- feature_pooling: `patch_positions_mean`
- task_aligned_pooling: `True`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.768888888888889`
- auc_minus_random_site_auc: `0.0`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `attn_out[0]`
- random_site_reason: none
- patching_valid: `True`
- restoration_alignment_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `-0.7149608955194895`
- probe_direction_cosine_to_control: `-0.024494275685223516`
- alignment_available: `True`

### `operator_combined_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- exploratory_signal: `False`
- controlled_probe_gate_passed: `False`
- candidate_mechanism_gate_passed: `False`
- highest_status: `insufficient_evidence`
- highest_status_semantics: highest claim ladder threshold reached for this cell
- claim_gate_passed: `False`
- status_kind: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- feature_pooling: `patch_positions_mean`
- task_aligned_pooling: `True`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.45777777777777773`
- auc_minus_random_site_auc: `0.03111111111111109`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.013333333333333308`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `mlp_out[0]`
- random_site_reason: none
- patching_valid: `True`
- restoration_alignment_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `1.0`
- probe_direction_cosine_to_control: `-0.024809684484843372`
- alignment_available: `True`

### `operator_gate_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- exploratory_signal: `False`
- controlled_probe_gate_passed: `False`
- candidate_mechanism_gate_passed: `False`
- highest_status: `insufficient_evidence`
- highest_status_semantics: highest claim ladder threshold reached for this cell
- claim_gate_passed: `False`
- status_kind: `insufficient_evidence`
- blockers: `['matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- feature_pooling: `patch_positions_mean`
- task_aligned_pooling: `True`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.6266666666666667`
- auc_minus_random_site_auc: `0.10222222222222221`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available`
- random_site_null_available: `True`
- selected_random_site: `resid_mid[0]`
- random_site_reason: none
- patching_valid: `True`
- restoration_alignment_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `0.7403431533931224`
- probe_direction_cosine_to_control: `-0.04194655459712951`
- alignment_available: `True`

### `operator_probs[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- exploratory_signal: `False`
- controlled_probe_gate_passed: `False`
- candidate_mechanism_gate_passed: `False`
- highest_status: `insufficient_evidence`
- highest_status_semantics: highest claim ladder threshold reached for this cell
- claim_gate_passed: `False`
- status_kind: `insufficient_evidence`
- blockers: `['shuffled-label null comparison failed', 'random-site null unavailable for this site-layer cell', 'matched-control comparison failed']`
- feature_pooling: `patch_positions_mean`
- task_aligned_pooling: `True`
- linear_probe_auc: `0.7688888888888888`
- auc_minus_shuffled_auc: `-0.21777777777777785`
- auc_minus_random_site_auc: `None`
- auc_minus_matched_control_auc: `None`
- target_vs_decoy_specificity: `0.3866666666666666`
- random_site_status: `unavailable_no_compatible_matched_dimensionality_site`
- random_site_null_available: `False`
- selected_random_site: `None`
- random_site_reason: no non-candidate random-site null with matched dimensionality and compatible site type; this is a null-feasibility limit, not an implementation failure
- patching_valid: `False`
- restoration_alignment_valid: `True`
- patching_reason: site metadata marks this site as non-continuous
- mediation_fraction_valid: `False`
- mediation_fraction: `None`
- probe_direction_cosine_to_control: `None`
- alignment_available: `False`

### `operator_suppress_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- exploratory_signal: `False`
- controlled_probe_gate_passed: `False`
- candidate_mechanism_gate_passed: `False`
- highest_status: `insufficient_evidence`
- highest_status_semantics: highest claim ladder threshold reached for this cell
- claim_gate_passed: `False`
- status_kind: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- feature_pooling: `patch_positions_mean`
- task_aligned_pooling: `True`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.6844444444444444`
- auc_minus_random_site_auc: `0.0`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `attn_out[0]`
- random_site_reason: none
- patching_valid: `True`
- restoration_alignment_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `-0.06496073402831301`
- probe_direction_cosine_to_control: `-0.05307981533431828`
- alignment_available: `True`

### `operator_transform_out[0]|family=negation_scope`
- claim_gate: `insufficient_evidence`
- exploratory_signal: `False`
- controlled_probe_gate_passed: `False`
- candidate_mechanism_gate_passed: `False`
- highest_status: `insufficient_evidence`
- highest_status_semantics: highest claim ladder threshold reached for this cell
- claim_gate_passed: `False`
- status_kind: `insufficient_evidence`
- blockers: `['random-site null comparison failed', 'matched-control comparison failed', 'target-vs-decoy specificity gate failed']`
- feature_pooling: `patch_positions_mean`
- task_aligned_pooling: `True`
- linear_probe_auc: `1.0`
- auc_minus_shuffled_auc: `0.15555555555555556`
- auc_minus_random_site_auc: `0.03555555555555556`
- auc_minus_matched_control_auc: `0.0`
- target_vs_decoy_specificity: `0.0`
- random_site_status: `available_but_candidate_did_not_beat_null_after_correction`
- random_site_null_available: `True`
- selected_random_site: `resid_post[0]`
- random_site_reason: none
- patching_valid: `True`
- restoration_alignment_valid: `True`
- patching_reason: none
- mediation_fraction_valid: `True`
- mediation_fraction: `-0.006860426615086898`
- probe_direction_cosine_to_control: `-0.05528845848351143`
- alignment_available: `True`

## Limitations
- Exploratory runs and probe-only runs cannot support confirmatory mechanism claims.
- Missing matched controls, missing decoys, missing random-site nulls, invalid statistics, or noncanonical controls cap the affected claim gates.
- Missing random-site nulls are feasibility limits for the affected `(site x layer)` cell, not automatic implementation failures and not a run-wide cap.
- Mean-sequence pooling is exploratory/diagnostic for Tier-1 and cannot support `candidate_mechanism_evidence`; confirmatory claims require task-aligned pooling.
- FDR-BH reports both tested metric cells and invalid/unavailable cells with reasons; unavailable cells are not assigned meaningful p-values.
- Candidate-to-control alignment is not cross-architecture universality evidence.
- Low alignment is not representational novelty evidence by itself.
- This Tier-1 status is single-seed, checkpoint-backed, statistically controlled evidence when gates pass; it is not a replicated finding.
- This report is not evidence that the mechanism is universal, replicated, solved a task family, lowered superposition, or proves a causal mechanism in general.
