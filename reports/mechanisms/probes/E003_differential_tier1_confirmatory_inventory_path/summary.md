# Tier-1 Mechanism Probe Suite Summary

This report uses a mechanism-probe-specific claim ladder, distinct from the repository-wide experiment status vocabulary.
An exploratory signal is not a passed confirmatory claim gate.

## Run
- experiment_id: `E003_qkv_architecture_gauntlet`
- candidate: `differential`
- checkpoint: `runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt`
- task_file: `configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml`
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
- expected_control_checkpoint: `runs/screen/standard_refactor_control_30m_seed1_rung500_7752266a764e/checkpoints/ckpt_last.pt`
- actual_control_checkpoint: `runs/screen/standard_refactor_control_30m_seed1_rung500_7752266a764e/checkpoints/ckpt_last.pt`
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
- tested_cells: `['branch_delta[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'branch_delta[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'branch_delta[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'branch_delta[0]|family=negation_scope|metric=component_patch_restoration', 'branch_delta[0]|family=negation_scope|metric=full_layer_patch_restoration', 'branch_delta[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'branch_delta[0]|family=negation_scope|metric=mediation_fraction', 'branch_delta[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'neg_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'neg_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'neg_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'neg_out[0]|family=negation_scope|metric=component_patch_restoration', 'neg_out[0]|family=negation_scope|metric=full_layer_patch_restoration', 'neg_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'neg_out[0]|family=negation_scope|metric=mediation_fraction', 'neg_out[0]|family=negation_scope|metric=target_vs_decoy_specificity', 'pos_out[0]|family=negation_scope|metric=auc_minus_matched_control_auc', 'pos_out[0]|family=negation_scope|metric=auc_minus_random_site_auc', 'pos_out[0]|family=negation_scope|metric=auc_minus_shuffled_auc', 'pos_out[0]|family=negation_scope|metric=component_patch_restoration', 'pos_out[0]|family=negation_scope|metric=full_layer_patch_restoration', 'pos_out[0]|family=negation_scope|metric=linear_probe_auc_minus_0_5', 'pos_out[0]|family=negation_scope|metric=mediation_fraction', 'pos_out[0]|family=negation_scope|metric=target_vs_decoy_specificity']`
- invalid_or_unavailable_cells: `[]`

## Random-Site Null Pool
- scope: `complete preset-declared Tier-1 random-site null family`
- selection_policy: same-layer non-candidate sites declared in the preset; selection still requires actual captured feature dimensionality and compatible tensor kind
- declared_sites: `['attn_out[0]', 'resid_mid[0]', 'mlp_out[0]', 'resid_post[0]']`

## Site Results
### `branch_delta[0]|family=negation_scope`
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
- auc_minus_shuffled_auc: `0.36444444444444446`
- auc_minus_random_site_auc: `0.0`
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
- mediation_fraction: `1.0000002699649644`
- probe_direction_cosine_to_control: `0.048047945607904964`
- alignment_available: `True`

### `neg_out[0]|family=negation_scope`
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
- auc_minus_shuffled_auc: `0.5155555555555555`
- auc_minus_random_site_auc: `0.07111111111111112`
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
- mediation_fraction: `0.6225063870527899`
- probe_direction_cosine_to_control: `0.040785393314477836`
- alignment_available: `True`

### `pos_out[0]|family=negation_scope`
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
- auc_minus_shuffled_auc: `0.5822222222222222`
- auc_minus_random_site_auc: `0.0`
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
- mediation_fraction: `0.19220024302372488`
- probe_direction_cosine_to_control: `0.04965359890192786`
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
