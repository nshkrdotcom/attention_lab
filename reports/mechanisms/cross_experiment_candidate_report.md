# Cross-Experiment Mechanism Candidate Report

Generated from structured backfill inventories. It is not a training-result claim.
Inventory source commits: e2582bb208bf1c795035ab49b9fc553d4ee8ff14.

## Promote Full Mechanism Run (`promote_full_mechanism_run`)
- `differential_qkv_anti_value_30m_seed1_rung500` (E003_qkv_architecture_gauntlet, `differential_qkv_anti_value`): evidence=checkpoint_recompute; checkpoint=available; next=run matched full mechanism probe from checkpoint before full promotion
- `scope_gated_qkv_30m_seed1_rung500` (E003_qkv_architecture_gauntlet, `scope_gated_qkv`): evidence=checkpoint_recompute; checkpoint=available; next=run matched full mechanism probe from checkpoint before full promotion
- `operator_valued_attention_30m_seed2_rung500` (E004_operator_binding_qkv_gauntlet, `operator_valued_attention`): evidence=checkpoint_recompute; checkpoint=available; next=run matched full mechanism probe from checkpoint before full promotion

## Diagnostic Rescue (`diagnostic_rescue`)
- `dynamic_value_query_conditioned_attention_30m_seed2_rung500` (E004_operator_binding_qkv_gauntlet, `dynamic_value_query_conditioned_attention`): evidence=checkpoint_recompute; checkpoint=available; next=run gate/delta post-hoc probe and causal ablation

## Profiling Redesign (`profiling_redesign`)
- `q3k3v3_role_routed_attention_30m_seed2_rung020` (E004_operator_binding_qkv_gauntlet, `q3k3v3_role_routed_attention`): evidence=checkpoint_recompute; checkpoint=available; next=profile low-batch role streams and redesign throughput

## Route Specialization Workbench (`route_specialization_workbench`)
- `multi_qkv_position_rotation_3track_global_30m_seed1` (E002_multitrack_qkv_shift_register, `multi_qkv_position_rotation_3track_global`): evidence=checkpoint_recompute; checkpoint=available; next=run route replacement and track ablation matrix
- `multi_qkv_static_3track_global_30m_seed1` (E002_multitrack_qkv_shift_register, `multi_qkv_static_3track_global`): evidence=checkpoint_recompute; checkpoint=available; next=run route replacement and track ablation matrix

## Cp Diagnostic Followup (`cp_diagnostic_followup`)
- `cp_bilinear_r8_30m_seed1` (E001_cp_trilinear_attention, `cp_bilinear`): evidence=checkpoint_recompute; checkpoint=available; next=run lambda/null and CP contribution probes
- `cp_trilinear_r8_30m_seed1` (E001_cp_trilinear_attention, `cp_trilinear`): evidence=checkpoint_recompute; checkpoint=available; next=run lambda/null and CP contribution probes

## Unsupported Or Incomplete (`unsupported_or_incomplete`)
- `standard_30m_seed1` (E001_cp_trilinear_attention, `standard`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `multi_qkv_train_rotation_3track_global_30m_seed1` (E002_multitrack_qkv_shift_register, `multi_qkv_train_rotation_3track_global`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `standard_refactor_control_30m_seed1` (E002_multitrack_qkv_shift_register, `standard`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `differential_qkv_anti_value_30m_seed1_rung020` (E003_qkv_architecture_gauntlet, `differential_qkv_anti_value`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `differential_qkv_anti_value_30m_seed1_rung150` (E003_qkv_architecture_gauntlet, `differential_qkv_anti_value`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `scope_gated_qkv_30m_seed1_rung020` (E003_qkv_architecture_gauntlet, `scope_gated_qkv`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `scope_gated_qkv_30m_seed1_rung150` (E003_qkv_architecture_gauntlet, `scope_gated_qkv`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `standard_refactor_control_30m_seed1_rung020` (E003_qkv_architecture_gauntlet, `standard`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `standard_refactor_control_30m_seed1_rung150` (E003_qkv_architecture_gauntlet, `standard`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `standard_refactor_control_30m_seed1_rung500` (E003_qkv_architecture_gauntlet, `standard`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `dynamic_value_query_conditioned_attention_30m_seed2_rung020` (E004_operator_binding_qkv_gauntlet, `dynamic_value_query_conditioned_attention`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `dynamic_value_query_conditioned_attention_30m_seed2_rung150` (E004_operator_binding_qkv_gauntlet, `dynamic_value_query_conditioned_attention`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `operator_valued_attention_30m_seed2_rung020` (E004_operator_binding_qkv_gauntlet, `operator_valued_attention`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `operator_valued_attention_30m_seed2_rung150` (E004_operator_binding_qkv_gauntlet, `operator_valued_attention`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `standard_refactor_control_30m_seed2_rung020` (E004_operator_binding_qkv_gauntlet, `standard`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `standard_refactor_control_30m_seed2_rung150` (E004_operator_binding_qkv_gauntlet, `standard`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts
- `standard_refactor_control_30m_seed2_rung500` (E004_operator_binding_qkv_gauntlet, `standard`): evidence=checkpoint_recompute; checkpoint=available; next=complete missing checkpoint, diagnostics, or report artifacts

## Not Evaluated (`not_evaluated`)
- `cp_trilinear_r8_lambda0_30m_seed1` (E001_cp_trilinear_attention, `cp_trilinear`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `standard_refactor_control_30m_seed1` (E001_cp_trilinear_attention, `standard`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `multi_qkv_layer_shift_3track_30m_seed1` (E002_multitrack_qkv_shift_register, `multi_qkv_layer_shift`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `multi_qkv_softmix_3track_30m_seed1` (E002_multitrack_qkv_shift_register, `multi_qkv_softmix`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `multi_qkv_static_3track_30m_seed1` (E002_multitrack_qkv_shift_register, `multi_qkv_static`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `multi_qkv_train_and_layer_shift_3track_30m_seed1` (E002_multitrack_qkv_shift_register, `multi_qkv_train_and_layer_shift`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `multi_qkv_train_shift_3track_30m_seed1` (E002_multitrack_qkv_shift_register, `multi_qkv_train_shift`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `multi_qkv_train_shift_warmup_3track_30m_seed1` (E002_multitrack_qkv_shift_register, `multi_qkv_train_shift_warmup`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `standard_30m_seed1` (E002_multitrack_qkv_shift_register, `standard`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `differential_qkv_anti_value_30m_seed1` (E003_qkv_architecture_gauntlet, `differential_qkv_anti_value`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `scope_gated_qkv_30m_seed1` (E003_qkv_architecture_gauntlet, `scope_gated_qkv`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `standard_refactor_control_30m_seed1` (E003_qkv_architecture_gauntlet, `standard`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `dynamic_value_query_conditioned_attention_30m_seed2` (E004_operator_binding_qkv_gauntlet, `dynamic_value_query_conditioned_attention`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `operator_valued_attention_30m_seed2` (E004_operator_binding_qkv_gauntlet, `operator_valued_attention`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `q3k3v3_role_routed_attention_30m_seed2` (E004_operator_binding_qkv_gauntlet, `q3k3v3_role_routed_attention`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist
- `standard_refactor_control_30m_seed2` (E004_operator_binding_qkv_gauntlet, `standard`): evidence=not_available; checkpoint=checkpoint_unavailable; next=do not classify scientifically until artifacts exist

## What Cannot Be Concluded
- Survival-screen pass does not establish semantic mechanism roles.
- Missing historical activations cannot be reconstructed without saved tensors.
- Checkpoint availability only means post-hoc recomputation is possible.
- Validation-loss differences are not architecture evidence without matched controls and diagnostics.
