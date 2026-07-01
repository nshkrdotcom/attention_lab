# E004 Results Template

Status: prepared_not_run.

## Gauntlet Summary

Fill from:

```text
reports/experiments/E004_operator_binding_qkv_gauntlet/gauntlet_report.json
reports/experiments/E004_operator_binding_qkv_gauntlet/gauntlet_report.md
```

## Candidate Decisions

| Candidate | Final rung | Decision | Reason | Next action |
| --- | --- | --- | --- | --- |
| `operator_valued_attention_30m_seed2` | not_run | not_run | no artifacts yet | run gauntlet |
| `q3k3v3_role_routed_attention_30m_seed2` | not_run | not_run | no artifacts yet | run gauntlet |
| `dynamic_value_query_conditioned_attention_30m_seed2` | not_run | not_run | no artifacts yet | run gauntlet |

## Required Report Fields

E004 gauntlet reports should include common policy fields plus variant diagnostic summaries:

```text
experiment_id
created_at
policy_path
control_run_name
candidate
rung
attention_type
status
promotion_recommendation
machine_decision
decision_reason
final_val_loss
loss_ratio_vs_control
median_tokens_per_sec
speed_ratio_vs_control
peak_vram_allocated_mb
vram_ratio_vs_control
mechanism_check_name
mechanism_active
diagnostics_non_degenerate
next_action
```

For `operator_valued_attention`:

```text
operator_prob_entropy_mean
operator_prob_add_mean
operator_prob_suppress_mean
operator_prob_gate_mean
operator_prob_transform_mean
operator_prob_bind_mean
operator_combined_output_norm_max
```

For `q3k3v3_role_routed_attention`:

```text
q3_content_output_norm_max
q3_operator_output_norm_max
q3_binding_output_norm_max
q3_content_operator_interaction_norm_max
q3_content_binding_interaction_norm_max
q3_operator_binding_interaction_norm_max
```

For `dynamic_value_query_conditioned_attention`:

```text
dynamic_value_gate_mean
dynamic_value_gate_std
dynamic_value_delta_norm_max
dynamic_value_delta_to_static_ratio_max
```

## Claim Boundary

Do not add scientific claims until actual train/eval/summary/verify artifacts support them. Passing a gauntlet screen is not a model-quality claim.
