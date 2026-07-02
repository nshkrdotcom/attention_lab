# E004_operator_binding_qkv_gauntlet Gauntlet Report

Created: 2026-07-01T23:55:28+00:00
Policy: `configs/experiments/E004_operator_binding_qkv_gauntlet/gauntlet_policy.yaml`
Control: `standard_refactor_control_30m_seed2`

| Candidate | Rung | Attention | Decision | Reason | Next |
| --- | --- | --- | --- | --- | --- |
| standard_refactor_control_30m_seed2_rung020 | rung020 | standard | advance | passed gauntlet policy | queue_rung150 |
| standard_refactor_control_30m_seed2_rung150 | rung150 | standard | advance | passed gauntlet policy | queue_rung500 |
| standard_refactor_control_30m_seed2_rung500 | rung500 | standard | advance | passed gauntlet policy | manual_full_row_required |
| operator_valued_attention_30m_seed2_rung020 | rung020 | operator_valued_attention | advance | passed gauntlet policy | queue_rung150 |
| operator_valued_attention_30m_seed2_rung150 | rung150 | operator_valued_attention | advance | passed gauntlet policy | queue_rung500 |
| operator_valued_attention_30m_seed2_rung500 | rung500 | operator_valued_attention | advance | passed gauntlet policy | manual_full_row_required |
| q3k3v3_role_routed_attention_30m_seed2_rung020 | rung020 | q3k3v3_role_routed_attention | needs_investigation | promotion report recommendation is 'needs_investigation'; screen throughput was below the configured baseline threshold | needs_investigation |
| dynamic_value_query_conditioned_attention_30m_seed2_rung020 | rung020 | dynamic_value_query_conditioned_attention | advance | passed gauntlet policy | queue_rung150 |
| dynamic_value_query_conditioned_attention_30m_seed2_rung150 | rung150 | dynamic_value_query_conditioned_attention | advance | passed gauntlet policy | queue_rung500 |
| dynamic_value_query_conditioned_attention_30m_seed2_rung500 | rung500 | dynamic_value_query_conditioned_attention | killed | mechanism diagnostics are missing or degenerate; promotion report recommendation is 'kill'; mechanism diagnostics were degenerate | killed |
