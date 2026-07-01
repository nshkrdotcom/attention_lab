# E003_qkv_architecture_gauntlet Gauntlet Report

Created: 2026-07-01T20:48:38+00:00
Policy: `configs/experiments/E003_qkv_architecture_gauntlet/gauntlet_policy.yaml`
Control: `standard_refactor_control_30m_seed1`

| Candidate | Rung | Attention | Decision | Reason | Next |
| --- | --- | --- | --- | --- | --- |
| standard_refactor_control_30m_seed1_rung020 | rung020 | standard | advance | passed gauntlet policy | queue_rung150 |
| standard_refactor_control_30m_seed1_rung150 | rung150 | standard | advance | passed gauntlet policy | queue_rung500 |
| standard_refactor_control_30m_seed1_rung500 | rung500 | standard | advance | passed gauntlet policy | manual_full_row_required |
| differential_qkv_anti_value_30m_seed1_rung020 | rung020 | differential_qkv_anti_value | advance | passed gauntlet policy | queue_rung150 |
| differential_qkv_anti_value_30m_seed1_rung150 | rung150 | differential_qkv_anti_value | advance | passed gauntlet policy | queue_rung500 |
| differential_qkv_anti_value_30m_seed1_rung500 | rung500 | differential_qkv_anti_value | advance | passed gauntlet policy | manual_full_row_required |
| scope_gated_qkv_30m_seed1_rung020 | rung020 | scope_gated_qkv | advance | passed gauntlet policy | queue_rung150 |
| scope_gated_qkv_30m_seed1_rung150 | rung150 | scope_gated_qkv | advance | passed gauntlet policy | queue_rung500 |
| scope_gated_qkv_30m_seed1_rung500 | rung500 | scope_gated_qkv | advance | passed gauntlet policy | manual_full_row_required |
