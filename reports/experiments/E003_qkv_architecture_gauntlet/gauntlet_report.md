# E003_qkv_architecture_gauntlet Gauntlet Summary

Created: 2026-07-01T20:48:38+00:00
Policy: `configs/experiments/E003_qkv_architecture_gauntlet/gauntlet_policy.yaml`
Control: `standard_refactor_control_30m_seed1`

| Candidate | Rung | Decision | Final val loss | Mechanism active | Next action |
| --- | --- | --- | ---: | --- | --- |
| standard_refactor_control_30m_seed1_rung020 | rung020 | advance | 9.703710556030273 | None | queue_rung150 |
| standard_refactor_control_30m_seed1_rung150 | rung150 | advance | 6.399200439453125 | None | queue_rung500 |
| standard_refactor_control_30m_seed1_rung500 | rung500 | advance | 5.592936992645264 | None | manual_full_row_required |
| differential_qkv_anti_value_30m_seed1_rung020 | rung020 | advance | 9.712251663208008 | True | queue_rung150 |
| differential_qkv_anti_value_30m_seed1_rung150 | rung150 | advance | 6.400367736816406 | True | queue_rung500 |
| differential_qkv_anti_value_30m_seed1_rung500 | rung500 | advance | 5.598229885101318 | True | manual_full_row_required |
| scope_gated_qkv_30m_seed1_rung020 | rung020 | advance | 9.777275085449219 | True | queue_rung150 |
| scope_gated_qkv_30m_seed1_rung150 | rung150 | advance | 6.4305949211120605 | True | queue_rung500 |
| scope_gated_qkv_30m_seed1_rung500 | rung500 | advance | 5.597848415374756 | True | manual_full_row_required |
