# Mechanism Backfill Inventory: E001_cp_trilinear_attention

- generated_from_commit: `0b81cf2804733cb95688a45ff31f2556e38a3ce7`
- repo_root_relative: `True`

| run | attention | checkpoint | diagnostics | evidence | posthoc |
| --- | --- | --- | --- | --- | --- |
| cp_bilinear_r8_30m_seed1 | cp_bilinear | available | available | checkpoint_recompute | checkpoint_recompute_available |
| cp_trilinear_r8_30m_seed1 | cp_trilinear | available | available | checkpoint_recompute | checkpoint_recompute_available |
| cp_trilinear_r8_lambda0_30m_seed1 | cp_trilinear | checkpoint_unavailable | missing | not_available | checkpoint_unavailable |
| standard_30m_seed1 | standard | available | not_recorded | checkpoint_recompute | checkpoint_recompute_available |
| standard_refactor_control_30m_seed1 | standard | checkpoint_unavailable | not_recorded | not_available | checkpoint_unavailable |
