# E001 Mechanism Investigation Addendum

Derived mechanism inventory:

```text
reports/mechanisms/backfill/E001_cp_trilinear_attention/
```

Checkpoint-recompute is available for `standard_30m_seed1`, `cp_bilinear_r8_30m_seed1`, and `cp_trilinear_r8_30m_seed1`. `cp_trilinear_r8_lambda0_30m_seed1` is `checkpoint_unavailable`.

Native hook support includes standard GPT sites plus `cp_score`, `cp_output`, and `cp_lambda`. `cp_rank_component[layer, rank]` is declared but full tensor capture is unsupported until optimized. Historical activations were not saved and cannot be recovered.
