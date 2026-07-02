# E002 Mechanism Investigation Addendum

Derived mechanism inventory:

```text
reports/mechanisms/backfill/E002_multitrack_qkv_shift_register/
```

Checkpoint-recompute is available for the canonical completed global Multi-QKV runs. Old skeleton configs remain `not_available`.

Native hook support includes standard GPT sites plus `selected_track`, `track_q`, `track_k`, `track_v`, and `track_out`. The appropriate next analysis is route specialization: route replacement matrices, Q/K/V-only swaps, and per-position patching from checkpoint-recomputed caches.
