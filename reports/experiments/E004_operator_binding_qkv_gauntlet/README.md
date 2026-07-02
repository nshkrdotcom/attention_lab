# E004 Operator-Binding QKV Gauntlet

Status: report directory prepared; no E004 gauntlet run evidence is claimed yet.

This directory holds gauntlet reports, promotion decisions, templates, and future verified E004 results.

Expected gauntlet outputs:

```text
gauntlet_report.json
gauntlet_report.md
promotion/
```

The gauntlet report is screen evidence only. It records which rung each candidate reached, why the machine policy advanced or blocked it, and whether final screen success is ready for manual full promotion.

Do not interpret validation loss as a scientific result unless the matched control and candidate have verified full-run artifacts.

## Mechanism Investigation Addendum

Derived backfill artifacts now live under:

```text
reports/mechanisms/backfill/E004_operator_binding_qkv_gauntlet/
```

The base full-run configs have no full-run checkpoints in this tree. Rung checkpoints exist for the standard control, operator-valued attention, dynamic-value attention, and Q3K3V3 where their rungs ran. The mechanism substrate can recompute operator probabilities/outputs, dynamic gates/deltas, and Q3 role streams from those checkpoints for small prompt batches.

The generated cross-experiment report classifies `operator_valued_attention_30m_seed2_rung500` as a mechanism-probe promotion candidate, `dynamic_value_query_conditioned_attention_30m_seed2_rung500` as diagnostic rescue, and `q3k3v3_role_routed_attention_30m_seed2_rung020` as profiling/redesign. These are next-action labels, not scientific conclusions.
