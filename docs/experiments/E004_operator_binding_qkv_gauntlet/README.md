# E004 Operator-Binding QKV Gauntlet

E004 is a screen-first gauntlet for operator-valued, role-routed, and dynamic-value attention mechanisms. Passing a rung means the candidate is stable enough to inspect under that screen budget; it is not a model-improvement claim.

## Mechanism Investigation Addendum

Derived mechanism inventory:

```text
reports/mechanisms/backfill/E004_operator_binding_qkv_gauntlet/
```

Base full-run configs do not currently have full-run checkpoints. Rung checkpoints exist for the standard control, operator-valued attention, dynamic-value attention, and Q3K3V3 where their rungs ran.

Native hook support exposes operator probabilities and operator outputs, dynamic gates/deltas, Q3 role streams and pair products, and standard GPT sites. The generated cross-experiment report labels `operator_valued_attention_30m_seed2_rung500` as a mechanism-probe promotion candidate, `dynamic_value_query_conditioned_attention_30m_seed2_rung500` as diagnostic rescue, and `q3k3v3_role_routed_attention_30m_seed2_rung020` as profiling/redesign. These are next actions, not scientific conclusions.
