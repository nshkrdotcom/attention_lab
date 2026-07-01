# Hypothesis: dynamic_value_query_conditioned_attention_30m_seed2

## Architecture Summary

`dynamic_value_query_conditioned_attention` keeps standard causal Q/K routing but gates the retrieved value content with a receiver-conditioned read-mode gate before the output projection.

## Mechanistic Hypothesis

The same source content may need different receiver-conditioned read modes. This variant tests dynamic value reinterpretation while keeping the attention route close to standard attention.

## Survival

Survival means loss descends, no NaN/Inf appears, checkpoints exist, and the dynamic value gate remains nondegenerate through screen rungs.

## Mechanism Activity

Activity means static content norm, gated content norm, and value delta norm are positive; gate mean and std are finite; gate std is nonzero; and the gate is not saturated exactly at 0 or 1.

## Nearest Boring Explanations

The mechanism may be active only because the module has more parameters or more computation than the standard control.

The gate may act as a generic learned rescaling rather than a dynamic read-mode probe.

## Kill Criteria

Kill if diagnostics are missing, the gate saturates, value delta is zero, content path is zero, values are nonfinite, loss does not descend, or checkpoints are missing.

## Diagnostics Emitted

`attention_diagnostics.jsonl` includes gate mean/std/min/max, gate entropy proxy, static and gated content norms, delta norm, delta-to-static ratio, pairwise-mode flag, and gate source.

## Required Controls For Future Work

Compare against `standard_refactor_control_30m_seed2` at the same rung. Future follow-up needs parameter-count and compute controls before interpreting loss or behavior.
