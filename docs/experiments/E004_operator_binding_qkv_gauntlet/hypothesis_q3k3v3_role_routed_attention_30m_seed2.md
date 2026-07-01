# Hypothesis: q3k3v3_role_routed_attention_30m_seed2

## Architecture Summary

`q3k3v3_role_routed_attention` creates three typed Q/K/V role streams inside the attention module: content-like, operator-like, and binding-like. The default E004 version uses diagonal role interactions and projects content, operator, binding, and pair-product streams back to the residual dimension.

## Mechanistic Hypothesis

Standard Q/K/V may force content-like, operator-like, and binding-like factors into one compatibility and value geometry. Separate role streams may expose whether these paths can remain active and separable during early pretraining.

## Survival

Survival means loss descends, no NaN/Inf appears, checkpoints exist, and all three role streams remain active through screen rungs.

## Mechanism Activity

Activity means content, operator, and binding output norms are positive, role ratios are finite, and at least one pair interaction is nonzero when pair products are enabled.

## Nearest Boring Explanations

The mechanism may be active only because the module has more parameters or more computation than the standard control.

The role labels may remain diagnostic labels only, with no stable content/operator/binding-like specialization.

## Kill Criteria

Kill if any role stream is zero across diagnostics, pair products are all zero when enabled, role ratios are nonfinite, diagnostics are missing, loss does not descend, or checkpoints are missing.

## Diagnostics Emitted

`attention_diagnostics.jsonl` includes role output norms, pair-interaction norms, role-to-total ratios, diagonal attention entropy estimates, and role/grid flags.

## Required Controls For Future Work

Compare against `standard_refactor_control_30m_seed2` at the same rung. Future follow-up needs parameter-count and compute controls before interpreting loss or behavior.
