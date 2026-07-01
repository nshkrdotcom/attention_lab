# Hypothesis: operator_valued_attention_30m_seed2

## Architecture Summary

`operator_valued_attention` retrieves content with causal Q/K/V attention, then routes the retrieved message through a fixed small set of update modes: add, suppress, gate, transform, and bind. The suppress path is explicitly negative-signed.

## Mechanistic Hypothesis

Some useful attention writes may be operator-like rather than one fixed OV write. A router over fixed update modes may expose whether the module uses multiple control-like write paths during early pretraining.

## Survival

Survival means loss descends, no NaN/Inf appears, checkpoints exist, and the operator router remains nondegenerate through screen rungs.

## Mechanism Activity

Activity means diagnostics show positive combined output norm, at least two active operator output norms, finite nonzero router entropy, finite probabilities, positive suppress scale, and no total router collapse to one operator.

## Nearest Boring Explanations

The mechanism may be active only because the module has more parameters or more computation than the standard control.

The router may learn a generic mixture of projections without any operator-like specialization.

## Kill Criteria

Kill if diagnostics are missing, probabilities are nonfinite, the router collapses to one operator, all non-add paths are zero, suppress scale is nonpositive, loss does not descend, or checkpoints are missing.

## Diagnostics Emitted

`attention_diagnostics.jsonl` includes operator probabilities, router entropy, operator argmax fractions, branch output norms, combined output norm, and suppress scale.

## Required Controls For Future Work

Compare against `standard_refactor_control_30m_seed2` at the same rung. Future follow-up needs parameter-count and compute controls before interpreting loss or behavior.
