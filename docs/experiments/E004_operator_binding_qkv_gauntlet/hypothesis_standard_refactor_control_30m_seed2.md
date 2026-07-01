# Hypothesis: standard_refactor_control_30m_seed2

## Architecture Summary

Standard GPT-style causal self-attention with packed Q/K/V, causal SDPA, and a single output projection.

## Mechanistic Hypothesis

The control calibrates loss descent, speed, VRAM, and checkpoint behavior for the same E004 data and training contract.

## Survival

Survival means the control completes gauntlet screen rungs with descending loss, no NaN/Inf, and checkpoint artifacts.

## Mechanism Activity

No non-standard mechanism is present. Mechanism activity is not required for the standard control.

## Nearest Boring Explanations

Candidate differences may reflect shared harness changes, seed noise, or extra parameters rather than architecture-specific behavior.

## Kill Criteria

Kill or investigate if the control fails to train, misses checkpoints, produces NaN/Inf, or diverges from the fixed E004 contract.

## Diagnostics Emitted

No attention-mechanism diagnostics are required for standard attention.

## Required Controls For Future Work

All E004 candidates compare against this matched standard control at the same rung before any direct interpretation.
