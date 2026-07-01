# 0901 Rebuild 01: Experiment Math and Variant Contract

## Purpose

This document defines the exact experiment math for the first Multi-QKV Shift Register implementation.

The source idea is narrow:

```text
Use a pool of same-shape Q/K/V matrices.
Select one bundled Q/K/V triple by deterministic schedule.
Force the same physical Q/K/V weights to operate under different phase/depth conditions.
Measure whether that pressure creates more portable attention subroutines.
```

This is not typed content/control/binding stream attention. The pool is 3 Q matrices, 3 K matrices, and 3 V matrices, all same-shape and same-role, with exactly one bundled triple active for a given forward path.

## Definitions

```text
L = number of transformer layers
l = zero-indexed layer index
t = zero-indexed optimizer/training step
p = zero-indexed token position inside the current sequence
K = track_count = 3
r = active track index in {0, 1, 2}
d = model embedding dimension
```

For each globally shared bank:

```text
W_Q[0], W_Q[1], W_Q[2]
W_K[0], W_K[1], W_K[2]
W_V[0], W_V[1], W_V[2]
```

Each projection has the same shape as the standard attention projection it replaces. The implementation may use separate Q/K/V projections or an equivalent packed `c_attn_bank[r]: Linear(d, 3d)` projection.

## Bundled-Triple Rule

The first implementation uses bundled Q/K/V track selection:

```text
Q = Q_bank[r]
K = K_bank[r]
V = V_bank[r]
```

or, with packed projections:

```text
Q, K, V = split(c_attn_bank[r](x))
```

It must not select Q, K, and V independently unless `r_q == r_k == r_v` for every token, layer, and step.

## Global-Sharing Rule

The first implementation uses one global 3-track Q/K/V bank shared by all transformer blocks:

```text
model.multi_qkv_bank
block_0.attn -> references model.multi_qkv_bank
block_1.attn -> references model.multi_qkv_bank
...
```

Layer-local banks are a later ablation only. Tests must prove all Multi-QKV blocks reference the same bank object or the same underlying bank parameters.

## Standard Attention Reference

Standard causal self-attention is:

```text
Q, K, V = split(c_attn(x))
A = softmax((Q K^T) / sqrt(d_head), causal_mask=True)
Y = A V
out = c_proj(Y)
```

Multi-QKV first-build variants replace only the Q/K/V projection source. They preserve causal masking, scaled dot-product semantics, output projection semantics, dropout semantics, residual/block ordering, and the shared optimizer/training loop except for explicit step/position/mode context threading.

## Variant A: Static Depth-Only Cycle

Canonical run:

```text
multi_qkv_static_3track_global_30m_seed1
```

Attention type:

```text
multi_qkv_static_3track_global
```

Formula:

```text
active_track(l, t, p, mode) = l mod 3
```

Mode does not affect A. For a 6-layer model:

```text
layer 0 -> track 0
layer 1 -> track 1
layer 2 -> track 2
layer 3 -> track 0
layer 4 -> track 1
layer 5 -> track 2
```

A is the static global-bank control. It tests whether a globally shared 3-track Q/K/V bank trains stably under a fixed depth cycle. It is not evidence for train-time or inference-persistent rotation by itself.

## Variant B: Train-Time Step Rotation, Frozen Eval Layout

Canonical run:

```text
multi_qkv_train_rotation_3track_global_30m_seed1
```

Attention type:

```text
multi_qkv_train_rotation_3track_global
```

Training formula:

```text
active_track(l, t, p, mode="train") = (l + t) mod 3
```

Evaluation and generation formula:

```text
active_track(l, t, p, mode in {"eval", "generate"}) = l mod 3
```

The training step clock exists only during training. B must not use `(l + t) mod 3` during eval or generation.

During training, B requires a real step integer. This is valid:

```python
model(idx, targets, step=step, schedule_mode="train")
```

This is an error:

```python
model(idx, targets, step=None, schedule_mode="train")
```

Do not write `step = step or 0` inside B. During eval/generation, B must not require `step`.

## Variant C: Position Rotation Active at Train and Inference

Canonical run:

```text
multi_qkv_position_rotation_3track_global_30m_seed1
```

Attention type:

```text
multi_qkv_position_rotation_3track_global
```

Formula:

```text
active_track(l, t, p, mode) = (l + p) mod 3
```

Mode does not affect C.

Position is the zero-indexed sequence-local token position used by this repo's positional embedding path:

```python
pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
```

Current generation does not use a KV cache. For `multi_qkv_position_rotation_3track_global`, position IDs during generation are recomputed for the full cropped context window passed to `GPT.forward`. Therefore the current implementation uses window-relative positions during generation. If incremental KV-cache generation is added later, E002 C must define and test whether routing uses absolute generated-token positions or window-relative positions before the KV-cache path is enabled.

C cannot use a single scalar active track for the whole sequence. It computes all three Q/K/V projections and selects per position before attention. This is intentionally correctness-first and may be slower; throughput and VRAM must be reported.

## Shared Attention Computation

After Q/K/V selection, every variant uses the same causal attention computation:

```python
q, k, v = split_qkv(qkv)
q = q.view(B, T, n_head, head_dim).transpose(1, 2)
k = k.view(B, T, n_head, head_dim).transpose(1, 2)
v = v.view(B, T, n_head, head_dim).transpose(1, 2)
y = scaled_causal_attention(q, k, v)
out = resid_dropout(c_proj(y))
```

No non-causal attention, cross-token leakage, or new value mixing is allowed.

## One-Track Identity Control

The implementation supports one-track Multi-QKV modules for unit tests:

```yaml
qkv_track_count: 1
qkv_global_bank: true
```

In the one-track case, `active_track(...) = 0` for all modes, layers, steps, and positions. A one-track Multi-QKV attention module must match standard attention exactly when Q/K/V and output projection weights are copied and dropout is disabled or controlled. This is a wiring correctness test, not a full experiment variant.

## Schedule Mode Contract

All model forward paths accept explicit schedule context:

```python
def forward(
    idx,
    targets=None,
    *,
    step=None,
    schedule_mode=None,
):
    ...
```

Required behavior:

```text
standard: ignores step and schedule_mode
cp_bilinear/cp_trilinear: ignores step and schedule_mode unless diagnostics need it
multi_qkv_static_3track_global: ignores step and schedule_mode for routing
multi_qkv_train_rotation_3track_global: uses step only when schedule_mode == "train"; freezes to layer_idx mod 3 in eval/generate
multi_qkv_position_rotation_3track_global: uses position in all modes
```

Training passes `schedule_mode="train"`, eval loss and HellaSwag pass `schedule_mode="eval"`, and generation passes `schedule_mode="generate"`. Model internals may infer from `self.training` only as backwards-compatible fallback.

## Parameter-Count Contract

Reports must inspect and record parameter counts for:

```text
standard_refactor_control_30m_seed1
multi_qkv_static_3track_global_30m_seed1
multi_qkv_train_rotation_3track_global_30m_seed1
multi_qkv_position_rotation_3track_global_30m_seed1
```

Because the first-build design uses a globally shared Q/K/V bank, parameter count may differ from standard attention. Do not assume matched count. Reports must include total parameters, trainable parameters, attention projection parameters, and non-attention parameters.

Nearest controls:

```text
standard_refactor_control: verifies shared-path plumbing did not alter standard attention
multi_qkv_static_3track_global: controls for global 3-track bank and static cycling
multi_qkv_train_rotation_3track_global: tests train-time phase exposure beyond static cycling
multi_qkv_position_rotation_3track_global: tests inference-persistent position clock beyond static cycling
```

## Diagnostics Contract

Every Multi-QKV first-build variant emits diagnostics sufficient to answer:

```text
Which track was active?
Did all tracks receive gradient over time?
Did inactive hard-switch tracks remain inactive in a selected forward path?
Did the global bank actually serve multiple layers?
Did eval freeze B correctly?
Did C route by position rather than one scalar per layer?
```

Minimum fields:

```json
{
  "attention_type": "multi_qkv_train_rotation_3track_global",
  "route_formula": "(layer_idx + step) % track_count during train; layer_idx % track_count during eval/generate",
  "uses_global_bank": true,
  "track_count": 3,
  "layer_idx": 0,
  "step": 17,
  "schedule_mode": "train",
  "active_track_index": 2,
  "active_track_counts": {"0": 0, "1": 0, "2": 1024},
  "per_track_gradient_norm": {"0": 0.0, "1": 0.0, "2": 0.123},
  "per_track_qkv_weight_norm": {"0": 12.3, "1": 12.1, "2": 12.4},
  "position_routing_enabled": false,
  "eval_freeze_mode": false
}
```

For C, `active_track_index` may be null because routing is per-position; `active_track_counts` is mandatory.

## Required Config Fields

Canonical configs use:

```yaml
model:
  attention_type: multi_qkv_static_3track_global
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_mod
```

For B:

```yaml
model:
  attention_type: multi_qkv_train_rotation_3track_global
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_plus_step_train_layer_eval
```

For C:

```yaml
model:
  attention_type: multi_qkv_position_rotation_3track_global
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_plus_position
```

Validation rejects `qkv_track_count < 1`, canonical A/B/C with `qkv_track_count != 3`, canonical A/B/C with `qkv_global_bank = false`, unknown `qkv_route_formula`, or `multi_qkv_*` attention without required QKV fields.

## Canonical Config Matrix

```text
configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

All share the E002 30M shape and training budget.

`configs/experiments/E002_multitrack_qkv_shift_register/standard_30m_seed1.yaml` remains runnable only as a legacy/noncanonical comparison config. It is not part of the four canonical first-build configs. Validation reports four canonical first-build configs, one legacy/auxiliary runnable config, five runnable configs total, and six unimplemented old skeleton configs.

## Hypothesis Documents

Each canonical run has a hypothesis document with exactly:

```text
CLAIM:
KILL_CONDITION:
MECHANISM_PROOF:
NEAREST_BORING_EXPLANATION:
CONTROL_THAT_RULES_IT_OUT:
```

B and C use `multi_qkv_static_3track_global_30m_seed1` as the key control, not only standard attention.

## Full-Run Claim Boundary

The implementation pass may prepare scripts. It may not claim full-run results. Variants may be described as implemented, tested, config-valid, and manual-script prepared. They may not be described as successful, better, worse, confirmed, falsified, or stable at full run until human/operator runs produce verified artifacts.

## Comparison Contract

First comparisons answer:

1. Did `standard_refactor_control_30m_seed1` match standard expectations?
2. Did `multi_qkv_static_3track_global_30m_seed1` train stably?
3. Did `multi_qkv_train_rotation_3track_global_30m_seed1` outperform A on verified metrics?
4. Did `multi_qkv_position_rotation_3track_global_30m_seed1` outperform A after accounting for throughput and VRAM?

## Kill Conditions

Kill or deprioritize if standard refactor control differs unexpectedly, A cannot train stably, diagnostics fail to prove global-bank routing, B silently uses step zero when missing step, B uses step rotation during eval/generation, C implements scalar layer routing instead of per-position routing, causal masking breaks, checkpoint reload fails, manifest verification fails, diagnostics are degenerate, or destructive/off-route tests show no measurable route effect.

## Promotion Criteria

Do not implement second-round variants until A/B/C are complete and interpreted. Future work may include decoupled coprime Q/K/V clocks, LoRA-delta rotation, learned soft routing, stochastic routing, annealed soft-to-hard routing, or layer-local bank ablation.
