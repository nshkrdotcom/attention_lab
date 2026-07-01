# 0901 Rebuild 02: Model and Registry Implementation

## Purpose

This document specifies the model, attention, registry, training, eval, and config implementation for the first-build Multi-QKV Shift Register experiment.

It implements:

```text
bundled-triple
globally-shared
deterministic-modular
full-dense-matrix
hard-switch
```

for exactly:

```text
multi_qkv_static_3track_global
multi_qkv_train_rotation_3track_global
multi_qkv_position_rotation_3track_global
```

## Files Added

```text
src/attention_lab/models/attention/multi_qkv_common.py
src/attention_lab/models/attention/multi_qkv_static.py
src/attention_lab/models/attention/multi_qkv_train_rotation.py
src/attention_lab/models/attention/multi_qkv_position_rotation.py
```

## Files Modified

```text
src/attention_lab/models/attention/__init__.py
src/attention_lab/models/attention/registry.py
src/attention_lab/models/gpt.py
src/attention_lab/training/config.py
src/attention_lab/training/train.py
src/attention_lab/training/attention_diagnostics.py
src/attention_lab/evals/loss_eval.py
src/attention_lab/evals/generation_eval.py
src/attention_lab/evals/hellaswag_eval.py
src/attention_lab/training/inspect_model_config.py
reports/schema/attention_diagnostics.schema.json
```

Wrapper scripts continue to call the backing modules.

## Design Overview

Standard attention gives each block its own Q/K/V projection. First-build Multi-QKV gives the model one global bank:

```text
MultiQKVGlobalBank
  c_attn_bank[0]  # Linear(n_embd, 3*n_embd)
  c_attn_bank[1]  # Linear(n_embd, 3*n_embd)
  c_attn_bank[2]  # Linear(n_embd, 3*n_embd)
```

Every Multi-QKV attention block references that same bank. Each block still owns its own output projection:

```text
block[l].attn.c_proj
```

`c_proj`, dropout modules, residual paths, layer norms, and MLPs remain layer-local. The source idea is specifically about a pool of Q/K/V matrices, not global output projections.

## Core Data Structures

`src/attention_lab/models/attention/multi_qkv_common.py` defines:

```python
ScheduleMode = Literal["train", "eval", "generate"]

@dataclass(frozen=True)
class MultiQKVRouteContext:
    layer_idx: int
    step: int | None
    schedule_mode: ScheduleMode
    position_ids: torch.Tensor | None = None
```

The route context is explicit. B must not rely only on `self.training`.

## Global Bank

`MultiQKVGlobalBank` owns the packed global bank:

```python
self.c_attn_bank = nn.ModuleList(
    nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    for _ in range(track_count)
)
```

`project_track(x, track_idx)` returns packed Q/K/V for one bundled track. `project_all_tracks(x)` returns packed Q/K/V for all tracks for C's per-position routing. `per_track_weight_norm()` and `per_track_gradient_norm()` return dictionaries keyed by string track id.

The compatibility alias `MultiQKVSharedBank = MultiQKVGlobalBank` may exist, but canonical docs and new code use `MultiQKVGlobalBank`.

## Base Attention

`MultiQKVGlobalCausalSelfAttention`:

```text
- stores layer_idx
- references the shared qkv_bank
- owns layer-local c_proj
- handles scalar routes for A/B
- handles per-position routes for C
- emits diagnostics
```

It uses `F.scaled_dot_product_attention(..., is_causal=True)` or equivalent causal attention after assembling Q/K/V. The implementation must not introduce non-causal attention or independent Q/K/V track choices.

## Scalar Routing for A/B

A/B compute one packed Q/K/V projection:

```python
track_idx = self.select_scalar_track(context)
qkv = self.qkv_bank.project_track(x, track_idx)
out = self._attention_from_qkv(qkv)
```

Only the selected track participates in that forward path.

## Position Routing for C

C computes all packed projections and assembles a packed Q/K/V tensor by position:

```python
active_tracks = (layer_idx + position_ids) % track_count
qkv_by_track = self.qkv_bank.project_all_tracks(x)
qkv = torch.empty_like(qkv_by_track[0])
for track_idx in range(track_count):
    mask = active_tracks == track_idx
    qkv[:, mask, :] = qkv_by_track[track_idx][:, mask, :]
out = self._attention_from_qkv(qkv)
```

C is more expensive by construction. Reports must include throughput and VRAM.

## Variant Classes

A:

```text
MultiQKVStaticGlobalCausalSelfAttention
attention_type = "multi_qkv_static_3track_global"
route_formula = "layer_idx % track_count"
select_scalar_track = layer_idx % track_count
```

B:

```text
MultiQKVTrainRotationGlobalCausalSelfAttention
attention_type = "multi_qkv_train_rotation_3track_global"
route_formula = "(layer_idx + step) % track_count during train; layer_idx % track_count during eval/generate"
train: (layer_idx + step) % track_count
eval/generate: layer_idx % track_count
```

Missing step in train mode is an error. Do not use `step or 0`.

C:

```text
MultiQKVPositionRotationGlobalCausalSelfAttention
attention_type = "multi_qkv_position_rotation_3track_global"
route_formula = "(layer_idx + position) % track_count"
select_position_tracks = (layer_idx + position_ids) % track_count
```

C must not be scalar-routed.

## Registry

`build_attention(config, *, layer_idx, qkv_bank)` keeps standard and CP behavior intact and dispatches Multi-QKV types only when a shared bank is supplied. It must not instantiate a new bank inside the registry. The implementation may keep `shared_qkv_bank` as a compatibility alias, but new code should pass `qkv_bank`.

## GPTConfig

Required fields:

```python
qkv_track_count: int = 1
qkv_global_bank: bool = False
qkv_route_formula: str | None = None
```

Compatibility aliases may exist:

```python
multi_qkv_track_count
multi_qkv_global
```

Canonical E002 configs use `qkv_*`.

## Model-Level Shared Bank

`GPT.__init__` creates the bank only for Multi-QKV attention types:

```python
self.multi_qkv_bank = MultiQKVGlobalBank(config) if uses_multi_qkv else None
```

Blocks receive the same bank object:

```python
Block(config, layer_idx=i, qkv_bank=self.multi_qkv_bank)
```

Tests must prove object/parameter identity across blocks.

## Forward Context

All attention modules support:

```python
def forward(
    self,
    x,
    *,
    step=None,
    schedule_mode=None,
    position_ids=None,
):
    ...
```

Standard and CP modules ignore the new context.

`GPT.forward` normalizes missing mode as backwards-compatible fallback:

```python
schedule_mode = "train" if self.training else "eval"
```

Training calls:

```python
model(x, y, step=step, schedule_mode="train")
```

Eval loss and HellaSwag call:

```python
model(x, y, schedule_mode="eval")
```

Generation calls:

```python
model(idx_cond, schedule_mode="generate")
```

Current generation does not use a KV cache. For `multi_qkv_position_rotation_3track_global`, position IDs during generation are recomputed for the full cropped context window passed to `GPT.forward`. Therefore the current implementation uses window-relative positions during generation. If incremental KV-cache generation is added later, E002 C must define and test whether routing uses absolute generated-token positions or window-relative positions before the KV-cache path is enabled.

## Config Validation

Implemented attention types include the three Multi-QKV canonical names. Validation requires:

```text
qkv_track_count == 3
qkv_global_bank is true
qkv_route_formula matches attention_type
```

The same canonical invariant is enforced when constructing `GPT` directly from Python. `GPT(GPTConfig(attention_type="multi_qkv_static_3track_global"))` must fail unless `qkv_track_count=3`, `qkv_global_bank=True`, and the expected `qkv_route_formula` are explicitly set. One-track identity remains a direct attention-module/unit-test path, not a canonical runnable `GPT` model path.

Route formula mapping:

```text
multi_qkv_static_3track_global -> layer_mod
multi_qkv_train_rotation_3track_global -> layer_plus_step_train_layer_eval
multi_qkv_position_rotation_3track_global -> layer_plus_position
```

Strict unknown-key behavior remains.

## Inspect Model Config

Inspection reports:

```text
attention_type
qkv_track_count
qkv_global_bank
qkv_route_formula
parameters_excluding_positional
parameters_including_positional
trainable_parameters
attention_projection_parameters
non_attention_parameters
```

This is required because global bank sharing changes parameter accounting.

## Diagnostics

The existing diagnostics collector calls `attention_diagnostics(step=step, layer=layer_idx)` for modules that define it. Multi-QKV diagnostics include:

```text
attention_type
route_formula
uses_global_bank
track_count
layer_idx
layer
step
last_forward_step
schedule_mode
active_track_index
active_track_counts
track_gradient_norm
per_track_gradient_norm
per_track_qkv_weight_norm
per_track_output_norm
track_entropy
position_routing_enabled
eval_freeze_mode
```

Field names preserve compatibility with `qkv_track_activity`.

## Alias Policy

The old skeleton attention names are not canonical. Do not promote `multi_qkv_layer_shift`, `multi_qkv_train_shift`, `multi_qkv_softmix`, or `multi_qkv_train_shift_warmup` in first build. They remain unimplemented/future work.

## Invariants

Global sharing:

```python
banks = [block.attn.qkv_bank for block in model.transformer.h]
assert all(bank is banks[0] for bank in banks)
```

Bundled triple:

```python
qkv = attention.qkv_bank.project_track(x, track)
q, k, v = qkv.split(n_embd, dim=2)
```

B eval freeze:

```text
train_track = (layer_idx + step) % 3
eval_track = layer_idx % 3
generate_track = layer_idx % 3
```

C per-position routing:

```text
active_tracks = (layer_idx + position_ids) % 3
```

Standard compatibility:

```text
standard outputs are unchanged by step/schedule_mode/position_id threading
```

## Common Failure Modes

Do not create a bank inside each attention layer. Do not use `step or 0`. Do not use train-step rotation during eval/generation. Do not implement C as scalar routing. Do not leave standard/CP forward signatures incompatible with the shared Block call. Do not emit diagnostics with empty activity fields and call that mechanism-active.

## Manual Full-Run Boundary

This implementation prepares code and scripts. It must not execute:

```text
scripts/experiments/E002_multitrack_qkv_shift_register/run_all_full_initial.sh
```

unless a human explicitly starts manual full-run execution after implementation is complete.
