# 0901 Rebuild 03: TDD and QC Plan

## Purpose

This document defines the required test-driven development and quality-control plan for the first-build Multi-QKV Shift Register implementation.

The implementation is accepted only when tests prove the actual experiment properties:

```text
- global shared Q/K/V bank
- bundled Q/K/V track selection
- correct A/B/C route formulas
- B train/eval/generate schedule split
- C per-position routing
- unchanged standard and CP behavior
- non-degenerate diagnostics
- valid configs and manual scripts
```

The tests extend the repo's existing attention, config, train, queue, summary, and verification test style. They must not bypass the harness or weaken existing checks.

## TDD Rule

Add tests before or alongside each implementation step. Do not implement all model code first and then add superficial tests.

Required sequence:

```text
1. Add config validation tests for new fields.
2. Add standard/CP compatibility tests for new forward signatures.
3. Add global bank tests.
4. Add A formula tests.
5. Add B formula and eval-freeze tests.
6. Add C per-position tests.
7. Add one-track identity tests.
8. Add causal masking tests.
9. Add training/eval/generate context-threading tests.
10. Add diagnostics tests.
11. Add config/script validation tests.
12. Run full QC.
```

## Test Files

Required new or updated files:

```text
tests/test_attention_multi_qkv_global.py
tests/test_model_multi_qkv_global.py
tests/test_train_multi_qkv_step_position_threading.py
tests/test_validate_experiment_e002_multi_qkv.py
tests/test_validate_experiment_e002.py
tests/test_qkv_rotation_diagnostics.py
tests/test_model_standard.py
tests/test_attention_cp_bilinear.py
tests/test_attention_cp_trilinear.py
tests/test_config.py
tests/test_attention_diagnostics.py
tests/test_eval_loss.py
tests/test_hellaswag_eval.py
tests/test_train_tiny.py
tests/test_verify_run.py
```

Do not weaken existing tests.

## Shared Tiny Helpers

Test helpers may live locally in the test files. Tiny configs should use:

```text
block_size = 8
vocab_size = 64
n_layer = 3
n_head = 2
n_embd = 16
dropout = 0.0
bias = false
```

Multi-QKV tiny configs must set:

```yaml
model:
  qkv_track_count: 3
  qkv_global_bank: true
  qkv_route_formula: layer_mod | layer_plus_step_train_layer_eval | layer_plus_position
```

The one-track identity helper may use:

```yaml
model:
  attention_type: multi_qkv_static_3track_global
  qkv_track_count: 1
  qkv_global_bank: true
  qkv_route_formula: layer_mod
```

The one-track case is a wiring test only, not a runnable E002 experiment config.

## Attention Module Tests

`tests/test_attention_multi_qkv_global.py` must directly test:

```text
[x] MultiQKVGlobalBank constructs three packed c_attn_bank tracks.
[x] project_track returns [B, T, 3 * n_embd].
[x] project_track rejects invalid direct track indexes.
[x] A: select_scalar_track(context) = layer_idx mod 3.
[x] B train: select_scalar_track(context) = (layer_idx + step) mod 3.
[x] B train missing step raises ValueError.
[x] B eval/generate freezes to layer_idx mod 3.
[x] C: select_position_tracks(context) = (layer_idx + position_ids) mod 3.
[x] C rejects wrong position_ids length.
[x] A/B/C forward shapes match input hidden shape.
[x] A/B/C causal masks prevent future-token influence.
[x] One-track packed Multi-QKV attention matches standard attention when c_attn and c_proj weights are copied.
[x] Inactive scalar hard-switch tracks receive no gradient for a selected forward path.
```

The direct formula tests should use `MultiQKVRouteContext`; forward tests should pass `schedule_mode` and `position_ids` explicitly.

## Full Model Tests

`tests/test_model_multi_qkv_global.py` must test:

```text
[x] Every Multi-QKV block references the same model.multi_qkv_bank object.
[x] Bank parameters are identical objects across blocks.
[x] A/B/C GPT forwards return [B, T, vocab_size] logits and finite scalar loss.
[x] Standard attention outputs are unchanged by step and schedule_mode threading.
[x] CP bilinear/trilinear models accept step and schedule_mode.
[x] B eval at model level freezes tracks to [0, 1, 2] for a 3-layer tiny model.
[x] B train at step 1 routes tracks to [1, 2, 0].
[x] C records per-position track counts and scalar active_track_index is None.
[x] Position ids reach C attention.
[x] E002 candidate parameter deltas inspect successfully.
```

If a full-model one-track identity copy is too broad, the lower-level attention identity test is acceptable, but there must be at least one identity test proving packed Multi-QKV can exactly reproduce standard attention when `track_count=1`.

## Train/Eval/Generate Context Tests

`tests/test_train_multi_qkv_step_position_threading.py` must prove:

```text
[x] A tiny train run passes the optimizer step to train-rotation attention.
[x] The tiny run writes attention diagnostics.
[x] Verification passes for the tiny run.
[x] Generation uses schedule_mode="generate".
[x] B generation freezes to layer_idx mod 3.
[x] Position ids reach C attention through model forward.
```

Tiny training may run on CPU and must remain small. Full 3000-step E002 runs are forbidden in automated QC.

## Diagnostics Tests

`tests/test_qkv_rotation_diagnostics.py` must prove diagnostics are not decorative:

```text
[x] A/B/C diagnostics emit attention_type, route_formula, uses_global_bank, track_count, layer_idx, step, last_forward_step, schedule_mode, active_track_index, active_track_counts, track_gradient_norm, per_track_gradient_norm, per_track_qkv_weight_norm, per_track_output_norm, track_entropy, position_routing_enabled, and eval_freeze_mode.
[x] Scalar A/B hard routing gives gradient only to the active track for layer 0, step 0.
[x] C per-position routing gives gradient to all tracks for layer 0 with sequence length 8.
[x] C diagnostics count all tracks.
[x] Diagnostic rows are JSON serializable.
[x] The queue qkv_track_activity mechanism check accepts nonzero diagnostics.
[x] The queue qkv_track_activity mechanism check rejects zero diagnostics.
```

Mechanism tests must use `src/attention_lab/queue/mechanism_checks.py`, not a duplicate checker.

## Config and Experiment Tests

`tests/test_validate_experiment_e002_multi_qkv.py` and existing config tests must prove:

```text
[x] Canonical E002 initial configs validate.
[x] Multi-QKV rejects qkv_global_bank=false.
[x] Multi-QKV rejects wrong qkv_route_formula.
[x] Multi-QKV rejects wrong qkv_track_count.
[x] E002 metadata includes canonical initial config files.
[x] E002 metadata distinguishes the four canonical first-build configs from the legacy/noncanonical runnable `standard_30m_seed1.yaml`.
[x] Old skeleton variants remain experimental_unimplemented.
[x] E002 run dirs are unique and under the E002 run directory.
```

Canonical first-build configs:

```text
configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

## Existing Test Updates

Required compatibility coverage:

```text
[x] Standard attention accepts schedule kwargs without behavior change.
[x] CP bilinear/trilinear attention accepts schedule kwargs.
[x] Direct canonical Multi-QKV `GPT` construction rejects missing or invalid `qkv_track_count`, `qkv_global_bank`, and `qkv_route_formula`.
[x] One-track identity remains a direct attention-module test path rather than a canonical runnable `GPT` model.
[x] C generation uses `schedule_mode="generate"` and window-relative positions from the cropped full context because the generation path has no KV cache.
[x] qkv_track_count, qkv_global_bank, and qkv_route_formula are accepted.
[x] unknown qkv-ish fields remain rejected.
[x] multi_qkv_* attention types require required qkv fields.
[x] diagnostics collector handles Multi-QKV diagnostics.
```

## Tiny Training Requirement

At least one tiny Multi-QKV training test must run and verify:

```text
- model instantiates
- train step runs
- loss is finite
- run artifacts are created
- attention diagnostics are emitted
- verify_run passes for the tiny run
```

Parametrized tiny training across A/B/C is acceptable only if runtime remains modest. Otherwise one integration test for train-rotation plus unit-level coverage for A/C is enough.

## Fast Local QC

Run from repo root:

```bash
uv run ruff check .
uv run pytest tests/test_attention_multi_qkv_global.py
uv run pytest tests/test_model_multi_qkv_global.py
uv run pytest tests/test_qkv_rotation_diagnostics.py
uv run pytest tests/test_validate_experiment_e002_multi_qkv.py
```

## Full QC

Required before commit:

```bash
uv run ruff check .
uv run pytest
uv run pytest --run-integration
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
```

Inspect all canonical initial configs:

```bash
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

Each inspect command must report:

```text
- attention_type
- qkv_track_count
- qkv_global_bank
- qkv_route_formula
- total parameters
- trainable parameters
- attention projection/global qkv bank parameters
```

## Acceptance Criteria

The implementation is not complete until:

```text
[ ] ruff passes.
[ ] all normal tests pass.
[ ] integration tests pass.
[ ] E002 validation passes.
[ ] all four canonical configs inspect successfully.
[ ] standard attention invariance test passes.
[ ] CP compatibility tests pass.
[ ] global-bank identity tests pass.
[ ] A/B/C formula tests pass.
[ ] B eval-freeze tests pass.
[ ] C per-position tests pass.
[ ] diagnostics tests pass with nonzero activity.
[ ] config rejection tests pass for invalid Multi-QKV settings.
```

## No Fake Green

Do not skip required implementation behavior. Skips are allowed only for environment-dependent GPU/full-run tests, not for Multi-QKV implementation behavior.

Bad:

```python
@pytest.mark.skip("multi_qkv not implemented yet")
def test_multi_qkv_global_bank_constructs_three_packed_tracks():
    ...
```

Good:

```python
def test_multi_qkv_global_bank_constructs_three_packed_tracks():
    ...
```

## No Full Runs During TDD

The implementation tests may run tiny training only.

They must not run:

```text
scripts/experiments/E002_multitrack_qkv_shift_register/run_all_full_initial.sh
```

or any other 3000-step E002 full-run script.

## Bug Triage Rules

If a test fails, fix the implementation or the test's incorrect assumption. Do not weaken tests unless the docset contract is wrong.

Examples:

```text
- If B uses step during eval, fix B.
- If C emits scalar active_track_index instead of counts, fix C diagnostics.
- If all layers have different bank objects, fix GPT construction.
- If standard outputs differ after adding kwargs, fix standard path.
- If qkv_global_bank=false is accepted for A/B/C, fix config validation.
```

## Required Final QC Log

The final agent response after implementation must include:

```text
uv run ruff check .                         PASSED/FAILED
uv run pytest                               PASSED/FAILED
uv run pytest --run-integration             PASSED/FAILED
uv run scripts/validate_experiment.py ...   PASSED/FAILED
inspect_model_config x4                     PASSED/FAILED
```

If any command failed, include:

```text
- failing command
- failure summary
- files likely involved
- whether the code was committed
```

Do not commit if QC is not green unless the human explicitly instructs otherwise.

## Handoff To Next Doc

The next document must define diagnostics and mechanism checks in more detail, including:

```text
- JSONL diagnostics fields
- schema updates
- qkv_track_activity mechanism-check expectations
- destructive/off-route test script
- what counts as mechanism-active evidence
- what counts as degenerate or insufficient evidence
```
