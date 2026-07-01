# 0901 Rebuild 06: Agent Execution Prompt and Final Acceptance

## Purpose

This is the final copy-paste implementation prompt for the coding agent. Use it from the root of the Attention Lab codebase.

The implementation target is the initial E002 Multi-QKV Shift Register experiment family. It is not a vague nine-stream
architecture, typed content/control/binding stream design, learned router, LoRA-delta variant, stochastic router, warmup
router, or coprime decoupled Q/K/V clock experiment.

The first-build experiment is exactly:

```text
bundled-triple
globally-shared
deterministic-modular
full-dense-matrix
hard-switch
```

## Implementation Prompt

You are working in the root of the Attention Lab codebase.

Implement the initial E002 Multi-QKV Shift Register experiment as a real, tested, repo-integrated architecture family.

The source idea is a pool of same-role Q/K/V matrices: 3 Q matrices, 3 K matrices, and 3 V matrices, where one bundled
Q/K/V triple is active according to a deterministic schedule. The point is to deny a physical Q/K/V matrix a fixed depth
home, not to give Q/K/V typed semantic roles.

## Required First-Build Variants

Standard refactor control:

```text
run name: standard_refactor_control_30m_seed1
attention_type: standard
purpose: prove shared-path step/schedule/position threading does not alter standard behavior
```

A: static global cycle:

```text
run name: multi_qkv_static_3track_global_30m_seed1
attention_type: multi_qkv_static_3track_global
formula: active_track(layer_idx, step, position, mode) = layer_idx mod 3
```

B: train-time rotation with eval freeze:

```text
run name: multi_qkv_train_rotation_3track_global_30m_seed1
attention_type: multi_qkv_train_rotation_3track_global
training formula: active_track(layer_idx, step, position, mode="train") = (layer_idx + step) mod 3
eval/generation formula: active_track(layer_idx, step, position, mode in {"eval", "generate"}) = layer_idx mod 3
```

Missing training step is an error. Do not use `step or 0`.

C: position rotation active at train and inference:

```text
run name: multi_qkv_position_rotation_3track_global_30m_seed1
attention_type: multi_qkv_position_rotation_3track_global
formula: active_track(layer_idx, step, position, mode) = (layer_idx + position) mod 3
```

C must route per position and must not collapse to scalar layer routing.

## Required Docset

Create exactly this docset:

```text
docs/implementation/0901_multiqkv_shift_register/
  0901_rebuild_00_docset_index_and_master_contract.md
  0901_rebuild_01_experiment_math_and_variant_contract.md
  0901_rebuild_02_model_and_registry_implementation.md
  0901_rebuild_03_tdd_and_qc_plan.md
  0901_rebuild_04_diagnostics_and_mechanism_checks.md
  0901_rebuild_05_configs_scripts_and_manual_run_loop.md
  0901_rebuild_06_agent_execution_prompt_and_final_acceptance.md
```

Keep docs aligned with the actual final implementation and current repo CLIs.

## Required Code Files

Add:

```text
src/attention_lab/models/attention/multi_qkv_common.py
src/attention_lab/models/attention/multi_qkv_static.py
src/attention_lab/models/attention/multi_qkv_train_rotation.py
src/attention_lab/models/attention/multi_qkv_position_rotation.py
scripts/qkv_track_destructive_test.py
```

Modify as needed:

```text
src/attention_lab/models/attention/__init__.py
src/attention_lab/models/attention/registry.py
src/attention_lab/models/gpt.py
src/attention_lab/training/config.py
src/attention_lab/training/train.py
src/attention_lab/training/attention_diagnostics.py
src/attention_lab/queue/mechanism_checks.py
scripts/train.py
scripts/eval_loss.py
scripts/eval_generate.py
scripts/eval_hellaswag.py
scripts/inspect_model_config.py
scripts/verify_run.py
scripts/compare_runs.py
```

Do not rewrite the training harness. Preserve existing script interfaces where possible.

## Core Architecture Requirements

Global Q/K/V bank:

```text
MultiQKVGlobalBank
  c_attn_bank[0]: Linear(n_embd, 3*n_embd)
  c_attn_bank[1]: Linear(n_embd, 3*n_embd)
  c_attn_bank[2]: Linear(n_embd, 3*n_embd)
```

Every transformer block must reference the same bank object or same underlying parameters:

```python
banks = [block.attn.qkv_bank for block in model.transformer.h]
assert all(bank is banks[0] for bank in banks)
```

Bundled Q/K/V selection:

```python
qkv = qkv_bank.project_track(x, active_track)
q, k, v = qkv.split(n_embd, dim=2)
```

Do not select Q, K, and V from separate tracks in A/B/C.

Keep `c_proj` layer-local. The experimental variable is globally shared Q/K/V projection, not output projection sharing.

Thread explicit schedule context through model/block/attention:

```python
model(idx, targets=None, *, step=None, schedule_mode="eval")
```

Required call sites:

```python
model(x, y, step=step, schedule_mode="train")
model(x, y, step=None, schedule_mode="eval")
model(idx_cond, targets=None, step=None, schedule_mode="generate")
```

Current generation does not use a KV cache. For `multi_qkv_position_rotation_3track_global`, position IDs during generation are recomputed for the full cropped context window passed to `GPT.forward`. Therefore the current implementation uses window-relative positions during generation. If incremental KV-cache generation is added later, E002 C must define and test whether routing uses absolute generated-token positions or window-relative positions before the KV-cache path is enabled.

## Required Config Fields

Add:

```python
qkv_track_count: int = 1
qkv_global_bank: bool = False
qkv_route_formula: str | None = None
```

For canonical A/B/C, validation requires:

```text
qkv_track_count == 3
qkv_global_bank == true
qkv_route_formula matches attention_type
```

Allowed route formulas:

```text
layer_mod
layer_plus_step_train_layer_eval
layer_plus_position
```

Register:

```text
multi_qkv_static_3track_global
multi_qkv_train_rotation_3track_global
multi_qkv_position_rotation_3track_global
```

Keep existing `standard`, `cp_bilinear`, and `cp_trilinear` behavior working. Standard and CP attention forward methods must
accept and ignore schedule kwargs.

## Required Tests

Add:

```text
tests/test_attention_multi_qkv_global.py
tests/test_model_multi_qkv_global.py
tests/test_train_multi_qkv_step_position_threading.py
tests/test_validate_experiment_e002_multi_qkv.py
tests/test_qkv_rotation_diagnostics.py
```

Coverage must include:

```text
[ ] bank construction and projection shapes
[ ] invalid track index rejection
[ ] A/B/C formulas
[ ] B missing training step raises
[ ] B eval/generate freeze
[ ] C per-position routing and position validation
[ ] A/B/C forward shape and causal masking
[ ] global bank identity and shared parameter identity
[ ] standard invariance under schedule kwargs
[ ] CP compatibility under schedule kwargs
[ ] one-track identity against standard attention
[ ] train/eval/generation schedule context threading
[ ] diagnostics required fields
[ ] A/B hard-routing gradients active-track only
[ ] C gradients reach all tracks for sequence length >= 3
[ ] qkv_track_activity pass/reject behavior
[ ] canonical config validation and invalid config rejection
[ ] destructive route override and script output
```

No core implementation test may be skipped because Multi-QKV is missing.

## Required Diagnostics And Mechanism Checks

Multi-QKV diagnostics must include:

```json
{
  "schema_version": 1,
  "experiment_id": "E002_multitrack_qkv_shift_register",
  "run_name": "multi_qkv_static_3track_global_30m_seed1",
  "attention_type": "multi_qkv_static_3track_global",
  "route_formula": "layer_idx % track_count",
  "uses_global_bank": true,
  "track_count": 3,
  "layer_idx": 0,
  "layer": 0,
  "step": 250,
  "last_forward_step": 250,
  "schedule_mode": "train",
  "active_track_index": 0,
  "active_track_counts": {"0": 1024, "1": 0, "2": 0},
  "track_gradient_norm": 0.014,
  "per_track_gradient_norm": {"0": 0.014, "1": 0.0, "2": 0.0},
  "per_track_qkv_weight_norm": {"0": 11.92, "1": 12.04, "2": 11.88},
  "position_routing_enabled": false,
  "eval_freeze_mode": false
}
```

For C, `active_track_index` is `null`, `active_track_counts` must show multiple tracks for length >= 3, and
`position_routing_enabled` is true.

Update `reports/schema/attention_diagnostics.schema.json` without breaking CP diagnostics.

`qkv_track_activity` in `src/attention_lab/queue/mechanism_checks.py` must reject empty, all-zero, or decorative diagnostics.
It must require global-bank evidence, track count 3, per-track weight norms, nonzero per-track gradients, positive
`active_track_counts`, and variant-specific routing evidence.

## Required Destructive Test Script

Add:

```text
scripts/qkv_track_destructive_test.py
```

It must load a config/checkpoint, run a natural forward pass, perturb routing, rerun, and write:

```text
evals/qkv_track_destructive_test.json
```

Required perturbations:

```text
rotate_tracks
force_track
zero_selected
```

Output includes `schema_version`, `experiment_id`, `run_name`, checkpoint/config paths, `num_batches`, perturbation loss/logit
deltas, and `destructive_test_passed`.

Do not write this file unless the script actually ran.

## Required Configs, Hypotheses, Scripts, Reports, And Plan

Create canonical configs:

```text
configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

Use the existing E002 30M/FineWeb-Edu 100M shape and 3000-step budget. B and C require
`multi_qkv_static_3track_global_30m_seed1` as their primary control.

`standard_30m_seed1.yaml` may remain runnable only as a legacy/noncanonical comparison config. It is not part of the four
canonical first-build configs. Validation should report four canonical first-build configs, one legacy/auxiliary runnable
config, five runnable configs total, and six unimplemented old skeleton configs.

Create hypothesis docs:

```text
docs/experiments/E002_multitrack_qkv_shift_register/hypothesis_standard_refactor_control_30m_seed1.md
docs/experiments/E002_multitrack_qkv_shift_register/hypothesis_multi_qkv_static_3track_global_30m_seed1.md
docs/experiments/E002_multitrack_qkv_shift_register/hypothesis_multi_qkv_train_rotation_3track_global_30m_seed1.md
docs/experiments/E002_multitrack_qkv_shift_register/hypothesis_multi_qkv_position_rotation_3track_global_30m_seed1.md
```

Each hypothesis doc must include:

```text
CLAIM:
KILL_CONDITION:
MECHANISM_PROOF:
NEAREST_BORING_EXPLANATION:
CONTROL_THAT_RULES_IT_OUT:
```

Create executable scripts:

```text
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_standard_refactor_control.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_static_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_train_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_position_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_all_full_initial.sh
scripts/experiments/E002_multitrack_qkv_shift_register/compare_initial_full_runs.sh
```

Scripts must use current repo CLIs, reject existing run artifacts, call verify_data/train/verify/eval/summarize/final verify,
and print destructive-test commands for Multi-QKV runs.

Update reports:

```text
reports/experiments/E002_multitrack_qkv_shift_register/README.md
reports/experiments/E002_multitrack_qkv_shift_register/results_template.md
reports/experiments/E002_multitrack_qkv_shift_register/initial_comparison_template.json
```

Do not create fake completed results.

Update:

```text
docs/experiments/E002_multitrack_qkv_shift_register_plan.md
```

It must include A/B/C formulas, global-sharing and bundled-triple requirements, non-goals, canonical run matrix, diagnostics,
manual full-run boundary, manual command sequence, evidence levels, and kill criteria.

## Required Comparison Behavior

Initial E002 comparison must include:

```text
final validation loss
best validation loss
perplexity
tokens/sec
VRAM
total/trainable/attention projection parameters
global qkv bank parameters
HellaSwag result if available
mechanism_check_passed
destructive_test_passed
evidence_level
```

Missing or degenerate diagnostics for A/B/C means `evidence_level = insufficient_evidence`.

## Manual Full-Run Boundary

Allowed during implementation:

```text
unit tests
integration tests
tiny/sanity training tests
config validation
model inspection
script static validation
```

Forbidden during implementation:

```text
running run_all_full_initial.sh
running full 3000-step E002 jobs
approving full queue runs
creating fake run summaries
hand-writing eval metrics
hand-writing destructive test outputs
claiming results
```

## Implementation Order

```text
1. Create the 0901 rebuilt docset.
2. Add config validation fields and tests.
3. Thread step/schedule_mode/position_ids through standard GPT, Block, train, eval, and generation paths.
4. Update standard/CP attention forward signatures to accept and ignore schedule kwargs.
5. Add tests proving standard and CP behavior remain valid.
6. Implement MultiQKVGlobalBank.
7. Implement A static global route.
8. Implement B train rotation with eval freeze.
9. Implement C position rotation.
10. Wire registry.
11. Add diagnostics and schema updates.
12. Update qkv_track_activity mechanism check.
13. Add destructive test script.
14. Add canonical configs.
15. Add hypothesis docs.
16. Add full-run wrapper scripts.
17. Update reports and E002 plan.
18. Run QC.
19. Perform self-review against every doc in the docset.
20. Commit and push only if QC is green.
```

## Required QC Commands

Run:

```bash
uv run ruff check .
uv run pytest
uv run pytest --run-integration
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
```

Inspect all canonical configs:

```bash
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

All must pass before committing.

## Final Self-Review Checklist

```text
[ ] The seven-file docset exists with no extra files.
[ ] A/B/C formulas in docs match code.
[ ] Multi-QKV bank is globally shared.
[ ] A/B use scalar hard routing.
[ ] C uses per-position routing.
[ ] B raises on missing training step.
[ ] B freezes to layer_idx mod 3 during eval/generation.
[ ] Standard output is unchanged by schedule kwargs.
[ ] CP attention still works.
[ ] One-track identity test passes.
[ ] Causal masking tests pass.
[ ] Diagnostics include required fields.
[ ] qkv_track_activity rejects degenerate diagnostics.
[ ] Destructive test script exists and is tested.
[ ] Canonical configs validate.
[ ] Full-run scripts exist and reject existing run dirs.
[ ] Reports do not claim fake results.
[ ] E002 plan documents formulas, diagnostics, run loop, evidence levels, and kill criteria.
[ ] No full 3000-step E002 run was executed during implementation.
[ ] QC is green.
```

## Commit And Push

Only if QC is green:

```bash
git status --short
git add .
git commit -m "Implement initial E002 global multi-QKV rotation experiments"
git push
```

## Final Response Requirements

Include:

```text
- commit hash
- push status
- QC commands run and pass/fail result
- concise files-changed summary
- explicit statement that full 3000-step E002 runs were not executed
- exact manual next commands for the human/operator
```

Use repo-accurate checkpoint paths in handoff commands:

```bash
# From codebase root
git status --short
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register

scripts/experiments/E002_multitrack_qkv_shift_register/run_full_standard_refactor_control.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_static_global.sh

uv run scripts/qkv_track_destructive_test.py \
  --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml \
  --checkpoint runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  --out runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/evals/qkv_track_destructive_test.json \
  --num-batches 4

scripts/experiments/E002_multitrack_qkv_shift_register/run_full_train_rotation_global.sh

uv run scripts/qkv_track_destructive_test.py \
  --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml \
  --checkpoint runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  --out runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1/evals/qkv_track_destructive_test.json \
  --num-batches 4

scripts/experiments/E002_multitrack_qkv_shift_register/run_full_position_rotation_global.sh

uv run scripts/qkv_track_destructive_test.py \
  --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml \
  --checkpoint runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  --out runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1/evals/qkv_track_destructive_test.json \
  --num-batches 4

scripts/experiments/E002_multitrack_qkv_shift_register/compare_initial_full_runs.sh
```

## Result Claim Rule

Do not claim the architecture works. Do not claim A/B/C are stable at full scale. Do not claim validation improvements.

At implementation completion, the only valid claim is:

```text
The initial E002 global Multi-QKV A/B/C experiments are implemented, tested, documented, and ready for manual full-run execution.
```
