# E003 QKV Architecture Gauntlet Plan

Status: implementation prepared; no training evidence is claimed until real gauntlet screen artifacts exist.

## Hypothesis

Standard Q/K/V may be a bottleneck for feature separation and mechanistic legibility. E003 tests whether nonstandard QKV decompositions can remain trainable while exposing separable, nondegenerate mechanism diagnostics.

This is not an efficiency experiment. Throughput and VRAM are safety gates, not the main claim target.

## First Variants

- `differential_qkv_anti_value`: positive and negative/suppressive QKV branches with `Y = Y_pos - lambda * Y_neg`.
- `scope_gated_qkv`: standard Q/K/V plus an attended scope/operator stream and receiver-side write gate.
- `standard_refactor_control_30m_seed1`: standard-attention control for shared-path changes and gauntlet ratios.

## Fixed Contract

Direct comparisons hold fixed the FineWeb-Edu 100M manifest, GPT-2 tokenizer, seed, batch construction, optimizer, learning-rate schedule, model scale, validation cadence, checkpoint cadence, and verification path.

Core base budget:

```text
block_size: 1024
n_layer: 6
n_head: 6
n_embd: 384
B: 4
T: 1024
total_batch_size: 262144
max_steps: 3000
```

The gauntlet generates screen-rung configs that override step, validation, save, and diagnostics cadence. It does not overwrite base full-run configs.

## Gauntlet Rungs

The committed policy is:

```text
rung020 -> rung150 -> rung500
```

Each rung is a real executed `SCREEN` job with its own generated config and run directory. `SANITY` is not used as evidence.

Advancement is decided from structured promotion reports, ledger rows, checkpoints, metrics, and diagnostics. The operator should not need to eyeball loss curves to decide whether a candidate advances.

Policy gates include:

- no NaN/Inf
- expected steps reached
- checkpoint present
- loss descended
- diagnostics present and nondegenerate for non-standard attention
- mechanism activity check passed
- loss, speed, and VRAM not catastrophic relative to control when a same-rung control report exists

## Full-Run Boundary

Full 3000-step runs are not launched by default. If all screen rungs pass, the gauntlet report marks:

```text
ready_for_manual_full_promotion: true
```

Full runs still require clean promotion evidence and existing queue approval preflight. The gauntlet must not set `full_run_approved` directly.

## Operator Commands

```bash
uv run scripts/validate_experiment.py --id E003_qkv_architecture_gauntlet
uv run attn-queue gauntlet-plan --experiment E003_qkv_architecture_gauntlet --policy configs/experiments/E003_qkv_architecture_gauntlet/gauntlet_policy.yaml
uv run attn-queue gauntlet-run --experiment E003_qkv_architecture_gauntlet --policy configs/experiments/E003_qkv_architecture_gauntlet/gauntlet_policy.yaml --once
uv run attn-queue gauntlet-report --experiment E003_qkv_architecture_gauntlet
```

Use `scripts/experiments/E003_qkv_architecture_gauntlet/run_gauntlet.sh` for the full preflight plus serial gauntlet workflow.

## Claim Boundary

At implementation completion, the valid claim is only:

```text
E003 QKV gauntlet code, configs, diagnostics, and docs are implemented and ready for screen-first execution.
```

No scientific result exists until actual screen or full-run artifacts pass the required checks.
