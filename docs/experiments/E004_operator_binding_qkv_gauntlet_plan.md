# E004 Operator-Binding QKV Gauntlet Plan

Status: implementation prepared; no E004 training evidence is claimed until real gauntlet screen artifacts exist.

## Claim Boundary

E004 is not an efficiency experiment. E004 is not a model-improvement claim. It tests whether alternative attention blocks can survive early pretraining while exposing inspectable internal mechanisms.

The screen question is:

```text
Can these architectures train without collapse while producing nondegenerate, inspectable internal mechanisms?
```

Passing a screen rung means only that the candidate remained stable enough to inspect under the configured rung. It does not mean the architecture is better than standard attention and does not imply semantic specialization.

## Variants

E004 uses the E003 gauntlet infrastructure with one standard control and three candidates:

```text
standard_refactor_control_30m_seed2
operator_valued_attention_30m_seed2
q3k3v3_role_routed_attention_30m_seed2
dynamic_value_query_conditioned_attention_30m_seed2
```

The implemented attention types are:

```text
operator_valued_attention
q3k3v3_role_routed_attention
dynamic_value_query_conditioned_attention
```

## Mechanism Hypotheses

`operator_valued_attention` tests whether retrieved content can be routed through multiple fixed update modes, including an explicitly negative suppressive contribution.

`q3k3v3_role_routed_attention` tests whether content-like, operator-like, and binding-like Q/K/V role streams can remain simultaneously active and separable.

`dynamic_value_query_conditioned_attention` tests whether a receiver token can use a dynamic read-mode gate to reinterpret the same retrieved value content.

These are architectural probes. The diagnostic names are labels for inspection, not claims that the model has learned logical operators, semantic binding, or scope.

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

## Gauntlet Rungs

The gauntlet decides advancement from `rung020` to `rung150` to `rung500` using structured metrics, checkpoints, promotion reports, and mechanism diagnostics:

```text
rung020 -> rung150 -> rung500
```

Each rung is a real `SCREEN` job with its own generated config and run directory. Full 3000-step runs are not run by default.

## Full-Run Boundary

Full runs require the existing promotion-report and full-run approval preflight. The gauntlet must not set `full_run_approved` directly. If all screen rungs pass, the report can mark a candidate as ready for manual full promotion, but the operator still approves through the queue full-run gate.

## Operator Commands

```bash
uv run scripts/validate_experiment.py --id E004_operator_binding_qkv_gauntlet
uv run attn-queue gauntlet-plan --experiment E004_operator_binding_qkv_gauntlet --policy configs/experiments/E004_operator_binding_qkv_gauntlet/gauntlet_policy.yaml
uv run attn-queue gauntlet-run --experiment E004_operator_binding_qkv_gauntlet --policy configs/experiments/E004_operator_binding_qkv_gauntlet/gauntlet_policy.yaml --once
uv run attn-queue gauntlet-report --experiment E004_operator_binding_qkv_gauntlet
```

Use `scripts/experiments/E004_operator_binding_qkv_gauntlet/run_gauntlet.sh` for the full preflight plus serial gauntlet workflow.
