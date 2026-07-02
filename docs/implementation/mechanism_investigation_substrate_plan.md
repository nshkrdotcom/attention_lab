# Mechanism Investigation Substrate Plan

Status: implemented as native infrastructure with derived backfill inventories.

This plan records the Phase 0 audit and the mechanism substrate scope for moving E001-E004 from survival screening toward reproducible mechanism investigation. It is not a scientific result and does not claim any new mechanism interpretation.

## Existing Artifact Locations

Backfill reads source evidence from these locations:

```text
configs/experiments/E001_cp_trilinear_attention/
configs/experiments/E002_multitrack_qkv_shift_register/
configs/experiments/E003_qkv_architecture_gauntlet/
configs/experiments/E004_operator_binding_qkv_gauntlet/
reports/experiments/E001_cp_trilinear_attention/
reports/experiments/E002_multitrack_qkv_shift_register/
reports/experiments/E003_qkv_architecture_gauntlet/
reports/experiments/E004_operator_binding_qkv_gauntlet/
runs/experiments/E001_cp_trilinear_attention/
runs/experiments/E002_multitrack_qkv_shift_register/
runs/screen/
data/queue.db
```

Derived mechanism artifacts are written only under:

```text
reports/mechanisms/backfill/
reports/mechanisms/cross_experiment_candidate_report.md
reports/mechanisms/probes/
```

Historical run directories and historical experiment reports remain read-only evidence inputs.

## Expected Checkpoint Locations

The checkpoint convention is:

```text
<run_dir>/checkpoints/ckpt_last.pt
```

Current backfill inventories found checkpoints for:

```text
E001:
  standard_30m_seed1
  cp_bilinear_r8_30m_seed1
  cp_trilinear_r8_30m_seed1

E002:
  standard_refactor_control_30m_seed1
  multi_qkv_static_3track_global_30m_seed1
  multi_qkv_train_rotation_3track_global_30m_seed1
  multi_qkv_position_rotation_3track_global_30m_seed1

E003:
  standard_refactor_control_30m_seed1_rung020/rung150/rung500
  differential_qkv_anti_value_30m_seed1_rung020/rung150/rung500
  scope_gated_qkv_30m_seed1_rung020/rung150/rung500

E004:
  standard_refactor_control_30m_seed2_rung020/rung150/rung500
  operator_valued_attention_30m_seed2_rung020/rung150/rung500
  dynamic_value_query_conditioned_attention_30m_seed2_rung020/rung150/rung500
  q3k3v3_role_routed_attention_30m_seed2_rung020
```

Base E003/E004 full-run configs do not currently have full-run checkpoints. They should be treated as planned/not available, even when a rung checkpoint exists for a derived rung config.

## Backfill Levels

The backfill system distinguishes:

```text
artifact_summary:
  Existing configs, reports, diagnostics, run summaries, promotion reports, or gauntlet reports can be summarized.

checkpoint_recompute:
  A checkpoint exists, so activations and interventions can be recomputed for new small prompt batches.

not_available:
  No checkpoint or required artifact exists. Historical activations cannot be recovered.
```

Current summary:

```text
checkpoint_recompute:
  E001 standard/cp_bilinear/cp_trilinear completed runs.
  E002 canonical standard/static/train-rotation/position-rotation completed runs.
  E003 rung020/rung150/rung500 screen checkpoints.
  E004 rung020/rung150/rung500 screen checkpoints where those rungs ran.

not_available:
  E001 cp_trilinear_r8_lambda0_30m_seed1 and standard_refactor_control_30m_seed1.
  E002 old skeleton configs and legacy standard_30m_seed1.
  E003 base full-run configs without rung suffix.
  E004 base full-run configs without rung suffix.
```

Every unavailable item is emitted in the derived `missing_artifacts.md` files as `checkpoint_unavailable`, `missing`, or `not_recorded`.

## Hook Sites Supported In This Task

Standard sites:

```text
resid_pre[layer]
attn_q[layer]
attn_k[layer]
attn_v[layer]
attn_out[layer]
resid_mid[layer]
mlp_out[layer]
resid_post[layer]
logits
```

Architecture-specific sites:

```text
operator_valued_attention:
  operator_probs[layer]
  operator_add_out[layer]
  operator_suppress_out[layer]
  operator_gate_out[layer]
  operator_transform_out[layer]
  operator_bind_out[layer]
  operator_combined_out[layer]

differential_qkv_anti_value:
  pos_q[layer]
  pos_k[layer]
  pos_v[layer]
  neg_q[layer]
  neg_k[layer]
  neg_v[layer]
  pos_out[layer]
  neg_out[layer]
  branch_delta[layer]
  lambda[layer]

scope_gated_qkv:
  content_out[layer]
  scope_out[layer]
  gate[layer]
  content_scope_product[layer]
  gated_content[layer]

multi_qkv_static_3track_global / multi_qkv_train_rotation_3track_global / multi_qkv_position_rotation_3track_global:
  track_q[layer, track]
  track_k[layer, track]
  track_v[layer, track]
  selected_track[layer]
  track_out[layer]

cp_bilinear / cp_trilinear:
  cp_score[layer]
  cp_output[layer]
  cp_lambda[layer]
  cp_rank_component[layer, rank] is declared but full tensor capture is unsupported until an optimized path exists.

dynamic_value_query_conditioned_attention:
  static_value_content[layer]
  dynamic_gate[layer]
  dynamic_delta[layer]
  dynamic_value_output[layer]

q3k3v3_role_routed_attention:
  content_out[layer]
  operator_out[layer]
  binding_out[layer]
  content_operator_product[layer]
  content_binding_product[layer]
  operator_binding_product[layer]
```

Hook-site docs are generated deterministically from `attention_lab.mechanisms.hook_sites`.

## Intervention Operations Supported In This Task

Native interventions are implemented through `run_with_interventions`:

```text
zero:
  replace a site with zeros_like(site_tensor)

mean_ablate:
  replace a site with its batch/token mean broadcast back to the original shape

replace:
  replace a site with a shape-compatible explicit tensor or cache tensor

scale:
  multiply a site by a scalar

patch_from_cache:
  replace a full site, or selected batch/token positions, from a compatible ActivationCache
```

Patch compatibility validates attention type and tensor shape. Missing or failed intervention sites are returned explicitly.

## Post-Hoc Probe Workflow

Use a checkpoint-backed candidate only:

```bash
uv run scripts/run_mechanism_probe.py \
  --config configs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2_rung500.yaml \
  --checkpoint runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt \
  --prompt "The history of mathematics" \
  --sites operator_probs,operator_suppress_out,operator_bind_out \
  --interventions zero \
  --output-dir reports/mechanisms/probes/E004_operator_valued_rung500
```

The probe writes:

```text
activation_summary.json
intervention_summary.json
probe_report.md
```

It recomputes activations from checkpoints and prompt batches. It does not recover historical activations that were never saved.

## TransformerLens Boundary

TransformerLens compatibility is not required for this substrate. The native cache preserves architecture-specific names and tensor semantics first. A thin adapter can be added later only for clear standard equivalents such as `resid_pre`, `attn_q`, `attn_k`, `attn_v`, `attn_out`, `resid_post`, and `logits`.
