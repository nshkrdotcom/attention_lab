#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/holiday_backfill configs/mechanisms reports/mechanisms/probes
LOG="logs/holiday_backfill/e001_e004_backdata_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== START $(date -u) ==="
git rev-parse HEAD || true

uv sync
uv run scripts/verify_cuda.py
uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes

uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
uv run scripts/validate_experiment.py --id E003_qkv_architecture_gauntlet
uv run scripts/validate_experiment.py --id E004_operator_binding_qkv_gauntlet

cat > configs/mechanisms/quick_probe_prompts.txt <<'PROMPTS'
The history of mathematics
The dog is not friendly.
The dog is friendly.
The dog is not only friendly but loyal.
The patient is not stable.
The doctor did not say the patient was stable.
PROMPTS

uv run scripts/backfill_mechanism_inventory.py --experiments E004 E003 E002 E001 --repo-root . --output-root reports/mechanisms/backfill
uv run scripts/compare_mechanism_candidates.py --backfill-root reports/mechanisms/backfill --output reports/mechanisms/cross_experiment_candidate_report.md

python - <<'PY'
import json
from pathlib import Path

root = Path(".")
bad = []
for inv_path in sorted(Path("reports/mechanisms/backfill").glob("*/inventory.json")):
    inv = json.loads(inv_path.read_text())
    for row in inv["candidates"]:
        cp = row.get("checkpoint_path")
        status = row.get("checkpoint_status")
        run = row.get("run_name")
        if status == "available" and (not cp or not (root / cp).exists()):
            bad.append((str(inv_path), run, status, cp))
        if status != "available" and cp:
            bad.append((str(inv_path), run, status, cp))
if bad:
    print("BAD CHECKPOINT STATUS")
    for x in bad:
        print("\t".join(map(str, x)))
    raise SystemExit(1)
print("OK: every available checkpoint path exists on disk")
PY

probe() {
  local config="$1"
  local checkpoint="$2"
  local out="$3"
  local sites="$4"
  local intervention_sites="${5:-}"

  if [[ -n "$intervention_sites" ]]; then
    uv run scripts/run_mechanism_probe.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --prompts-file configs/mechanisms/quick_probe_prompts.txt \
      --sites "$sites" \
      --intervention-sites "$intervention_sites" \
      --interventions zero,scale \
      --layer 0 \
      --scale 0.0 \
      --output-dir "$out" \
      --device cuda
  else
    uv run scripts/run_mechanism_probe.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --prompts-file configs/mechanisms/quick_probe_prompts.txt \
      --sites "$sites" \
      --interventions zero,scale \
      --layer 0 \
      --scale 0.0 \
      --output-dir "$out" \
      --device cuda
  fi
}

probe_capture_only() {
  local config="$1"
  local checkpoint="$2"
  local out="$3"
  local sites="$4"

  uv run scripts/run_mechanism_probe.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --prompts-file configs/mechanisms/quick_probe_prompts.txt \
    --sites "$sites" \
    --layer 0 \
    --output-dir "$out" \
    --device cuda
}

# E001
probe configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml \
  runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E001_standard_quick \
  attn_q,attn_k,attn_v,attn_out

probe configs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1.yaml \
  runs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E001_cp_bilinear_quick \
  attn_q,attn_k,attn_v,attn_out,cp_score,cp_output,cp_lambda

probe configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1.yaml \
  runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E001_cp_trilinear_quick \
  attn_q,attn_k,attn_v,attn_out,cp_score,cp_output,cp_lambda

# E002
probe configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml \
  runs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E002_standard_refactor_control_quick \
  attn_q,attn_k,attn_v,attn_out

probe configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml \
  runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E002_static_global_quick \
  selected_track,track_q,track_k,track_v,track_out \
  track_q,track_k,track_v,track_out

probe configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml \
  runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E002_train_rotation_quick \
  selected_track,track_q,track_k,track_v,track_out \
  track_q,track_k,track_v,track_out

probe configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml \
  runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E002_position_rotation_quick \
  selected_track,track_q,track_k,track_v,track_out \
  track_q,track_k,track_v,track_out

probe_capture_only configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml \
  runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E002_position_rotation_capture_quick \
  selected_track,track_q,track_k,track_v,track_out

# E003 rung500
probe configs/experiments/E003_qkv_architecture_gauntlet/standard_refactor_control_30m_seed1_rung500.yaml \
  runs/screen/standard_refactor_control_30m_seed1_rung500_7752266a764e/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E003_standard_rung500_quick \
  attn_q,attn_k,attn_v,attn_out

probe configs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1_rung500.yaml \
  runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E003_differential_rung500_inventory_path \
  pos_q,pos_k,pos_v,neg_q,neg_k,neg_v,pos_out,neg_out,branch_delta,lambda

probe configs/experiments/E003_qkv_architecture_gauntlet/scope_gated_qkv_30m_seed1_rung500.yaml \
  runs/screen/scope_gated_qkv_30m_seed1_rung500_bb3de557aae8/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E003_scope_gated_rung500_inventory_path \
  content_out,scope_out,gate,content_scope_product,gated_content

# E004 rung500 / q3 rung020
probe configs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2_rung500.yaml \
  runs/screen/standard_refactor_control_30m_seed2_rung500_3cc31db15c20/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E004_standard_rung500_quick \
  attn_q,attn_k,attn_v,attn_out

probe configs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2_rung500.yaml \
  runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E004_operator_valued_rung500_inventory_path \
  operator_probs,operator_add_out,operator_suppress_out,operator_gate_out,operator_transform_out,operator_bind_out,operator_combined_out

probe configs/experiments/E004_operator_binding_qkv_gauntlet/dynamic_value_query_conditioned_attention_30m_seed2_rung500.yaml \
  runs/screen/dynamic_value_query_conditioned_attention_30m_seed2_rung500_99b5756e77ed/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E004_dynamic_value_rung500_diagnostic_rescue \
  static_value_content,dynamic_gate,dynamic_delta,dynamic_value_output

probe configs/experiments/E004_operator_binding_qkv_gauntlet/q3k3v3_role_routed_attention_30m_seed2_rung020.yaml \
  runs/screen/q3k3v3_role_routed_attention_30m_seed2_rung020_e640cc594862/checkpoints/ckpt_last.pt \
  reports/mechanisms/probes/E004_q3k3v3_rung020_quick \
  content_out,operator_out,binding_out,content_operator_product,content_binding_product,operator_binding_product

uv run scripts/backfill_mechanism_inventory.py --experiments E004 E003 E002 E001 --repo-root . --output-root reports/mechanisms/backfill
uv run scripts/compare_mechanism_candidates.py --backfill-root reports/mechanisms/backfill --output reports/mechanisms/cross_experiment_candidate_report.md

uv run ruff check .
uv run pytest

echo "=== PROBES ==="
find reports/mechanisms/probes -maxdepth 2 -type f | sort

echo "=== REPORT ==="
sed -n '1,280p' reports/mechanisms/cross_experiment_candidate_report.md

echo "=== GIT ==="
git status --short
git diff --stat

echo "=== END $(date -u) ==="
