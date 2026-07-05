#!/usr/bin/env bash
# Promotes the E003/E004 architectures that cleanly passed gauntlet screening
# (differential_qkv_anti_value, scope_gated_qkv, operator_valued_attention) to
# a full 3000-step training run, plus the E004 standard-attention control at
# seed2 (needed so operator_valued_attention has a matched, equally-trained
# control to compare against -- E003's control is reused for free from
# E002's already-completed standard_refactor_control_30m_seed1 checkpoint,
# since that config is identical to E003's control config).
#
# dynamic_value_query_conditioned_attention and q3k3v3_role_routed_attention
# are deliberately NOT included: the former was killed at rung500 on a
# mechanism-diagnostics gate (not loss/stability), the latter never passed
# rung150 on a throughput gate. Both need their own fix (a diagnostic
# rescue, a speed profile/redesign) before a full run is worth it.
#
# Does not use `set -e` at the top level: an unattended overnight batch
# should not let one architecture's failure prevent the other three from
# running. Each run is isolated in its own subshell and failures are
# collected and reported at the end.

set -uo pipefail

cd "$(dirname "$0")/../.."

export ATTENTION_LAB_I_UNDERSTAND_THIS_IS_A_PROMOTED_FULL_RUN=1

FAILURES=()
STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo "================================================================"
echo "E003/E004 full-run promotion batch starting: ${STARTED_AT}"
echo "================================================================"

echo ""
echo "=== Verifying data manifest (once, shared across all runs) ==="
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes

run_architecture() {
  local name="$1"
  local config="$2"
  local run_dir="$3"

  echo ""
  echo "################################################################"
  echo "### Starting full run: ${name}"
  echo "### Config:  ${config}"
  echo "### Run dir: ${run_dir}"
  echo "### Started: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "################################################################"

  if (
    set -e
    echo "--- [${name}] train.py ---"
    uv run scripts/train.py --config "${config}" --overwrite

    echo "--- [${name}] verify_run.py (post-training) ---"
    uv run scripts/verify_run.py --run_dir "${run_dir}" \
      --expect-complete-training --expect-sample --expect-data-manifest

    echo "--- [${name}] eval_loss.py ---"
    uv run scripts/eval_loss.py \
      --checkpoint "${run_dir}/checkpoints/ckpt_last.pt" \
      --data_root data/fineweb_edu_100m

    echo "--- [${name}] eval_generate.py ---"
    uv run scripts/eval_generate.py \
      --checkpoint "${run_dir}/checkpoints/ckpt_last.pt" \
      --prompt "The history of mathematics"

    echo "--- [${name}] eval_hellaswag.py ---"
    uv run scripts/eval_hellaswag.py \
      --checkpoint "${run_dir}/checkpoints/ckpt_last.pt" \
      --max_examples 100

    echo "--- [${name}] summarize_run.py ---"
    uv run scripts/summarize_run.py --run_dir "${run_dir}"

    echo "--- [${name}] verify_run.py (final, full expectations) ---"
    uv run scripts/verify_run.py --run_dir "${run_dir}" \
      --expect-complete-training --expect-sample --expect-eval-loss \
      --expect-hellaswag --expect-data-manifest
  ); then
    echo "=== [${name}] COMPLETE: $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
  else
    echo "!!! [${name}] FAILED -- see output above. Continuing to next architecture. !!!" >&2
    FAILURES+=("${name}")
  fi
}

run_architecture \
  "differential_qkv_anti_value (E003)" \
  "configs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1.yaml" \
  "runs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1"

run_architecture \
  "scope_gated_qkv (E003)" \
  "configs/experiments/E003_qkv_architecture_gauntlet/scope_gated_qkv_30m_seed1.yaml" \
  "runs/experiments/E003_qkv_architecture_gauntlet/scope_gated_qkv_30m_seed1"

run_architecture \
  "operator_valued_attention (E004)" \
  "configs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2.yaml" \
  "runs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2"

run_architecture \
  "standard_refactor_control_30m_seed2 (E004's matched control -- E003 reuses E002's existing seed1 control checkpoint instead of retraining)" \
  "configs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2.yaml" \
  "runs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2"

FINISHED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""
echo "================================================================"
echo "Batch finished: ${FINISHED_AT} (started: ${STARTED_AT})"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  echo "All four runs completed successfully."
else
  echo "FAILED (${#FAILURES[@]}): ${FAILURES[*]}"
  echo "Re-run just the failed ones by copying their run_architecture block into a new script."
fi
echo "================================================================"
