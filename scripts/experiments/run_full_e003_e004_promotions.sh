#!/usr/bin/env bash
# Promotes the E003/E004 architectures that cleanly passed gauntlet screening
# (differential_qkv_anti_value, scope_gated_qkv, operator_valued_attention) to
# a full 3000-step training run, plus the E004 standard-attention control at
# seed2 (needed so operator_valued_attention has a matched, equally-trained
# control to compare against -- E003's control is reused for free from
# E002's already-completed standard_refactor_control_30m_seed1 checkpoint,
# which matches E003's standard control on the model, data, seed, batch,
# optimizer, LR schedule, and token-budget fields that matter for this
# mechanism-control comparison).
#
# dynamic_value_query_conditioned_attention and q3k3v3_role_routed_attention
# are deliberately NOT included: the former was killed at rung500 on a
# mechanism-diagnostics gate (not loss/stability), the latter never passed
# rung150 on a throughput gate. Both need their own fix (a diagnostic
# rescue, a speed profile/redesign) before a full run is worth it.
#
# Runs entirely in the FOREGROUND. No nohup, no backgrounding (&), and no
# train.py log redirection or pipe wrapper. Run this directly in a terminal
# you intend to leave open (or a tmux/screen session if you want it to survive
# a disconnect) -- everything train.py prints goes straight to your screen,
# live, as it happens, each line timestamped at the source (see
# src/attention_lab/training/train.py's `_ts()` helper).
#
# Does not use `set -e` at the top level: an unattended overnight batch
# should not let one architecture's failure prevent the other three from
# running. Each run is isolated in its own subshell and failures are
# collected and reported at the end.
#
# Training is checkpoint-aware. If a run already has
# checkpoints/ckpt_last.pt, train.py resumes from it; otherwise the run starts
# fresh with --overwrite. This keeps the script compatible with the optional
# second-terminal stall watcher, which kills a wedged train.py process and lets
# this foreground script restart from the latest checkpoint.

set -uo pipefail

cd "$(dirname "$0")/../.."

export ATTENTION_LAB_I_UNDERSTAND_THIS_IS_A_PROMOTED_FULL_RUN=1

FAILURES=()
TRAIN_RESTART_LIMIT="${ATTENTION_LAB_TRAIN_RESTART_LIMIT:-20}"
TRAIN_RESTART_DELAY_SECONDS="${ATTENTION_LAB_TRAIN_RESTART_DELAY_SECONDS:-15}"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

training_already_complete() {
  local run_dir="$1"
  uv run scripts/verify_run.py --run_dir "${run_dir}" \
    --expect-complete-training --expect-sample --expect-data-manifest \
    >/dev/null 2>&1
}

run_training_until_complete() {
  local name="$1"
  local config="$2"
  local run_dir="$3"
  local checkpoint="${run_dir}/checkpoints/ckpt_last.pt"
  local attempt=1
  local status=0

  while true; do
    if training_already_complete "${run_dir}"; then
      echo "[$(ts)] [${name}] training artifacts already verify complete; skipping train.py"
      return 0
    fi

    if [[ -f "${checkpoint}" ]]; then
      echo "[$(ts)] [${name}] train.py attempt ${attempt}/${TRAIN_RESTART_LIMIT}: resuming from ${checkpoint}"
      uv run scripts/train.py --config "${config}" --resume "${checkpoint}"
    else
      echo "[$(ts)] [${name}] train.py attempt ${attempt}/${TRAIN_RESTART_LIMIT}: no checkpoint found; starting fresh with --overwrite"
      uv run scripts/train.py --config "${config}" --overwrite
    fi
    status=$?

    if [[ ${status} -eq 0 ]]; then
      return 0
    fi

    echo "[$(ts)] [${name}] train.py exited with status ${status}"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "[$(ts)] [${name}] no checkpoint exists; cannot resume this run"
      return "${status}"
    fi
    if (( attempt >= TRAIN_RESTART_LIMIT )); then
      echo "[$(ts)] [${name}] restart limit reached; leaving run incomplete"
      return "${status}"
    fi

    attempt=$((attempt + 1))
    echo "[$(ts)] [${name}] retrying from latest checkpoint after ${TRAIN_RESTART_DELAY_SECONDS}s"
    sleep "${TRAIN_RESTART_DELAY_SECONDS}"
  done
}

STARTED_AT="$(ts)"

echo "[${STARTED_AT}] E003/E004 full-run promotion batch starting"

echo "[$(ts)] Verifying data manifest (once, shared across all runs)"
if ! uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes; then
  echo "[$(ts)] Data manifest verification failed; refusing to start promotion batch"
  exit 1
fi

run_architecture() {
  local name="$1"
  local config="$2"
  local run_dir="$3"

  echo "[$(ts)] ==== Starting full run: ${name} ===="
  echo "[$(ts)] config:  ${config}"
  echo "[$(ts)] run_dir: ${run_dir}"

  if (
    set -e
    echo "[$(ts)] [${name}] train.py starting or resuming"
    run_training_until_complete "${name}" "${config}" "${run_dir}"
    echo "[$(ts)] [${name}] train.py finished, running post-training checks"

    uv run scripts/verify_run.py --run_dir "${run_dir}" \
      --expect-complete-training --expect-sample --expect-data-manifest

    uv run scripts/eval_loss.py \
      --checkpoint "${run_dir}/checkpoints/ckpt_last.pt" \
      --data_root data/fineweb_edu_100m

    uv run scripts/eval_generate.py \
      --checkpoint "${run_dir}/checkpoints/ckpt_last.pt" \
      --prompt "The history of mathematics"

    uv run scripts/eval_hellaswag.py \
      --checkpoint "${run_dir}/checkpoints/ckpt_last.pt" \
      --max_examples 100

    uv run scripts/summarize_run.py --run_dir "${run_dir}"

    uv run scripts/verify_run.py --run_dir "${run_dir}" \
      --expect-complete-training --expect-sample --expect-eval-loss \
      --expect-hellaswag --expect-data-manifest
  ); then
    echo "[$(ts)] [${name}] COMPLETE"
  else
    echo "[$(ts)] [${name}] FAILED -- see output above. Continuing to next architecture."
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

FINISHED_AT="$(ts)"
echo "[${FINISHED_AT}] Batch finished (started: ${STARTED_AT})"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  echo "[${FINISHED_AT}] All four runs completed successfully."
else
  echo "[${FINISHED_AT}] FAILED (${#FAILURES[@]}): ${FAILURES[*]}"
  echo "[${FINISHED_AT}] Re-run just the failed ones by copying their run_architecture block into a new script."
fi
