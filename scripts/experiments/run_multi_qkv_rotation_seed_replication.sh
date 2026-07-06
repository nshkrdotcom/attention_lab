#!/usr/bin/env bash
# Replicates multi_qkv_static and multi_qkv_train_rotation to 2 more seeds
# each (1338, 1339; seed1 already exists) to test whether a real spelunking
# finding is robust or a single-seed fluke: on a synthetic induction probe,
# multi_qkv_train_rotation scored an exact 0.0% across 5 trials while
# multi_qkv_static scored 24.2%, despite both reducing to the IDENTICAL
# layer_idx % track_count routing formula at eval time (verified directly
# against both source files -- see docs/mechanisms/spelunking_toolkit.md).
# The leading hypothesis is that train_rotation's per-track weights end up
# less specialized because each track was trained under a shifting role
# assignment rather than one fixed one -- this script's job is to find out
# whether that dissociation replicates across seeds before trusting it as
# a real architectural effect.
#
# Runs entirely in the FOREGROUND, synchronously, in this terminal. No
# nohup, no backgrounding. Leave this terminal open (or use tmux/screen if
# you want it to survive a disconnect).
#
# Estimated cost: ~5h/run x 4 runs =~ 20h total (multi-QKV variants ran at
# ~5h each for a comparable config last session). Run this only if you want
# that follow-up question answered; it isn't required for anything else in
# this pass.

set -uo pipefail

cd "$(dirname "$0")/../.."

export ATTENTION_LAB_I_UNDERSTAND_THIS_IS_A_PROMOTED_FULL_RUN=1

FAILURES=()

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

training_already_complete() {
  local run_dir="$1"
  uv run scripts/verify_run.py --run_dir "${run_dir}" \
    --expect-complete-training --expect-sample --expect-data-manifest \
    >/dev/null 2>&1
}

run_architecture() {
  local name="$1"
  local config="$2"
  local run_dir="$3"
  local checkpoint="${run_dir}/checkpoints/ckpt_last.pt"

  echo "[$(ts)] ==== Starting: ${name} ===="
  echo "[$(ts)] config:  ${config}"
  echo "[$(ts)] run_dir: ${run_dir}"

  if training_already_complete "${run_dir}"; then
    echo "[$(ts)] [${name}] training artifacts already verify complete; skipping"
    return 0
  fi

  if (
    set -e
    if [[ -f "${checkpoint}" ]]; then
      echo "[$(ts)] [${name}] resuming from ${checkpoint}"
      uv run scripts/train.py --config "${config}" --resume "${checkpoint}"
    else
      echo "[$(ts)] [${name}] no checkpoint found; starting fresh with --overwrite"
      uv run scripts/train.py --config "${config}" --overwrite
    fi

    uv run scripts/verify_run.py --run_dir "${run_dir}" \
      --expect-complete-training --expect-sample --expect-data-manifest

    uv run scripts/eval_loss.py \
      --checkpoint "${checkpoint}" --data_root data/fineweb_edu_100m

    uv run scripts/summarize_run.py --run_dir "${run_dir}"
  ); then
    echo "[$(ts)] [${name}] COMPLETE"
  else
    echo "[$(ts)] [${name}] FAILED -- see output above. Continuing to next architecture."
    FAILURES+=("${name}")
  fi
}

STARTED_AT="$(ts)"
echo "[${STARTED_AT}] multi_qkv rotation seed-replication batch starting"

echo "[$(ts)] Verifying data manifest (once, shared across all runs)"
if ! uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes; then
  echo "[$(ts)] Data manifest verification failed; refusing to start"
  exit 1
fi

run_architecture \
  "multi_qkv_static seed1338" \
  "configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1338.yaml" \
  "runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1338"

run_architecture \
  "multi_qkv_static seed1339" \
  "configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1339.yaml" \
  "runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1339"

run_architecture \
  "multi_qkv_train_rotation seed1338" \
  "configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1338.yaml" \
  "runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1338"

run_architecture \
  "multi_qkv_train_rotation seed1339" \
  "configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1339.yaml" \
  "runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1339"

FINISHED_AT="$(ts)"
echo "[${FINISHED_AT}] Batch finished (started: ${STARTED_AT})"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  echo "[${FINISHED_AT}] All four runs completed successfully."
  echo "[${FINISHED_AT}] Next: re-run scripts/spelunk_checkpoint.py --induction-probe-pattern-len 20 against"
  echo "[${FINISHED_AT}] each new checkpoint (5 seeds each, as in the original sweep) to see if the"
  echo "[${FINISHED_AT}] static-vs-train_rotation dissociation replicates."
else
  echo "[${FINISHED_AT}] FAILED (${#FAILURES[@]}): ${FAILURES[*]}"
fi
