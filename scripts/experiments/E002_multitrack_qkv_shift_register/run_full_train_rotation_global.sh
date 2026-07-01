#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

if [[ "${ATTENTION_LAB_I_UNDERSTAND_THIS_IS_A_PROMOTED_FULL_RUN:-}" != "1" ]]; then
  echo "Refusing direct full run. Use the screen-first queue workflow and approve a clean promotion report first." >&2
  echo "Set ATTENTION_LAB_I_UNDERSTAND_THIS_IS_A_PROMOTED_FULL_RUN=1 only for a manually promoted full run." >&2
  exit 2
fi

CONFIG="configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml"
RUN_DIR="runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1"

if [[ -d "${RUN_DIR}/checkpoints" || -f "${RUN_DIR}/metrics.jsonl" || -f "${RUN_DIR}/evals/run_summary.json" || -f "${RUN_DIR}/data_manifest.json" ]]; then
  echo "Refusing to overwrite existing run artifacts in ${RUN_DIR}" >&2
  exit 1
fi

uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
uv run scripts/train.py --config "${CONFIG}" --overwrite
uv run scripts/verify_run.py --run_dir "${RUN_DIR}" --expect-complete-training --expect-sample --expect-data-manifest
uv run scripts/eval_loss.py --checkpoint "${RUN_DIR}/checkpoints/ckpt_last.pt" --data_root data/fineweb_edu_100m
uv run scripts/eval_generate.py --checkpoint "${RUN_DIR}/checkpoints/ckpt_last.pt" --prompt "The history of mathematics"
uv run scripts/eval_hellaswag.py --checkpoint "${RUN_DIR}/checkpoints/ckpt_last.pt" --max_examples 100
uv run scripts/summarize_run.py --run_dir "${RUN_DIR}"
uv run scripts/verify_run.py --run_dir "${RUN_DIR}" --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag --expect-data-manifest

echo
echo "Full run completed and verified: ${RUN_DIR}"
echo "Next destructive test:"
echo "uv run scripts/qkv_track_destructive_test.py --config ${CONFIG} --checkpoint ${RUN_DIR}/checkpoints/ckpt_last.pt --out ${RUN_DIR}/evals/qkv_track_destructive_test.json --num-batches 4"
