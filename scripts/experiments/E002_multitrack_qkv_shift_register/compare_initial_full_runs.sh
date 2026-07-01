#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPORT_DIR="reports/experiments/E002_multitrack_qkv_shift_register"
STANDARD_RUN="runs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1"
STATIC_RUN="runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1"
TRAIN_ROTATION_RUN="runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1"
POSITION_ROTATION_RUN="runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1"

require_file() {
  local run_dir="$1"
  local rel_path="$2"
  if [[ ! -f "${run_dir}/${rel_path}" ]]; then
    echo "Missing required E002 comparison artifact: ${run_dir}/${rel_path}" >&2
    echo "Run the full train/eval/summarize/verify and mechanism diagnostic steps before comparing." >&2
    exit 1
  fi
}

require_standard_artifacts() {
  local run_dir="$1"
  require_file "${run_dir}" "evals/run_summary.json"
  require_file "${run_dir}" "evals/val_loss.json"
  require_file "${run_dir}" "evals/hellaswag.json"
  require_file "${run_dir}" "checkpoints/ckpt_last.pt"
  uv run scripts/verify_run.py \
    --run_dir "${run_dir}" \
    --expect-complete-training \
    --expect-sample \
    --expect-eval-loss \
    --expect-hellaswag \
    --expect-data-manifest >/dev/null
}

require_multi_qkv_artifacts() {
  local run_dir="$1"
  require_standard_artifacts "${run_dir}"
  require_file "${run_dir}" "evals/attention_diagnostics.jsonl"
  require_file "${run_dir}" "evals/qkv_track_destructive_test.json"
  python - "${run_dir}/evals/qkv_track_destructive_test.json" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
if payload.get("destructive_test_passed") is True:
    raise SystemExit(0)
print(f"Multi-QKV destructive test did not pass: {path}", file=sys.stderr)
raise SystemExit(1)
PY
}

require_standard_artifacts "${STANDARD_RUN}"
require_multi_qkv_artifacts "${STATIC_RUN}"
require_multi_qkv_artifacts "${TRAIN_ROTATION_RUN}"
require_multi_qkv_artifacts "${POSITION_ROTATION_RUN}"

uv run scripts/compare_runs.py \
  --experiment E002_multitrack_qkv_shift_register \
  --baseline "${STANDARD_RUN}" \
  --candidate "${STATIC_RUN}" \
  --json-out "${REPORT_DIR}/comparison_static_global_vs_standard_refactor.json"

uv run scripts/compare_runs.py \
  --experiment E002_multitrack_qkv_shift_register \
  --baseline "${STATIC_RUN}" \
  --candidate "${TRAIN_ROTATION_RUN}" \
  --json-out "${REPORT_DIR}/comparison_train_rotation_global_vs_static_global.json"

uv run scripts/compare_runs.py \
  --experiment E002_multitrack_qkv_shift_register \
  --baseline "${STATIC_RUN}" \
  --candidate "${POSITION_ROTATION_RUN}" \
  --json-out "${REPORT_DIR}/comparison_position_rotation_global_vs_static_global.json"
