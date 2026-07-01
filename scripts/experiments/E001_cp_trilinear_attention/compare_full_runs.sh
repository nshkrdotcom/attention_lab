#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPORT_DIR="reports/experiments/E001_cp_trilinear_attention"
STANDARD_RUN="runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1"
BILINEAR_RUN="runs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1"
TRILINEAR_RUN="runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1"
LAMBDA0_RUN="runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_lambda0_30m_seed1"

require_file() {
  local run_dir="$1"
  local rel_path="$2"
  if [[ ! -f "${run_dir}/${rel_path}" ]]; then
    echo "Missing required E001 comparison artifact: ${run_dir}/${rel_path}" >&2
    echo "Run the promoted full train/eval/summarize/verify and mechanism diagnostic steps before comparing." >&2
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

require_cp_artifacts() {
  local run_dir="$1"
  require_standard_artifacts "${run_dir}"
  require_file "${run_dir}" "evals/attention_diagnostics.jsonl"
  python - "${run_dir}/evals/attention_diagnostics.jsonl" <<'PY'
import json
import sys

path = sys.argv[1]
for line in open(path, encoding="utf-8"):
    row = json.loads(line)
    value = row.get("cp_gradient_norm")
    if value is not None and float(value) > 1e-6:
        raise SystemExit(0)
print(f"CP diagnostics did not show cp_gradient_norm > 1e-6: {path}", file=sys.stderr)
raise SystemExit(1)
PY
}

has_cp_artifacts() {
  local run_dir="$1"
  [[ -f "${run_dir}/evals/run_summary.json" ]] \
    && [[ -f "${run_dir}/evals/val_loss.json" ]] \
    && [[ -f "${run_dir}/evals/hellaswag.json" ]] \
    && [[ -f "${run_dir}/checkpoints/ckpt_last.pt" ]] \
    && [[ -f "${run_dir}/evals/attention_diagnostics.jsonl" ]]
}

require_standard_artifacts "${STANDARD_RUN}"
require_cp_artifacts "${BILINEAR_RUN}"
require_cp_artifacts "${TRILINEAR_RUN}"

uv run scripts/compare_runs.py \
  --experiment E001_cp_trilinear_attention \
  --baseline "${STANDARD_RUN}" \
  --candidate "${BILINEAR_RUN}" \
  --json-out "${REPORT_DIR}/comparison_cp_bilinear_r8_vs_standard.json"

uv run scripts/compare_runs.py \
  --experiment E001_cp_trilinear_attention \
  --baseline "${STANDARD_RUN}" \
  --candidate "${TRILINEAR_RUN}" \
  --json-out "${REPORT_DIR}/comparison_cp_trilinear_r8_vs_standard.json"

if has_cp_artifacts "${LAMBDA0_RUN}"; then
  require_cp_artifacts "${LAMBDA0_RUN}"
  uv run scripts/compare_runs.py \
    --experiment E001_cp_trilinear_attention \
    --baseline "${STANDARD_RUN}" \
    --candidate "${LAMBDA0_RUN}" \
    --json-out "${REPORT_DIR}/comparison_cp_trilinear_lambda0_vs_standard.json"
else
  echo "Skipping lambda0 comparison; lambda0 is a manual promoted control and lacks required full artifacts." >&2
fi

uv run scripts/compare_runs.py \
  --experiment E001_cp_trilinear_attention \
  --baseline "${BILINEAR_RUN}" \
  --candidate "${TRILINEAR_RUN}" \
  --json-out "${REPORT_DIR}/comparison_cp_trilinear_r8_vs_cp_bilinear_r8.json"
