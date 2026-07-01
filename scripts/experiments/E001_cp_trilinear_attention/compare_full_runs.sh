#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPORT_DIR="reports/experiments/E001_cp_trilinear_attention"
STANDARD_RUN="runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1"
BILINEAR_RUN="runs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1"
TRILINEAR_RUN="runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1"
LAMBDA0_RUN="runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_lambda0_30m_seed1"

require_summary() {
  local run_dir="$1"
  if [[ ! -f "${run_dir}/evals/run_summary.json" ]]; then
    echo "Missing run summary: ${run_dir}/evals/run_summary.json" >&2
    echo "Run the corresponding full-run script and summarize step before comparing." >&2
    exit 1
  fi
}

require_summary "${STANDARD_RUN}"
require_summary "${BILINEAR_RUN}"
require_summary "${TRILINEAR_RUN}"
require_summary "${LAMBDA0_RUN}"

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

uv run scripts/compare_runs.py \
  --experiment E001_cp_trilinear_attention \
  --baseline "${STANDARD_RUN}" \
  --candidate "${LAMBDA0_RUN}" \
  --json-out "${REPORT_DIR}/comparison_cp_trilinear_lambda0_vs_standard.json"

uv run scripts/compare_runs.py \
  --experiment E001_cp_trilinear_attention \
  --baseline "${BILINEAR_RUN}" \
  --candidate "${TRILINEAR_RUN}" \
  --json-out "${REPORT_DIR}/comparison_cp_trilinear_r8_vs_cp_bilinear_r8.json"
