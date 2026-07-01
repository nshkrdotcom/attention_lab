#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

EXPERIMENT="E003_qkv_architecture_gauntlet"
REPORT_JSON="reports/experiments/${EXPERIMENT}/gauntlet_report.json"

if [[ "${1:-}" == "--full" ]]; then
  for run_name in \
    standard_refactor_control_30m_seed1 \
    differential_qkv_anti_value_30m_seed1 \
    scope_gated_qkv_30m_seed1; do
    run_dir="runs/experiments/${EXPERIMENT}/${run_name}"
    if [[ ! -f "${run_dir}/evals/run_summary.json" ]]; then
      echo "Missing verified full-run summary: ${run_dir}/evals/run_summary.json" >&2
      echo "Use the gauntlet report for screen-rung comparison until full runs exist." >&2
      exit 2
    fi
  done
  uv run scripts/compare_runs.py \
    --experiment "${EXPERIMENT}" \
    --baseline "runs/experiments/${EXPERIMENT}/standard_refactor_control_30m_seed1" \
    --candidate "runs/experiments/${EXPERIMENT}/differential_qkv_anti_value_30m_seed1" \
    --json-out "reports/experiments/${EXPERIMENT}/comparison_differential_vs_standard.json"
  uv run scripts/compare_runs.py \
    --experiment "${EXPERIMENT}" \
    --baseline "runs/experiments/${EXPERIMENT}/standard_refactor_control_30m_seed1" \
    --candidate "runs/experiments/${EXPERIMENT}/scope_gated_qkv_30m_seed1" \
    --json-out "reports/experiments/${EXPERIMENT}/comparison_scope_gated_vs_standard.json"
  exit 0
fi

if [[ ! -f "${REPORT_JSON}" ]]; then
  echo "Missing gauntlet screen report: ${REPORT_JSON}" >&2
  echo "Run scripts/experiments/${EXPERIMENT}/run_gauntlet.sh first." >&2
  exit 2
fi

uv run scripts/experiments/${EXPERIMENT}/summarize_gauntlet.py
