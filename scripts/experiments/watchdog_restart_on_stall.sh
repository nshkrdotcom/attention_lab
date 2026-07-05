#!/usr/bin/env bash
# Optional second-terminal stall watcher for the E003/E004 full-run promotion
# batch. It does not launch training itself. Run it alongside
# run_full_e003_e004_promotions.sh; when metrics.jsonl stops updating, this
# script captures live diagnostics, kills the stuck train.py process, and lets
# the foreground promotion script resume from ckpt_last.pt.

set -uo pipefail

cd "$(dirname "$0")/../.."

POLL_SECONDS="${ATTENTION_LAB_STALL_POLL_SECONDS:-60}"
STALE_SECONDS="${ATTENTION_LAB_STALL_SECONDS:-240}"
TERM_GRACE_SECONDS="${ATTENTION_LAB_STALL_TERM_GRACE_SECONDS:-20}"
DRY_RUN="${ATTENTION_LAB_STALL_DRY_RUN:-0}"

TARGET_NAMES=(
  "differential_qkv_anti_value (E003)"
  "scope_gated_qkv (E003)"
  "operator_valued_attention (E004)"
  "standard_refactor_control_30m_seed2 (E004)"
)

TARGET_CONFIGS=(
  "configs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1.yaml"
  "configs/experiments/E003_qkv_architecture_gauntlet/scope_gated_qkv_30m_seed1.yaml"
  "configs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2.yaml"
  "configs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2.yaml"
)

TARGET_RUN_DIRS=(
  "runs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1"
  "runs/experiments/E003_qkv_architecture_gauntlet/scope_gated_qkv_30m_seed1"
  "runs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2"
  "runs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2"
)

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

train_python_pid_for_config() {
  local config="$1"
  ps -eo pid=,ppid=,comm=,args= | awk -v config="${config}" '
    index($0, config) > 0 && index($0, "scripts/train.py") > 0 && $3 ~ /^python/ {
      print $1
      exit
    }
  '
}

parent_uv_pid_for_train() {
  local pid="$1"
  local ppid
  ppid="$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d " ")"
  if [[ -z "${ppid}" ]]; then
    return 0
  fi
  if ps -p "${ppid}" -o args= 2>/dev/null | grep -q "uv run scripts/train.py"; then
    printf "%s\n" "${ppid}"
  fi
}

run_with_optional_sudo() {
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo -n env "PATH=${PATH}" "$@"
  else
    "$@"
  fi
}

capture_diagnostics() {
  local name="$1"
  local config="$2"
  local run_dir="$3"
  local pid="$4"
  local metrics_age="$5"
  local diag_dir="${run_dir}/stall_diagnostics/$(date -u +"%Y%m%dT%H%M%SZ")"

  mkdir -p "${diag_dir}"
  {
    echo "captured_at_utc: $(ts)"
    echo "run: ${name}"
    echo "config: ${config}"
    echo "run_dir: ${run_dir}"
    echo "pid: ${pid}"
    echo "metrics_age_seconds: ${metrics_age}"
    echo "stale_threshold_seconds: ${STALE_SECONDS}"
  } >"${diag_dir}/summary.txt"

  if [[ -r "/proc/${pid}/status" ]]; then
    cp "/proc/${pid}/status" "${diag_dir}/proc_status.txt" || true
  fi
  ps -eo pid,ppid,stat,pcpu,pmem,etime,args >"${diag_dir}/process_snapshot.txt" 2>&1 || true
  top -bn1 -H -p "${pid}" >"${diag_dir}/top_threads.txt" 2>&1 || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv >"${diag_dir}/nvidia_smi_query.csv" 2>&1 || true
    nvidia-smi -q >"${diag_dir}/nvidia_smi_q.txt" 2>&1 || true
  fi

  if command -v dmesg >/dev/null 2>&1; then
    run_with_optional_sudo dmesg -T --level=err,warn >"${diag_dir}/dmesg_warn_err.txt" 2>&1 || true
    run_with_optional_sudo dmesg -T >"${diag_dir}/dmesg_all.txt" 2>&1 || true
  fi

  if command -v journalctl >/dev/null 2>&1; then
    journalctl -k --since "30 minutes ago" >"${diag_dir}/journalctl_kernel_last30m.txt" 2>&1 || true
  fi

  if command -v timeout >/dev/null 2>&1; then
    run_with_optional_sudo timeout 45s uv run --with py-spy py-spy dump --pid "${pid}" \
      >"${diag_dir}/py_spy_dump.txt" 2>&1 || true
  else
    run_with_optional_sudo uv run --with py-spy py-spy dump --pid "${pid}" \
      >"${diag_dir}/py_spy_dump.txt" 2>&1 || true
  fi

  echo "[$(ts)] captured stall diagnostics in ${diag_dir}"
}

terminate_train_process() {
  local pid="$1"
  local parent_pid
  parent_pid="$(parent_uv_pid_for_train "${pid}" || true)"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(ts)] dry run: would terminate train.py pid ${pid}"
    return 0
  fi

  echo "[$(ts)] sending SIGTERM to train.py pid ${pid}"
  kill -TERM "${pid}" 2>/dev/null || true
  sleep "${TERM_GRACE_SECONDS}"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[$(ts)] train.py pid ${pid} still alive; sending SIGKILL"
    kill -KILL "${pid}" 2>/dev/null || true
  fi

  if [[ -n "${parent_pid}" ]] && kill -0 "${parent_pid}" 2>/dev/null; then
    echo "[$(ts)] sending SIGTERM to parent uv pid ${parent_pid}"
    kill -TERM "${parent_pid}" 2>/dev/null || true
  fi
}

echo "[$(ts)] E003/E004 stall watcher starting"
echo "[$(ts)] poll_seconds=${POLL_SECONDS} stale_seconds=${STALE_SECONDS} dry_run=${DRY_RUN}"
echo "[$(ts)] run ./scripts/experiments/run_full_e003_e004_promotions.sh in the foreground terminal"

while true; do
  now="$(date +%s)"
  active_count=0

  for index in "${!TARGET_CONFIGS[@]}"; do
    name="${TARGET_NAMES[${index}]}"
    config="${TARGET_CONFIGS[${index}]}"
    run_dir="${TARGET_RUN_DIRS[${index}]}"
    metrics_path="${run_dir}/metrics.jsonl"
    pid="$(train_python_pid_for_config "${config}")"

    if [[ -z "${pid}" ]]; then
      continue
    fi

    active_count=$((active_count + 1))
    if [[ ! -f "${metrics_path}" ]]; then
      echo "[$(ts)] [${name}] train.py pid ${pid} active; metrics file not created yet"
      continue
    fi

    metrics_mtime="$(stat -c %Y "${metrics_path}")"
    metrics_age=$((now - metrics_mtime))
    if (( metrics_age > STALE_SECONDS )); then
      echo "[$(ts)] [${name}] metrics stale for ${metrics_age}s; capturing diagnostics before termination"
      capture_diagnostics "${name}" "${config}" "${run_dir}" "${pid}" "${metrics_age}"
      terminate_train_process "${pid}"
      break
    fi

    echo "[$(ts)] [${name}] train.py pid ${pid} active; metrics age ${metrics_age}s"
  done

  if (( active_count == 0 )); then
    echo "[$(ts)] no active target train.py process found"
  fi

  sleep "${POLL_SECONDS}"
done
