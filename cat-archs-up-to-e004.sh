#!/usr/bin/env bash
set -euo pipefail

# Run from repo root.
# Usage:
#   ./scripts/cat_attention_archs_and_reports.sh > archs_reports_E001_E004.txt

cat_file() {
  local path="$1"

  echo
  echo "================================================================================"
  echo "FILE: ${path}"
  echo "================================================================================"
  echo

  if [[ -f "$path" ]]; then
    cat "$path"
  else
    echo "MISSING: $path" >&2
  fi
}

echo "ATTENTION_LAB E001-E004 ARCHITECTURES + REPORTS"

# ==============================================================================
# Shared / baseline architecture
# ==============================================================================

cat_file "src/attention_lab/models/attention/standard.py"

# ==============================================================================
# E001: CP trilinear attention
# ==============================================================================

cat_file "reports/experiments/E001_cp_trilinear_attention/report.md"
cat_file "reports/experiments/E001_cp_trilinear_attention/arch_diags.md"

cat_file "src/attention_lab/models/attention/cp_common.py"
cat_file "src/attention_lab/models/attention/cp_bilinear.py"
cat_file "src/attention_lab/models/attention/cp_trilinear.py"

# Legacy / reserved CP path if present in the repo
cat_file "src/attention_lab/models/attention/trilinear_cp.py"

# ==============================================================================
# E002: Multi-QKV shift/register family
# ==============================================================================

cat_file "reports/experiments/E002_multitrack_qkv_shift_register/report.md"
cat_file "reports/experiments/E002_multitrack_qkv_shift_register/arch_diags.md"

cat_file "src/attention_lab/models/attention/multi_qkv_common.py"
cat_file "src/attention_lab/models/attention/multi_qkv_static.py"
cat_file "src/attention_lab/models/attention/multi_qkv_train_rotation.py"
cat_file "src/attention_lab/models/attention/multi_qkv_position_rotation.py"

# ==============================================================================
# E003: QKV architecture gauntlet
# ==============================================================================

cat_file "reports/experiments/E003_qkv_architecture_gauntlet/report.md"
cat_file "reports/experiments/E003_qkv_architecture_gauntlet/arch_diags.md"

cat_file "src/attention_lab/models/attention/differential_qkv.py"
cat_file "src/attention_lab/models/attention/scope_gated_qkv.py"

# ==============================================================================
# E004: Operator / binding / dynamic value gauntlet
# ==============================================================================

cat_file "reports/experiments/E004_operator_binding_qkv_gauntlet/report.md"
cat_file "reports/experiments/E004_operator_binding_qkv_gauntlet/arch_diags.md"

cat_file "src/attention_lab/models/attention/operator_valued.py"
cat_file "src/attention_lab/models/attention/dynamic_value_qc.py"
cat_file "src/attention_lab/models/attention/q3k3v3_role_routed.py"

