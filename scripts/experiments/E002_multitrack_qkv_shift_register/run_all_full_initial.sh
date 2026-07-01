#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

scripts/experiments/E002_multitrack_qkv_shift_register/run_full_standard_refactor_control.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_static_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_train_rotation_global.sh
scripts/experiments/E002_multitrack_qkv_shift_register/run_full_position_rotation_global.sh

echo
echo "Initial E002 full runs completed."
echo "Now run qkv_track_destructive_test.py for A/B/C, then compare_initial_full_runs.sh."
