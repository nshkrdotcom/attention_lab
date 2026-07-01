#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
Refusing to launch all E002 full runs.

E002 is screen-first:
  1. queue/screen standard_refactor_control
  2. queue/screen static/train/position variants
  3. generate promotion reports
  4. approve only selected full runs
  5. run full only after promotion

Use README.md and docs/experiments/E002_multitrack_qkv_shift_register_plan.md for the screen-first workflow.
EOF

exit 2
