#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
Refusing to run all E001 full runs.

E001 is screen-first:
  1. queue/screen standard, cp_bilinear, and cp_trilinear candidates
  2. generate promotion reports
  3. approve only selected full runs
  4. treat lambda0 as a manual promoted control only after active CP evidence warrants it

Use README.md and docs/experiments/E001_cp_trilinear_attention_plan.md for the screen-first workflow.
EOF

exit 2
