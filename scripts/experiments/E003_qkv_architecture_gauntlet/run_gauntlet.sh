#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

ALLOW_FULL=0
if [[ "${1:-}" == "--allow-full" ]]; then
  ALLOW_FULL=1
elif [[ "${1:-}" != "" ]]; then
  echo "usage: $0 [--allow-full]" >&2
  exit 2
fi

POLICY="configs/experiments/E003_qkv_architecture_gauntlet/gauntlet_policy.yaml"
EXPERIMENT="E003_qkv_architecture_gauntlet"

uv run scripts/verify_cuda.py
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
uv run scripts/validate_experiment.py --id "${EXPERIMENT}"

uv run attn-queue gauntlet-plan \
  --experiment "${EXPERIMENT}" \
  --policy "${POLICY}"

if [[ "${ALLOW_FULL}" == "1" ]]; then
  uv run attn-queue gauntlet-run \
    --experiment "${EXPERIMENT}" \
    --policy "${POLICY}" \
    --until-blocked \
    --allow-full
else
  uv run attn-queue gauntlet-run \
    --experiment "${EXPERIMENT}" \
    --policy "${POLICY}" \
    --until-blocked
fi

uv run attn-queue gauntlet-report --experiment "${EXPERIMENT}"
