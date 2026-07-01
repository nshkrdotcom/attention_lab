#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run scripts/prepare_fineweb_edu.py \
  --out_dir data/fineweb_edu_100m \
  --train_tokens 100000000 \
  --val_tokens 4000000
uv run scripts/write_data_manifest.py \
  --data_root data/fineweb_edu_100m \
  --out data/fineweb_edu_100m/manifest.json
uv run scripts/verify_data.py \
  --data_root data/fineweb_edu_100m \
  --manifest data/fineweb_edu_100m/manifest.json \
  --verify_hashes
