#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run scripts/train.py --config configs/baseline_15m_fineweb100m_sanity.yaml

