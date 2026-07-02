#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_lab.mechanisms.claim_gates import ClaimGateResult
from attention_lab.mechanisms.summary import render_summary_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a Tier-1 mechanism probe suite summary.md.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", help="Defaults to <input-dir>/summary.md")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    metrics_path = input_dir / "metrics.json"
    gates_path = input_dir / "claim_gates.json"
    if not metrics_path.exists():
        raise SystemExit(f"metrics.json does not exist: {metrics_path}")
    if not gates_path.exists():
        raise SystemExit(f"claim_gates.json does not exist: {gates_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    gate = ClaimGateResult(
        status=gates["status"],
        status_vocabulary=gates["status_vocabulary"],
        reasons=list(gates.get("reasons", [])),
        caps=list(gates.get("caps", [])),
    )
    output = Path(args.output) if args.output else input_dir / "summary.md"
    output.write_text(render_summary_markdown(metrics, gate), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
