#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from attention_lab.mechanisms.summary import load_suite_artifacts, render_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate summary.md for a mechanism probe suite output directory.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    metrics, claim_gates = load_suite_artifacts(output_dir)
    (output_dir / "summary.md").write_text(render_summary(metrics, claim_gates), encoding="utf-8")
    print(f"wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
