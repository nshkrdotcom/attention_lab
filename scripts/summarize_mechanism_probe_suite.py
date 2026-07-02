#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from attention_lab.mechanisms.summary import load_suite_artifacts, render_summary, validate_suite_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate summary.md for a mechanism probe suite output directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate", action="store_true", help="Validate required suite artifact structure.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    metrics, claim_gates = load_suite_artifacts(output_dir)
    (output_dir / "summary.md").write_text(render_summary(metrics, claim_gates), encoding="utf-8")
    if args.validate:
        errors = validate_suite_artifacts(output_dir)
        if errors:
            raise SystemExit("invalid mechanism probe suite artifacts: " + "; ".join(errors))
        print(f"validated {output_dir}")
    print(f"wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
