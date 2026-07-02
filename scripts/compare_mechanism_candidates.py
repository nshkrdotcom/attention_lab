#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from attention_lab.mechanisms.reports import write_cross_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-experiment mechanism candidate report.")
    parser.add_argument("--backfill-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    backfill_root = Path(args.backfill_root)
    if not backfill_root.exists():
        raise SystemExit(f"backfill root does not exist: {backfill_root}")
    output = Path(args.output)
    write_cross_experiment_report(backfill_root, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
