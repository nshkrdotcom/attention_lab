#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from attention_lab.mechanisms.backfill import (
    build_registered_experiment_inventory,
    resolve_experiment_ids,
    write_backfill_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill mechanism investigation inventories.")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment ids or aliases, e.g. E001,E002")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--output-root", default="reports/mechanisms/backfill", help="Derived backfill output root")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    experiment_ids = resolve_experiment_ids(args.experiments, repo_root)
    if not experiment_ids:
        raise SystemExit("no experiments requested")
    output_root = repo_root / args.output_root
    for experiment_id in experiment_ids:
        inventory = build_registered_experiment_inventory(experiment_id, repo_root)
        write_backfill_outputs(inventory, output_root / experiment_id)
        print(f"wrote {output_root / experiment_id / 'inventory.json'}")


if __name__ == "__main__":
    main()
