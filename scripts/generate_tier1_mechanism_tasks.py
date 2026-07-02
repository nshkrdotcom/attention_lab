#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from attention_lab.mechanisms.task_generation import (
    TIER1_CANDIDATES,
    generate_tier1_negation_suite,
    validate_tier1_suite_file,
    write_tier1_negation_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or validate deterministic Tier-1 mechanism-probe task suites."
    )
    parser.add_argument("--output", required=True, help="Task suite YAML to write or validate.")
    parser.add_argument("--candidate", required=True, choices=TIER1_CANDIDATES)
    parser.add_argument("--pairs-per-family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.validate_only:
        validate_tier1_suite_file(output, min_n=args.pairs_per_family)
        print(f"validated {output}")
        return

    suite = generate_tier1_negation_suite(
        candidate=args.candidate,
        pairs_per_family=args.pairs_per_family,
        seed=args.seed,
    )
    write_tier1_negation_suite(output, suite)
    validate_tier1_suite_file(output, min_n=args.pairs_per_family)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
