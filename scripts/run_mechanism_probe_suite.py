#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from attention_lab.mechanisms.suite import run_probe_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tier-1 mechanism probe suite from real checkpoints.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", help="Candidate config override; defaults to the Tier-1 preset config.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hypothesis-doc")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--sites", help="Comma-separated candidate site bases. Defaults to preset Tier-1 sites.")
    parser.add_argument("--site-spec-file", help="Exploratory-only metadata for non-preset sites.")
    parser.add_argument(
        "--feature-pooling",
        default="auto",
        choices=("auto", "mean_sequence", "final_token", "answer_position", "patch_positions_mean"),
    )
    parser.add_argument("--control-mode", default="matched", choices=("matched", "none"))
    parser.add_argument("--control-checkpoint")
    parser.add_argument("--control-config")
    parser.add_argument("--force-noncanonical-control", action="store_true")
    parser.add_argument(
        "--allow-diagnostic-with-missing-control",
        action="store_true",
        help="Allow a confirmatory-shaped run to complete as insufficient-evidence diagnostics when matched controls are missing.",
    )
    parser.add_argument("--min-n", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    sites = [site.strip() for site in args.sites.split(",") if site.strip()] if args.sites else None
    try:
        run_probe_suite(
            experiment_id=args.experiment_id,
            candidate=args.candidate,
            config=args.config,
            checkpoint=Path(args.checkpoint),
            task_file=Path(args.task_file),
            output_dir=Path(args.output_dir),
            hypothesis_doc=Path(args.hypothesis_doc) if args.hypothesis_doc else None,
            exploratory=args.exploratory,
            probe_only=args.probe_only,
            sites=sites,
            control_mode=args.control_mode,
            control_checkpoint=Path(args.control_checkpoint) if args.control_checkpoint else None,
            control_config=Path(args.control_config) if args.control_config else None,
            min_n=args.min_n,
            bootstrap_samples=args.bootstrap_samples,
            fdr_alpha=args.fdr_alpha,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            force_noncanonical_control=args.force_noncanonical_control,
            feature_pooling=args.feature_pooling,
            site_spec_file=Path(args.site_spec_file) if args.site_spec_file else None,
            allow_diagnostic_with_missing_control=args.allow_diagnostic_with_missing_control,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
