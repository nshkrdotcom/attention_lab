#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec, run_with_interventions
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small post-hoc mechanism probe from a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--prompts-file")
    parser.add_argument("--sites", required=True, help="Comma-separated hook site bases")
    parser.add_argument("--interventions", default="", help="Comma-separated intervention names: zero,scale")
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.exists():
        raise SystemExit(f"config does not exist: {config_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
    prompts = _load_prompts(args.prompt, args.prompts_file)
    if not prompts:
        raise SystemExit("provide --prompt or --prompts-file")

    config = load_config(config_path)
    model_config = config_from_dict(config["model"], config["data"])
    model = GPT(model_config)
    checkpoint = load_checkpoint(checkpoint_path, device=args.device)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    input_ids = _encode_prompts(prompts, block_size=model_config.block_size).to(args.device)
    sites = [site.strip() for site in args.sites.split(",") if site.strip()]
    with torch.no_grad():
        capture = capture_activations(
            model,
            input_ids,
            sites=sites,
            detach=True,
            cpu=True,
            checkpoint_path=checkpoint_path,
            batch_metadata={"prompts": prompts},
            schedule_mode="eval",
        )
        intervention_summary = {}
        for intervention_name in [name.strip() for name in args.interventions.split(",") if name.strip()]:
            kind = InterventionKind(intervention_name)
            specs = [
                InterventionSpec(site=site, layer=0, kind=kind, scale=args.scale if kind == InterventionKind.SCALE else None)
                for site in sites
            ]
            result = run_with_interventions(model, input_ids, specs, capture_sites=sites, schedule_mode="eval")
            intervention_summary[intervention_name] = {
                "applied": result.applied_interventions,
                "missing_or_failed": result.missing_or_failed_interventions,
                "logit_delta_norm": float((result.logits.detach().cpu() - capture.logits.detach().cpu()).float().norm()),
            }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture.cache.save_metadata_summary(output_dir / "activation_summary.json")
    (output_dir / "intervention_summary.json").write_text(
        json.dumps({"schema_version": 1, "interventions": intervention_summary}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_probe_report(output_dir / "probe_report.md", config_path, checkpoint_path, capture, intervention_summary)
    print(f"wrote {output_dir}")


def _load_prompts(prompt: str | None, prompts_file: str | None) -> list[str]:
    prompts = []
    if prompt:
        prompts.append(prompt)
    if prompts_file:
        path = Path(prompts_file)
        if not path.exists():
            raise SystemExit(f"prompts file does not exist: {path}")
        prompts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return prompts


def _encode_prompts(prompts: list[str], *, block_size: int) -> torch.Tensor:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    encoded = [enc.encode(prompt)[:block_size] for prompt in prompts]
    max_len = max(1, max(len(tokens) for tokens in encoded))
    padded = []
    for tokens in encoded:
        if not tokens:
            tokens = [0]
        padded.append(tokens + [0] * (max_len - len(tokens)))
    return torch.tensor(padded, dtype=torch.long)


def _write_probe_report(
    path: Path,
    config_path: Path,
    checkpoint_path: Path,
    capture,
    intervention_summary: dict,
) -> None:
    lines = [
        "# Mechanism Probe Report",
        "",
        f"- config: `{config_path}`",
        f"- checkpoint: `{checkpoint_path}`",
        f"- attention_type: `{capture.cache.attention_type}`",
        f"- captured_sites: {len(capture.cache.records)}",
        f"- missing_sites: {len(capture.missing_sites)}",
        f"- interventions: {', '.join(intervention_summary) if intervention_summary else 'none'}",
        "",
    ]
    if capture.missing_sites:
        lines.append("## Missing Sites")
        for site, missing in capture.missing_sites.items():
            lines.append(f"- `{site}`: {missing.status} ({missing.reason})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
