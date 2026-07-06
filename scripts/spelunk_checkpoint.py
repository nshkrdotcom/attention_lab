#!/usr/bin/env python3
"""Spelunk a checkpoint: capture real per-layer attention weights, visualize
them, and optionally score a synthetic induction probe. This is exploratory
instrumentation, not a confirmatory hypothesis test -- it doesn't assume what
you're looking for in advance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken
import torch

from attention_lab.mechanisms.activations import load_mechanism_model
from attention_lab.mechanisms.attention_reconstruction import reconstruct_standard_attention_weights
from attention_lab.mechanisms.capture import capture_activations
from attention_lab.mechanisms.hook_sites import get_hook_site_specs
from attention_lab.mechanisms.synthetic_prompts import build_induction_probe, induction_accuracy
from attention_lab.mechanisms.visualize import plot_attention_heatmap, plot_track_selection_histogram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", help="Real text prompt to run through the model.")
    parser.add_argument(
        "--induction-probe-pattern-len",
        type=int,
        help="Build a synthetic induction probe with this pattern length instead of --prompt.",
    )
    parser.add_argument("--induction-probe-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.prompt and not args.induction_probe_pattern_len:
        raise SystemExit("provide --prompt or --induction-probe-pattern-len")

    loaded = load_mechanism_model(args.config, args.checkpoint, device=args.device)
    loaded.model.eval()

    induction_probe = None
    if args.induction_probe_pattern_len:
        induction_probe = build_induction_probe(
            vocab_size=loaded.vocab_size,
            pattern_len=args.induction_probe_pattern_len,
            generator=torch.Generator().manual_seed(args.induction_probe_seed),
        )
        input_ids = induction_probe.input_ids.to(args.device)
    else:
        enc = tiktoken.get_encoding("gpt2")
        input_ids = torch.tensor([enc.encode(args.prompt)]).to(args.device)

    attention_type = loaded.attention_type
    site_names = {spec.name for spec in get_hook_site_specs(attention_type)}
    has_attn_weights_site = "attn_weights[layer]" in site_names
    has_selected_track_site = "selected_track[layer]" in site_names

    requested_sites = ["attn_q", "attn_k"]
    if has_attn_weights_site:
        requested_sites.append("attn_weights")
    if has_selected_track_site:
        requested_sites.append("selected_track")

    with torch.no_grad():
        result = capture_activations(
            loaded.model,
            input_ids,
            sites=requested_sites,
            detach=True,
            cpu=True,
            checkpoint_path=Path(args.checkpoint),
            schedule_mode="eval",
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_layer = loaded.model.config.n_layer
    generated_files = []
    for layer in range(n_layer):
        weights_key = f"attn_weights[{layer}]"
        if weights_key in result.cache.records:
            weights = result.cache.records[weights_key].tensor
        elif f"attn_q[{layer}]" in result.cache.records:
            q = result.cache.records[f"attn_q[{layer}]"].tensor
            k = result.cache.records[f"attn_k[{layer}]"].tensor
            weights = reconstruct_standard_attention_weights(q, k)
        else:
            continue

        heatmap_path = output_dir / f"attn_heatmap_layer{layer}.png"
        plot_attention_heatmap(weights, heatmap_path, title=f"{attention_type} layer {layer}")
        generated_files.append(str(heatmap_path))

        track_key = f"selected_track[{layer}]"
        if track_key in result.cache.records:
            track_path = output_dir / f"track_selection_layer{layer}.png"
            plot_track_selection_histogram(
                result.cache.records[track_key].tensor,
                track_count=loaded.model.config.qkv_track_count,
                out_path=track_path,
                title=f"{attention_type} layer {layer}",
            )
            generated_files.append(str(track_path))

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "attention_type": attention_type,
        "n_layer": n_layer,
        "generated_files": generated_files,
    }
    if induction_probe is not None:
        summary["induction_probe"] = {
            "pattern_len": args.induction_probe_pattern_len,
            "seed": args.induction_probe_seed,
            "accuracy": induction_accuracy(result.logits, induction_probe),
            "n_scorable_positions": len(induction_probe.induction_positions),
        }
    (output_dir / "spelunk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
