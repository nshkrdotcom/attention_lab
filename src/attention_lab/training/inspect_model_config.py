from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.config import load_config


def _parameter_breakdown(model: GPT) -> dict[str, int]:
    total = 0
    trainable = 0
    attention_projection = 0
    global_qkv_bank = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        if ".attn." in name or name.startswith("multi_qkv_bank."):
            attention_projection += count
        if name.startswith("multi_qkv_bank."):
            global_qkv_bank += count
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "attention_projection_parameters": attention_projection,
        "global_qkv_bank_parameters": global_qkv_bank,
        "non_attention_parameters": total - attention_projection,
    }


def _parameter_counts(config_path: str | Path) -> tuple[dict[str, Any], int, int, dict[str, int]]:
    config = load_config(config_path)
    model_config = config_from_dict(config["model"], config["data"])
    model = GPT(model_config)
    return config, model.num_parameters(non_embedding=True), model.num_parameters(non_embedding=False), _parameter_breakdown(model)


def inspect_model_config(config_path: str | Path, baseline_config_path: str | Path | None = None) -> dict[str, Any]:
    config, parameters_excluding_positional, parameters_including_positional, breakdown = _parameter_counts(config_path)
    model_config = config_from_dict(config["model"], config["data"])
    train_config = config["train"]
    micro_tokens = int(train_config["B"]) * int(train_config["T"])
    total_batch_size = int(train_config["total_batch_size"])
    result = {
        "config": str(config_path),
        "run_name": config["run"]["name"],
        "attention_type": model_config.attention_type,
        "cp_rank": model_config.cp_rank,
        "cp_lambda_init": model_config.cp_lambda_init,
        "cp_lambda_trainable": model_config.cp_lambda_trainable,
        "cp_lambda_fixed": model_config.cp_lambda_fixed,
        "qkv_track_count": model_config.qkv_track_count,
        "qkv_global_bank": model_config.qkv_global_bank,
        "qkv_route_formula": model_config.qkv_route_formula,
        "multi_qkv_track_count": model_config.multi_qkv_track_count,
        "multi_qkv_global": model_config.multi_qkv_global,
        "n_layer": model_config.n_layer,
        "n_head": model_config.n_head,
        "n_embd": model_config.n_embd,
        "block_size": model_config.block_size,
        "vocab_size": model_config.vocab_size,
        "parameters_excluding_positional": parameters_excluding_positional,
        "parameters_including_positional": parameters_including_positional,
        "trainable_parameters": breakdown["trainable_parameters"],
        "attention_projection_parameters": breakdown["attention_projection_parameters"],
        "global_qkv_bank_parameters": breakdown["global_qkv_bank_parameters"],
        "non_attention_parameters": breakdown["non_attention_parameters"],
        "estimated_tokens_per_step": total_batch_size,
        "estimated_gradient_accum_steps": total_batch_size // micro_tokens,
    }
    if baseline_config_path is not None:
        _, baseline_excluding_positional, baseline_including_positional, _ = _parameter_counts(baseline_config_path)
        delta_excluding = parameters_excluding_positional - baseline_excluding_positional
        delta_including = parameters_including_positional - baseline_including_positional
        result.update(
            {
                "baseline_config": str(baseline_config_path),
                "baseline_parameters_excluding_positional": baseline_excluding_positional,
                "baseline_parameters_including_positional": baseline_including_positional,
                "parameter_delta_vs_baseline": delta_excluding,
                "parameter_delta_including_positional_vs_baseline": delta_including,
                "percent_delta_vs_baseline": (delta_excluding / baseline_excluding_positional) * 100.0,
                "percent_delta_including_positional_vs_baseline": (
                    delta_including / baseline_including_positional
                )
                * 100.0,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    result = inspect_model_config(args.config, baseline_config_path=args.baseline_config)
    for key, value in result.items():
        print(f"{key}: {value}")
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
