import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.data_manifest import sha256_file
from attention_lab.training.runtime import autocast_context, device_type_from_device, dtype_from_name


def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask
    avg_loss = masked_shift_losses.sum(dim=1) / shift_mask.sum(dim=1)
    return int(avg_loss.argmin().item())


def default_output(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "evals" / "hellaswag.json"
    return Path("reports") / "hellaswag.json"


@torch.no_grad()
def run_hellaswag(args: argparse.Namespace) -> dict:
    from attention_lab.evals.hellaswag_data import HELLASWAG_URLS, download, iterate_examples_from_path, render_example

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    device_type = device_type_from_device(device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config = checkpoint["config"]
    model_config = config_from_dict(config["model"], config["data"])
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    dtype = dtype_from_name(args.dtype or config["train"].get("dtype", "bfloat16"))
    data_path = download(args.split)

    num_correct_norm = 0
    num_total = 0
    for example in iterate_examples_from_path(data_path):
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with autocast_context(device_type, dtype):
            logits, _ = model(tokens, schedule_mode="eval")
        pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
        if args.max_examples and num_total >= args.max_examples:
            break

    result = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_total": num_total,
        "num_correct_norm": num_correct_norm,
        "accuracy_norm": num_correct_norm / num_total if num_total else 0.0,
        "data_path": str(data_path),
        "data_url": HELLASWAG_URLS[args.split],
        "data_sha256": sha256_file(data_path),
    }
    out_path = Path(args.out) if args.out else default_output(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(json.dumps(run_hellaswag(args), indent=2))


if __name__ == "__main__":
    main()
