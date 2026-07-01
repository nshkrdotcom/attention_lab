import argparse
from pathlib import Path

import torch
import tiktoken
from torch.nn import functional as F

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.runtime import autocast_context, device_type_from_device, dtype_from_name


@torch.no_grad()
def generate_text(
    model: GPT,
    enc,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    device: str,
    device_type: str,
    dtype: torch.dtype,
    seed: int,
) -> list[str]:
    model.eval()
    start_ids = enc.encode(prompt)
    if start_ids and max(start_ids) >= model.config.vocab_size:
        raise ValueError(
            f"Prompt contains token id {max(start_ids)} but model vocab_size is {model.config.vocab_size}. "
            "Use a checkpoint/config with the matching tokenizer vocabulary."
        )
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(num_samples, 1)
    generator_device = device if device.startswith("cuda") else "cpu"
    sample_rng = torch.Generator(device=generator_device)
    sample_rng.manual_seed(seed)

    for _ in range(max_new_tokens):
        # No KV cache is used here. Each generation step recomputes the cropped
        # full context, so GPT.forward assigns window-relative position ids
        # 0..T-1; E002 position-rotation routing follows those same ids.
        idx_cond = x[:, -model.config.block_size :]
        with autocast_context(device_type, dtype):
            logits, _ = model(idx_cond, schedule_mode="generate")
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=sample_rng)
        x = torch.cat((x, next_id), dim=1)
    return [enc.decode(row.tolist()) for row in x]


def default_sample_output(checkpoint_path: Path, step: int) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        name = "sample_step_last.txt" if checkpoint_path.name == "ckpt_last.pt" else f"sample_step_{step:06d}.txt"
        return checkpoint_path.parent.parent / "samples" / name
    return Path("reports") / "sample.txt"


def run_generate(args: argparse.Namespace) -> list[str]:
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

    dtype = dtype_from_name(args.dtype or config["train"].get("dtype", "bfloat16"))
    enc = tiktoken.get_encoding(config["data"].get("tokenizer", "gpt2"))
    samples = generate_text(
        model,
        enc,
        prompt=args.prompt,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        device=device,
        device_type=device_type,
        dtype=dtype,
        seed=args.seed,
    )

    step = int(checkpoint.get("step", 0))
    out_path = Path(args.out) if args.out else default_sample_output(checkpoint_path, step)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n---\n\n".join(samples) + "\n", encoding="utf-8")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="The history of mathematics")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    samples = run_generate(args)
    for i, sample in enumerate(samples):
        print(f"sample {i}:\n{sample}\n")


if __name__ == "__main__":
    main()
