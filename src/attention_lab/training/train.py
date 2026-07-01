import argparse
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import tiktoken
from torch.nn.parallel import DistributedDataParallel as DDP

from attention_lab.evals.generation_eval import generate_text
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.attention_diagnostics import append_attention_diagnostics, collect_attention_diagnostics
from attention_lab.training.checkpointing import load_checkpoint, restore_rng_state, save_checkpoint
from attention_lab.training.config import load_config, save_config
from attention_lab.training.data_manifest import copy_manifest_to_run
from attention_lab.training.data_loader import TokenShardLoader
from attention_lab.training.environment import environment_text, git_commit_text
from attention_lab.training.gpu_metrics import collect_gpu_metrics
from attention_lab.training.metrics import MetricsLogger
from attention_lab.training.optim import build_optimizer
from attention_lab.training.resume import (
    assert_model_state_compatible,
    validate_resume_compatibility,
    validate_resume_data_manifest,
)
from attention_lab.training.runtime import autocast_context, device_type_from_device, dtype_from_name


def setup_distributed(requested_device: str) -> tuple[bool, int, int, int, str, bool]:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA.")
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        return True, rank, local_rank, world_size, device, rank == 0

    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = requested_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is False.")
    return False, 0, 0, 1, device, True


def get_lr(step: int, train_config: dict[str, Any]) -> float:
    max_lr = float(train_config["learning_rate"])
    min_lr = float(train_config["min_lr"])
    warmup_steps = int(train_config.get("warmup_steps", 0))
    max_steps = int(train_config["max_steps"])
    if warmup_steps > 0 and step <= warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: TokenShardLoader,
    val_steps: int,
    device: str,
    device_type: str,
    dtype: torch.dtype,
    ddp: bool,
) -> float:
    model.eval()
    loader.reset()
    val_loss_accum = torch.zeros((), device=device)
    for _ in range(val_steps):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        with autocast_context(device_type, dtype):
            _, loss = model(x, y, schedule_mode="eval")
        val_loss_accum += loss.detach() / val_steps
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    return float(val_loss_accum.item())


def prepare_run_dir(out_dir: Path, config: dict[str, Any], config_path: Path, overwrite: bool, resume: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite and not resume:
        raise FileExistsError(
            f"Run directory already exists and is not empty: {out_dir}. "
            "Use --overwrite for a fresh run or --resume to continue."
        )
    if overwrite and out_dir.exists() and not resume:
        shutil.rmtree(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "evals").mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")
    (out_dir / "config_source.txt").write_text(str(config_path) + "\n", encoding="utf-8")
    (out_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    (out_dir / "git_commit.txt").write_text(git_commit_text(Path.cwd()), encoding="utf-8")
    manifest_hash = copy_manifest_to_run(config["data"]["data_root"], out_dir)
    if manifest_hash is not None:
        with (out_dir / "environment.txt").open("a", encoding="utf-8") as f:
            f.write(f"\ndata_manifest_sha256: {manifest_hash}\n")


def write_samples(
    raw_model: GPT,
    out_dir: Path,
    step: int,
    config: dict[str, Any],
    device: str,
    device_type: str,
    dtype: torch.dtype,
) -> None:
    sample_config = config.get("sample", {})
    prompt = sample_config.get("prompt", "Hello, I'm a language model,")
    enc = tiktoken.get_encoding(config["data"].get("tokenizer", "gpt2"))
    samples = generate_text(
        raw_model,
        enc,
        prompt=prompt,
        num_samples=int(sample_config.get("num_samples", 4)),
        max_new_tokens=int(sample_config.get("max_new_tokens", 64)),
        top_k=int(sample_config.get("top_k", 50)),
        temperature=float(sample_config.get("temperature", 1.0)),
        device=device,
        device_type=device_type,
        dtype=dtype,
        seed=int(sample_config.get("seed", 42)),
    )
    sample_path = out_dir / "samples" / f"sample_step_{step:06d}.txt"
    sample_path.write_text("\n\n---\n\n".join(samples) + "\n", encoding="utf-8")
    if step == int(config["train"]["max_steps"]):
        (out_dir / "samples" / "sample_step_last.txt").write_text(sample_path.read_text(encoding="utf-8"), encoding="utf-8")


def train(config_path: str | Path, overwrite: bool = False, resume_path: str | None = None) -> None:
    if overwrite and resume_path is not None:
        raise ValueError("--overwrite and --resume cannot be used together")
    config_path = Path(config_path)
    config = load_config(config_path)
    train_config = config["train"]
    run_config = config["run"]
    data_config = config["data"]
    out_dir = Path(run_config["out_dir"])

    ddp, rank, _, world_size, device, master_process = setup_distributed(train_config.get("device", "auto"))
    device_type = device_type_from_device(device)
    dtype = dtype_from_name(train_config.get("dtype", "bfloat16"))
    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = load_checkpoint(resume_path, device=device)
        validate_resume_compatibility(
            config,
            resume_checkpoint["config"],
            checkpoint_step=int(resume_checkpoint["step"]),
        )
        validate_resume_data_manifest(out_dir, data_config["data_root"], checkpoint=resume_checkpoint)

    if master_process:
        prepare_run_dir(out_dir, config, config_path, overwrite=overwrite, resume=resume_path is not None)
        print(f"using device: {device}")
    if ddp:
        dist.barrier()

    seed = int(run_config.get("seed", 1337)) + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("high")

    B = int(train_config["B"])
    T = int(train_config["T"])
    total_batch_size = int(train_config["total_batch_size"])
    if total_batch_size % (B * T * world_size) != 0:
        raise ValueError("total_batch_size must be divisible by B * T * world_size")
    grad_accum_steps = total_batch_size // (B * T * world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = TokenShardLoader(data_config["data_root"], B, T, rank, world_size, "train", master_process)
    val_loader = TokenShardLoader(data_config["data_root"], B, T, rank, world_size, "val", master_process)

    model_config = config_from_dict(config["model"], data_config)
    raw_model = GPT(model_config)
    raw_model.to(device)
    if master_process:
        print(f"model parameters: {raw_model.num_parameters():,}")

    model: torch.nn.Module = raw_model
    if bool(train_config.get("compile", False)):
        model = torch.compile(raw_model)
    if ddp:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    optimizer = build_optimizer(
        raw_model,
        weight_decay=float(train_config["weight_decay"]),
        learning_rate=float(train_config["learning_rate"]),
        device_type=device_type,
        master_process=master_process,
    )

    start_step = 0
    last_train_loss = None
    if resume_path:
        checkpoint = resume_checkpoint
        assert checkpoint is not None
        assert_model_state_compatible(raw_model, checkpoint["model"])
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("train_loader_state") is not None:
            train_loader.load_state_dict(checkpoint["train_loader_state"])
        else:
            raise ValueError("Checkpoint does not contain train_loader_state; exact resume is not possible.")
        restore_rng_state(checkpoint)
        start_step = int(checkpoint["step"])
        last_train_loss = checkpoint.get("train_loss")
        if master_process:
            print(f"resumed from {resume_path} at step {start_step}")

    logger = MetricsLogger(out_dir, append=resume_path is not None) if master_process else None
    if master_process and resume_path is not None:
        resume_text = str(Path(resume_path)) + "\n"
        (out_dir / "resume_from.txt").write_text(resume_text, encoding="utf-8")
        logger.log(
            {
                "event": "resume",
                "step": start_step,
                "checkpoint": str(Path(resume_path)),
                "resume_from": str(Path(resume_path)),
            }
        )
    max_steps = int(train_config["max_steps"])
    val_every = int(train_config["val_every"])
    val_steps = int(train_config["val_steps"])
    save_every = int(train_config["save_every"])
    log_every = int(train_config["log_every"])
    sample_every = int(config.get("sample", {}).get("sample_every", save_every))
    attention_diagnostics_every = int(config.get("diagnostics", {}).get("attention_diagnostics_every", 0))

    try:
        if start_step == 0 and bool(train_config.get("eval_at_start", True)):
            val_loss = evaluate_loss(model, val_loader, val_steps, device, device_type, dtype, ddp)
            if master_process:
                logger.log(
                    {
                        "event": "val",
                        "step": 0,
                        "val_loss": val_loss,
                        "val_perplexity": math.exp(val_loss),
                        **collect_gpu_metrics(device_type),
                    }
                )
                print(f"step     0 | val loss: {val_loss:.4f} | val ppl: {math.exp(val_loss):.2f}")

        for step in range(start_step + 1, max_steps + 1):
            t0 = time.time()
            lr = get_lr(step, train_config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_accum = torch.zeros((), device=device)
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                if ddp:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                with autocast_context(device_type, dtype):
                    _, loss = model(x, y, step=step, schedule_mode="train")
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["grad_clip"]))
            attention_diagnostic_rows = []
            if (
                master_process
                and attention_diagnostics_every > 0
                and (step % attention_diagnostics_every == 0 or step == max_steps)
            ):
                attention_diagnostic_rows = collect_attention_diagnostics(raw_model, step)
            optimizer.step()
            if master_process and attention_diagnostic_rows:
                append_attention_diagnostics(
                    out_dir,
                    attention_diagnostic_rows,
                    run_name=str(run_config["name"]),
                )
            if device_type == "cuda":
                torch.cuda.synchronize()

            dt = time.time() - t0
            tokens_processed = B * T * grad_accum_steps * world_size
            tokens_per_sec = tokens_processed / dt
            last_train_loss = float(loss_accum.item())
            memory_metrics = collect_gpu_metrics(device_type)

            if master_process and (step % log_every == 0 or step == 1):
                logger.log(
                    {
                        "event": "train",
                        "step": step,
                        "train_loss": last_train_loss,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "dt_ms": dt * 1000,
                        "tokens_per_sec": tokens_per_sec,
                        **memory_metrics,
                    }
                )
                print(
                    f"step {step:5d} | loss: {last_train_loss:.6f} | lr {lr:.4e} | "
                    f"norm: {float(grad_norm):.4f} | dt: {dt * 1000:.2f}ms | "
                    f"tok/sec: {tokens_per_sec:.2f}"
                )

            val_loss = None
            if step % val_every == 0 or step == max_steps:
                val_loss = evaluate_loss(model, val_loader, val_steps, device, device_type, dtype, ddp)
                if master_process:
                    logger.log(
                        {
                            "event": "val",
                            "step": step,
                            "val_loss": val_loss,
                            "val_perplexity": math.exp(val_loss),
                            **memory_metrics,
                        }
                    )
                    print(f"step {step:5d} | val loss: {val_loss:.4f} | val ppl: {math.exp(val_loss):.2f}")

            if master_process and sample_every > 0 and (step % sample_every == 0 or step == max_steps):
                write_samples(raw_model, out_dir, step, config, device, device_type, dtype)

            if master_process and (step % save_every == 0 or step == max_steps):
                checkpoint_path = save_checkpoint(
                    out_dir,
                    raw_model,
                    optimizer,
                    config,
                    step,
                    train_loss=last_train_loss,
                    val_loss=val_loss,
                    train_loader_state=train_loader.state_dict(),
                )
                logger.log({"event": "checkpoint", "step": step, "checkpoint": str(checkpoint_path)})
    finally:
        if logger is not None:
            logger.close()
        if ddp:
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run directory.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    args = parser.parse_args()
    train(args.config, overwrite=args.overwrite, resume_path=args.resume)


if __name__ == "__main__":
    main()
