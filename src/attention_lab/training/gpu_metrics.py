from __future__ import annotations

import os
import subprocess
from typing import Any

import torch


def _mb(value: int | float) -> float:
    return float(value) / 1024**2


def nvidia_smi_process_memory_mb(pid: int | None = None) -> float | None:
    pid = os.getpid() if pid is None else pid
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    total = 0.0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            memory_mb = float(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            total += memory_mb
    return total if total > 0 else None


def collect_gpu_metrics(device_type: str) -> dict[str, Any]:
    if device_type != "cuda" or not torch.cuda.is_available():
        return {
            "current_vram_allocated_mb": None,
            "peak_vram_allocated_mb": None,
            "current_vram_reserved_mb": None,
            "peak_vram_reserved_mb": None,
            "peak_vram_mb": None,
            "nvidia_smi_memory_mb": None,
        }
    peak_allocated = _mb(torch.cuda.max_memory_allocated())
    return {
        "current_vram_allocated_mb": _mb(torch.cuda.memory_allocated()),
        "peak_vram_allocated_mb": peak_allocated,
        "current_vram_reserved_mb": _mb(torch.cuda.memory_reserved()),
        "peak_vram_reserved_mb": _mb(torch.cuda.max_memory_reserved()),
        "peak_vram_mb": peak_allocated,
        "nvidia_smi_memory_mb": nvidia_smi_process_memory_mb(),
    }
