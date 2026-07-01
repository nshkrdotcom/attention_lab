from contextlib import nullcontext

import torch


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def autocast_context(device_type: str, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype)


def device_type_from_device(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"

