from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass
class ActivationRecord:
    site: str
    layer: int | None
    tensor: torch.Tensor
    metadata: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        tensor = self.tensor
        detached = tensor.detach().float()
        summary: dict[str, Any] = {
            "site": self.site,
            "layer": self.layer,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "device": str(tensor.device),
            "requires_grad": bool(tensor.requires_grad),
            "metadata": _json_safe(self.metadata),
        }
        if detached.numel() > 0:
            summary.update(
                {
                    "mean": float(detached.mean().item()),
                    "std": float(detached.std(unbiased=False).item()),
                    "min": float(detached.min().item()),
                    "max": float(detached.max().item()),
                    "norm": float(detached.norm().item()),
                }
            )
        return summary


@dataclass
class ActivationCache:
    records: Mapping[str, ActivationRecord]
    model_name: str
    attention_type: str
    checkpoint_path: Path | None
    config_hash: str | None
    batch_metadata: dict[str, Any]

    def metadata_summary(self, *, tensor_saved: bool = False) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_name": self.model_name,
            "attention_type": self.attention_type,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path is not None else None,
            "config_hash": self.config_hash,
            "batch_metadata": _json_safe(self.batch_metadata),
            "records": {
                key: {**record.summary(), "tensor_saved": tensor_saved}
                for key, record in sorted(self.records.items(), key=lambda item: item[0])
            },
        }

    def save_metadata_summary(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata_summary(tensor_saved=False), indent=2, sort_keys=True), encoding="utf-8")

    def save(self, path: str | Path, *, include_tensors: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"metadata": self.metadata_summary(tensor_saved=include_tensors)}
        if include_tensors:
            payload["records"] = {
                key: {
                    "site": record.site,
                    "layer": record.layer,
                    "tensor": record.tensor.detach().cpu(),
                    "metadata": record.metadata,
                }
                for key, record in self.records.items()
            }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "ActivationCache":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        records = {
            key: ActivationRecord(
                site=value["site"],
                layer=value["layer"],
                tensor=value["tensor"],
                metadata=value.get("metadata", {}),
            )
            for key, value in payload.get("records", {}).items()
        }
        checkpoint_path = metadata.get("checkpoint_path")
        return cls(
            records=records,
            model_name=metadata["model_name"],
            attention_type=metadata["attention_type"],
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
            config_hash=metadata.get("config_hash"),
            batch_metadata=metadata.get("batch_metadata", {}),
        )


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().float()
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
    }
    if detached.numel() > 0:
        result.update(
            {
                "mean": float(detached.mean().item()),
                "std": float(detached.std(unbiased=False).item()),
                "min": float(detached.min().item()),
                "max": float(detached.max().item()),
                "norm": float(detached.norm().item()),
            }
        )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return tuple(_json_safe(item) for item in value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return tensor_summary(value)
    return value
