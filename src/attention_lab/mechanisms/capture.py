from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from attention_lab.mechanisms.cache import ActivationCache, ActivationRecord, tensor_summary
from attention_lab.mechanisms.hook_sites import format_site_name, get_hook_site_status, site_base
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec


@dataclass
class MissingHookSite:
    site: str
    status: str
    reason: str


@dataclass
class CaptureResult:
    logits: torch.Tensor
    loss: torch.Tensor | None
    cache: ActivationCache
    missing_sites: dict[str, MissingHookSite]


class ActivationRecorder:
    def __init__(
        self,
        *,
        model,
        sites: list[str] | None = None,
        detach: bool = False,
        cpu: bool = False,
        dtype: torch.dtype | None = None,
        interventions: list[InterventionSpec] | None = None,
        checkpoint_path: Path | None = None,
        batch_metadata: dict[str, Any] | None = None,
    ):
        self.model = model
        self.attention_type = getattr(getattr(model, "config", None), "attention_type", "standard")
        self.model_name = model.__class__.__name__
        self.config_hash = hash_model_config(getattr(model, "config", None))
        self.detach = detach
        self.cpu = cpu
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.batch_metadata = batch_metadata or {}
        self.capture_all = sites is None
        self.requested_sites = list(sites or [])
        self.requested_bases = {site_base(site) for site in self.requested_sites}
        self.records: dict[str, ActivationRecord] = {}
        self.missing_sites: dict[str, MissingHookSite] = {}
        self.interventions = interventions or []
        self.applied_interventions: list[dict[str, Any]] = []
        self.failed_interventions: list[dict[str, Any]] = []
        self.before_after_summaries: dict[str, dict[str, Any]] = {}
        self._seen_interventions: set[int] = set()

    def wants(self, site: str, key: str) -> bool:
        if self.capture_all:
            return True
        return site in self.requested_bases or key in self.requested_sites or any(
            self._spec_targets_key(spec, site, key) for spec in self.interventions
        )

    def record(
        self,
        site: str,
        tensor: torch.Tensor,
        *,
        layer: int | None = None,
        track: int | None = None,
        rank: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        key = format_site_name(site, layer=layer, track=track, rank=rank)
        if not self.wants(site, key):
            return tensor

        modified = tensor
        for index, spec in enumerate(self.interventions):
            if not self._spec_targets_key(spec, site, key):
                continue
            before = modified
            try:
                modified = self._apply_intervention(spec, modified, key)
            except Exception as exc:
                self.failed_interventions.append(
                    {"site": key, "kind": spec.kind.value, "reason": str(exc)}
                )
                raise
            self._seen_interventions.add(index)
            self.applied_interventions.append({"site": key, "kind": spec.kind.value})
            self.before_after_summaries[key] = {
                "before": tensor_summary(before),
                "after": tensor_summary(modified),
            }

        record_tensor = self._prepare_tensor(modified)
        self.records[key] = ActivationRecord(
            site=key,
            layer=layer,
            tensor=record_tensor,
            metadata=metadata or {},
        )
        return modified

    def finalize(self) -> None:
        if self.capture_all:
            return
        seen_bases = {site_base(key) for key in self.records}
        seen_keys = set(self.records)
        for requested in self.requested_sites:
            requested_base = site_base(requested)
            if requested in seen_keys or requested_base in seen_bases:
                continue
            status = get_hook_site_status(self.attention_type, requested)
            if status.declared and not status.runtime_supported:
                self.missing_sites[requested] = MissingHookSite(
                    site=requested,
                    status="unsupported",
                    reason=status.reason or "unsupported",
                )
            else:
                self.missing_sites[requested] = MissingHookSite(
                    site=requested,
                    status="missing",
                    reason=status.reason or "site was not emitted by this forward pass",
                )

    def to_cache(self) -> ActivationCache:
        return ActivationCache(
            records=dict(self.records),
            model_name=self.model_name,
            attention_type=self.attention_type,
            checkpoint_path=self.checkpoint_path,
            config_hash=self.config_hash,
            batch_metadata=dict(self.batch_metadata),
        )

    def missing_intervention_summaries(self) -> list[dict[str, Any]]:
        missing = []
        for index, spec in enumerate(self.interventions):
            if index not in self._seen_interventions:
                missing.append(
                    {
                        "site": spec.site if spec.layer is None else format_site_name(spec.site, layer=spec.layer),
                        "kind": spec.kind.value,
                        "reason": "intervention site was not emitted by this forward pass",
                    }
                )
        return missing

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        value = tensor
        if self.detach:
            value = value.detach()
        if self.dtype is not None:
            value = value.to(dtype=self.dtype)
        if self.cpu:
            value = value.cpu()
        return value

    def _spec_targets_key(self, spec: InterventionSpec, site: str, key: str) -> bool:
        if spec.layer is not None:
            if key != format_site_name(spec.site, layer=spec.layer) and not key.startswith(f"{spec.site}[{spec.layer},"):
                return False
        return spec.site == site or spec.site == key

    def _apply_intervention(self, spec: InterventionSpec, tensor: torch.Tensor, key: str) -> torch.Tensor:
        if spec.kind == InterventionKind.ZERO:
            return torch.zeros_like(tensor)
        if spec.kind == InterventionKind.SCALE:
            if spec.scale is None:
                raise ValueError("scale intervention requires scale")
            return tensor * float(spec.scale)
        if spec.kind == InterventionKind.MEAN_ABLATE:
            if tensor.ndim >= 2:
                dims = tuple(range(0, min(2, tensor.ndim)))
                return tensor.mean(dim=dims, keepdim=True).expand_as(tensor)
            return tensor.mean().expand_as(tensor)
        if spec.kind == InterventionKind.REPLACE:
            replacement = self._replacement_tensor(spec, key)
            if replacement.shape != tensor.shape:
                raise ValueError(f"replacement shape {tuple(replacement.shape)} does not match {tuple(tensor.shape)}")
            return replacement.to(device=tensor.device, dtype=tensor.dtype)
        if spec.kind == InterventionKind.PATCH_FROM_CACHE:
            replacement = self._replacement_tensor(spec, key)
            return self._patch_tensor(tensor, replacement.to(device=tensor.device, dtype=tensor.dtype), spec)
        raise ValueError(f"unsupported intervention kind: {spec.kind}")

    def _replacement_tensor(self, spec: InterventionSpec, key: str) -> torch.Tensor:
        if spec.value is not None:
            return spec.value
        if spec.source_cache is None:
            raise ValueError("replacement requires value or source_cache")
        if spec.source_cache.attention_type != self.attention_type:
            raise ValueError(
                f"source cache attention_type={spec.source_cache.attention_type!r} does not match "
                f"model attention_type={self.attention_type!r}"
            )
        source_key = spec.source_site
        if source_key is None:
            source_key = key
        elif spec.layer is not None and "[" not in source_key:
            source_key = format_site_name(source_key, layer=spec.layer)
        if source_key not in spec.source_cache.records:
            raise ValueError(f"source cache does not contain site {source_key!r}")
        return spec.source_cache.records[source_key].tensor

    def _patch_tensor(self, tensor: torch.Tensor, replacement: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        if replacement.shape != tensor.shape:
            raise ValueError(f"replacement shape {tuple(replacement.shape)} does not match {tuple(tensor.shape)}")
        patched = tensor.clone()
        batch_index = spec.batch_indices if spec.batch_indices is not None else slice(None)
        token_index = spec.token_indices if spec.token_indices is not None else slice(None)
        if tensor.ndim >= 3:
            patched[batch_index, token_index, ...] = replacement[batch_index, token_index, ...]
        elif tensor.ndim == 2:
            patched[batch_index, token_index] = replacement[batch_index, token_index]
        else:
            patched = replacement.clone()
        return patched


def capture_activations(
    model,
    input_ids: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    sites: list[str] | None = None,
    detach: bool = False,
    cpu: bool = False,
    dtype: torch.dtype | None = None,
    checkpoint_path: str | Path | None = None,
    batch_metadata: dict[str, Any] | None = None,
    step: int | None = None,
    schedule_mode: str | None = None,
) -> CaptureResult:
    recorder = ActivationRecorder(
        model=model,
        sites=sites,
        detach=detach,
        cpu=cpu,
        dtype=dtype,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path is not None else None,
        batch_metadata=batch_metadata,
    )
    logits, loss = model(
        input_ids,
        labels,
        step=step,
        schedule_mode=schedule_mode,
        activation_recorder=recorder,
    )
    recorder.finalize()
    return CaptureResult(logits=logits, loss=loss, cache=recorder.to_cache(), missing_sites=recorder.missing_sites)


def hash_model_config(config: Any) -> str | None:
    if config is None:
        return None
    if hasattr(config, "__dataclass_fields__"):
        payload = asdict(config)
    elif isinstance(config, dict):
        payload = config
    else:
        return None
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
