from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal

import torch
import torch.nn as nn
from torch.nn import functional as F

ScheduleMode = Literal["train", "eval", "generate"]


@dataclass(frozen=True)
class MultiQKVRouteContext:
    """Routing context for deterministic Multi-QKV track selection."""

    layer_idx: int
    step: int | None
    schedule_mode: ScheduleMode
    position_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class MultiQKVDebugRouteOverride:
    """Temporary route perturbation for destructive/off-route tests."""

    mode: Literal["none", "force_track", "rotate_tracks", "zero_selected"] = "none"
    forced_track: int | None = None


MULTI_QKV_ATTENTION_TYPES = {
    "multi_qkv_static_3track_global",
    "multi_qkv_train_rotation_3track_global",
    "multi_qkv_position_rotation_3track_global",
}
MULTI_QKV_ROUTE_FORMULAS = {
    "multi_qkv_static_3track_global": "layer_mod",
    "multi_qkv_train_rotation_3track_global": "layer_plus_step_train_layer_eval",
    "multi_qkv_position_rotation_3track_global": "layer_plus_position",
}


def is_multi_qkv_attention(attention_type: str) -> bool:
    return attention_type in MULTI_QKV_ATTENTION_TYPES


def validate_canonical_multi_qkv_model_config(config: Any) -> None:
    """Enforce the canonical _3track_global contract at model construction."""

    attention_type = getattr(config, "attention_type", "standard")
    if not is_multi_qkv_attention(attention_type):
        return

    track_count = int(getattr(config, "qkv_track_count", 1))
    if track_count != 3:
        raise ValueError(f"{attention_type} requires qkv_track_count=3, got {track_count}")
    if getattr(config, "qkv_global_bank", False) is not True:
        raise ValueError(f"{attention_type} requires qkv_global_bank=True")
    expected_route = MULTI_QKV_ROUTE_FORMULAS[attention_type]
    route_formula = getattr(config, "qkv_route_formula", None)
    if route_formula != expected_route:
        raise ValueError(f"{attention_type} requires qkv_route_formula={expected_route!r}")


class MultiQKVGlobalBank(nn.Module):
    """Globally shared bank of packed bundled Q/K/V projection tracks."""

    def __init__(self, config: Any):
        super().__init__()
        track_count = int(getattr(config, "qkv_track_count", getattr(config, "multi_qkv_track_count", 3)))
        if track_count <= 0:
            raise ValueError("qkv_track_count must be positive")
        self.track_count = track_count
        self.n_embd = int(config.n_embd)
        self.c_attn_bank = nn.ModuleList(
            [nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) for _ in range(track_count)]
        )

    def project_track(self, x: torch.Tensor, track: int) -> torch.Tensor:
        if track < 0 or track >= self.track_count:
            raise IndexError(f"track={track} outside [0, {self.track_count})")
        return self.c_attn_bank[track](x)

    def project_all_tracks(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.project_track(x, track) for track in range(self.track_count)]

    def qkv_weight_norms(self) -> list[float]:
        values = []
        for track in range(self.track_count):
            total = torch.zeros((), device=self.c_attn_bank[track].weight.device)
            for parameter in self.c_attn_bank[track].parameters():
                total = total + parameter.detach().float().pow(2).sum()
            values.append(float(total.sqrt().item()))
        return values

    def qkv_gradient_norms(self) -> list[float]:
        values = []
        for track in range(self.track_count):
            total = None
            for parameter in self.c_attn_bank[track].parameters():
                if parameter.grad is None:
                    continue
                grad_total = parameter.grad.detach().float().pow(2).sum()
                total = grad_total if total is None else total + grad_total
            values.append(0.0 if total is None else float(total.sqrt().item()))
        return values

    def qkv_weight_norm_dict(self) -> dict[str, float]:
        return {str(index): value for index, value in enumerate(self.qkv_weight_norms())}

    def qkv_gradient_norm_dict(self) -> dict[str, float]:
        return {str(index): value for index, value in enumerate(self.qkv_gradient_norms())}


MultiQKVSharedBank = MultiQKVGlobalBank


class MultiQKVBaseCausalSelfAttention(nn.Module):
    """Causal self-attention using a globally shared hard-switched Q/K/V bank."""

    attention_type = "multi_qkv_base"
    route_formula = "undefined"
    position_routing_enabled = False
    eval_freeze_mode = False

    def __init__(
        self,
        config: Any,
        *,
        layer_idx: int,
        qkv_bank: MultiQKVGlobalBank | None = None,
        shared_qkv_bank: MultiQKVGlobalBank | None = None,
    ):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if qkv_bank is not None and shared_qkv_bank is not None and qkv_bank is not shared_qkv_bank:
            raise ValueError("qkv_bank and shared_qkv_bank must reference the same global bank")
        bank = qkv_bank if qkv_bank is not None else shared_qkv_bank
        if bank is None:
            raise ValueError("Multi-QKV attention requires a qkv_bank")
        self.layer_idx = int(layer_idx)
        self.__dict__["qkv_bank"] = bank
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.last_diagnostics: dict[str, Any] | None = None
        self._last_active_track_index: int | None = None
        self._last_active_track_counts: dict[str, int] | None = None
        self._last_schedule_mode: str | None = None
        self._last_step: int | None = None
        self.debug_route_override: MultiQKVDebugRouteOverride | None = None

    @property
    def track_count(self) -> int:
        return self.qkv_bank.track_count

    def select_scalar_track(self, context: MultiQKVRouteContext) -> int:
        raise NotImplementedError

    def select_position_tracks(
        self,
        context: MultiQKVRouteContext,
        *,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError

    def active_track_indices(
        self,
        *,
        step: int | None,
        positions: torch.Tensor,
        schedule_mode: str,
    ) -> torch.Tensor:
        context = MultiQKVRouteContext(
            layer_idx=self.layer_idx,
            step=step,
            schedule_mode=schedule_mode,
            position_ids=positions,
        )
        if self.position_routing_enabled:
            active_tracks = self.select_position_tracks(
                context,
                seq_len=positions.numel(),
                device=positions.device,
            )
        else:
            active_tracks = torch.tensor(self.select_scalar_track(context), dtype=torch.long, device=positions.device)
        return self._apply_debug_route_override(active_tracks)

    def _apply_debug_route_override(self, active_tracks: torch.Tensor) -> torch.Tensor:
        override = self.debug_route_override
        if override is None or override.mode == "none" or override.mode == "zero_selected":
            return active_tracks
        if override.mode == "rotate_tracks":
            return (active_tracks + 1) % self.track_count
        if override.mode == "force_track":
            if override.forced_track is None:
                raise ValueError("force_track override requires forced_track")
            forced = int(override.forced_track)
            if forced < 0 or forced >= self.track_count:
                raise ValueError(f"forced_track={forced} outside [0, {self.track_count})")
            return torch.full_like(active_tracks, forced)
        raise ValueError(f"unknown Multi-QKV debug route override mode: {override.mode}")

    def _split_qkv_heads(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split(self.n_embd, dim=2)
        return self._split_heads(q), self._split_heads(k), self._split_heads(v)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.size()
        head_size = channels // self.n_head
        return x.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)

    def _select_per_position(self, projections: list[torch.Tensor], active_tracks: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(projections, dim=2)
        batch_size, seq_len, _, channels = stacked.shape
        if active_tracks.dim() == 1:
            index = active_tracks.view(1, seq_len, 1, 1).expand(batch_size, seq_len, 1, channels)
        elif active_tracks.dim() == 2:
            index = active_tracks.view(batch_size, seq_len, 1, 1).expand(batch_size, seq_len, 1, channels)
        else:
            raise ValueError("active track tensor must be scalar, [T], or [B, T]")
        return stacked.gather(dim=2, index=index).squeeze(2)

    def _project(self, x: torch.Tensor, active_tracks: torch.Tensor) -> torch.Tensor:
        if active_tracks.dim() == 0:
            return self.qkv_bank.project_track(x, int(active_tracks.item()))

        return self._select_per_position(self.qkv_bank.project_all_tracks(x), active_tracks)

    def _track_counts(self, active_tracks: torch.Tensor, seq_len: int) -> dict[str, int]:
        if active_tracks.dim() == 0:
            values = [0 for _ in range(self.track_count)]
            values[int(active_tracks.item())] = seq_len
            return {str(index): value for index, value in enumerate(values)}
        counts = torch.bincount(active_tracks.reshape(-1).cpu(), minlength=self.track_count)
        return {str(index): int(counts[index].item()) for index in range(self.track_count)}

    def _record_diagnostics(
        self,
        *,
        step: int | None,
        schedule_mode: str,
        active_tracks: torch.Tensor,
        seq_len: int,
        selected_qkv: torch.Tensor,
        attention: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            counts = self._track_counts(active_tracks.detach(), seq_len)
            self._last_active_track_index = int(active_tracks.item()) if active_tracks.dim() == 0 else None
            self._last_active_track_counts = counts
            self._last_schedule_mode = schedule_mode
            self._last_step = step
            count_tensor = torch.tensor(list(counts.values()), dtype=torch.float32)
            probs = count_tensor / count_tensor.sum().clamp_min(1.0)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
            per_track_output_norm = {str(track): 0.0 for track in range(self.track_count)}
            for track_key, count in counts.items():
                track = int(track_key)
                if count == 0:
                    continue
                selected_q = selected_qkv[..., : self.n_embd]
                if active_tracks.dim() == 0:
                    per_track_output_norm[track_key] = float(selected_q.detach().float().norm().item())
                else:
                    mask = active_tracks == track
                    if mask.dim() == 1:
                        values = selected_q[:, mask]
                    else:
                        values = selected_q[mask]
                    per_track_output_norm[track_key] = (
                        float(values.detach().float().norm().item()) if values.numel() else 0.0
                    )
            attention_entropy = -(
                attention.detach().float() * attention.detach().float().clamp_min(1e-12).log()
            ).sum(dim=-1)
            self.last_diagnostics = {
                "schema_version": 1,
                "attention_type": self.attention_type,
                "track_count": self.track_count,
                "active_track_index": self._last_active_track_index,
                "active_track_counts": counts,
                "per_track_output_norm": per_track_output_norm,
                "track_output_delta": max(per_track_output_norm.values()) if per_track_output_norm else 0.0,
                "track_entropy": float(entropy.item()),
                "route_formula": self.route_formula,
                "uses_global_bank": True,
                "layer_idx": self.layer_idx,
                "step": step,
                "last_forward_step": step,
                "schedule_mode": schedule_mode,
                "position_routing_enabled": self.position_routing_enabled,
                "eval_freeze_mode": self.eval_freeze_mode,
                "attention_entropy_mean": float(attention_entropy.mean().item()),
                "attention_entropy_std": float(attention_entropy.std(unbiased=False).item()),
            }

    def attention_diagnostics(self, step: int, layer: int) -> dict[str, Any] | None:
        if self.last_diagnostics is None:
            return None
        return {
            **self.last_diagnostics,
            "step": self.last_diagnostics.get("step", step),
            "layer": layer,
            "track_gradient_norm": self._active_track_gradient_norm(),
            "per_track_gradient_norm": self.qkv_bank.qkv_gradient_norm_dict(),
            "per_track_qkv_weight_norm": self.qkv_bank.qkv_weight_norm_dict(),
        }

    def _active_track_gradient_norm(self) -> float | None:
        if self.last_diagnostics is None:
            return None
        active_track = self.last_diagnostics.get("active_track_index")
        if active_track is None:
            return None
        value = self.qkv_bank.qkv_gradient_norm_dict().get(str(active_track))
        return None if value is None else float(value)

    def forward(
        self,
        x: torch.Tensor,
        *,
        step: int | None = None,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        schedule_mode: str | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        del layer_idx
        batch_size, seq_len, channels = x.size()
        if position_ids is not None:
            positions = position_ids
        if positions is None:
            positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        else:
            positions = positions.to(device=x.device, dtype=torch.long)
        if schedule_mode is None:
            schedule_mode = "train" if self.training else "eval"
        if schedule_mode not in {"train", "eval", "generate"}:
            raise ValueError("schedule_mode must be one of: train, eval, generate")

        active_tracks = self.active_track_indices(step=step, positions=positions, schedule_mode=schedule_mode).to(
            device=x.device, dtype=torch.long
        )
        qkv = self._project(x, active_tracks)
        if self.debug_route_override is not None and self.debug_route_override.mode == "zero_selected":
            qkv = torch.zeros_like(qkv)
        q, k, v = self._split_qkv_heads(qkv)

        head_size = channels // self.n_head
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        self._record_diagnostics(
            step=step,
            schedule_mode=schedule_mode,
            active_tracks=active_tracks,
            seq_len=seq_len,
            selected_qkv=qkv,
            attention=attention,
        )
        attention = self.attn_dropout(attention)
        y = attention @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


MultiQKVGlobalCausalSelfAttention = MultiQKVBaseCausalSelfAttention


@contextmanager
def override_multi_qkv_routes(
    model: nn.Module,
    override: MultiQKVDebugRouteOverride,
) -> Iterator[None]:
    modules = [module for module in model.modules() if isinstance(module, MultiQKVBaseCausalSelfAttention)]
    previous = [module.debug_route_override for module in modules]
    try:
        for module in modules:
            module.debug_route_override = override
        yield
    finally:
        for module, old_value in zip(modules, previous):
            module.debug_route_override = old_value
