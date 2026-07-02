from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HookSiteSpec:
    name: str
    family: str
    tensor_kind: str
    layer_indexed: bool
    shape_semantics: tuple[str, ...]
    architecture: str | None
    description: str


@dataclass(frozen=True)
class HookSiteStatus:
    name: str
    declared: bool
    runtime_supported: bool
    reason: str | None = None
