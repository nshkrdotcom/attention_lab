from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_ACTIVITY_THRESHOLD = 1e-6
QKV_FAMILY_PREFIXES = ("multi_qkv_", "qkv_shift_")


@dataclass(frozen=True)
class MechanismCheckResult:
    active: bool | None
    note: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.active is True

    @property
    def reason(self) -> str:
        return self.note


def mechanism_check_name(attention_type: str, queue_config: dict[str, Any] | None = None) -> str | None:
    queue_config = queue_config or {}
    explicit = queue_config.get("mechanism_check")
    if explicit:
        return str(explicit)
    if attention_type == "standard":
        return None
    if attention_type in {"cp_bilinear", "cp_trilinear"}:
        return "cp_gradient_norm"
    if attention_type.startswith(QKV_FAMILY_PREFIXES):
        return "qkv_track_activity"
    return None


def load_diagnostic_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_mechanism_activity(
    *,
    attention_type: str,
    diagnostics_path: str | Path,
    queue_config: dict[str, Any] | None = None,
    threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
) -> MechanismCheckResult:
    queue_config = queue_config or {}
    check_name = mechanism_check_name(attention_type, queue_config)
    if check_name is None:
        return MechanismCheckResult(None, "standard attention has no mechanism activity requirement")

    rows = load_diagnostic_rows(diagnostics_path)
    if not rows:
        if queue_config.get("allow_missing_diagnostics", False):
            return MechanismCheckResult(None, "missing diagnostics allowed by queue.allow_missing_diagnostics")
        return MechanismCheckResult(None, f"missing diagnostics for mechanism_check={check_name}")

    if check_name == "cp_gradient_norm":
        return _check_cp_gradient_norm(rows, threshold)
    if check_name == "qkv_track_activity":
        return _check_qkv_track_activity(rows, threshold, attention_type=attention_type)
    return MechanismCheckResult(None, f"unknown mechanism_check={check_name}")


def _check_cp_gradient_norm(rows: list[dict[str, Any]], threshold: float) -> MechanismCheckResult:
    saw_value = False
    for row in rows:
        value = row.get("cp_gradient_norm")
        if value is None:
            continue
        saw_value = True
        if float(value) > threshold:
            return MechanismCheckResult(True, "cp_gradient_norm above threshold", {"rows_seen": len(rows)})
    if saw_value:
        return MechanismCheckResult(False, "cp_gradient_norm never exceeded threshold", {"rows_seen": len(rows)})
    return MechanismCheckResult(None, "diagnostics did not contain cp_gradient_norm", {"rows_seen": len(rows)})


def _check_qkv_track_activity(
    rows: list[dict[str, Any]],
    threshold: float,
    *,
    attention_type: str,
) -> MechanismCheckResult:
    qkv_rows = [row for row in rows if str(row.get("attention_type", "")).startswith(QKV_FAMILY_PREFIXES)]
    if not qkv_rows:
        if attention_type.startswith(QKV_FAMILY_PREFIXES):
            return _check_legacy_qkv_activity(rows, threshold)
        return MechanismCheckResult(None, "diagnostics did not contain Multi-QKV rows", {"rows_seen": len(rows)})

    details = _qkv_details(qkv_rows, threshold)
    common_failure = _qkv_common_failure(qkv_rows, threshold)
    if common_failure is not None:
        return MechanismCheckResult(False, common_failure, details)

    attention_types = {str(row.get("attention_type")) for row in qkv_rows}
    if len(attention_types) != 1:
        return MechanismCheckResult(False, "diagnostics contain mixed Multi-QKV attention types", details)
    attention_type = next(iter(attention_types))

    if attention_type == "multi_qkv_static_3track_global":
        failure = _check_static_qkv_rows(qkv_rows)
    elif attention_type == "multi_qkv_train_rotation_3track_global":
        failure = _check_train_rotation_qkv_rows(qkv_rows)
    elif attention_type == "multi_qkv_position_rotation_3track_global":
        failure = _check_position_rotation_qkv_rows(qkv_rows, threshold)
    else:
        failure = None

    if failure is not None:
        return MechanismCheckResult(False, failure, details)
    return MechanismCheckResult(True, "qkv_track_activity passed", details)


def _check_legacy_qkv_activity(rows: list[dict[str, Any]], threshold: float) -> MechanismCheckResult:
    """Compatibility path for pre-canonical QKV diagnostics.

    Older E002 skeletons and future QKV checks may emit mechanism fields without
    the full canonical A/B/C metadata. These are acceptable for screen-level
    compatibility only when a numeric activity field is nonzero.
    """

    for row in rows:
        values = [
            _max_nested_float(row.get("per_track_gradient_norm")),
            _as_float(row.get("track_gradient_norm")),
            _as_float(row.get("track_output_delta")),
            _as_float(row.get("branch_off_logit_delta")),
        ]
        if any(value is not None and value > threshold for value in values):
            return MechanismCheckResult(
                True,
                "legacy qkv diagnostic activity above threshold",
                {"rows_seen": len(rows)},
            )
    return MechanismCheckResult(False, "legacy qkv diagnostics never exceeded threshold", {"rows_seen": len(rows)})


def _qkv_details(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tracks_with_nonzero_gradients: set[str] = set()
    active_tracks_seen: set[str] = set()
    train_steps_seen: set[int] = set()
    route_formula_seen: set[str] = set()
    for row in rows:
        if row.get("schedule_mode") == "train" and row.get("step") is not None:
            train_steps_seen.add(int(row["step"]))
        if row.get("route_formula") is not None:
            route_formula_seen.add(str(row["route_formula"]))
        active_track = row.get("active_track_index")
        if active_track is not None:
            active_tracks_seen.add(str(active_track))
        for key, value in _track_dict(row.get("per_track_gradient_norm")).items():
            if float(value) > threshold:
                tracks_with_nonzero_gradients.add(key)
    return {
        "rows_seen": len(rows),
        "attention_type": rows[0].get("attention_type"),
        "tracks_with_nonzero_gradients": sorted(tracks_with_nonzero_gradients),
        "active_tracks_seen": sorted(active_tracks_seen),
        "train_steps_seen": sorted(train_steps_seen),
        "route_formula_seen": sorted(route_formula_seen),
    }


def _qkv_common_failure(rows: list[dict[str, Any]], threshold: float) -> str | None:
    saw_nonzero_gradient = False
    for row in rows:
        if row.get("uses_global_bank") is not True:
            return "uses_global_bank was not true for every Multi-QKV diagnostic row"
        if int(row.get("track_count") or 0) != 3:
            return "track_count was not 3 for canonical Multi-QKV diagnostics"

        weight_norms = _track_dict(row.get("per_track_qkv_weight_norm"))
        if set(weight_norms) != {"0", "1", "2"}:
            return "per_track_qkv_weight_norm missing required track keys"

        counts = _count_dict(row.get("active_track_counts"))
        if not counts:
            return "active_track_counts missing"
        if sum(counts.values()) <= 0:
            return "active_track_counts sum to zero"

        gradients = _track_dict(row.get("per_track_gradient_norm"))
        if any(value > threshold for value in gradients.values()):
            saw_nonzero_gradient = True
    if not saw_nonzero_gradient:
        return "all per_track_gradient_norm values were zero"
    return None


def _check_static_qkv_rows(rows: list[dict[str, Any]]) -> str | None:
    tracks_seen: set[str] = set()
    max_layer = -1
    for row in rows:
        if row.get("route_formula") != "layer_idx % track_count":
            return "static route_formula did not match layer_idx % track_count"
        if row.get("position_routing_enabled") is not False:
            return "static diagnostics unexpectedly enabled position routing"
        if row.get("eval_freeze_mode") is not False:
            return "static diagnostics unexpectedly enabled eval freeze mode"
        active_track = row.get("active_track_index")
        if active_track not in {0, 1, 2}:
            return "static diagnostics missing scalar active_track_index"
        counts = _count_dict(row.get("active_track_counts"))
        active_key = str(active_track)
        if counts.get(active_key, 0) != sum(counts.values()):
            return "static active_track_counts were not scalar hard-routed"
        tracks_seen.add(active_key)
        if row.get("layer_idx") is not None:
            max_layer = max(max_layer, int(row["layer_idx"]))
    if max_layer >= 2 and tracks_seen != {"0", "1", "2"}:
        return "static diagnostics did not show all three tracks across layers"
    return None


def _check_train_rotation_qkv_rows(rows: list[dict[str, Any]]) -> str | None:
    train_steps: set[int] = set()
    tracks_by_layer: dict[int, set[int]] = {}
    for row in rows:
        route = str(row.get("route_formula", ""))
        if "layer_idx + step" not in route or "eval/generate" not in route:
            return "train-rotation route_formula did not describe train rotation and eval freeze"
        if row.get("position_routing_enabled") is not False:
            return "train-rotation diagnostics unexpectedly enabled position routing"
        if row.get("eval_freeze_mode") is not True:
            return "train-rotation diagnostics did not mark eval_freeze_mode"
        counts = _count_dict(row.get("active_track_counts"))
        active_track = row.get("active_track_index")
        if active_track not in {0, 1, 2}:
            return "train-rotation diagnostics missing scalar active_track_index"
        if counts.get(str(active_track), 0) != sum(counts.values()):
            return "train-rotation active_track_counts were not scalar hard-routed"
        layer_idx = int(row.get("layer_idx", row.get("layer", 0)))
        if row.get("schedule_mode") == "train":
            if row.get("last_forward_step") is None:
                return "train-rotation train row had null last_forward_step"
            train_steps.add(int(row["last_forward_step"]))
            tracks_by_layer.setdefault(layer_idx, set()).add(int(active_track))
        elif row.get("schedule_mode") in {"eval", "generate"}:
            if int(active_track) != layer_idx % 3:
                return "train-rotation eval/generate row did not freeze to layer_idx % track_count"
    if len(train_steps) > 1 and not any(len(tracks) > 1 for tracks in tracks_by_layer.values()):
        return "train rotation did not show changing active tracks across steps"
    if train_steps and max(train_steps) > 0 and train_steps == {0}:
        return "train rotation rows all looked like step zero"
    return None


def _check_position_rotation_qkv_rows(rows: list[dict[str, Any]], threshold: float) -> str | None:
    saw_all_track_gradients = False
    for row in rows:
        if row.get("route_formula") != "(layer_idx + position) % track_count":
            return "position route_formula did not match (layer_idx + position) % track_count"
        if row.get("position_routing_enabled") is not True:
            return "position diagnostics did not enable position routing"
        if row.get("eval_freeze_mode") is not False:
            return "position diagnostics unexpectedly enabled eval freeze mode"
        if row.get("active_track_index") is not None:
            return "position diagnostics emitted scalar active_track_index"
        counts = _count_dict(row.get("active_track_counts"))
        if sum(1 for value in counts.values() if value > 0) <= 1 and sum(counts.values()) >= 3:
            return "position diagnostics used only one active track"
        gradients = _track_dict(row.get("per_track_gradient_norm"))
        if all(gradients.get(str(track), 0.0) > threshold for track in range(3)):
            saw_all_track_gradients = True
    if not saw_all_track_gradients:
        return "position diagnostics did not show nonzero gradients for all tracks in any row"
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _max_nested_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        numbers = [float(item) for item in value.values()]
        return max(numbers) if numbers else None
    if isinstance(value, (list, tuple)):
        numbers = [float(item) for item in value]
        return max(numbers) if numbers else None
    return float(value)


def _track_dict(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): float(item or 0.0) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {str(index): float(item or 0.0) for index, item in enumerate(value)}
    return {}


def _count_dict(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): int(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {str(index): int(item) for index, item in enumerate(value)}
    return {}
