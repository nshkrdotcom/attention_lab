from __future__ import annotations

import argparse
import importlib.util
import json
from copy import deepcopy

import pytest
import torch

from attention_lab.models.attention.multi_qkv_common import MultiQKVDebugRouteOverride, override_multi_qkv_routes
from attention_lab.models.gpt import GPT, GPTConfig, config_from_dict
from attention_lab.queue.mechanism_checks import evaluate_mechanism_activity
from attention_lab.training.checkpointing import save_checkpoint
from attention_lab.training.attention_diagnostics import collect_attention_diagnostics
from attention_lab.training.config import save_config
from attention_lab.training.optim import build_optimizer


ROUTES = {
    "multi_qkv_static_3track_global": "layer_mod",
    "multi_qkv_train_rotation_3track_global": "layer_plus_step_train_layer_eval",
    "multi_qkv_position_rotation_3track_global": "layer_plus_position",
}


def tiny_multi_qkv_config(attention_type: str) -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=3,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type=attention_type,
        qkv_track_count=3,
        qkv_global_bank=True,
        qkv_route_formula=ROUTES[attention_type],
    )


@pytest.mark.parametrize(
    "attention_type",
    [
        "multi_qkv_static_3track_global",
        "multi_qkv_train_rotation_3track_global",
        "multi_qkv_position_rotation_3track_global",
    ],
)
def test_multi_qkv_attention_diagnostics_emit_required_fields(attention_type: str):
    torch.manual_seed(1)
    model = GPT(tiny_multi_qkv_config(attention_type))
    idx = torch.randint(0, 64, (2, 8))

    _, loss = model(idx, idx, step=0, schedule_mode="train")
    assert loss is not None
    loss.backward()

    diag = model.transformer.h[0].attn.attention_diagnostics(step=0, layer=0)
    assert diag is not None
    for key in (
        "attention_type",
        "active_track_index",
        "active_track_counts",
        "per_track_gradient_norm",
        "per_track_qkv_weight_norm",
        "per_track_output_norm",
        "track_entropy",
        "route_formula",
        "uses_global_bank",
        "layer_idx",
        "step",
        "last_forward_step",
        "schedule_mode",
        "track_count",
        "track_gradient_norm",
        "position_routing_enabled",
        "eval_freeze_mode",
    ):
        assert key in diag
    assert diag["attention_type"] == attention_type
    assert diag["uses_global_bank"] is True
    assert diag["track_count"] == 3
    assert diag["schedule_mode"] == "train"
    assert diag["last_forward_step"] == 0
    if attention_type == "multi_qkv_position_rotation_3track_global":
        assert diag["track_gradient_norm"] is None
        assert all(diag["per_track_gradient_norm"][str(track)] > 0.0 for track in range(3))
    else:
        assert diag["track_gradient_norm"] is not None and diag["track_gradient_norm"] > 0.0


@pytest.mark.parametrize(
    "attention_type",
    [
        "multi_qkv_static_3track_global",
        "multi_qkv_train_rotation_3track_global",
    ],
)
def test_scalar_hard_routing_gradients_go_to_active_track_only(attention_type: str):
    torch.manual_seed(2)
    config = tiny_multi_qkv_config(attention_type)
    config.n_layer = 1
    model = GPT(config)
    idx = torch.randint(0, 64, (2, 8))

    _, loss = model(idx, idx, step=0, schedule_mode="train")
    assert loss is not None
    loss.backward()

    diag = model.transformer.h[0].attn.attention_diagnostics(step=0, layer=0)
    assert diag is not None
    grads = diag["per_track_gradient_norm"]

    assert grads["0"] > 0.0
    assert grads["1"] == 0.0
    assert grads["2"] == 0.0


def test_position_routing_gradients_reach_all_tracks_for_layer0():
    torch.manual_seed(3)
    model = GPT(tiny_multi_qkv_config("multi_qkv_position_rotation_3track_global"))
    idx = torch.randint(0, 64, (2, 8))

    _, loss = model(idx, idx, step=0, schedule_mode="train")
    assert loss is not None
    loss.backward()

    diag = model.transformer.h[0].attn.attention_diagnostics(step=0, layer=0)
    assert diag is not None
    grads = diag["per_track_gradient_norm"]

    assert grads["0"] > 0.0
    assert grads["1"] > 0.0
    assert grads["2"] > 0.0


def test_position_rotation_diagnostics_count_all_tracks():
    torch.manual_seed(4)
    model = GPT(tiny_multi_qkv_config("multi_qkv_position_rotation_3track_global"))
    idx = torch.randint(0, 64, (2, 8))
    _, loss = model(idx, idx, step=2, schedule_mode="train")
    assert loss is not None
    loss.backward()
    rows = collect_attention_diagnostics(model, step=2)
    first = rows[0]
    assert first["active_track_index"] is None
    assert first["active_track_counts"] == {"0": 3, "1": 3, "2": 2}
    assert all(count > 0 for count in first["active_track_counts"].values())


def test_diagnostics_rows_are_json_serializable():
    model = GPT(tiny_multi_qkv_config("multi_qkv_train_rotation_3track_global"))
    model.eval()
    idx = torch.randint(0, 64, (1, 8))
    model(idx, schedule_mode="eval")
    rows = collect_attention_diagnostics(model, step=0)
    json.dumps(rows)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _qkv_row(
    *,
    attention_type: str,
    route_formula: str,
    layer_idx: int,
    step: int,
    active_track_index: int | None,
    active_track_counts: dict[str, int],
    per_track_gradient_norm: dict[str, float],
    schedule_mode: str = "train",
    last_forward_step: int | None = None,
    position_routing_enabled: bool = False,
    eval_freeze_mode: bool = False,
):
    return {
        "schema_version": 1,
        "experiment_id": "E002_multitrack_qkv_shift_register",
        "run_name": "tiny_qkv_test",
        "attention_type": attention_type,
        "route_formula": route_formula,
        "uses_global_bank": True,
        "track_count": 3,
        "layer_idx": layer_idx,
        "layer": layer_idx,
        "step": step,
        "last_forward_step": step if last_forward_step is None else last_forward_step,
        "schedule_mode": schedule_mode,
        "active_track_index": active_track_index,
        "active_track_counts": active_track_counts,
        "track_gradient_norm": None if active_track_index is None else per_track_gradient_norm[str(active_track_index)],
        "per_track_gradient_norm": per_track_gradient_norm,
        "per_track_qkv_weight_norm": {"0": 1.0, "1": 1.0, "2": 1.0},
        "position_routing_enabled": position_routing_enabled,
        "eval_freeze_mode": eval_freeze_mode,
    }


def _scalar_counts(active_track: int) -> dict[str, int]:
    return {str(track): (8 if track == active_track else 0) for track in range(3)}


def _scalar_grads(active_track: int) -> dict[str, float]:
    return {str(track): (0.1 if track == active_track else 0.0) for track in range(3)}


def _load_destructive_runner(repo_root):
    script_path = repo_root / "scripts" / "qkv_track_destructive_test.py"
    spec = importlib.util.spec_from_file_location("qkv_track_destructive_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_destructive_test


def test_qkv_track_activity_mechanism_check_accepts_valid_static_diagnostics(tmp_path):
    diagnostics_path = tmp_path / "attention_diagnostics.jsonl"
    _write_jsonl(
        diagnostics_path,
        [
            _qkv_row(
                attention_type="multi_qkv_static_3track_global",
                route_formula="layer_idx % track_count",
                layer_idx=layer_idx,
                step=0,
                active_track_index=layer_idx,
                active_track_counts=_scalar_counts(layer_idx),
                per_track_gradient_norm=_scalar_grads(layer_idx),
            )
            for layer_idx in range(3)
        ],
    )

    verdict = evaluate_mechanism_activity(
        attention_type="multi_qkv_static_3track_global",
        diagnostics_path=diagnostics_path,
    )

    assert verdict.passed
    assert verdict.details["tracks_with_nonzero_gradients"] == ["0", "1", "2"]


def test_qkv_track_activity_mechanism_check_accepts_valid_train_rotation_diagnostics(tmp_path):
    diagnostics_path = tmp_path / "attention_diagnostics.jsonl"
    rows = []
    route = "(layer_idx + step) % track_count during train; layer_idx % track_count during eval/generate"
    for step in (0, 1):
        for layer_idx in range(3):
            active = (layer_idx + step) % 3
            rows.append(
                _qkv_row(
                    attention_type="multi_qkv_train_rotation_3track_global",
                    route_formula=route,
                    layer_idx=layer_idx,
                    step=step,
                    active_track_index=active,
                    active_track_counts=_scalar_counts(active),
                    per_track_gradient_norm=_scalar_grads(active),
                    eval_freeze_mode=True,
                )
            )
    _write_jsonl(diagnostics_path, rows)

    verdict = evaluate_mechanism_activity(
        attention_type="multi_qkv_train_rotation_3track_global",
        diagnostics_path=diagnostics_path,
    )

    assert verdict.passed
    assert verdict.details["active_tracks_seen"] == ["0", "1", "2"]
    assert verdict.details["train_steps_seen"] == [0, 1]


def test_qkv_track_activity_mechanism_check_accepts_valid_position_diagnostics(tmp_path):
    diagnostics_path = tmp_path / "attention_diagnostics.jsonl"
    _write_jsonl(
        diagnostics_path,
        [
            _qkv_row(
                attention_type="multi_qkv_position_rotation_3track_global",
                route_formula="(layer_idx + position) % track_count",
                layer_idx=0,
                step=0,
                active_track_index=None,
                active_track_counts={"0": 3, "1": 3, "2": 2},
                per_track_gradient_norm={"0": 0.1, "1": 0.2, "2": 0.3},
                position_routing_enabled=True,
            )
        ],
    )

    verdict = evaluate_mechanism_activity(
        attention_type="multi_qkv_position_rotation_3track_global",
        diagnostics_path=diagnostics_path,
    )

    assert verdict.passed
    assert verdict.details["tracks_with_nonzero_gradients"] == ["0", "1", "2"]


def test_qkv_track_activity_mechanism_check_rejects_empty_diagnostics(tmp_path):
    diagnostics_path = tmp_path / "attention_diagnostics.jsonl"
    diagnostics_path.write_text("", encoding="utf-8")

    verdict = evaluate_mechanism_activity(
        attention_type="multi_qkv_static_3track_global",
        diagnostics_path=diagnostics_path,
    )

    assert not verdict.passed


def test_qkv_track_activity_mechanism_check_rejects_zero_diagnostics(tmp_path):
    diagnostics_path = tmp_path / "attention_diagnostics.jsonl"
    _write_jsonl(
        diagnostics_path,
        [
            _qkv_row(
                attention_type="multi_qkv_static_3track_global",
                route_formula="layer_idx % track_count",
                layer_idx=layer_idx,
                step=0,
                active_track_index=layer_idx,
                active_track_counts=_scalar_counts(layer_idx),
                per_track_gradient_norm={"0": 0.0, "1": 0.0, "2": 0.0},
            )
            for layer_idx in range(3)
        ],
    )

    verdict = evaluate_mechanism_activity(
        attention_type="multi_qkv_static_3track_global",
        diagnostics_path=diagnostics_path,
    )

    assert not verdict.passed
    assert "zero" in verdict.reason


def test_qkv_track_activity_mechanism_check_rejects_position_scalar_only_routing(tmp_path):
    diagnostics_path = tmp_path / "attention_diagnostics.jsonl"
    _write_jsonl(
        diagnostics_path,
        [
            _qkv_row(
                attention_type="multi_qkv_position_rotation_3track_global",
                route_formula="(layer_idx + position) % track_count",
                layer_idx=0,
                step=0,
                active_track_index=None,
                active_track_counts={"0": 8, "1": 0, "2": 0},
                per_track_gradient_norm={"0": 0.1, "1": 0.0, "2": 0.0},
                position_routing_enabled=True,
            )
        ],
    )

    verdict = evaluate_mechanism_activity(
        attention_type="multi_qkv_position_rotation_3track_global",
        diagnostics_path=diagnostics_path,
    )

    assert not verdict.passed
    assert "one active track" in verdict.reason


def test_qkv_track_activity_mechanism_check_rejects_train_rotation_missing_step(tmp_path):
    diagnostics_path = tmp_path / "attention_diagnostics.jsonl"
    _write_jsonl(
        diagnostics_path,
        [
            _qkv_row(
                attention_type="multi_qkv_train_rotation_3track_global",
                route_formula=(
                    "(layer_idx + step) % track_count during train; "
                    "layer_idx % track_count during eval/generate"
                ),
                layer_idx=0,
                step=0,
                last_forward_step=None,
                active_track_index=0,
                active_track_counts=_scalar_counts(0),
                per_track_gradient_norm=_scalar_grads(0),
                eval_freeze_mode=True,
            )
            | {"last_forward_step": None}
        ],
    )

    verdict = evaluate_mechanism_activity(
        attention_type="multi_qkv_train_rotation_3track_global",
        diagnostics_path=diagnostics_path,
    )

    assert not verdict.passed
    assert "null last_forward_step" in verdict.reason


def test_destructive_route_override_resets_after_context_manager():
    model = GPT(tiny_multi_qkv_config("multi_qkv_static_3track_global"))
    modules = [block.attn for block in model.transformer.h]
    assert all(module.debug_route_override is None for module in modules)

    override = MultiQKVDebugRouteOverride(mode="rotate_tracks")
    with override_multi_qkv_routes(model, override):
        assert all(module.debug_route_override is override for module in modules)

    assert all(module.debug_route_override is None for module in modules)


def test_qkv_track_destructive_test_writes_expected_json(tmp_path, repo_root, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    config = deepcopy(tiny_config(tmp_path, data_root, max_steps=1))
    config["run"]["name"] = "tiny_multi_qkv_static"
    config["run"]["out_dir"] = str(tmp_path / "runs" / "tiny_multi_qkv_static")
    config["model"].update(
        {
            "attention_type": "multi_qkv_static_3track_global",
            "qkv_track_count": 3,
            "qkv_global_bank": True,
            "qkv_route_formula": "layer_mod",
            "n_layer": 3,
            "n_head": 2,
        }
    )
    config_path = tmp_path / "tiny_multi_qkv_static.yaml"
    save_config(config, config_path)

    model = GPT(config_from_dict(config["model"], config["data"]))
    optimizer = build_optimizer(model, weight_decay=0.1, learning_rate=0.001, device_type="cpu", master_process=False)
    checkpoint = save_checkpoint(config["run"]["out_dir"], model, optimizer, config, step=0)
    out_path = tmp_path / "qkv_track_destructive_test.json"

    run_destructive_test = _load_destructive_runner(repo_root)
    result = run_destructive_test(
        argparse.Namespace(
            config=str(config_path),
            checkpoint=str(checkpoint),
            data_root=str(data_root),
            split="val",
            B=2,
            T=8,
            num_batches=1,
            dtype="float32",
            device="cpu",
            mode="eval",
            perturbation=["rotate_tracks"],
            forced_track=None,
            out=str(out_path),
        )
    )

    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == result
    assert result["schema_version"] == 1
    assert result["experiment_id"] == "E002_multitrack_qkv_shift_register"
    assert result["run_name"] == "tiny_multi_qkv_static"
    assert result["perturbations"]
    assert {"name", "natural_loss", "perturbed_loss", "loss_delta", "mean_abs_logit_delta", "max_abs_logit_delta"}.issubset(
        result["perturbations"][0]
    )
    assert isinstance(result["destructive_test_passed"], bool)


def test_attention_diagnostics_schema_includes_multi_qkv_and_cp_fields(repo_root):
    schema = json.loads((repo_root / "reports/schema/attention_diagnostics.schema.json").read_text(encoding="utf-8"))
    properties = schema["properties"]

    for key in (
        "schema_version",
        "experiment_id",
        "run_name",
        "route_formula",
        "uses_global_bank",
        "track_count",
        "last_forward_step",
        "schedule_mode",
        "active_track_index",
        "active_track_counts",
        "track_gradient_norm",
        "per_track_gradient_norm",
        "per_track_qkv_weight_norm",
        "position_routing_enabled",
        "eval_freeze_mode",
    ):
        assert key in properties

    for key in ("cp_gradient_norm", "cp_score_mean", "cp_to_standard_score_std_ratio"):
        assert key in properties


def test_qkv_track_activity_mechanism_check_accepts_legacy_nonzero_output_delta(tmp_path):
    diagnostics_path = tmp_path / "attention_diagnostics.jsonl"
    diagnostics_path.write_text(
        json.dumps(
            {
                "attention_type": "qkv_shift_legacy",
                "track_output_delta": 0.1,
                "track_gradient_norm": 0.1,
                "per_track_gradient_norm": {"0": 0.1, "1": 0.0, "2": 0.0},
                "active_track_counts": {"0": 8, "1": 0, "2": 0},
                "uses_global_bank": True,
                "track_count": 3,
                "per_track_qkv_weight_norm": {"0": 1.0, "1": 1.0, "2": 1.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    verdict = evaluate_mechanism_activity(
        attention_type="multi_qkv_static_3track_global",
        diagnostics_path=diagnostics_path,
    )

    assert verdict.active is True
