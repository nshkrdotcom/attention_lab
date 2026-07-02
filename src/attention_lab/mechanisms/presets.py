from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SitePreset:
    site: str
    layer: int
    tensor_kind: str
    control_sites: tuple[str, ...]
    continuous: bool = True

    @property
    def key(self) -> str:
        return f"{self.site}[{self.layer}]"


@dataclass(frozen=True)
class ControlPreset:
    run_name: str
    config: Path
    checkpoint: Path
    attention_type: str = "standard"


@dataclass(frozen=True)
class MechanismPreset:
    experiment_id: str
    candidate: str
    aliases: tuple[str, ...]
    attention_type: str
    tier: str
    status: str
    executable: bool
    config: Path | None
    checkpoint: Path | None
    sites: tuple[SitePreset, ...]
    matched_control: ControlPreset | None
    single_seed: bool = True


E003_CONTROL_SEED1 = ControlPreset(
    run_name="standard_refactor_control_30m_seed1_rung500",
    config=Path("configs/experiments/E003_qkv_architecture_gauntlet/standard_refactor_control_30m_seed1_rung500.yaml"),
    checkpoint=Path(
        "runs/screen/standard_refactor_control_30m_seed1_rung500_7752266a764e/checkpoints/ckpt_last.pt"
    ),
)

E004_CONTROL_SEED2 = ControlPreset(
    run_name="standard_refactor_control_30m_seed2_rung500",
    config=Path("configs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2_rung500.yaml"),
    checkpoint=Path(
        "runs/screen/standard_refactor_control_30m_seed2_rung500_3cc31db15c20/checkpoints/ckpt_last.pt"
    ),
)


PRESETS: tuple[MechanismPreset, ...] = (
    MechanismPreset(
        experiment_id="E003_qkv_architecture_gauntlet",
        candidate="differential",
        aliases=("differential", "differential_qkv_anti_value", "differential_qkv_anti_value_30m_seed1_rung500"),
        attention_type="differential_qkv_anti_value",
        tier="tier1",
        status="executable",
        executable=True,
        config=Path("configs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1_rung500.yaml"),
        checkpoint=Path(
            "runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt"
        ),
        sites=(
            SitePreset("branch_delta", 0, "activation", ("attn_out", "resid_mid")),
            SitePreset("pos_out", 0, "activation", ("attn_out", "resid_mid")),
            SitePreset("neg_out", 0, "activation", ("attn_out", "resid_mid")),
        ),
        matched_control=E003_CONTROL_SEED1,
    ),
    MechanismPreset(
        experiment_id="E004_operator_binding_qkv_gauntlet",
        candidate="operator_valued",
        aliases=("operator_valued", "operator_valued_attention", "operator_valued_attention_30m_seed2_rung500"),
        attention_type="operator_valued_attention",
        tier="tier1",
        status="executable",
        executable=True,
        config=Path(
            "configs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2_rung500.yaml"
        ),
        checkpoint=Path(
            "runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt"
        ),
        sites=(
            SitePreset("operator_combined_out", 0, "activation", ("attn_out", "resid_mid")),
            SitePreset("operator_add_out", 0, "activation", ("attn_out", "resid_mid")),
            SitePreset("operator_suppress_out", 0, "activation", ("attn_out", "resid_mid")),
            SitePreset("operator_gate_out", 0, "activation", ("attn_out", "resid_mid")),
            SitePreset("operator_transform_out", 0, "activation", ("attn_out", "resid_mid")),
            SitePreset("operator_bind_out", 0, "activation", ("attn_out", "resid_mid")),
            SitePreset("operator_probs", 0, "probability", (), continuous=True),
        ),
        matched_control=E004_CONTROL_SEED2,
    ),
    MechanismPreset(
        experiment_id="E003_qkv_architecture_gauntlet",
        candidate="scope_gated",
        aliases=("scope_gated", "scope_gated_qkv", "scope_gated_qkv_30m_seed1_rung500"),
        attention_type="scope_gated_qkv",
        tier="tier1_followup",
        status="stub_not_executable",
        executable=False,
        config=Path("configs/experiments/E003_qkv_architecture_gauntlet/scope_gated_qkv_30m_seed1_rung500.yaml"),
        checkpoint=Path("runs/screen/scope_gated_qkv_30m_seed1_rung500_bb3de557aae8/checkpoints/ckpt_last.pt"),
        sites=(),
        matched_control=E003_CONTROL_SEED1,
    ),
    MechanismPreset(
        experiment_id="E004_operator_binding_qkv_gauntlet",
        candidate="dynamic_value",
        aliases=(
            "dynamic_value",
            "dynamic_value_query_conditioned_attention",
            "dynamic_value_query_conditioned_attention_30m_seed2_rung500",
        ),
        attention_type="dynamic_value_query_conditioned_attention",
        tier="tier2",
        status="stub_not_executable",
        executable=False,
        config=Path(
            "configs/experiments/E004_operator_binding_qkv_gauntlet/"
            "dynamic_value_query_conditioned_attention_30m_seed2_rung500.yaml"
        ),
        checkpoint=Path(
            "runs/screen/dynamic_value_query_conditioned_attention_30m_seed2_rung500_99b5756e77ed/checkpoints/ckpt_last.pt"
        ),
        sites=(),
        matched_control=E004_CONTROL_SEED2,
    ),
    MechanismPreset(
        experiment_id="E004_operator_binding_qkv_gauntlet",
        candidate="q3k3v3",
        aliases=("q3k3v3", "q3k3v3_role_routed_attention", "q3k3v3_role_routed_attention_30m_seed2_rung020"),
        attention_type="q3k3v3_role_routed_attention",
        tier="tier3",
        status="stub_not_executable",
        executable=False,
        config=Path("configs/experiments/E004_operator_binding_qkv_gauntlet/q3k3v3_role_routed_attention_30m_seed2_rung020.yaml"),
        checkpoint=Path("runs/screen/q3k3v3_role_routed_attention_30m_seed2_rung020_e640cc594862/checkpoints/ckpt_last.pt"),
        sites=(),
        matched_control=E004_CONTROL_SEED2,
    ),
)


def get_preset(experiment_id: str, candidate: str) -> MechanismPreset:
    normalized = candidate.strip()
    for preset in PRESETS:
        if preset.experiment_id == experiment_id and normalized in preset.aliases:
            return preset
    known = ", ".join(sorted({alias for preset in PRESETS for alias in preset.aliases if preset.experiment_id == experiment_id}))
    raise ValueError(f"unknown mechanism probe preset for {experiment_id}/{candidate}. Known candidates: {known}")
