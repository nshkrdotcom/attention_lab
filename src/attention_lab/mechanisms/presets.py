from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SitePreset:
    site: str
    layer: int
    tensor_kind: str
    control_site: str | None
    continuous: bool = True

    @property
    def key(self) -> str:
        return f"{self.site}[{self.layer}]"


@dataclass(frozen=True)
class ControlPreset:
    run_name: str
    config_path: Path
    checkpoint_path: Path
    attention_type: str = "standard"


@dataclass(frozen=True)
class MechanismProbePreset:
    experiment_id: str
    candidate: str
    aliases: tuple[str, ...]
    tier: str
    executable: bool
    status: str
    attention_type: str
    run_name: str
    config_path: Path
    expected_checkpoint_path: Path
    matched_control: ControlPreset | None
    target_sites: tuple[SitePreset, ...]
    random_site_pool: tuple[SitePreset, ...]
    notes: str = ""


E003_CONTROL_SEED1 = ControlPreset(
    run_name="standard_refactor_control_30m_seed1_rung500",
    config_path=Path("configs/experiments/E003_qkv_architecture_gauntlet/standard_refactor_control_30m_seed1_rung500.yaml"),
    checkpoint_path=Path("runs/screen/standard_refactor_control_30m_seed1_rung500_7752266a764e/checkpoints/ckpt_last.pt"),
)

E004_CONTROL_SEED2 = ControlPreset(
    run_name="standard_refactor_control_30m_seed2_rung500",
    config_path=Path("configs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2_rung500.yaml"),
    checkpoint_path=Path("runs/screen/standard_refactor_control_30m_seed2_rung500_3cc31db15c20/checkpoints/ckpt_last.pt"),
)


E003_DIFFERENTIAL_PRESET = MechanismProbePreset(
    experiment_id="E003_qkv_architecture_gauntlet",
    candidate="differential",
    aliases=("differential", "differential_qkv", "differential_qkv_anti_value"),
    tier="tier1",
    executable=True,
    status="executable",
    attention_type="differential_qkv_anti_value",
    run_name="differential_qkv_anti_value_30m_seed1_rung500",
    config_path=Path("configs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1_rung500.yaml"),
    expected_checkpoint_path=Path(
        "runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt"
    ),
    matched_control=E003_CONTROL_SEED1,
    target_sites=(
        SitePreset("branch_delta", 0, "activation", "attn_out"),
        SitePreset("pos_out", 0, "activation", "attn_out"),
        SitePreset("neg_out", 0, "activation", "attn_out"),
    ),
    random_site_pool=(
        SitePreset("attn_out", 0, "activation", "attn_out"),
        SitePreset("resid_mid", 0, "activation", "resid_mid"),
        SitePreset("mlp_out", 0, "activation", "mlp_out"),
        SitePreset("resid_post", 0, "activation", "resid_post"),
    ),
    notes="Tier-1 E003 differential candidate; matched to seed1 standard refactor rung500 control.",
)


E004_OPERATOR_PRESET = MechanismProbePreset(
    experiment_id="E004_operator_binding_qkv_gauntlet",
    candidate="operator_valued",
    aliases=("operator", "operator_valued", "operator_valued_attention"),
    tier="tier1",
    executable=True,
    status="executable",
    attention_type="operator_valued_attention",
    run_name="operator_valued_attention_30m_seed2_rung500",
    config_path=Path(
        "configs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2_rung500.yaml"
    ),
    expected_checkpoint_path=Path(
        "runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt"
    ),
    matched_control=E004_CONTROL_SEED2,
    target_sites=(
        SitePreset("operator_probs", 0, "probability", None),
        SitePreset("operator_add_out", 0, "activation", "attn_out"),
        SitePreset("operator_suppress_out", 0, "activation", "attn_out"),
        SitePreset("operator_gate_out", 0, "activation", "attn_out"),
        SitePreset("operator_transform_out", 0, "activation", "attn_out"),
        SitePreset("operator_bind_out", 0, "activation", "attn_out"),
        SitePreset("operator_combined_out", 0, "activation", "attn_out"),
    ),
    random_site_pool=(
        SitePreset("attn_out", 0, "activation", "attn_out"),
        SitePreset("resid_mid", 0, "activation", "resid_mid"),
        SitePreset("mlp_out", 0, "activation", "mlp_out"),
        SitePreset("resid_post", 0, "activation", "resid_post"),
    ),
    notes="Tier-1 E004 operator-valued candidate; matched to seed2 standard refactor rung500 control.",
)


STUB_PRESETS = (
    MechanismProbePreset(
        experiment_id="E003_qkv_architecture_gauntlet",
        candidate="scope_gated",
        aliases=("scope_gated", "scope_gated_qkv"),
        tier="tier2",
        executable=False,
        status="stub_not_executable",
        attention_type="scope_gated_qkv",
        run_name="scope_gated_qkv_30m_seed1_rung500",
        config_path=Path("configs/experiments/E003_qkv_architecture_gauntlet/scope_gated_qkv_30m_seed1_rung500.yaml"),
        expected_checkpoint_path=Path("runs/screen/scope_gated_qkv_30m_seed1_rung500_bb3de557aae8/checkpoints/ckpt_last.pt"),
        matched_control=E003_CONTROL_SEED1,
        target_sites=(),
        random_site_pool=(),
        notes="Deferred Tier-2 preset; not executable in the Tier-1 suite.",
    ),
    MechanismProbePreset(
        experiment_id="E004_operator_binding_qkv_gauntlet",
        candidate="dynamic_value",
        aliases=("dynamic_value", "dynamic_value_query_conditioned_attention"),
        tier="tier2",
        executable=False,
        status="stub_not_executable",
        attention_type="dynamic_value_query_conditioned_attention",
        run_name="dynamic_value_query_conditioned_attention_30m_seed2_rung500",
        config_path=Path(
            "configs/experiments/E004_operator_binding_qkv_gauntlet/"
            "dynamic_value_query_conditioned_attention_30m_seed2_rung500.yaml"
        ),
        expected_checkpoint_path=Path(
            "runs/screen/dynamic_value_query_conditioned_attention_30m_seed2_rung500_99b5756e77ed/"
            "checkpoints/ckpt_last.pt"
        ),
        matched_control=E004_CONTROL_SEED2,
        target_sites=(),
        random_site_pool=(),
        notes="Diagnostic-rescue preset; not executable in the Tier-1 suite.",
    ),
    MechanismProbePreset(
        experiment_id="E004_operator_binding_qkv_gauntlet",
        candidate="q3k3v3",
        aliases=("q3k3v3", "q3k3v3_role_routed_attention"),
        tier="tier3",
        executable=False,
        status="stub_not_executable",
        attention_type="q3k3v3_role_routed_attention",
        run_name="q3k3v3_role_routed_attention_30m_seed2_rung020",
        config_path=Path(
            "configs/experiments/E004_operator_binding_qkv_gauntlet/q3k3v3_role_routed_attention_30m_seed2_rung020.yaml"
        ),
        expected_checkpoint_path=Path(
            "runs/screen/q3k3v3_role_routed_attention_30m_seed2_rung020_e640cc594862/checkpoints/ckpt_last.pt"
        ),
        matched_control=E004_CONTROL_SEED2,
        target_sites=(),
        random_site_pool=(),
        notes="Profiling/redesign preset; not executable in the Tier-1 suite.",
    ),
)


PRESETS = (E003_DIFFERENTIAL_PRESET, E004_OPERATOR_PRESET, *STUB_PRESETS)


def resolve_preset(experiment_id: str, candidate: str) -> MechanismProbePreset:
    normalized = candidate.strip().lower().replace("-", "_")
    for preset in PRESETS:
        if preset.experiment_id != experiment_id:
            continue
        if normalized == preset.candidate or normalized in preset.aliases:
            return preset
    known = sorted(
        preset.candidate
        for preset in PRESETS
        if preset.experiment_id == experiment_id
    )
    raise ValueError(f"unknown mechanism probe candidate {candidate!r} for {experiment_id}; known: {known}")


def site_presets_for_names(preset: MechanismProbePreset, names: list[str] | None) -> tuple[SitePreset, ...]:
    if not names:
        return preset.target_sites
    by_name = {site.site: site for site in preset.target_sites}
    selected = []
    for name in names:
        base = name.split("[", 1)[0]
        if base in by_name:
            selected.append(by_name[base])
        else:
            selected.append(SitePreset(base, 0, "activation", "attn_out"))
    return tuple(selected)
