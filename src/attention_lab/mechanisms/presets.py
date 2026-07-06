from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SitePreset:
    site: str
    layer: int
    tensor_kind: str
    control_site: str | None
    continuous: bool = True
    full_layer_site: str | None = "attn_out"
    canonical: bool = True
    noncanonical_reason: str | None = None
    no_control_reason: str | None = None
    no_full_layer_comparator_reason: str | None = None

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
        SitePreset(
            "operator_probs",
            0,
            "probability",
            None,
            continuous=False,
            full_layer_site=None,
            no_control_reason="standard matched control has no operator probability site",
            no_full_layer_comparator_reason=(
                "low-dimensional operator probability site is capture/probe-only; no validated continuous "
                "patch/restoration intervention exists"
            ),
        ),
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


E002_CONTROL_SEED1_FULL = ControlPreset(
    run_name="standard_refactor_control_30m_seed1",
    config_path=Path("configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml"),
    checkpoint_path=Path(
        "runs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1/checkpoints/ckpt_last.pt"
    ),
)

E004_CONTROL_SEED2_FULL = ControlPreset(
    run_name="standard_refactor_control_30m_seed2",
    config_path=Path("configs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2.yaml"),
    checkpoint_path=Path(
        "runs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2/checkpoints/ckpt_last.pt"
    ),
)


E003_DIFFERENTIAL_FULL_PRESET = MechanismProbePreset(
    experiment_id="E003_qkv_architecture_gauntlet",
    candidate="differential_full",
    aliases=("differential_full", "differential_qkv_anti_value_full"),
    tier="tier1",
    executable=True,
    status="executable",
    attention_type="differential_qkv_anti_value",
    run_name="differential_qkv_anti_value_30m_seed1",
    config_path=Path("configs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1.yaml"),
    expected_checkpoint_path=Path(
        "runs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1/checkpoints/ckpt_last.pt"
    ),
    matched_control=E002_CONTROL_SEED1_FULL,
    target_sites=E003_DIFFERENTIAL_PRESET.target_sites,
    random_site_pool=E003_DIFFERENTIAL_PRESET.random_site_pool,
    notes=(
        "Same hypothesis and sites as the rung500 differential preset, re-pointed at the full "
        "3000-step checkpoint (promoted 2026-07-05) with a properly matched, equally-full-depth "
        "control (E002's standard_refactor_control_30m_seed1, config-identical to E003's own "
        "control apart from metadata) -- this makes the control canonical without needing "
        "--force-noncanonical-control, unlike re-running the rung500 preset against the new "
        "checkpoint."
    ),
)


E004_OPERATOR_FULL_PRESET = MechanismProbePreset(
    experiment_id="E004_operator_binding_qkv_gauntlet",
    candidate="operator_valued_full",
    aliases=("operator_valued_full", "operator_full"),
    tier="tier1",
    executable=True,
    status="executable",
    attention_type="operator_valued_attention",
    run_name="operator_valued_attention_30m_seed2",
    config_path=Path("configs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2.yaml"),
    expected_checkpoint_path=Path(
        "runs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2/checkpoints/ckpt_last.pt"
    ),
    matched_control=E004_CONTROL_SEED2_FULL,
    target_sites=E004_OPERATOR_PRESET.target_sites,
    random_site_pool=E004_OPERATOR_PRESET.random_site_pool,
    notes=(
        "Same hypothesis and sites as the rung500 operator-valued preset, re-pointed at the full "
        "3000-step checkpoint (promoted 2026-07-05) with a newly-promoted, equally-full-depth "
        "seed2 standard control -- makes the control canonical without --force-noncanonical-control."
    ),
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


PRESETS = (
    E003_DIFFERENTIAL_PRESET,
    E004_OPERATOR_PRESET,
    E003_DIFFERENTIAL_FULL_PRESET,
    E004_OPERATOR_FULL_PRESET,
    *STUB_PRESETS,
)


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


def site_presets_for_names(
    preset: MechanismProbePreset,
    names: list[str] | None,
    *,
    exploratory: bool = False,
    site_spec_file: str | Path | None = None,
) -> tuple[SitePreset, ...]:
    if not names:
        return preset.target_sites
    by_name = {site.site: site for site in preset.target_sites}
    by_key = {site.key: site for site in preset.target_sites}
    exploratory_specs = _load_exploratory_site_specs(site_spec_file) if site_spec_file else {}
    selected = []
    for name in names:
        base, layer = _parse_requested_site(name)
        key = f"{base}[{layer}]" if layer is not None else None
        if key is not None and key in by_key:
            selected.append(by_key[key])
        elif layer is None and base in by_name:
            selected.append(by_name[base])
        elif layer is not None and base in by_name:
            raise ValueError(
                f"unknown confirmatory site {name!r} for {preset.candidate}; "
                f"declared layer for {base!r} is {by_name[base].layer}"
            )
        elif exploratory:
            if base not in exploratory_specs:
                raise ValueError(
                    f"unknown exploratory site {base!r} requires explicit metadata via --site-spec-file"
                )
            if layer is not None and exploratory_specs[base].layer != layer:
                raise ValueError(
                    f"exploratory site {name!r} does not match --site-spec-file layer {exploratory_specs[base].layer}"
                )
            selected.append(exploratory_specs[base])
        else:
            known = ", ".join(sorted(by_name))
            raise ValueError(
                f"unknown confirmatory site {base!r} for {preset.candidate}; declared Tier-1 sites: {known}"
            )
    return tuple(selected)


def _parse_requested_site(name: str) -> tuple[str, int | None]:
    if "[" not in name:
        return name, None
    base, rest = name.split("[", 1)
    if not rest.endswith("]"):
        raise ValueError(f"invalid site specifier {name!r}")
    raw_layer = rest[:-1]
    if not raw_layer.isdigit():
        raise ValueError(f"invalid site layer in {name!r}")
    return base, int(raw_layer)


def _load_exploratory_site_specs(path: str | Path) -> dict[str, SitePreset]:
    spec_path = Path(path)
    raw = spec_path.read_text(encoding="utf-8")
    payload = json.loads(raw) if spec_path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError("--site-spec-file must contain a mapping")
    rows = payload.get("sites")
    if not isinstance(rows, list):
        raise ValueError("--site-spec-file must contain a sites list")
    specs: dict[str, SitePreset] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"site-spec row {index} must be a mapping")
        specs[_required_site_field(row, "site", index)] = _site_preset_from_spec(row, index)
    return specs


def _site_preset_from_spec(row: dict[str, Any], index: int) -> SitePreset:
    site = _required_site_field(row, "site", index)
    layer = row.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ValueError(f"site-spec row {index} requires integer layer")
    tensor_kind = _required_site_field(row, "tensor_kind", index)
    continuous = row.get("continuous")
    if not isinstance(continuous, bool):
        raise ValueError(f"site-spec row {index} requires boolean continuous")
    control_site = row.get("control_site")
    no_control_reason = row.get("no_control_reason")
    if control_site is None and not no_control_reason:
        raise ValueError(f"site-spec row {index} requires control_site or no_control_reason")
    if control_site is not None and not isinstance(control_site, str):
        raise ValueError(f"site-spec row {index} control_site must be a string when present")
    full_layer_site = row.get("full_layer_site")
    no_full_layer_reason = row.get("no_full_layer_comparator_reason")
    if full_layer_site is None and not no_full_layer_reason:
        raise ValueError(f"site-spec row {index} requires full_layer_site or no_full_layer_comparator_reason")
    if full_layer_site is not None and not isinstance(full_layer_site, str):
        raise ValueError(f"site-spec row {index} full_layer_site must be a string when present")
    return SitePreset(
        site=site,
        layer=layer,
        tensor_kind=tensor_kind,
        control_site=control_site,
        continuous=continuous,
        full_layer_site=full_layer_site,
        canonical=False,
        noncanonical_reason="exploratory site metadata supplied outside Tier-1 preset registry",
        no_control_reason=str(no_control_reason) if no_control_reason else None,
        no_full_layer_comparator_reason=str(no_full_layer_reason) if no_full_layer_reason else None,
    )


def _required_site_field(row: dict[str, Any], field_name: str, index: int) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"site-spec row {index} requires non-empty {field_name}")
    return value
