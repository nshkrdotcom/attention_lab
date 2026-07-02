from __future__ import annotations

from collections.abc import Iterable

from attention_lab.mechanisms.specs import HookSiteSpec, HookSiteStatus


class UnknownAttentionTypeError(ValueError):
    pass


def _site(
    name: str,
    family: str,
    tensor_kind: str,
    shape: tuple[str, ...],
    description: str,
    *,
    architecture: str | None = None,
    layer_indexed: bool = True,
) -> HookSiteSpec:
    return HookSiteSpec(
        name=name,
        family=family,
        tensor_kind=tensor_kind,
        layer_indexed=layer_indexed,
        shape_semantics=shape,
        architecture=architecture,
        description=description,
    )


STANDARD_SITES = (
    _site("resid_pre[layer]", "standard", "activation", ("batch", "token", "embed"), "Block input residual."),
    _site("attn_q[layer]", "standard", "activation", ("batch", "head", "token", "head_dim"), "Attention queries."),
    _site("attn_k[layer]", "standard", "activation", ("batch", "head", "token", "head_dim"), "Attention keys."),
    _site("attn_v[layer]", "standard", "activation", ("batch", "head", "token", "head_dim"), "Attention values."),
    _site("attn_out[layer]", "standard", "activation", ("batch", "token", "embed"), "Attention output."),
    _site("resid_mid[layer]", "standard", "activation", ("batch", "token", "embed"), "Residual after attention."),
    _site("mlp_out[layer]", "standard", "activation", ("batch", "token", "embed"), "MLP output."),
    _site("resid_post[layer]", "standard", "activation", ("batch", "token", "embed"), "Residual after MLP."),
    _site("logits", "standard", "activation", ("batch", "token", "vocab"), "Final token logits.", layer_indexed=False),
)

OPERATOR_SITES = (
    _site(
        "operator_probs[layer]",
        "operator_valued_attention",
        "probability",
        ("batch", "token", "operator"),
        "Router probabilities over add/suppress/gate/transform/bind operators.",
        architecture="operator_valued_attention",
    ),
    _site(
        "operator_add_out[layer]",
        "operator_valued_attention",
        "activation",
        ("batch", "token", "embed"),
        "Add operator output.",
        architecture="operator_valued_attention",
    ),
    _site(
        "operator_suppress_out[layer]",
        "operator_valued_attention",
        "activation",
        ("batch", "token", "embed"),
        "Negative signed suppress operator output.",
        architecture="operator_valued_attention",
    ),
    _site(
        "operator_gate_out[layer]",
        "operator_valued_attention",
        "activation",
        ("batch", "token", "embed"),
        "Gate operator output.",
        architecture="operator_valued_attention",
    ),
    _site(
        "operator_transform_out[layer]",
        "operator_valued_attention",
        "activation",
        ("batch", "token", "embed"),
        "Transform operator output or explicit zero tensor when disabled.",
        architecture="operator_valued_attention",
    ),
    _site(
        "operator_bind_out[layer]",
        "operator_valued_attention",
        "activation",
        ("batch", "token", "embed"),
        "Bind operator output or explicit zero tensor when disabled.",
        architecture="operator_valued_attention",
    ),
    _site(
        "operator_combined_out[layer]",
        "operator_valued_attention",
        "activation",
        ("batch", "token", "embed"),
        "Probability-weighted operator mixture before residual dropout.",
        architecture="operator_valued_attention",
    ),
)

DIFFERENTIAL_SITES = tuple(
    _site(name, "differential_qkv_anti_value", kind, shape, description, architecture="differential_qkv_anti_value")
    for name, kind, shape, description in (
        ("pos_q[layer]", "activation", ("batch", "head", "token", "head_dim"), "Positive branch query tensor."),
        ("pos_k[layer]", "activation", ("batch", "head", "token", "head_dim"), "Positive branch key tensor."),
        ("pos_v[layer]", "activation", ("batch", "head", "token", "head_dim"), "Positive branch value tensor."),
        ("neg_q[layer]", "activation", ("batch", "head", "token", "head_dim"), "Negative branch query tensor."),
        ("neg_k[layer]", "activation", ("batch", "head", "token", "head_dim"), "Negative branch key tensor."),
        ("neg_v[layer]", "activation", ("batch", "head", "token", "head_dim"), "Negative branch value tensor."),
        ("pos_out[layer]", "activation", ("batch", "token", "embed"), "Merged positive branch attention output."),
        ("neg_out[layer]", "activation", ("batch", "token", "embed"), "Merged negative branch attention output."),
        ("branch_delta[layer]", "activation", ("batch", "token", "embed"), "pos_out - lambda * neg_out."),
        ("lambda[layer]", "parameter", (), "Positive subtractive branch scale."),
    )
)

SCOPE_SITES = tuple(
    _site(name, "scope_gated_qkv", kind, ("batch", "token", "embed"), description, architecture="scope_gated_qkv")
    for name, kind, description in (
        ("content_out[layer]", "activation", "Content stream attention output."),
        ("scope_out[layer]", "activation", "Scope stream attention output after scope scaling."),
        ("gate[layer]", "gate", "Receiver-side gate in [0, 1]."),
        ("content_scope_product[layer]", "activation", "Elementwise content * scope product."),
        ("gated_content[layer]", "activation", "Elementwise gate * content output."),
    )
)

MULTI_QKV_SITES = (
    _site(
        "track_q[layer, track]",
        "multi_qkv",
        "activation",
        ("batch", "token", "embed"),
        "Selected or available Q projection for a routed track.",
        architecture="multi_qkv",
    ),
    _site(
        "track_k[layer, track]",
        "multi_qkv",
        "activation",
        ("batch", "token", "embed"),
        "Selected or available K projection for a routed track.",
        architecture="multi_qkv",
    ),
    _site(
        "track_v[layer, track]",
        "multi_qkv",
        "activation",
        ("batch", "token", "embed"),
        "Selected or available V projection for a routed track.",
        architecture="multi_qkv",
    ),
    _site(
        "selected_track[layer]",
        "multi_qkv",
        "route",
        ("token",),
        "Scalar or per-token selected route track.",
        architecture="multi_qkv",
    ),
    _site(
        "track_out[layer]",
        "multi_qkv",
        "activation",
        ("batch", "token", "embed"),
        "Routed attention output before the layer-local projection.",
        architecture="multi_qkv",
    ),
)

CP_SITES = (
    _site(
        "cp_score[layer]",
        "cp",
        "score",
        ("batch", "head", "query_token", "key_token"),
        "Raw CP score augmentation before lambda scaling.",
        architecture="cp",
    ),
    _site(
        "cp_output[layer]",
        "cp",
        "score",
        ("batch", "head", "query_token", "key_token"),
        "Lambda-scaled CP score contribution.",
        architecture="cp",
    ),
    _site(
        "cp_rank_component[layer, rank]",
        "cp",
        "score",
        ("batch", "head", "query_token", "key_token"),
        "Per-rank CP score component. Declared but summarized only until optimized.",
        architecture="cp",
    ),
    _site("cp_lambda[layer]", "cp", "parameter", (), "CP branch scale.", architecture="cp"),
)

DYNAMIC_VALUE_SITES = tuple(
    _site(
        name,
        "dynamic_value_query_conditioned_attention",
        kind,
        ("batch", "token", "embed"),
        description,
        architecture="dynamic_value_query_conditioned_attention",
    )
    for name, kind, description in (
        ("static_value_content[layer]", "activation", "Static value content before dynamic read-mode gating."),
        ("dynamic_gate[layer]", "gate", "Receiver-conditioned dynamic value gate."),
        ("dynamic_delta[layer]", "activation", "dynamic_value_output - static_value_content."),
        ("dynamic_value_output[layer]", "activation", "Gate-conditioned value content before output projection."),
    )
)

Q3K3V3_SITES = tuple(
    _site(
        name,
        "q3k3v3_role_routed_attention",
        "activation",
        ("batch", "token", "embed"),
        description,
        architecture="q3k3v3_role_routed_attention",
    )
    for name, description in (
        ("content_out[layer]", "Content role stream output."),
        ("operator_out[layer]", "Operator role stream output."),
        ("binding_out[layer]", "Binding role stream output."),
        ("content_operator_product[layer]", "Elementwise content * operator product."),
        ("content_binding_product[layer]", "Elementwise content * binding product."),
        ("operator_binding_product[layer]", "Elementwise operator * binding product."),
    )
)

ARCHITECTURE_SITES: dict[str, tuple[HookSiteSpec, ...]] = {
    "standard": (),
    "operator_valued_attention": OPERATOR_SITES,
    "differential_qkv_anti_value": DIFFERENTIAL_SITES,
    "scope_gated_qkv": SCOPE_SITES,
    "multi_qkv_static_3track_global": MULTI_QKV_SITES,
    "multi_qkv_train_rotation_3track_global": MULTI_QKV_SITES,
    "multi_qkv_position_rotation_3track_global": MULTI_QKV_SITES,
    "cp_bilinear": CP_SITES,
    "cp_trilinear": CP_SITES,
    "dynamic_value_query_conditioned_attention": DYNAMIC_VALUE_SITES,
    "q3k3v3_role_routed_attention": Q3K3V3_SITES,
}

UNSUPPORTED_RUNTIME_SITES = {
    ("cp_bilinear", "cp_rank_component[layer, rank]"): "full per-rank tensor capture is summary-only until optimized",
    ("cp_trilinear", "cp_rank_component[layer, rank]"): "full per-rank tensor capture is summary-only until optimized",
}


def site_base(name: str) -> str:
    return name.split("[", 1)[0]


def format_site_name(site: str, *, layer: int | None = None, track: int | None = None, rank: int | None = None) -> str:
    if layer is None:
        return site
    if track is not None:
        return f"{site}[{layer},{track}]"
    if rank is not None:
        return f"{site}[{layer},{rank}]"
    return f"{site}[{layer}]"


def _normalize_declared_name(name: str) -> str:
    base = site_base(name)
    if ", track" in name or base in {"track_q", "track_k", "track_v"}:
        return f"{base}[layer, track]"
    if ", rank" in name or base == "cp_rank_component":
        return f"{base}[layer, rank]"
    if "[" in name:
        return f"{base}[layer]"
    return base


def _known_attention_types() -> set[str]:
    return set(ARCHITECTURE_SITES)


def get_hook_site_specs(attention_type: str, *, include_standard: bool = True) -> tuple[HookSiteSpec, ...]:
    if attention_type not in ARCHITECTURE_SITES:
        raise UnknownAttentionTypeError(f"unknown attention_type for hook-site registry: {attention_type}")
    specs: list[HookSiteSpec] = []
    if include_standard:
        specs.extend(STANDARD_SITES)
    specs.extend(ARCHITECTURE_SITES[attention_type])
    return tuple(specs)


def get_hook_site_status(attention_type: str, site_name: str) -> HookSiteStatus:
    specs = get_hook_site_specs(attention_type)
    declared = {_normalize_declared_name(spec.name): spec for spec in specs}
    normalized = _normalize_declared_name(site_name)
    if normalized not in declared:
        return HookSiteStatus(site_name, declared=False, runtime_supported=False, reason="site is not declared")
    reason = UNSUPPORTED_RUNTIME_SITES.get((attention_type, normalized))
    if reason is not None:
        return HookSiteStatus(site_name, declared=True, runtime_supported=False, reason=reason)
    return HookSiteStatus(site_name, declared=True, runtime_supported=True)


def get_hook_site_spec(attention_type: str, site_name: str) -> HookSiteSpec | None:
    specs = get_hook_site_specs(attention_type)
    declared = {_normalize_declared_name(spec.name): spec for spec in specs}
    spec = declared.get(_normalize_declared_name(site_name))
    if spec is not None:
        return spec
    requested_base = site_base(site_name)
    for candidate in specs:
        if site_base(candidate.name) == requested_base:
            return candidate
    return None


def is_discrete_hook_site(attention_type: str, site_name: str) -> bool:
    try:
        spec = get_hook_site_spec(attention_type, site_name)
    except UnknownAttentionTypeError:
        return False
    return spec is not None and spec.tensor_kind == "route"


def supported_site_names(attention_type: str) -> list[str]:
    names = []
    for spec in get_hook_site_specs(attention_type):
        status = get_hook_site_status(attention_type, spec.name)
        if status.runtime_supported:
            names.append(spec.name)
    return names


def unsupported_site_names(attention_type: str) -> dict[str, str]:
    unsupported = {}
    for spec in get_hook_site_specs(attention_type):
        status = get_hook_site_status(attention_type, spec.name)
        if status.declared and not status.runtime_supported:
            unsupported[spec.name] = status.reason or "unsupported"
    return unsupported


def iter_attention_types() -> Iterable[str]:
    return tuple(sorted(_known_attention_types()))


def render_hook_site_docs() -> str:
    lines = ["# Mechanism Hook Sites", ""]
    for attention_type in sorted(_known_attention_types()):
        lines.append(f"## {attention_type}")
        for spec in sorted(get_hook_site_specs(attention_type), key=lambda item: item.name):
            status = get_hook_site_status(attention_type, spec.name)
            marker = "supported" if status.runtime_supported else f"unsupported: {status.reason}"
            lines.append(
                f"- `{spec.name}` [{spec.family}/{spec.tensor_kind}; {marker}] "
                f"{', '.join(spec.shape_semantics) or 'scalar'} - {spec.description}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
