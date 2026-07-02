from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


MECHANISM_PROBE_STATUSES = (
    "insufficient_evidence",
    "exploratory_probe_signal",
    "controlled_probe_signal",
    "candidate_mechanism_evidence",
)


@dataclass(frozen=True)
class ClaimGateResult:
    status: str
    status_vocabulary: str
    reasons: list[str]
    caps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_claim_gate(metrics: dict[str, Any]) -> ClaimGateResult:
    reasons: list[str] = []
    caps: list[str] = []

    if not metrics.get("has_real_probe_metrics", False):
        reasons.append("real trained probe metrics are missing")
        return _result("insufficient_evidence", reasons, caps)

    if metrics.get("exploratory", False):
        caps.append("exploratory")
        reasons.append("exploratory runs cannot make confirmatory mechanism claims")
        return _result("exploratory_probe_signal", reasons, caps)

    if metrics.get("probe_only", False):
        caps.append("probe_only")
        reasons.append("probe-only runs cannot reach candidate_mechanism_evidence")
        return _result("exploratory_probe_signal", reasons, caps)

    blockers = [
        ("minimum_n_passed", "minimum N failed"),
        ("confirmatory_floor_passed", "confirmatory task-suite size floor failed"),
        ("grouped_split_passed", "grouped split discipline did not pass"),
        ("hypothesis_doc_valid", "valid pre-registered hypothesis doc is missing"),
        ("matched_control_available", "matched control evidence is missing"),
        ("control_canonical", "control pairing is noncanonical or seed-mismatched"),
        ("random_site_null_available", "random-site null is unavailable"),
        ("stats_valid", "bootstrap/statistical results are invalid"),
        ("fdr_primary_passed", "FDR-corrected primary statistical gate failed"),
        ("bootstrap_primary_ci_excludes_null", "primary bootstrap CI does not exclude the null"),
        ("decoy_specificity_passed", "target-vs-decoy specificity gate failed"),
    ]
    for key, message in blockers:
        if not metrics.get(key, False):
            reasons.append(message)
    if metrics.get("min_n_below_floor", False):
        reasons.append("--min-n is below the committed confirmatory floor")
    if metrics.get("raw_delta_only", False):
        reasons.append("raw activation delta alone cannot pass a claim gate")

    if reasons:
        if any("noncanonical" in reason or "seed-mismatched" in reason for reason in reasons):
            caps.append("noncanonical_control")
        return _result("insufficient_evidence", reasons, caps)

    causal_valid = (
        metrics.get("patching_valid", False)
        and metrics.get("restoration_valid", False)
        and metrics.get("mediation_fraction_valid", False)
    )
    if not causal_valid:
        reasons.append("causal patch/restoration metrics are invalid or unavailable")
        return _result("controlled_probe_signal", reasons, caps)

    return _result("candidate_mechanism_evidence", reasons, caps)


def _result(status: str, reasons: list[str], caps: list[str]) -> ClaimGateResult:
    if status not in MECHANISM_PROBE_STATUSES:
        raise ValueError(f"unknown mechanism-probe status: {status}")
    return ClaimGateResult(
        status=status,
        status_vocabulary="mechanism_probe_scoped",
        reasons=list(reasons),
        caps=list(caps),
    )
