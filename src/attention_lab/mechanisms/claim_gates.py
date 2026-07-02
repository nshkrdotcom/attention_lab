from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


INSUFFICIENT_EVIDENCE = "insufficient_evidence"
EXPLORATORY_PROBE_SIGNAL = "exploratory_probe_signal"
CONTROLLED_PROBE_SIGNAL = "controlled_probe_signal"
CANDIDATE_MECHANISM_EVIDENCE = "candidate_mechanism_evidence"

MECHANISM_PROBE_STATUS_LADDER = (
    INSUFFICIENT_EVIDENCE,
    EXPLORATORY_PROBE_SIGNAL,
    CONTROLLED_PROBE_SIGNAL,
    CANDIDATE_MECHANISM_EVIDENCE,
)


@dataclass(frozen=True)
class CellGateInputs:
    exploratory: bool
    probe_only: bool
    hypothesis_doc_valid: bool
    real_probe_metrics: bool
    min_n_passed: bool
    confirmatory_floor_met: bool
    grouped_split: bool
    matched_control_available: bool
    canonical_control: bool
    noncanonical_control: bool
    shuffled_null_passed: bool
    random_site_null_available: bool
    random_site_null_passed: bool
    matched_control_passed: bool
    primary_fdr_passed: bool
    primary_ci_passed: bool
    specificity_fdr_passed: bool
    specificity_ci_passed: bool
    patching_valid: bool
    mediation_valid: bool
    patching_fdr_passed: bool = True
    full_layer_patching_fdr_passed: bool = True
    mediation_fdr_passed: bool = True
    task_aligned_pooling: bool = True
    restoration_alignment_valid: bool = True
    canonical_site: bool = True
    force_noncanonical_control: bool = False
    extra_blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class CellGateResult:
    status: str
    blockers: tuple[str, ...]
    caps: tuple[str, ...]
    exploratory_signal: bool
    controlled_probe_gate_passed: bool
    candidate_mechanism_gate_passed: bool
    highest_status: str
    claim_gate_passed: bool
    status_kind: str
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blockers": list(self.blockers),
            "caps": list(self.caps),
            "exploratory_signal": self.exploratory_signal,
            "controlled_probe_gate_passed": self.controlled_probe_gate_passed,
            "candidate_mechanism_gate_passed": self.candidate_mechanism_gate_passed,
            "highest_status": self.highest_status,
            "highest_status_semantics": "highest claim ladder threshold reached for this cell",
            "claim_gate_passed": self.claim_gate_passed,
            "claim_gate_passed_semantics": (
                "candidate_mechanism_gate_passed compatibility alias; "
                "use controlled_probe_gate_passed or candidate_mechanism_gate_passed for precise consumers"
            ),
            "status_kind": self.status_kind,
            "inputs": self.inputs,
            "status_vocabulary_scope": (
                "mechanism-probe scoped; distinct from the repository's broader experiment status vocabulary"
            ),
        }


def evaluate_cell_claim_gate(inputs: CellGateInputs) -> CellGateResult:
    blockers: list[str] = list(inputs.extra_blockers)
    caps: list[str] = []
    if not inputs.real_probe_metrics:
        blockers.append("real trained probe metrics are missing")
    if not inputs.min_n_passed:
        blockers.append("minimum N failed")
    if not inputs.grouped_split:
        blockers.append("grouped split discipline missing")

    if blockers:
        return _result(INSUFFICIENT_EVIDENCE, blockers, caps, inputs)

    if inputs.exploratory:
        caps.append("exploratory mode caps claims below confirmatory evidence")
        return _result(EXPLORATORY_PROBE_SIGNAL, blockers, caps, inputs)
    if inputs.probe_only:
        caps.append("probe-only mode skips causal patching/restoration and caps claims")
        return _result(EXPLORATORY_PROBE_SIGNAL, blockers, caps, inputs)
    if not inputs.hypothesis_doc_valid:
        blockers.append("confirmatory run requires a valid pre-registered hypothesis doc")
        return _result(INSUFFICIENT_EVIDENCE, blockers, caps, inputs)
    if not inputs.confirmatory_floor_met:
        blockers.append("confirmatory task-suite size floor failed")
    if not inputs.matched_control_available:
        blockers.append("matched control evidence is unavailable")
    if not inputs.canonical_control:
        blockers.append("control pairing is noncanonical or seed-mismatched")
        if inputs.noncanonical_control or inputs.force_noncanonical_control:
            caps.append("noncanonical controls can never reach candidate_mechanism_evidence")
    if not inputs.shuffled_null_passed:
        blockers.append("shuffled-label null comparison failed")
    if not inputs.random_site_null_available:
        blockers.append("random-site null unavailable for this site-layer cell")
        caps.append("missing random-site null caps only this site-layer cell")
    elif not inputs.random_site_null_passed:
        blockers.append("random-site null comparison failed")
    if not inputs.matched_control_passed:
        blockers.append("matched-control comparison failed")
    if not inputs.primary_fdr_passed or not inputs.primary_ci_passed:
        blockers.append("primary probe metric failed corrected statistical gate")
    if not inputs.specificity_fdr_passed or not inputs.specificity_ci_passed:
        blockers.append("target-vs-decoy specificity gate failed")

    if blockers:
        return _result(INSUFFICIENT_EVIDENCE, blockers, caps, inputs)

    if not inputs.restoration_alignment_valid:
        blockers.append("restoration patching alignment metadata is invalid")
        return _result(CONTROLLED_PROBE_SIGNAL, blockers, caps, inputs)
    if not inputs.patching_valid or not inputs.mediation_valid:
        blockers.append("valid causal patching/restoration and mediation metrics are required")
        return _result(CONTROLLED_PROBE_SIGNAL, blockers, caps, inputs)
    if not inputs.task_aligned_pooling:
        blockers.append("candidate_mechanism_evidence requires task-aligned feature pooling")
        return _result(CONTROLLED_PROBE_SIGNAL, blockers, caps, inputs)
    if not inputs.canonical_site:
        blockers.append("candidate_mechanism_evidence requires a canonical Tier-1 preset site")
        return _result(CONTROLLED_PROBE_SIGNAL, blockers, caps, inputs)
    if not inputs.patching_fdr_passed or not inputs.full_layer_patching_fdr_passed or not inputs.mediation_fdr_passed:
        blockers.append("restoration/mediation metrics failed corrected statistical gate")
        return _result(CONTROLLED_PROBE_SIGNAL, blockers, caps, inputs)

    return _result(CANDIDATE_MECHANISM_EVIDENCE, blockers, caps, inputs)


def overall_status(cell_results: list[CellGateResult]) -> str:
    if not cell_results:
        return INSUFFICIENT_EVIDENCE
    rank = {status: index for index, status in enumerate(MECHANISM_PROBE_STATUS_LADDER)}
    return max((result.status for result in cell_results), key=lambda status: rank[status])


def _result(status: str, blockers: list[str], caps: list[str], inputs: CellGateInputs) -> CellGateResult:
    exploratory_signal = status == EXPLORATORY_PROBE_SIGNAL
    controlled_probe_gate_passed = status in {CONTROLLED_PROBE_SIGNAL, CANDIDATE_MECHANISM_EVIDENCE}
    candidate_mechanism_gate_passed = status == CANDIDATE_MECHANISM_EVIDENCE
    claim_gate_passed = candidate_mechanism_gate_passed
    if candidate_mechanism_gate_passed:
        status_kind = "candidate_mechanism_claim"
    elif controlled_probe_gate_passed:
        status_kind = "controlled_probe_claim"
    elif exploratory_signal:
        status_kind = "exploratory_signal"
    else:
        status_kind = "insufficient_evidence"
    return CellGateResult(
        status=status,
        blockers=tuple(dict.fromkeys(blockers)),
        caps=tuple(dict.fromkeys(caps)),
        exploratory_signal=exploratory_signal,
        controlled_probe_gate_passed=controlled_probe_gate_passed,
        candidate_mechanism_gate_passed=candidate_mechanism_gate_passed,
        highest_status=status,
        claim_gate_passed=claim_gate_passed,
        status_kind=status_kind,
        inputs=inputs.__dict__,
    )
