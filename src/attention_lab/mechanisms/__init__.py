"""Native mechanism inspection substrate for Attention Lab."""

from attention_lab.mechanisms.cache import ActivationCache, ActivationRecord
from attention_lab.mechanisms.capture import CaptureResult, capture_activations
from attention_lab.mechanisms.diagnostics import normalize_diagnostic_row, normalize_diagnostics_jsonl
from attention_lab.mechanisms.hook_sites import get_hook_site_specs
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec, run_with_interventions
from attention_lab.mechanisms.specs import HookSiteSpec

__all__ = [
    "ActivationCache",
    "ActivationRecord",
    "CaptureResult",
    "HookSiteSpec",
    "InterventionKind",
    "InterventionSpec",
    "capture_activations",
    "get_hook_site_specs",
    "normalize_diagnostic_row",
    "normalize_diagnostics_jsonl",
    "run_with_interventions",
]
