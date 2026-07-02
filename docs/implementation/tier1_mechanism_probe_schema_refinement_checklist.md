# Tier-1 Mechanism Probe Schema Refinement Checklist

Follow-up remediation for the Tier-1 mechanism-probe suite after review of the first remediation pass.

## Items

- [x] Split ambiguous claim-gate semantics into explicit machine fields:
  - `exploratory_signal`
  - `controlled_probe_gate_passed`
  - `candidate_mechanism_gate_passed`
  - `highest_status`
- [x] Retain `claim_gate_passed` only as a compatibility alias for `candidate_mechanism_gate_passed`.
- [x] Ensure `controlled_probe_signal` can pass the controlled-probe threshold without implying valid causal patching/restoration.
- [x] Update per-run overall fields:
  - `overall_exploratory_signal`
  - `overall_controlled_probe_gate_passed`
  - `overall_candidate_mechanism_gate_passed`
- [x] Document that run-level `overall_*` gate booleans are existential over evaluated cells, not all-cell or primary-cell pass indicators.
- [x] Document `highest_status` as the highest mechanism-probe claim ladder threshold reached for a cell.
- [x] Update summary rendering and artifact validation to require the explicit gate fields.
- [x] Enforce built-in Tier-1 deterministic regeneration during confirmatory suite execution, not only in validate-only generation commands.
- [x] Add a regression where a built-in task suite is tampered with and `content_sha256` is recomputed, then confirmatory execution still rejects it before checkpoint loading.
- [x] Document that `content_sha256` is an integrity check, not standalone provenance.
- [x] Document that future task generators need an equivalent regeneration validator before supporting confirmatory claims.
- [x] Document that the random-site null pool is the complete preset-declared Tier-1 null family, not an unrestricted hook sweep.
- [x] Reword checkpoint availability docs as local reconciliation facts for generated artifacts, not portable repository truth.
- [x] Regenerate committed Tier-1 suite artifacts with the refined schema.
- [x] Run targeted and full QC.
- [x] Commit and push the refinement.

## Scientific Boundary

This pass does not change the Tier-1 scientific result. The regenerated E003/E004 confirmatory artifacts should remain `insufficient_evidence` unless the actual probe/null/control/specificity/restoration metrics clear their gates.
