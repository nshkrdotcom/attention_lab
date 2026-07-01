#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


EXPERIMENT = "E003_qkv_architecture_gauntlet"
REPORT_DIR = Path("reports") / "experiments" / EXPERIMENT
REPORT_JSON = REPORT_DIR / "gauntlet_report.json"
REPORT_MD = REPORT_DIR / "gauntlet_report.md"


def main() -> None:
    if not REPORT_JSON.exists():
        raise SystemExit(f"Missing gauntlet report: {REPORT_JSON}")
    payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    lines = [
        f"# {EXPERIMENT} Gauntlet Summary",
        "",
        f"Created: {payload.get('created_at')}",
        f"Policy: `{payload.get('policy_path')}`",
        f"Control: `{payload.get('control_run_name')}`",
        "",
        "| Candidate | Rung | Decision | Final val loss | Mechanism active | Next action |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for decision in payload.get("decisions", []):
        lines.append(
            "| {candidate} | {rung} | {machine_decision} | {loss} | {mechanism} | {next_action} |".format(
                candidate=decision.get("candidate") or "",
                rung=decision.get("rung") or "",
                machine_decision=decision.get("machine_decision") or "",
                loss=decision.get("final_val_loss") if decision.get("final_val_loss") is not None else "",
                mechanism=decision.get("mechanism_active"),
                next_action=decision.get("next_action") or "",
            )
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {REPORT_MD}")


if __name__ == "__main__":
    main()
