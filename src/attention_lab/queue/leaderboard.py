from __future__ import annotations

from datetime import datetime
from typing import Any


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "---"
    return f"{float(value):.{digits}f}"


def _fmt_speed(value: Any) -> str:
    if value is None or value == "":
        return "---"
    value = float(value)
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def _fmt_vram(value: Any) -> str:
    if value is None or value == "":
        return "---"
    return f"{float(value) / 1024:.1f}G"


def _fmt_hot(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return "1" if int(value) else "0"


def _fmt_bool(value: Any) -> str:
    return "Y" if bool(value) else "N"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "~"


def render_leaderboard(
    rows: list[dict[str, Any]],
    *,
    now: str | None = None,
    min_stage: str | None = None,
    sort: str | None = None,
) -> str:
    rows = _filter_rows(rows, min_stage=min_stage)
    rows = _sort_rows(rows, sort=sort)
    now = now or datetime.now().strftime("%Y-%m-%d %H:%M")
    running = next((row for row in rows if row.get("status") == "RUNNING"), None)
    running_text = "none"
    if running is not None:
        step = running.get("step_reached") or "---"
        running_text = f"{running.get('config_name')} step {step}"

    lines = [
        f"QUEUE STATUS  {now}  |  running: {running_text}",
        "=" * 96,
        "RUNNING / RECENT",
        "  run_name                             attn_type       stage  status    loss    ppl    tok/s  vram  hs  appr notes",
    ]
    recent = [row for row in rows if row.get("status") in {"RUNNING", "PASSED"}]
    if not recent:
        lines.append("  ---")
    for row in recent[-12:]:
        lines.append(_format_row(row))

    pending = [row for row in rows if row.get("status") == "PENDING"]
    lines.extend(["", f"PENDING ({len(pending)} in queue)"])
    if not pending:
        lines.append("  ---")
    for row in pending[:20]:
        lines.append(
            f"  {_truncate(row.get('config_name', ''), 36):36} "
            f"{row.get('attention_type') or '---':14} {row.get('stage'):6} "
            f"PENDING approved={_fmt_bool(row.get('full_run_approved'))}"
        )

    failed = [row for row in rows if row.get("status") in {"FAILED", "KILLED"}]
    lines.extend(["", "KILLED / FAILED (recent)"])
    if not failed:
        lines.append("  ---")
    for row in failed[-12:]:
        lines.append(_format_row(row, include_failure=True))
    lines.extend(["", "=" * 96, "hs = mechanism_active  |  to add: attn-queue add configs/experiments/<ID>/..."])
    return "\n".join(lines) + "\n"


def _filter_rows(rows: list[dict[str, Any]], *, min_stage: str | None) -> list[dict[str, Any]]:
    if min_stage is None or min_stage == "SCREEN":
        return list(rows)
    if min_stage == "FULL":
        return [row for row in rows if row.get("stage") == "FULL"]
    return list(rows)


def _sort_rows(rows: list[dict[str, Any]], *, sort: str | None) -> list[dict[str, Any]]:
    if sort is None:
        return list(rows)
    if sort == "loss":
        return sorted(rows, key=lambda row: _number_sort_key(row.get("final_val_loss")))
    if sort == "ppl":
        return sorted(rows, key=lambda row: _number_sort_key(row.get("final_ppl")))
    if sort == "speed":
        return sorted(rows, key=lambda row: _number_sort_key(row.get("median_tokens_per_sec"), reverse=True))
    return list(rows)


def _number_sort_key(value: Any, *, reverse: bool = False) -> tuple[int, float]:
    if value is None or value == "":
        return (1, 0.0)
    number = float(value)
    return (0, -number if reverse else number)


def _format_row(row: dict[str, Any], *, include_failure: bool = False) -> str:
    notes = row.get("failure_class") if include_failure and row.get("failure_class") else row.get("notes") or ""
    return (
        f"  {_truncate(row.get('config_name', ''), 36):36} "
        f"{_truncate(row.get('attention_type') or '---', 14):14} "
        f"{row.get('stage') or '---':6} "
        f"{row.get('status') or '---':8} "
        f"{_fmt_float(row.get('final_val_loss')):7} "
        f"{_fmt_float(row.get('final_ppl'), digits=1):6} "
        f"{_fmt_speed(row.get('median_tokens_per_sec')):6} "
        f"{_fmt_vram(row.get('peak_vram_allocated_mb')):5} "
        f"{_fmt_hot(row.get('mechanism_active')):2} "
        f"{_fmt_bool(row.get('full_run_approved')):4} "
        f"{_truncate(notes, 40)}"
    )
