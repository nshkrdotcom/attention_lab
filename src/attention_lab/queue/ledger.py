from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attention_lab.training.config import load_config

STAGES = {"SCREEN", "FULL"}
STATUSES = {"PENDING", "RUNNING", "PASSED", "FAILED", "KILLED"}
FAILURE_CLASSES = {
    "NAN",
    "FLAT_LOSS",
    "DEAD_GRAD",
    "COMPILE_ERROR",
    "OOM",
    "SLOW",
    "VERIFY_FAIL",
    "RUN_DIR_EXISTS",
    "UNKNOWN",
}
BASELINE_ROW_ID = "__baseline__"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def hash_config_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:12]


class QueueLedger:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")

    def close(self) -> None:
        self.conn.close()

    def initialize(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                config_path TEXT NOT NULL,
                config_name TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                attention_type TEXT,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                failure_class TEXT,
                enqueued_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                step_reached INTEGER,
                final_val_loss REAL,
                best_val_loss REAL,
                final_ppl REAL,
                median_tokens_per_sec REAL,
                peak_vram_allocated_mb REAL,
                hellaswag_acc REAL,
                ablation_logit_delta REAL,
                mechanism_active INTEGER,
                full_run_approved INTEGER NOT NULL DEFAULT 0,
                allow_overwrite_existing_run_dir INTEGER NOT NULL DEFAULT 0,
                notes TEXT
            )
            """
        )
        self._migrate_schema()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(runs)").fetchall()}
        migrations = {
            "full_run_approved": "ALTER TABLE runs ADD COLUMN full_run_approved INTEGER NOT NULL DEFAULT 0",
            "allow_overwrite_existing_run_dir": (
                "ALTER TABLE runs ADD COLUMN allow_overwrite_existing_run_dir INTEGER NOT NULL DEFAULT 0"
            ),
        }
        for column, statement in migrations.items():
            if column not in columns:
                self.conn.execute(statement)

    def enqueue_config(self, config_path: str | Path, config: dict[str, Any], content: bytes) -> str:
        run_id = hash_config_bytes(content)
        existing = self.get_run(run_id)
        if existing is not None:
            return run_id
        queue_config = config.get("queue", {})
        self._raise_on_run_dir_collision(
            str(config["run"]["out_dir"]),
            allow_terminal_reuse=bool(queue_config.get("allow_overwrite_existing_run_dir", False)),
        )
        self.conn.execute(
            """
            INSERT INTO runs (
                id, config_path, config_name, run_dir, attention_type, stage, status,
                enqueued_at, full_run_approved, allow_overwrite_existing_run_dir, notes
            ) VALUES (?, ?, ?, ?, ?, 'SCREEN', 'PENDING', ?, ?, ?, '')
            """,
            (
                run_id,
                str(config_path),
                Path(config_path).stem,
                str(config["run"]["out_dir"]),
                config["model"].get("attention_type", "standard"),
                utc_now(),
                int(queue_config.get("full_run_approved", False)),
                int(queue_config.get("allow_overwrite_existing_run_dir", False)),
            ),
        )
        self.conn.commit()
        return run_id

    def _raise_on_run_dir_collision(self, run_dir: str, *, allow_terminal_reuse: bool = False) -> None:
        rows = self.conn.execute(
            "SELECT id, config_name, status FROM runs WHERE run_dir = ? AND id != ?",
            (run_dir, BASELINE_ROW_ID),
        ).fetchall()
        for row in rows:
            if row["status"] in {"PENDING", "RUNNING", "PASSED"} or not allow_terminal_reuse:
                raise ValueError(
                    f"run.out_dir collision with {row['config_name']} ({row['status']}): {run_dir}"
                )

    def scan_inbox(self, inbox_dir: str | Path) -> dict[str, Any]:
        inbox_dir = Path(inbox_dir)
        inserted = 0
        skipped = 0
        errors = []
        for config_path in sorted(inbox_dir.glob("*.yaml")):
            try:
                content = config_path.read_bytes()
                config = load_config(config_path)
                run_id = hash_config_bytes(content)
                if self.get_run(run_id) is not None:
                    skipped += 1
                    continue
                self.enqueue_config(config_path, config, content)
                inserted += 1
            except Exception as exc:  # noqa: BLE001 - queue must not crash on a bad inbox config
                errors.append({"path": str(config_path), "error": str(exc)})
        return {"inserted": inserted, "skipped": skipped, "errors": errors}

    def get_run(self, run_id_or_name: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id_or_name,)).fetchone()
        if row is None:
            row = self.conn.execute(
                "SELECT * FROM runs WHERE config_name = ? ORDER BY enqueued_at DESC LIMIT 1",
                (run_id_or_name,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_runs(self, *, stage: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM runs WHERE id != ?"
        params: list[Any] = [BASELINE_ROW_ID]
        if stage is not None:
            query += " AND stage = ?"
            params.append(stage)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY enqueued_at ASC"
        return [dict(row) for row in self.conn.execute(query, params).fetchall()]

    def get_pending_screens(self) -> list[dict[str, Any]]:
        return self.list_runs(stage="SCREEN", status="PENDING")

    def get_pending_full_runs(self) -> list[dict[str, Any]]:
        return self.list_runs(stage="FULL", status="PENDING")

    def mark_started(self, run_id: str) -> None:
        self.conn.execute(
            "UPDATE runs SET status = 'RUNNING', started_at = ?, finished_at = NULL WHERE id = ?",
            (utc_now(), run_id),
        )
        self.conn.commit()

    def promote_to_full(self, run_id: str, *, notes: str | None = None) -> None:
        if notes is None:
            self.conn.execute("UPDATE runs SET stage = 'FULL', status = 'PENDING', failure_class = NULL WHERE id = ?", (run_id,))
        else:
            self.conn.execute(
                "UPDATE runs SET stage = 'FULL', status = 'PENDING', failure_class = NULL, notes = ? WHERE id = ?",
                (notes, run_id),
            )
        self.conn.commit()

    def mark_passed(
        self,
        run_id: str,
        *,
        step_reached: int | None = None,
        final_val_loss: float | None = None,
        best_val_loss: float | None = None,
        final_ppl: float | None = None,
        median_tokens_per_sec: float | None = None,
        peak_vram_allocated_mb: float | None = None,
        hellaswag_acc: float | None = None,
        mechanism_active: bool | None = None,
        notes: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE runs
            SET status = 'PASSED',
                failure_class = NULL,
                finished_at = ?,
                step_reached = COALESCE(?, step_reached),
                final_val_loss = COALESCE(?, final_val_loss),
                best_val_loss = COALESCE(?, best_val_loss),
                final_ppl = COALESCE(?, final_ppl),
                median_tokens_per_sec = COALESCE(?, median_tokens_per_sec),
                peak_vram_allocated_mb = COALESCE(?, peak_vram_allocated_mb),
                hellaswag_acc = COALESCE(?, hellaswag_acc),
                mechanism_active = COALESCE(?, mechanism_active),
                notes = COALESCE(?, notes)
            WHERE id = ?
            """,
            (
                utc_now(),
                step_reached,
                final_val_loss,
                best_val_loss,
                final_ppl,
                median_tokens_per_sec,
                peak_vram_allocated_mb,
                hellaswag_acc,
                None if mechanism_active is None else int(mechanism_active),
                notes,
                run_id,
            ),
        )
        self.conn.commit()

    def mark_failed(
        self,
        run_id: str,
        *,
        failure_class: str = "UNKNOWN",
        killed: bool = False,
        step_reached: int | None = None,
        mechanism_active: bool | None = None,
        notes: str | None = None,
    ) -> None:
        if failure_class not in FAILURE_CLASSES:
            failure_class = "UNKNOWN"
        status = "KILLED" if killed else "FAILED"
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?,
                failure_class = ?,
                finished_at = ?,
                step_reached = COALESCE(?, step_reached),
                mechanism_active = COALESCE(?, mechanism_active),
                notes = COALESCE(?, notes)
            WHERE id = ?
            """,
            (
                status,
                failure_class,
                utc_now(),
                step_reached,
                None if mechanism_active is None else int(mechanism_active),
                notes,
                run_id,
            ),
        )
        self.conn.commit()

    def update_notes(self, run_id_or_name: str, notes: str) -> None:
        row = self.get_run(run_id_or_name)
        if row is None:
            raise KeyError(f"Unknown queue run: {run_id_or_name}")
        self.conn.execute("UPDATE runs SET notes = ? WHERE id = ?", (notes, row["id"]))
        self.conn.commit()

    def update_config_path(self, run_id_or_name: str, config_path: str | Path) -> None:
        row = self.get_run(run_id_or_name)
        if row is None:
            raise KeyError(f"Unknown queue run: {run_id_or_name}")
        self.conn.execute("UPDATE runs SET config_path = ? WHERE id = ?", (str(config_path), row["id"]))
        self.conn.commit()

    def set_full_run_approved(self, run_id_or_name: str, approved: bool) -> None:
        row = self.get_run(run_id_or_name)
        if row is None:
            raise KeyError(f"Unknown queue run: {run_id_or_name}")
        self.conn.execute("UPDATE runs SET full_run_approved = ? WHERE id = ?", (int(approved), row["id"]))
        self.conn.commit()

    def append_notes(self, run_id_or_name: str, notes: str) -> None:
        row = self.get_run(run_id_or_name)
        if row is None:
            raise KeyError(f"Unknown queue run: {run_id_or_name}")
        current = row.get("notes") or ""
        combined = f"{current}\n{notes}".strip()
        self.update_notes(row["id"], combined)

    def requeue(self, run_id_or_name: str) -> None:
        row = self.get_run(run_id_or_name)
        if row is None:
            raise KeyError(f"Unknown queue run: {run_id_or_name}")
        if row["status"] not in {"FAILED", "KILLED"} or row["stage"] != "FULL":
            raise ValueError("Only FAILED or KILLED FULL runs can be requeued")
        self.conn.execute(
            "UPDATE runs SET status = 'PENDING', failure_class = NULL, started_at = NULL, finished_at = NULL WHERE id = ?",
            (row["id"],),
        )
        self.conn.commit()

    def reset_interrupted(self) -> int:
        cursor = self.conn.execute(
            "UPDATE runs SET status = 'PENDING' WHERE status = 'RUNNING' AND finished_at IS NULL"
        )
        self.conn.commit()
        return int(cursor.rowcount)

    def update_baseline_screen_tokens_per_sec(self, tokens_per_sec: float) -> None:
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO runs (
                id, config_path, config_name, run_dir, attention_type, stage, status,
                enqueued_at, finished_at, median_tokens_per_sec, notes
            ) VALUES (?, '', ?, '', 'standard', 'FULL', 'PASSED', ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                median_tokens_per_sec = excluded.median_tokens_per_sec,
                finished_at = excluded.finished_at,
                notes = excluded.notes
            """,
            (BASELINE_ROW_ID, BASELINE_ROW_ID, now, now, float(tokens_per_sec), "baseline_screen_tokens_per_sec"),
        )
        self.conn.commit()

    def get_baseline_screen_tokens_per_sec(self) -> float | None:
        row = self.conn.execute(
            "SELECT median_tokens_per_sec FROM runs WHERE id = ?",
            (BASELINE_ROW_ID,),
        ).fetchone()
        if row is None or row["median_tokens_per_sec"] is None:
            return None
        return float(row["median_tokens_per_sec"])
