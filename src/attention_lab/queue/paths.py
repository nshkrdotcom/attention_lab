from __future__ import annotations

from pathlib import Path


def ensure_queue_dirs(root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    paths = {
        "queue": root / "queue",
        "inbox": root / "queue" / "inbox",
        "active": root / "queue" / "active",
        "full_pending": root / "queue" / "full_pending",
        "done": root / "queue" / "done",
        "failed": root / "queue" / "failed",
        "data": root / "data",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def default_db_path(root: str | Path = ".") -> Path:
    return Path(root) / "data" / "queue.db"


def default_pid_path(root: str | Path = ".") -> Path:
    return Path(root) / "data" / "queue.pid"
