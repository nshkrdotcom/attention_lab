from __future__ import annotations

import shutil
from pathlib import Path


QUEUE_CONFIG_STATE_DIRS = {"inbox", "full_pending"}


def queue_root_from_inbox_config(config_path: str | Path) -> Path | None:
    path = Path(config_path)
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index] == "queue" and parts[index + 1] in QUEUE_CONFIG_STATE_DIRS:
            prefix = parts[:index]
            return Path(*prefix) if prefix else Path(".")
    return None


def copy_to_active(config_path: str | Path) -> Path | None:
    root = queue_root_from_inbox_config(config_path)
    if root is None:
        return None
    config_path = Path(config_path)
    active_path = root / "queue" / "active" / config_path.name
    active_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, active_path)
    return active_path


def finalize_config(config_path: str | Path, state: str) -> Path | None:
    if state not in {"done", "failed"}:
        raise ValueError("state must be done or failed")
    root = queue_root_from_inbox_config(config_path)
    if root is None:
        return None
    config_path = Path(config_path)
    target = root / "queue" / state / config_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        shutil.copy2(config_path, target)
        if "queue" in config_path.parts and any(state in config_path.parts for state in QUEUE_CONFIG_STATE_DIRS):
            config_path.unlink()
    active_path = root / "queue" / "active" / config_path.name
    active_path.unlink(missing_ok=True)
    return target


def clear_active(config_path: str | Path) -> None:
    root = queue_root_from_inbox_config(config_path)
    if root is None:
        return
    active_path = root / "queue" / "active" / Path(config_path).name
    active_path.unlink(missing_ok=True)


def move_to_full_pending(config_path: str | Path) -> Path | None:
    root = queue_root_from_inbox_config(config_path)
    if root is None:
        clear_active(config_path)
        return None
    config_path = Path(config_path)
    target = root / "queue" / "full_pending" / config_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        shutil.copy2(config_path, target)
        config_path.unlink()
    clear_active(config_path)
    return target
