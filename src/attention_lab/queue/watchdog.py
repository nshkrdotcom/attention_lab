from __future__ import annotations

import signal
import time
import os
from pathlib import Path
from typing import Callable

from attention_lab.queue.discipline import default_hypothesis_path, validate_hypothesis_doc
from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.paths import default_pid_path
from attention_lab.queue.runner import run_full
from attention_lab.queue.screener import run_screen
from attention_lab.training.config import load_config

ScreenRunner = Callable[[dict, QueueLedger], dict]
FullRunner = Callable[[dict, QueueLedger], dict]


class Watchdog:
    def __init__(
        self,
        ledger: QueueLedger,
        *,
        inbox_dir: str | Path = "queue/inbox",
        pid_path: str | Path = "data/queue.pid",
        screen_runner: ScreenRunner | None = None,
        full_runner: FullRunner | None = None,
        sleep_seconds: int = 60,
    ):
        self.ledger = ledger
        self.inbox_dir = Path(inbox_dir)
        self.pid_path = Path(pid_path)
        self.screen_runner = screen_runner or (lambda row, ledger: run_screen(row, ledger))
        self.full_runner = full_runner or (lambda row, ledger: run_full(row, ledger))
        self.sleep_seconds = sleep_seconds
        self.should_stop = False

    def install_signal_handlers(self) -> None:
        def _handle_sigterm(signum, frame):  # noqa: ARG001
            self.should_stop = True

        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)

    def write_pid(self) -> None:
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    def run_once(self) -> dict:
        scan_result = self.ledger.scan_inbox(self.inbox_dir)
        pending_screens = self.ledger.get_pending_screens()
        if pending_screens:
            row = pending_screens[0]
            return {"action": "screen", "scan": scan_result, "result": self.screen_runner(row, self.ledger)}

        for row in self.ledger.get_pending_full_runs():
            ready, note = self._full_run_ready(row)
            if ready:
                if note:
                    self.ledger.update_notes(row["id"], note)
                return {"action": "full", "scan": scan_result, "result": self.full_runner(row, self.ledger)}
            if note:
                self.ledger.update_notes(row["id"], note)
        return {"action": "sleep", "scan": scan_result}

    def run_forever(self) -> None:
        self.install_signal_handlers()
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_pid()
        self.ledger.reset_interrupted()
        try:
            while not self.should_stop:
                result = self.run_once()
                if result["action"] == "sleep":
                    time.sleep(self.sleep_seconds)
        finally:
            if self.pid_path.exists():
                self.pid_path.unlink()

    def _full_run_ready(self, row: dict) -> tuple[bool, str | None]:
        config = load_config(row["config_path"])
        queue_config = config.get("queue", {})
        notes = []

        if not bool(row.get("full_run_approved")):
            return False, "waiting for queue.full_run_approved: true"

        requires_run = queue_config.get("requires_run")
        attention_type = config["model"].get("attention_type", "standard")
        skip_control_check = queue_config.get("skip_control_check", False)
        if attention_type != "standard" and not requires_run and not skip_control_check:
            return False, "waiting for queue.requires_run or explicit queue.skip_control_check"
        if attention_type != "standard" and skip_control_check:
            notes.append("WARNING: control dependency explicitly skipped")

        if requires_run:
            required = self.ledger.get_run(requires_run)
            if required is None or required.get("status") != "PASSED":
                return False, f"waiting for required run to pass: {requires_run}"

        if queue_config.get("skip_hypothesis_check", False):
            notes.append("WARNING: hypothesis check explicitly skipped")
            return True, "\n".join(notes)

        hypothesis_path = default_hypothesis_path(row["config_path"], config)
        hypothesis = validate_hypothesis_doc(hypothesis_path)
        if not hypothesis.ok:
            return False, f"missing or incomplete hypothesis doc: {hypothesis.path}"
        return True, "\n".join(notes) if notes else None


def main() -> None:
    ledger = QueueLedger("data/queue.db")
    ledger.initialize()
    Watchdog(ledger, pid_path=default_pid_path()).run_forever()
