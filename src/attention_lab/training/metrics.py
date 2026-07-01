import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "time",
    "event",
    "step",
    "train_loss",
    "val_loss",
    "val_perplexity",
    "lr",
    "grad_norm",
    "dt_ms",
    "tokens_per_sec",
    "peak_vram_mb",
    "peak_vram_allocated_mb",
    "peak_vram_reserved_mb",
    "current_vram_allocated_mb",
    "current_vram_reserved_mb",
    "nvidia_smi_memory_mb",
    "checkpoint",
    "resume_from",
]


class MetricsLogger:
    def __init__(self, out_dir: str | Path, append: bool = False):
        self.out_dir = Path(out_dir)
        self.jsonl_path = self.out_dir / "metrics.jsonl"
        self.csv_path = self.out_dir / "metrics.csv"
        mode = "a" if append else "w"
        self.jsonl_file = self.jsonl_path.open(mode, encoding="utf-8")
        self.csv_file = self.csv_path.open(mode, newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not append or self.csv_path.stat().st_size == 0:
            self.csv_writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
        self.jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.jsonl_file.flush()
        self.csv_writer.writerow({key: record.get(key) for key in FIELDNAMES})
        self.csv_file.flush()

    def close(self) -> None:
        self.jsonl_file.close()
        self.csv_file.close()
