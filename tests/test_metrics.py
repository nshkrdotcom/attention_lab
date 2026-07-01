from __future__ import annotations

import csv
import json
import math

from attention_lab.training.metrics import MetricsLogger


def test_metrics_logger_writes_jsonl_and_csv(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log(
        {
            "event": "train",
            "step": 1,
            "train_loss": 2.0,
            "tokens_per_sec": 10.0,
            "peak_vram_allocated_mb": 11.0,
            "peak_vram_reserved_mb": 22.0,
            "current_vram_allocated_mb": 7.0,
            "current_vram_reserved_mb": 14.0,
            "peak_vram_mb": 11.0,
        }
    )
    logger.log({"event": "val", "step": 1, "val_loss": 1.5, "val_perplexity": math.exp(1.5)})
    logger.close()

    jsonl_rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["event"] for row in jsonl_rows] == ["train", "val"]
    assert math.isclose(jsonl_rows[1]["val_perplexity"], math.exp(jsonl_rows[1]["val_loss"]))

    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert [row["event"] for row in csv_rows] == ["train", "val"]
    assert csv_rows[0]["peak_vram_allocated_mb"] == "11.0"
    assert csv_rows[0]["peak_vram_reserved_mb"] == "22.0"
