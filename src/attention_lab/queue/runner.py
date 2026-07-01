from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.state_files import copy_to_active, finalize_config
from attention_lab.training.config import load_config


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], Path], CommandResult]


def build_full_pipeline(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    data_root: str | Path,
    manifest_path: str | Path,
) -> list[list[str]]:
    run_dir = str(run_dir)
    data_root = str(data_root)
    manifest_path = str(manifest_path)
    ckpt_last = f"{run_dir}/checkpoints/ckpt_last.pt"
    return [
        [
            "uv",
            "run",
            "scripts/verify_data.py",
            "--data_root",
            data_root,
            "--manifest",
            manifest_path,
            "--verify_hashes",
        ],
        ["uv", "run", "scripts/train.py", "--config", str(config_path), "--overwrite"],
        [
            "uv",
            "run",
            "scripts/verify_run.py",
            "--run_dir",
            run_dir,
            "--expect-complete-training",
            "--expect-sample",
            "--expect-data-manifest",
        ],
        ["uv", "run", "scripts/eval_loss.py", "--checkpoint", ckpt_last, "--data_root", data_root],
        [
            "uv",
            "run",
            "scripts/eval_generate.py",
            "--checkpoint",
            ckpt_last,
            "--prompt",
            "The history of mathematics",
        ],
        ["uv", "run", "scripts/eval_hellaswag.py", "--checkpoint", ckpt_last, "--max_examples", "100"],
        ["uv", "run", "scripts/summarize_run.py", "--run_dir", run_dir],
        [
            "uv",
            "run",
            "scripts/verify_run.py",
            "--run_dir",
            run_dir,
            "--expect-complete-training",
            "--expect-sample",
            "--expect-eval-loss",
            "--expect-hellaswag",
            "--expect-data-manifest",
        ],
    ]


def classify_failure(returncode: int, stderr: str, *, verify_step: bool = False) -> str:
    text = stderr.lower()
    if "out of memory" in text or "cuda oom" in text:
        return "OOM"
    if "nan" in text or " inf" in text or "infinite" in text:
        return "NAN"
    if verify_step:
        return "VERIFY_FAIL"
    if returncode != 0:
        return "UNKNOWN"
    return "UNKNOWN"


def capture_git_state(run_dir: str | Path, *, repo_root: str | Path = ".") -> str | None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    diff = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    commit_text = commit.stdout.strip() if commit.returncode == 0 else f"UNKNOWN ({commit.stderr.strip()})"
    diff_text = diff.stdout if diff.returncode == 0 else f"git diff failed: {diff.stderr}"
    warning = "WARNING: source tree was dirty at full-run start" if diff_text.strip() else None
    state = [
        f"commit: {commit_text}",
        f"dirty: {bool(diff_text.strip())}",
    ]
    if warning:
        state.append(warning)
    state.extend(["diff:", diff_text])
    (run_dir / "git_state.txt").write_text("\n".join(state), encoding="utf-8")
    return warning


def default_command_runner(cmd: list[str], log_path: Path) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        for line in process.stdout:
            output_lines.append(line)
            log.write(line)
            print(line, end="")
            sys.stdout.flush()
    returncode = process.wait()
    output = "".join(output_lines)
    return CommandResult(returncode=returncode, stdout=output, stderr=output)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_path_for_data_root(data_root: str | Path) -> Path:
    return Path(data_root) / "manifest.json"


def _is_verify_step(cmd: Sequence[str]) -> bool:
    return "scripts/verify_run.py" in cmd


def existing_run_artifacts(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    artifacts = [
        run_dir / "checkpoints",
        run_dir / "metrics.jsonl",
        run_dir / "evals" / "run_summary.json",
        run_dir / "data_manifest.json",
    ]
    return [str(path) for path in artifacts if path.exists()]


REQUIRED_SUMMARY_FIELDS = {
    "max_step",
    "final_val_loss",
    "best_val_loss",
    "final_val_perplexity",
    "median_tokens_per_sec",
}


def validate_full_run_artifacts(run_dir: str | Path) -> tuple[dict, dict, str | None]:
    run_dir = Path(run_dir)
    required_files = [
        run_dir / "evals" / "run_summary.json",
        run_dir / "evals" / "val_loss.json",
        run_dir / "evals" / "hellaswag.json",
        run_dir / "checkpoints" / "ckpt_last.pt",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        return {}, {}, f"missing required full-run artifacts: {', '.join(missing)}"

    summary = _read_json(run_dir / "evals" / "run_summary.json")
    missing_summary = sorted(field for field in REQUIRED_SUMMARY_FIELDS if summary.get(field) is None)
    if missing_summary:
        return summary, {}, f"run_summary.json missing required fields: {missing_summary}"

    hellaswag = _read_json(run_dir / "evals" / "hellaswag.json")
    if hellaswag.get("accuracy_norm") is None and hellaswag.get("accuracy") is None:
        return summary, hellaswag, "hellaswag.json missing accuracy_norm/accuracy"
    return summary, hellaswag, None


def run_full(
    row: dict,
    ledger: QueueLedger,
    *,
    command_runner: CommandRunner = default_command_runner,
    repo_root: str | Path = ".",
) -> dict:
    run_id = row["id"]
    config_path = Path(row["config_path"])
    config = load_config(config_path)
    run_dir = Path(config["run"]["out_dir"])
    data_root = Path(config["data"]["data_root"])
    manifest_path = _manifest_path_for_data_root(data_root)
    log_path = run_dir / "queue_runner.log"
    queue_config = config.get("queue", {})

    if not bool(row.get("allow_overwrite_existing_run_dir")) and not queue_config.get(
        "allow_overwrite_existing_run_dir", False
    ):
        artifacts = existing_run_artifacts(run_dir)
        if artifacts:
            notes = f"existing run_dir artifacts; refusing overwrite: {', '.join(artifacts)}"
            ledger.mark_failed(run_id, failure_class="RUN_DIR_EXISTS", notes=notes)
            finalize_config(config_path, "failed")
            return {"ok": False, "failure_class": "RUN_DIR_EXISTS", "notes": notes}

    ledger.mark_started(run_id)
    copy_to_active(config_path)
    warning = capture_git_state(run_dir, repo_root=repo_root)
    if warning is not None:
        ledger.append_notes(run_id, warning)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(warning + "\n")

    for cmd in build_full_pipeline(
        config_path=config_path,
        run_dir=run_dir,
        data_root=data_root,
        manifest_path=manifest_path,
    ):
        result = command_runner(cmd, log_path)
        if result.returncode != 0:
            failure_class = classify_failure(result.returncode, result.stderr, verify_step=_is_verify_step(cmd))
            ledger.mark_failed(run_id, failure_class=failure_class, notes=f"failed command: {' '.join(cmd)}")
            finalize_config(config_path, "failed")
            return {"ok": False, "failure_class": failure_class, "command": cmd}

    summary, hellaswag, artifact_error = validate_full_run_artifacts(run_dir)
    if artifact_error is not None:
        ledger.mark_failed(run_id, failure_class="VERIFY_FAIL", notes=artifact_error)
        finalize_config(config_path, "failed")
        return {"ok": False, "failure_class": "VERIFY_FAIL", "notes": artifact_error}

    ledger.mark_passed(
        run_id,
        step_reached=summary.get("max_step"),
        final_val_loss=summary.get("final_val_loss"),
        best_val_loss=summary.get("best_val_loss"),
        final_ppl=summary.get("final_val_perplexity"),
        median_tokens_per_sec=summary.get("median_tokens_per_sec"),
        peak_vram_allocated_mb=summary.get("peak_vram_allocated_mb") or summary.get("peak_vram_mb"),
        hellaswag_acc=hellaswag.get("accuracy_norm") if hellaswag.get("accuracy_norm") is not None else hellaswag.get("accuracy"),
    )
    finalize_config(config_path, "done")
    return {"ok": True, "run_id": run_id}
