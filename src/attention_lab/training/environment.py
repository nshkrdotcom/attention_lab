import platform
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return f"{command[0]}: not found"
    return result.stdout.strip()


def git_commit_text(cwd: str | Path) -> str:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    status = _run(["git", "status", "--short"], cwd=cwd)
    return f"commit: {commit}\n\nstatus:\n{status}\n"


def environment_text() -> str:
    lines = [
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        f"uv: {_run(['uv', '--version'])}",
    ]
    try:
        import torch

        lines.extend(
            [
                f"torch: {torch.__version__}",
                f"cuda_available: {torch.cuda.is_available()}",
                f"cuda_version: {torch.version.cuda}",
            ]
        )
        if torch.cuda.is_available():
            lines.append(f"cuda_device: {torch.cuda.get_device_name(0)}")
            lines.append(f"bf16_supported: {torch.cuda.is_bf16_supported()}")
    except Exception as exc:
        lines.append(f"torch_import_error: {exc}")
    lines.append("\nnvidia-smi:\n" + _run(["nvidia-smi"]))
    return "\n".join(lines) + "\n"

