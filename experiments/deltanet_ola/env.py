from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from experiments.deltanet_ola.fla_paths import fla_root


def _git_value(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _submodule_commit() -> str:
    root = fla_root()
    if not root.exists():
        return "missing"
    file_value = _submodule_commit_from_files(root)
    if file_value != "unknown":
        return file_value
    value = _git_value(["rev-parse", "HEAD"], root)
    if value != "unknown":
        return value
    parent = root.parents[1]
    status = _git_value(["submodule", "status", "3rdparty/flash-linear-attention"], parent)
    return status.split()[0].lstrip("-+") if status != "unknown" and status.split() else "unknown"


def _submodule_commit_from_files(root: Path) -> str:
    git_pointer = root / ".git"
    if not git_pointer.exists() or git_pointer.is_dir():
        return "unknown"
    try:
        pointer_text = git_pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    prefix = "gitdir:"
    if not pointer_text.startswith(prefix):
        return "unknown"
    git_dir = (root / pointer_text[len(prefix) :].strip()).resolve()
    head_path = git_dir / "HEAD"
    try:
        head_text = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    if not head_text.startswith("ref:"):
        return head_text
    ref_path = git_dir / head_text.removeprefix("ref:").strip()
    try:
        return ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        packed_refs = git_dir / "packed-refs"
        try:
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                sha, _, ref = line.partition(" ")
                if ref == head_text.removeprefix("ref:").strip():
                    return sha
        except OSError:
            return "unknown"
    return "unknown"


def collect_run_metadata(device: torch.device, dtype: str = "unknown") -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "project_commit": _git_value(["rev-parse", "HEAD"], project_root),
        "fla_commit": _submodule_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": ",".join(gpu_names),
        "device": str(device),
        "dtype": dtype,
    }
