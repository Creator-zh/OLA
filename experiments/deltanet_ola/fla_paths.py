from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def fla_root() -> Path:
    return project_root() / "3rdparty" / "flash-linear-attention"


def ensure_fla_on_path() -> Path:
    root = fla_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root

