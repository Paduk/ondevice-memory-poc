"""Import the repository-local PalmClaw Ubuntu evaluation runtime."""

from __future__ import annotations

import sys
from pathlib import Path


def enable_ubuntu_runtime() -> Path:
    source = Path(__file__).resolve().parents[1] / "ubuntu" / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"PalmClaw Ubuntu source not found: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return source
