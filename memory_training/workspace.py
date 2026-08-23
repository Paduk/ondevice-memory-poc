"""Create the isolated NVMe workspace without copying canonical datasets."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_WORKSPACE_ROOT,
    TARGET_MODELS,
    WORKSPACE_DIRECTORIES,
)


def initialize_workspace(workspace: Path, data_root: Path) -> dict[str, object]:
    workspace = workspace.resolve()
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    workspace.mkdir(parents=True, exist_ok=True)
    for relative in WORKSPACE_DIRECTORIES:
        (workspace / relative).mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "palmclaw-memory-training-workspace-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(workspace),
        "canonical_data_root": str(data_root),
        "data_is_copied": False,
        "target_models": [
            {
                "key": model.key,
                "hf_id": model.hf_id,
                "ollama_tag": model.ollama_tag,
            }
            for model in TARGET_MODELS
        ],
    }
    manifest_path = workspace / "workspace.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    manifest = initialize_workspace(args.workspace, args.data_root)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
