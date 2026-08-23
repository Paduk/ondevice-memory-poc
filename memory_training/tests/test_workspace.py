import json
from pathlib import Path

from memory_training.config import WORKSPACE_DIRECTORIES
from memory_training.workspace import initialize_workspace


def test_initialize_workspace_keeps_dataset_external(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    workspace = tmp_path / "workspace"

    manifest = initialize_workspace(workspace, data_root)

    assert manifest["data_is_copied"] is False
    assert (
        json.loads((workspace / "workspace.json").read_text())["data_is_copied"]
        is False
    )
    for relative in WORKSPACE_DIRECTORIES:
        assert (workspace / relative).is_dir()
