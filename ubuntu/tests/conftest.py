from __future__ import annotations

from pathlib import Path

import pytest

from palmclaw_ubuntu.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    package_root = Path(__file__).resolve().parents[1] / "src" / "palmclaw_ubuntu"
    return Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "palmclaw.db",
        workspace_root=tmp_path / "data" / "workspaces",
        shared_workspace_root=tmp_path / "data" / "shared",
        builtin_skills_root=package_root / "builtin_skills",
        workspace_skills_root=tmp_path / "data" / "skills",
        backend="fake",
        agent_model=None,
        memory_model=None,
        model_timeout_seconds=2,
        tool_timeout_seconds=2,
        max_tool_rounds=4,
        max_history_messages=40,
        max_tool_result_chars=20_000,
        memory_trigger_messages=2,
        max_file_bytes=100_000,
    )
