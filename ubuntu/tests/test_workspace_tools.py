from __future__ import annotations

from pathlib import Path

import pytest

from palmclaw_ubuntu.models import ToolCall
from palmclaw_ubuntu.tools import (
    FileReadTool,
    FileWriteTool,
    ToolExecutionContext,
    ToolRegistry,
)
from palmclaw_ubuntu.workspace import WorkspaceResolver


def _registry(tmp_path: Path) -> tuple[ToolRegistry, WorkspaceResolver]:
    resolver = WorkspaceResolver(
        tmp_path / "workspaces",
        tmp_path / "shared",
    )
    registry = ToolRegistry(
        [
            FileReadTool(
                resolver,
                timeout_seconds=1,
                max_file_bytes=1_000,
            ),
            FileWriteTool(
                resolver,
                timeout_seconds=1,
                max_file_bytes=1_000,
            ),
        ]
    )
    return registry, resolver


def test_file_write_then_read(tmp_path):
    registry, _ = _registry(tmp_path)
    context = ToolExecutionContext(session_id="session-1")
    write_result, _ = registry.execute(
        ToolCall(
            id="write-1",
            name="file_write",
            arguments={
                "path": "notes/hello.txt",
                "content": "hello",
                "overwrite": False,
            },
        ),
        context,
    )
    read_result, _ = registry.execute(
        ToolCall(
            id="read-1",
            name="file_read",
            arguments={"path": "notes/hello.txt"},
        ),
        context,
    )
    assert not write_result.is_error
    assert write_result.tool_call_id == "write-1"
    assert not read_result.is_error
    assert '"content": "hello"' in read_result.content


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "/tmp/outside.txt",
        "session://../../outside.txt",
        "shared://../outside.txt",
    ],
)
def test_workspace_escape_is_blocked(tmp_path, path):
    registry, _ = _registry(tmp_path)
    result, _ = registry.execute(
        ToolCall(
            id="write-escape",
            name="file_write",
            arguments={
                "path": path,
                "content": "no",
                "overwrite": False,
            },
        ),
        ToolExecutionContext(session_id="session-1"),
    )
    assert result.is_error


def test_invalid_arguments_are_rejected_before_execution(tmp_path):
    registry, resolver = _registry(tmp_path)
    result, _ = registry.execute(
        ToolCall(
            id="invalid",
            name="file_write",
            arguments={"path": "missing-content.txt"},
        ),
        ToolExecutionContext(session_id="session-1"),
    )
    assert result.is_error
    assert result.metadata["error_code"] == "invalid_arguments"
    assert not (resolver.session_root("session-1") / "missing-content.txt").exists()


def test_symlink_escape_is_blocked(tmp_path):
    registry, resolver = _registry(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = resolver.session_root("session-1") / "outside-link"
    link.symlink_to(outside, target_is_directory=True)

    result, _ = registry.execute(
        ToolCall(
            id="read-symlink",
            name="file_read",
            arguments={"path": "outside-link/secret.txt"},
        ),
        ToolExecutionContext(session_id="session-1"),
    )

    assert result.is_error
    assert result.metadata["error_code"] == "permission_denied"


def test_hard_link_is_blocked(tmp_path):
    registry, resolver = _registry(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    hard_link = resolver.session_root("session-1") / "linked.txt"
    hard_link.hardlink_to(outside)

    result, _ = registry.execute(
        ToolCall(
            id="read-hard-link",
            name="file_read",
            arguments={"path": "linked.txt"},
        ),
        ToolExecutionContext(session_id="session-1"),
    )

    assert result.is_error
    assert result.metadata["error_code"] == "permission_denied"


def test_one_session_cannot_reach_another_session_workspace(tmp_path):
    registry, resolver = _registry(tmp_path)
    second_root = resolver.session_root("session-2")
    (second_root / "private.txt").write_text("private", encoding="utf-8")

    result, _ = registry.execute(
        ToolCall(
            id="cross-session",
            name="file_read",
            arguments={"path": "../session-2/private.txt"},
        ),
        ToolExecutionContext(session_id="session-1"),
    )

    assert result.is_error
    assert result.metadata["error_code"] == "permission_denied"
