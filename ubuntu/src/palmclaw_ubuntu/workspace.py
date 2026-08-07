from __future__ import annotations

import re
from pathlib import Path

_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class WorkspaceResolver:
    def __init__(
        self,
        workspace_root: Path,
        shared_workspace_root: Path,
    ):
        self.workspace_root = workspace_root.resolve()
        self.shared_workspace_root = shared_workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.shared_workspace_root.mkdir(parents=True, exist_ok=True)

    def session_root(self, session_id: str) -> Path:
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("Invalid session ID for workspace")
        root = (self.workspace_root / session_id).resolve()
        self._require_under(root, self.workspace_root)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def resolve(
        self,
        session_id: str,
        raw_path: str,
        *,
        must_exist: bool = False,
    ) -> Path:
        value = raw_path.strip()
        if not value:
            value = "."
        if value.startswith("session://"):
            root = self.session_root(session_id)
            relative = value.removeprefix("session://").lstrip("/\\")
        elif value.startswith("shared://"):
            root = self.shared_workspace_root
            relative = value.removeprefix("shared://").lstrip("/\\")
        else:
            candidate_input = Path(value)
            if candidate_input.is_absolute():
                raise PermissionError("Absolute paths are not allowed")
            root = self.session_root(session_id)
            relative = value

        candidate = (root / relative).resolve(strict=False)
        self._require_under(candidate, root)
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"Path does not exist: {raw_path}")
        return candidate

    @staticmethod
    def _require_under(candidate: Path, root: Path) -> None:
        if candidate != root and not candidate.is_relative_to(root):
            raise PermissionError("Path escapes the allowed workspace")

    def display_path(self, session_id: str, path: Path) -> str:
        resolved = path.resolve(strict=False)
        session_root = self.session_root(session_id)
        if resolved == session_root:
            return "session://"
        if resolved.is_relative_to(session_root):
            return "session://" + resolved.relative_to(session_root).as_posix()
        if resolved == self.shared_workspace_root:
            return "shared://"
        if resolved.is_relative_to(self.shared_workspace_root):
            return (
                "shared://"
                + resolved.relative_to(self.shared_workspace_root).as_posix()
            )
        raise PermissionError("Path is outside the allowed workspace")
