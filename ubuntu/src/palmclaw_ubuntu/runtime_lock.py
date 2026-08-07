from __future__ import annotations

import fcntl
from pathlib import Path
from typing import TextIO


class RuntimeLease:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO | None = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                self._file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            self._file.close()
            self._file = None
            raise RuntimeError(
                "Another PalmClaw runtime is active for this data directory"
            ) from exc

    def close(self) -> None:
        if self._file is None:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None

    def __enter__(self) -> RuntimeLease:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
