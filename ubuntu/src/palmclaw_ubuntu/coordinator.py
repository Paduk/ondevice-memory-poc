from __future__ import annotations

import threading
from dataclasses import replace

from palmclaw_ubuntu.agent import AgentLoop, RunResult
from palmclaw_ubuntu.background_memory import (
    BackgroundFactMemoryWorker,
    BackgroundMemoryPatchWorker,
)
from palmclaw_ubuntu.privacy import redact_secrets


class SessionTurnCoordinator:
    def __init__(
        self,
        agent_loop: AgentLoop,
        *,
        max_concurrent_sessions: int,
        patch_worker: (
            BackgroundMemoryPatchWorker | BackgroundFactMemoryWorker | None
        ) = None,
    ):
        if max_concurrent_sessions < 1:
            raise ValueError("max_concurrent_sessions must be at least 1")
        self.agent_loop = agent_loop
        self.patch_worker = patch_worker
        self._capacity = threading.BoundedSemaphore(max_concurrent_sessions)
        self._guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}

    def run(self, session_id: str, user_text: str) -> RunResult:
        session_lock = self._session_lock(session_id)
        with session_lock, self._capacity:
            result = self.agent_loop.run(session_id, user_text)
            if self.patch_worker is None or result.status != "completed":
                return result
            try:
                job_id = self.patch_worker.enqueue_turn(
                    session_id,
                    result.turn_id,
                )
            except Exception as exc:
                return replace(
                    result,
                    patch_queue_error=redact_secrets(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            return replace(result, patch_job_id=job_id)

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._guard:
            return self._session_locks.setdefault(
                session_id,
                threading.Lock(),
            )
