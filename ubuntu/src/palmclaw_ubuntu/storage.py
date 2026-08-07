from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.fact_memory_schema import (
    fact_memory_record_key,
    normalize_capability_hints,
    normalize_fact_memory_identity,
)
from palmclaw_ubuntu.models import (
    AMemLink,
    AMemNote,
    AMemNoteVersion,
    ChatMessage,
    FactMemoryIdentity,
    FactMemoryRecord,
    MemoryCandidate,
    MemoryEvidence,
    MemoryGateDecision,
    MemoryRecord,
    ModelUsage,
    ToolCall,
    ToolMemoryIdentity,
    ToolMemoryRecord,
)
from palmclaw_ubuntu.privacy import (
    redact_data,
    redact_data_for_cloud,
    redact_secrets,
)
from palmclaw_ubuntu.tool_memory_schema import (
    normalize_identifier,
    normalize_tool_memory_identity,
    split_tool_memory_conditions,
    tool_memory_entity_id,
    tool_memory_identity_family_key,
    tool_memory_record_key,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def utc_after(seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _json(value: Any) -> str:
    return json.dumps(
        redact_data(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _evaluation_json(value: Any) -> str:
    safe_value, _ = redact_data_for_cloud(value)
    return _json(safe_value)


def _idempotency_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StoredSession:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StoredMessage:
    id: int
    session_id: str
    turn_id: str | None
    role: str
    content: str
    tool_calls: tuple[ToolCall, ...]
    tool_call_id: str | None
    provider_items: tuple[dict[str, Any], ...]
    created_at: str

    def as_chat_message(self) -> ChatMessage:
        return ChatMessage(
            role=self.role,
            content=self.content,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
            provider_items=self.provider_items,
        )


class SQLiteRepository:
    def __init__(self, database_path: Path | str):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")
        if self.database_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._apply_migrations()
        self._backfill_tool_memory_identity_metadata()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def clone_in_memory(self) -> SQLiteRepository:
        """Create an isolated, transactionally consistent in-memory clone."""

        clone = SQLiteRepository(":memory:")
        try:
            with self._lock, clone._lock:
                self._connection.backup(clone._connection)
                clone._connection.row_factory = sqlite3.Row
                clone._connection.execute("PRAGMA foreign_keys=ON")
            return clone
        except Exception:
            clone.close()
            raise

    def backup_to(self, database_path: Path | str) -> None:
        """Write a transactionally consistent clone to a new SQLite file."""

        destination = Path(database_path).expanduser().resolve()
        if self.database_path != ":memory:" and destination == Path(
            self.database_path
        ).expanduser().resolve():
            raise ValueError("SQLite backup destination must be different")
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(destination))
        try:
            with self._lock:
                self._connection.backup(target)
        finally:
            target.close()

    def _apply_migrations(self) -> None:
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            applied = {
                int(row["version"])
                for row in self._connection.execute(
                    "SELECT version FROM schema_migrations"
                )
            }
            migration_root = resources.files("palmclaw_ubuntu.migrations")
            migrations = sorted(
                entry
                for entry in migration_root.iterdir()
                if entry.name.endswith(".sql")
            )
            for entry in migrations:
                version_text = entry.name.split("_", 1)[0]
                version = int(version_text)
                if version in applied:
                    continue
                self._connection.executescript(entry.read_text(encoding="utf-8"))
                self._connection.execute(
                    """
                    INSERT INTO schema_migrations(version, name, applied_at)
                    VALUES (?, ?, ?)
                    """,
                    (version, entry.name, utc_now()),
                )
            self._connection.commit()

    def _backfill_tool_memory_identity_metadata(self) -> None:
        with self._lock:
            columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(tool_memory_records)"
                ).fetchall()
            }
            if "identity_family_key" not in columns:
                return
            rows = self._connection.execute(
                """
                SELECT *
                FROM tool_memory_records
                WHERE identity_family_key IS NULL
                """
            ).fetchall()
            if not rows:
                return
            updates = []
            for row in rows:
                identity = ToolMemoryIdentity(
                    user_id=str(row["user_id"]),
                    tool_domain=str(row["tool_domain"]),
                    topic=str(row["topic"]),
                    scope=str(row["scope"]),
                    scope_key=str(row["scope_key"]),
                    conditions=json.loads(row["conditions_json"]),
                )
                identity_conditions, applicability = (
                    split_tool_memory_conditions(identity.conditions)
                )
                updates.append(
                    (
                        tool_memory_entity_id(identity),
                        _json(identity_conditions),
                        _json(applicability),
                        tool_memory_identity_family_key(identity),
                        row["id"],
                    )
                )
            with self._connection:
                self._connection.executemany(
                    """
                    UPDATE tool_memory_records
                    SET entity_id = ?,
                        identity_conditions_json = ?,
                        applicability_json = ?,
                        identity_family_key = ?
                    WHERE id = ?
                    """,
                    updates,
                )
                self._connection.commit()

    @contextmanager
    def _write_scope(self) -> Iterator[None]:
        with self._lock:
            owns_transaction = not self._connection.in_transaction
            if owns_transaction:
                self._connection.execute("BEGIN")
            try:
                yield
            except BaseException:
                if owns_transaction:
                    self._connection.rollback()
                raise
            else:
                if owns_transaction:
                    self._connection.commit()

    @contextmanager
    def memory_patch_transaction(self) -> Iterator[None]:
        with self._lock:
            if self._connection.in_transaction:
                raise RuntimeError("A SQLite transaction is already active")
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    @contextmanager
    def amem_transaction(self) -> Iterator[None]:
        """Commit one validated A-MEM ingestion step atomically."""

        with self._lock:
            if self._connection.in_transaction:
                raise RuntimeError("A SQLite transaction is already active")
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def create_session(self, title: str = "Chat") -> StoredSession:
        session_id = str(uuid.uuid4())
        now = utc_now()
        normalized_title = redact_secrets(title.strip() or "Chat")
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO sessions(id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, normalized_title, now, now),
            )
        return StoredSession(session_id, normalized_title, now, now)

    def list_sessions(self) -> list[StoredSession]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [StoredSession(**dict(row)) for row in rows]

    def get_session(self, session_id: str) -> StoredSession | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        return StoredSession(**dict(row)) if row else None

    def require_session(self, session_id: str) -> StoredSession:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Unknown session: {session_id}")
        return session

    def set_active_session(self, session_id: str) -> None:
        self.require_session(session_id)
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_state(key, value, updated_at)
                VALUES ('active_session_id', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (session_id, now),
            )

    def get_active_session_id(self) -> str | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT value
                FROM runtime_state
                WHERE key = 'active_session_id'
                """
            ).fetchone()
        if row is None:
            return None
        session_id = str(row["value"])
        return session_id if self.get_session(session_id) else None

    def create_turn(self, session_id: str) -> str:
        self.require_session(session_id)
        turn_id = str(uuid.uuid4())
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO turns(
                    id, session_id, status, round_count, started_at
                )
                VALUES (?, ?, 'running', 0, ?)
                """,
                (turn_id, session_id, now),
            )
        return turn_id

    def recover_interrupted_execution(self) -> dict[str, int]:
        now = utc_now()
        with self._lock, self._connection:
            running_turns = [
                str(row["id"])
                for row in self._connection.execute(
                    "SELECT id FROM turns WHERE status = 'running'"
                ).fetchall()
            ]
            if running_turns:
                placeholders = ",".join("?" for _ in running_turns)
                tool_cursor = self._connection.execute(
                    f"""
                    UPDATE tool_calls
                    SET status = 'interrupted',
                        error = 'Process restarted before Tool completion',
                        updated_at = ?
                    WHERE turn_id IN ({placeholders})
                      AND status IN ('proposed', 'executing')
                    """,
                    (now, *running_turns),
                )
                self._connection.execute(
                    f"""
                    UPDATE turns
                    SET status = 'interrupted',
                        ended_at = ?,
                        terminal_reason = 'process_restart'
                    WHERE id IN ({placeholders})
                    """,
                    (now, *running_turns),
                )
            else:
                tool_cursor = None
            consolidation_cursor = self._connection.execute(
                """
                UPDATE consolidation_runs
                SET status = 'interrupted',
                    error = 'Process restarted during consolidation',
                    ended_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
            completed_patch_jobs = self._connection.execute(
                """
                UPDATE memory_patch_jobs
                SET status = 'completed',
                    completed_at = ?,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE status = 'running'
                  AND EXISTS (
                      SELECT 1
                      FROM memory_patch_runs
                      WHERE (
                            memory_patch_runs.job_id = memory_patch_jobs.id
                            OR memory_patch_runs.id =
                                memory_patch_jobs.batch_run_id
                        )
                        AND memory_patch_runs.status = 'completed'
                  )
                """,
                (now, now),
            )
            interrupted_patch_jobs = self._connection.execute(
                """
                UPDATE memory_patch_jobs
                SET status = CASE
                        WHEN attempt_count >= max_attempts
                        THEN 'failed'
                        ELSE 'retryable'
                    END,
                    available_at = ?,
                    lease_expires_at = NULL,
                    last_error = 'Process restarted during memory patching',
                    completed_at = CASE
                        WHEN attempt_count >= max_attempts
                        THEN ?
                        ELSE completed_at
                    END,
                    updated_at = ?
                WHERE status = 'running'
                """,
                (now, now, now),
            )
            interrupted_patch_runs = self._connection.execute(
                """
                UPDATE memory_patch_runs
                SET status = 'interrupted',
                    error = 'Process restarted during memory patching',
                    ended_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
            interrupted_tool_memory_retrievals = self._connection.execute(
                """
                UPDATE tool_memory_retrieval_runs
                SET status = 'failed',
                    candidate_count = 0,
                    selected_count = 0,
                    selected_tokens = 0,
                    error = 'Process restarted during Tool memory retrieval',
                    ended_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
            interrupted_amem_constructions = self._connection.execute(
                """
                UPDATE amem_construction_runs
                SET status = 'interrupted',
                    error = 'Process restarted during A-MEM construction',
                    ended_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
            interrupted_amem_evolutions = self._connection.execute(
                """
                UPDATE amem_evolution_events
                SET status = 'interrupted',
                    error = 'Process restarted during A-MEM evolution',
                    ended_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
            interrupted_amem_retrievals = self._connection.execute(
                """
                UPDATE amem_retrieval_runs
                SET status = 'interrupted',
                    error = 'Process restarted during A-MEM retrieval',
                    ended_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
            interrupted_compact_amem_episodes = self._connection.execute(
                """
                UPDATE compact_amem_episode_runs
                SET status = 'interrupted',
                    error = 'Process restarted during Compact A-MEM compaction',
                    ended_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
        recovery = {
            "turns": len(running_turns),
            "tools": tool_cursor.rowcount if tool_cursor else 0,
            "consolidations": consolidation_cursor.rowcount,
        }
        for key, count in (
            ("memory_patch_jobs_completed", completed_patch_jobs.rowcount),
            ("memory_patch_jobs_interrupted", interrupted_patch_jobs.rowcount),
            ("memory_patch_runs_interrupted", interrupted_patch_runs.rowcount),
            (
                "tool_memory_retrievals_interrupted",
                interrupted_tool_memory_retrievals.rowcount,
            ),
            (
                "amem_constructions_interrupted",
                interrupted_amem_constructions.rowcount,
            ),
            (
                "amem_evolutions_interrupted",
                interrupted_amem_evolutions.rowcount,
            ),
            (
                "amem_retrievals_interrupted",
                interrupted_amem_retrievals.rowcount,
            ),
            (
                "compact_amem_episodes_interrupted",
                interrupted_compact_amem_episodes.rowcount,
            ),
        ):
            if count:
                recovery[key] = count
        return recovery

    def finish_turn(
        self,
        turn_id: str,
        status: str,
        round_count: int,
        terminal_reason: str,
    ) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE turns
                SET status = ?,
                    round_count = ?,
                    ended_at = ?,
                    terminal_reason = ?
                WHERE id = ?
                """,
                (
                    status,
                    round_count,
                    utc_now(),
                    redact_secrets(terminal_reason),
                    turn_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown turn: {turn_id}")

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        turn_id: str | None = None,
        tool_calls: Sequence[ToolCall] = (),
        tool_call_id: str | None = None,
        provider_items: Sequence[dict[str, Any]] = (),
    ) -> int:
        safe_content = redact_secrets(content)
        calls_json = (
            _json(
                [
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": dict(call.arguments),
                    }
                    for call in tool_calls
                ]
            )
            if tool_calls
            else None
        )
        provider_items_json = _json(list(provider_items)) if provider_items else None
        now = utc_now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO messages(
                    session_id,
                    turn_id,
                    role,
                    content,
                    tool_calls_json,
                    tool_call_id,
                    provider_items_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_id,
                    role,
                    safe_content,
                    calls_json,
                    tool_call_id,
                    provider_items_json,
                    now,
                ),
            )
            self._connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        return int(cursor.lastrowid)

    def list_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        after_id: int | None = None,
    ) -> list[StoredMessage]:
        conditions = ["session_id = ?"]
        parameters: list[Any] = [session_id]
        if after_id is not None:
            conditions.append("id > ?")
            parameters.append(after_id)
        query = f"""
            SELECT
                id,
                session_id,
                turn_id,
                role,
                content,
                tool_calls_json,
                tool_call_id,
                provider_items_json,
                created_at
            FROM messages
            WHERE {" AND ".join(conditions)}
            ORDER BY id ASC
        """
        if limit is not None:
            query = f"""
                SELECT *
                FROM ({query})
                ORDER BY id DESC
                LIMIT ?
            """
            parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                query,
                tuple(parameters),
            ).fetchall()
        if limit is not None:
            rows = list(reversed(rows))
        return [self._decode_message(row) for row in rows]

    @staticmethod
    def _decode_message(row: sqlite3.Row) -> StoredMessage:
        calls_data = (
            json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else []
        )
        calls = tuple(
            ToolCall(
                id=item["id"],
                name=item["name"],
                arguments=item.get("arguments", {}),
            )
            for item in calls_data
        )
        provider_items = tuple(
            json.loads(row["provider_items_json"]) if row["provider_items_json"] else []
        )
        return StoredMessage(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            turn_id=row["turn_id"],
            role=str(row["role"]),
            content=str(row["content"]),
            tool_calls=calls,
            tool_call_id=row["tool_call_id"],
            provider_items=provider_items,
            created_at=str(row["created_at"]),
        )

    def record_tool_call(
        self,
        turn_id: str,
        round_number: int,
        call: ToolCall,
        *,
        side_effect: str,
        retry_safety: str,
        fingerprint: str,
        status: str = "proposed",
    ) -> None:
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO tool_calls(
                    id,
                    turn_id,
                    round_number,
                    name,
                    arguments_json,
                    status,
                    side_effect,
                    retry_safety,
                    fingerprint,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call.id,
                    turn_id,
                    round_number,
                    call.name,
                    _json(dict(call.arguments)),
                    status,
                    side_effect,
                    retry_safety,
                    fingerprint,
                    now,
                    now,
                ),
            )

    def mark_tool_call_executing(self, call_id: str) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE tool_calls
                SET status = 'executing', updated_at = ?
                WHERE id = ? AND status = 'proposed'
                """,
                (utc_now(), call_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Tool call is not executable from its current state: {call_id}"
                )

    def record_tool_result(
        self,
        call_id: str,
        *,
        content: str,
        is_error: bool,
        metadata: dict[str, Any],
        duration_ms: int,
        status: str | None = None,
    ) -> None:
        safe_content = redact_secrets(content)
        resolved_status = status or ("failed" if is_error else "succeeded")
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE tool_calls
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    resolved_status,
                    safe_content if is_error else None,
                    now,
                    call_id,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO tool_results(
                    id,
                    tool_call_id,
                    content,
                    is_error,
                    metadata_json,
                    duration_ms,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    call_id,
                    safe_content,
                    int(is_error),
                    _json(metadata),
                    duration_ms,
                    now,
                ),
            )

    def latest_memory(
        self,
        session_id: str,
        scope: str = "session",
    ) -> str:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT content
                FROM memories
                WHERE session_id = ?
                  AND scope = ?
                  AND status = 'verified'
                  AND fact_key IS NULL
                ORDER BY version DESC
                LIMIT 1
                """,
                (session_id, scope),
            ).fetchone()
        return str(row["content"]) if row else ""

    def list_memories(
        self,
        session_id: str,
        *,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        status_clause = "" if include_superseded else "AND status = 'verified'"
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM memories
                WHERE (session_id = ? OR scope = 'global')
                {status_clause}
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        return [self._decode_json_columns(dict(row), ("value_json",)) for row in rows]

    def active_structured_memories(
        self,
        session_id: str,
    ) -> list[MemoryRecord]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM memories
                WHERE status = 'verified'
                  AND fact_key IS NOT NULL
                  AND (
                    (scope = 'session' AND session_id = ?)
                    OR scope = 'global'
                  )
                ORDER BY created_at DESC
                """,
                (session_id,),
            ).fetchall()
        return [self._decode_memory_record(row) for row in rows]

    def active_fact_memory(self, fact_key: str) -> MemoryRecord | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM memories
                WHERE fact_key = ? AND status = 'verified'
                LIMIT 1
                """,
                (fact_key,),
            ).fetchone()
        return self._decode_memory_record(row) if row else None

    def insert_tool_memory_record(
        self,
        *,
        session_id: str,
        identity: ToolMemoryIdentity,
        value: Any,
        memory_type: str,
        confidence: float,
        version: int,
        status: str = "active",
        supersedes_id: str | None = None,
        merged_into_id: str | None = None,
        evidence: Sequence[MemoryEvidence] = (),
        record_id: str | None = None,
        patch_proposal_id: str | None = None,
        status_reason: str = "record_created",
        allow_cross_session_evidence: bool = False,
    ) -> ToolMemoryRecord:
        self.require_session(session_id)
        normalized = normalize_tool_memory_identity(identity)
        record_key = tool_memory_record_key(normalized)
        identity_conditions, applicability = split_tool_memory_conditions(
            normalized.conditions
        )
        entity_id = tool_memory_entity_id(normalized)
        identity_family_key = tool_memory_identity_family_key(normalized)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if version < 1:
            raise ValueError("version must be positive")
        allowed_statuses = {
            "active",
            "superseded",
            "merged",
            "deleted",
            "rejected",
        }
        if status not in allowed_statuses:
            raise ValueError(f"Unsupported tool memory status: {status}")
        resolved_id = record_id or str(uuid.uuid4())
        now = utc_now()
        with self._write_scope():
            for source in evidence:
                row = self._connection.execute(
                    "SELECT session_id FROM messages WHERE id = ?",
                    (source.message_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown evidence message: {source.message_id}")
                if (
                    str(row["session_id"]) != session_id
                    and not allow_cross_session_evidence
                ):
                    raise ValueError(
                        "Tool memory evidence must belong to the record session"
                    )
            self._connection.execute(
                """
                INSERT INTO tool_memory_records(
                    id,
                    session_id,
                    record_key,
                    user_id,
                    tool_domain,
                    topic,
                    scope,
                    scope_key,
                    conditions_json,
                    value_json,
                    memory_type,
                    status,
                    confidence,
                    version,
                    supersedes_id,
                    merged_into_id,
                    created_at,
                    updated_at,
                    entity_id,
                    identity_conditions_json,
                    applicability_json,
                    identity_family_key
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    resolved_id,
                    session_id,
                    record_key,
                    normalized.user_id,
                    normalized.tool_domain,
                    normalized.topic,
                    normalized.scope,
                    normalized.scope_key,
                    _json(dict(normalized.conditions)),
                    _json(value),
                    redact_secrets(memory_type.strip()),
                    status,
                    confidence,
                    version,
                    supersedes_id,
                    merged_into_id,
                    now,
                    now,
                    entity_id,
                    _json(identity_conditions),
                    _json(applicability),
                    identity_family_key,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO tool_memory_record_status_events(
                    record_id,
                    status,
                    reason,
                    patch_proposal_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    status,
                    redact_secrets(status_reason),
                    patch_proposal_id,
                    now,
                ),
            )
            for source in evidence:
                self._connection.execute(
                    """
                    INSERT INTO tool_memory_record_sources(
                        record_id,
                        message_id,
                        evidence_text,
                        start_char,
                        end_char,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_id,
                        source.message_id,
                        redact_secrets(source.quote),
                        source.start_char,
                        source.end_char,
                        now,
                    ),
                )
            row = self._connection.execute(
                "SELECT * FROM tool_memory_records WHERE id = ?",
                (resolved_id,),
            ).fetchone()
        assert row is not None
        return self._decode_tool_memory_record(row)

    def list_tool_memory_records(
        self,
        session_id: str,
        *,
        active_only: bool = True,
        user_id: str | None = None,
        tool_domain: str | None = None,
        topic: str | None = None,
    ) -> list[ToolMemoryRecord]:
        clauses = ["session_id = ?"]
        parameters: list[Any] = [session_id]
        if active_only:
            clauses.append("status = 'active'")
        for column, value in (
            ("user_id", user_id),
            ("tool_domain", tool_domain),
            ("topic", topic),
        ):
            if value is None:
                continue
            clauses.append(f"{column} = ?")
            parameters.append(normalize_identifier(value))
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM tool_memory_records
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC, version DESC
                """,
                tuple(parameters),
            ).fetchall()
        return [self._decode_tool_memory_record(row) for row in rows]

    def active_tool_memory_record(
        self,
        record_key: str,
    ) -> ToolMemoryRecord | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM tool_memory_records
                WHERE record_key = ? AND status = 'active'
                LIMIT 1
                """,
                (record_key,),
            ).fetchone()
        return self._decode_tool_memory_record(row) if row else None

    def tool_memory_record(self, record_id: str) -> ToolMemoryRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM tool_memory_records WHERE id = ?",
                (record_id,),
            ).fetchone()
        return self._decode_tool_memory_record(row) if row else None

    def tool_memory_record_sources(
        self,
        record_id: str,
    ) -> tuple[MemoryEvidence, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT message_id, evidence_text, start_char, end_char
                FROM tool_memory_record_sources
                WHERE record_id = ?
                ORDER BY message_id, start_char, evidence_text
                """,
                (record_id,),
            ).fetchall()
        return tuple(
            MemoryEvidence(
                message_id=int(row["message_id"]),
                quote=str(row["evidence_text"]),
                start_char=row["start_char"],
                end_char=row["end_char"],
            )
            for row in rows
        )

    def next_tool_memory_version(self, record_key: str) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) AS version
                FROM tool_memory_records
                WHERE record_key = ?
                """,
                (record_key,),
            ).fetchone()
        return int(row["version"]) + 1

    def transition_tool_memory_record(
        self,
        record_id: str,
        *,
        status: str,
        reason: str,
        patch_proposal_id: str,
        merged_into_id: str | None = None,
    ) -> ToolMemoryRecord:
        if status not in {"superseded", "merged", "deleted", "rejected"}:
            raise ValueError(f"Unsupported record transition status: {status}")
        now = utc_now()
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE tool_memory_records
                SET status = ?,
                    merged_into_id = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (status, merged_into_id, now, record_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Tool memory record is not active: {record_id}")
            self._connection.execute(
                """
                INSERT INTO tool_memory_record_status_events(
                    record_id,
                    status,
                    reason,
                    patch_proposal_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    status,
                    redact_secrets(reason),
                    patch_proposal_id,
                    now,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM tool_memory_records WHERE id = ?",
                (record_id,),
            ).fetchone()
        assert row is not None
        return self._decode_tool_memory_record(row)

    def link_merged_tool_memory_record(
        self,
        record_id: str,
        *,
        merged_into_id: str,
    ) -> None:
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE tool_memory_records
                SET merged_into_id = ?, updated_at = ?
                WHERE id = ? AND status = 'merged'
                """,
                (merged_into_id, utc_now(), record_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Tool memory record is not merged: {record_id}")

    def insert_fact_memory_record(
        self,
        *,
        session_id: str,
        identity: FactMemoryIdentity,
        value: Any,
        memory_type: str,
        confidence: float,
        version: int,
        capability_hints: Sequence[str] = (),
        status: str = "active",
        supersedes_id: str | None = None,
        merged_into_id: str | None = None,
        evidence: Sequence[MemoryEvidence] = (),
        record_id: str | None = None,
        patch_event_id: str | None = None,
        status_reason: str = "record_created",
        allow_cross_session_evidence: bool = False,
        bundle_id: str | None = None,
    ) -> FactMemoryRecord:
        """Store a versioned fact without binding it to a concrete tool schema."""
        self.require_session(session_id)
        normalized = normalize_fact_memory_identity(identity)
        record_key = fact_memory_record_key(normalized)
        hints = normalize_capability_hints(capability_hints)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if version < 1:
            raise ValueError("version must be positive")
        if status not in {
            "active",
            "superseded",
            "merged",
            "deleted",
            "review",
        }:
            raise ValueError(f"Unsupported fact memory status: {status}")
        normalized_type = normalize_identifier(memory_type)
        if not normalized_type:
            raise ValueError("memory_type cannot be empty")
        resolved_id = record_id or str(uuid.uuid4())
        resolved_bundle_id = bundle_id or _fact_memory_bundle_id(
            session_id=session_id,
            entity_id=normalized.entity_id,
            evidence=evidence,
            record_id=resolved_id,
        )
        now = utc_now()
        with self._write_scope():
            for source in evidence:
                row = self._connection.execute(
                    "SELECT session_id FROM messages WHERE id = ?",
                    (source.message_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown evidence message: {source.message_id}")
                if (
                    str(row["session_id"]) != session_id
                    and not allow_cross_session_evidence
                ):
                    raise ValueError(
                        "Fact memory evidence must belong to the record session"
                    )
            self._connection.execute(
                """
                INSERT INTO fact_memory_records(
                    id,
                    session_id,
                    record_key,
                    user_id,
                    entity_id,
                    predicate,
                    identity_conditions_json,
                    applicability_json,
                    capability_hints_json,
                    value_json,
                    memory_type,
                    status,
                    confidence,
                    version,
                    supersedes_id,
                    merged_into_id,
                    bundle_id,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    session_id,
                    record_key,
                    normalized.user_id,
                    normalized.entity_id,
                    normalized.predicate,
                    _json(dict(normalized.identity_conditions)),
                    _json(dict(normalized.applicability)),
                    _json(hints),
                    _json(value),
                    normalized_type,
                    status,
                    confidence,
                    version,
                    supersedes_id,
                    merged_into_id,
                    resolved_bundle_id,
                    now,
                    now,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO fact_memory_record_status_events(
                    record_id,
                    status,
                    reason,
                    patch_event_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    status,
                    redact_secrets(status_reason),
                    patch_event_id,
                    now,
                ),
            )
            for source in evidence:
                self._connection.execute(
                    """
                    INSERT INTO fact_memory_record_sources(
                        record_id,
                        message_id,
                        evidence_text,
                        start_char,
                        end_char,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_id,
                        source.message_id,
                        redact_secrets(source.quote),
                        source.start_char,
                        source.end_char,
                        now,
                    ),
                )
            row = self._connection.execute(
                "SELECT * FROM fact_memory_records WHERE id = ?",
                (resolved_id,),
            ).fetchone()
        assert row is not None
        return self._decode_fact_memory_record(row)

    def list_fact_memory_records(
        self,
        session_id: str,
        *,
        active_only: bool = True,
        user_id: str | None = None,
        entity_id: str | None = None,
        predicate: str | None = None,
        status: str | None = None,
    ) -> list[FactMemoryRecord]:
        clauses = ["session_id = ?"]
        parameters: list[Any] = [session_id]
        if active_only:
            clauses.append("status = 'active'")
        elif status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        for column, value in (
            ("user_id", user_id),
            ("entity_id", entity_id),
            ("predicate", predicate),
        ):
            if value is None:
                continue
            clauses.append(f"{column} = ?")
            parameters.append(normalize_identifier(value))
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM fact_memory_records
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC, version DESC
                """,
                tuple(parameters),
            ).fetchall()
        return [self._decode_fact_memory_record(row) for row in rows]

    def store_fact_memory_embedding(
        self,
        *,
        record_id: str,
        model_id: str,
        dimensions: int,
        vector: Sequence[float],
    ) -> None:
        if dimensions < 1 or len(vector) != dimensions:
            raise ValueError("Invalid Fact memory embedding dimensions")
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO fact_memory_embeddings(
                    record_id, model_id, dimensions, vector_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(record_id, model_id, dimensions)
                DO UPDATE SET vector_json = excluded.vector_json,
                              created_at = excluded.created_at
                """,
                (
                    record_id,
                    model_id,
                    dimensions,
                    _json([float(value) for value in vector]),
                    utc_now(),
                ),
            )

    def fact_memory_embeddings(
        self,
        record_ids: Sequence[str],
        *,
        model_id: str,
        dimensions: int,
    ) -> dict[str, tuple[float, ...]]:
        normalized = tuple(dict.fromkeys(str(item) for item in record_ids))
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT record_id, vector_json
                FROM fact_memory_embeddings
                WHERE record_id IN ({placeholders})
                  AND model_id = ?
                  AND dimensions = ?
                """,
                (*normalized, model_id, dimensions),
            ).fetchall()
        return {
            str(row["record_id"]): tuple(
                float(value) for value in json.loads(row["vector_json"])
            )
            for row in rows
        }

    def begin_fact_memory_retrieval(
        self,
        *,
        session_id: str,
        turn_id: str | None,
        query: str,
        user_id: str,
        mode: str,
        top_k: int,
        token_budget: int,
        query_context: Mapping[str, Any],
        route: Mapping[str, Any],
    ) -> str:
        self.require_session(session_id)
        if turn_id is not None:
            with self._lock:
                turn = self._connection.execute(
                    "SELECT session_id FROM turns WHERE id = ?",
                    (turn_id,),
                ).fetchone()
            if turn is None:
                raise KeyError(f"Unknown Fact retrieval turn: {turn_id}")
            if str(turn["session_id"]) != session_id:
                raise ValueError(
                    "Fact memory retrieval turn belongs to another session"
                )
        run_id = str(uuid.uuid4())
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO fact_memory_retrieval_runs(
                    id, session_id, turn_id, query, user_id, mode, top_k,
                    token_budget, query_context_json, route_json, status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    session_id,
                    turn_id,
                    redact_secrets(query),
                    normalize_identifier(user_id),
                    mode,
                    top_k,
                    token_budget,
                    _json(dict(query_context)),
                    _json(dict(route)),
                    utc_now(),
                ),
            )
        return run_id

    def complete_fact_memory_retrieval(
        self,
        run_id: str,
        candidates: Sequence[Mapping[str, Any]],
        *,
        selected_count: int,
        selected_tokens: int,
        latency_ms: int,
    ) -> None:
        with self._write_scope():
            for candidate in candidates:
                self._connection.execute(
                    """
                    INSERT INTO fact_memory_retrieval_candidates(
                        retrieval_run_id, record_id, rank, bm25_score,
                        embedding_score, entity_score, condition_score,
                        route_score, combined_score, selected,
                        exclusion_reason, token_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        candidate["record_id"],
                        candidate.get("rank"),
                        candidate.get("bm25_score", 0.0),
                        candidate.get("embedding_score", 0.0),
                        candidate.get("entity_score", 0.0),
                        candidate.get("condition_score", 0.0),
                        candidate.get("route_score", 0.0),
                        candidate.get("combined_score", 0.0),
                        int(bool(candidate.get("selected"))),
                        candidate.get("exclusion_reason"),
                        candidate.get("token_count", 0),
                    ),
                )
            cursor = self._connection.execute(
                """
                UPDATE fact_memory_retrieval_runs
                SET status = 'completed', candidate_count = ?,
                    selected_count = ?, selected_tokens = ?, latency_ms = ?,
                    ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    len(candidates),
                    selected_count,
                    selected_tokens,
                    latency_ms,
                    utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running Fact memory retrieval not found: {run_id}")

    def fail_fact_memory_retrieval(
        self,
        run_id: str,
        *,
        error: str,
        latency_ms: int,
    ) -> None:
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE fact_memory_retrieval_runs
                SET status = 'failed', candidate_count = 0,
                    selected_count = 0, selected_tokens = 0, latency_ms = ?,
                    error = ?, ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (latency_ms, redact_secrets(error), utc_now(), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running Fact memory retrieval not found: {run_id}")

    def fact_memory_retrieval_trace(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._connection.execute(
                "SELECT * FROM fact_memory_retrieval_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown Fact memory retrieval: {run_id}")
            candidates = self._connection.execute(
                """
                SELECT
                    fact_memory_retrieval_candidates.*,
                    fact_memory_records.entity_id,
                    fact_memory_records.predicate,
                    fact_memory_records.identity_conditions_json,
                    fact_memory_records.applicability_json,
                    fact_memory_records.capability_hints_json,
                    fact_memory_records.value_json,
                    fact_memory_records.version,
                    fact_memory_records.confidence
                FROM fact_memory_retrieval_candidates
                JOIN fact_memory_records
                  ON fact_memory_records.id =
                     fact_memory_retrieval_candidates.record_id
                WHERE retrieval_run_id = ?
                ORDER BY selected DESC, rank, record_id
                """,
                (run_id,),
            ).fetchall()
        return {
            "run": self._decode_json_columns(
                dict(run),
                ("query_context_json", "route_json"),
            ),
            "candidates": [
                self._decode_json_columns(
                    dict(row),
                    (
                        "identity_conditions_json",
                        "applicability_json",
                        "capability_hints_json",
                        "value_json",
                    ),
                )
                for row in candidates
            ],
        }

    def begin_compact_amem_episode(
        self,
        *,
        session_id: str,
        episode_index: int,
        start_timestamp: str,
        end_timestamp: str,
        source_message_ids: Sequence[int],
        backend: str,
        model_id: str,
        prompt_version: str,
        schema_version: str,
        idempotency_key: str,
    ) -> str:
        self.require_session(session_id)
        if episode_index < 0 or not source_message_ids:
            raise ValueError("Invalid Compact A-MEM episode identity")
        run_id = str(uuid.uuid4())
        now = utc_now()
        with self._write_scope():
            self._connection.execute(
                """
                UPDATE compact_amem_episode_runs
                SET status = 'interrupted',
                    error = 'Superseded by a resumed Compact A-MEM episode',
                    ended_at = ?
                WHERE session_id = ? AND episode_index = ?
                  AND status = 'running'
                """,
                (now, session_id, episode_index),
            )
            self._connection.execute(
                """
                INSERT INTO compact_amem_episode_runs(
                    id, session_id, episode_index, start_timestamp,
                    end_timestamp, source_message_ids_json, backend, model_id,
                    prompt_version, schema_version, idempotency_key, status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    session_id,
                    episode_index,
                    start_timestamp,
                    end_timestamp,
                    _json(list(source_message_ids)),
                    backend,
                    model_id,
                    prompt_version,
                    schema_version,
                    idempotency_key,
                    now,
                ),
            )
        return run_id

    def completed_compact_amem_episode_indices(
        self,
        session_id: str,
    ) -> frozenset[int]:
        self.require_session(session_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT DISTINCT episode_index
                FROM compact_amem_episode_runs
                WHERE session_id = ? AND status = 'completed'
                """,
                (session_id,),
            ).fetchall()
        return frozenset(int(row["episode_index"]) for row in rows)

    def completed_compact_amem_episode_sources(
        self,
        session_id: str,
    ) -> dict[int, tuple[int, ...]]:
        self.require_session(session_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT episode_index, source_message_ids_json, started_at
                FROM compact_amem_episode_runs
                WHERE session_id = ? AND status = 'completed'
                ORDER BY episode_index, started_at
                """,
                (session_id,),
            ).fetchall()
        return {
            int(row["episode_index"]): tuple(
                int(value)
                for value in json.loads(row["source_message_ids_json"])
            )
            for row in rows
        }

    def complete_compact_amem_episode(
        self,
        run_id: str,
        *,
        output: Mapping[str, Any],
        usage: ModelUsage,
    ) -> None:
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE compact_amem_episode_runs
                SET status = 'completed', output_json = ?, usage_json = ?,
                    ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    _json(dict(output)),
                    _json(usage.as_dict()),
                    utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running Compact A-MEM episode not found: {run_id}")

    def fail_compact_amem_episode(self, run_id: str, *, error: str) -> None:
        self._fail_amem_run("compact_amem_episode_runs", run_id, error)

    def insert_compact_amem_note_provenance(
        self,
        *,
        note_id: str,
        episode_run_id: str,
        note_ordinal: int,
        memory_kind: str,
        source_message_ids: Sequence[int],
    ) -> None:
        if note_ordinal < 0 or not source_message_ids:
            raise ValueError("Invalid Compact A-MEM note provenance")
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO compact_amem_notes(
                    note_id, episode_run_id, note_ordinal, memory_kind
                )
                VALUES (?, ?, ?, ?)
                """,
                (note_id, episode_run_id, note_ordinal, memory_kind),
            )
            self._connection.executemany(
                """
                INSERT INTO compact_amem_note_sources(
                    note_id, source_message_id, source_ordinal
                )
                VALUES (?, ?, ?)
                """,
                tuple(
                    (note_id, source_id, ordinal)
                    for ordinal, source_id in enumerate(source_message_ids)
                ),
            )

    def compact_amem_note_sources(self, note_id: str) -> tuple[int, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT source_message_id FROM compact_amem_note_sources
                WHERE note_id = ? ORDER BY source_ordinal
                """,
                (note_id,),
            ).fetchall()
        return tuple(int(row["source_message_id"]) for row in rows)

    def compact_amem_graph_fingerprint(self, session_id: str) -> str:
        base = self.amem_graph_fingerprint(session_id)
        with self._lock:
            notes = [
                dict(row)
                for row in self._connection.execute(
                    """
                    SELECT compact.*
                    FROM compact_amem_notes AS compact
                    JOIN amem_notes AS notes ON notes.id = compact.note_id
                    WHERE notes.session_id = ?
                    ORDER BY compact.note_id
                    """,
                    (session_id,),
                ).fetchall()
            ]
            sources = [
                dict(row)
                for row in self._connection.execute(
                    """
                    SELECT sources.*
                    FROM compact_amem_note_sources AS sources
                    JOIN amem_notes AS notes ON notes.id = sources.note_id
                    WHERE notes.session_id = ?
                    ORDER BY sources.note_id, sources.source_ordinal
                    """,
                    (session_id,),
                ).fetchall()
            ]
        canonical = json.dumps(
            {"amem_graph": base, "compact_notes": notes, "sources": sources},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def insert_amem_note(
        self,
        *,
        session_id: str,
        source_message_id: int,
        timestamp: str,
        speaker: str,
        content: str,
        note_id: str | None = None,
    ) -> AMemNote:
        self.require_session(session_id)
        if source_message_id < 0:
            raise ValueError("A-MEM source_message_id cannot be negative")
        resolved_id = note_id or str(uuid.uuid4())
        now = utc_now()
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO amem_notes(
                    id, session_id, source_message_id, timestamp, speaker,
                    content, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    resolved_id,
                    session_id,
                    source_message_id,
                    timestamp,
                    speaker,
                    redact_secrets(content),
                    now,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM amem_notes WHERE id = ?",
                (resolved_id,),
            ).fetchone()
        assert row is not None
        return self._decode_amem_note(row)

    def get_amem_note(self, note_id: str) -> AMemNote | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM amem_notes WHERE id = ?",
                (note_id,),
            ).fetchone()
        return self._decode_amem_note(row) if row else None

    def amem_note_by_source(
        self,
        session_id: str,
        source_message_id: int,
    ) -> AMemNote | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM amem_notes
                WHERE session_id = ? AND source_message_id = ?
                """,
                (session_id, source_message_id),
            ).fetchone()
        return self._decode_amem_note(row) if row else None

    def list_amem_notes(
        self,
        session_id: str,
        *,
        active_only: bool = True,
    ) -> tuple[AMemNote, ...]:
        status_clause = "AND status = 'active'" if active_only else ""
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM amem_notes
                WHERE session_id = ? {status_clause}
                ORDER BY timestamp, source_message_id, id
                """,
                (session_id,),
            ).fetchall()
        return tuple(self._decode_amem_note(row) for row in rows)

    def insert_amem_note_version(
        self,
        *,
        note_id: str,
        context: str,
        keywords: Sequence[str],
        tags: Sequence[str],
        created_by_event_id: str | None = None,
        version_id: str | None = None,
    ) -> AMemNoteVersion:
        resolved_id = version_id or str(uuid.uuid4())
        with self._write_scope():
            note = self._connection.execute(
                "SELECT session_id FROM amem_notes WHERE id = ?",
                (note_id,),
            ).fetchone()
            if note is None:
                raise KeyError(f"Unknown A-MEM note: {note_id}")
            if created_by_event_id is not None:
                event = self._connection.execute(
                    """
                    SELECT session_id FROM amem_evolution_events WHERE id = ?
                    """,
                    (created_by_event_id,),
                ).fetchone()
                if event is None:
                    raise KeyError(
                        f"Unknown A-MEM evolution event: {created_by_event_id}"
                    )
                if str(event["session_id"]) != str(note["session_id"]):
                    raise ValueError(
                        "A-MEM version event belongs to another session"
                    )
            latest = self._connection.execute(
                """
                SELECT * FROM amem_note_versions
                WHERE note_id = ? AND status = 'active'
                """,
                (note_id,),
            ).fetchone()
            version = int(latest["version"]) + 1 if latest else 1
            supersedes_id = str(latest["id"]) if latest else None
            if latest:
                self._connection.execute(
                    """
                    UPDATE amem_note_versions
                    SET status = 'superseded'
                    WHERE id = ? AND status = 'active'
                    """,
                    (supersedes_id,),
                )
            self._connection.execute(
                """
                INSERT INTO amem_note_versions(
                    id, note_id, version, context, keywords_json, tags_json,
                    status, supersedes_id, created_by_event_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
                """,
                (
                    resolved_id,
                    note_id,
                    version,
                    redact_secrets(context),
                    _json(list(keywords)),
                    _json(list(tags)),
                    supersedes_id,
                    created_by_event_id,
                    utc_now(),
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM amem_note_versions WHERE id = ?",
                (resolved_id,),
            ).fetchone()
        assert row is not None
        return self._decode_amem_note_version(row)

    def latest_amem_note_version(
        self,
        note_id: str,
    ) -> AMemNoteVersion | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM amem_note_versions
                WHERE note_id = ? AND status = 'active'
                ORDER BY version DESC LIMIT 1
                """,
                (note_id,),
            ).fetchone()
        return self._decode_amem_note_version(row) if row else None

    def amem_note_versions(
        self,
        note_id: str,
    ) -> tuple[AMemNoteVersion, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM amem_note_versions
                WHERE note_id = ? ORDER BY version, created_at, id
                """,
                (note_id,),
            ).fetchall()
        return tuple(self._decode_amem_note_version(row) for row in rows)

    def store_amem_note_embedding(
        self,
        *,
        note_version_id: str,
        model_id: str,
        dimensions: int,
        vector: Sequence[float],
    ) -> None:
        if dimensions < 1 or len(vector) != dimensions:
            raise ValueError("Invalid A-MEM embedding dimensions")
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO amem_note_embeddings(
                    note_version_id, model_id, dimensions, vector_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(note_version_id, model_id, dimensions)
                DO UPDATE SET vector_json = excluded.vector_json,
                              created_at = excluded.created_at
                """,
                (
                    note_version_id,
                    model_id,
                    dimensions,
                    _json([float(value) for value in vector]),
                    utc_now(),
                ),
            )

    def amem_note_embeddings(
        self,
        note_version_ids: Sequence[str],
        *,
        model_id: str,
        dimensions: int,
    ) -> dict[str, tuple[float, ...]]:
        normalized = tuple(dict.fromkeys(note_version_ids))
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT note_version_id, vector_json
                FROM amem_note_embeddings
                WHERE note_version_id IN ({placeholders})
                  AND model_id = ? AND dimensions = ?
                """,
                (*normalized, model_id, dimensions),
            ).fetchall()
        return {
            str(row["note_version_id"]): tuple(
                float(value) for value in json.loads(row["vector_json"])
            )
            for row in rows
        }

    def insert_amem_link(
        self,
        *,
        session_id: str,
        note_a_id: str,
        note_b_id: str,
        created_by_note_id: str,
        decision: str,
        evolution_event_id: str | None = None,
        similarity_score: float | None = None,
        link_id: str | None = None,
    ) -> AMemLink:
        if note_a_id == note_b_id:
            raise ValueError("A-MEM self-links are not allowed")
        left_note_id, right_note_id = sorted((note_a_id, note_b_id))
        if created_by_note_id not in {left_note_id, right_note_id}:
            raise ValueError("A-MEM link creator must be one of its endpoints")
        with self._write_scope():
            rows = self._connection.execute(
                """
                SELECT id, session_id FROM amem_notes WHERE id IN (?, ?)
                """,
                (left_note_id, right_note_id),
            ).fetchall()
            if len(rows) != 2:
                raise KeyError("A-MEM link endpoint not found")
            if any(str(row["session_id"]) != session_id for row in rows):
                raise ValueError("A-MEM links cannot cross sessions")
            if evolution_event_id is not None:
                event = self._connection.execute(
                    """
                    SELECT session_id FROM amem_evolution_events WHERE id = ?
                    """,
                    (evolution_event_id,),
                ).fetchone()
                if event is None:
                    raise KeyError(
                        f"Unknown A-MEM evolution event: {evolution_event_id}"
                    )
                if str(event["session_id"]) != session_id:
                    raise ValueError("A-MEM link event belongs to another session")
            resolved_id = link_id or str(uuid.uuid4())
            self._connection.execute(
                """
                INSERT INTO amem_links(
                    id, session_id, left_note_id, right_note_id,
                    created_by_note_id, evolution_event_id, similarity_score,
                    decision, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    session_id,
                    left_note_id,
                    right_note_id,
                    created_by_note_id,
                    evolution_event_id,
                    similarity_score,
                    decision,
                    utc_now(),
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM amem_links WHERE id = ?",
                (resolved_id,),
            ).fetchone()
        assert row is not None
        return self._decode_amem_link(row)

    def list_amem_links(
        self,
        session_id: str,
        *,
        note_id: str | None = None,
    ) -> tuple[AMemLink, ...]:
        note_clause = (
            "AND (left_note_id = ? OR right_note_id = ?)" if note_id else ""
        )
        parameters: tuple[Any, ...] = (
            (session_id, note_id, note_id) if note_id else (session_id,)
        )
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM amem_links
                WHERE session_id = ? {note_clause}
                ORDER BY created_at, left_note_id, right_note_id
                """,
                parameters,
            ).fetchall()
        return tuple(self._decode_amem_link(row) for row in rows)

    def amem_graph_fingerprint(self, session_id: str) -> str:
        """Hash graph content while deliberately excluding retrieval traces."""

        self.require_session(session_id)
        queries = (
            (
                "notes",
                """
                SELECT * FROM amem_notes
                WHERE session_id = ? ORDER BY id
                """,
            ),
            (
                "versions",
                """
                SELECT versions.*
                FROM amem_note_versions AS versions
                JOIN amem_notes AS notes ON notes.id = versions.note_id
                WHERE notes.session_id = ?
                ORDER BY versions.note_id, versions.version, versions.id
                """,
            ),
            (
                "embeddings",
                """
                SELECT embeddings.*
                FROM amem_note_embeddings AS embeddings
                JOIN amem_note_versions AS versions
                  ON versions.id = embeddings.note_version_id
                JOIN amem_notes AS notes ON notes.id = versions.note_id
                WHERE notes.session_id = ?
                ORDER BY embeddings.note_version_id,
                         embeddings.model_id,
                         embeddings.dimensions
                """,
            ),
            (
                "links",
                """
                SELECT * FROM amem_links
                WHERE session_id = ?
                ORDER BY left_note_id, right_note_id, id
                """,
            ),
        )
        with self._lock:
            payload = {
                name: [
                    dict(row)
                    for row in self._connection.execute(
                        statement,
                        (session_id,),
                    ).fetchall()
                ]
                for name, statement in queries
            }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def amem_metrics(self, session_id: str) -> dict[str, Any]:
        """Return persisted A-MEM graph, generation, and retrieval usage."""

        self.require_session(session_id)
        with self._lock:
            counts = {
                "note_count": self._connection.execute(
                    "SELECT COUNT(*) FROM amem_notes WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0],
                "version_count": self._connection.execute(
                    """
                    SELECT COUNT(*) FROM amem_note_versions AS versions
                    JOIN amem_notes AS notes ON notes.id = versions.note_id
                    WHERE notes.session_id = ?
                    """,
                    (session_id,),
                ).fetchone()[0],
                "embedding_count": self._connection.execute(
                    """
                    SELECT COUNT(*) FROM amem_note_embeddings AS embeddings
                    JOIN amem_note_versions AS versions
                      ON versions.id = embeddings.note_version_id
                    JOIN amem_notes AS notes ON notes.id = versions.note_id
                    WHERE notes.session_id = ?
                    """,
                    (session_id,),
                ).fetchone()[0],
                "link_count": self._connection.execute(
                    "SELECT COUNT(*) FROM amem_links WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0],
            }
            construction = self._connection.execute(
                """
                SELECT status, usage_json FROM amem_construction_runs
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
            evolution = self._connection.execute(
                """
                SELECT status, usage_json FROM amem_evolution_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
            compaction = self._connection.execute(
                """
                SELECT status, usage_json FROM compact_amem_episode_runs
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
            retrieval = self._connection.execute(
                """
                SELECT status, candidate_count, selected_count,
                       selected_tokens, latency_ms
                FROM amem_retrieval_runs WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()

        def status_counts(rows: Sequence[sqlite3.Row]) -> dict[str, int]:
            result: dict[str, int] = {}
            for row in rows:
                status = str(row["status"])
                result[status] = result.get(status, 0) + 1
            return dict(sorted(result.items()))

        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
        for row in (*construction, *compaction, *evolution):
            values = json.loads(row["usage_json"] or "{}")
            for key in usage:
                usage[key] += int(values.get(key, 0) or 0)
        provider_calls = [
            call
            for call in self.model_calls_for_sessions((session_id,))
            if str(call.get("role", "")).startswith("amem_")
        ]
        provider_roles: dict[str, int] = {}
        provider_costs: dict[str, float] = {}
        provider_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
        estimated_cost_usd = 0.0
        privacy = {
            "detected_count": 0,
            "redacted_count": 0,
            "sensitive_chars_before": 0,
            "sensitive_chars_after": 0,
        }
        for call in provider_calls:
            role = str(call.get("role", "unknown"))
            provider_roles[role] = provider_roles.get(role, 0) + 1
            call_usage = dict(call.get("usage") or {})
            for key in provider_usage:
                provider_usage[key] += int(call_usage.get(key, 0) or 0)
            metadata = dict(call.get("metadata") or {})
            call_cost = float(
                metadata.get("estimated_cost_usd", 0.0) or 0.0
            )
            estimated_cost_usd += call_cost
            provider_costs[role] = provider_costs.get(role, 0.0) + call_cost
            call_privacy = dict(metadata.get("privacy") or {})
            for key in privacy:
                privacy[key] += int(call_privacy.get(key, 0) or 0)
        return {
            **{key: int(value) for key, value in counts.items()},
            "generation": {
                "construction": status_counts(construction),
                "compaction": status_counts(compaction),
                "evolution": status_counts(evolution),
                "call_count": len(construction) + len(compaction) + len(evolution),
                "usage": usage,
            },
            "retrieval": {
                "runs": status_counts(retrieval),
                "run_count": len(retrieval),
                "candidate_count": sum(
                    int(row["candidate_count"] or 0) for row in retrieval
                ),
                "selected_count": sum(
                    int(row["selected_count"] or 0) for row in retrieval
                ),
                "selected_tokens": sum(
                    int(row["selected_tokens"] or 0) for row in retrieval
                ),
                "latency_ms": sum(
                    int(row["latency_ms"] or 0) for row in retrieval
                ),
            },
            "provider": {
                "call_count": len(provider_calls),
                "failed_call_count": sum(
                    bool(call.get("error")) for call in provider_calls
                ),
                "roles": dict(sorted(provider_roles.items())),
                "costs_by_role_usd": {
                    role: round(value, 8)
                    for role, value in sorted(provider_costs.items())
                },
                "latency_ms": sum(
                    int(call.get("latency_ms", 0) or 0)
                    for call in provider_calls
                ),
                "usage": provider_usage,
                "estimated_cost_usd": round(estimated_cost_usd, 8),
                "privacy": privacy,
            },
        }

    def begin_amem_construction(
        self,
        *,
        session_id: str,
        source_message_id: int,
        backend: str,
        model_id: str,
        prompt_version: str,
        schema_version: str,
        idempotency_key: str,
    ) -> str:
        self.require_session(session_id)
        run_id = str(uuid.uuid4())
        with self._write_scope():
            now = utc_now()
            self._connection.execute(
                """
                UPDATE amem_construction_runs
                SET status = 'interrupted',
                    error = 'Superseded by a resumed A-MEM construction',
                    ended_at = ?
                WHERE session_id = ? AND source_message_id = ?
                  AND status = 'running'
                """,
                (now, session_id, source_message_id),
            )
            self._connection.execute(
                """
                INSERT INTO amem_construction_runs(
                    id, session_id, source_message_id, backend, model_id,
                    prompt_version, schema_version, idempotency_key, status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    session_id,
                    source_message_id,
                    backend,
                    model_id,
                    prompt_version,
                    schema_version,
                    idempotency_key,
                    now,
                ),
            )
        return run_id

    def complete_amem_construction(
        self,
        run_id: str,
        *,
        note_id: str,
        output: Mapping[str, Any],
        usage: ModelUsage,
    ) -> None:
        with self._write_scope():
            run_note = self._connection.execute(
                """
                SELECT runs.session_id AS run_session_id,
                       notes.session_id AS note_session_id
                FROM amem_construction_runs AS runs
                JOIN amem_notes AS notes ON notes.id = ?
                WHERE runs.id = ? AND runs.status = 'running'
                """,
                (note_id, run_id),
            ).fetchone()
            if run_note is None:
                raise KeyError(
                    f"Running A-MEM construction or note not found: {run_id}"
                )
            if run_note["run_session_id"] != run_note["note_session_id"]:
                raise ValueError("A-MEM construction note belongs to another session")
            cursor = self._connection.execute(
                """
                UPDATE amem_construction_runs
                SET status = 'completed', note_id = ?, output_json = ?,
                    usage_json = ?, ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    note_id,
                    _json(dict(output)),
                    _json(usage.as_dict()),
                    utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running A-MEM construction not found: {run_id}")

    def fail_amem_construction(self, run_id: str, *, error: str) -> None:
        self._fail_amem_run("amem_construction_runs", run_id, error)

    def begin_amem_evolution(
        self,
        *,
        session_id: str,
        new_note_id: str,
        backend: str,
        model_id: str,
        prompt_version: str,
        schema_version: str,
        idempotency_key: str,
    ) -> str:
        note = self.get_amem_note(new_note_id)
        if note is None:
            raise KeyError(f"Unknown A-MEM note: {new_note_id}")
        if note.session_id != session_id:
            raise ValueError("A-MEM evolution note belongs to another session")
        event_id = str(uuid.uuid4())
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO amem_evolution_events(
                    id, session_id, new_note_id, backend, model_id,
                    prompt_version, schema_version, idempotency_key, status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    event_id,
                    session_id,
                    new_note_id,
                    backend,
                    model_id,
                    prompt_version,
                    schema_version,
                    idempotency_key,
                    utc_now(),
                ),
            )
        return event_id

    def complete_amem_evolution(
        self,
        event_id: str,
        *,
        decision: Mapping[str, Any],
        usage: ModelUsage,
    ) -> None:
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE amem_evolution_events
                SET status = 'completed', decision_json = ?, usage_json = ?,
                    ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (_json(dict(decision)), _json(usage.as_dict()), utc_now(), event_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running A-MEM evolution not found: {event_id}")

    def fail_amem_evolution(self, event_id: str, *, error: str) -> None:
        self._fail_amem_run("amem_evolution_events", event_id, error)

    def begin_amem_retrieval(
        self,
        *,
        session_id: str,
        query: str,
        mode: str,
        top_k: int,
        token_budget: int,
        embedding_model_id: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> str:
        self.require_session(session_id)
        if top_k < 1 or token_budget < 0:
            raise ValueError("Invalid A-MEM retrieval limits")
        if (embedding_model_id is None) != (embedding_dimensions is None):
            raise ValueError("A-MEM embedding model and dimensions must be paired")
        run_id = str(uuid.uuid4())
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO amem_retrieval_runs(
                    id, session_id, query, mode, top_k, token_budget,
                    embedding_model_id, embedding_dimensions, status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    session_id,
                    redact_secrets(query),
                    mode,
                    top_k,
                    token_budget,
                    embedding_model_id,
                    embedding_dimensions,
                    utc_now(),
                ),
            )
        return run_id

    def complete_amem_retrieval(
        self,
        run_id: str,
        candidates: Sequence[Mapping[str, Any]],
        *,
        selected_count: int,
        selected_tokens: int,
        latency_ms: int,
    ) -> None:
        with self._write_scope():
            run = self._connection.execute(
                """
                SELECT session_id FROM amem_retrieval_runs
                WHERE id = ? AND status = 'running'
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Running A-MEM retrieval not found: {run_id}")
            for candidate in candidates:
                note = self._connection.execute(
                    """
                    SELECT amem_notes.session_id
                    FROM amem_notes
                    JOIN amem_note_versions
                      ON amem_note_versions.note_id = amem_notes.id
                     AND amem_note_versions.id = ?
                    WHERE amem_notes.id = ?
                    """,
                    (candidate["note_version_id"], candidate["note_id"]),
                ).fetchone()
                if note is None or str(note["session_id"]) != str(run["session_id"]):
                    raise ValueError(
                        "A-MEM retrieval candidate/version is outside the session"
                    )
                self._connection.execute(
                    """
                    INSERT INTO amem_retrieval_candidates(
                        retrieval_run_id, note_id, note_version_id, source,
                        rank, embedding_score, combined_score, selected,
                        exclusion_reason, token_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        candidate["note_id"],
                        candidate["note_version_id"],
                        candidate.get("source", "seed"),
                        candidate.get("rank"),
                        candidate.get("embedding_score", 0.0),
                        candidate.get("combined_score", 0.0),
                        int(bool(candidate.get("selected"))),
                        candidate.get("exclusion_reason"),
                        candidate.get("token_count", 0),
                    ),
                )
            cursor = self._connection.execute(
                """
                UPDATE amem_retrieval_runs
                SET status = 'completed', candidate_count = ?,
                    selected_count = ?, selected_tokens = ?, latency_ms = ?,
                    ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    len(candidates),
                    selected_count,
                    selected_tokens,
                    latency_ms,
                    utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running A-MEM retrieval not found: {run_id}")

    def fail_amem_retrieval(
        self,
        run_id: str,
        *,
        error: str,
        latency_ms: int = 0,
    ) -> None:
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE amem_retrieval_runs
                SET status = 'failed', candidate_count = 0,
                    selected_count = 0, selected_tokens = 0, latency_ms = ?,
                    error = ?, ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (latency_ms, redact_secrets(error), utc_now(), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running A-MEM retrieval not found: {run_id}")

    def amem_retrieval_trace(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._connection.execute(
                "SELECT * FROM amem_retrieval_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown A-MEM retrieval: {run_id}")
            candidates = self._connection.execute(
                """
                SELECT amem_retrieval_candidates.*, amem_notes.content,
                       amem_notes.timestamp, amem_notes.speaker,
                       amem_note_versions.context,
                       amem_note_versions.keywords_json,
                       amem_note_versions.tags_json
                FROM amem_retrieval_candidates
                JOIN amem_notes
                  ON amem_notes.id = amem_retrieval_candidates.note_id
                JOIN amem_note_versions
                  ON amem_note_versions.id =
                     amem_retrieval_candidates.note_version_id
                WHERE retrieval_run_id = ?
                ORDER BY selected DESC, rank, note_id
                """,
                (run_id,),
            ).fetchall()
        return {
            "run": dict(run),
            "candidates": [
                self._decode_json_columns(
                    dict(row),
                    ("keywords_json", "tags_json"),
                )
                for row in candidates
            ],
        }

    def amem_construction_trace(self, run_id: str) -> dict[str, Any]:
        return self._amem_run_trace("amem_construction_runs", run_id)

    def amem_evolution_trace(self, event_id: str) -> dict[str, Any]:
        return self._amem_run_trace("amem_evolution_events", event_id)

    def compact_amem_episode_trace(self, run_id: str) -> dict[str, Any]:
        return self._amem_run_trace("compact_amem_episode_runs", run_id)

    def _amem_run_trace(self, table: str, run_id: str) -> dict[str, Any]:
        json_columns = {
            "amem_construction_runs": ("output_json", "usage_json"),
            "amem_evolution_events": ("decision_json", "usage_json"),
            "compact_amem_episode_runs": (
                "source_message_ids_json",
                "output_json",
                "usage_json",
            ),
        }
        if table not in json_columns:
            raise ValueError(f"Unsupported A-MEM run table: {table}")
        with self._lock:
            row = self._connection.execute(
                f"SELECT * FROM {table} WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown A-MEM run: {run_id}")
        return self._decode_json_columns(dict(row), json_columns[table])

    def _fail_amem_run(self, table: str, run_id: str, error: str) -> None:
        if table not in {
            "amem_construction_runs",
            "amem_evolution_events",
            "compact_amem_episode_runs",
        }:
            raise ValueError(f"Unsupported A-MEM run table: {table}")
        with self._write_scope():
            cursor = self._connection.execute(
                f"""
                UPDATE {table}
                SET status = 'failed', error = ?, ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (redact_secrets(error), utc_now(), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running A-MEM run not found: {run_id}")

    @staticmethod
    def _decode_amem_note(row: sqlite3.Row) -> AMemNote:
        return AMemNote(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            source_message_id=int(row["source_message_id"]),
            timestamp=str(row["timestamp"]),
            speaker=str(row["speaker"]),
            content=str(row["content"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _decode_amem_note_version(row: sqlite3.Row) -> AMemNoteVersion:
        return AMemNoteVersion(
            id=str(row["id"]),
            note_id=str(row["note_id"]),
            version=int(row["version"]),
            context=str(row["context"]),
            keywords=tuple(json.loads(row["keywords_json"])),
            tags=tuple(json.loads(row["tags_json"])),
            status=str(row["status"]),
            supersedes_id=(
                str(row["supersedes_id"])
                if row["supersedes_id"] is not None
                else None
            ),
            created_by_event_id=(
                str(row["created_by_event_id"])
                if row["created_by_event_id"] is not None
                else None
            ),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _decode_amem_link(row: sqlite3.Row) -> AMemLink:
        return AMemLink(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            left_note_id=str(row["left_note_id"]),
            right_note_id=str(row["right_note_id"]),
            created_by_note_id=str(row["created_by_note_id"]),
            evolution_event_id=(
                str(row["evolution_event_id"])
                if row["evolution_event_id"] is not None
                else None
            ),
            similarity_score=(
                float(row["similarity_score"])
                if row["similarity_score"] is not None
                else None
            ),
            decision=str(row["decision"]),
            created_at=str(row["created_at"]),
        )

    def active_versioned_fact_memory_record(
        self,
        record_key: str,
    ) -> FactMemoryRecord | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM fact_memory_records
                WHERE record_key = ? AND status = 'active'
                LIMIT 1
                """,
                (record_key,),
            ).fetchone()
        return self._decode_fact_memory_record(row) if row else None

    def fact_memory_record(self, record_id: str) -> FactMemoryRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM fact_memory_records WHERE id = ?",
                (record_id,),
            ).fetchone()
        return self._decode_fact_memory_record(row) if row else None

    def fact_memory_record_versions(
        self,
        record_key: str,
    ) -> tuple[FactMemoryRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM fact_memory_records
                WHERE record_key = ?
                ORDER BY version, created_at
                """,
                (record_key,),
            ).fetchall()
        return tuple(self._decode_fact_memory_record(row) for row in rows)

    def fact_memory_record_sources(
        self,
        record_id: str,
    ) -> tuple[MemoryEvidence, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT message_id, evidence_text, start_char, end_char
                FROM fact_memory_record_sources
                WHERE record_id = ?
                ORDER BY message_id, start_char, evidence_text
                """,
                (record_id,),
            ).fetchall()
        return tuple(
            MemoryEvidence(
                message_id=int(row["message_id"]),
                quote=str(row["evidence_text"]),
                start_char=row["start_char"],
                end_char=row["end_char"],
            )
            for row in rows
        )

    def next_fact_memory_version(self, record_key: str) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) AS version
                FROM fact_memory_records
                WHERE record_key = ?
                """,
                (record_key,),
            ).fetchone()
        return int(row["version"]) + 1

    def transition_fact_memory_record(
        self,
        record_id: str,
        *,
        status: str,
        reason: str,
        patch_event_id: str,
        merged_into_id: str | None = None,
    ) -> FactMemoryRecord:
        if status not in {"superseded", "merged", "deleted", "review"}:
            raise ValueError(f"Unsupported fact record transition: {status}")
        now = utc_now()
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE fact_memory_records
                SET status = ?, merged_into_id = ?, updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (status, merged_into_id, now, record_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Fact memory record is not active: {record_id}")
            self._connection.execute(
                """
                INSERT INTO fact_memory_record_status_events(
                    record_id,
                    status,
                    reason,
                    patch_event_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    status,
                    redact_secrets(reason),
                    patch_event_id,
                    now,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM fact_memory_records WHERE id = ?",
                (record_id,),
            ).fetchone()
        assert row is not None
        return self._decode_fact_memory_record(row)

    def link_merged_fact_memory_record(
        self,
        record_id: str,
        *,
        merged_into_id: str,
    ) -> None:
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE fact_memory_records
                SET merged_into_id = ?, updated_at = ?
                WHERE id = ? AND status = 'merged'
                """,
                (merged_into_id, utc_now(), record_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Fact memory record is not merged: {record_id}")

    def begin_fact_memory_patch_event(
        self,
        *,
        event_id: str,
        session_id: str,
        operation: str,
        patch: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        confidence: float,
        idempotency_key: str,
        target_record_id: str | None,
        merge_record_ids: Sequence[str],
        proposed_record_key: str | None,
    ) -> str:
        self.require_session(session_id)
        now = utc_now()
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO fact_memory_patch_events(
                    id,
                    session_id,
                    operation,
                    target_record_id,
                    merge_record_ids_json,
                    proposed_record_key,
                    patch_json,
                    evidence_json,
                    confidence,
                    status,
                    idempotency_key,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    operation,
                    target_record_id,
                    _json(tuple(merge_record_ids)),
                    proposed_record_key,
                    _json(dict(patch)),
                    _json(tuple(evidence)),
                    confidence,
                    _idempotency_digest(idempotency_key),
                    now,
                ),
            )
        return event_id

    def resolve_fact_memory_patch_event(
        self,
        event_id: str,
        *,
        status: str,
        result_record_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        if status not in {"applied", "rejected", "review"}:
            raise ValueError(f"Unsupported fact patch event status: {status}")
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE fact_memory_patch_events
                SET status = ?,
                    result_record_id = ?,
                    reason = ?,
                    resolved_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (
                    status,
                    result_record_id,
                    redact_secrets(reason or ""),
                    utc_now(),
                    event_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Fact patch event is not pending: {event_id}")

    def fact_memory_patch_event_by_idempotency(
        self,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM fact_memory_patch_events
                WHERE idempotency_key = ?
                """,
                (_idempotency_digest(idempotency_key),),
            ).fetchone()
        if row is None:
            return None
        return self._decode_json_columns(
            dict(row),
            ("merge_record_ids_json", "patch_json", "evidence_json"),
        )

    def fact_memory_record_detail(self, record_id: str) -> dict[str, Any]:
        record = self.fact_memory_record(record_id)
        if record is None:
            raise KeyError(f"Unknown fact memory record: {record_id}")
        with self._lock:
            events = self._connection.execute(
                """
                SELECT status, reason, patch_event_id, created_at
                FROM fact_memory_record_status_events
                WHERE record_id = ?
                ORDER BY id
                """,
                (record_id,),
            ).fetchall()
        return {
            "record": record,
            "sources": self.fact_memory_record_sources(record_id),
            "status_events": [dict(row) for row in events],
        }

    def record_fact_memory_candidate_event(
        self,
        *,
        run_id: str,
        sequence_index: int,
        candidate: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        decision: str,
        status: str,
        idempotency_key: str,
        target_record_id: str | None = None,
        merge_record_ids: Sequence[str] = (),
        result_record_id: str | None = None,
        match_score: float | None = None,
        reason: str = "",
        validation: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> str:
        if decision not in {
            "ADD",
            "UPDATE",
            "MERGE",
            "DELETE",
            "NOOP",
            "REVIEW",
            "REJECT",
        }:
            raise ValueError(f"Unsupported Fact link decision: {decision}")
        if status not in {"applied", "noop", "review", "rejected"}:
            raise ValueError(f"Unsupported Fact candidate status: {status}")
        if sequence_index < 0:
            raise ValueError("Fact candidate sequence_index cannot be negative")
        self.memory_patch_run(run_id)
        resolved_id = event_id or str(uuid.uuid4())
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO fact_memory_candidate_events(
                    id,
                    memory_patch_run_id,
                    sequence_index,
                    candidate_json,
                    evidence_json,
                    decision,
                    status,
                    target_record_id,
                    merge_record_ids_json,
                    result_record_id,
                    match_score,
                    reason,
                    validation_json,
                    idempotency_key,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    run_id,
                    sequence_index,
                    _json(dict(candidate)),
                    _json(tuple(evidence)),
                    decision,
                    status,
                    target_record_id,
                    _json(tuple(merge_record_ids)),
                    result_record_id,
                    match_score,
                    redact_secrets(reason),
                    _json(dict(validation or {})),
                    _idempotency_digest(idempotency_key),
                    utc_now(),
                ),
            )
        return resolved_id

    def fact_memory_candidate_events(
        self,
        run_id: str,
    ) -> list[dict[str, Any]]:
        self.memory_patch_run(run_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM fact_memory_candidate_events
                WHERE memory_patch_run_id = ?
                ORDER BY sequence_index
                """,
                (run_id,),
            ).fetchall()
        return [
            self._decode_json_columns(
                dict(row),
                (
                    "candidate_json",
                    "evidence_json",
                    "merge_record_ids_json",
                    "validation_json",
                ),
            )
            for row in rows
        ]

    def fact_memory_candidate_events_for_session(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        self.require_session(session_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT fact_memory_candidate_events.*
                FROM fact_memory_candidate_events
                JOIN memory_patch_runs
                  ON memory_patch_runs.id =
                     fact_memory_candidate_events.memory_patch_run_id
                WHERE memory_patch_runs.session_id = ?
                ORDER BY
                    memory_patch_runs.started_at,
                    fact_memory_candidate_events.sequence_index
                """,
                (session_id,),
            ).fetchall()
        return [
            self._decode_json_columns(
                dict(row),
                (
                    "candidate_json",
                    "evidence_json",
                    "merge_record_ids_json",
                    "validation_json",
                ),
            )
            for row in rows
        ]

    def fact_memory_metrics(self, session_id: str) -> dict[str, Any]:
        self.require_session(session_id)
        with self._lock:
            event_rows = self._connection.execute(
                """
                SELECT
                    fact_memory_candidate_events.decision,
                    fact_memory_candidate_events.status,
                    fact_memory_candidate_events.validation_json
                FROM fact_memory_candidate_events
                JOIN memory_patch_runs
                  ON memory_patch_runs.id =
                     fact_memory_candidate_events.memory_patch_run_id
                WHERE memory_patch_runs.session_id = ?
                """,
                (session_id,),
            ).fetchall()
            record_rows = self._connection.execute(
                """
                SELECT status, version
                FROM fact_memory_records
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
            model_rows = self._connection.execute(
                """
                SELECT role, usage_json
                FROM model_calls
                WHERE session_id = ?
                  AND role IN (
                      'fact_memory_extraction',
                      'fact_memory_semantic_validation'
                  )
                """,
                (session_id,),
            ).fetchall()
        decision_counts: dict[str, int] = {}
        candidate_status_counts: dict[str, int] = {}
        validation_code_counts: dict[str, int] = {}
        for row in event_rows:
            decision = str(row["decision"])
            status = str(row["status"])
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            candidate_status_counts[status] = (
                candidate_status_counts.get(status, 0) + 1
            )
            validation = json.loads(row["validation_json"] or "{}")
            code = str(validation.get("code", "")).strip()
            if code:
                validation_code_counts[code] = (
                    validation_code_counts.get(code, 0) + 1
                )
        record_status_counts: dict[str, int] = {}
        versioned_record_count = 0
        for row in record_rows:
            status = str(row["status"])
            record_status_counts[status] = (
                record_status_counts.get(status, 0) + 1
            )
            if int(row["version"]) > 1:
                versioned_record_count += 1
        extraction_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
        semantic_usage = dict(extraction_usage)
        extraction_call_count = 0
        semantic_call_count = 0
        for row in model_rows:
            item = json.loads(row["usage_json"] or "{}")
            if row["role"] == "fact_memory_semantic_validation":
                target = semantic_usage
                semantic_call_count += 1
            else:
                target = extraction_usage
                extraction_call_count += 1
            for key in target:
                target[key] += int(item.get(key, 0))
        return {
            "candidate_count": len(event_rows),
            "decision_counts": decision_counts,
            "candidate_status_counts": candidate_status_counts,
            "validation_code_counts": validation_code_counts,
            "record_count": len(record_rows),
            "active_record_count": record_status_counts.get("active", 0),
            "record_status_counts": record_status_counts,
            "versioned_record_count": versioned_record_count,
            "model_call_count": extraction_call_count,
            "model_usage": extraction_usage,
            "semantic_model_call_count": semantic_call_count,
            "semantic_model_usage": semantic_usage,
        }

    def enqueue_memory_patch_job(
        self,
        *,
        session_id: str,
        source_turn_id: str,
        max_attempts: int,
    ) -> str:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        with self._write_scope():
            turn = self._connection.execute(
                """
                SELECT session_id, status
                FROM turns
                WHERE id = ?
                """,
                (source_turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"Unknown source turn: {source_turn_id}")
            if str(turn["session_id"]) != session_id:
                raise ValueError("Patch job turn must belong to its session")
            if str(turn["status"]) != "completed":
                raise ValueError("Only completed turns can enter the patch queue")
            cursor_row = self._connection.execute(
                """
                SELECT COALESCE(MAX(id), 0) AS cursor
                FROM messages
                WHERE turn_id = ?
                """,
                (source_turn_id,),
            ).fetchone()
            source_cursor = int(cursor_row["cursor"])
            if source_cursor < 1:
                raise ValueError("Patch job source turn has no messages")
            existing = self._connection.execute(
                """
                SELECT id
                FROM memory_patch_jobs
                WHERE source_turn_id = ?
                """,
                (source_turn_id,),
            ).fetchone()
            if existing is not None:
                return str(existing["id"])
            job_id = str(uuid.uuid4())
            now = utc_now()
            self._connection.execute(
                """
                INSERT INTO memory_patch_jobs(
                    id,
                    session_id,
                    source_turn_id,
                    source_message_cursor,
                    status,
                    attempt_count,
                    max_attempts,
                    available_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    session_id,
                    source_turn_id,
                    source_cursor,
                    max_attempts,
                    now,
                    now,
                    now,
                ),
            )
            return job_id

    def reset_memory_patch_jobs_for_replay(
        self,
        *,
        session_id: str,
        max_attempts: int,
    ) -> int:
        """Queue a completed session for a controlled supplemental pass."""

        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.require_session(session_id)
        now = utc_now()
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE memory_patch_jobs
                SET status = 'queued',
                    attempt_count = 0,
                    max_attempts = ?,
                    available_at = ?,
                    claimed_at = NULL,
                    lease_expires_at = NULL,
                    completed_at = NULL,
                    last_error = NULL,
                    updated_at = ?,
                    batch_run_id = NULL
                WHERE session_id = ?
                """,
                (max_attempts, now, now, session_id),
            )
        return int(cursor.rowcount)

    def claim_memory_patch_jobs(
        self,
        *,
        limit: int,
        lease_seconds: float,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("Patch job claim limit must be positive")
        if lease_seconds <= 0:
            raise ValueError("Patch job lease must be positive")
        now = utc_now()
        with self._write_scope():
            self._recover_expired_memory_patch_jobs(now)
            clauses = [
                "status IN ('queued', 'retryable')",
                "available_at <= ?",
                "attempt_count < max_attempts",
            ]
            parameters: list[Any] = [now]
            if session_id is not None:
                self.require_session(session_id)
                clauses.append("session_id = ?")
                parameters.append(session_id)
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM memory_patch_jobs
                WHERE {' AND '.join(clauses)}
                ORDER BY source_message_cursor, created_at, id
                LIMIT ?
                """,
                (*parameters, limit),
            ).fetchall()
            claimed: list[dict[str, Any]] = []
            for row in rows:
                cursor = self._connection.execute(
                    """
                    UPDATE memory_patch_jobs
                    SET status = 'running',
                        attempt_count = attempt_count + 1,
                        claimed_at = ?,
                        lease_expires_at = ?,
                        updated_at = ?
                    WHERE id = ?
                      AND status IN ('queued', 'retryable')
                    """,
                    (
                        now,
                        utc_after(lease_seconds),
                        now,
                        row["id"],
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                claimed_row = self._connection.execute(
                    "SELECT * FROM memory_patch_jobs WHERE id = ?",
                    (row["id"],),
                ).fetchone()
                claimed.append(dict(claimed_row))
        return claimed

    def complete_memory_patch_job(self, job_id: str) -> None:
        self.complete_memory_patch_jobs((job_id,))

    def complete_memory_patch_jobs(self, job_ids: Sequence[str]) -> None:
        normalized = tuple(dict.fromkeys(str(job_id) for job_id in job_ids))
        if not normalized:
            raise ValueError("At least one memory patch job is required")
        now = utc_now()
        with self._write_scope():
            placeholders = ",".join("?" for _ in normalized)
            cursor = self._connection.execute(
                f"""
                UPDATE memory_patch_jobs
                SET status = 'completed',
                    completed_at = ?,
                    lease_expires_at = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE id IN ({placeholders}) AND status = 'running'
                """,
                (now, now, *normalized),
            )
            if cursor.rowcount != len(normalized):
                rows = self._connection.execute(
                    f"""
                    SELECT id, status
                    FROM memory_patch_jobs
                    WHERE id IN ({placeholders})
                    """,
                    normalized,
                ).fetchall()
                statuses = {str(row["id"]): str(row["status"]) for row in rows}
                invalid = [
                    job_id
                    for job_id in normalized
                    if statuses.get(job_id) != "completed"
                ]
                if invalid:
                    raise KeyError(
                        f"Running memory patch jobs not found: {invalid}"
                    )

    def link_memory_patch_jobs_to_batch_run(
        self,
        job_ids: Sequence[str],
        *,
        run_id: str,
        session_id: str,
    ) -> None:
        normalized = tuple(dict.fromkeys(str(job_id) for job_id in job_ids))
        if len(normalized) < 2:
            raise ValueError("A batch patch run requires at least two jobs")
        run = self.memory_patch_run(run_id)
        if (
            str(run["session_id"]) != session_id
            or run["source_turn_id"] is not None
            or run["job_id"] is not None
        ):
            raise ValueError("Batch patch run must be session-scoped")
        placeholders = ",".join("?" for _ in normalized)
        with self._write_scope():
            rows = self._connection.execute(
                f"""
                SELECT id, session_id, status, batch_run_id
                FROM memory_patch_jobs
                WHERE id IN ({placeholders})
                """,
                normalized,
            ).fetchall()
            by_id = {str(row["id"]): row for row in rows}
            invalid = [
                job_id
                for job_id in normalized
                if job_id not in by_id
                or str(by_id[job_id]["session_id"]) != session_id
                or str(by_id[job_id]["status"]) != "running"
            ]
            if invalid:
                raise ValueError(f"Invalid batch patch jobs: {invalid}")
            self._connection.execute(
                f"""
                UPDATE memory_patch_jobs
                SET batch_run_id = ?, updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (run_id, utc_now(), *normalized),
            )

    def retry_memory_patch_job(
        self,
        job_id: str,
        *,
        error: str,
        retry_delay_seconds: float,
    ) -> str:
        if retry_delay_seconds < 0:
            raise ValueError("Patch retry delay cannot be negative")
        now = utc_now()
        with self._write_scope():
            row = self._connection.execute(
                """
                SELECT attempt_count, max_attempts
                FROM memory_patch_jobs
                WHERE id = ? AND status = 'running'
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Running memory patch job not found: {job_id}")
            final = int(row["attempt_count"]) >= int(row["max_attempts"])
            status = "failed" if final else "retryable"
            self._connection.execute(
                """
                UPDATE memory_patch_jobs
                SET status = ?,
                    available_at = ?,
                    lease_expires_at = NULL,
                    completed_at = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    utc_after(retry_delay_seconds),
                    now if final else None,
                    redact_secrets(error),
                    now,
                    job_id,
                ),
            )
        return status

    def memory_patch_jobs(
        self,
        *,
        session_id: str | None = None,
        statuses: Sequence[str] = (),
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("Patch job list limit must be positive")
        allowed = {"queued", "running", "retryable", "completed", "failed"}
        invalid = set(statuses) - allowed
        if invalid:
            raise ValueError(f"Unsupported patch job statuses: {sorted(invalid)}")
        clauses: list[str] = []
        parameters: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            parameters.append(session_id)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(statuses)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM memory_patch_jobs
                {where}
                ORDER BY source_message_cursor, created_at, id
                LIMIT ?
                """,
                (*parameters, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def memory_patch_queue_status(
        self,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        clauses = "WHERE session_id = ?" if session_id is not None else ""
        parameters = (session_id,) if session_id is not None else ()
        with self._lock:
            counts = self._connection.execute(
                f"""
                SELECT status, COUNT(*) AS count
                FROM memory_patch_jobs
                {clauses}
                GROUP BY status
                """,
                parameters,
            ).fetchall()
            cursor_rows = self._connection.execute(
                f"""
                SELECT source_message_cursor, status
                FROM memory_patch_jobs
                {clauses}
                ORDER BY source_message_cursor, created_at, id
                """,
                parameters,
            ).fetchall()
        status_counts = {str(row["status"]): int(row["count"]) for row in counts}
        processed_cursor = 0
        for row in cursor_rows:
            if str(row["status"]) != "completed":
                break
            processed_cursor = int(row["source_message_cursor"])
        return {
            "status_counts": status_counts,
            "ready_count": (
                status_counts.get("queued", 0)
                + status_counts.get("retryable", 0)
            ),
            "processed_message_cursor": processed_cursor,
        }

    def memory_patch_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM memory_patch_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown memory patch job: {job_id}")
        return dict(row)

    def turn_memory_messages(self, turn_id: str) -> list[StoredMessage]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM messages
                WHERE turn_id = ?
                ORDER BY id
                """,
                (turn_id,),
            ).fetchall()
        return [self._decode_message(row) for row in rows]

    def _recover_expired_memory_patch_jobs(self, now: str) -> None:
        self._connection.execute(
            """
            UPDATE memory_patch_jobs
            SET status = 'completed',
                completed_at = ?,
                lease_expires_at = NULL,
                updated_at = ?
            WHERE status = 'running'
              AND lease_expires_at <= ?
              AND EXISTS (
                  SELECT 1
                  FROM memory_patch_runs
                  WHERE (
                        memory_patch_runs.job_id = memory_patch_jobs.id
                        OR memory_patch_runs.id =
                            memory_patch_jobs.batch_run_id
                    )
                    AND memory_patch_runs.status = 'completed'
              )
            """,
            (now, now, now),
        )
        self._connection.execute(
            """
            UPDATE memory_patch_jobs
            SET status = CASE
                    WHEN attempt_count >= max_attempts
                    THEN 'failed'
                    ELSE 'retryable'
                END,
                available_at = ?,
                lease_expires_at = NULL,
                last_error = 'Memory patch worker lease expired',
                completed_at = CASE
                    WHEN attempt_count >= max_attempts
                    THEN ?
                    ELSE completed_at
                END,
                updated_at = ?
            WHERE status = 'running'
              AND lease_expires_at <= ?
            """,
            (now, now, now, now),
        )

    def begin_memory_patch_run(
        self,
        *,
        session_id: str,
        source_turn_id: str | None,
        backend: str,
        model_id: str,
        prompt_version: str,
        schema_version: str,
        status: str = "queued",
        job_id: str | None = None,
    ) -> str:
        self.require_session(session_id)
        if status not in {"queued", "running"}:
            raise ValueError("A memory patch run must start queued or running")
        if source_turn_id is not None:
            with self._lock:
                row = self._connection.execute(
                    "SELECT session_id FROM turns WHERE id = ?",
                    (source_turn_id,),
                ).fetchone()
            if row is None:
                raise KeyError(f"Unknown source turn: {source_turn_id}")
            if str(row["session_id"]) != session_id:
                raise ValueError("Patch source turn must belong to the run session")
        if job_id is not None:
            job = self.memory_patch_job(job_id)
            if (
                str(job["session_id"]) != session_id
                or job["source_turn_id"] != source_turn_id
            ):
                raise ValueError("Patch run must match its queued job")
        run_id = str(uuid.uuid4())
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO memory_patch_runs(
                    id,
                    session_id,
                    source_turn_id,
                    backend,
                    model_id,
                    prompt_version,
                    schema_version,
                    job_id,
                    status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    source_turn_id,
                    redact_secrets(backend),
                    redact_secrets(model_id),
                    redact_secrets(prompt_version),
                    redact_secrets(schema_version),
                    job_id,
                    status,
                    utc_now(),
                ),
            )
        return run_id

    def start_memory_patch_run(self, run_id: str) -> None:
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE memory_patch_runs
                SET status = 'running'
                WHERE id = ? AND status = 'queued'
                """,
                (run_id,),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Queued memory patch run not found: {run_id}")

    def record_memory_patch_proposal(
        self,
        *,
        run_id: str,
        operation: str,
        patch: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
        confidence: float,
        idempotency_key: str,
        target_record_id: str | None = None,
        proposed_record_key: str | None = None,
        validation: Mapping[str, Any] | None = None,
        sequence_index: int | None = None,
    ) -> str:
        normalized_operation = operation.strip().upper()
        if normalized_operation not in {"ADD", "UPDATE", "MERGE", "DELETE"}:
            raise ValueError(f"Unsupported memory patch operation: {operation}")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        normalized_idempotency = idempotency_key.strip()
        if not normalized_idempotency:
            raise ValueError("idempotency_key cannot be empty")
        if sequence_index is not None and sequence_index < 0:
            raise ValueError("sequence_index cannot be negative")
        patch_json = _evaluation_json(dict(patch))
        evidence_json = _evaluation_json([dict(item) for item in evidence])
        validation_json = _evaluation_json(dict(validation or {}))
        with self._write_scope():
            existing = self._connection.execute(
                """
                SELECT *
                FROM memory_patch_proposals
                WHERE idempotency_key = ?
                """,
                (normalized_idempotency,),
            ).fetchone()
            if existing is not None:
                expected = (
                    run_id,
                    normalized_operation,
                    target_record_id,
                    proposed_record_key,
                    patch_json,
                    evidence_json,
                    confidence,
                    validation_json,
                    sequence_index,
                )
                actual = (
                    existing["run_id"],
                    existing["operation"],
                    existing["target_record_id"],
                    existing["proposed_record_key"],
                    existing["patch_json"],
                    existing["evidence_json"],
                    existing["confidence"],
                    existing["validation_json"],
                    existing["sequence_index"],
                )
                if actual != expected:
                    raise ValueError(
                        "idempotency_key already belongs to a different proposal"
                    )
                return str(existing["id"])
            proposal_id = str(uuid.uuid4())
            self._connection.execute(
                """
                INSERT INTO memory_patch_proposals(
                    id,
                    run_id,
                    operation,
                    target_record_id,
                    proposed_record_key,
                    patch_json,
                    evidence_json,
                    confidence,
                    status,
                    idempotency_key,
                    created_at,
                    validation_json,
                    sequence_index
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                """,
                (
                    proposal_id,
                    run_id,
                    normalized_operation,
                    target_record_id,
                    proposed_record_key,
                    patch_json,
                    evidence_json,
                    confidence,
                    normalized_idempotency,
                    utc_now(),
                    validation_json,
                    sequence_index,
                ),
            )
        return proposal_id

    def resolve_memory_patch_proposal(
        self,
        proposal_id: str,
        *,
        status: str,
        result_record_id: str | None = None,
        rejection_reason: str | None = None,
    ) -> None:
        if status not in {"applied", "rejected"}:
            raise ValueError("Patch proposal status must be applied or rejected")
        if status == "applied" and result_record_id is None:
            raise ValueError("Applied patch proposal requires a result record")
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE memory_patch_proposals
                SET status = ?,
                    result_record_id = ?,
                    rejection_reason = ?,
                    applied_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (
                    status,
                    result_record_id,
                    (
                        redact_secrets(rejection_reason)
                        if rejection_reason is not None
                        else None
                    ),
                    utc_now(),
                    proposal_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(
                    f"Pending memory patch proposal not found: {proposal_id}"
                )

    def finish_memory_patch_run(
        self,
        run_id: str,
        *,
        status: str,
        usage: ModelUsage | None = None,
        error: str | None = None,
    ) -> None:
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError("Unsupported final memory patch run status")
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE memory_patch_runs
                SET status = ?,
                    usage_json = ?,
                    error = ?,
                    ended_at = ?
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (
                    status,
                    _json((usage or ModelUsage()).as_dict()),
                    redact_secrets(error) if error else None,
                    utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Active memory patch run not found: {run_id}")

    def memory_patch_trace(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._connection.execute(
                "SELECT * FROM memory_patch_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown memory patch run: {run_id}")
            proposals = self._connection.execute(
                """
                SELECT *
                FROM memory_patch_proposals
                WHERE run_id = ?
                ORDER BY
                    sequence_index IS NULL,
                    sequence_index,
                    created_at,
                    id
                """,
                (run_id,),
            ).fetchall()
            model_calls = self._connection.execute(
                """
                SELECT *
                FROM model_calls
                WHERE memory_patch_run_id = ?
                ORDER BY created_at, id
                """,
                (run_id,),
            ).fetchall()
            job = (
                self._connection.execute(
                    "SELECT * FROM memory_patch_jobs WHERE id = ?",
                    (run["job_id"],),
                ).fetchone()
                if run["job_id"]
                else None
            )
            batch_jobs = self._connection.execute(
                """
                SELECT *
                FROM memory_patch_jobs
                WHERE batch_run_id = ?
                ORDER BY source_message_cursor, created_at, id
                """,
                (run_id,),
            ).fetchall()
            fact_candidates = self._connection.execute(
                """
                SELECT *
                FROM fact_memory_candidate_events
                WHERE memory_patch_run_id = ?
                ORDER BY sequence_index
                """,
                (run_id,),
            ).fetchall()
        return {
            "run": self._decode_json_columns(dict(run), ("usage_json",)),
            "job": dict(job) if job is not None else None,
            "jobs": [dict(row) for row in batch_jobs],
            "proposals": [
                self._decode_json_columns(
                    dict(row),
                    ("patch_json", "evidence_json", "validation_json"),
                )
                for row in proposals
            ],
            "fact_candidates": [
                self._decode_json_columns(
                    dict(row),
                    (
                        "candidate_json",
                        "evidence_json",
                        "merge_record_ids_json",
                        "validation_json",
                    ),
                )
                for row in fact_candidates
            ],
            "model_calls": [
                self._decode_json_columns(
                    dict(row),
                    ("usage_json", "metadata_json"),
                )
                for row in model_calls
            ],
        }

    def tool_memory_patch_metrics(self, session_id: str) -> dict[str, Any]:
        self.require_session(session_id)
        with self._lock:
            run_rows = self._connection.execute(
                """
                SELECT id, status
                FROM memory_patch_runs
                WHERE session_id = ?
                ORDER BY started_at, id
                """,
                (session_id,),
            ).fetchall()
            proposal_rows = self._connection.execute(
                """
                SELECT
                    memory_patch_proposals.operation,
                    memory_patch_proposals.status,
                    memory_patch_proposals.validation_json
                FROM memory_patch_proposals
                JOIN memory_patch_runs
                  ON memory_patch_runs.id = memory_patch_proposals.run_id
                WHERE memory_patch_runs.session_id = ?
                ORDER BY memory_patch_proposals.created_at,
                         memory_patch_proposals.id
                """,
                (session_id,),
            ).fetchall()
            record_rows = self._connection.execute(
                """
                SELECT status, version
                FROM tool_memory_records
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
            duplicate_row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM (
                    SELECT record_key
                    FROM tool_memory_records
                    WHERE session_id = ? AND status = 'active'
                    GROUP BY record_key
                    HAVING COUNT(*) > 1
                )
                """,
                (session_id,),
            ).fetchone()
            evidence_row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM tool_memory_record_sources
                JOIN tool_memory_records
                  ON tool_memory_records.id =
                     tool_memory_record_sources.record_id
                WHERE tool_memory_records.session_id = ?
                """,
                (session_id,),
            ).fetchone()

        run_status_counts: dict[str, int] = {}
        for row in run_rows:
            status = str(row["status"])
            run_status_counts[status] = run_status_counts.get(status, 0) + 1
        operation_counts: dict[str, int] = {}
        proposal_status_counts: dict[str, int] = {}
        validation_code_counts: dict[str, int] = {}
        evidence_accepted = 0
        for row in proposal_rows:
            operation = str(row["operation"])
            status = str(row["status"])
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
            proposal_status_counts[status] = (
                proposal_status_counts.get(status, 0) + 1
            )
            validation = json.loads(row["validation_json"] or "{}")
            if validation.get("accepted") is True:
                evidence_accepted += 1
            code = validation.get("code")
            if code:
                normalized = str(code)
                validation_code_counts[normalized] = (
                    validation_code_counts.get(normalized, 0) + 1
                )
        record_status_counts: dict[str, int] = {}
        versioned_records = 0
        for row in record_rows:
            status = str(row["status"])
            record_status_counts[status] = record_status_counts.get(status, 0) + 1
            versioned_records += int(int(row["version"]) > 1)
        proposal_count = len(proposal_rows)
        rejected_codes = tuple(validation_code_counts)
        return {
            "run_count": len(run_rows),
            "run_status_counts": dict(sorted(run_status_counts.items())),
            "latest_run_id": str(run_rows[-1]["id"]) if run_rows else None,
            "proposal_count": proposal_count,
            "proposal_status_counts": dict(
                sorted(proposal_status_counts.items())
            ),
            "operation_counts": dict(sorted(operation_counts.items())),
            "validation_code_counts": dict(
                sorted(validation_code_counts.items())
            ),
            "patch_success_rate": (
                proposal_status_counts.get("applied", 0) / proposal_count
                if proposal_count
                else 1.0
            ),
            "evidence_acceptance_rate": (
                evidence_accepted / proposal_count if proposal_count else 1.0
            ),
            "duplicate_rejection_count": sum(
                count
                for code, count in validation_code_counts.items()
                if "duplicate" in code
            ),
            "conflict_rejection_count": sum(
                validation_code_counts[code]
                for code in rejected_codes
                if "conflict" in code or "contradiction" in code
            ),
            "unsupported_evidence_count": sum(
                validation_code_counts[code]
                for code in rejected_codes
                if code
                in {
                    "empty_evidence",
                    "invalid_evidence_span",
                    "missing_evidence",
                    "missing_evidence_message",
                    "unsupported_evidence",
                    "unsupported_value",
                }
            ),
            "record_count": len(record_rows),
            "active_record_count": record_status_counts.get("active", 0),
            "record_status_counts": dict(sorted(record_status_counts.items())),
            "versioned_record_count": versioned_records,
            "duplicate_active_key_count": int(duplicate_row["count"]),
            "evidence_source_count": int(evidence_row["count"]),
            "queue": self.memory_patch_queue_status(session_id=session_id),
        }

    def memory_patch_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM memory_patch_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown memory patch run: {run_id}")
        return self._decode_json_columns(dict(row), ("usage_json",))

    def memory_patch_proposal_by_idempotency(
        self,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM memory_patch_proposals
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return self._decode_json_columns(
            dict(row),
            ("patch_json", "evidence_json", "validation_json"),
        )

    def tool_memory_record_detail(self, record_id: str) -> dict[str, Any]:
        record = self.tool_memory_record(record_id)
        if record is None:
            raise KeyError(f"Unknown tool memory record: {record_id}")
        with self._lock:
            status_events = self._connection.execute(
                """
                SELECT status, reason, patch_proposal_id, created_at
                FROM tool_memory_record_status_events
                WHERE record_id = ?
                ORDER BY id
                """,
                (record_id,),
            ).fetchall()
        return {
            "record": record,
            "sources": self.tool_memory_record_sources(record_id),
            "status_events": [dict(row) for row in status_events],
        }

    def routed_tool_memory_records(
        self,
        *,
        user_id: str,
        routes: Sequence[tuple[str, Sequence[str]]],
    ) -> list[ToolMemoryRecord]:
        if not routes:
            return []
        route_clauses: list[str] = []
        parameters: list[Any] = [normalize_identifier(user_id)]
        for domain, topics in routes:
            normalized_domain = normalize_identifier(domain)
            normalized_topics = tuple(
                dict.fromkeys(
                    normalize_identifier(topic)
                    for topic in topics
                    if normalize_identifier(topic)
                )
            )
            if normalized_topics:
                placeholders = ",".join("?" for _ in normalized_topics)
                route_clauses.append(
                    f"(tool_domain = ? AND topic IN ({placeholders}))"
                )
                parameters.extend((normalized_domain, *normalized_topics))
            else:
                route_clauses.append("(tool_domain = ?)")
                parameters.append(normalized_domain)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM tool_memory_records
                WHERE user_id = ?
                  AND status = 'active'
                  AND ({' OR '.join(route_clauses)})
                ORDER BY updated_at DESC, version DESC, id
                """,
                tuple(parameters),
            ).fetchall()
        return [self._decode_tool_memory_record(row) for row in rows]

    def store_tool_memory_embedding(
        self,
        *,
        record_id: str,
        model_id: str,
        dimensions: int,
        vector: Sequence[float],
    ) -> None:
        if dimensions < 1 or len(vector) != dimensions:
            raise ValueError("Tool memory embedding dimensions do not match")
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO tool_memory_embeddings(
                    record_id,
                    model_id,
                    dimensions,
                    vector_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(record_id, model_id, dimensions) DO UPDATE SET
                    vector_json = excluded.vector_json,
                    created_at = excluded.created_at
                """,
                (
                    record_id,
                    redact_secrets(model_id),
                    dimensions,
                    _json(list(vector)),
                    utc_now(),
                ),
            )

    def tool_memory_embeddings(
        self,
        record_ids: Sequence[str],
        *,
        model_id: str,
        dimensions: int,
    ) -> dict[str, tuple[float, ...]]:
        if not record_ids:
            return {}
        placeholders = ",".join("?" for _ in record_ids)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT record_id, vector_json
                FROM tool_memory_embeddings
                WHERE record_id IN ({placeholders})
                  AND model_id = ?
                  AND dimensions = ?
                """,
                (*record_ids, model_id, dimensions),
            ).fetchall()
        return {
            str(row["record_id"]): tuple(
                float(value) for value in json.loads(row["vector_json"])
            )
            for row in rows
        }

    def store_tool_schema_embeddings(
        self,
        embeddings: Mapping[str, tuple[str, Sequence[float]]],
        *,
        model_id: str,
        dimensions: int,
    ) -> None:
        if dimensions < 1:
            raise ValueError("Tool schema embedding dimensions must be positive")
        rows = []
        for tool_name, (fingerprint, vector) in embeddings.items():
            if len(vector) != dimensions:
                raise ValueError("Tool schema embedding dimensions do not match")
            rows.append(
                (
                    tool_name,
                    fingerprint,
                    redact_secrets(model_id),
                    dimensions,
                    _json(list(vector)),
                    utc_now(),
                )
            )
        if not rows:
            return
        with self._write_scope():
            self._connection.executemany(
                """
                INSERT INTO tool_schema_embeddings(
                    tool_name,
                    schema_fingerprint,
                    model_id,
                    dimensions,
                    vector_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    tool_name,
                    schema_fingerprint,
                    model_id,
                    dimensions
                ) DO UPDATE SET
                    vector_json = excluded.vector_json,
                    created_at = excluded.created_at
                """,
                rows,
            )

    def tool_schema_embeddings(
        self,
        fingerprints: Mapping[str, str],
        *,
        model_id: str,
        dimensions: int,
    ) -> dict[str, tuple[float, ...]]:
        if not fingerprints:
            return {}
        tool_names = tuple(fingerprints)
        placeholders = ",".join("?" for _ in tool_names)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT tool_name, schema_fingerprint, vector_json
                FROM tool_schema_embeddings
                WHERE tool_name IN ({placeholders})
                  AND model_id = ?
                  AND dimensions = ?
                """,
                (*tool_names, model_id, dimensions),
            ).fetchall()
        return {
            str(row["tool_name"]): tuple(
                float(value) for value in json.loads(row["vector_json"])
            )
            for row in rows
            if fingerprints.get(str(row["tool_name"]))
            == str(row["schema_fingerprint"])
        }

    def begin_tool_memory_retrieval(
        self,
        *,
        session_id: str,
        turn_id: str | None,
        query: str,
        user_id: str,
        mode: str,
        top_k: int,
        token_budget: int,
        route: Mapping[str, Any],
        classifier_used: bool,
    ) -> str:
        self.require_session(session_id)
        if turn_id is not None:
            with self._lock:
                turn = self._connection.execute(
                    "SELECT session_id FROM turns WHERE id = ?",
                    (turn_id,),
                ).fetchone()
            if turn is None:
                raise KeyError(f"Unknown retrieval turn: {turn_id}")
            if str(turn["session_id"]) != session_id:
                raise ValueError(
                    "Tool memory retrieval turn belongs to another session"
                )
        run_id = str(uuid.uuid4())
        with self._write_scope():
            self._connection.execute(
                """
                INSERT INTO tool_memory_retrieval_runs(
                    id,
                    session_id,
                    turn_id,
                    query,
                    user_id,
                    mode,
                    top_k,
                    token_budget,
                    route_json,
                    classifier_used,
                    status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    session_id,
                    turn_id,
                    redact_secrets(query),
                    normalize_identifier(user_id),
                    mode,
                    top_k,
                    token_budget,
                    _json(dict(route)),
                    int(classifier_used),
                    utc_now(),
                ),
            )
        return run_id

    def complete_tool_memory_retrieval(
        self,
        run_id: str,
        candidates: Sequence[Mapping[str, Any]],
        *,
        selected_count: int,
        selected_tokens: int,
        latency_ms: int,
    ) -> None:
        now = utc_now()
        with self._write_scope():
            for candidate in candidates:
                self._connection.execute(
                    """
                    INSERT INTO tool_memory_retrieval_candidates(
                        retrieval_run_id,
                        record_id,
                        route_domain,
                        route_topic,
                        rank,
                        bm25_score,
                        embedding_score,
                        combined_score,
                        selected,
                        exclusion_reason,
                        token_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        candidate["record_id"],
                        candidate.get("route_domain"),
                        candidate.get("route_topic"),
                        candidate.get("rank"),
                        candidate.get("bm25_score", 0.0),
                        candidate.get("embedding_score", 0.0),
                        candidate.get("combined_score", 0.0),
                        int(bool(candidate.get("selected"))),
                        candidate.get("exclusion_reason"),
                        candidate.get("token_count", 0),
                    ),
                )
            cursor = self._connection.execute(
                """
                UPDATE tool_memory_retrieval_runs
                SET status = 'completed',
                    candidate_count = ?,
                    selected_count = ?,
                    selected_tokens = ?,
                    latency_ms = ?,
                    ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    len(candidates),
                    selected_count,
                    selected_tokens,
                    latency_ms,
                    now,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running Tool memory retrieval not found: {run_id}")

    def fail_tool_memory_retrieval(
        self,
        run_id: str,
        *,
        error: str,
        latency_ms: int,
    ) -> None:
        with self._write_scope():
            cursor = self._connection.execute(
                """
                UPDATE tool_memory_retrieval_runs
                SET status = 'failed',
                    candidate_count = 0,
                    selected_count = 0,
                    selected_tokens = 0,
                    latency_ms = ?,
                    error = ?,
                    ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    latency_ms,
                    redact_secrets(error),
                    utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running Tool memory retrieval not found: {run_id}")

    def list_tool_memory_retrievals(
        self,
        session_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM tool_memory_retrieval_runs
                WHERE session_id = ?
                ORDER BY started_at DESC, id
                LIMIT ?
                """,
                (session_id, max(1, limit)),
            ).fetchall()
        return [
            self._decode_json_columns(dict(row), ("route_json",))
            for row in rows
        ]

    def tool_memory_retrieval_trace(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._connection.execute(
                "SELECT * FROM tool_memory_retrieval_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown Tool memory retrieval: {run_id}")
            candidates = self._connection.execute(
                """
                SELECT
                    tool_memory_retrieval_candidates.*,
                    tool_memory_records.user_id,
                    tool_memory_records.tool_domain,
                    tool_memory_records.topic,
                    tool_memory_records.scope,
                    tool_memory_records.scope_key,
                    tool_memory_records.conditions_json,
                    tool_memory_records.value_json,
                    tool_memory_records.version,
                    tool_memory_records.confidence,
                    tool_memory_records.entity_id,
                    tool_memory_records.identity_conditions_json,
                    tool_memory_records.applicability_json,
                    tool_memory_records.identity_family_key
                FROM tool_memory_retrieval_candidates
                JOIN tool_memory_records
                  ON tool_memory_records.id =
                     tool_memory_retrieval_candidates.record_id
                WHERE retrieval_run_id = ?
                ORDER BY selected DESC, rank, record_id
                """,
                (run_id,),
            ).fetchall()
        return {
            "run": self._decode_json_columns(dict(run), ("route_json",)),
            "candidates": [
                self._decode_json_columns(
                    dict(row),
                    (
                        "conditions_json",
                        "value_json",
                        "identity_conditions_json",
                        "applicability_json",
                    ),
                )
                for row in candidates
            ],
        }

    @staticmethod
    def _decode_memory_record(row: sqlite3.Row) -> MemoryRecord:
        value = json.loads(row["value_json"]) if row["value_json"] else ""
        return MemoryRecord(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            scope=str(row["scope"]),
            content=str(row["content"]),
            status=str(row["status"]),
            version=int(row["version"]),
            subject=str(row["subject"]),
            predicate=str(row["predicate"]),
            value=str(value),
            fact_key=str(row["fact_key"]),
            memory_type=str(row["memory_type"]),
            confidence=float(row["confidence"]),
            sensitivity=str(row["sensitivity"]),
            supersedes_id=row["supersedes_id"],
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _decode_tool_memory_record(row: sqlite3.Row) -> ToolMemoryRecord:
        return ToolMemoryRecord(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            record_key=str(row["record_key"]),
            user_id=str(row["user_id"]),
            tool_domain=str(row["tool_domain"]),
            topic=str(row["topic"]),
            scope=str(row["scope"]),
            scope_key=str(row["scope_key"]),
            conditions=json.loads(row["conditions_json"]),
            value=json.loads(row["value_json"]),
            memory_type=str(row["memory_type"]),
            status=str(row["status"]),
            confidence=float(row["confidence"]),
            version=int(row["version"]),
            supersedes_id=row["supersedes_id"],
            merged_into_id=row["merged_into_id"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            entity_id=row["entity_id"],
            identity_conditions=json.loads(row["identity_conditions_json"]),
            applicability=json.loads(row["applicability_json"]),
            identity_family_key=row["identity_family_key"],
        )

    @staticmethod
    def _decode_fact_memory_record(row: sqlite3.Row) -> FactMemoryRecord:
        return FactMemoryRecord(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            record_key=str(row["record_key"]),
            user_id=str(row["user_id"]),
            entity_id=str(row["entity_id"]),
            predicate=str(row["predicate"]),
            identity_conditions=json.loads(row["identity_conditions_json"]),
            applicability=json.loads(row["applicability_json"]),
            capability_hints=tuple(json.loads(row["capability_hints_json"])),
            value=json.loads(row["value_json"]),
            memory_type=str(row["memory_type"]),
            status=str(row["status"]),
            confidence=float(row["confidence"]),
            version=int(row["version"]),
            supersedes_id=row["supersedes_id"],
            merged_into_id=row["merged_into_id"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            bundle_id=row["bundle_id"],
        )

    def apply_structured_memory(
        self,
        *,
        session_id: str,
        candidate: MemoryCandidate,
        fact_key: str,
        consolidation_run_id: str,
        model_id: str,
        schema_version: str,
        rejection_reason: str | None = None,
        gate_decision: MemoryGateDecision | None = None,
    ) -> tuple[MemoryRecord, str]:
        memory_id = str(uuid.uuid4())
        safe_value = redact_secrets(candidate.value.strip())
        content = redact_secrets(
            f"{candidate.subject}.{candidate.predicate}: {safe_value}"
        )
        now = utc_now()
        with self._lock, self._connection:
            active = self._connection.execute(
                """
                SELECT *
                FROM memories
                WHERE fact_key = ? AND status = 'verified'
                LIMIT 1
                """,
                (fact_key,),
            ).fetchone()
            version_row = self._connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) AS version
                FROM memories
                WHERE fact_key = ?
                """,
                (fact_key,),
            ).fetchone()
            version = int(version_row["version"]) + 1
            final_status = "verified"
            action = "created"
            supersedes_id = None
            resolved_rejection = rejection_reason
            requested_decision = (
                gate_decision.decision if gate_decision is not None else "auto"
            )
            if rejection_reason or requested_decision == "reject":
                final_status = "rejected"
                action = "rejected"
                resolved_rejection = rejection_reason or gate_decision.reason
            elif requested_decision == "review":
                final_status = "review"
                action = "review"
                resolved_rejection = gate_decision.reason
            elif active is not None:
                active_value = (
                    json.loads(active["value_json"]) if active["value_json"] else ""
                )
                if str(active_value) == safe_value:
                    final_status = "rejected"
                    action = "duplicate"
                    resolved_rejection = "duplicate_active_value"
                elif requested_decision in {"auto", "supersede"}:
                    supersedes_id = str(active["id"])
                    self._connection.execute(
                        """
                        UPDATE memories
                        SET status = 'superseded'
                        WHERE id = ? AND status = 'verified'
                        """,
                        (supersedes_id,),
                    )
                    self._connection.execute(
                        """
                        INSERT INTO memory_status_events(
                            memory_id, status, reason, created_at
                        )
                        VALUES (?, 'superseded', ?, ?)
                        """,
                        (supersedes_id, f"superseded_by:{memory_id}", now),
                    )
                    action = "superseded"
                else:
                    final_status = "review"
                    action = "review"
                    resolved_rejection = "unresolved_conflict"
            self._connection.execute(
                """
                INSERT INTO memories(
                    id,
                    session_id,
                    scope,
                    content,
                    status,
                    version,
                    created_at,
                    subject,
                    predicate,
                    value_json,
                    fact_key,
                    memory_type,
                    confidence,
                    sensitivity,
                    supersedes_id,
                    consolidation_run_id,
                    rejection_reason,
                    model_id,
                    schema_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    session_id,
                    candidate.scope,
                    content,
                    "proposed",
                    version,
                    now,
                    redact_secrets(candidate.subject.strip()),
                    redact_secrets(candidate.predicate.strip()),
                    _json(safe_value),
                    fact_key,
                    candidate.memory_type,
                    candidate.confidence,
                    candidate.sensitivity,
                    supersedes_id,
                    consolidation_run_id,
                    resolved_rejection,
                    model_id,
                    schema_version,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO memory_status_events(
                    memory_id, status, reason, created_at
                )
                VALUES (?, 'proposed', NULL, ?)
                """,
                (memory_id, now),
            )
            if gate_decision is not None:
                self._connection.execute(
                    """
                    INSERT INTO memory_gate_decisions(
                        memory_id,
                        decision,
                        evidence_relation,
                        conflict_relation,
                        reason,
                        policy_version,
                        pii_categories_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        gate_decision.decision,
                        gate_decision.evidence_relation,
                        gate_decision.conflict_relation,
                        gate_decision.reason,
                        gate_decision.policy_version,
                        _json(list(gate_decision.pii_categories)),
                        now,
                    ),
                )
            self._connection.execute(
                """
                UPDATE memories
                SET status = ?
                WHERE id = ? AND status = 'proposed'
                """,
                (final_status, memory_id),
            )
            self._connection.execute(
                """
                INSERT INTO memory_status_events(
                    memory_id, status, reason, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    memory_id,
                    final_status,
                    resolved_rejection,
                    now,
                ),
            )
            for evidence in candidate.evidence:
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO memory_sources(
                        memory_id,
                        message_id,
                        evidence_text,
                        start_char,
                        end_char,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        evidence.message_id,
                        redact_secrets(evidence.quote),
                        evidence.start_char,
                        evidence.end_char,
                        now,
                    ),
                )
            row = self._connection.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._decode_memory_record(row), action

    def store_memory_embedding(
        self,
        *,
        memory_id: str,
        model_id: str,
        dimensions: int,
        vector: Sequence[float],
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO memory_embeddings(
                    memory_id,
                    model_id,
                    dimensions,
                    vector_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, model_id, dimensions) DO UPDATE SET
                    vector_json = excluded.vector_json,
                    created_at = excluded.created_at
                """,
                (
                    memory_id,
                    model_id,
                    dimensions,
                    _json(list(vector)),
                    utc_now(),
                ),
            )

    def memory_embeddings(
        self,
        memory_ids: Sequence[str],
        *,
        model_id: str,
        dimensions: int,
    ) -> dict[str, tuple[float, ...]]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT memory_id, vector_json
                FROM memory_embeddings
                WHERE memory_id IN ({placeholders})
                  AND model_id = ?
                  AND dimensions = ?
                """,
                (*memory_ids, model_id, dimensions),
            ).fetchall()
        return {
            str(row["memory_id"]): tuple(
                float(value) for value in json.loads(row["vector_json"])
            )
            for row in rows
        }

    def memory_detail(self, memory_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown memory: {memory_id}")
            sources = self._connection.execute(
                """
                SELECT *
                FROM memory_sources
                WHERE memory_id = ?
                ORDER BY message_id
                """,
                (memory_id,),
            ).fetchall()
            embeddings = self._connection.execute(
                """
                SELECT model_id, dimensions, created_at
                FROM memory_embeddings
                WHERE memory_id = ?
                ORDER BY model_id, dimensions
                """,
                (memory_id,),
            ).fetchall()
            status_events = self._connection.execute(
                """
                SELECT status, reason, created_at
                FROM memory_status_events
                WHERE memory_id = ?
                ORDER BY id
                """,
                (memory_id,),
            ).fetchall()
            gate_decision = self._connection.execute(
                """
                SELECT *
                FROM memory_gate_decisions
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()
        memory = self._decode_json_columns(dict(row), ("value_json",))
        memory["sources"] = [dict(source) for source in sources]
        memory["embeddings"] = [dict(embedding) for embedding in embeddings]
        memory["status_events"] = [dict(event) for event in status_events]
        memory["gate_decision"] = (
            self._decode_json_columns(
                dict(gate_decision),
                ("pii_categories_json",),
            )
            if gate_decision
            else None
        )
        return memory

    def list_memory_reviews(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT
                    memories.*,
                    memory_gate_decisions.decision AS gate_decision,
                    memory_gate_decisions.evidence_relation,
                    memory_gate_decisions.conflict_relation,
                    memory_gate_decisions.reason AS gate_reason,
                    memory_gate_decisions.policy_version,
                    memory_gate_decisions.pii_categories_json
                FROM memories
                LEFT JOIN memory_gate_decisions
                  ON memory_gate_decisions.memory_id = memories.id
                WHERE memories.status = 'review'
                  AND (
                    (memories.scope = 'session' AND memories.session_id = ?)
                    OR memories.scope = 'global'
                  )
                ORDER BY memories.created_at
                """,
                (session_id,),
            ).fetchall()
        return [
            self._decode_json_columns(
                dict(row),
                ("value_json", "pii_categories_json"),
            )
            for row in rows
        ]

    def resolve_memory_review(
        self,
        memory_id: str,
        *,
        decision: str,
        reason: str,
    ) -> dict[str, Any]:
        if decision not in {"accept", "reject"}:
            raise ValueError("review decision must be 'accept' or 'reject'")
        safe_reason = redact_secrets(reason.strip() or "manual_review")
        now = utc_now()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM memories WHERE id = ? AND status = 'review'",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Memory is not awaiting review: {memory_id}")
            final_status = "rejected"
            event_reason = f"manual_reject:{safe_reason}"
            if decision == "accept":
                active = self._connection.execute(
                    """
                    SELECT *
                    FROM memories
                    WHERE fact_key = ? AND status = 'verified'
                    LIMIT 1
                    """,
                    (row["fact_key"],),
                ).fetchone()
                if active is not None and active["value_json"] == row["value_json"]:
                    final_status = "rejected"
                    event_reason = "duplicate_active_value"
                else:
                    final_status = "verified"
                    event_reason = f"manual_accept:{safe_reason}"
                    if active is not None:
                        self._connection.execute(
                            """
                            UPDATE memories
                            SET status = 'superseded'
                            WHERE id = ? AND status = 'verified'
                            """,
                            (active["id"],),
                        )
                        self._connection.execute(
                            """
                            INSERT INTO memory_status_events(
                                memory_id, status, reason, created_at
                            )
                            VALUES (?, 'superseded', ?, ?)
                            """,
                            (
                                active["id"],
                                f"superseded_by:{memory_id}:manual_review",
                                now,
                            ),
                        )
                        self._connection.execute(
                            """
                            UPDATE memories
                            SET supersedes_id = ?
                            WHERE id = ?
                            """,
                            (active["id"], memory_id),
                        )
            self._connection.execute(
                """
                UPDATE memories
                SET status = ?, rejection_reason = ?
                WHERE id = ? AND status = 'review'
                """,
                (
                    final_status,
                    None if final_status == "verified" else event_reason,
                    memory_id,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO memory_status_events(
                    memory_id, status, reason, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, final_status, event_reason, now),
            )
        return self.memory_detail(memory_id)

    def memory_versions(self, fact_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM memories
                WHERE fact_key = ?
                ORDER BY version ASC
                """,
                (fact_key,),
            ).fetchall()
        return [self._decode_json_columns(dict(row), ("value_json",)) for row in rows]

    def begin_retrieval(
        self,
        *,
        session_id: str,
        turn_id: str | None,
        query: str,
        mode: str,
        top_k: int,
        token_budget: int,
        embedding_backend: str | None,
        embedding_model_id: str | None,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO retrieval_runs(
                    id,
                    session_id,
                    turn_id,
                    query,
                    mode,
                    top_k,
                    token_budget,
                    embedding_backend,
                    embedding_model_id,
                    status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    session_id,
                    turn_id,
                    redact_secrets(query),
                    mode,
                    top_k,
                    token_budget,
                    embedding_backend,
                    embedding_model_id,
                    utc_now(),
                ),
            )
        return run_id

    def complete_retrieval(
        self,
        run_id: str,
        candidates: Sequence[dict[str, Any]],
        *,
        selected_count: int,
    ) -> None:
        now = utc_now()
        with self._lock, self._connection:
            for candidate in candidates:
                self._connection.execute(
                    """
                    INSERT INTO retrieval_candidates(
                        retrieval_run_id,
                        memory_id,
                        rank,
                        bm25_score,
                        embedding_score,
                        combined_score,
                        selected,
                        exclusion_reason,
                        token_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        candidate["memory_id"],
                        candidate.get("rank"),
                        candidate.get("bm25_score", 0.0),
                        candidate.get("embedding_score", 0.0),
                        candidate.get("combined_score", 0.0),
                        int(bool(candidate.get("selected"))),
                        candidate.get("exclusion_reason"),
                        candidate.get("token_count", 0),
                    ),
                )
            self._connection.execute(
                """
                UPDATE retrieval_runs
                SET status = 'completed',
                    selected_count = ?,
                    ended_at = ?
                WHERE id = ?
                """,
                (selected_count, now, run_id),
            )

    def fail_retrieval(self, run_id: str, error: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE retrieval_runs
                SET status = 'failed', error = ?, ended_at = ?
                WHERE id = ?
                """,
                (redact_secrets(error), utc_now(), run_id),
            )

    def list_retrieval_runs(
        self,
        session_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM retrieval_runs
                WHERE session_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def retrieval_trace(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._connection.execute(
                "SELECT * FROM retrieval_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown retrieval run: {run_id}")
            candidates = self._connection.execute(
                """
                SELECT
                    retrieval_candidates.*,
                    memories.content,
                    memories.scope,
                    memories.version,
                    memories.fact_key
                FROM retrieval_candidates
                JOIN memories
                  ON memories.id = retrieval_candidates.memory_id
                WHERE retrieval_run_id = ?
                ORDER BY selected DESC, rank, combined_score DESC
                """,
                (run_id,),
            ).fetchall()
        return {
            "run": dict(run),
            "candidates": [dict(candidate) for candidate in candidates],
        }

    def consolidation_window(
        self,
        session_id: str,
        trigger_messages: int,
    ) -> list[StoredMessage]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COALESCE(MAX(end_message_id), 0) AS watermark
                FROM consolidation_runs
                WHERE session_id = ?
                  AND status = 'completed'
                """,
                (session_id,),
            ).fetchone()
        messages = self.list_messages(
            session_id,
            after_id=int(row["watermark"]),
        )
        countable = [
            message for message in messages if message.role in {"user", "assistant"}
        ]
        return messages if len(countable) >= trigger_messages else []

    def begin_consolidation(
        self,
        session_id: str,
        start_message_id: int,
        end_message_id: int,
        *,
        backend: str,
        model_id: str,
        prompt_version: str,
        schema_version: str,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO consolidation_runs(
                    id,
                    session_id,
                    start_message_id,
                    end_message_id,
                    backend,
                    model_id,
                    prompt_version,
                    schema_version,
                    status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    session_id,
                    start_message_id,
                    end_message_id,
                    backend,
                    model_id,
                    prompt_version,
                    schema_version,
                    utc_now(),
                ),
            )
        return run_id

    def complete_consolidation(
        self,
        run_id: str,
        session_id: str,
        content: str,
        usage: ModelUsage,
    ) -> str:
        memory_id = str(uuid.uuid4())
        safe_content = redact_secrets(content.strip())
        now = utc_now()
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) AS version
                FROM memories
                WHERE session_id = ? AND scope = 'session'
                """,
                (session_id,),
            ).fetchone()
            version = int(row["version"]) + 1
            self._connection.execute(
                """
                UPDATE memories
                SET status = 'superseded'
                WHERE session_id = ?
                  AND scope = 'session'
                  AND status = 'verified'
                """,
                (session_id,),
            )
            self._connection.execute(
                """
                INSERT INTO memories(
                    id,
                    session_id,
                    scope,
                    content,
                    status,
                    version,
                    created_at
                )
                VALUES (?, ?, 'session', ?, 'verified', ?, ?)
                """,
                (memory_id, session_id, safe_content, version, now),
            )
            self._connection.execute(
                """
                UPDATE consolidation_runs
                SET status = 'completed',
                    output = ?,
                    usage_json = ?,
                    ended_at = ?
                WHERE id = ?
                """,
                (safe_content, _json(usage.as_dict()), now, run_id),
            )
        return memory_id

    def complete_structured_consolidation(
        self,
        run_id: str,
        *,
        output: dict[str, Any],
        usage: ModelUsage,
    ) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE consolidation_runs
                SET status = 'completed',
                    output = ?,
                    usage_json = ?,
                    ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    _json(output),
                    _json(usage.as_dict()),
                    utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Consolidation is not running: {run_id}")

    def complete_empty_consolidation(
        self,
        run_id: str,
        usage: ModelUsage,
    ) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE consolidation_runs
                SET status = 'completed',
                    output = '',
                    usage_json = ?,
                    ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (_json(usage.as_dict()), utc_now(), run_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Consolidation is not running: {run_id}")

    def consolidation_progress(self, session_id: str) -> dict[str, Any]:
        self.require_session(session_id)
        with self._lock:
            watermark_row = self._connection.execute(
                """
                SELECT COALESCE(MAX(end_message_id), 0) AS watermark,
                       COUNT(*) AS completed_runs
                FROM consolidation_runs
                WHERE session_id = ? AND status = 'completed'
                """,
                (session_id,),
            ).fetchone()
            status_rows = self._connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM consolidation_runs
                WHERE session_id = ?
                GROUP BY status
                """,
                (session_id,),
            ).fetchall()
            completed_rows = self._connection.execute(
                """
                SELECT id
                FROM consolidation_runs
                WHERE session_id = ? AND status = 'completed'
                ORDER BY end_message_id, started_at, id
                """,
                (session_id,),
            ).fetchall()
        messages = self.list_messages(session_id)
        watermark = int(watermark_row["watermark"])
        return {
            "message_count": len(messages),
            "completed_message_count": sum(
                message.id <= watermark for message in messages
            ),
            "pending_message_count": sum(
                message.id > watermark for message in messages
            ),
            "watermark": watermark,
            "completed_runs": int(watermark_row["completed_runs"]),
            "completed_run_ids": [str(row["id"]) for row in completed_rows],
            "run_status_counts": {
                str(row["status"]): int(row["count"]) for row in status_rows
            },
        }

    def consolidation_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM consolidation_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown consolidation run: {run_id}")
        return self._decode_json_columns(dict(row), ("usage_json",))

    def fail_consolidation(
        self,
        run_id: str,
        error: str,
        *,
        retryable: bool = False,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE consolidation_runs
                SET status = ?, error = ?, ended_at = ?
                WHERE id = ?
                """,
                (
                    "retryable" if retryable else "failed",
                    redact_secrets(error),
                    utc_now(),
                    run_id,
                ),
            )

    def record_model_call(
        self,
        *,
        session_id: str,
        role: str,
        backend: str,
        model_id: str,
        prompt_version: str,
        schema_version: str | None,
        latency_ms: int,
        usage: ModelUsage,
        response_id: str | None,
        metadata: dict[str, Any],
        turn_id: str | None = None,
        consolidation_run_id: str | None = None,
        memory_patch_run_id: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO model_calls(
                    id,
                    session_id,
                    turn_id,
                    consolidation_run_id,
                    memory_patch_run_id,
                    role,
                    backend,
                    model_id,
                    prompt_version,
                    schema_version,
                    response_id,
                    latency_ms,
                    usage_json,
                    metadata_json,
                    error,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    session_id,
                    turn_id,
                    consolidation_run_id,
                    memory_patch_run_id,
                    role,
                    backend,
                    model_id,
                    prompt_version,
                    schema_version,
                    response_id,
                    latency_ms,
                    _json(usage.as_dict()),
                    _json(metadata),
                    redact_secrets(error) if error else None,
                    utc_now(),
                ),
            )

    def memory_backend_report(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            call_rows = self._connection.execute(
                """
                SELECT model_calls.*, consolidation_runs.status AS run_status
                FROM model_calls
                LEFT JOIN consolidation_runs
                  ON consolidation_runs.id = model_calls.consolidation_run_id
                WHERE model_calls.session_id = ?
                  AND model_calls.role = 'memory'
                ORDER BY model_calls.created_at
                """,
                (session_id,),
            ).fetchall()
            outcome_rows = self._connection.execute(
                """
                SELECT
                    consolidation_runs.backend,
                    memory_gate_decisions.decision,
                    COUNT(*) AS count
                FROM memories
                JOIN consolidation_runs
                  ON consolidation_runs.id = memories.consolidation_run_id
                LEFT JOIN memory_gate_decisions
                  ON memory_gate_decisions.memory_id = memories.id
                WHERE consolidation_runs.session_id = ?
                GROUP BY
                    consolidation_runs.backend,
                    memory_gate_decisions.decision
                """,
                (session_id,),
            ).fetchall()
        calls = [
            self._decode_json_columns(
                dict(row),
                ("usage_json", "metadata_json"),
            )
            for row in call_rows
        ]
        backends: dict[str, dict[str, Any]] = {}
        for call in calls:
            backend = str(call["backend"])
            summary = backends.setdefault(
                backend,
                {
                    "backend": backend,
                    "calls": 0,
                    "errors": 0,
                    "retryable_runs": 0,
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "cloud_sensitive_spans_before": 0,
                    "cloud_sensitive_spans_after": 0,
                    "gate_outcomes": {},
                },
            )
            summary["calls"] += 1
            summary["errors"] += int(bool(call.get("error")))
            summary["retryable_runs"] += int(call.get("run_status") == "retryable")
            summary["latency_ms"] += int(call["latency_ms"])
            usage = call.get("usage", {})
            summary["input_tokens"] += int(usage.get("input_tokens", 0))
            summary["output_tokens"] += int(usage.get("output_tokens", 0))
            metadata = call.get("metadata", {})
            summary["estimated_cost_usd"] += float(
                metadata.get("estimated_cost_usd", 0.0)
            )
            privacy = metadata.get("privacy", {})
            if privacy.get("destination") == "cloud":
                summary["cloud_sensitive_spans_before"] += int(
                    privacy.get("detected_count", 0)
                )
                summary["cloud_sensitive_spans_after"] += int(
                    privacy.get("sensitive_chars_after", 0)
                )
        for outcome in outcome_rows:
            backend = str(outcome["backend"])
            summary = backends.setdefault(
                backend,
                {
                    "backend": backend,
                    "calls": 0,
                    "errors": 0,
                    "retryable_runs": 0,
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "cloud_sensitive_spans_before": 0,
                    "cloud_sensitive_spans_after": 0,
                    "gate_outcomes": {},
                },
            )
            decision = str(outcome["decision"] or "legacy")
            summary["gate_outcomes"][decision] = int(outcome["count"])
        for summary in backends.values():
            summary["average_latency_ms"] = (
                round(summary["latency_ms"] / summary["calls"], 2)
                if summary["calls"]
                else 0.0
            )
            summary["estimated_cost_usd"] = round(
                summary["estimated_cost_usd"],
                8,
            )
        return {
            "session_id": session_id,
            "backends": sorted(backends.values(), key=lambda item: item["backend"]),
            "runs": calls,
        }

    def model_calls_for_sessions(
        self,
        session_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        unique_ids = tuple(dict.fromkeys(str(item) for item in session_ids))
        if not unique_ids:
            return []
        for session_id in unique_ids:
            self.require_session(session_id)
        placeholders = ",".join("?" for _ in unique_ids)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM model_calls
                WHERE session_id IN ({placeholders})
                ORDER BY created_at, id
                """,
                unique_ids,
            ).fetchall()
        return [
            self._decode_json_columns(
                dict(row),
                ("usage_json", "metadata_json"),
            )
            for row in rows
        ]

    def get_trace(self, turn_id: str) -> dict[str, Any]:
        with self._lock:
            turn = self._connection.execute(
                "SELECT * FROM turns WHERE id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"Unknown turn: {turn_id}")
            calls = self._connection.execute(
                """
                SELECT *
                FROM tool_calls
                WHERE turn_id = ?
                ORDER BY round_number, created_at
                """,
                (turn_id,),
            ).fetchall()
            tool_results = self._connection.execute(
                """
                SELECT tool_results.*
                FROM tool_results
                JOIN tool_calls
                  ON tool_calls.id = tool_results.tool_call_id
                WHERE tool_calls.turn_id = ?
                ORDER BY tool_results.created_at
                """,
                (turn_id,),
            ).fetchall()
            model_calls = self._connection.execute(
                """
                SELECT *
                FROM model_calls
                WHERE turn_id = ?
                ORDER BY created_at
                """,
                (turn_id,),
            ).fetchall()
        return {
            "turn": dict(turn),
            "tool_calls": [
                self._decode_json_columns(
                    dict(row),
                    ("arguments_json",),
                )
                for row in calls
            ],
            "tool_results": [
                self._decode_json_columns(
                    dict(row),
                    ("metadata_json",),
                )
                for row in tool_results
            ],
            "model_calls": [
                self._decode_json_columns(
                    dict(row),
                    ("usage_json", "metadata_json"),
                )
                for row in model_calls
            ],
        }

    @staticmethod
    def _decode_json_columns(
        item: dict[str, Any],
        columns: Iterable[str],
    ) -> dict[str, Any]:
        for column in columns:
            if item.get(column):
                item[column.removesuffix("_json")] = json.loads(item[column])
            item.pop(column, None)
        return item

    def list_turns(
        self,
        session_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM turns
                WHERE session_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def begin_evaluation(
        self,
        *,
        name: str,
        dataset_name: str,
        dataset_version: str,
        dataset_sha256: str,
        execution_mode: str,
        profiles: Sequence[str],
        seed: int,
        repetitions: int,
    ) -> str:
        run_id = str(uuid.uuid4())
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO evaluation_runs(
                    id,
                    name,
                    dataset_name,
                    dataset_version,
                    dataset_sha256,
                    execution_mode,
                    profiles_json,
                    seed,
                    repetitions,
                    status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    redact_secrets(name.strip() or "Evaluation"),
                    dataset_name,
                    dataset_version,
                    dataset_sha256,
                    execution_mode,
                    _evaluation_json(list(profiles)),
                    seed,
                    repetitions,
                    utc_now(),
                ),
            )
        return run_id

    def record_evaluation_case(
        self,
        *,
        evaluation_run_id: str,
        profile: str,
        case_id: str,
        repetition: int,
        status: str,
        latency_ms: int,
        expected: dict[str, Any],
        actual: dict[str, Any],
        metrics: dict[str, Any],
        error: str | None = None,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO evaluation_cases(
                    evaluation_run_id,
                    profile,
                    case_id,
                    repetition,
                    status,
                    latency_ms,
                    expected_json,
                    actual_json,
                    metrics_json,
                    error,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_run_id,
                    profile,
                    case_id,
                    repetition,
                    status,
                    latency_ms,
                    _evaluation_json(expected),
                    _evaluation_json(actual),
                    _evaluation_json(metrics),
                    redact_secrets(error) if error else None,
                    utc_now(),
                ),
            )

    def complete_evaluation(
        self,
        run_id: str,
        *,
        metrics: dict[str, Any],
        artifact_dir: str,
        with_errors: bool = False,
    ) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE evaluation_runs
                SET status = ?,
                    metrics_json = ?,
                    artifact_dir = ?,
                    ended_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    "completed_with_errors" if with_errors else "completed",
                    _evaluation_json(metrics),
                    str(Path(artifact_dir).resolve()),
                    utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Evaluation is not running: {run_id}")

    def refresh_evaluation_report(
        self,
        run_id: str,
        *,
        metrics: dict[str, Any],
        artifact_dir: str,
    ) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE evaluation_runs
                SET metrics_json = ?, artifact_dir = ?
                WHERE id = ?
                  AND status IN ('completed', 'completed_with_errors')
                """,
                (
                    _evaluation_json(metrics),
                    str(Path(artifact_dir).resolve()),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Evaluation is not completed and refreshable: {run_id}"
                )

    def fail_evaluation(self, run_id: str, error: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE evaluation_runs
                SET status = 'failed', error = ?, ended_at = ?
                WHERE id = ?
                """,
                (redact_secrets(error), utc_now(), run_id),
            )

    def list_evaluation_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM evaluation_runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            self._decode_json_columns(
                dict(row),
                ("profiles_json", "metrics_json"),
            )
            for row in rows
        ]

    def evaluation_detail(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._connection.execute(
                "SELECT * FROM evaluation_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown evaluation run: {run_id}")
            cases = self._connection.execute(
                """
                SELECT *
                FROM evaluation_cases
                WHERE evaluation_run_id = ?
                ORDER BY profile, repetition, case_id
                """,
                (run_id,),
            ).fetchall()
        return {
            "run": self._decode_json_columns(
                dict(run),
                ("profiles_json", "metrics_json"),
            ),
            "cases": [
                self._decode_json_columns(
                    dict(row),
                    ("expected_json", "actual_json", "metrics_json"),
                )
                for row in cases
            ],
        }


def _fact_memory_bundle_id(
    *,
    session_id: str,
    entity_id: str,
    evidence: Sequence[MemoryEvidence],
    record_id: str,
) -> str:
    """Derive a stable source-event bundle without merging unrelated facts."""

    message_ids = sorted({int(item.message_id) for item in evidence})
    if not message_ids:
        return f"fact:{record_id}"
    payload = json.dumps(
        [session_id, normalize_identifier(entity_id), message_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "event:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
