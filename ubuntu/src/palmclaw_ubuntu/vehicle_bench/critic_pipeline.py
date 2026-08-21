from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from palmclaw_ubuntu.vehicle_bench.critic import (
    REJECT_REASON_CODES,
    ValidatedVehicleMemoryCriticResult,
    VehicleCriticRollbackRequestPayload,
    VehicleMemoryAdjudication,
    VehicleMemoryCriticResponse,
    build_defer_keep_result,
)
from palmclaw_ubuntu.vehicle_bench.critic_retrieval import (
    CausalTurnRetriever,
    CriticRetrievalResult,
)
from palmclaw_ubuntu.vehicle_bench.critic_trace import (
    CriticMethod,
    CriticSourceTurn,
)

CriticRunStatus = Literal["RUNNING", "COMPLETED", "UNRESOLVED"]
CRITIC_STATE_SCHEMA_VERSION = "vehiclemembench-v2-critic-state-v1"


class VehicleMemoryCriticReviewer(Protocol):
    def review(
        self,
        *,
        source_turn: CriticSourceTurn,
        approved_memory: str,
        max_memory_chars: int,
    ) -> ValidatedVehicleMemoryCriticResult: ...


class VehicleMemoryAdjudicator(Protocol):
    def adjudicate(
        self,
        *,
        source_turn: CriticSourceTurn,
        approved_memory: str,
        luna_reject: ValidatedVehicleMemoryCriticResult,
        recovered_context: Sequence[CriticSourceTurn] = (),
        max_memory_chars: int,
    ) -> VehicleMemoryAdjudication: ...

    def repair_omission(
        self,
        *,
        source_turn: CriticSourceTurn,
        approved_memory: str,
        rollback_request: VehicleCriticRollbackRequestPayload,
        trigger_message_id: int,
        max_memory_chars: int,
    ) -> ValidatedVehicleMemoryCriticResult: ...


@dataclass(frozen=True)
class CriticRunState:
    method: CriticMethod
    scenario_index: int
    source_fingerprint: str
    turn_count: int
    next_turn_index: int
    approved_memory: str
    status: CriticRunStatus
    replay_generation: int
    pending_repair: Mapping[str, Any] | None


@dataclass(frozen=True)
class CriticPipelineRunResult:
    method: CriticMethod
    scenario_index: int
    status: CriticRunStatus
    processed_turns: int
    turn_count: int
    replay_generation: int
    final_memory: str
    final_memory_sha256: str
    attempt_count: int
    rollback_count: int


class CriticCheckpointStore:
    """Transactional canonical state for one method/scenario trajectory."""

    def __init__(
        self,
        path: Path | str,
        *,
        turns: tuple[CriticSourceTurn, ...],
    ) -> None:
        if not turns:
            raise ValueError("Critic checkpoint requires source turns")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.turns = turns
        self.method = turns[0].method
        self.scenario_index = turns[0].scenario_index
        _validate_source_turns(turns)
        self.source_fingerprint = _source_fingerprint(turns)
        self._initialize()
        self._validate_checkpoint()

    def state(self) -> CriticRunState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_state WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("Critic run state is missing")
        pending = _decode_optional_object(row["pending_repair_json"])
        return CriticRunState(
            method=row["method"],
            scenario_index=int(row["scenario_index"]),
            source_fingerprint=row["source_fingerprint"],
            turn_count=int(row["turn_count"]),
            next_turn_index=int(row["next_turn_index"]),
            approved_memory=row["approved_memory"],
            status=row["status"],
            replay_generation=int(row["replay_generation"]),
            pending_repair=pending,
        )

    def bind_run_config(self, config: Mapping[str, Any]) -> None:
        encoded = _encode_json(config)
        fingerprint = _text_sha256(encoded)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._state_row(connection)
            recorded = state["config_fingerprint"]
            if recorded is None:
                connection.execute(
                    """
                    UPDATE run_state
                    SET run_config_json = ?, config_fingerprint = ?, updated_at = ?
                    WHERE singleton = 1
                    """,
                    (encoded, fingerprint, _now()),
                )
            elif str(recorded) != fingerprint:
                raise ValueError("Critic checkpoint run configuration does not match")

    def record_attempt(
        self,
        *,
        turn: CriticSourceTurn,
        replay_generation: int,
        stage: str,
        payload: Mapping[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_attempts (
                    turn_index, message_id, replay_generation, stage,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_index,
                    turn.message_id,
                    replay_generation,
                    stage,
                    _encode_json(payload),
                    _now(),
                ),
            )

    def commit_turn(
        self,
        *,
        turn: CriticSourceTurn,
        result: ValidatedVehicleMemoryCriticResult,
        stage: str,
        resolution_metadata: Mapping[str, Any] | None = None,
    ) -> CriticRunState:
        if not result.commit_ready or result.approved_memory_after is None:
            raise ValueError("Cannot commit an unresolved critic result")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._state_row(connection)
            if state["status"] != "RUNNING":
                raise ValueError("Cannot commit to a non-running critic trajectory")
            if int(state["next_turn_index"]) != turn.turn_index:
                raise ValueError("Critic commit turn does not match checkpoint")
            approved_before = str(state["approved_memory"])
            before_sha256 = _text_sha256(approved_before)
            if result.approved_memory_before_sha256 != before_sha256:
                raise ValueError("Critic result before-memory hash mismatch")
            generation = int(state["replay_generation"])
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_snapshots (
                    memory_sha256, content, content_chars, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    result.approved_memory_after_sha256,
                    result.approved_memory_after,
                    len(result.approved_memory_after),
                    _now(),
                ),
            )
            connection.execute(
                """
                INSERT INTO turn_results (
                    turn_index, message_id, replay_generation, active, stage,
                    result_json, resolution_json, before_sha256, after_sha256,
                    created_at
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_index,
                    turn.message_id,
                    generation,
                    stage,
                    _encode_json(_validated_result_payload(result)),
                    _encode_json(dict(resolution_metadata or {})),
                    before_sha256,
                    result.approved_memory_after_sha256,
                    _now(),
                ),
            )
            next_turn_index = turn.turn_index + 1
            status = (
                "COMPLETED" if next_turn_index == len(self.turns) else "RUNNING"
            )
            pending_json = state["pending_repair_json"]
            if pending_json:
                pending = _decode_optional_object(pending_json)
                if pending and int(pending["target_turn_index"]) == turn.turn_index:
                    pending_json = None
            connection.execute(
                """
                UPDATE run_state
                SET next_turn_index = ?, approved_memory = ?, status = ?,
                    pending_repair_json = ?, updated_at = ?
                WHERE singleton = 1
                """,
                (
                    next_turn_index,
                    result.approved_memory_after,
                    status,
                    pending_json,
                    _now(),
                ),
            )
        return self.state()

    def rollback_to(
        self,
        *,
        target_turn: CriticSourceTurn,
        trigger_turn: CriticSourceTurn,
        request: VehicleCriticRollbackRequestPayload,
        retrieval: CriticRetrievalResult,
    ) -> CriticRunState:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._state_row(connection)
            if state["status"] != "RUNNING":
                raise ValueError("Cannot rollback a non-running trajectory")
            next_turn_index = int(state["next_turn_index"])
            if not 0 <= target_turn.turn_index < next_turn_index:
                raise ValueError("Rollback target is not in the committed prefix")
            repeated = connection.execute(
                """
                SELECT COUNT(*) FROM rollback_events
                WHERE trigger_message_id = ? AND target_message_id = ?
                """,
                (trigger_turn.message_id, target_turn.message_id),
            ).fetchone()[0]
            if int(repeated) > 0:
                raise ValueError("Repeated rollback pair would create a replay loop")
            memory_before = self._memory_before_turn(
                connection, target_turn.turn_index
            )
            old_generation = int(state["replay_generation"])
            new_generation = old_generation + 1
            connection.execute(
                "UPDATE turn_results SET active = 0 "
                "WHERE active = 1 AND turn_index >= ?",
                (target_turn.turn_index,),
            )
            pending = {
                "target_turn_index": target_turn.turn_index,
                "target_message_id": target_turn.message_id,
                "trigger_turn_index": trigger_turn.turn_index,
                "trigger_message_id": trigger_turn.message_id,
                "rollback_request": request.model_dump(mode="json"),
                "retrieval": retrieval.as_dict(),
            }
            connection.execute(
                """
                INSERT INTO rollback_events (
                    trigger_turn_index, trigger_message_id, target_turn_index,
                    target_message_id, from_generation, to_generation,
                    request_json, retrieval_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trigger_turn.turn_index,
                    trigger_turn.message_id,
                    target_turn.turn_index,
                    target_turn.message_id,
                    old_generation,
                    new_generation,
                    _encode_json(request.model_dump(mode="json")),
                    _encode_json(retrieval.as_dict()),
                    _now(),
                ),
            )
            connection.execute(
                """
                UPDATE run_state
                SET next_turn_index = ?, approved_memory = ?, status = 'RUNNING',
                    replay_generation = ?, pending_repair_json = ?, updated_at = ?
                WHERE singleton = 1
                """,
                (
                    target_turn.turn_index,
                    memory_before,
                    new_generation,
                    _encode_json(pending),
                    _now(),
                ),
            )
        return self.state()

    def mark_unresolved(
        self,
        *,
        turn: CriticSourceTurn,
        code: str,
        detail: Mapping[str, Any],
    ) -> CriticRunState:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._state_row(connection)
            if state["status"] != "RUNNING":
                raise ValueError("Cannot mark a non-running trajectory unresolved")
            connection.execute(
                """
                INSERT INTO unresolved_events (
                    turn_index, message_id, replay_generation, code,
                    detail_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_index,
                    turn.message_id,
                    int(state["replay_generation"]),
                    code,
                    _encode_json(detail),
                    _now(),
                ),
            )
            connection.execute(
                "UPDATE run_state SET status = 'UNRESOLVED', updated_at = ? "
                "WHERE singleton = 1",
                (_now(),),
            )
        return self.state()

    def rollback_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM rollback_events").fetchone()
        return int(row[0])

    def attempt_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM model_attempts").fetchone()
        return int(row[0])

    def attempts(self) -> tuple[Mapping[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM model_attempts ORDER BY id"
            ).fetchall()
        return tuple(
            {
                "turn_index": int(row["turn_index"]),
                "message_id": int(row["message_id"]),
                "replay_generation": int(row["replay_generation"]),
                "stage": row["stage"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        )

    def rollback_events(self) -> tuple[Mapping[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM rollback_events ORDER BY id"
            ).fetchall()
        return tuple(
            {
                "trigger_turn_index": int(row["trigger_turn_index"]),
                "trigger_message_id": int(row["trigger_message_id"]),
                "target_turn_index": int(row["target_turn_index"]),
                "target_message_id": int(row["target_message_id"]),
                "from_generation": int(row["from_generation"]),
                "to_generation": int(row["to_generation"]),
                "request": json.loads(row["request_json"]),
                "retrieval": json.loads(row["retrieval_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        )

    def active_results(self) -> tuple[Mapping[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT turn_results.*, memory_snapshots.content
                    AS approved_memory_after
                FROM turn_results
                JOIN memory_snapshots
                  ON memory_snapshots.memory_sha256 = turn_results.after_sha256
                WHERE turn_results.active = 1
                ORDER BY turn_results.turn_index
                """
            ).fetchall()
        return tuple(
            {
                "turn_index": int(row["turn_index"]),
                "message_id": int(row["message_id"]),
                "replay_generation": int(row["replay_generation"]),
                "stage": row["stage"],
                "result": {
                    **json.loads(row["result_json"]),
                    "approved_memory_after": row["approved_memory_after"],
                    "approved_memory_after_sha256": row["after_sha256"],
                },
                "resolution": json.loads(row["resolution_json"]),
                "before_sha256": row["before_sha256"],
                "after_sha256": row["after_sha256"],
            }
            for row in rows
        )

    def unresolved_events(self) -> tuple[Mapping[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM unresolved_events "
                "WHERE resolved_at IS NULL ORDER BY id"
            ).fetchall()
        return self._decode_unresolved_events(rows)

    def unresolved_history(self) -> tuple[Mapping[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM unresolved_events ORDER BY id"
            ).fetchall()
        return self._decode_unresolved_events(rows)

    def reopen_deferable_unresolved(self) -> Mapping[str, Any]:
        allowed = {
            "TERRA_FINAL_REJECT",
            "NO_CAUSAL_CONTEXT",
            "ROLLBACK_LIMIT_REACHED",
            "ROLLBACK_REJECTED",
            "ROLLBACK_REPAIR_REJECT",
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._state_row(connection)
            if state["status"] != "UNRESOLVED":
                raise ValueError("Critic trajectory is not unresolved")
            rows = connection.execute(
                "SELECT * FROM unresolved_events WHERE resolved_at IS NULL "
                "ORDER BY id DESC"
            ).fetchall()
            if len(rows) != 1:
                raise ValueError("Expected exactly one open unresolved event")
            row = rows[0]
            if row["code"] not in allowed:
                raise ValueError(
                    f"Unresolved code is not deferable: {row['code']}"
                )
            resolved_at = _now()
            connection.execute(
                "UPDATE unresolved_events SET resolved_at = ?, "
                "resolution_code = 'DEFER_KEEP' WHERE id = ?",
                (resolved_at, int(row["id"])),
            )
            connection.execute(
                "UPDATE run_state SET status = 'RUNNING', updated_at = ? "
                "WHERE singleton = 1",
                (resolved_at,),
            )
        return self._decode_unresolved_events((row,))[0]

    @staticmethod
    def _decode_unresolved_events(
        rows: Sequence[sqlite3.Row],
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            {
                "id": int(row["id"]),
                "turn_index": int(row["turn_index"]),
                "message_id": int(row["message_id"]),
                "replay_generation": int(row["replay_generation"]),
                "code": row["code"],
                "detail": json.loads(row["detail_json"]),
                "created_at": row["created_at"],
                "resolved_at": row["resolved_at"],
                "resolution_code": row["resolution_code"],
            }
            for row in rows
        )

    def export_labels(self, path: Path | str) -> None:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        lines = [
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in self.active_results()
        ]
        temporary_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)

    def export_review_queue(self, path: Path | str) -> None:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        lines = [
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in self.unresolved_events()
        ]
        temporary_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS run_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version TEXT NOT NULL,
                    method TEXT NOT NULL,
                    scenario_index INTEGER NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    run_config_json TEXT,
                    config_fingerprint TEXT,
                    turn_count INTEGER NOT NULL,
                    next_turn_index INTEGER NOT NULL,
                    approved_memory TEXT NOT NULL,
                    status TEXT NOT NULL,
                    replay_generation INTEGER NOT NULL,
                    pending_repair_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turn_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_index INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    replay_generation INTEGER NOT NULL,
                    active INTEGER NOT NULL CHECK (active IN (0, 1)),
                    stage TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    resolution_json TEXT NOT NULL,
                    before_sha256 TEXT NOT NULL,
                    after_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (turn_index, replay_generation)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_turn_result
                ON turn_results(turn_index) WHERE active = 1;
                CREATE TABLE IF NOT EXISTS memory_snapshots (
                    memory_sha256 TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_chars INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_index INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    replay_generation INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rollback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_turn_index INTEGER NOT NULL,
                    trigger_message_id INTEGER NOT NULL,
                    target_turn_index INTEGER NOT NULL,
                    target_message_id INTEGER NOT NULL,
                    from_generation INTEGER NOT NULL,
                    to_generation INTEGER NOT NULL,
                    request_json TEXT NOT NULL,
                    retrieval_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS unresolved_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_index INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    replay_generation INTEGER NOT NULL,
                    code TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolution_code TEXT
                );
                """
            )
            unresolved_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(unresolved_events)"
                ).fetchall()
            }
            if "resolved_at" not in unresolved_columns:
                connection.execute(
                    "ALTER TABLE unresolved_events ADD COLUMN resolved_at TEXT"
                )
            if "resolution_code" not in unresolved_columns:
                connection.execute(
                    "ALTER TABLE unresolved_events "
                    "ADD COLUMN resolution_code TEXT"
                )
            row = connection.execute(
                "SELECT * FROM run_state WHERE singleton = 1"
            ).fetchone()
            if row is None:
                now = _now()
                connection.execute(
                    """
                    INSERT INTO run_state (
                        singleton, schema_version, method, scenario_index,
                        source_fingerprint, run_config_json, config_fingerprint,
                        turn_count, next_turn_index,
                        approved_memory, status, replay_generation,
                        pending_repair_json, created_at, updated_at
                    ) VALUES (
                        1, ?, ?, ?, ?, NULL, NULL, ?, 0, '', 'RUNNING', 0,
                        NULL, ?, ?
                    )
                    """,
                    (
                        CRITIC_STATE_SCHEMA_VERSION,
                        self.method,
                        self.scenario_index,
                        self.source_fingerprint,
                        len(self.turns),
                        now,
                        now,
                    ),
                )
            else:
                expected = (
                    CRITIC_STATE_SCHEMA_VERSION,
                    self.method,
                    self.scenario_index,
                    self.source_fingerprint,
                    len(self.turns),
                )
                actual = (
                    row["schema_version"],
                    row["method"],
                    int(row["scenario_index"]),
                    row["source_fingerprint"],
                    int(row["turn_count"]),
                )
                if actual != expected:
                    raise ValueError(
                        "Critic checkpoint source/configuration does not match"
                    )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _validate_checkpoint(self) -> None:
        with self._connect() as connection:
            state = self._state_row(connection)
            status = str(state["status"])
            if status not in {"RUNNING", "COMPLETED", "UNRESOLVED"}:
                raise ValueError("Critic checkpoint has an invalid status")
            next_turn_index = int(state["next_turn_index"])
            if not 0 <= next_turn_index <= len(self.turns):
                raise ValueError("Critic checkpoint next turn is out of bounds")
            if status == "COMPLETED" and next_turn_index != len(self.turns):
                raise ValueError("Completed critic checkpoint is missing turns")
            if status != "COMPLETED" and next_turn_index == len(self.turns):
                raise ValueError("Incomplete critic checkpoint has no remaining turn")
            rows = connection.execute(
                """
                SELECT turn_results.*, memory_snapshots.content
                    AS approved_memory_after
                FROM turn_results
                JOIN memory_snapshots
                  ON memory_snapshots.memory_sha256 = turn_results.after_sha256
                WHERE turn_results.active = 1
                ORDER BY turn_results.turn_index
                """
            ).fetchall()
            indexes = [int(row["turn_index"]) for row in rows]
            if indexes != list(range(next_turn_index)):
                raise ValueError("Critic checkpoint active turn prefix is inconsistent")
            previous_memory = ""
            for row in rows:
                after = str(row["approved_memory_after"])
                if row["before_sha256"] != _text_sha256(previous_memory):
                    raise ValueError("Critic checkpoint before-memory chain is invalid")
                if row["after_sha256"] != _text_sha256(after):
                    raise ValueError("Critic checkpoint after-memory hash is invalid")
                previous_memory = after
            if previous_memory != str(state["approved_memory"]):
                raise ValueError("Critic checkpoint approved memory is inconsistent")
            pending = _decode_optional_object(state["pending_repair_json"])
            if pending is not None and (
                status != "RUNNING"
                or int(
                    pending.get("target_turn_index", -1)
                )
                != next_turn_index
            ):
                raise ValueError("Critic checkpoint pending repair is inconsistent")

    @staticmethod
    def _state_row(connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM run_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("Critic run state is missing")
        return row

    @staticmethod
    def _memory_before_turn(
        connection: sqlite3.Connection,
        turn_index: int,
    ) -> str:
        if turn_index == 0:
            return ""
        row = connection.execute(
            """
            SELECT memory_snapshots.content AS approved_memory_after
            FROM turn_results
            JOIN memory_snapshots
              ON memory_snapshots.memory_sha256 = turn_results.after_sha256
            WHERE turn_results.active = 1 AND turn_results.turn_index = ?
            """,
            (turn_index - 1,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Rollback checkpoint predecessor is missing")
        return str(row["approved_memory_after"])


class VehicleMemoryCriticPipeline:
    def __init__(
        self,
        *,
        turns: tuple[CriticSourceTurn, ...],
        store: CriticCheckpointStore,
        luna: VehicleMemoryCriticReviewer,
        terra: VehicleMemoryAdjudicator,
        max_memory_chars: int,
        retrieval_top_k: int = 8,
        retrieval_neighbor_window: int = 1,
        max_rollbacks: int = 16,
    ) -> None:
        _validate_source_turns(turns)
        if store.source_fingerprint != _source_fingerprint(turns):
            raise ValueError("Critic pipeline source differs from checkpoint")
        if max_memory_chars < 1:
            raise ValueError("Critic pipeline memory limit must be positive")
        if max_rollbacks < 0:
            raise ValueError("Critic pipeline rollback limit cannot be negative")
        self.turns = turns
        self.store = store
        self.luna = luna
        self.terra = terra
        self.max_memory_chars = max_memory_chars
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_neighbor_window = retrieval_neighbor_window
        self.max_rollbacks = max_rollbacks
        self.retriever = CausalTurnRetriever(turns)
        self.store.bind_run_config(
            {
                "state_schema_version": CRITIC_STATE_SCHEMA_VERSION,
                "max_memory_chars": max_memory_chars,
                "retrieval_top_k": retrieval_top_k,
                "retrieval_neighbor_window": retrieval_neighbor_window,
                "max_rollbacks": max_rollbacks,
                "luna": _model_contract(luna),
                "terra": _model_contract(terra),
            }
        )

    def resume_unresolved_as_defer_keep(self) -> CriticRunState:
        state = self.store.state()
        if state.status != "UNRESOLVED":
            return state
        event = self.store.reopen_deferable_unresolved()
        turn = self.turns[state.next_turn_index]
        detail = event.get("detail")
        raw_response: Mapping[str, Any] = {}
        if isinstance(detail, Mapping):
            final = detail.get("final")
            if isinstance(final, Mapping):
                candidate = final.get("response")
                if isinstance(candidate, Mapping):
                    raw_response = candidate
        reason_code = str(raw_response.get("reason_code", "AMBIGUOUS_EVIDENCE"))
        if reason_code not in REJECT_REASON_CODES:
            reason_code = "AMBIGUOUS_EVIDENCE"
        reason = str(
            raw_response.get(
                "reason",
                "The prior critic trajectory could not resolve this turn safely.",
            )
        ).strip()
        if not reason or "\n" in reason or "\r" in reason or len(reason) > 320:
            reason = "The prior critic trajectory could not resolve this turn safely."
        rejected = VehicleMemoryCriticResponse(
            method=turn.method,
            verdict="REJECT",
            decision=None,
            reason_code=reason_code,  # type: ignore[arg-type]
            reason=reason,
            metadata={
                "resumed_unresolved_event_id": event["id"],
                "resumed_unresolved_code": event["code"],
            },
        )
        self._commit_defer_keep(
            turn,
            approved_memory=state.approved_memory,
            rejected_response=rejected,
            code=str(event["code"]),
            detail={"resumed_unresolved_event_id": event["id"]},
            stage="defer_keep_resumed_unresolved",
        )
        return self.store.state()

    def run(self, *, max_commits: int | None = None) -> CriticPipelineRunResult:
        if max_commits is not None and max_commits < 1:
            raise ValueError("Critic pipeline max_commits must be positive")
        committed = 0
        while True:
            state = self.store.state()
            if state.status != "RUNNING":
                return self._run_result(state)
            if max_commits is not None and committed >= max_commits:
                return self._run_result(state)
            turn = self.turns[state.next_turn_index]
            if state.pending_repair is not None:
                self._run_pending_repair(turn, state)
                if self.store.state().status == "RUNNING":
                    committed += 1
                continue

            luna_result = self.luna.review(
                source_turn=turn,
                approved_memory=state.approved_memory,
                max_memory_chars=self.max_memory_chars,
            )
            self.store.record_attempt(
                turn=turn,
                replay_generation=state.replay_generation,
                stage="luna",
                payload=_validated_result_payload(luna_result),
            )
            if luna_result.commit_ready:
                self.store.commit_turn(
                    turn=turn,
                    result=luna_result,
                    stage="luna",
                )
                committed += 1
                continue
            action = self._run_adjudication(turn, state, luna_result)
            if action == "committed":
                committed += 1

    def _run_adjudication(
        self,
        turn: CriticSourceTurn,
        state: CriticRunState,
        luna_result: ValidatedVehicleMemoryCriticResult,
    ) -> Literal["committed", "rollback", "unresolved"]:
        initial = self.terra.adjudicate(
            source_turn=turn,
            approved_memory=state.approved_memory,
            luna_reject=luna_result,
            max_memory_chars=self.max_memory_chars,
        )
        self.store.record_attempt(
            turn=turn,
            replay_generation=state.replay_generation,
            stage="terra_initial",
            payload=initial.as_dict(),
        )
        if initial.resolution == "FINAL":
            return self._commit_or_stop_terra_final(
                turn, initial, stage="terra_initial"
            )
        if initial.resolution != "CONTEXT_REQUIRED" or (
            initial.context_request is None
        ):
            return self._stop_unresolved(
                turn,
                code="INVALID_INITIAL_ADJUDICATION",
                detail=initial.as_dict(),
            )
        retrieval = self.retriever.retrieve(
            turn,
            initial.context_request,
            top_k=self.retrieval_top_k,
            neighbor_window=self.retrieval_neighbor_window,
        )
        if not retrieval.hits:
            return self._commit_defer_keep(
                turn,
                approved_memory=state.approved_memory,
                rejected_response=luna_result.response,
                code="NO_CAUSAL_CONTEXT",
                detail={
                    "adjudication": initial.as_dict(),
                    "retrieval": retrieval.as_dict(),
                },
            )
        recovered = self.terra.adjudicate(
            source_turn=turn,
            approved_memory=state.approved_memory,
            luna_reject=luna_result,
            recovered_context=retrieval.turns,
            max_memory_chars=self.max_memory_chars,
        )
        self.store.record_attempt(
            turn=turn,
            replay_generation=state.replay_generation,
            stage="terra_recovered",
            payload={
                "adjudication": recovered.as_dict(),
                "retrieval": retrieval.as_dict(),
            },
        )
        if recovered.resolution == "FINAL":
            return self._commit_or_stop_terra_final(
                turn,
                recovered,
                stage="terra_recovered",
                retrieval=retrieval,
            )
        if recovered.resolution != "ROLLBACK_REQUIRED" or (
            recovered.rollback_request is None
        ):
            return self._stop_unresolved(
                turn,
                code="RECOVERY_NOT_RESOLVED",
                detail={
                    "adjudication": recovered.as_dict(),
                    "retrieval": retrieval.as_dict(),
                },
            )
        if self.store.rollback_count() >= self.max_rollbacks:
            return self._commit_defer_keep(
                turn,
                approved_memory=state.approved_memory,
                rejected_response=luna_result.response,
                code="ROLLBACK_LIMIT_REACHED",
                detail=recovered.as_dict(),
            )
        target = self.retriever.turn_for_message_id(
            recovered.rollback_request.source_message_id
        )
        try:
            self.store.rollback_to(
                target_turn=target,
                trigger_turn=turn,
                request=recovered.rollback_request,
                retrieval=retrieval,
            )
        except ValueError as exc:
            return self._commit_defer_keep(
                turn,
                approved_memory=state.approved_memory,
                rejected_response=luna_result.response,
                code="ROLLBACK_REJECTED",
                detail={"error": str(exc), "adjudication": recovered.as_dict()},
            )
        return "rollback"

    def _run_pending_repair(
        self,
        turn: CriticSourceTurn,
        state: CriticRunState,
    ) -> None:
        pending = dict(state.pending_repair or {})
        if int(pending.get("target_turn_index", -1)) != turn.turn_index:
            self._stop_unresolved(
                turn,
                code="INVALID_PENDING_REPAIR",
                detail=pending,
            )
            return
        request = VehicleCriticRollbackRequestPayload.model_validate(
            pending.get("rollback_request")
        )
        repair = self.terra.repair_omission(
            source_turn=turn,
            approved_memory=state.approved_memory,
            rollback_request=request,
            trigger_message_id=int(pending["trigger_message_id"]),
            max_memory_chars=self.max_memory_chars,
        )
        self.store.record_attempt(
            turn=turn,
            replay_generation=state.replay_generation,
            stage="terra_rollback_repair",
            payload=_validated_result_payload(repair),
        )
        if not repair.commit_ready and repair.response.verdict == "REJECT":
            self._commit_defer_keep(
                turn,
                approved_memory=state.approved_memory,
                rejected_response=repair.response,
                code="ROLLBACK_REPAIR_REJECT",
                detail={
                    "repair": _validated_result_payload(repair),
                    "pending": pending,
                },
                stage="defer_keep_rollback_repair",
            )
            return
        if (
            not repair.commit_ready
            or repair.response.verdict != "CORRECT"
            or repair.response.decision != "UPDATE"
        ):
            self._stop_unresolved(
                turn,
                code="ROLLBACK_REPAIR_FAILED",
                detail=_validated_result_payload(repair),
            )
            return
        self.store.commit_turn(
            turn=turn,
            result=repair,
            stage="terra_rollback_repair",
            resolution_metadata=pending,
        )

    def _commit_or_stop_terra_final(
        self,
        turn: CriticSourceTurn,
        adjudication: VehicleMemoryAdjudication,
        *,
        stage: str,
        retrieval: CriticRetrievalResult | None = None,
    ) -> Literal["committed", "unresolved"]:
        result = adjudication.final_result
        if (
            result is not None
            and not result.commit_ready
            and result.response.verdict == "REJECT"
        ):
            return self._commit_defer_keep(
                turn,
                approved_memory=self.store.state().approved_memory,
                rejected_response=result.response,
                code="TERRA_FINAL_REJECT",
                detail={
                    "adjudication": adjudication.as_dict(),
                    "retrieval": (
                        retrieval.as_dict() if retrieval is not None else None
                    ),
                },
                stage=f"{stage}_defer_keep",
            )
        if result is None or not result.commit_ready:
            return self._stop_unresolved(
                turn,
                code="TERRA_FINAL_REJECT",
                detail=adjudication.as_dict(),
            )
        resolution = {
            "adjudication": adjudication.as_dict(),
            "retrieval": retrieval.as_dict() if retrieval is not None else None,
        }
        self.store.commit_turn(
            turn=turn,
            result=result,
            stage=stage,
            resolution_metadata=resolution,
        )
        return "committed"

    def _commit_defer_keep(
        self,
        turn: CriticSourceTurn,
        *,
        approved_memory: str,
        rejected_response: VehicleMemoryCriticResponse,
        code: str,
        detail: Mapping[str, Any],
        stage: str = "defer_keep",
    ) -> Literal["committed"]:
        deferred = build_defer_keep_result(
            source_turn=turn,
            approved_memory=approved_memory,
            rejected_response=rejected_response,
            defer_code=code,
            max_memory_chars=self.max_memory_chars,
        )
        self.store.commit_turn(
            turn=turn,
            result=deferred,
            stage=stage,
            resolution_metadata={
                "defer_keep": True,
                "defer_code": code,
                "source_reject": rejected_response.as_dict(),
                "detail": dict(detail),
            },
        )
        return "committed"

    def _stop_unresolved(
        self,
        turn: CriticSourceTurn,
        *,
        code: str,
        detail: Mapping[str, Any],
    ) -> Literal["unresolved"]:
        self.store.mark_unresolved(turn=turn, code=code, detail=detail)
        return "unresolved"

    def _run_result(self, state: CriticRunState) -> CriticPipelineRunResult:
        return CriticPipelineRunResult(
            method=state.method,
            scenario_index=state.scenario_index,
            status=state.status,
            processed_turns=state.next_turn_index,
            turn_count=state.turn_count,
            replay_generation=state.replay_generation,
            final_memory=state.approved_memory,
            final_memory_sha256=_text_sha256(state.approved_memory),
            attempt_count=self.store.attempt_count(),
            rollback_count=self.store.rollback_count(),
        )


def _validate_source_turns(turns: tuple[CriticSourceTurn, ...]) -> None:
    if not turns:
        raise ValueError("Critic source turns cannot be empty")
    method = turns[0].method
    scenario = turns[0].scenario_index
    for expected, turn in enumerate(turns):
        if turn.turn_index != expected:
            raise ValueError("Critic source turn indexes must be contiguous")
        if turn.method != method or turn.scenario_index != scenario:
            raise ValueError("Critic source turns cross method or scenario")
        if expected and turn.message_id <= turns[expected - 1].message_id:
            raise ValueError("Critic source message IDs must increase")


def _source_fingerprint(turns: tuple[CriticSourceTurn, ...]) -> str:
    digest = hashlib.sha256()
    for turn in turns:
        digest.update(
            _encode_json(
                {
                    "method": turn.method,
                    "scenario_index": turn.scenario_index,
                    "turn_index": turn.turn_index,
                    "message_id": turn.message_id,
                    "content": turn.content,
                    "original_decision": turn.original_decision,
                    "original_before_sha256": turn.original_before_sha256,
                    "original_after_sha256": turn.original_after_sha256,
                }
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _validated_result_payload(
    result: ValidatedVehicleMemoryCriticResult,
) -> dict[str, Any]:
    return {
        "response": result.response.as_dict(),
        "approved_memory_before_sha256": result.approved_memory_before_sha256,
        "approved_memory_after_sha256": result.approved_memory_after_sha256,
        "patch_stats": dict(result.patch_stats),
        "commit_ready": result.commit_ready,
    }


def _model_contract(model: object) -> dict[str, Any]:
    instructions = getattr(model, "instructions", None)
    repair_instructions = getattr(model, "repair_instructions", None)
    allowlist = getattr(model, "pii_allowlist", ())
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_id": getattr(model, "model_id", None),
        "prompt_version": getattr(model, "prompt_version", None),
        "repair_prompt_version": getattr(model, "repair_prompt_version", None),
        "reasoning_effort": getattr(model, "reasoning_effort", None),
        "max_output_tokens": getattr(model, "max_output_tokens", None),
        "max_semantic_retries": getattr(model, "max_semantic_retries", None),
        "redact_pii": getattr(model, "redact_pii", None),
        "instructions_sha256": (
            _text_sha256(instructions) if isinstance(instructions, str) else None
        ),
        "repair_instructions_sha256": (
            _text_sha256(repair_instructions)
            if isinstance(repair_instructions, str)
            else None
        ),
        "pii_allowlist_sha256": _text_sha256(
            json.dumps(tuple(allowlist), ensure_ascii=False)
        ),
    }


def _encode_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_optional_object(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise ValueError("Critic checkpoint JSON field is not an object")
    return decoded


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()
