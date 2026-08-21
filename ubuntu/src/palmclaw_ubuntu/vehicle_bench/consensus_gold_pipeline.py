from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from palmclaw_ubuntu.vehicle_bench.consensus_gold import (
    CONSENSUS_GOLD_PROMPT_VERSION,
    CONSENSUS_GOLD_SCHEMA_VERSION,
    CandidateGateResult,
    CompactionConsensusResult,
    CompactionGateResult,
    CompactionInput,
    ConsensusResult,
    GeneratedCombinedCandidate,
    GeneratedCompactionCandidate,
    GoldGenerationInput,
    build_compaction_consensus,
    build_compaction_input,
    build_consensus,
    build_gold_generation_input,
    evaluate_candidate,
    evaluate_compaction_candidate,
    text_sha256,
)
from palmclaw_ubuntu.vehicle_bench.memory import VehicleHistoryEntry

GOLD_CHECKPOINT_SCHEMA_VERSION = "vehiclemembench-v2-consensus-checkpoint-v1"
GOLD_LABEL_VERSION = "vehiclemembench-v2-consensus-turn-label-v1"

type GoldRunStatus = Literal["RUNNING", "PAUSED_REVIEW", "COMPLETED", "FAILED"]


class CandidateGenerator(Protocol):
    def generate(
        self,
        generation_input: GoldGenerationInput,
        *,
        sample_id: Literal["A", "B", "C"],
    ) -> GeneratedCombinedCandidate: ...


class CandidateResolver(Protocol):
    def resolve(
        self,
        *,
        generation_input: GoldGenerationInput,
        gate_results: Sequence[CandidateGateResult],
        consensus: ConsensusResult,
    ) -> GeneratedCombinedCandidate | None: ...


class CompactionGenerator(Protocol):
    def generate(
        self,
        compaction_input: CompactionInput,
        *,
        sample_id: Literal["A", "B", "C"],
    ) -> GeneratedCompactionCandidate: ...


class CompactionResolver(Protocol):
    def resolve(
        self,
        *,
        compaction_input: CompactionInput,
        gate_results: Sequence[CompactionGateResult],
        consensus: CompactionConsensusResult,
    ) -> GeneratedCompactionCandidate | None: ...


class GoldTurnEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=1_024)


class GoldTurnLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = GOLD_LABEL_VERSION
    scenario_index: int = Field(ge=1)
    turn_index: int = Field(ge=0)
    message_id: int = Field(ge=1)
    date: str
    decision: Literal["NO_OP", "UPDATE"]
    operations: list[dict[str, str]]
    reason: str = Field(min_length=1, max_length=320)
    evidence: list[GoldTurnEvidence]
    validation_path: tuple[str, ...]
    before_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    after_memory: str
    patch_stats: dict[str, int]
    add_count_since_compaction: int = Field(ge=0)
    compaction_triggered: bool = False


@dataclass(frozen=True)
class GoldRunState:
    scenario_index: int
    source_fingerprint: str
    turn_count: int
    next_turn_index: int
    approved_memory: str
    approved_memory_sha256: str
    add_count_since_compaction: int
    status: GoldRunStatus
    pending_review: Mapping[str, Any] | None


@dataclass(frozen=True)
class GoldPipelineRunResult:
    scenario_index: int
    status: GoldRunStatus
    processed_turns: int
    turn_count: int
    final_memory: str
    final_memory_sha256: str
    add_count_since_compaction: int


class GoldCheckpointStore:
    """Transactional state for one Scenario-level consensus-Gold trajectory."""

    def __init__(
        self,
        path: Path | str,
        *,
        scenario_index: int,
        entries: Sequence[VehicleHistoryEntry],
    ) -> None:
        if scenario_index < 1:
            raise ValueError("Gold checkpoint scenario index must be positive")
        if not entries:
            raise ValueError("Gold checkpoint requires source turns")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.scenario_index = scenario_index
        self.entries = tuple(entries)
        self.source_fingerprint = _source_fingerprint(scenario_index, self.entries)
        self._initialize()
        self._validate()

    def state(self) -> GoldRunState:
        with self._connect() as connection:
            row = self._state_row(connection)
        return _state_from_row(row)

    def bind_run_config(self, config: Mapping[str, Any]) -> None:
        encoded = _encode_json(dict(config))
        fingerprint = text_sha256(encoded)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._state_row(connection)
            recorded = row["config_fingerprint"]
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
                raise ValueError("Gold checkpoint run configuration does not match")

    def record_gate_result(
        self,
        *,
        turn_index: int,
        stage: str,
        attempt_index: int,
        input_sha256: str,
        result: CandidateGateResult,
    ) -> None:
        self._record_attempt(
            turn_index=turn_index,
            stage=stage,
            sample_id=result.sample_id,
            attempt_index=attempt_index,
            input_sha256=input_sha256,
            status=result.status,
            payload=result.model_dump(mode="json"),
        )

    def record_compaction_gate_result(
        self,
        *,
        turn_index: int,
        stage: str,
        attempt_index: int,
        input_sha256: str,
        result: CompactionGateResult,
    ) -> None:
        self._record_attempt(
            turn_index=turn_index,
            stage=stage,
            sample_id=result.sample_id,
            attempt_index=attempt_index,
            input_sha256=input_sha256,
            status=result.status,
            payload=result.model_dump(mode="json"),
        )

    def record_error(
        self,
        *,
        turn_index: int,
        stage: str,
        sample_id: str,
        attempt_index: int,
        input_sha256: str,
        error: Exception,
    ) -> None:
        self._record_attempt(
            turn_index=turn_index,
            stage=stage,
            sample_id=sample_id,
            attempt_index=attempt_index,
            input_sha256=input_sha256,
            status="ERROR",
            payload={"error_type": type(error).__name__, "message": str(error)},
        )

    def cached_pass(
        self,
        *,
        turn_index: int,
        stage: str,
        sample_id: str,
        input_sha256: str,
    ) -> CandidateGateResult | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM model_attempts
                WHERE turn_index = ? AND stage = ? AND sample_id = ?
                  AND input_sha256 = ? AND status = 'PASS'
                ORDER BY attempt_index DESC, id DESC
                LIMIT 1
                """,
                (turn_index, stage, sample_id, input_sha256),
            ).fetchone()
        if row is None:
            return None
        return CandidateGateResult.model_validate_json(row["payload_json"])

    def cached_compaction_pass(
        self,
        *,
        turn_index: int,
        stage: str,
        sample_id: str,
        input_sha256: str,
    ) -> CompactionGateResult | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM model_attempts
                WHERE turn_index = ? AND stage = ? AND sample_id = ?
                  AND input_sha256 = ? AND status = 'PASS'
                ORDER BY attempt_index DESC, id DESC
                LIMIT 1
                """,
                (turn_index, stage, sample_id, input_sha256),
            ).fetchone()
        if row is None:
            return None
        return CompactionGateResult.model_validate_json(row["payload_json"])

    def attempt_count(
        self,
        *,
        turn_index: int,
        stage: str,
        sample_id: str,
        input_sha256: str,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM model_attempts
                WHERE turn_index = ? AND stage = ? AND sample_id = ?
                  AND input_sha256 = ?
                """,
                (turn_index, stage, sample_id, input_sha256),
            ).fetchone()
        return int(row["count"])

    def record_consensus(
        self,
        *,
        turn_index: int,
        input_sha256: str,
        consensus: BaseModel,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO consensus_results (
                    turn_index, input_sha256, payload_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    turn_index,
                    input_sha256,
                    _encode_json(consensus.model_dump(mode="json")),
                    _now(),
                ),
            )

    def commit_turn(
        self,
        label: GoldTurnLabel,
        *,
        allow_paused_review: bool = False,
    ) -> GoldRunState:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._state_row(connection)
            allowed_statuses = (
                {"RUNNING", "PAUSED_REVIEW"}
                if allow_paused_review
                else {"RUNNING"}
            )
            if state["status"] not in allowed_statuses:
                raise ValueError("Cannot commit to a non-running Gold trajectory")
            if int(state["next_turn_index"]) != label.turn_index:
                raise ValueError("Gold commit turn does not match checkpoint")
            if str(state["approved_memory_sha256"]) != label.before_memory_sha256:
                raise ValueError("Gold label before-memory hash mismatch")
            if text_sha256(label.after_memory) != label.after_memory_sha256:
                raise ValueError("Gold label after-memory hash mismatch")
            connection.execute(
                """
                INSERT INTO turn_results (
                    turn_index, message_id, label_json,
                    before_memory_sha256, after_memory_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    label.turn_index,
                    label.message_id,
                    _encode_json(label.model_dump(mode="json")),
                    label.before_memory_sha256,
                    label.after_memory_sha256,
                    _now(),
                ),
            )
            next_turn = label.turn_index + 1
            status = "COMPLETED" if next_turn == len(self.entries) else "RUNNING"
            connection.execute(
                """
                UPDATE run_state
                SET next_turn_index = ?, approved_memory = ?,
                    approved_memory_sha256 = ?, add_count_since_compaction = ?,
                    status = ?, pending_review_json = NULL, updated_at = ?
                WHERE singleton = 1
                """,
                (
                    next_turn,
                    label.after_memory,
                    label.after_memory_sha256,
                    label.add_count_since_compaction,
                    status,
                    _now(),
                ),
            )
            updated = self._state_row(connection)
        return _state_from_row(updated)

    def pause_review(
        self,
        *,
        turn_index: int,
        reason: str,
        payload: Mapping[str, Any],
    ) -> GoldRunState:
        review = {"turn_index": turn_index, "reason": reason, "payload": dict(payload)}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._state_row(connection)
            if int(state["next_turn_index"]) != turn_index:
                raise ValueError("Gold review turn does not match checkpoint")
            connection.execute(
                """
                UPDATE run_state
                SET status = 'PAUSED_REVIEW', pending_review_json = ?, updated_at = ?
                WHERE singleton = 1
                """,
                (_encode_json(review), _now()),
            )
            updated = self._state_row(connection)
        return _state_from_row(updated)

    def active_labels(self) -> tuple[GoldTurnLabel, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT label_json FROM turn_results ORDER BY turn_index"
            ).fetchall()
        return tuple(
            GoldTurnLabel.model_validate_json(row["label_json"]) for row in rows
        )

    def attempts(self) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM model_attempts ORDER BY id"
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def _record_attempt(
        self,
        *,
        turn_index: int,
        stage: str,
        sample_id: str,
        attempt_index: int,
        input_sha256: str,
        status: str,
        payload: Mapping[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_attempts (
                    turn_index, stage, sample_id, attempt_index,
                    input_sha256, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_index,
                    stage,
                    sample_id,
                    attempt_index,
                    input_sha256,
                    status,
                    _encode_json(dict(payload)),
                    _now(),
                ),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS run_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version TEXT NOT NULL,
                    scenario_index INTEGER NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    turn_count INTEGER NOT NULL,
                    next_turn_index INTEGER NOT NULL,
                    approved_memory TEXT NOT NULL,
                    approved_memory_sha256 TEXT NOT NULL,
                    add_count_since_compaction INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    pending_review_json TEXT,
                    run_config_json TEXT,
                    config_fingerprint TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_index INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS model_attempt_lookup
                ON model_attempts (
                    turn_index, stage, sample_id, input_sha256, attempt_index
                );
                CREATE TABLE IF NOT EXISTS consensus_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_index INTEGER NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turn_results (
                    turn_index INTEGER PRIMARY KEY,
                    message_id INTEGER NOT NULL,
                    label_json TEXT NOT NULL,
                    before_memory_sha256 TEXT NOT NULL,
                    after_memory_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            row = connection.execute(
                "SELECT singleton FROM run_state WHERE singleton = 1"
            ).fetchone()
            if row is None:
                empty_sha256 = text_sha256("")
                connection.execute(
                    """
                    INSERT INTO run_state (
                        singleton, schema_version, scenario_index,
                        source_fingerprint, turn_count, next_turn_index,
                        approved_memory, approved_memory_sha256,
                        add_count_since_compaction, status, updated_at
                    ) VALUES (1, ?, ?, ?, ?, 0, '', ?, 0, 'RUNNING', ?)
                    """,
                    (
                        GOLD_CHECKPOINT_SCHEMA_VERSION,
                        self.scenario_index,
                        self.source_fingerprint,
                        len(self.entries),
                        empty_sha256,
                        _now(),
                    ),
                )

    def _validate(self) -> None:
        state = self.state()
        if state.scenario_index != self.scenario_index:
            raise ValueError("Gold checkpoint scenario does not match source")
        if state.source_fingerprint != self.source_fingerprint:
            raise ValueError("Gold checkpoint source fingerprint does not match")
        if state.turn_count != len(self.entries):
            raise ValueError("Gold checkpoint turn count does not match source")
        if not 0 <= state.next_turn_index <= state.turn_count:
            raise ValueError("Gold checkpoint next turn is out of bounds")
        if text_sha256(state.approved_memory) != state.approved_memory_sha256:
            raise ValueError("Gold checkpoint approved-memory hash is invalid")
        labels = self.active_labels()
        if len(labels) != state.next_turn_index:
            raise ValueError("Gold checkpoint committed turn prefix is inconsistent")
        previous_sha256 = text_sha256("")
        for index, label in enumerate(labels):
            if label.turn_index != index:
                raise ValueError("Gold checkpoint turn sequence is invalid")
            if label.before_memory_sha256 != previous_sha256:
                raise ValueError("Gold checkpoint memory chain is invalid")
            previous_sha256 = label.after_memory_sha256
        if previous_sha256 != state.approved_memory_sha256:
            raise ValueError("Gold checkpoint final memory is inconsistent")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _state_row(connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM run_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("Gold checkpoint state is missing")
        return row


class ConsensusGoldPipeline:
    def __init__(
        self,
        *,
        scenario_index: int,
        entries: Sequence[VehicleHistoryEntry],
        store: GoldCheckpointStore,
        luna: CandidateGenerator,
        terra: CandidateResolver,
        sol: CandidateResolver,
        compactor_luna: CompactionGenerator | None = None,
        compactor_terra: CompactionResolver | None = None,
        compactor_sol: CompactionResolver | None = None,
        max_candidate_attempts: int = 2,
        max_memory_chars: int = 8_192,
        compaction_add_threshold: int = 30,
        compaction_target_ratio: float = 0.70,
    ) -> None:
        if max_candidate_attempts < 1:
            raise ValueError("Gold candidate attempts must be positive")
        if max_memory_chars < 1:
            raise ValueError("Gold memory character limit must be positive")
        if compaction_add_threshold < 1:
            raise ValueError("Gold compaction threshold must be positive")
        if not 0 < compaction_target_ratio < 1:
            raise ValueError("Gold compaction ratio must be between zero and one")
        self.scenario_index = scenario_index
        self.entries = tuple(entries)
        self.store = store
        self.luna = luna
        self.terra = terra
        self.sol = sol
        self.compactor_luna = compactor_luna
        self.compactor_terra = compactor_terra
        self.compactor_sol = compactor_sol
        self.max_candidate_attempts = max_candidate_attempts
        self.max_memory_chars = max_memory_chars
        self.compaction_add_threshold = compaction_add_threshold
        self.compaction_target_ratio = compaction_target_ratio
        self.store.bind_run_config(
            {
                "schema_version": CONSENSUS_GOLD_SCHEMA_VERSION,
                "prompt_version": CONSENSUS_GOLD_PROMPT_VERSION,
                "max_candidate_attempts": max_candidate_attempts,
                "max_memory_chars": max_memory_chars,
                "compaction_add_threshold": compaction_add_threshold,
                "compaction_target_ratio": compaction_target_ratio,
                "luna": _component_identity(luna),
                "terra": _component_identity(terra),
                "sol": _component_identity(sol),
                "compactor_luna": _component_identity(compactor_luna),
                "compactor_terra": _component_identity(compactor_terra),
                "compactor_sol": _component_identity(compactor_sol),
            }
        )

    def run(self, *, max_commits: int | None = None) -> GoldPipelineRunResult:
        if max_commits is not None and max_commits < 1:
            raise ValueError("Gold max commits must be positive")
        state = self.store.state()
        if state.status != "RUNNING":
            return self._result(state, 0)
        processed = 0
        while state.status == "RUNNING" and state.next_turn_index < len(self.entries):
            if max_commits is not None and processed >= max_commits:
                break
            state = self._run_turn(state)
            if state.status == "PAUSED_REVIEW":
                break
            processed += 1
        return self._result(state, processed)

    def resume_human_candidate(
        self,
        candidate: GeneratedCombinedCandidate,
    ) -> GoldRunState:
        state = self.store.state()
        if state.status != "PAUSED_REVIEW":
            raise ValueError("Gold trajectory is not waiting for Human Review")
        turn_index = state.next_turn_index
        generation_input = build_gold_generation_input(
            scenario_index=self.scenario_index,
            turn_index=turn_index,
            entry=self.entries[turn_index],
            previous_memory=state.approved_memory,
        )
        selected = evaluate_candidate(
            candidate,
            generation_input=generation_input,
            max_memory_chars=self.max_memory_chars,
        )
        self.store.record_gate_result(
            turn_index=turn_index,
            stage="HUMAN",
            attempt_index=1,
            input_sha256=generation_input.input_sha256,
            result=selected,
        )
        if selected.status != "PASS" or selected.after_memory is None:
            raise ValueError(
                "Human Review candidate failed rule validation: "
                f"{list(selected.error_codes)}"
            )
        return self._finalize_selected(
            state=state,
            generation_input=generation_input,
            selected=selected,
            validation_path=("human_review", "rule_validator"),
            allow_paused_review=True,
        )

    def resume_human_compaction(
        self,
        candidate: GeneratedCompactionCandidate,
    ) -> GoldRunState:
        state = self.store.state()
        pending = state.pending_review
        if state.status != "PAUSED_REVIEW" or pending is None:
            raise ValueError("Gold trajectory is not waiting for Human Review")
        payload = pending.get("payload", {})
        if payload.get("review_kind") != "COMPACTION":
            raise ValueError("Gold trajectory is not waiting for compaction review")
        turn_index = state.next_turn_index
        generation_input = build_gold_generation_input(
            scenario_index=self.scenario_index,
            turn_index=turn_index,
            entry=self.entries[turn_index],
            previous_memory=state.approved_memory,
        )
        selected = CandidateGateResult.model_validate(
            payload.get("selected_turn_result")
        )
        selected = evaluate_candidate(
            selected.candidate,
            generation_input=generation_input,
            max_memory_chars=self.max_memory_chars,
        )
        if selected.status != "PASS" or selected.after_memory is None:
            raise ValueError("Stored turn candidate failed replay validation")
        compaction_input = build_compaction_input(
            scenario_index=self.scenario_index,
            turn_index=turn_index,
            post_patch_memory=selected.after_memory,
            target_ratio=self.compaction_target_ratio,
        )
        compacted = evaluate_compaction_candidate(
            candidate,
            compaction_input=compaction_input,
            max_memory_chars=self.max_memory_chars,
        )
        self.store.record_compaction_gate_result(
            turn_index=turn_index,
            stage="COMPACTION_HUMAN",
            attempt_index=1,
            input_sha256=compaction_input.input_sha256,
            result=compacted,
        )
        if compacted.status != "PASS" or compacted.after_memory is None:
            raise ValueError(
                "Human compaction failed rule validation: "
                f"{list(compacted.error_codes)}"
            )
        validation_path = tuple(payload.get("turn_validation_path", ())) + (
            "compaction_human_review",
            "compaction_rule_validator",
        )
        return self._commit_selected(
            generation_input=generation_input,
            selected=selected,
            validation_path=validation_path,
            final_memory=compacted.after_memory,
            final_memory_sha256=str(compacted.after_memory_sha256),
            next_add_count=0,
            compaction_triggered=True,
            allow_paused_review=True,
        )

    def _run_turn(self, state: GoldRunState) -> GoldRunState:
        turn_index = state.next_turn_index
        entry = self.entries[turn_index]
        generation_input = build_gold_generation_input(
            scenario_index=self.scenario_index,
            turn_index=turn_index,
            entry=entry,
            previous_memory=state.approved_memory,
        )
        gate_results = self._luna_results(generation_input)
        if len(gate_results) != 3:
            return self.store.pause_review(
                turn_index=turn_index,
                reason="LUNA_GENERATION_FAILED",
                payload={
                    "review_kind": "TURN",
                    "available_results": len(gate_results),
                },
            )
        consensus = build_consensus(gate_results)
        self.store.record_consensus(
            turn_index=turn_index,
            input_sha256=generation_input.input_sha256,
            consensus=consensus,
        )
        selected: CandidateGateResult | None = None
        validation_path: tuple[str, ...]
        if consensus.route == "AUTO_ACCEPT":
            selected = next(
                result
                for result in gate_results
                if result.sample_id == consensus.accepted_sample_id
            )
            validation_path = ("luna_consensus", "rule_validator")
        else:
            selected = self._resolve(
                resolver=self.terra,
                stage="TERRA",
                generation_input=generation_input,
                gate_results=gate_results,
                consensus=consensus,
            )
            validation_path = ("luna_consensus", "terra", "rule_validator")
            if selected is None:
                selected = self._resolve(
                    resolver=self.sol,
                    stage="SOL",
                    generation_input=generation_input,
                    gate_results=gate_results,
                    consensus=consensus,
                )
                validation_path = (
                    "luna_consensus",
                    "terra",
                    "sol",
                    "rule_validator",
                )
        if selected is None or selected.after_memory is None:
            return self.store.pause_review(
                turn_index=turn_index,
                reason="ESCALATION_UNRESOLVED",
                payload={
                    "review_kind": "TURN",
                    "consensus": consensus.model_dump(mode="json"),
                    "gate_results": [
                        result.model_dump(mode="json") for result in gate_results
                    ],
                },
            )

        return self._finalize_selected(
            state=state,
            generation_input=generation_input,
            selected=selected,
            validation_path=validation_path,
        )

    def _finalize_selected(
        self,
        *,
        state: GoldRunState,
        generation_input: GoldGenerationInput,
        selected: CandidateGateResult,
        validation_path: tuple[str, ...],
        allow_paused_review: bool = False,
    ) -> GoldRunState:
        turn_index = generation_input.turn_index
        if selected.after_memory is None:
            raise ValueError("Cannot finalize a candidate without resulting memory")
        next_add_count = state.add_count_since_compaction + int(
            selected.patch_stats.get("add_count", 0)
        )
        final_memory = selected.after_memory
        final_memory_sha256 = str(selected.after_memory_sha256)
        compaction_triggered = False
        if next_add_count >= self.compaction_add_threshold:
            compaction_review_payload = {
                "review_kind": "COMPACTION",
                "selected_turn_result": selected.model_dump(mode="json"),
                "turn_validation_path": list(validation_path),
                "post_patch_memory": selected.after_memory,
                "post_patch_memory_sha256": selected.after_memory_sha256,
                "add_count_since_compaction": next_add_count,
            }
            if self.compactor_luna is None or self.compactor_sol is None:
                return self.store.pause_review(
                    turn_index=turn_index,
                    reason="COMPACTION_MODELS_NOT_CONFIGURED",
                    payload=compaction_review_payload,
                )
            compaction_input = build_compaction_input(
                scenario_index=self.scenario_index,
                turn_index=turn_index,
                post_patch_memory=selected.after_memory,
                target_ratio=self.compaction_target_ratio,
            )
            compaction_results = self._compaction_luna_results(compaction_input)
            if len(compaction_results) != 3:
                return self.store.pause_review(
                    turn_index=turn_index,
                    reason="COMPACTION_GENERATION_FAILED",
                    payload={
                        **compaction_review_payload,
                        "available_results": len(compaction_results),
                        "gate_results": [
                            result.model_dump(mode="json")
                            for result in compaction_results
                        ],
                    },
                )
            compaction_consensus = build_compaction_consensus(compaction_results)
            self.store.record_consensus(
                turn_index=turn_index,
                input_sha256=compaction_input.input_sha256,
                consensus=compaction_consensus,
            )
            compacted: CompactionGateResult | None = None
            if compaction_consensus.route == "AUTO_ACCEPT":
                compacted = next(
                    result
                    for result in compaction_results
                    if result.sample_id == compaction_consensus.accepted_sample_id
                )
                validation_path += (
                    "compaction_luna_consensus",
                    "compaction_rule_validator",
                )
            else:
                compacted = self._resolve_compaction(
                    resolver=self.compactor_sol,
                    stage="COMPACTION_SOL",
                    compaction_input=compaction_input,
                    gate_results=compaction_results,
                    consensus=compaction_consensus,
                )
                validation_path += (
                    "compaction_luna_consensus",
                    "compaction_sol",
                    "compaction_rule_validator",
                )
            if compacted is None or compacted.after_memory is None:
                return self.store.pause_review(
                    turn_index=turn_index,
                    reason="COMPACTION_ESCALATION_UNRESOLVED",
                    payload={
                        **compaction_review_payload,
                        "turn_validation_path": list(validation_path),
                        "consensus": compaction_consensus.model_dump(mode="json"),
                        "gate_results": [
                            result.model_dump(mode="json")
                            for result in compaction_results
                        ],
                    },
                )
            final_memory = compacted.after_memory
            final_memory_sha256 = str(compacted.after_memory_sha256)
            next_add_count = 0
            compaction_triggered = True

        return self._commit_selected(
            generation_input=generation_input,
            selected=selected,
            validation_path=validation_path,
            final_memory=final_memory,
            final_memory_sha256=final_memory_sha256,
            next_add_count=next_add_count,
            compaction_triggered=compaction_triggered,
            allow_paused_review=allow_paused_review,
        )

    def _commit_selected(
        self,
        *,
        generation_input: GoldGenerationInput,
        selected: CandidateGateResult,
        validation_path: tuple[str, ...],
        final_memory: str,
        final_memory_sha256: str,
        next_add_count: int,
        compaction_triggered: bool,
        allow_paused_review: bool,
    ) -> GoldRunState:
        payload = selected.candidate.payload
        label = GoldTurnLabel(
            scenario_index=self.scenario_index,
            turn_index=generation_input.turn_index,
            message_id=generation_input.message_id,
            date=generation_input.date,
            decision=payload.decision,
            operations=[operation.model_dump() for operation in payload.operations],
            reason=payload.reason,
            evidence=[
                GoldTurnEvidence(
                    message_id=generation_input.message_id,
                    quote=evidence.quote,
                )
                for evidence in payload.evidence
            ],
            validation_path=validation_path,
            before_memory_sha256=selected.before_memory_sha256,
            after_memory_sha256=final_memory_sha256,
            after_memory=final_memory,
            patch_stats=selected.patch_stats,
            add_count_since_compaction=next_add_count,
            compaction_triggered=compaction_triggered,
        )
        return self.store.commit_turn(
            label,
            allow_paused_review=allow_paused_review,
        )

    def _luna_results(
        self,
        generation_input: GoldGenerationInput,
    ) -> tuple[CandidateGateResult, ...]:
        sample_ids: tuple[Literal["A", "B", "C"], ...] = ("A", "B", "C")
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = tuple(
                executor.map(
                    lambda sample_id: self._luna_result(
                        generation_input,
                        sample_id,
                    ),
                    sample_ids,
                )
            )
        return tuple(result for result in results if result is not None)

    def _luna_result(
        self,
        generation_input: GoldGenerationInput,
        sample_id: Literal["A", "B", "C"],
    ) -> CandidateGateResult | None:
        cached = self.store.cached_pass(
            turn_index=generation_input.turn_index,
            stage="LUNA",
            sample_id=sample_id,
            input_sha256=generation_input.input_sha256,
        )
        if cached is not None:
            return cached
        latest: CandidateGateResult | None = None
        attempts = self.store.attempt_count(
            turn_index=generation_input.turn_index,
            stage="LUNA",
            sample_id=sample_id,
            input_sha256=generation_input.input_sha256,
        )
        while attempts < self.max_candidate_attempts:
            attempt_index = attempts + 1
            try:
                candidate = self.luna.generate(
                    generation_input,
                    sample_id=sample_id,
                )
                latest = evaluate_candidate(
                    candidate,
                    generation_input=generation_input,
                    max_memory_chars=self.max_memory_chars,
                )
                self.store.record_gate_result(
                    turn_index=generation_input.turn_index,
                    stage="LUNA",
                    attempt_index=attempt_index,
                    input_sha256=generation_input.input_sha256,
                    result=latest,
                )
            except (RuntimeError, ValueError) as error:
                self.store.record_error(
                    turn_index=generation_input.turn_index,
                    stage="LUNA",
                    sample_id=sample_id,
                    attempt_index=attempt_index,
                    input_sha256=generation_input.input_sha256,
                    error=error,
                )
            attempts += 1
            if latest is not None and latest.status == "PASS":
                break
        return latest

    def _resolve(
        self,
        *,
        resolver: CandidateResolver,
        stage: Literal["TERRA", "SOL"],
        generation_input: GoldGenerationInput,
        gate_results: Sequence[CandidateGateResult],
        consensus: ConsensusResult,
    ) -> CandidateGateResult | None:
        try:
            candidate = resolver.resolve(
                generation_input=generation_input,
                gate_results=gate_results,
                consensus=consensus,
            )
        except (RuntimeError, ValueError) as error:
            self.store.record_error(
                turn_index=generation_input.turn_index,
                stage=stage,
                sample_id=stage,
                attempt_index=1,
                input_sha256=generation_input.input_sha256,
                error=error,
            )
            return None
        if candidate is None:
            return None
        result = evaluate_candidate(
            candidate,
            generation_input=generation_input,
            max_memory_chars=self.max_memory_chars,
        )
        self.store.record_gate_result(
            turn_index=generation_input.turn_index,
            stage=stage,
            attempt_index=1,
            input_sha256=generation_input.input_sha256,
            result=result,
        )
        return result if result.status == "PASS" else None

    def _compaction_luna_results(
        self,
        compaction_input: CompactionInput,
    ) -> tuple[CompactionGateResult, ...]:
        if self.compactor_luna is None:
            return ()
        sample_ids: tuple[Literal["A", "B", "C"], ...] = ("A", "B", "C")
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = tuple(
                executor.map(
                    lambda sample_id: self._compaction_luna_result(
                        compaction_input,
                        sample_id,
                    ),
                    sample_ids,
                )
            )
        return tuple(result for result in results if result is not None)

    def _compaction_luna_result(
        self,
        compaction_input: CompactionInput,
        sample_id: Literal["A", "B", "C"],
    ) -> CompactionGateResult | None:
        if self.compactor_luna is None:
            return None
        cached = self.store.cached_compaction_pass(
            turn_index=compaction_input.turn_index,
            stage="COMPACTION_LUNA",
            sample_id=sample_id,
            input_sha256=compaction_input.input_sha256,
        )
        if cached is not None:
            return cached
        latest: CompactionGateResult | None = None
        attempts = self.store.attempt_count(
            turn_index=compaction_input.turn_index,
            stage="COMPACTION_LUNA",
            sample_id=sample_id,
            input_sha256=compaction_input.input_sha256,
        )
        while attempts < self.max_candidate_attempts:
            attempt_index = attempts + 1
            try:
                candidate = self.compactor_luna.generate(
                    compaction_input,
                    sample_id=sample_id,
                )
                latest = evaluate_compaction_candidate(
                    candidate,
                    compaction_input=compaction_input,
                    max_memory_chars=self.max_memory_chars,
                )
                self.store.record_compaction_gate_result(
                    turn_index=compaction_input.turn_index,
                    stage="COMPACTION_LUNA",
                    attempt_index=attempt_index,
                    input_sha256=compaction_input.input_sha256,
                    result=latest,
                )
            except (RuntimeError, ValueError) as error:
                self.store.record_error(
                    turn_index=compaction_input.turn_index,
                    stage="COMPACTION_LUNA",
                    sample_id=sample_id,
                    attempt_index=attempt_index,
                    input_sha256=compaction_input.input_sha256,
                    error=error,
                )
            attempts += 1
            if latest is not None and latest.status == "PASS":
                break
        return latest

    def _resolve_compaction(
        self,
        *,
        resolver: CompactionResolver,
        stage: Literal["COMPACTION_TERRA", "COMPACTION_SOL"],
        compaction_input: CompactionInput,
        gate_results: Sequence[CompactionGateResult],
        consensus: CompactionConsensusResult,
    ) -> CompactionGateResult | None:
        try:
            candidate = resolver.resolve(
                compaction_input=compaction_input,
                gate_results=gate_results,
                consensus=consensus,
            )
        except (RuntimeError, ValueError) as error:
            self.store.record_error(
                turn_index=compaction_input.turn_index,
                stage=stage,
                sample_id=stage.removeprefix("COMPACTION_"),
                attempt_index=1,
                input_sha256=compaction_input.input_sha256,
                error=error,
            )
            return None
        if candidate is None:
            return None
        result = evaluate_compaction_candidate(
            candidate,
            compaction_input=compaction_input,
            max_memory_chars=self.max_memory_chars,
        )
        self.store.record_compaction_gate_result(
            turn_index=compaction_input.turn_index,
            stage=stage,
            attempt_index=1,
            input_sha256=compaction_input.input_sha256,
            result=result,
        )
        return result if result.status == "PASS" else None

    def _result(
        self,
        state: GoldRunState,
        processed: int,
    ) -> GoldPipelineRunResult:
        return GoldPipelineRunResult(
            scenario_index=state.scenario_index,
            status=state.status,
            processed_turns=processed,
            turn_count=state.turn_count,
            final_memory=state.approved_memory,
            final_memory_sha256=state.approved_memory_sha256,
            add_count_since_compaction=state.add_count_since_compaction,
        )


def _source_fingerprint(
    scenario_index: int,
    entries: Sequence[VehicleHistoryEntry],
) -> str:
    payload = {
        "scenario_index": scenario_index,
        "turns": [
            {
                "turn_index": index,
                "line_number": entry.line_number,
                "timestamp": entry.timestamp.isoformat(),
                "raw_sha256": text_sha256(entry.raw),
            }
            for index, entry in enumerate(entries)
        ],
    }
    return text_sha256(_encode_json(payload))


def _state_from_row(row: sqlite3.Row) -> GoldRunState:
    pending = (
        json.loads(row["pending_review_json"]) if row["pending_review_json"] else None
    )
    return GoldRunState(
        scenario_index=int(row["scenario_index"]),
        source_fingerprint=str(row["source_fingerprint"]),
        turn_count=int(row["turn_count"]),
        next_turn_index=int(row["next_turn_index"]),
        approved_memory=str(row["approved_memory"]),
        approved_memory_sha256=str(row["approved_memory_sha256"]),
        add_count_since_compaction=int(row["add_count_since_compaction"]),
        status=row["status"],
        pending_review=pending,
    )


def _encode_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _component_identity(component: object | None) -> dict[str, str] | None:
    if component is None:
        return None
    model_id = getattr(component, "model_id", None)
    instructions = getattr(component, "instructions", None)
    result = {
        "class": f"{type(component).__module__}.{type(component).__qualname__}",
        "model_id": model_id if isinstance(model_id, str) else "",
    }
    if isinstance(instructions, str):
        result["instructions_sha256"] = text_sha256(instructions)
    return result


def _now() -> str:
    return datetime.now(UTC).isoformat()
