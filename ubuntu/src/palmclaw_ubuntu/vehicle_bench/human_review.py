from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from palmclaw_ubuntu.vehicle_bench.consensus_gold import (
    CONSENSUS_GOLD_SCHEMA_VERSION,
    CombinedTurnCandidatePayload,
    CompactionCandidatePayload,
    CompactionInput,
    GeneratedCombinedCandidate,
    GeneratedCompactionCandidate,
    GoldGenerationInput,
    text_sha256,
)
from palmclaw_ubuntu.vehicle_bench.consensus_gold_pipeline import GoldRunState
from palmclaw_ubuntu.vehicle_bench.memory import VehicleHistoryEntry

HUMAN_REVIEW_SCHEMA_VERSION = "vehiclemembench-v2-human-review-v1"

type ReviewStatus = Literal["PENDING", "SUBMITTED", "APPLIED", "REJECTED"]
type ReviewAction = Literal["SELECT", "NO_OP", "CORRECT"]


class HumanReviewSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer: str = Field(min_length=1, max_length=120)
    action: ReviewAction
    reason: str = Field(min_length=1, max_length=320)
    selected_sample_id: Literal["A", "B", "C", "TERRA", "SOL"] | None = None
    corrected_candidate: CombinedTurnCandidatePayload | None = None
    corrected_memory: str | None = Field(default=None, max_length=32_768)

    @field_validator("reviewer", "reason")
    @classmethod
    def normalize_single_line(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("Reviewer and reason must be one non-empty line")
        return normalized

    @field_validator("corrected_memory")
    @classmethod
    def normalize_corrected_memory(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Corrected memory cannot be empty")
        return normalized

    @model_validator(mode="after")
    def validate_action(self) -> HumanReviewSubmission:
        if self.action == "SELECT":
            if (
                self.selected_sample_id is None
                or self.corrected_candidate is not None
                or self.corrected_memory is not None
            ):
                raise ValueError("SELECT requires only selected_sample_id")
        elif self.action == "NO_OP":
            if (
                self.selected_sample_id is not None
                or self.corrected_candidate is not None
                or self.corrected_memory is not None
            ):
                raise ValueError("NO_OP cannot include a candidate")
        elif self.selected_sample_id is not None or (
            (self.corrected_candidate is None) == (self.corrected_memory is None)
        ):
            raise ValueError("CORRECT requires exactly one correction payload")
        return self


@dataclass(frozen=True)
class HumanReviewRecord:
    review_id: str
    scenario_index: int
    turn_index: int
    checkpoint_path: str
    request_sha256: str
    status: ReviewStatus
    request: dict[str, Any]
    submission: HumanReviewSubmission | None
    created_at: str
    submitted_at: str | None
    applied_at: str | None
    apply_error: str | None


class HumanReviewQueue:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def register_paused(
        self,
        *,
        checkpoint_path: Path | str,
        state: GoldRunState,
        entry: VehicleHistoryEntry,
    ) -> HumanReviewRecord:
        if state.status != "PAUSED_REVIEW" or state.pending_review is None:
            raise ValueError("Only a paused Gold state can enter Human Review")
        checkpoint = str(Path(checkpoint_path).expanduser().resolve())
        request = {
            "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
            "scenario_index": state.scenario_index,
            "turn_index": state.next_turn_index,
            "checkpoint_path": checkpoint,
            "source_fingerprint": state.source_fingerprint,
            "approved_memory_before": state.approved_memory,
            "approved_memory_before_sha256": state.approved_memory_sha256,
            "current_turn": {
                "message_id": entry.line_number,
                "date": entry.date,
                "raw": entry.raw,
                "content": entry.content,
            },
            "pending_review": dict(state.pending_review),
            "escalation_attempts": _load_escalation_attempts(
                checkpoint,
                state.next_turn_index,
            ),
        }
        encoded = _encode_json(request)
        request_sha256 = text_sha256(encoded)
        base_review_id = (
            f"s{state.scenario_index:02d}-t{state.next_turn_index:06d}-"
            f"{request_sha256[:12]}"
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT review_id, status FROM reviews
                WHERE scenario_index = ? AND turn_index = ?
                  AND checkpoint_path = ? AND request_sha256 = ?
                ORDER BY created_at, review_id
                """,
                (
                    state.scenario_index,
                    state.next_turn_index,
                    checkpoint,
                    request_sha256,
                ),
            ).fetchall()
            active = next(
                (row for row in reversed(existing) if row["status"] != "REJECTED"),
                None,
            )
            if active is not None:
                review_id = str(active["review_id"])
            else:
                review_id = base_review_id
                if existing:
                    review_id = f"{base_review_id}-r{len(existing) + 1}"
            connection.execute(
                """
                INSERT INTO reviews (
                    review_id, schema_version, scenario_index, turn_index,
                    checkpoint_path, request_sha256, status, request_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                ON CONFLICT(review_id) DO NOTHING
                """,
                (
                    review_id,
                    HUMAN_REVIEW_SCHEMA_VERSION,
                    state.scenario_index,
                    state.next_turn_index,
                    checkpoint,
                    request_sha256,
                    encoded,
                    _now(),
                ),
            )
        return self.get(review_id)

    def list(
        self,
        *,
        status: ReviewStatus | None = None,
    ) -> tuple[HumanReviewRecord, ...]:
        query = "SELECT * FROM reviews"
        parameters: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY created_at, review_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    def get(self, review_id: str) -> HumanReviewRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Human review not found: {review_id}")
        return _record_from_row(row)

    def submit(
        self,
        review_id: str,
        submission: HumanReviewSubmission,
        *,
        expected_request_sha256: str,
    ) -> HumanReviewRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Human review not found: {review_id}")
            if row["request_sha256"] != expected_request_sha256:
                raise ValueError("Human review request hash has changed")
            if row["status"] != "PENDING":
                raise ValueError(f"Human review is already {row['status']}")
            request = json.loads(row["request_json"])
            review_kind = (
                request.get("pending_review", {})
                .get("payload", {})
                .get("review_kind", "TURN")
            )
            if review_kind == "COMPACTION":
                if submission.action == "NO_OP" or (
                    submission.action == "CORRECT"
                    and submission.corrected_memory is None
                ):
                    raise ValueError(
                        "Compaction review requires SELECT or memory correction"
                    )
            elif submission.action == "CORRECT" and (
                submission.corrected_candidate is None
            ):
                raise ValueError("Turn review requires a Combined candidate correction")
            connection.execute(
                """
                UPDATE reviews
                SET status = 'SUBMITTED', submission_json = ?, submitted_at = ?,
                    apply_error = NULL
                WHERE review_id = ?
                """,
                (
                    _encode_json(submission.model_dump(mode="json")),
                    _now(),
                    review_id,
                ),
            )
        return self.get(review_id)

    def submitted_for(
        self,
        *,
        scenario_index: int,
        turn_index: int,
        checkpoint_path: Path | str,
    ) -> HumanReviewRecord | None:
        checkpoint = str(Path(checkpoint_path).expanduser().resolve())
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM reviews
                WHERE scenario_index = ? AND turn_index = ?
                  AND checkpoint_path = ? AND status = 'SUBMITTED'
                ORDER BY submitted_at
                LIMIT 1
                """,
                (scenario_index, turn_index, checkpoint),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def mark_applied(self, review_id: str) -> HumanReviewRecord:
        return self._set_result(review_id, status="APPLIED", error=None)

    def mark_apply_error(self, review_id: str, error: Exception) -> HumanReviewRecord:
        return self._set_result(
            review_id,
            status="REJECTED",
            error=f"{type(error).__name__}: {error}",
        )

    def _set_result(
        self,
        review_id: str,
        *,
        status: Literal["REJECTED", "APPLIED"],
        error: str | None,
    ) -> HumanReviewRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Human review not found: {review_id}")
            if row["status"] != "SUBMITTED":
                raise ValueError("Only a submitted review can be applied")
            connection.execute(
                """
                UPDATE reviews
                SET status = ?, applied_at = ?, apply_error = ?
                WHERE review_id = ?
                """,
                (status, _now() if status == "APPLIED" else None, error, review_id),
            )
        return self.get(review_id)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    scenario_index INTEGER NOT NULL,
                    turn_index INTEGER NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    submission_json TEXT,
                    created_at TEXT NOT NULL,
                    submitted_at TEXT,
                    applied_at TEXT,
                    apply_error TEXT
                );
                CREATE INDEX IF NOT EXISTS reviews_status_lookup
                ON reviews(status, scenario_index, turn_index);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection


def build_reviewed_candidate(
    record: HumanReviewRecord,
    generation_input: GoldGenerationInput,
) -> GeneratedCombinedCandidate:
    submission = record.submission
    if record.status != "SUBMITTED" or submission is None:
        raise ValueError("Human Review has not been submitted")
    request = record.request
    if (
        request.get("pending_review", {})
        .get("payload", {})
        .get("review_kind")
        == "COMPACTION"
    ):
        raise ValueError("Human Review is for compaction, not a turn candidate")
    if request.get("source_fingerprint") is None:
        raise ValueError("Human Review source fingerprint is missing")
    if request.get("approved_memory_before_sha256") != text_sha256(
        generation_input.previous_memory
    ):
        raise ValueError("Human Review before-memory hash does not match")
    if request.get("turn_index") != generation_input.turn_index:
        raise ValueError("Human Review turn does not match")

    if submission.action == "NO_OP":
        payload = CombinedTurnCandidatePayload(
            decision="NO_OP",
            operations=[],
            reason=submission.reason,
            evidence=[],
        )
    elif submission.action == "CORRECT":
        if submission.corrected_candidate is None:
            raise RuntimeError("Corrected Human Review candidate is missing")
        payload = submission.corrected_candidate
    else:
        gate_results = (
            request.get("pending_review", {})
            .get("payload", {})
            .get("gate_results", [])
        )
        selected = next(
            (
                result
                for result in gate_results
                if result.get("sample_id") == submission.selected_sample_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("Selected Human Review candidate is unavailable")
        payload = CombinedTurnCandidatePayload.model_validate(
            selected.get("candidate", {}).get("payload")
        )
    return GeneratedCombinedCandidate(
        sample_id="HUMAN",
        payload=payload,
        response_id=None,
        model_id=f"human:{submission.reviewer}",
        prompt_version=HUMAN_REVIEW_SCHEMA_VERSION,
        schema_version=CONSENSUS_GOLD_SCHEMA_VERSION,
        input_sha256=generation_input.input_sha256,
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        },
    )


def build_reviewed_compaction_candidate(
    record: HumanReviewRecord,
    compaction_input: CompactionInput,
) -> GeneratedCompactionCandidate:
    submission = record.submission
    if record.status != "SUBMITTED" or submission is None:
        raise ValueError("Human Review has not been submitted")
    pending_payload = record.request.get("pending_review", {}).get("payload", {})
    if pending_payload.get("review_kind") != "COMPACTION":
        raise ValueError("Human Review is not for compaction")
    if pending_payload.get("post_patch_memory_sha256") != (
        compaction_input.post_patch_memory_sha256
    ):
        raise ValueError("Human Review post-patch memory hash does not match")

    if submission.action == "CORRECT":
        if submission.corrected_memory is None:
            raise RuntimeError("Corrected Human Review memory is missing")
        next_memory = submission.corrected_memory
    elif submission.action == "SELECT":
        gate_results = pending_payload.get("gate_results", [])
        selected = next(
            (
                result
                for result in gate_results
                if result.get("sample_id") == submission.selected_sample_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("Selected Human Review compaction is unavailable")
        payload = CompactionCandidatePayload.model_validate(
            selected.get("candidate", {}).get("payload")
        )
        next_memory = payload.next_memory
    else:
        raise ValueError("Compaction review does not support NO_OP")
    return GeneratedCompactionCandidate(
        sample_id="HUMAN",
        payload=CompactionCandidatePayload(next_memory=next_memory),
        response_id=None,
        model_id=f"human:{submission.reviewer}",
        prompt_version=HUMAN_REVIEW_SCHEMA_VERSION,
        schema_version=CONSENSUS_GOLD_SCHEMA_VERSION,
        input_sha256=compaction_input.input_sha256,
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        },
    )


def _record_from_row(row: sqlite3.Row) -> HumanReviewRecord:
    submission = (
        HumanReviewSubmission.model_validate_json(row["submission_json"])
        if row["submission_json"]
        else None
    )
    return HumanReviewRecord(
        review_id=str(row["review_id"]),
        scenario_index=int(row["scenario_index"]),
        turn_index=int(row["turn_index"]),
        checkpoint_path=str(row["checkpoint_path"]),
        request_sha256=str(row["request_sha256"]),
        status=row["status"],
        request=json.loads(row["request_json"]),
        submission=submission,
        created_at=str(row["created_at"]),
        submitted_at=row["submitted_at"],
        applied_at=row["applied_at"],
        apply_error=row["apply_error"],
    )


def _encode_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_escalation_attempts(
    checkpoint_path: str,
    turn_index: int,
) -> list[dict[str, Any]]:
    with sqlite3.connect(checkpoint_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT stage, sample_id, attempt_index, status, payload_json, created_at
            FROM model_attempts
            WHERE turn_index = ?
              AND stage IN ('TERRA', 'SOL', 'COMPACTION_TERRA', 'COMPACTION_SOL')
            ORDER BY id
            """,
            (turn_index,),
        ).fetchall()
    return [
        {
            "stage": row["stage"],
            "sample_id": row["sample_id"],
            "attempt_index": row["attempt_index"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def _now() -> str:
    return datetime.now(UTC).isoformat()
