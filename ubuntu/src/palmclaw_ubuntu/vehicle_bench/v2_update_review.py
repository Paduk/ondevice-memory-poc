from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from palmclaw_ubuntu.vehicle_bench.v2_update_adjudicator import (
    UPDATE_ADJUDICATOR_SCHEMA_VERSION,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_audit import (
    EventAuditCase,
    event_audit_case_from_dict,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_judge import (
    UpdateEventJudgePayload,
    ground_update_event_decision,
)

UPDATE_AUDIT_REVIEW_SCHEMA_VERSION = "vehiclemembench-v2-update-audit-review-v1"

ReviewStatus = Literal["PENDING", "SUBMITTED", "APPLIED", "REJECTED"]


class UpdateAuditHumanSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=320)
    decision: UpdateEventJudgePayload

    @field_validator("reviewer", "reason")
    @classmethod
    def normalize_single_line(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("Reviewer and reason must be one non-empty line")
        return normalized


@dataclass(frozen=True)
class UpdateAuditReviewRecord:
    review_id: str
    scenario_index: int
    turn_index: int
    checkpoint_path: str
    request_sha256: str
    status: ReviewStatus
    request: dict[str, Any]
    submission: UpdateAuditHumanSubmission | None
    created_at: str
    submitted_at: str | None
    applied_at: str | None
    apply_error: str | None


class UpdateAuditReviewQueue:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def register_deferred(
        self,
        *,
        case: EventAuditCase,
        consensus: dict[str, Any],
        sol_report: dict[str, Any],
        checkpoint_path: Path | str,
    ) -> UpdateAuditReviewRecord:
        if sol_report.get("schema_version") != UPDATE_ADJUDICATOR_SCHEMA_VERSION:
            raise ValueError("Invalid SOL report for UPDATE audit review")
        if sol_report.get("resolution") != "DEFER_HUMAN":
            raise ValueError("Only DEFER_HUMAN can enter UPDATE audit review")
        if sol_report.get("case_id") != case.case_id:
            raise ValueError("UPDATE audit review case mismatch")
        request = {
            "schema_version": UPDATE_AUDIT_REVIEW_SCHEMA_VERSION,
            "review_kind": "UPDATE_AUDIT",
            "scenario_index": case.scenario_index,
            "turn_index": case.timeline_index,
            "case": case.as_dict(),
            "dual_judge_consensus": consensus,
            "sol_report": sol_report,
            "current_turn": {
                "raw": (
                    f"{case.dialogue[-1].speaker_name}: {case.dialogue[-1].text}"
                    if case.dialogue
                    else "(empty event)"
                )
            },
        }
        encoded = _encode_json(request)
        request_sha256 = hashlib.sha256(encoded.encode()).hexdigest()
        checkpoint = str(Path(checkpoint_path).expanduser().resolve())
        review_id = (
            f"update-audit-s{case.scenario_index:02d}-"
            f"{case.event_id}-{request_sha256[:12]}"
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
                    UPDATE_AUDIT_REVIEW_SCHEMA_VERSION,
                    case.scenario_index,
                    case.timeline_index,
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
    ) -> tuple[UpdateAuditReviewRecord, ...]:
        query = "SELECT * FROM reviews"
        parameters: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY created_at, review_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    def get(self, review_id: str) -> UpdateAuditReviewRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"UPDATE audit review not found: {review_id}")
        return _record_from_row(row)

    def submit(
        self,
        review_id: str,
        submission: UpdateAuditHumanSubmission,
        *,
        expected_request_sha256: str,
    ) -> UpdateAuditReviewRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"UPDATE audit review not found: {review_id}")
            if row["request_sha256"] != expected_request_sha256:
                raise ValueError("UPDATE audit review request hash has changed")
            if row["status"] != "PENDING":
                raise ValueError(f"UPDATE audit review is already {row['status']}")
            request = json.loads(row["request_json"])
            case = event_audit_case_from_dict(request["case"])
            grounded = ground_update_event_decision(case, submission.decision)
            stored = {
                **submission.model_dump(mode="json"),
                "grounded_decision": grounded,
            }
            connection.execute(
                """
                UPDATE reviews
                SET status = 'SUBMITTED', submission_json = ?, submitted_at = ?,
                    apply_error = NULL
                WHERE review_id = ?
                """,
                (_encode_json(stored), _now(), review_id),
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
                CREATE INDEX IF NOT EXISTS update_audit_reviews_status
                ON reviews(status, scenario_index, turn_index);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection


def _record_from_row(row: sqlite3.Row) -> UpdateAuditReviewRecord:
    submission = None
    if row["submission_json"]:
        raw = json.loads(row["submission_json"])
        raw.pop("grounded_decision", None)
        submission = UpdateAuditHumanSubmission.model_validate(raw)
    return UpdateAuditReviewRecord(
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


def _now() -> str:
    return datetime.now(UTC).isoformat()
