from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from palmclaw_ubuntu.vehicle_bench.v2_update_audit import EventAuditCase

UPDATE_JUDGE_PROMPT_VERSION = "vehiclemembench-v2-update-event-judge-v1"
UPDATE_JUDGE_SCHEMA_VERSION = "vehiclemembench-v2-update-event-result-v1"
UPDATE_CONSENSUS_SCHEMA_VERSION = "vehiclemembench-v2-update-consensus-v1"

ExpectedUpdateStatus = Literal[
    "FOUND",
    "MISSING",
    "UNSUPPORTED_BY_DIALOGUE",
    "WRONG_VALUE",
    "WRONG_SUBJECT",
    "WRONG_CONDITION",
]
CandidateUpdateStatus = Literal[
    "VALID",
    "SPURIOUS",
    "DUPLICATE",
    "PREMATURE",
    "LATE",
]
MemoryResult = Literal[
    "FAITHFUL",
    "INFORMATION_LOSS",
    "CONTRADICTED",
    "MALFORMED",
]
EventVerdict = Literal["PASS", "PARTIAL", "FAIL"]
JudgeRole = Literal["luna", "terra"]

UPDATE_EVENT_JUDGE_INSTRUCTIONS = """
Audit one turn-wise vehicle-memory event. Decide whether the candidate UPDATEs
or NO_OP behavior faithfully capture the durable vehicle preferences supported
by the causal event dialogue.

The supplied expected_updates define which structured fields must be audited,
but they are NEVER dialogue evidence. Every factual conclusion must be grounded
in an exact quote from event_dialogue. Do not use future events, world knowledge,
likely defaults, or facts that appear only in expected_updates.

For each expected update, return its exact source_update_index and classify it:
- FOUND: a candidate correctly commits the subject, value, and condition.
- MISSING: the dialogue supports it but no candidate commits it.
- UNSUPPORTED_BY_DIALOGUE: the expected fields are not established by dialogue.
- WRONG_VALUE / WRONG_SUBJECT / WRONG_CONDITION: a candidate commits it
  incorrectly in that specific way.

For each supplied candidate update, return its exact record_id and classify it:
- VALID, SPURIOUS, DUPLICATE, PREMATURE, or LATE.

earliest_evidence_turn_id is the earliest dialogue turn after which the complete
subject, value, and condition are supported. Evidence quotes must be exact,
non-empty substrings of their cited turns. Judge event_after_memory against
event_before_memory and all valid updates. Reasons must be short. Output only
the requested structured object.
""".strip()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UpdateAuditEvidence(_StrictModel):
    turn_id: str = Field(min_length=1, max_length=192)
    quote: str = Field(min_length=1, max_length=2_048)

    @field_validator("turn_id", "quote")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class ExpectedUpdateDecision(_StrictModel):
    source_update_index: int = Field(ge=0)
    status: ExpectedUpdateStatus
    matched_candidate_record_id: str | None = Field(default=None, max_length=320)
    earliest_evidence_turn_id: str | None = Field(default=None, max_length=192)
    evidence: list[UpdateAuditEvidence] = Field(max_length=8)
    reason: str = Field(min_length=1, max_length=512)

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_missing_shape(self) -> ExpectedUpdateDecision:
        no_candidate = {"MISSING", "UNSUPPORTED_BY_DIALOGUE"}
        if self.status in no_candidate and self.matched_candidate_record_id is not None:
            raise ValueError(f"{self.status} cannot match a candidate")
        if self.status not in no_candidate and self.matched_candidate_record_id is None:
            raise ValueError(f"{self.status} requires a matched candidate")
        unsupported = self.status == "UNSUPPORTED_BY_DIALOGUE"
        if unsupported and self.earliest_evidence_turn_id is not None:
            raise ValueError("Unsupported dialogue cannot have earliest evidence")
        if not unsupported and self.earliest_evidence_turn_id is None:
            raise ValueError(f"{self.status} requires earliest evidence turn")
        if not unsupported and not self.evidence:
            raise ValueError(f"{self.status} requires grounded evidence")
        return self


class CandidateUpdateDecision(_StrictModel):
    record_id: str = Field(min_length=1, max_length=320)
    status: CandidateUpdateStatus
    matched_source_update_indexes: list[int] = Field(max_length=32)
    evidence: list[UpdateAuditEvidence] = Field(max_length=8)
    reason: str = Field(min_length=1, max_length=512)

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        return value.strip()

    @field_validator("matched_source_update_indexes")
    @classmethod
    def unique_sorted_indexes(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("matched source update indexes cannot be negative")
        if values != sorted(set(values)):
            raise ValueError("matched source update indexes must be sorted/unique")
        return values


class UpdateEventJudgePayload(_StrictModel):
    case_id: str = Field(min_length=1, max_length=320)
    verdict: EventVerdict
    expected_updates: list[ExpectedUpdateDecision] = Field(max_length=32)
    candidate_updates: list[CandidateUpdateDecision] = Field(max_length=32)
    memory_result: MemoryResult
    reason: str = Field(min_length=1, max_length=768)

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        return value.strip()


class OpenAIV2UpdateEventJudge:
    def __init__(
        self,
        model_id: str,
        *,
        judge_role: JudgeRole,
        timeout_seconds: float,
        reasoning_effort: str,
        max_output_tokens: int = 16_384,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Update Event Judge model ID is required")
        self.model_id = model_id
        self.judge_role = judge_role
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def audit(self, case: EventAuditCase) -> dict[str, Any]:
        provider_input = _provider_input(case)
        response = self._client.responses.parse(
            model=self.model_id,
            instructions=UPDATE_EVENT_JUDGE_INSTRUCTIONS,
            input=json.dumps(
                provider_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            text_format=UpdateEventJudgePayload,
            max_output_tokens=self.max_output_tokens,
            reasoning={"effort": self.reasoning_effort},
            store=False,
        )
        status = getattr(response, "status", None)
        if status in {"failed", "incomplete", "cancelled"}:
            raise RuntimeError(f"Update Event Judge returned status {status}")
        parsed = UpdateEventJudgePayload.model_validate(
            getattr(response, "output_parsed", None)
        )
        grounded = ground_update_event_decision(case, parsed)
        return {
            "schema_version": UPDATE_JUDGE_SCHEMA_VERSION,
            "prompt_version": UPDATE_JUDGE_PROMPT_VERSION,
            "judge_role": self.judge_role,
            "model_id": self.model_id,
            "response_id": getattr(response, "id", None),
            "case_id": case.case_id,
            "case_input_sha256": case.input_sha256,
            "provider_input_sha256": _canonical_sha256(provider_input),
            "decision": grounded,
            "material_key": material_semantic_key(grounded),
            "usage": _response_usage(response),
        }


def ground_update_event_decision(
    case: EventAuditCase,
    payload: UpdateEventJudgePayload,
) -> dict[str, Any]:
    if payload.case_id != case.case_id:
        raise ValueError(f"Update Judge case mismatch: {payload.case_id}")
    expected_indexes = list(range(len(case.expected_updates)))
    actual_indexes = [item.source_update_index for item in payload.expected_updates]
    if actual_indexes != expected_indexes:
        raise ValueError(
            f"Expected update order mismatch: {actual_indexes} != {expected_indexes}"
        )
    candidate_ids = [item["record_id"] for item in case.candidate_updates]
    actual_candidate_ids = [item.record_id for item in payload.candidate_updates]
    if actual_candidate_ids != candidate_ids:
        raise ValueError(
            "Candidate update order mismatch: "
            f"{actual_candidate_ids} != {candidate_ids}"
        )
    candidate_id_set = set(candidate_ids)
    expected_index_set = set(expected_indexes)
    turn_by_id = {turn.turn_id: turn for turn in case.dialogue}
    turn_index_by_id = {
        turn.turn_id: index for index, turn in enumerate(case.dialogue)
    }

    expected_results = []
    for item in payload.expected_updates:
        matched = item.matched_candidate_record_id
        if matched is not None and matched not in candidate_id_set:
            raise ValueError(f"Unknown matched candidate record: {matched}")
        evidence = _ground_evidence(item.evidence, turn_by_id)
        earliest = item.earliest_evidence_turn_id
        if earliest is not None and earliest not in turn_by_id:
            raise ValueError(f"Unknown earliest evidence turn: {earliest}")
        boundary_offset = None
        if matched is not None and earliest is not None:
            candidate = next(
                value
                for value in case.candidate_updates
                if value["record_id"] == matched
            )
            boundary_offset = int(candidate["event_turn_index"]) - (
                turn_index_by_id[earliest]
            )
        expected_results.append(
            {
                **item.model_dump(mode="json", exclude={"evidence"}),
                "evidence": evidence,
                "boundary_offset": boundary_offset,
            }
        )

    candidate_results = []
    for item in payload.candidate_updates:
        if set(item.matched_source_update_indexes) - expected_index_set:
            raise ValueError(f"Unknown matched source update: {item.record_id}")
        candidate_results.append(
            {
                **item.model_dump(mode="json", exclude={"evidence"}),
                "evidence": _ground_evidence(item.evidence, turn_by_id),
            }
        )
    return {
        "case_id": case.case_id,
        "verdict": payload.verdict,
        "expected_updates": expected_results,
        "candidate_updates": candidate_results,
        "memory_result": payload.memory_result,
        "reason": payload.reason,
    }


def material_semantic_key(decision: dict[str, Any]) -> dict[str, Any]:
    """Return only structured semantics; free text/evidence wording is ignored."""

    return {
        "verdict": decision["verdict"],
        "expected_updates": [
            {
                "source_update_index": item["source_update_index"],
                "status": item["status"],
                "matched_candidate_record_id": item["matched_candidate_record_id"],
                "earliest_evidence_turn_id": item["earliest_evidence_turn_id"],
                "boundary_offset": item["boundary_offset"],
            }
            for item in decision["expected_updates"]
        ],
        "candidate_updates": [
            {
                "record_id": item["record_id"],
                "status": item["status"],
                "matched_source_update_indexes": item[
                    "matched_source_update_indexes"
                ],
            }
            for item in decision["candidate_updates"]
        ],
        "memory_result": decision["memory_result"],
    }


def build_dual_judge_consensus(
    luna_report: dict[str, Any],
    terra_report: dict[str, Any],
) -> dict[str, Any]:
    for report, role in ((luna_report, "luna"), (terra_report, "terra")):
        if report.get("schema_version") != UPDATE_JUDGE_SCHEMA_VERSION:
            raise ValueError(f"Invalid {role} Judge schema version")
        if report.get("prompt_version") != UPDATE_JUDGE_PROMPT_VERSION:
            raise ValueError(f"Invalid {role} Judge prompt version")
        if report.get("judge_role") != role:
            raise ValueError(f"Invalid {role} Judge role")
    for field in ("case_id", "case_input_sha256"):
        if luna_report.get(field) != terra_report.get(field):
            raise ValueError(f"Dual Judge {field} mismatch")
    luna_key = luna_report["material_key"]
    terra_key = terra_report["material_key"]
    agreed = luna_key == terra_key
    return {
        "schema_version": UPDATE_CONSENSUS_SCHEMA_VERSION,
        "case_id": luna_report["case_id"],
        "case_input_sha256": luna_report["case_input_sha256"],
        "status": "AGREED" if agreed else "DISAGREED",
        "material_key": luna_key if agreed else None,
        "adopted_decision": luna_report["decision"] if agreed else None,
        "material_diff": [] if agreed else _material_diff(luna_key, terra_key),
        "judge_reports": {
            "luna": luna_report,
            "terra": terra_report,
        },
    }


def _provider_input(case: EventAuditCase) -> dict[str, Any]:
    case_payload = case.as_dict()
    case_payload.pop("case_id")
    case_payload.pop("input_sha256")
    if _canonical_sha256(case_payload) != case.input_sha256:
        raise ValueError(f"Event audit case input hash mismatch: {case.case_id}")
    return {
        "case_id": case.case_id,
        "event_dialogue": [
            {
                "event_turn_index": index,
                "turn_id": turn.turn_id,
                "speaker_id": turn.speaker_id,
                "speaker_name": turn.speaker_name,
                "text": turn.text,
            }
            for index, turn in enumerate(case.dialogue)
        ],
        "event_before_memory": case.event_before_memory,
        "event_after_memory": case.event_after_memory,
        "expected_updates": list(case.expected_updates),
        "candidate_updates": list(case.candidate_updates),
    }


def _ground_evidence(
    items: list[UpdateAuditEvidence],
    turn_by_id: dict[str, Any],
) -> list[dict[str, str]]:
    grounded = []
    for item in items:
        turn = turn_by_id.get(item.turn_id)
        if turn is None:
            raise ValueError(f"Unknown evidence turn: {item.turn_id}")
        if item.quote not in turn.text:
            raise ValueError(f"Evidence quote absent from turn: {item.turn_id}")
        grounded.append(
            {
                "turn_id": item.turn_id,
                "quote": item.quote,
                "turn_text": turn.text,
            }
        )
    return grounded


def _material_diff(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": path, "luna": left, "terra": right}]
    if isinstance(left, dict):
        differences = []
        for key in sorted(set(left).union(right)):
            differences.extend(
                _material_diff(left.get(key), right.get(key), f"{path}.{key}")
            )
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": path, "luna": left, "terra": right}]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            differences.extend(
                _material_diff(left_item, right_item, f"{path}[{index}]")
            )
        return differences
    return [] if left == right else [{"path": path, "luna": left, "terra": right}]


def _canonical_sha256(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
