from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from palmclaw_ubuntu.vehicle_bench.v2_update_audit import (
    AuditDialogueTurn,
    EventAuditCase,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_judge import (
    OpenAIV2UpdateEventJudge,
    UpdateEventJudgePayload,
    build_dual_judge_consensus,
)


def _case() -> EventAuditCase:
    record_id = "hybrid:s01:vehicle-e1:vehicle-e1-turn-001"
    dialogue = (
        AuditDialogueTurn(
            turn_id="vehicle-e1-turn-001",
            speaker_id="p0",
            speaker_name="Person 0",
            text="I prefer the HUD brightness at level 7.",
        ),
    )
    case = EventAuditCase(
        schema_version="test",
        case_id="hybrid:s01:vehicle-e1",
        method="hybrid",
        scenario_index=1,
        source_stage2_sha256="a" * 64,
        source_memory_artifact_sha256="b" * 64,
        event_id="vehicle-e1",
        timeline_index=0,
        chain_kind="vehicle",
        reasoning_type="preference_conflict",
        selection_reason="EXPECTED_OR_GENERATED_UPDATE",
        dialogue=dialogue,
        event_before_memory="",
        event_after_memory="- Person 0: HUD brightness=7",
        expected_updates=(
            {
                "subject_id": "p0",
                "attribute_path": "carcontrol_HUD_set_brightness_level.level",
                "previous_value": None,
                "new_value": 7,
                "context_arguments": [],
                "condition": None,
                "supersedes_event_id": None,
            },
        ),
        candidate_updates=(
            {
                "record_id": record_id,
                "event_turn_index": 0,
                "turn_id": "vehicle-e1-turn-001",
                "source_update_indexes": [0],
                "operations": [
                    {
                        "op": "add",
                        "target": "",
                        "content": "- Person 0: HUD brightness=7",
                    }
                ],
            },
        ),
        no_op_reason_codes=(),
        input_sha256="pending",
    )
    payload = case.as_dict()
    payload.pop("case_id")
    payload.pop("input_sha256")
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return replace(case, input_sha256=digest)


def _payload(*, quote: str, reason: str, candidate_status: str = "VALID"):
    record_id = _case().candidate_updates[0]["record_id"]
    return {
        "case_id": _case().case_id,
        "verdict": "PASS" if candidate_status == "VALID" else "PARTIAL",
        "expected_updates": [
            {
                "source_update_index": 0,
                "status": "FOUND",
                "matched_candidate_record_id": record_id,
                "earliest_evidence_turn_id": "vehicle-e1-turn-001",
                "evidence": [
                    {"turn_id": "vehicle-e1-turn-001", "quote": quote}
                ],
                "reason": reason,
            }
        ],
        "candidate_updates": [
            {
                "record_id": record_id,
                "status": candidate_status,
                "matched_source_update_indexes": [0],
                "evidence": [
                    {"turn_id": "vehicle-e1-turn-001", "quote": quote}
                ],
                "reason": reason,
            }
        ],
        "memory_result": "FAITHFUL",
        "reason": reason,
    }


class _FakeResponses:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="response-1",
            status="completed",
            output_parsed=UpdateEventJudgePayload.model_validate(self.payload),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                input_tokens_details=SimpleNamespace(cached_tokens=10),
            ),
        )


def _judge(role: str, payload: dict[str, object]):
    responses = _FakeResponses(payload)
    judge = OpenAIV2UpdateEventJudge(
        f"test-{role}",
        judge_role=role,
        timeout_seconds=1,
        reasoning_effort="medium",
        client=SimpleNamespace(responses=responses),
    )
    return judge, responses


def test_semantically_equal_judges_agree_despite_text_differences() -> None:
    luna, _ = _judge(
        "luna",
        _payload(quote="HUD brightness at level 7", reason="Preference stated."),
    )
    terra, _ = _judge(
        "terra",
        _payload(
            quote="I prefer the HUD brightness at level 7.",
            reason="The speaker explicitly confirms the setting.",
        ),
    )

    consensus = build_dual_judge_consensus(
        luna.audit(_case()),
        terra.audit(_case()),
    )

    assert consensus["status"] == "AGREED"
    assert consensus["material_diff"] == []


def test_material_candidate_difference_requires_sol() -> None:
    luna, _ = _judge(
        "luna",
        _payload(quote="HUD brightness at level 7", reason="Valid."),
    )
    terra, _ = _judge(
        "terra",
        _payload(
            quote="HUD brightness at level 7",
            reason="The update is late.",
            candidate_status="LATE",
        ),
    )

    consensus = build_dual_judge_consensus(
        luna.audit(_case()),
        terra.audit(_case()),
    )

    assert consensus["status"] == "DISAGREED"
    assert any(
        "candidate_updates" in item["path"]
        for item in consensus["material_diff"]
    )


def test_judge_rejects_ungrounded_quote() -> None:
    judge, _ = _judge(
        "luna",
        _payload(quote="not in the dialogue", reason="Invalid evidence."),
    )

    with pytest.raises(ValueError, match="Evidence quote absent"):
        judge.audit(_case())
