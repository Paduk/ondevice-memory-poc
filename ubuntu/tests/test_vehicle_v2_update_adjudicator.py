from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from palmclaw_ubuntu.vehicle_bench.v2_update_adjudicator import (
    OpenAIV2UpdateAuditAdjudicator,
    SolUpdateAdjudicationPayload,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_audit import (
    AuditDialogueTurn,
    EventAuditCase,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_judge import (
    UPDATE_CONSENSUS_SCHEMA_VERSION,
)


def audit_case() -> EventAuditCase:
    record_id = "hybrid:s01:event-1:event-1-turn-001"
    case = EventAuditCase(
        schema_version="test",
        case_id="hybrid:s01:event-1",
        method="hybrid",
        scenario_index=1,
        source_stage2_sha256="a" * 64,
        source_memory_artifact_sha256="b" * 64,
        event_id="event-1",
        timeline_index=0,
        chain_kind="vehicle",
        reasoning_type="preference_conflict",
        selection_reason="EXPECTED_OR_GENERATED_UPDATE",
        dialogue=(
            AuditDialogueTurn(
                turn_id="event-1-turn-001",
                speaker_id="p0",
                speaker_name="Person 0",
                text="I prefer the HUD brightness at level 7.",
            ),
        ),
        event_before_memory="",
        event_after_memory="- Person 0: HUD brightness=7",
        expected_updates=(
            {
                "subject_id": "p0",
                "attribute_path": "hud.level",
                "previous_value": None,
                "new_value": 7,
            },
        ),
        candidate_updates=(
            {
                "record_id": record_id,
                "event_turn_index": 0,
                "turn_id": "event-1-turn-001",
                "operations": [],
            },
        ),
        no_op_reason_codes=(),
        input_sha256="pending",
    )
    body = case.as_dict()
    body.pop("case_id")
    body.pop("input_sha256")
    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return replace(case, input_sha256=digest)


def audit_decision() -> dict[str, object]:
    case = audit_case()
    return {
        "case_id": case.case_id,
        "verdict": "PASS",
        "expected_updates": [
            {
                "source_update_index": 0,
                "status": "FOUND",
                "matched_candidate_record_id": case.candidate_updates[0]["record_id"],
                "earliest_evidence_turn_id": "event-1-turn-001",
                "evidence": [
                    {
                        "turn_id": "event-1-turn-001",
                        "quote": "HUD brightness at level 7",
                    }
                ],
                "reason": "The preference is explicit.",
            }
        ],
        "candidate_updates": [
            {
                "record_id": case.candidate_updates[0]["record_id"],
                "status": "VALID",
                "matched_source_update_indexes": [0],
                "evidence": [
                    {
                        "turn_id": "event-1-turn-001",
                        "quote": "HUD brightness at level 7",
                    }
                ],
                "reason": "The candidate preserves the stated value.",
            }
        ],
        "memory_result": "FAITHFUL",
        "reason": "The update is grounded and complete.",
    }


def disagreed_consensus() -> dict[str, object]:
    decision = audit_decision()
    return {
        "schema_version": UPDATE_CONSENSUS_SCHEMA_VERSION,
        "case_id": audit_case().case_id,
        "case_input_sha256": audit_case().input_sha256,
        "status": "DISAGREED",
        "material_key": None,
        "adopted_decision": None,
        "material_diff": [{"path": "verdict", "luna": "PASS", "terra": "FAIL"}],
        "judge_reports": {
            "luna": {"decision": decision},
            "terra": {"decision": {**decision, "verdict": "FAIL"}},
        },
    }


class _FakeResponses:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def parse(self, **_kwargs):
        return SimpleNamespace(
            id="sol-response-1",
            status="completed",
            output_parsed=SolUpdateAdjudicationPayload.model_validate(self.payload),
            usage=SimpleNamespace(
                input_tokens=200,
                output_tokens=30,
                total_tokens=230,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
            ),
        )


def _adjudicator(payload: dict[str, object]):
    return OpenAIV2UpdateAuditAdjudicator(
        "test-sol",
        timeout_seconds=1,
        client=SimpleNamespace(responses=_FakeResponses(payload)),
    )


def test_sol_resolves_and_grounds_final_decision() -> None:
    case = audit_case()
    report = _adjudicator(
        {
            "case_id": case.case_id,
            "resolution": "RESOLVE",
            "decision": audit_decision(),
            "reason": "The dialogue directly supports Luna's interpretation.",
        }
    ).adjudicate(case, disagreed_consensus())

    assert report["resolution"] == "RESOLVE"
    assert report["decision"]["expected_updates"][0]["boundary_offset"] == 0
    assert report["usage"]["cached_tokens"] == 20


def test_sol_can_defer_without_forcing_a_decision() -> None:
    case = audit_case()
    report = _adjudicator(
        {
            "case_id": case.case_id,
            "resolution": "DEFER_HUMAN",
            "decision": None,
            "reason": "Two grounded interpretations remain plausible.",
        }
    ).adjudicate(case, disagreed_consensus())

    assert report["resolution"] == "DEFER_HUMAN"
    assert report["decision"] is None


def test_sol_resolution_rejects_ungrounded_evidence() -> None:
    case = audit_case()
    decision = audit_decision()
    decision["expected_updates"][0]["evidence"][0]["quote"] = "not present"
    adjudicator = _adjudicator(
        {
            "case_id": case.case_id,
            "resolution": "RESOLVE",
            "decision": decision,
            "reason": "Supposedly resolved.",
        }
    )

    with pytest.raises(ValueError, match="Evidence quote absent"):
        adjudicator.adjudicate(case, disagreed_consensus())
