from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from palmclaw_ubuntu.models import ModelUsage
from palmclaw_ubuntu.vehicle_bench.critic import (
    VEHICLE_COMBINED_CRITIC_SCHEMA_VERSION,
    VEHICLE_CRITIC_CONTEXT_SCHEMA_VERSION,
    VEHICLE_MEMORY_ADJUDICATOR_PROMPT_VERSION,
    VEHICLE_MEMORY_CRITIC_PROMPT_VERSION,
    VEHICLE_MEMORY_REPAIR_PROMPT_VERSION,
    VEHICLE_SUMMARY_ADJUDICATOR_SCHEMA_VERSION,
    VEHICLE_SUMMARY_CRITIC_SCHEMA_VERSION,
    CriticEvidence,
    CriticPatchOperation,
    OpenAIVehicleMemoryAdjudicatorModel,
    OpenAIVehicleMemoryCriticModel,
    VehicleCombinedAdjudicationPayload,
    VehicleCombinedCriticPayload,
    VehicleCriticContextRequestPayload,
    VehicleCriticRollbackRequestPayload,
    VehicleMemoryCriticResponse,
    VehicleSummaryAdjudicationPayload,
    VehicleSummaryCriticPayload,
    validate_vehicle_memory_critic_response,
)
from palmclaw_ubuntu.vehicle_bench.critic_trace import CriticSourceTurn


class FakeResponses:
    def __init__(self, parsed: object, *, status: str = "completed") -> None:
        self.parsed = parsed
        self.status = status
        self.requests: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            id="critic-response",
            status=self.status,
            output_parsed=self.parsed,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                input_tokens_details=SimpleNamespace(cached_tokens=10),
            ),
        )


class ScriptedResponses(FakeResponses):
    def __init__(self, parsed_sequence: list[object]) -> None:
        super().__init__(parsed_sequence[0])
        self.parsed_sequence = list(parsed_sequence)

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.parsed = self.parsed_sequence.pop(0)
        return super().parse(**kwargs)


def _source_turn(
    method: str,
    *,
    original_decision: str = "NO_OP",
    content: str = "[VehicleMemBench history turn=0]\nGary prefers green lights.",
    before_memory: str = "### Gary\n- Ambient color: orange",
    after_memory: str | None = None,
    turn_index: int = 0,
    message_id: int = 1,
) -> CriticSourceTurn:
    if after_memory is None:
        after_memory = before_memory
    return CriticSourceTurn(
        method=method,  # type: ignore[arg-type]
        scenario_index=1,
        turn_index=turn_index,
        message_id=message_id,
        role="user",
        content=content,
        date="2025-01-01",
        consolidation_run_id="run-1",
        model_call_id="call-1",
        original_decision=original_decision,  # type: ignore[arg-type]
        original_before_memory=before_memory,
        original_after_memory=after_memory,
        original_before_sha256="before",
        original_after_sha256="after",
        recorded_provider_after_sha256=None,
        storage_transformed=False,
        model_id="gpt-5.6-luna",
        prompt_version="source-prompt",
        schema_version="source-schema",
        usage={},
        latency_ms=10,
        metadata={},
    )


def _model(parsed: object) -> tuple[OpenAIVehicleMemoryCriticModel, FakeResponses]:
    responses = FakeResponses(parsed)
    model = OpenAIVehicleMemoryCriticModel(
        "gpt-5.6-luna",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )
    return model, responses


def _adjudicator(
    parsed: object,
) -> tuple[OpenAIVehicleMemoryAdjudicatorModel, FakeResponses]:
    responses = FakeResponses(parsed)
    model = OpenAIVehicleMemoryAdjudicatorModel(
        "gpt-5.6-terra",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )
    return model, responses


def _luna_reject(source: CriticSourceTurn):
    return validate_vehicle_memory_critic_response(
        VehicleMemoryCriticResponse(
            method=source.method,
            verdict="REJECT",
            decision=None,
            reason_code="NEEDS_CAUSAL_CONTEXT",
            reason="The reference requires an earlier causal turn.",
        ),
        source_turn=source,
        approved_memory=source.original_before_memory,
        max_memory_chars=8_192,
    )


def test_summary_critic_uses_separate_strict_schema_and_validates_update() -> None:
    next_memory = "### Gary\n- Ambient color: green"
    model, responses = _model(
        {
            "verdict": "CORRECT",
            "decision": "UPDATE",
            "reason_code": "MISSED_DURABLE_INFORMATION",
            "reason": "The turn explicitly changes Gary's durable preference.",
            "evidence": [
                {"message_id": 1, "quote": "prefers green lights"}
            ],
            "next_memory": next_memory,
        }
    )
    source = _source_turn("summary")

    result = model.review(
        source_turn=source,
        approved_memory=source.original_before_memory,
        max_memory_chars=8_192,
    )

    request = responses.requests[0]
    assert request["model"] == "gpt-5.6-luna"
    assert request["text_format"].__name__ == "_VehicleSummaryCriticTransport"
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "low"}
    assert "next memory" in str(request["instructions"])
    provider_input = json.loads(str(request["input"]))
    assert provider_input["method"] == "summary"
    assert provider_input["current_turn"]["message_id"] == 1
    assert result.commit_ready is True
    assert result.approved_memory_after == next_memory
    assert result.response.usage == ModelUsage(100, 20, 120, 10)
    assert (
        result.response.metadata["schema_version"]
        == VEHICLE_SUMMARY_CRITIC_SCHEMA_VERSION
    )
    assert result.response.metadata["prompt_version"] == (
        VEHICLE_MEMORY_CRITIC_PROMPT_VERSION
    )


def test_combined_critic_applies_exact_patch_atomically() -> None:
    model, responses = _model(
        {
            "verdict": "CORRECT",
            "decision": "UPDATE",
            "reason_code": "MISSED_DURABLE_INFORMATION",
            "reason": "The stored color must be replaced with the explicit value.",
            "evidence": [
                {"message_id": 1, "quote": "prefers green lights"}
            ],
            "operations": [
                {
                    "op": "replace",
                    "target": "- Ambient color: orange",
                    "content": "- Ambient color: green",
                }
            ],
        }
    )
    source = _source_turn("combined")

    result = model.review(
        source_turn=source,
        approved_memory=source.original_before_memory,
        max_memory_chars=8_192,
    )

    assert (
        responses.requests[0]["text_format"].__name__
        == "_VehicleCombinedCriticTransport"
    )
    assert "exact complete-line" in str(responses.requests[0]["instructions"])
    assert result.approved_memory_after == "### Gary\n- Ambient color: green"
    assert result.patch_stats == {
        "operation_count": 1,
        "add_count": 0,
        "replace_count": 1,
        "delete_count": 0,
        "inserted_characters": 22,
        "deleted_characters": 23,
    }
    assert (
        result.response.metadata["schema_version"]
        == VEHICLE_COMBINED_CRITIC_SCHEMA_VERSION
    )


def test_critic_accept_noop_preserves_memory_and_hash() -> None:
    model, _ = _model(
        {
            "verdict": "ACCEPT",
            "decision": "NO_OP",
            "reason_code": "TRANSIENT_COMMAND_ONLY",
            "reason": "The turn is only a transient command.",
            "evidence": [],
            "next_memory": None,
        }
    )
    source = _source_turn(
        "summary",
        content="[VehicleMemBench history turn=0]\nSet the fan to 2 now.",
    )

    result = model.review(
        source_turn=source,
        approved_memory=source.original_before_memory,
        max_memory_chars=8_192,
    )

    assert result.approved_memory_after == source.original_before_memory
    assert result.approved_memory_after_sha256 == (
        result.approved_memory_before_sha256
    )


def test_critic_retries_cross_field_semantic_error_and_aggregates_usage() -> None:
    responses = ScriptedResponses(
        [
            {
                "verdict": "REJECT",
                "decision": None,
                "reason_code": "NO_VEHICLE_PREFERENCE",
                "reason": "The turn does not state a durable preference.",
                "evidence": [],
                "next_memory": None,
            },
            {
                "verdict": "ACCEPT",
                "decision": "NO_OP",
                "reason_code": "NO_VEHICLE_PREFERENCE",
                "reason": "The turn does not state a durable preference.",
                "evidence": [],
                "next_memory": None,
            },
        ]
    )
    model = OpenAIVehicleMemoryCriticModel(
        "gpt-5.6-luna",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )
    source = _source_turn("summary")

    result = model.review(
        source_turn=source,
        approved_memory=source.original_before_memory,
        max_memory_chars=8_192,
    )

    assert result.response.verdict == "ACCEPT"
    assert result.response.usage == ModelUsage(200, 40, 240, 20)
    assert result.response.metadata["semantic_retry_count"] == 1
    assert len(result.response.metadata["attempt_latencies_ms"]) == 2
    assert len(result.response.metadata["semantic_retry_errors"]) == 1
    assert len(responses.requests) == 2
    assert "cross-field constraints" in str(responses.requests[1]["instructions"])


def test_critic_normalizes_source_dependent_correction_reason_error() -> None:
    operation = {
        "op": "replace",
        "target": "- Ambient color: orange",
        "content": "- Ambient color: green",
    }
    common = {
        "verdict": "CORRECT",
        "decision": "UPDATE",
        "reason": "The explicit durable setting requires a corrected patch.",
        "evidence": [{"message_id": 1, "quote": "prefers green lights"}],
        "operations": [operation],
    }
    responses = ScriptedResponses(
        [{**common, "reason_code": "MISSED_DURABLE_INFORMATION"}]
    )
    model = OpenAIVehicleMemoryCriticModel(
        "gpt-5.6-luna",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )
    source = _source_turn(
        "combined",
        original_decision="UPDATE",
        after_memory="### Gary\n- Ambient color: green",
    )

    result = model.review(
        source_turn=source,
        approved_memory=source.original_before_memory,
        max_memory_chars=8_192,
    )

    assert result.response.reason_code == "INCORRECT_DURABLE_INFORMATION"
    assert result.response.metadata["semantic_retry_count"] == 0
    assert result.response.metadata["semantic_normalizations"] == [
        "reason_code:MISSED_DURABLE_INFORMATION->INCORRECT_DURABLE_INFORMATION"
    ]
    assert len(responses.requests) == 1
    assert result.response.usage == ModelUsage(100, 20, 120, 10)


def test_critic_normalizes_redundant_correct_noop_to_accept() -> None:
    model, responses = _model(
        {
            "verdict": "CORRECT",
            "decision": "NO_OP",
            "reason_code": "NO_VEHICLE_PREFERENCE",
            "reason": "The turn contains no durable vehicle preference.",
            "evidence": [],
            "next_memory": None,
        }
    )
    source = _source_turn("summary", original_decision="NO_OP")

    result = model.review(
        source_turn=source,
        approved_memory=source.original_before_memory,
        max_memory_chars=8_192,
    )

    assert result.response.verdict == "ACCEPT"
    assert result.response.decision == "NO_OP"
    assert result.response.metadata["semantic_retry_count"] == 0
    assert result.response.metadata["semantic_normalizations"] == [
        "verdict:CORRECT->ACCEPT"
    ]
    assert len(responses.requests) == 1


def test_critic_reject_is_not_commit_ready() -> None:
    model, _ = _model(
        {
            "verdict": "REJECT",
            "decision": None,
            "reason_code": "NEEDS_CAUSAL_CONTEXT",
            "reason": "The referenced setting requires an earlier causal turn.",
            "evidence": [],
            "operations": [],
        }
    )
    source = _source_turn("combined")

    result = model.review(
        source_turn=source,
        approved_memory=source.original_before_memory,
        max_memory_chars=8_192,
    )

    assert result.commit_ready is False
    assert result.approved_memory_after is None
    assert result.approved_memory_after_sha256 is None


def test_accept_must_preserve_original_candidate_decision() -> None:
    response = VehicleMemoryCriticResponse(
        method="summary",
        verdict="ACCEPT",
        decision="UPDATE",
        reason_code="SUPPORTED_DURABLE_UPDATE",
        reason="The update is supported.",
        evidence=(CriticEvidence(1, "prefers green lights"),),
        next_memory="### Gary\n- Ambient color: green",
    )
    source = _source_turn("summary", original_decision="NO_OP")

    with pytest.raises(ValueError, match="preserve the original"):
        validate_vehicle_memory_critic_response(
            response,
            source_turn=source,
            approved_memory=source.original_before_memory,
            max_memory_chars=8_192,
        )


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        (CriticEvidence(2, "prefers green lights"), "current turn"),
        (CriticEvidence(1, "prefers blue lights"), "exact source text"),
    ],
)
def test_validator_rejects_invalid_current_turn_evidence(
    evidence: CriticEvidence,
    match: str,
) -> None:
    response = VehicleMemoryCriticResponse(
        method="summary",
        verdict="CORRECT",
        decision="UPDATE",
        reason_code="MISSED_DURABLE_INFORMATION",
        reason="The explicit preference should be retained.",
        evidence=(evidence,),
        next_memory="### Gary\n- Ambient color: green",
    )
    source = _source_turn("summary")

    with pytest.raises(ValueError, match=match):
        validate_vehicle_memory_critic_response(
            response,
            source_turn=source,
            approved_memory=source.original_before_memory,
            max_memory_chars=8_192,
        )


def test_validator_rejects_invalid_patch_without_mutating_input() -> None:
    approved_memory = "### Gary\n- Ambient color: orange"
    response = VehicleMemoryCriticResponse(
        method="combined",
        verdict="CORRECT",
        decision="UPDATE",
        reason_code="MISSED_DURABLE_INFORMATION",
        reason="The explicit preference should replace the old value.",
        evidence=(CriticEvidence(1, "prefers green lights"),),
        operations=(
            CriticPatchOperation(
                op="replace",
                target="- Ambient color: missing",
                content="- Ambient color: green",
            ),
        ),
    )
    source = _source_turn("combined")

    with pytest.raises(ValueError, match="occur exactly once"):
        validate_vehicle_memory_critic_response(
            response,
            source_turn=source,
            approved_memory=approved_memory,
            max_memory_chars=8_192,
        )
    assert approved_memory == "### Gary\n- Ambient color: orange"


def test_provider_validates_evidence_against_redacted_cloud_view() -> None:
    content = (
        "[VehicleMemBench history turn=0]\n"
        "Email driver@example.com prefers green lights."
    )
    model, responses = _model(
        {
            "verdict": "CORRECT",
            "decision": "UPDATE",
            "reason_code": "MISSED_DURABLE_INFORMATION",
            "reason": "The turn states a durable lighting preference.",
            "evidence": [
                {
                    "message_id": 1,
                    "quote": "[REDACTED_EMAIL] prefers green lights",
                }
            ],
            "next_memory": "### Driver\n- Ambient color: green",
        }
    )
    source = _source_turn("summary", content=content)

    result = model.review(
        source_turn=source,
        approved_memory=source.original_before_memory,
        max_memory_chars=8_192,
    )

    provider_input = json.loads(str(responses.requests[0]["input"]))
    assert "driver@example.com" not in provider_input["current_turn"]["content"]
    assert result.response.metadata["privacy"]["redacted_count"] == 1
    assert result.commit_ready is True


def test_method_schemas_are_strict_and_distinct() -> None:
    summary_schema = VehicleSummaryCriticPayload.model_json_schema()
    combined_schema = VehicleCombinedCriticPayload.model_json_schema()

    assert summary_schema["additionalProperties"] is False
    assert combined_schema["additionalProperties"] is False
    assert "next_memory" in summary_schema["properties"]
    assert "operations" not in summary_schema["properties"]
    assert "operations" in combined_schema["properties"]
    assert "next_memory" not in combined_schema["properties"]
    assert set(summary_schema["required"]) == set(summary_schema["properties"])
    assert set(combined_schema["required"]) == set(combined_schema["properties"])

    with pytest.raises(ValidationError):
        VehicleSummaryCriticPayload.model_validate(
            {
                "verdict": "ACCEPT",
                "decision": "NO_OP",
                "reason_code": "TRANSIENT_COMMAND_ONLY",
                "reason": "This is transient.",
                "evidence": [],
                "next_memory": "must be null",
            }
        )


def test_context_request_schema_is_strict_and_requires_a_hint() -> None:
    payload = VehicleCriticContextRequestPayload.model_validate(
        {
            "verdict": "CONTEXT_REQUIRED",
            "reason": "The setting reference needs earlier context.",
            "entity_hints": ["Gary"],
            "setting_hints": ["ambient color"],
            "temporal_hints": [],
        }
    )

    assert payload.entity_hints == ["Gary"]
    assert VEHICLE_CRITIC_CONTEXT_SCHEMA_VERSION.endswith("-v1")
    schema = VehicleCriticContextRequestPayload.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    with pytest.raises(ValidationError, match="at least one search hint"):
        VehicleCriticContextRequestPayload.model_validate(
            {
                "verdict": "CONTEXT_REQUIRED",
                "reason": "More context is required.",
                "entity_hints": [],
                "setting_hints": [],
                "temporal_hints": [],
            }
        )


def test_terra_initial_adjudication_can_request_bounded_context() -> None:
    source = _source_turn(
        "summary",
        content="[VehicleMemBench history turn=2]\nMake that setting green.",
        turn_index=2,
        message_id=3,
    )
    model, responses = _adjudicator(
        {
            "resolution": "CONTEXT_REQUIRED",
            "final": None,
            "context_request": {
                "verdict": "CONTEXT_REQUIRED",
                "reason": "The referenced setting needs earlier context.",
                "entity_hints": ["Gary"],
                "setting_hints": ["setting", "green"],
                "temporal_hints": [],
            },
            "rollback_request": None,
        }
    )

    result = model.adjudicate(
        source_turn=source,
        approved_memory=source.original_before_memory,
        luna_reject=_luna_reject(source),
        max_memory_chars=8_192,
    )

    request = responses.requests[0]
    assert (
        request["text_format"].__name__
        == "_VehicleSummaryAdjudicationTransport"
    )
    assert request["model"] == "gpt-5.6-terra"
    assert request["store"] is False
    provider_input = json.loads(str(request["input"]))
    assert provider_input["phase"] == "initial"
    assert provider_input["recovered_context"] == []
    assert result.resolution == "CONTEXT_REQUIRED"
    assert result.context_request is not None
    assert result.context_request.entity_hints == ["Gary"]
    assert result.metadata["schema_version"] == (
        VEHICLE_SUMMARY_ADJUDICATOR_SCHEMA_VERSION
    )
    assert result.metadata["prompt_version"] == (
        VEHICLE_MEMORY_ADJUDICATOR_PROMPT_VERSION
    )


def test_terra_recovered_context_resolves_current_turn_with_both_evidence() -> None:
    prior = _source_turn(
        "summary",
        content="[VehicleMemBench history turn=0]\nGary discussed ambient lighting.",
        before_memory="",
        turn_index=0,
        message_id=1,
    )
    current = _source_turn(
        "summary",
        content="[VehicleMemBench history turn=2]\nMake that ambient color green.",
        before_memory="### Gary\n- Ambient color: orange",
        turn_index=2,
        message_id=3,
    )
    model, _ = _adjudicator(
        {
            "resolution": "FINAL",
            "final": {
                "verdict": "CORRECT",
                "decision": "UPDATE",
                "reason_code": "MISSED_DURABLE_INFORMATION",
                "reason": "The current turn explicitly changes the referenced setting.",
                "evidence": [
                    {"message_id": 1, "quote": "ambient lighting"},
                    {"message_id": 3, "quote": "ambient color green"},
                ],
                "next_memory": "### Gary\n- Ambient color: green",
            },
            "context_request": None,
            "rollback_request": None,
        }
    )

    result = model.adjudicate(
        source_turn=current,
        approved_memory=current.original_before_memory,
        luna_reject=_luna_reject(current),
        recovered_context=(prior,),
        max_memory_chars=8_192,
    )

    assert result.resolution == "FINAL"
    assert result.final_result is not None
    assert result.final_result.commit_ready is True
    assert result.final_result.approved_memory_after.endswith("color: green")


def test_terra_recovered_context_can_request_prior_turn_rollback() -> None:
    prior = _source_turn(
        "combined",
        content=(
            "[VehicleMemBench history turn=0]\n"
            "Gary always prefers green ambient lighting."
        ),
        before_memory="",
        turn_index=0,
        message_id=1,
    )
    current = _source_turn(
        "combined",
        content="[VehicleMemBench history turn=2]\nUse my usual ambient setting.",
        before_memory="### Gary\n- Audio volume: 4",
        turn_index=2,
        message_id=3,
    )
    model, _ = _adjudicator(
        {
            "resolution": "ROLLBACK_REQUIRED",
            "final": None,
            "context_request": None,
            "rollback_request": {
                "source_message_id": 1,
                "quote": "always prefers green ambient lighting",
                "reason": "The earlier durable preference was omitted.",
            },
        }
    )

    result = model.adjudicate(
        source_turn=current,
        approved_memory=current.original_before_memory,
        luna_reject=_luna_reject(current),
        recovered_context=(prior,),
        max_memory_chars=8_192,
    )

    assert result.resolution == "ROLLBACK_REQUIRED"
    assert result.rollback_request is not None
    assert result.rollback_request.source_message_id == 1


def test_terra_rejects_rollback_to_unretrieved_turn() -> None:
    prior = _source_turn(
        "summary",
        content="[VehicleMemBench history turn=0]\nGary discussed lighting.",
        before_memory="",
        turn_index=0,
        message_id=1,
    )
    current = _source_turn(
        "summary",
        content="[VehicleMemBench history turn=2]\nUse my usual setting.",
        turn_index=2,
        message_id=3,
    )
    model, _ = _adjudicator(
        {
            "resolution": "ROLLBACK_REQUIRED",
            "final": None,
            "context_request": None,
            "rollback_request": {
                "source_message_id": 2,
                "quote": "missing quote",
                "reason": "A different turn was omitted.",
            },
        }
    )

    with pytest.raises(RuntimeError, match="not present"):
        model.adjudicate(
            source_turn=current,
            approved_memory=current.original_before_memory,
            luna_reject=_luna_reject(current),
            recovered_context=(prior,),
            max_memory_chars=8_192,
        )


def test_terra_repairs_selected_omission_against_rollback_memory() -> None:
    source = _source_turn(
        "summary",
        content=(
            "[VehicleMemBench history turn=0]\n"
            "Gary always prefers green ambient lighting."
        ),
        before_memory="",
        turn_index=0,
        message_id=1,
    )
    rollback_request = VehicleCriticRollbackRequestPayload(
        source_message_id=1,
        quote="always prefers green ambient lighting",
        reason="The durable preference was omitted.",
    )
    model, responses = _adjudicator(
        {
            "verdict": "CORRECT",
            "decision": "UPDATE",
            "reason_code": "MISSED_DURABLE_INFORMATION",
            "reason": "The source turn explicitly states a durable preference.",
            "evidence": [
                {
                    "message_id": 1,
                    "quote": "always prefers green ambient lighting",
                }
            ],
            "next_memory": "### Gary\n- Ambient color: green",
        }
    )

    result = model.repair_omission(
        source_turn=source,
        approved_memory="",
        rollback_request=rollback_request,
        trigger_message_id=3,
        max_memory_chars=8_192,
    )

    assert result.commit_ready is True
    assert result.approved_memory_after == "### Gary\n- Ambient color: green"
    assert (
        responses.requests[0]["text_format"].__name__
        == "_VehicleSummaryCriticTransport"
    )
    assert "earlier source turn" in str(responses.requests[0]["instructions"])
    assert "Return FINAL" not in str(responses.requests[0]["instructions"])
    assert result.response.metadata["prompt_version"] == (
        VEHICLE_MEMORY_REPAIR_PROMPT_VERSION
    )


def test_adjudicator_schemas_are_strict_and_fully_required() -> None:
    for payload_type in (
        VehicleSummaryAdjudicationPayload,
        VehicleCombinedAdjudicationPayload,
    ):
        schema = payload_type.model_json_schema()
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        for definition in schema.get("$defs", {}).values():
            if definition.get("type") != "object":
                continue
            assert definition["additionalProperties"] is False
            assert set(definition["required"]) == set(definition["properties"])
