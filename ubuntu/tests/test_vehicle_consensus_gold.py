from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from palmclaw_ubuntu.providers import (
    OpenAITemporalAwareRecursiveSummaryPatchMemoryModel,
)
from palmclaw_ubuntu.vehicle_bench.consensus_gold import (
    CONSENSUS_GOLD_PATCH_R2_PROMPT_VERSION,
    CONSENSUS_GOLD_SCHEMA_VERSION,
    V2_PATCH_R2_OUTPUT_INSTRUCTIONS,
    CombinedTurnCandidatePayload,
    CompactionCandidatePayload,
    GeneratedCombinedCandidate,
    GeneratedCompactionCandidate,
    OpenAIV2CandidateResolverModel,
    OpenAIV2CombinedCandidateModel,
    OpenAIV2CompactionCandidateModel,
    OpenAIV2CompactionResolverModel,
    build_compaction_consensus,
    build_compaction_input,
    build_consensus,
    build_gold_generation_input,
    evaluate_candidate,
    evaluate_compaction_candidate,
    render_vehicle_turn,
)
from palmclaw_ubuntu.vehicle_bench.memory import VehicleHistoryEntry


class FakeCreateResponses:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            id="v1-response",
            status="completed",
            output=[],
            usage=None,
        )


class FakeParseResponses:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            id=f"candidate-{len(self.requests)}",
            status="completed",
            output_parsed=self.payload,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                input_tokens_details=SimpleNamespace(cached_tokens=10),
            ),
        )


def _entry(content: str = "I prefer green ambient lighting.") -> VehicleHistoryEntry:
    timestamp = datetime.strptime("2025-04-10 13:00", "%Y-%m-%d %H:%M")
    return VehicleHistoryEntry(
        line_number=7,
        timestamp=timestamp,
        speaker="Gary",
        content=content,
        raw=f"[2025-04-10 13:00] Gary: {content}",
    )


def _update_payload(
    *,
    color: str = "green",
    quote: str = "prefer green ambient lighting",
) -> CombinedTurnCandidatePayload:
    return CombinedTurnCandidatePayload.model_validate(
        {
            "decision": "UPDATE",
            "operations": [
                {
                    "op": "replace",
                    "target": "- Ambient color: orange",
                    "content": f"- Ambient color: {color}",
                    "identity_key": "gary.ambient_color",
                    "temporal_action": "durable_upsert",
                    "temporal_cue": "",
                }
            ],
            "reason": "Gary explicitly changes his ambient-lighting preference.",
            "evidence": [{"quote": quote}],
        }
    )


def _candidate(
    sample_id: str,
    generation_input,
    *,
    payload: CombinedTurnCandidatePayload | None = None,
) -> GeneratedCombinedCandidate:
    return GeneratedCombinedCandidate(
        sample_id=sample_id,
        payload=payload or _update_payload(),
        response_id=f"response-{sample_id}",
        model_id="gpt-5.6-luna",
        prompt_version="test-prompt",
        schema_version=CONSENSUS_GOLD_SCHEMA_VERSION,
        input_sha256=generation_input.input_sha256,
        usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cached_tokens": 0,
        },
    )


def test_v2_builder_matches_v1_combined_provider_input() -> None:
    entry = _entry()
    turn = render_vehicle_turn(0, entry)
    responses = FakeCreateResponses()
    model = OpenAITemporalAwareRecursiveSummaryPatchMemoryModel(
        "memory-model",
        timeout_seconds=1,
        instructions="Temporal Combined instructions",
        update_cadence="history_entry",
        client=SimpleNamespace(responses=responses),
    )
    previous = "### Gary\n- Ambient color: orange"

    model.update(
        previous_memory=previous,
        date=entry.date,
        daily_history=turn,
    )
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=entry,
        previous_memory=previous,
    )

    assert responses.requests[0]["input"] == generation_input.provider_input
    assert generation_input.current_turn == turn
    assert generation_input.message_id == entry.line_number


def test_candidate_gate_applies_temporal_patch_and_checks_evidence() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )

    result = evaluate_candidate(
        _candidate("A", generation_input),
        generation_input=generation_input,
    )

    assert result.status == "PASS"
    assert result.error_codes == ()
    assert result.after_memory == "### Gary\n- Ambient color: green"
    assert result.patch_stats["replace_count"] == 1
    assert result.semantic_fingerprint is not None


def test_candidate_gate_rejects_non_current_evidence() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    payload = _update_payload(quote="a quote from another turn")

    result = evaluate_candidate(
        _candidate("A", generation_input, payload=payload),
        generation_input=generation_input,
    )

    assert result.status == "FAIL"
    assert result.error_codes == ("EVIDENCE_NOT_IN_CURRENT_TURN",)
    assert result.semantic_fingerprint is None


def test_candidate_gate_rejects_missing_patch_target() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    payload = _update_payload().model_copy(deep=True)
    payload.operations[0].target = "- Ambient color: missing"

    result = evaluate_candidate(
        _candidate("A", generation_input, payload=payload),
        generation_input=generation_input,
    )

    assert result.status == "FAIL"
    assert result.error_codes == ("PATCH_APPLY_FAILED",)
    assert result.after_memory is None


def test_consensus_auto_accepts_three_matching_candidates() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    results = tuple(
        evaluate_candidate(
            _candidate(sample_id, generation_input),
            generation_input=generation_input,
        )
        for sample_id in ("A", "B", "C")
    )

    consensus = build_consensus(results)

    assert consensus.route == "AUTO_ACCEPT"
    assert consensus.agreement == "3/3"
    assert consensus.accepted_sample_id == "A"


def test_consensus_ignores_identity_key_only_differences() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    identity_keys = (
        "gary.ambient_color",
        "gary.ambient_lighting_color",
        "gary.current_ambient_color",
    )
    payloads = []
    for identity_key in identity_keys:
        payload = _update_payload().model_copy(deep=True)
        payload.operations[0].identity_key = identity_key
        payloads.append(payload)
    results = tuple(
        evaluate_candidate(
            _candidate(sample_id, generation_input, payload=payload),
            generation_input=generation_input,
        )
        for sample_id, payload in zip(("A", "B", "C"), payloads, strict=True)
    )

    consensus = build_consensus(results)

    assert consensus.route == "AUTO_ACCEPT"
    assert consensus.agreement == "3/3"
    assert consensus.accepted_sample_id == "A"


def test_consensus_keeps_temporal_action_material() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    payloads = [_update_payload() for _ in range(3)]
    payloads[2].operations[0].temporal_action = "current_upsert"
    results = tuple(
        evaluate_candidate(
            _candidate(sample_id, generation_input, payload=payload),
            generation_input=generation_input,
        )
        for sample_id, payload in zip(("A", "B", "C"), payloads, strict=True)
    )

    consensus = build_consensus(results)

    assert consensus.route == "TERRA"
    assert consensus.agreement == "2/3"


def test_consensus_routes_material_two_of_three_to_terra() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    payloads = (_update_payload(), _update_payload(), _update_payload(color="blue"))
    results = tuple(
        evaluate_candidate(
            _candidate(sample_id, generation_input, payload=payload),
            generation_input=generation_input,
        )
        for sample_id, payload in zip(("A", "B", "C"), payloads, strict=True)
    )

    consensus = build_consensus(results)

    assert consensus.route == "TERRA"
    assert consensus.agreement == "2/3"
    assert consensus.preferred_sample_id == "A"
    assert consensus.accepted_sample_id is None


def test_openai_adapter_uses_same_input_for_three_independent_calls() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    payload = _update_payload().model_dump()
    responses = FakeParseResponses(payload)
    model = OpenAIV2CombinedCandidateModel(
        "gpt-5.6-luna",
        timeout_seconds=1,
        base_instructions="V1 Temporal Combined instructions",
        client=SimpleNamespace(responses=responses),
    )

    candidates = tuple(
        model.generate(generation_input, sample_id=sample_id)
        for sample_id in ("A", "B", "C")
    )

    assert [candidate.sample_id for candidate in candidates] == ["A", "B", "C"]
    assert len({candidate.input_sha256 for candidate in candidates}) == 1
    assert all(
        request["input"] == generation_input.provider_input
        for request in responses.requests
    )
    assert all(
        request["reasoning"] == {"effort": "medium"} for request in responses.requests
    )
    assert all("previous_response_id" not in request for request in responses.requests)


def test_patch_r2_adapter_uses_non_temporal_policy_and_distinct_version() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    payload = _update_payload().model_dump()
    payload["operations"][0]["temporal_action"] = "non_temporal"
    responses = FakeParseResponses(payload)
    model = OpenAIV2CombinedCandidateModel(
        "gpt-5.6-luna",
        timeout_seconds=1,
        base_instructions="V1 non-temporal Patch R2 instructions",
        output_instructions=V2_PATCH_R2_OUTPUT_INSTRUCTIONS,
        prompt_version=CONSENSUS_GOLD_PATCH_R2_PROMPT_VERSION,
        client=SimpleNamespace(responses=responses),
    )

    candidate = model.generate(generation_input, sample_id="A")

    assert candidate.prompt_version == CONSENSUS_GOLD_PATCH_R2_PROMPT_VERSION
    assert "temporal_action=non_temporal" in str(
        responses.requests[0]["instructions"]
    )


def test_terra_resolver_selects_supplied_candidate_without_extra_context() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    payloads = (_update_payload(), _update_payload(), _update_payload(color="blue"))
    results = tuple(
        evaluate_candidate(
            _candidate(sample_id, generation_input, payload=payload),
            generation_input=generation_input,
        )
        for sample_id, payload in zip(("A", "B", "C"), payloads, strict=True)
    )
    consensus = build_consensus(results)
    responses = FakeParseResponses(
        {
            "verdict": "SELECT",
            "selected_sample_id": "A",
            "corrected_candidate": None,
            "reason": "Candidate A is directly supported by the current turn.",
        }
    )
    resolver = OpenAIV2CandidateResolverModel(
        "gpt-5.6-terra",
        stage="TERRA",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    resolved = resolver.resolve(
        generation_input=generation_input,
        gate_results=results,
        consensus=consensus,
    )

    assert resolved is not None
    assert resolved.sample_id == "TERRA"
    assert resolved.payload == results[0].candidate.payload
    provider_input = json.loads(str(responses.requests[0]["input"]))
    assert set(provider_input) == {
        "approved_memory_before",
        "candidates",
        "consensus",
        "current_turn",
        "scenario_index",
        "turn_index",
    }
    assert responses.requests[0]["reasoning"] == {"effort": "high"}


def test_sol_resolver_can_return_corrected_candidate() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    payloads = (_update_payload(), _update_payload(), _update_payload(color="blue"))
    results = tuple(
        evaluate_candidate(
            _candidate(sample_id, generation_input, payload=payload),
            generation_input=generation_input,
        )
        for sample_id, payload in zip(("A", "B", "C"), payloads, strict=True)
    )
    responses = FakeParseResponses(
        {
            "verdict": "CORRECT",
            "selected_sample_id": None,
            "corrected_candidate": _update_payload().model_dump(),
            "reason": "The corrected patch follows the explicit evidence.",
        }
    )
    resolver = OpenAIV2CandidateResolverModel(
        "gpt-5.6-sol",
        stage="SOL",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    resolved = resolver.resolve(
        generation_input=generation_input,
        gate_results=results,
        consensus=build_consensus(results),
    )

    assert resolved is not None
    assert resolved.sample_id == "SOL"
    assert resolved.payload == _update_payload()


def test_resolver_uncertain_returns_no_candidate() -> None:
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=_entry(),
        previous_memory="### Gary\n- Ambient color: orange",
    )
    results = tuple(
        evaluate_candidate(
            _candidate(sample_id, generation_input),
            generation_input=generation_input,
        )
        for sample_id in ("A", "B", "C")
    )
    responses = FakeParseResponses(
        {
            "verdict": "UNCERTAIN",
            "selected_sample_id": None,
            "corrected_candidate": None,
            "reason": "The current evidence is insufficient to resolve safely.",
        }
    )
    resolver = OpenAIV2CandidateResolverModel(
        "gpt-5.6-terra",
        stage="TERRA",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    resolved = resolver.resolve(
        generation_input=generation_input,
        gate_results=results,
        consensus=build_consensus(results),
    )

    assert resolved is None


def test_compaction_gate_and_consensus_accept_matching_shorter_rewrites() -> None:
    compaction_input = build_compaction_input(
        scenario_index=1,
        turn_index=29,
        post_patch_memory=(
            "### Gary\n- Gary consistently prefers green ambient lighting in the car."
        ),
    )
    results = []
    for sample_id in ("A", "B", "C"):
        candidate = GeneratedCompactionCandidate(
            sample_id=sample_id,
            payload=CompactionCandidatePayload(
                next_memory="### Gary\n- Ambient: green"
            ),
            response_id=f"compact-{sample_id}",
            model_id="gpt-5.6-luna",
            prompt_version="test",
            schema_version=CONSENSUS_GOLD_SCHEMA_VERSION,
            input_sha256=compaction_input.input_sha256,
            usage={},
        )
        results.append(
            evaluate_compaction_candidate(
                candidate,
                compaction_input=compaction_input,
            )
        )

    consensus = build_compaction_consensus(results)

    assert all(result.status == "PASS" for result in results)
    assert consensus.route == "AUTO_ACCEPT"
    assert consensus.agreement == "3/3"


def test_compaction_disagreement_routes_directly_to_sol() -> None:
    compaction_input = build_compaction_input(
        scenario_index=2,
        turn_index=29,
        post_patch_memory=(
            "### Gary\n- Gary consistently prefers green ambient lighting in the car."
        ),
    )
    memories = (
        "### Gary\n- Ambient: green",
        "### Gary\n- Lighting: green",
        "### Gary\n- Panel: green",
    )
    results = []
    for sample_id, memory in zip(("A", "B", "C"), memories, strict=True):
        candidate = GeneratedCompactionCandidate(
            sample_id=sample_id,
            payload=CompactionCandidatePayload(next_memory=memory),
            response_id=f"compact-{sample_id}",
            model_id="gpt-5.6-luna",
            prompt_version="test",
            schema_version=CONSENSUS_GOLD_SCHEMA_VERSION,
            input_sha256=compaction_input.input_sha256,
            usage={},
        )
        results.append(
            evaluate_compaction_candidate(
                candidate,
                compaction_input=compaction_input,
            )
        )

    consensus = build_compaction_consensus(results)

    assert consensus.route == "SOL"
    assert consensus.agreement == "1/1/1"


def test_compaction_openai_adapters_use_fixed_effort_and_scoped_input() -> None:
    compaction_input = build_compaction_input(
        scenario_index=1,
        turn_index=29,
        post_patch_memory=(
            "### Gary\n- Gary consistently prefers green ambient lighting in the car."
        ),
    )
    generator_responses = FakeParseResponses(
        {"next_memory": "### Gary\n- Ambient: green"}
    )
    generator = OpenAIV2CompactionCandidateModel(
        "gpt-5.6-luna",
        timeout_seconds=1,
        base_instructions="V1 temporal compaction instructions",
        client=SimpleNamespace(responses=generator_responses),
    )
    candidates = tuple(
        generator.generate(compaction_input, sample_id=sample_id)
        for sample_id in ("A", "B", "C")
    )
    gates = tuple(
        evaluate_compaction_candidate(
            candidate,
            compaction_input=compaction_input,
        )
        for candidate in candidates
    )
    resolver_responses = FakeParseResponses(
        {
            "verdict": "SELECT",
            "selected_sample_id": "A",
            "corrected_memory": None,
            "reason": "Candidate A preserves the supplied memory compactly.",
        }
    )
    resolver = OpenAIV2CompactionResolverModel(
        "gpt-5.6-terra",
        stage="TERRA",
        timeout_seconds=1,
        client=SimpleNamespace(responses=resolver_responses),
    )

    resolved = resolver.resolve(
        compaction_input=compaction_input,
        gate_results=gates,
        consensus=build_compaction_consensus(gates),
    )

    assert resolved is not None
    assert all(
        request["reasoning"] == {"effort": "medium"}
        for request in generator_responses.requests
    )
    assert resolver_responses.requests[0]["reasoning"] == {"effort": "high"}
    provider_input = json.loads(str(resolver_responses.requests[0]["input"]))
    assert set(provider_input) == {
        "candidates",
        "consensus",
        "post_patch_memory",
        "scenario_index",
        "target_tokens",
        "turn_index",
    }
