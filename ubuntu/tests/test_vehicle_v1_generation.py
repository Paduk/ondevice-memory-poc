from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_EVENT_CHAIN_PROMPT_VERSION,
    V1_PERSONA_PROMPT_VERSION,
    V1_PUBLISHED_REASONING_COUNTS,
    OpenAIV1PersonaGenerationModel,
    V1EventChainPayload,
    V1GeneratedEventChains,
    V1GeneratedPersonaGroup,
    V1PersonaGroupPayload,
    V1PersonaSeed,
    build_persona_seed_groups,
    build_reasoning_type_plan,
    build_vehicle_attribute_catalog,
    generate_stage2_artifact,
    interleave_event_chains,
    load_persona_seeds,
    validate_stage2_contract,
    write_stage2_artifact,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1EventChainRecord,
    V1EventRecord,
    V1NamedValue,
    V1PersonaRecord,
    V1PreferenceUpdate,
    canonical_json_sha256,
)


def _tool_schemas() -> tuple[dict, ...]:
    return (
        {
            "name": "carcontrol_HUD_set_brightness_level",
            "description": "Set HUD brightness.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["level"],
                "additionalProperties": False,
            },
        },
        {
            "name": "carcontrol_navigation_stop",
            "description": "Stop navigation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    )


def _seeds() -> tuple[V1PersonaSeed, ...]:
    return tuple(
        V1PersonaSeed(
            seed_id=f"p{index}",
            source="persona-hub-elite",
            profile={"name": f"Person {index}", "occupation": "Researcher"},
        )
        for index in range(3)
    )


def _personas() -> tuple[V1PersonaRecord, ...]:
    return tuple(
        V1PersonaRecord(
            persona_id=f"p{index}",
            name=f"Person {index}",
            basic_profile=(
                V1NamedValue(key="age", value="35"),
                V1NamedValue(key="education", value="PhD"),
                V1NamedValue(key="occupation", value="Researcher"),
                V1NamedValue(key="mbti", value="INTJ"),
            ),
            cultural_interests=(V1NamedValue(key="music", value="jazz"),),
            lifestyle_habits=(V1NamedValue(key="travel", value="weekends"),),
            vehicle_preferences=(V1NamedValue(key="HUD", value="varied"),),
        )
        for index in range(3)
    )


def _event_payload(reasoning_types: tuple[str, ...]) -> V1EventChainPayload:
    background = []
    vehicle = []
    for chain_index in range(20):
        events = tuple(
            V1EventRecord(
                event_id=f"background-{chain_index}-{event_index}",
                timestamp=(
                    f"2025-01-{chain_index + 1:02d}T{8 + event_index:02d}:00"
                ),
                description="A non-vehicle life event.",
                participant_ids=("p0", "p1"),
            )
            for event_index in range(3)
        )
        background.append(
            V1EventChainRecord(
                chain_id=f"background-{chain_index}",
                kind="background",
                events=events,
            )
        )
    for chain_offset, reasoning_type in enumerate(reasoning_types):
        day = chain_offset + 21
        events = (
            V1EventRecord(
                event_id=f"vehicle-{chain_offset}-0",
                timestamp=f"2025-01-{day:02d}T08:00",
                description="Person 0 discovers a vehicle preference.",
                participant_ids=("p0", "p1"),
                preference_updates=(
                    V1PreferenceUpdate(
                        subject_id="p0",
                        attribute_path=(
                            "carcontrol_HUD_set_brightness_level.level"
                        ),
                        new_value=8,
                    ),
                ),
            ),
            V1EventRecord(
                event_id=f"vehicle-{chain_offset}-1",
                timestamp=f"2025-01-{day:02d}T09:00",
                description="Person 0 later confirms the contextual preference.",
                participant_ids=("p0", "p2"),
            ),
        )
        vehicle.append(
            V1EventChainRecord(
                chain_id=f"vehicle-{chain_offset}",
                kind="vehicle",
                reasoning_type=reasoning_type,
                delayed_query_seed="Apply Person 0's established HUD preference.",
                events=events,
            )
        )
    return V1EventChainPayload(
        background_chains=tuple(background),
        vehicle_chains=tuple(vehicle),
    )


class _PersonaModel:
    def generate(self, seeds, *, candidate_group_id):
        return V1GeneratedPersonaGroup(
            candidate_group_id=candidate_group_id,
            seed_persona_ids=tuple(seed.seed_id for seed in seeds),
            seed_personas=tuple(seeds),
            payload=V1PersonaGroupPayload(personas=_personas()),
            model_id="fake-persona",
            prompt_version=V1_PERSONA_PROMPT_VERSION,
            input_sha256="a" * 64,
            usage={},
        )


class _EventModel:
    def generate(
        self,
        personas,
        *,
        scenario_candidate_id,
        reasoning_types,
        vehicle_attributes,
    ):
        del personas
        catalog_payload = [item.model_dump(mode="json") for item in vehicle_attributes]
        return V1GeneratedEventChains(
            scenario_candidate_id=scenario_candidate_id,
            planned_reasoning_types=tuple(reasoning_types),
            payload=_event_payload(tuple(reasoning_types)),
            model_id="fake-event",
            prompt_version=V1_EVENT_CHAIN_PROMPT_VERSION,
            input_sha256="b" * 64,
            vehicle_catalog_sha256=canonical_json_sha256(catalog_payload),
            usage={},
        )


class _FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="response-1",
            status="completed",
            output_parsed=self.payload,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                input_tokens_details=SimpleNamespace(cached_tokens=10),
            ),
        )


def test_reasoning_plan_scales_published_distribution_exactly() -> None:
    plan = build_reasoning_type_plan(100)

    assert len(plan) == 100
    assert all(len(scenario) == 10 for scenario in plan)
    observed = Counter(item for scenario in plan for item in scenario)
    assert observed == {
        reasoning_type: count * 2
        for reasoning_type, count in V1_PUBLISHED_REASONING_COUNTS.items()
    }
    assert plan == build_reasoning_type_plan(100)


def test_persona_seed_loader_and_grouping_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "persona_seeds.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "seed_id": f"seed-{index}",
                    "source": "persona-hub-elite",
                    "profile": {"name": f"Person {index}"},
                }
            )
            for index in range(9)
        )
        + "\n",
        encoding="utf-8",
    )

    seeds = load_persona_seeds(path)
    first = build_persona_seed_groups(seeds, candidate_group_count=3)
    second = build_persona_seed_groups(seeds, candidate_group_count=3)

    assert first == second
    assert len(first) == 3
    assert len({item.seed_id for group in first for item in group}) == 9


def test_catalog_and_stage2_validator_bind_updates_to_tool_schema() -> None:
    catalog = build_vehicle_attribute_catalog(_tool_schemas())
    reasoning_types = build_reasoning_type_plan(1)[0]
    payload = _event_payload(reasoning_types)

    audit = validate_stage2_contract(
        personas=_personas(),
        event_payload=payload,
        planned_reasoning_types=reasoning_types,
        vehicle_attributes=catalog,
    )
    timeline = interleave_event_chains(payload.all_chains)

    assert len(catalog) == 1
    assert catalog[0].path == "carcontrol_HUD_set_brightness_level.level"
    assert audit.event_count == 80
    assert audit.attribute_update_count == 10
    assert len(timeline) == 80
    assert [item.timeline_index for item in timeline] == list(range(80))


def test_stage2_validator_rejects_invalid_vehicle_value() -> None:
    payload = _event_payload(build_reasoning_type_plan(1)[0])
    raw = payload.model_dump(mode="json")
    raw["vehicle_chains"][0]["events"][0]["preference_updates"][0][
        "new_value"
    ] = 99
    invalid = V1EventChainPayload.model_validate(raw)

    with pytest.raises(ValueError, match="invalid preference value"):
        validate_stage2_contract(
            personas=_personas(),
            event_payload=invalid,
            planned_reasoning_types=build_reasoning_type_plan(1)[0],
            vehicle_attributes=build_vehicle_attribute_catalog(_tool_schemas()),
        )


def test_stage2_orchestrator_writes_hash_checked_artifact(tmp_path: Path) -> None:
    artifact = generate_stage2_artifact(
        candidate_group_id="group-001",
        scenario_candidate_id="scenario-001",
        seeds=_seeds(),
        reasoning_types=build_reasoning_type_plan(1)[0],
        vehicle_attributes=build_vehicle_attribute_catalog(_tool_schemas()),
        persona_model=_PersonaModel(),
        event_model=_EventModel(),
    )
    path = write_stage2_artifact(tmp_path, artifact)
    stored = json.loads(path.read_text())

    assert artifact.audit.passed is True
    assert stored["artifact_sha256"] == artifact.artifact_sha256
    stored["audit"]["event_count"] = 79
    with pytest.raises(ValidationError, match="artifact_sha256"):
        type(artifact).model_validate(stored)


def test_openai_persona_adapter_uses_terra_reasoning_and_structured_output() -> None:
    payload = V1PersonaGroupPayload(personas=_personas())
    responses = _FakeResponses(payload)
    model = OpenAIV1PersonaGenerationModel(
        "paper-persona-model",
        timeout_seconds=30,
        client=SimpleNamespace(responses=responses),
    )

    generated = model.generate(_seeds(), candidate_group_id="group-001")

    assert generated.payload == payload
    assert generated.usage["cached_tokens"] == 10
    assert "temperature" not in responses.kwargs
    assert responses.kwargs["reasoning"] == {"effort": "medium"}
    assert responses.kwargs["text_format"] is V1PersonaGroupPayload
