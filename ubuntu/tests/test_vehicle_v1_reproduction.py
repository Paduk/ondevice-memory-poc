from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from palmclaw_ubuntu.vehicle_bench.dataset import (
    BenchmarkManifest,
    GoldToolCall,
    VehicleBenchmarkDataset,
    VehicleScenario,
    VehicleTask,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1DialogueTurnRecord,
    V1EventChainRecord,
    V1EventRecord,
    V1FinalQuizRecord,
    V1GoldToolCallRecord,
    V1NamedValue,
    V1PersonaRecord,
    V1PreferenceUpdate,
    V1ScenarioArtifact,
    build_v1_reference_profile,
    build_v1_reproduction_manifest,
    canonical_json_sha256,
    default_v1_reproduction_decisions,
    write_v1_reproduction_reference,
)


class _WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def _dataset(tmp_path: Path) -> VehicleBenchmarkDataset:
    tmp_path.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for scenario_index in (1, 2):
        history = tmp_path / f"history_{scenario_index}.txt"
        qa = tmp_path / f"qa_{scenario_index}.json"
        history.write_text(
            "[2025-01-01 08:00] Gary Allen: Set the HUD to my daytime setting.\n"
            "[2025-01-01 08:01] Pat Lee: I prefer the HUD at level 8.\n",
            encoding="utf-8",
        )
        qa_payload = {
            "related_to_vehicle_preference": [
                {
                    "gold_memory": "[January 1, 2025] Pat prefers HUD level 8.",
                    "reasoning_type": "conditional_constraint",
                    "query": "Use Pat's daytime HUD setting.",
                    "new_answer": [
                        "carcontrol_HUD_set_brightness_level(level=8)"
                    ],
                }
            ]
        }
        qa.write_text(json.dumps(qa_payload), encoding="utf-8")
        task = VehicleTask(
            id=f"vehicle-{scenario_index:02d}-00",
            scenario_index=scenario_index,
            event_index=0,
            query="Use Pat's daytime HUD setting.",
            gold_memory="[January 1, 2025] Pat prefers HUD level 8.",
            reasoning_type="conditional_constraint",
            gold_calls=(
                GoldToolCall(
                    name="carcontrol_HUD_set_brightness_level",
                    arguments={"level": 8},
                    source="carcontrol_HUD_set_brightness_level(level=8)",
                ),
            ),
        )
        scenarios.append(
            VehicleScenario(
                index=scenario_index,
                history_path=history,
                qa_path=qa,
                tasks=(task,),
            )
        )
    return VehicleBenchmarkDataset(
        root=tmp_path,
        scenarios=tuple(scenarios),
        tool_schemas=(
            {
                "name": "carcontrol_HUD_set_brightness_level",
                "description": "Set brightness.",
                "parameters": {"type": "object"},
            },
        ),
        manifest=BenchmarkManifest(
            upstream_commit="a" * 40,
            dataset_sha256="b" * 64,
            tool_schema_sha256="c" * 64,
            scenario_count=2,
            task_count=2,
            tool_count=1,
        ),
    )


def _complete_scenario() -> V1ScenarioArtifact:
    personas = tuple(
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
    chains = []
    turns = []
    quizzes = []
    for index in range(30):
        vehicle = index >= 20
        event_id = f"event-{index}"
        chain_id = f"chain-{index}"
        update = (
            V1PreferenceUpdate(
                subject_id="p0",
                attribute_path="HUD.brightness",
                new_value=8,
            ),
        ) if vehicle else ()
        event = V1EventRecord(
            event_id=event_id,
            timestamp=f"2025-01-{index + 1:02d}T08:00",
            description="A source event.",
            participant_ids=("p0", "p1"),
            preference_updates=update,
        )
        chains.append(
            V1EventChainRecord(
                chain_id=chain_id,
                kind="vehicle" if vehicle else "background",
                reasoning_type="conditional_constraint" if vehicle else None,
                delayed_query_seed="Use the delayed setting." if vehicle else None,
                events=(event,),
            )
        )
        turns.append(
            V1DialogueTurnRecord(
                turn_id=f"turn-{index}",
                source_event_id=event_id,
                timestamp=f"2025-01-{index + 1:02d}T08:00",
                speaker_id="p0",
                speaker_name="Person 0",
                text="A natural conversation turn.",
            )
        )
        if vehicle:
            quizzes.append(
                V1FinalQuizRecord(
                    quiz_id=f"quiz-{index}",
                    source_chain_id=chain_id,
                    reasoning_type="conditional_constraint",
                    gold_memory="Person 0 prefers HUD brightness 8.",
                    query="Apply the daytime preference.",
                    gold_calls=(
                        V1GoldToolCallRecord(
                            name="carcontrol_HUD_set_brightness_level",
                            arguments={"level": 8},
                        ),
                    ),
                    target_state={"HUD": {"brightness": 8}},
                    target_state_sha256=canonical_json_sha256(
                        {"HUD": {"brightness": 8}}
                    ),
                )
            )
    return V1ScenarioArtifact(
        scenario_id="scenario-001",
        personas=personas,
        event_chains=tuple(chains),
        dialogue_turns=tuple(turns),
        final_quizzes=tuple(quizzes),
    )


def test_default_decisions_keep_paper_first_and_scale_screening() -> None:
    decisions = default_v1_reproduction_decisions(target_scenario_count=100)
    by_key = {decision.key: decision for decision in decisions}

    assert by_key["persona_group_size"].provenance == "paper_specified"
    assert by_key["chains_per_scenario"].value == {
        "background": 20,
        "vehicle": 10,
    }
    assert by_key["qa_output_schema"].provenance == "v1_inferred"
    assert by_key["candidate_persona_group_count"].value == 200
    assert by_key["candidate_persona_group_count"].provenance == "project_defined"


def test_complete_scenario_schema_enforces_v1_counts_and_references() -> None:
    artifact = _complete_scenario()

    assert len(artifact.personas) == 3
    assert len(artifact.event_chains) == 30
    assert len(artifact.final_quizzes) == 10

    payload = artifact.model_dump(mode="json")
    payload["personas"] = payload["personas"][:2]
    with pytest.raises(ValidationError, match="exactly 3 personas"):
        V1ScenarioArtifact.model_validate(payload)


def test_reference_profile_is_deterministic_and_tracks_public_shapes(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)

    first = build_v1_reference_profile(dataset, token_counter=_WordCounter())
    second = build_v1_reference_profile(dataset, token_counter=_WordCounter())

    assert first == second
    assert first.observed_contract["public_output_totals"] == {
        "scenario_count": 2,
        "query_count": 2,
        "history_turn_count": 4,
        "history_token_count_o200k_base": 44,
        "gold_tool_call_count": 2,
    }
    assert first.observed_contract["qa"]["reasoning_type_counts"] == {
        "conditional_constraint": 2
    }
    assert first.observed_contract["qa"]["target_module_counts"] == {"HUD": 2}


def test_reference_writer_hashes_match_serialized_contract(tmp_path: Path) -> None:
    profile = build_v1_reference_profile(
        _dataset(tmp_path / "source"),
        token_counter=_WordCounter(),
    )
    manifest = build_v1_reproduction_manifest(profile)
    paths = write_v1_reproduction_reference(
        tmp_path / "output",
        profile=profile,
        manifest=manifest,
    )

    stored_profile = json.loads(paths["reference_profile"].read_text())
    stored_schema = json.loads(paths["scenario_schema"].read_text())
    stored_manifest = json.loads(paths["manifest"].read_text())

    assert canonical_json_sha256(stored_profile) == manifest.reference_profile_sha256
    assert canonical_json_sha256(stored_schema) == manifest.scenario_schema_sha256
    assert stored_manifest["provenance_policy"] == [
        "paper_specified",
        "v1_inferred",
        "project_defined",
    ]
