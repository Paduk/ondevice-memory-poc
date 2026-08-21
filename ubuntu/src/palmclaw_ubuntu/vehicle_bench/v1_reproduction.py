"""Contracts and reference profiling for VehicleMemBench V1 reimplementation."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from palmclaw_ubuntu.tokens import TokenCounter
from palmclaw_ubuntu.vehicle_bench.dataset import (
    REASONING_TYPES,
    GoldToolCall,
    VehicleBenchmarkDataset,
)
from palmclaw_ubuntu.vehicle_bench.memory import parse_vehicle_history

V1_REPRODUCTION_SCHEMA_VERSION = "vehiclemembench-v1-reproduction-schema-v1"
V1_REFERENCE_PROFILE_VERSION = "vehiclemembench-v1-reference-profile-v1"
V1_REPRODUCTION_MANIFEST_VERSION = "vehiclemembench-v1-reproduction-manifest-v1"
V1_PAPER_REFERENCE = "VehicleMemBench arXiv:2603.23840v1"

ProvenanceKind = Literal[
    "paper_specified",
    "v1_inferred",
    "project_defined",
]
UncertaintyLevel = Literal["none", "low", "medium", "high"]
ChainKind = Literal["background", "vehicle"]
V1ScalarValue = str | int | float | bool

_HISTORY_LINE_SHAPE = (
    r"^(?:\[YYYY-MM-DD HH:MM\]\s+)+Speaker Name:\s?Dialogue text$"
)
_TIMESTAMP_PREFIX = re.compile(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V1SpecDecision(_StrictModel):
    """One traceable choice in the reconstructed generation contract."""

    key: str = Field(min_length=1)
    value: JsonValue
    provenance: ProvenanceKind
    source_refs: tuple[str, ...] = Field(min_length=1)
    uncertainty: UncertaintyLevel
    rationale: str = Field(min_length=1)


class V1NamedValue(_StrictModel):
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)


class V1PersonaRecord(_StrictModel):
    persona_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    name: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    basic_profile: tuple[V1NamedValue, ...] = Field(min_length=1)
    cultural_interests: tuple[V1NamedValue, ...] = Field(min_length=1)
    lifestyle_habits: tuple[V1NamedValue, ...] = Field(min_length=1)
    vehicle_preferences: tuple[V1NamedValue, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_attribute_keys(self) -> V1PersonaRecord:
        for field_name in (
            "basic_profile",
            "cultural_interests",
            "lifestyle_habits",
            "vehicle_preferences",
        ):
            values = getattr(self, field_name)
            keys = [item.key.strip().casefold() for item in values]
            if len(set(keys)) != len(keys):
                raise ValueError(f"{field_name} keys must be unique")
        return self


class V1ArgumentValue(_StrictModel):
    name: str = Field(min_length=1)
    value: V1ScalarValue


class V1PreferenceUpdate(_StrictModel):
    subject_id: str = Field(min_length=1)
    attribute_path: str = Field(min_length=1)
    previous_value: V1ScalarValue | None = None
    new_value: V1ScalarValue
    context_arguments: tuple[V1ArgumentValue, ...] = ()
    condition: str | None = None
    supersedes_event_id: str | None = None

    @model_validator(mode="after")
    def validate_context_arguments(self) -> V1PreferenceUpdate:
        names = [argument.name for argument in self.context_arguments]
        if len(set(names)) != len(names):
            raise ValueError("context argument names must be unique")
        return self

    def context_argument_map(self) -> dict[str, V1ScalarValue]:
        return {
            argument.name: argument.value for argument in self.context_arguments
        }


class V1EventRecord(_StrictModel):
    event_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    timestamp: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
    description: str = Field(min_length=1)
    participant_ids: tuple[str, ...] = Field(min_length=1)
    preference_updates: tuple[V1PreferenceUpdate, ...] = ()


class V1EventChainRecord(_StrictModel):
    chain_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    kind: ChainKind
    reasoning_type: str | None = None
    delayed_query_seed: str | None = None
    events: tuple[V1EventRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_kind(self) -> V1EventChainRecord:
        if self.kind == "vehicle":
            if self.reasoning_type not in REASONING_TYPES:
                raise ValueError("vehicle chain requires an official reasoning_type")
            if not self.delayed_query_seed or not self.delayed_query_seed.strip():
                raise ValueError("vehicle chain requires a delayed_query_seed")
            if not any(event.preference_updates for event in self.events):
                raise ValueError("vehicle chain requires a preference update")
        elif self.reasoning_type is not None or self.delayed_query_seed is not None:
            raise ValueError(
                "background chain cannot declare reasoning_type or delayed_query_seed"
            )
        return self


class V1DialogueTurnRecord(_StrictModel):
    turn_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    source_event_id: str = Field(min_length=1)
    timestamp: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
    speaker_id: str = Field(min_length=1)
    speaker_name: str = Field(min_length=1)
    text: str = Field(min_length=1)


class V1GoldToolCallRecord(_StrictModel):
    name: str = Field(pattern=r"^carcontrol_[A-Za-z0-9_]+$")
    arguments: dict[str, JsonValue]


class V1FinalQuizRecord(_StrictModel):
    quiz_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    source_chain_id: str = Field(min_length=1)
    reasoning_type: str
    gold_memory: str = Field(min_length=1)
    query: str = Field(min_length=1)
    gold_calls: tuple[V1GoldToolCallRecord, ...] = Field(min_length=1)
    target_state: dict[str, JsonValue] = Field(min_length=1)
    target_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_reasoning_type(self) -> V1FinalQuizRecord:
        if self.reasoning_type not in REASONING_TYPES:
            raise ValueError("quiz requires an official reasoning_type")
        if canonical_json_sha256(self.target_state) != self.target_state_sha256:
            raise ValueError("target_state_sha256 does not match target_state")
        return self


class V1ScenarioArtifact(_StrictModel):
    """Canonical complete scenario exchanged between V1 generation stages."""

    schema_version: Literal[
        "vehiclemembench-v1-reproduction-schema-v1"
    ] = V1_REPRODUCTION_SCHEMA_VERSION
    scenario_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    personas: tuple[V1PersonaRecord, ...]
    event_chains: tuple[V1EventChainRecord, ...]
    dialogue_turns: tuple[V1DialogueTurnRecord, ...] = Field(min_length=1)
    final_quizzes: tuple[V1FinalQuizRecord, ...]

    @model_validator(mode="after")
    def validate_complete_scenario(self) -> V1ScenarioArtifact:
        if len(self.personas) != 3:
            raise ValueError("a complete V1 scenario requires exactly 3 personas")
        background = [
            chain for chain in self.event_chains if chain.kind == "background"
        ]
        vehicle = [chain for chain in self.event_chains if chain.kind == "vehicle"]
        if len(background) != 20 or len(vehicle) != 10:
            raise ValueError(
                "a complete V1 scenario requires 20 background and 10 vehicle chains"
            )
        if len(self.final_quizzes) != 10:
            raise ValueError("a complete V1 scenario requires exactly 10 final quizzes")

        persona_ids = {persona.persona_id for persona in self.personas}
        if len(persona_ids) != len(self.personas):
            raise ValueError("persona_id values must be unique")
        event_ids = {
            event.event_id for chain in self.event_chains for event in chain.events
        }
        if len(event_ids) != sum(len(chain.events) for chain in self.event_chains):
            raise ValueError("event_id values must be unique")
        chain_by_id = {chain.chain_id: chain for chain in self.event_chains}
        if len(chain_by_id) != len(self.event_chains):
            raise ValueError("chain_id values must be unique")
        for turn in self.dialogue_turns:
            if turn.source_event_id not in event_ids:
                raise ValueError("dialogue turn references an unknown event")
            if turn.speaker_id not in persona_ids:
                raise ValueError("dialogue turn references an unknown persona")
        quiz_chain_ids: set[str] = set()
        for quiz in self.final_quizzes:
            chain = chain_by_id.get(quiz.source_chain_id)
            if chain is None or chain.kind != "vehicle":
                raise ValueError("final quiz must reference a vehicle chain")
            if quiz.reasoning_type != chain.reasoning_type:
                raise ValueError("quiz and source chain reasoning_type must match")
            quiz_chain_ids.add(quiz.source_chain_id)
        if quiz_chain_ids != {chain.chain_id for chain in vehicle}:
            raise ValueError("every vehicle chain requires exactly one final quiz")
        return self


class V1ReferenceProfile(_StrictModel):
    profile_version: Literal[
        "vehiclemembench-v1-reference-profile-v1"
    ] = V1_REFERENCE_PROFILE_VERSION
    source_manifest: dict[str, JsonValue]
    reproduction_decisions: tuple[V1SpecDecision, ...]
    observed_contract: dict[str, JsonValue]


class V1ReproductionManifest(_StrictModel):
    manifest_version: Literal[
        "vehiclemembench-v1-reproduction-manifest-v1"
    ] = V1_REPRODUCTION_MANIFEST_VERSION
    scenario_schema_version: str
    reference_profile_version: str
    reference_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_scenario_count: int = Field(gt=0)
    candidate_persona_group_count: int = Field(gt=0)
    provenance_policy: tuple[ProvenanceKind, ...]
    decisions: tuple[V1SpecDecision, ...]


def default_v1_reproduction_decisions(
    *, target_scenario_count: int = 100
) -> tuple[V1SpecDecision, ...]:
    if target_scenario_count < 1:
        raise ValueError("target_scenario_count must be positive")
    candidate_count = target_scenario_count * 2
    return (
        V1SpecDecision(
            key="target_scenario_count",
            value=target_scenario_count,
            provenance="project_defined",
            source_refs=("PalmClaw V1 reproduction/V2 expansion objective",),
            uncertainty="none",
            rationale="The project requires 100 new V1-compatible scenarios.",
        ),
        V1SpecDecision(
            key="candidate_persona_group_count",
            value=candidate_count,
            provenance="project_defined",
            source_refs=(f"{V1_PAPER_REFERENCE} §2.1 Stage 1",),
            uncertainty="low",
            rationale=(
                "Scale the paper's 100-candidate/50-retained screening ratio "
                "to the 100-scenario target."
            ),
        ),
        V1SpecDecision(
            key="persona_group_size",
            value=3,
            provenance="paper_specified",
            source_refs=(f"{V1_PAPER_REFERENCE} §2.1 Stage 1",),
            uncertainty="none",
            rationale="The paper defines groups of three recurring vehicle occupants.",
        ),
        V1SpecDecision(
            key="chains_per_scenario",
            value={"background": 20, "vehicle": 10},
            provenance="paper_specified",
            source_refs=(f"{V1_PAPER_REFERENCE} Appendix B",),
            uncertainty="none",
            rationale=(
                "The released benchmark uses 20 distractor and 10 executable "
                "chains."
            ),
        ),
        V1SpecDecision(
            key="final_quizzes_per_scenario",
            value=10,
            provenance="paper_specified",
            source_refs=(f"{V1_PAPER_REFERENCE} Appendix B",),
            uncertainty="none",
            rationale="Each executable vehicle chain terminates in one delayed query.",
        ),
        V1SpecDecision(
            key="reasoning_types",
            value=sorted(REASONING_TYPES),
            provenance="paper_specified",
            source_refs=(f"{V1_PAPER_REFERENCE} §2.1 Stage 2 and Appendix G",),
            uncertainty="none",
            rationale=(
                "These are the five preference-evolution categories in the paper."
            ),
        ),
        V1SpecDecision(
            key="paper_generation_models",
            value={
                "profile_enrichment": {
                    "model": "Gemini-3-Pro-Preview",
                    "temperature": 0.7,
                },
                "event_chain_generation": {
                    "model": "Gemini-3-Pro-Preview",
                    "temperature": 0.8,
                },
                "dialogue_generation": {"model": "GPT-4.1", "temperature": 0.7},
                "answer_generation": {
                    "model": "Gemini-3-Pro-Preview",
                    "temperature": 0.0,
                },
            },
            provenance="paper_specified",
            source_refs=(f"{V1_PAPER_REFERENCE} Appendix E Table 7",),
            uncertainty="none",
            rationale=(
                "Record the exact stage-level model settings reported by the paper."
            ),
        ),
        V1SpecDecision(
            key="project_generation_model",
            value={
                "model": "gpt-5.6-terra",
                "reasoning_effort": "medium",
                "temperature": None,
                "applies_to": [
                    "profile_enrichment",
                    "event_chain_generation",
                    "dialogue_generation",
                    "answer_generation",
                ],
            },
            provenance="project_defined",
            source_refs=("PalmClaw model decision, 2026-08-20",),
            uncertainty="none",
            rationale=(
                "Use one available high-quality model while preserving the paper's "
                "original model settings separately for transparent comparison."
            ),
        ),
        V1SpecDecision(
            key="event_chain_generation_contract",
            value={
                "requires_multi_hop": True,
                "vehicle_chain_ends_in_delayed_executable_query": True,
                "events_span_weeks_or_months": True,
                "vehicle_attribute_must_match_device_range_table": True,
            },
            provenance="paper_specified",
            source_refs=(f"{V1_PAPER_REFERENCE} Appendix G Figure 5",),
            uncertainty="none",
            rationale="Use the shortened prompt's published semantic requirements.",
        ),
        V1SpecDecision(
            key="dialogue_generation_contract",
            value={
                "human_to_human_only": True,
                "vehicle_preferences_are_natural_asides": True,
                "speaker_identity_matches_event": True,
                "chronology_is_preserved": True,
                "minimum_background_dialogue_lines_per_event": 40,
            },
            provenance="paper_specified",
            source_refs=(f"{V1_PAPER_REFERENCE} Appendix I Figure 6",),
            uncertainty="none",
            rationale=(
                "Use the published dialogue constraints rather than infer new ones."
            ),
        ),
        V1SpecDecision(
            key="manual_quality_control",
            value={
                "annotators": 3,
                "acceptance": "unanimous_agreement",
                "checks": [
                    "persona_profile_validation",
                    "event_dialogue_alignment",
                    "query_relevance",
                    "answer_correctness",
                ],
            },
            provenance="paper_specified",
            source_refs=(f"{V1_PAPER_REFERENCE} Appendix F",),
            uncertainty="none",
            rationale=(
                "Preserve the paper's human verification protocol as the gold "
                "standard."
            ),
        ),
        V1SpecDecision(
            key="qa_output_schema",
            value={
                "root": "related_to_vehicle_preference",
                "fields": [
                    "gold_memory",
                    "reasoning_type",
                    "query",
                    "new_answer",
                ],
            },
            provenance="v1_inferred",
            source_refs=("Official V1 benchmark/qa_data/qa_1.json ... qa_50.json",),
            uncertainty="none",
            rationale="All 50 released QA files use this exact public output contract.",
        ),
        V1SpecDecision(
            key="persona_intermediate_schema",
            value=[
                "persona_id",
                "name",
                "aliases",
                "basic_profile",
                "cultural_interests",
                "lifestyle_habits",
                "vehicle_preferences",
            ],
            provenance="project_defined",
            source_refs=(f"{V1_PAPER_REFERENCE} §2.1 Stage 1",),
            uncertainty="medium",
            rationale=(
                "The paper names the attribute families but does not release its "
                "intermediate persona JSON schema."
            ),
        ),
        V1SpecDecision(
            key="event_intermediate_schema",
            value=[
                "event_id",
                "timestamp",
                "description",
                "participant_ids",
                "preference_updates",
            ],
            provenance="project_defined",
            source_refs=(f"{V1_PAPER_REFERENCE} §2.1 Stages 2–3",),
            uncertainty="high",
            rationale=(
                "The fields encode the published event semantics, but their exact "
                "original JSON representation is unavailable."
            ),
        ),
        V1SpecDecision(
            key="simulator_target_state_artifact",
            value="nested JSON state plus canonical hash",
            provenance="project_defined",
            source_refs=(f"{V1_PAPER_REFERENCE} §2.1 Stage 5 and §2.3",),
            uncertainty="low",
            rationale=(
                "The paper requires executable target-state validation but does not "
                "publish a generation-stage artifact schema."
            ),
        ),
        V1SpecDecision(
            key="event_dialogue_realizability",
            value={
                "minimum_participants_per_event": 2,
                "reserved_group_subject_id": "shared-vehicle",
            },
            provenance="project_defined",
            source_refs=(
                f"{V1_PAPER_REFERENCE} §2.1 Stage 4 and Appendix I Figure 6",
            ),
            uncertainty="low",
            rationale=(
                "Human-to-human dialogue cannot faithfully realize a one-person "
                "event; the reserved subject represents an explicit shared rule."
            ),
        ),
        V1SpecDecision(
            key="dialogue_generation_execution",
            value={
                "api_granularity": "one call per structured event",
                "background_turn_target": 40,
                "vehicle_turn_target": 26,
                "context": [
                    "current_event",
                    "same_chain_prior_events",
                    "recent_global_events_8",
                    "previous_preference_state",
                ],
                "timestamp_policy": "inherit source event timestamp",
            },
            provenance="project_defined",
            source_refs=(
                f"{V1_PAPER_REFERENCE} §2.1 Stage 4 and Appendix I Figure 6",
                "Official V1 history length distribution",
            ),
            uncertainty="medium",
            rationale=(
                "The paper specifies causal inputs and >=40 background lines but "
                "does not release batching, vehicle-event length, or timestamp code."
            ),
        ),
        V1SpecDecision(
            key="automatic_generation_validation",
            value={
                "tool_json_schema": True,
                "vehicleworld_execution_success": True,
                "quiz_must_change_state": True,
                "resumable_event_and_chain_checkpoints": True,
            },
            provenance="project_defined",
            source_refs=(
                f"{V1_PAPER_REFERENCE} §2.1 Stage 5 and Appendix F",
            ),
            uncertainty="low",
            rationale=(
                "Operationalize the paper's execution verification before Human "
                "Review while rejecting type-valid but simulator-invalid values."
            ),
        ),
        V1SpecDecision(
            key="history_line_schema",
            value=_HISTORY_LINE_SHAPE,
            provenance="v1_inferred",
            source_refs=(
                "Official V1 benchmark/history/history_1.txt ... history_50.txt",
            ),
            uncertainty="none",
            rationale="All released non-empty history lines follow this public format.",
        ),
        V1SpecDecision(
            key="unpublished_generation_details",
            value=[
                "full prompts",
                "intermediate event-chain JSON",
                "retry policy",
                "automatic validator details",
                "manual correction representation",
            ],
            provenance="project_defined",
            source_refs=(f"{V1_PAPER_REFERENCE} Appendix I (shortened prompts)",),
            uncertainty="high",
            rationale=(
                "These details are absent from the paper and released repository; "
                "later implementations must version and disclose conservative choices."
            ),
        ),
    )


def build_v1_reference_profile(
    dataset: VehicleBenchmarkDataset,
    *,
    token_counter: TokenCounter | None = None,
    target_scenario_count: int = 100,
) -> V1ReferenceProfile:
    """Extract deterministic V1 output constraints without using task scores."""

    counter = token_counter or TokenCounter("o200k_base")
    history_turn_counts: list[int] = []
    history_day_counts: list[int] = []
    history_speaker_counts: list[int] = []
    history_token_counts: list[int] = []
    history_character_counts: list[int] = []
    timestamps_per_line: list[int] = []
    chronology_violations = 0
    speaker_count_exceptions: list[JsonValue] = []
    timestamp_prefix_exceptions: list[JsonValue] = []

    queries_per_scenario: list[int] = []
    query_token_counts: list[int] = []
    gold_memory_token_counts: list[int] = []
    gold_memory_line_counts: list[int] = []
    calls_per_query: list[int] = []
    reasoning_type_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    module_counts: Counter[str] = Counter()
    qa_root_key_shapes: Counter[tuple[str, ...]] = Counter()
    qa_item_key_shapes: Counter[tuple[str, ...]] = Counter()

    for scenario in dataset.scenarios:
        raw_history = scenario.history_path.read_text(encoding="utf-8")
        entries = parse_vehicle_history(scenario.history_path)
        history_turn_counts.append(len(entries))
        history_day_counts.append(len({entry.date for entry in entries}))
        history_speaker_counts.append(len({entry.speaker for entry in entries}))
        history_token_counts.append(counter.count(raw_history))
        history_character_counts.append(len(raw_history))
        speakers = sorted({entry.speaker for entry in entries})
        if len(speakers) != 3:
            speaker_count_exceptions.append(
                {
                    "scenario_index": scenario.index,
                    "speaker_count": len(speakers),
                    "speaker_labels": speakers,
                }
            )
        for entry in entries:
            prefix_count = len(_TIMESTAMP_PREFIX.findall(entry.raw))
            timestamps_per_line.append(prefix_count)
            if prefix_count != 1:
                timestamp_prefix_exceptions.append(
                    {
                        "scenario_index": scenario.index,
                        "line_number": entry.line_number,
                        "timestamp_prefix_count": prefix_count,
                    }
                )
        chronology_violations += sum(
            current.timestamp < previous.timestamp
            for previous, current in zip(entries, entries[1:], strict=False)
        )

        raw_qa = _read_json_object(scenario.qa_path)
        qa_root_key_shapes[tuple(sorted(raw_qa))] += 1
        raw_items = raw_qa.get("related_to_vehicle_preference", [])
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, dict):
                    qa_item_key_shapes[tuple(sorted(item))] += 1

        queries_per_scenario.append(len(scenario.tasks))
        for task in scenario.tasks:
            query_token_counts.append(counter.count(task.query))
            gold_memory_token_counts.append(counter.count(task.gold_memory))
            gold_memory_line_counts.append(
                len([line for line in task.gold_memory.splitlines() if line.strip()])
            )
            calls_per_query.append(len(task.gold_calls))
            reasoning_type_counts[task.reasoning_type] += 1
            for call in task.gold_calls:
                tool_counts[call.name] += 1
                module_counts[_tool_module(call)] += 1

    observed_contract: dict[str, JsonValue] = {
        "history": {
            "line_shape": _HISTORY_LINE_SHAPE,
            "turns_per_scenario": _numeric_summary(history_turn_counts),
            "days_per_scenario": _numeric_summary(history_day_counts),
            "speakers_per_scenario": _numeric_summary(history_speaker_counts),
            "tokens_per_scenario_o200k_base": _numeric_summary(
                history_token_counts
            ),
            "characters_per_scenario": _numeric_summary(history_character_counts),
            "timestamp_prefixes_per_line": _numeric_summary(timestamps_per_line),
            "chronology_violations": chronology_violations,
            "speaker_count_exceptions": speaker_count_exceptions,
            "timestamp_prefix_exceptions": timestamp_prefix_exceptions,
        },
        "qa": {
            "root_key_shapes": _shape_counts(qa_root_key_shapes),
            "item_key_shapes": _shape_counts(qa_item_key_shapes),
            "queries_per_scenario": _numeric_summary(queries_per_scenario),
            "query_tokens_o200k_base": _numeric_summary(query_token_counts),
            "gold_memory_tokens_o200k_base": _numeric_summary(
                gold_memory_token_counts
            ),
            "gold_memory_lines": _numeric_summary(gold_memory_line_counts),
            "gold_calls_per_query": _numeric_summary(calls_per_query),
            "reasoning_type_counts": dict(sorted(reasoning_type_counts.items())),
            "tool_call_counts": dict(sorted(tool_counts.items())),
            "target_module_counts": dict(sorted(module_counts.items())),
        },
        "public_output_totals": {
            "scenario_count": len(dataset.scenarios),
            "query_count": sum(queries_per_scenario),
            "history_turn_count": sum(history_turn_counts),
            "history_token_count_o200k_base": sum(history_token_counts),
            "gold_tool_call_count": sum(calls_per_query),
        },
        "paper_stat_parity": {
            "history_turns_mean": _parity_value(
                statistics.fmean(history_turn_counts), 2690.0
            ),
            "history_tokens_mean_o200k_base": _parity_value(
                statistics.fmean(history_token_counts), 92819.0
            ),
            "query_tokens_mean_o200k_base": _parity_value(
                statistics.fmean(query_token_counts), 37.63
            ),
        },
    }
    return V1ReferenceProfile(
        source_manifest=dataset.manifest.as_dict(),
        reproduction_decisions=default_v1_reproduction_decisions(
            target_scenario_count=target_scenario_count
        ),
        observed_contract=observed_contract,
    )


def build_v1_reproduction_manifest(
    profile: V1ReferenceProfile,
    *,
    target_scenario_count: int = 100,
) -> V1ReproductionManifest:
    decisions = default_v1_reproduction_decisions(
        target_scenario_count=target_scenario_count
    )
    schema = V1ScenarioArtifact.model_json_schema()
    return V1ReproductionManifest(
        scenario_schema_version=V1_REPRODUCTION_SCHEMA_VERSION,
        reference_profile_version=profile.profile_version,
        reference_profile_sha256=canonical_json_sha256(
            profile.model_dump(mode="json")
        ),
        scenario_schema_sha256=canonical_json_sha256(schema),
        target_scenario_count=target_scenario_count,
        candidate_persona_group_count=target_scenario_count * 2,
        provenance_policy=(
            "paper_specified",
            "v1_inferred",
            "project_defined",
        ),
        decisions=decisions,
    )


def write_v1_reproduction_reference(
    output_dir: Path | str,
    *,
    profile: V1ReferenceProfile,
    manifest: V1ReproductionManifest,
) -> Mapping[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, tuple[Path, Any]] = {
        "reference_profile": (
            root / "reference_profile.json",
            profile.model_dump(mode="json"),
        ),
        "scenario_schema": (
            root / "scenario.schema.json",
            V1ScenarioArtifact.model_json_schema(),
        ),
        "manifest": (
            root / "manifest.json",
            manifest.model_dump(mode="json"),
        ),
    }
    for _, (path, payload) in artifacts.items():
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {name: path for name, (path, _) in artifacts.items()}


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _numeric_summary(values: Sequence[int]) -> dict[str, JsonValue]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
    }


def _shape_counts(counts: Counter[tuple[str, ...]]) -> list[JsonValue]:
    return [
        {"keys": list(keys), "count": count}
        for keys, count in sorted(counts.items())
    ]


def _parity_value(observed: float, paper_reported: float) -> dict[str, JsonValue]:
    return {
        "observed": round(observed, 3),
        "paper_reported": paper_reported,
        "absolute_delta": round(abs(observed - paper_reported), 3),
    }


def _tool_module(call: GoldToolCall) -> str:
    suffix = call.name.removeprefix("carcontrol_")
    return suffix.split("_", 1)[0]


def _read_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw
