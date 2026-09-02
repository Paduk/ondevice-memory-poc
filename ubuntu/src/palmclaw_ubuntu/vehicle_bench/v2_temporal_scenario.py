"""Structured Temporal/Current extension for VehicleMemBench V2 Hybrid."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_EVENT_CHAIN_INSTRUCTIONS,
    V1GeneratedEventChains,
    V1VehicleAttribute,
    interleave_event_chains,
    render_event_chain_generation_input,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1DialogueTurnRecord,
    V1EventChainRecord,
    V1EventRecord,
    V1PersonaRecord,
    V1PreferenceUpdate,
    canonical_json_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2GeneratedHybridEventAlignment,
    V2HybridEventAlignmentPayload,
    hybrid_memory_identity,
)

V2_TEMPORAL_EVENT_PROMPT_VERSION_V1 = (
    "vehiclemembench-v2-temporal-current-event-chain-v1"
)
V2_TEMPORAL_EVENT_PROMPT_VERSION_V2 = (
    "vehiclemembench-v2-temporal-current-event-chain-v2"
)
V2_TEMPORAL_EVENT_PROMPT_VERSION = (
    "vehiclemembench-v2-temporal-current-event-chain-v3"
)
V2_TEMPORAL_EVENT_PROMPT_VERSIONS = frozenset(
    {
        V2_TEMPORAL_EVENT_PROMPT_VERSION_V1,
        V2_TEMPORAL_EVENT_PROMPT_VERSION_V2,
        V2_TEMPORAL_EVENT_PROMPT_VERSION,
    }
)
V2_TEMPORAL_PLAN_SCHEMA_VERSION = "vehiclemembench-v2-temporal-plan-v1"
V2_TEMPORAL_MAX_UPDATE_CHECKPOINTS = 28
V2_TEMPORAL_DIALOGUE_PROMPT_VERSION = (
    "vehiclemembench-v2-temporal-dialogue-self-contained-v1"
)
V2_TEMPORAL_ALIGNMENT_PROMPT_VERSION = (
    "vehiclemembench-v2-temporal-alignment-self-contained-v1"
)

V2_TEMPORAL_DIALOGUE_INSTRUCTIONS = """
The input may contain required_exact_anchor_turns. Copy every anchor's speaker_id
and anchor_text exactly once as one dialogue turn. Do not paraphrase, split,
merge, or omit an anchor. Build a natural conversation around the frozen anchor,
with earlier proposals or discussion remaining non-committal. Do not repeat the
complete anchor fact in unrelated turns. When no anchor is supplied, follow the
base dialogue rules unchanged. When multiple anchors are supplied, include them
in ascending source_update_index order.
""".strip()

V2_TEMPORAL_ALIGNMENT_INSTRUCTIONS = """
For this Temporal dataset, each alignment must be solvable from current approved
memory and the selected current turn alone. Do not use an earlier event-prefix
proposal to fill information missing from the selected turn. For a temporary
override, wait for the first self-contained subject utterance containing the
value and temporary scope. For end_temporary, the approved memory already holds
the durable baseline and active override, so select the earliest explicit scope-
ending turn. Do not delay to a later repetition of the restored baseline.
""".strip()

TemporalFocus = Literal[
    "current_durable",
    "temporary_restore",
    "conditional",
    "correction_noop",
]
TemporalAction = Literal[
    "durable_upsert",
    "current_upsert",
    "temporary_override",
    "end_temporary",
    "conditional_upsert",
]
TemporalSplit = Literal["train", "validation", "test"]

_CURRENT_SCOPE = re.compile(
    r"\b(?:current|currently|observed|right now|at present)\b", re.IGNORECASE
)
_TEMPORARY_SCOPE = re.compile(
    r"\b(?:temporary|temporarily|until|for now|for the next|during recovery|"
    r"while recover(?:ing|y))\b",
    re.IGNORECASE,
)
_HARD_NEGATIVE_CUE = re.compile(
    r"\b(?:current|currently|temporary|temporarily|until|for now)\b",
    re.IGNORECASE,
)
_DIALOGUE_TEMPORARY_CUE = re.compile(
    r"\b(?:temporar(?:y|ily)|for now|until|for the next|for the duration|"
    r"for (?:this|that) [a-z][a-z'-]*|during\b|while\b)",
    re.IGNORECASE,
)
_DIALOGUE_END_CUE = re.compile(
    r"\b(?:no longer|has ended|is ended|ends|ended|(?:is|are|was|were) over|"
    r"finished|back to normal|"
    r"back to the baseline|fully recovered|recovered)\b",
    re.IGNORECASE,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V2TemporalScenarioProfile(_StrictModel):
    scenario_index: int = Field(ge=1, le=20)
    scenario_id: str = Field(pattern=r"^temporal-s\d{2}$")
    candidate_group: int = Field(ge=101, le=120)
    split: TemporalSplit
    focus: TemporalFocus
    required_actions: tuple[TemporalAction, ...]
    focus_minimum_count: int = Field(ge=1)
    hard_negative_minimum_count: int = Field(ge=1)


class V2TemporalTransition(_StrictModel):
    source_event_id: str = Field(min_length=1)
    source_update_index: int = Field(ge=0)
    temporal_action: TemporalAction
    temporal_cue: str = ""
    baseline_event_id: str | None = None
    baseline_source_update_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_action_shape(self) -> V2TemporalTransition:
        cue = self.temporal_cue.strip()
        has_baseline = self.baseline_event_id is not None
        has_baseline_index = self.baseline_source_update_index is not None
        if has_baseline != has_baseline_index:
            raise ValueError("baseline event and update index must appear together")
        if self.temporal_action in {"temporary_override", "end_temporary"}:
            if not cue or not has_baseline:
                raise ValueError(
                    f"{self.temporal_action} requires cue and baseline reference"
                )
        elif cue or has_baseline:
            raise ValueError(
                f"{self.temporal_action} cannot carry cue or baseline reference"
            )
        return self


class V2TemporalEventChainPayload(_StrictModel):
    background_chains: tuple[V1EventChainRecord, ...] = Field(
        min_length=20, max_length=20
    )
    vehicle_chains: tuple[V1EventChainRecord, ...] = Field(
        min_length=10, max_length=10
    )
    temporal_transitions: tuple[V2TemporalTransition, ...] = Field(min_length=1)


class V2GeneratedTemporalEventChains(_StrictModel):
    scenario_candidate_id: str = Field(min_length=1)
    profile: V2TemporalScenarioProfile
    planned_reasoning_types: tuple[str, ...] = Field(min_length=10, max_length=10)
    payload: V2TemporalEventChainPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = V2_TEMPORAL_EVENT_PROMPT_VERSION
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vehicle_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]
    normalization_notes: tuple[str, ...] = ()

    def as_v1_event_chains(self) -> V1GeneratedEventChains:
        return V1GeneratedEventChains(
            scenario_candidate_id=self.scenario_candidate_id,
            planned_reasoning_types=self.planned_reasoning_types,
            payload={
                "background_chains": self.payload.background_chains,
                "vehicle_chains": self.payload.vehicle_chains,
            },
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            input_sha256=self.input_sha256,
            vehicle_catalog_sha256=self.vehicle_catalog_sha256,
            response_id=self.response_id,
            usage=self.usage,
        )


def build_temporal_scenario_profiles() -> tuple[V2TemporalScenarioProfile, ...]:
    """Return the frozen 15/1/4 scenario-level split and focus schedule."""

    focus_by_index: dict[int, TemporalFocus] = {}
    groups: tuple[tuple[TemporalFocus, tuple[int, ...]], ...] = (
        ("current_durable", (1, 2, 3, 4, 17)),
        ("temporary_restore", (5, 6, 7, 8, 18)),
        ("conditional", (9, 10, 11, 12, 19)),
        ("correction_noop", (13, 14, 15, 16, 20)),
    )
    for focus, indexes in groups:
        for index in indexes:
            focus_by_index[index] = focus
    actions: tuple[TemporalAction, ...] = (
        "durable_upsert",
        "current_upsert",
        "temporary_override",
        "end_temporary",
        "conditional_upsert",
    )
    profiles = []
    for index in range(1, 21):
        split: TemporalSplit
        if index <= 15:
            split = "train"
        elif index == 16:
            split = "validation"
        else:
            split = "test"
        profiles.append(
            V2TemporalScenarioProfile(
                scenario_index=index,
                scenario_id=f"temporal-s{index:02d}",
                candidate_group=100 + index,
                split=split,
                focus=focus_by_index[index],
                required_actions=actions,
                focus_minimum_count=2,
                hard_negative_minimum_count=2,
            )
        )
    return tuple(profiles)


def temporal_scenario_profile(index: int) -> V2TemporalScenarioProfile:
    if not 1 <= index <= 20:
        raise ValueError("Temporal scenario index must be in [1, 20]")
    return build_temporal_scenario_profiles()[index - 1]


def build_temporal_dataset_plan() -> dict[str, Any]:
    profiles = build_temporal_scenario_profiles()
    return {
        "schema_version": "vehiclemembench-v2-temporal-dataset-plan-v1",
        "scenario_count": len(profiles),
        "split_counts": dict(sorted(Counter(item.split for item in profiles).items())),
        "focus_counts": dict(sorted(Counter(item.focus for item in profiles).items())),
        "models": {
            "persona_event_dialogue_alignment": "gpt-5.6-terra",
            "turn_and_final_quiz": "gpt-5.6-luna",
            "exception_review": "gpt-5.6-sol",
        },
        "quiz_counts_per_scenario": {"turn": 30, "final": 10},
        "memory_application": "deterministic_temporal_patch",
        "profiles": [item.model_dump(mode="json") for item in profiles],
    }


V2_TEMPORAL_EVENT_INSTRUCTIONS = f"""
{V1_EVENT_CHAIN_INSTRUCTIONS}

VehicleMemBench V2 Temporal/Current extension:
- For every background chain, reasoning_type and delayed_query_seed must both
  be null. Only vehicle chains may populate those fields.
- Return temporal_transitions in addition to the 20 background and 10 vehicle
  chains. Every preference_update must have exactly one transition keyed by its
  source event ID and zero-based update index.
- Return at most {V2_TEMPORAL_MAX_UPDATE_CHECKPOINTS} preference_updates and
  temporal_transitions in total. This reserves two of the 30 Turn Quiz slots
  for delayed and composite coverage.
- Every vehicle chain starts with a durable_upsert whose condition is null and
  which does not supersede another event. Keep that durable baseline active.
- Every scenario must contain current_upsert, temporary_override,
  end_temporary, and conditional_upsert as well as durable_upsert.
- current_upsert records an explicitly observed current vehicle setting. Its
  condition must clearly say current/currently/observed/right now/at present,
  and its memory identity must remain distinct from the durable baseline.
- temporary_override must be a separate update with an explicit temporary
  condition and an exact temporal_cue copied from its event description. It
  references, but must not supersede, the durable baseline.
- end_temporary must explicitly end a previous temporary_override. Its update
  supersedes the temporary event, restores the referenced durable baseline's
  subject/attribute/selectors/value/condition, and gives an exact ending cue
  copied from the event description.
- conditional_upsert has a reusable non-temporary condition.
- A durable correction preserves the durable identity, declares previous_value
  and supersedes_event_id, and changes new_value.
- Include vehicle-chain events that mention a current or temporary setting but
  contain no preference_update because no durable/current change is confirmed.
  These are hard-negative NO_OP turns.
- Temporal/current facts must be naturally motivated in descriptions and later
  dialogue. Final delayed queries use the latest valid state, while intermediate
  quiz generation will test state immediately after transitions.
- temporal_cue and baseline fields are empty/null for actions other than
  temporary_override and end_temporary.
- Follow the supplied scenario_profile. All action families occur in every
  scenario; repeat its focus family at least focus_minimum_count times.
""".strip()


class OpenAIV2TemporalEventChainGenerationModel:
    """Generate Stage 2 chains and an auditable Temporal sidecar together."""

    def __init__(
        self,
        model_id: str,
        *,
        timeout_seconds: float,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int = 32_768,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Temporal event generation model ID is required")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def generate(
        self,
        personas: Sequence[V1PersonaRecord],
        *,
        scenario_candidate_id: str,
        reasoning_types: Sequence[str],
        vehicle_attributes: Sequence[V1VehicleAttribute],
        profile: V2TemporalScenarioProfile,
    ) -> V2GeneratedTemporalEventChains:
        base_input = json.loads(
            render_event_chain_generation_input(
                personas,
                scenario_candidate_id=scenario_candidate_id,
                reasoning_types=reasoning_types,
                vehicle_attributes=vehicle_attributes,
            )
        )
        base_input["scenario_profile"] = profile.model_dump(mode="json")
        provider_input = json.dumps(
            base_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": V2_TEMPORAL_EVENT_INSTRUCTIONS,
            "input": provider_input,
            "text_format": V2TemporalEventChainPayload,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        response = self._client.responses.parse(**request)
        status = getattr(response, "status", None)
        if status in {"failed", "incomplete", "cancelled"}:
            raise RuntimeError(
                f"Temporal event generation provider returned status {status}"
            )
        payload = V2TemporalEventChainPayload.model_validate(response.output_parsed)
        catalog_payload = [item.model_dump(mode="json") for item in vehicle_attributes]
        return V2GeneratedTemporalEventChains(
            scenario_candidate_id=scenario_candidate_id,
            profile=profile,
            planned_reasoning_types=tuple(reasoning_types),
            payload=payload,
            model_id=self.model_id,
            input_sha256=canonical_json_sha256(base_input),
            vehicle_catalog_sha256=canonical_json_sha256(catalog_payload),
            response_id=getattr(response, "id", None),
            usage=_response_usage(response),
        )


def validate_temporal_event_payload(
    payload: V2TemporalEventChainPayload,
    *,
    profile: V2TemporalScenarioProfile,
) -> dict[str, Any]:
    """Validate exact coverage and lossless baseline/override relationships."""

    all_chains = (*payload.background_chains, *payload.vehicle_chains)
    events = {
        event.event_id: event for chain in all_chains for event in chain.events
    }
    if len(events) != sum(len(chain.events) for chain in all_chains):
        raise ValueError("Temporal payload contains duplicate event IDs")
    update_keys = {
        (event.event_id, index)
        for chain in payload.vehicle_chains
        for event in chain.events
        for index, _ in enumerate(event.preference_updates)
    }
    annotations = {
        (item.source_event_id, item.source_update_index): item
        for item in payload.temporal_transitions
    }
    if len(annotations) != len(payload.temporal_transitions):
        raise ValueError("Temporal transition keys must be unique")
    if len(annotations) > V2_TEMPORAL_MAX_UPDATE_CHECKPOINTS:
        raise ValueError(
            "Temporal scenario has too many update checkpoints: "
            f"{len(annotations)} > {V2_TEMPORAL_MAX_UPDATE_CHECKPOINTS}"
        )
    if set(annotations) != update_keys:
        missing = sorted(update_keys - set(annotations))
        extra = sorted(set(annotations) - update_keys)
        raise ValueError(
            "Temporal annotations do not cover updates: "
            f"missing={missing}, extra={extra}"
        )

    action_counts = Counter(
        item.temporal_action for item in payload.temporal_transitions
    )
    for action in profile.required_actions:
        if action_counts[action] < 1:
            raise ValueError(f"Temporal scenario is missing required action: {action}")
    focus_count = _focus_count(profile.focus, payload.temporal_transitions, events)
    if focus_count < profile.focus_minimum_count:
        raise ValueError(
            f"Temporal focus {profile.focus} has {focus_count}, requires "
            f"{profile.focus_minimum_count}"
        )

    event_order = {
        event.event_id: event.timestamp
        for event in sorted(events.values(), key=lambda item: item.timestamp)
    }
    first_update_keys = set()
    for chain in payload.vehicle_chains:
        first = next(
            (
                (event.event_id, index)
                for event in chain.events
                for index, _ in enumerate(event.preference_updates)
            ),
            None,
        )
        if first is None or annotations[first].temporal_action != "durable_upsert":
            raise ValueError(
                f"Temporal vehicle chain {chain.chain_id} must start durable"
            )
        first_update_keys.add(first)

    for key, annotation in annotations.items():
        event = events[key[0]]
        update = event.preference_updates[key[1]]
        action = annotation.temporal_action
        if action == "temporary_override" and not _DIALOGUE_TEMPORARY_CUE.search(
            event.description
        ):
            raise ValueError(
                f"Temporary meaning is absent from {event.event_id}"
            )
        if action == "end_temporary" and not _DIALOGUE_END_CUE.search(
            event.description
        ):
            raise ValueError(
                f"Temporary ending meaning is absent from {event.event_id}"
            )
        if action == "durable_upsert":
            if key in first_update_keys and (
                update.condition is not None
                or update.supersedes_event_id is not None
            ):
                raise ValueError("Initial durable baseline must be unconditional")
        elif action == "current_upsert":
            if not update.condition or not _CURRENT_SCOPE.search(update.condition):
                raise ValueError("current_upsert requires explicit current scope")
        elif action == "conditional_upsert":
            if not update.condition or _TEMPORARY_SCOPE.search(update.condition):
                raise ValueError(
                    "conditional_upsert requires a reusable non-temporary condition"
                )
        elif action == "temporary_override":
            if not update.condition or not _DIALOGUE_TEMPORARY_CUE.search(
                update.condition
            ):
                raise ValueError("temporary_override requires temporary condition")
            baseline = _referenced_update(annotation, events)
            _require_same_setting(update, baseline, require_value=False)
            if hybrid_memory_identity(update) == hybrid_memory_identity(baseline):
                raise ValueError("Temporary override must preserve a separate baseline")
            if update.supersedes_event_id is not None:
                raise ValueError("Temporary override cannot supersede its baseline")
        elif action == "end_temporary":
            baseline = _referenced_update(annotation, events)
            _require_same_setting(update, baseline, require_value=True)
            if update.condition != baseline.condition:
                raise ValueError("end_temporary must restore baseline condition")
            if not update.supersedes_event_id:
                raise ValueError("end_temporary must supersede temporary event")
            temporary_key = next(
                (
                    candidate
                    for candidate, item in annotations.items()
                    if candidate[0] == update.supersedes_event_id
                    and item.temporal_action == "temporary_override"
                ),
                None,
            )
            if temporary_key is None:
                raise ValueError("end_temporary does not reference a temporary event")
            temporary = events[temporary_key[0]].preference_updates[temporary_key[1]]
            if update.previous_value != temporary.new_value:
                raise ValueError("end_temporary previous_value must match override")
            temporary_position = (
                event_order[temporary_key[0]],
                temporary_key[1],
            )
            ending_position = (event_order[event.event_id], key[1])
            if temporary_position >= ending_position:
                raise ValueError("end_temporary must occur after its override")

    explicit_hard_negatives = sum(
        not event.preference_updates
        and bool(_HARD_NEGATIVE_CUE.search(event.description))
        for chain in payload.vehicle_chains
        for event in chain.events
    )
    transition_event_opportunities = len(
        {
            transition.source_event_id
            for transition in payload.temporal_transitions
            if transition.temporal_action
            in {"current_upsert", "temporary_override", "end_temporary"}
        }
    )
    hard_negative_opportunities = (
        explicit_hard_negatives + transition_event_opportunities
    )
    if hard_negative_opportunities < profile.hard_negative_minimum_count:
        raise ValueError(
            "Temporal hard-negative opportunity count "
            f"{hard_negative_opportunities} is below "
            f"{profile.hard_negative_minimum_count}"
        )
    correction_count = _durable_correction_count(payload.temporal_transitions, events)
    if correction_count < 1:
        raise ValueError("Temporal scenario requires a durable correction")
    return {
        "action_counts": dict(sorted(action_counts.items())),
        "focus": profile.focus,
        "focus_count": focus_count,
        "explicit_hard_negative_event_count": explicit_hard_negatives,
        "hard_negative_opportunity_count": hard_negative_opportunities,
        "durable_correction_count": correction_count,
        "passed": True,
    }


def normalize_temporal_event_payload(
    payload: V2TemporalEventChainPayload,
    *,
    personas: Sequence[V1PersonaRecord] = (),
) -> tuple[V2TemporalEventChainPayload, tuple[str, ...]]:
    """Normalize only deterministic, semantically lossless provider slips."""

    state: dict[str, tuple[Any, str]] = {}
    normalized_events = {}
    notes = []
    persona_by_id = {persona.persona_id: persona for persona in personas}
    for item in interleave_event_chains(
        (*payload.background_chains, *payload.vehicle_chains)
    ):
        event = item.event
        participant_ids = event.participant_ids
        unknown_indexes = [
            index
            for index, persona_id in enumerate(participant_ids)
            if persona_id not in persona_by_id
        ]
        if persona_by_id and len(unknown_indexes) == 1:
            used = {item for item in participant_ids if item in persona_by_id}
            description = _normalized_phrase(event.description)
            candidates = [
                persona.persona_id
                for persona in personas
                if persona.persona_id not in used
                and any(
                    _phrase_occurs(description, label)
                    for label in (persona.name, *persona.aliases)
                )
            ]
            if len(candidates) == 1:
                repaired = list(participant_ids)
                bad_id = repaired[unknown_indexes[0]]
                repaired[unknown_indexes[0]] = candidates[0]
                participant_ids = tuple(repaired)
                notes.append(
                    f"{event.event_id}: repaired unambiguous participant_id "
                    f"{bad_id!r} -> {candidates[0]!r}"
                )
        updates = []
        for index, update in enumerate(event.preference_updates):
            identity = hybrid_memory_identity(update)
            if (
                update.supersedes_event_id is None
                and update.previous_value is not None
                and identity not in state
            ):
                notes.append(
                    f"{event.event_id}/{index}: cleared dangling previous_value "
                    "from new scoped identity"
                )
                update = update.model_copy(update={"previous_value": None})
                identity = hybrid_memory_identity(update)
            if update.supersedes_event_id is not None:
                obsolete = [
                    key
                    for key, (_, source_event_id) in state.items()
                    if source_event_id == update.supersedes_event_id
                    and key != identity
                ]
                for key in obsolete:
                    del state[key]
            state[identity] = (update.new_value, event.event_id)
            updates.append(update)
        normalized_events[event.event_id] = event.model_copy(
            update={
                "participant_ids": participant_ids,
                "preference_updates": tuple(updates),
            }
        )

    def rebuild(chain: V1EventChainRecord) -> V1EventChainRecord:
        return chain.model_copy(
            update={
                "events": tuple(
                    normalized_events[event.event_id] for event in chain.events
                )
            }
        )

    normalized = payload.model_copy(
        update={
            "background_chains": tuple(
                rebuild(chain) for chain in payload.background_chains
            ),
            "vehicle_chains": tuple(
                rebuild(chain) for chain in payload.vehicle_chains
            ),
        }
    )
    return normalized, tuple(notes)


def _normalized_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _phrase_occurs(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = _normalized_phrase(phrase)
    return bool(
        normalized_phrase
        and re.search(
            rf"(?:^|\s){re.escape(normalized_phrase)}(?:$|\s)",
            normalized_text,
        )
    )


def build_temporal_plan_artifact(
    generated: V2GeneratedTemporalEventChains,
    *,
    source_stage2_sha256: str,
) -> dict[str, Any]:
    audit = validate_temporal_event_payload(
        generated.payload,
        profile=generated.profile,
    )
    body = {
        "schema_version": V2_TEMPORAL_PLAN_SCHEMA_VERSION,
        "source_stage2_sha256": source_stage2_sha256,
        "scenario_profile": generated.profile.model_dump(mode="json"),
        "prompt_version": generated.prompt_version,
        "model_id": generated.model_id,
        "input_sha256": generated.input_sha256,
        "response_id": generated.response_id,
        "usage": generated.usage,
        "temporal_transitions": [
            item.model_dump(mode="json")
            for item in generated.payload.temporal_transitions
        ],
        "audit": audit,
    }
    return {**body, "artifact_sha256": canonical_json_sha256(body)}


def write_temporal_plan(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_temporal_plan(
    path: Path,
    *,
    source_stage2_sha256: str,
    required: bool = True,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Load and authenticate a temporal sidecar for Hybrid replay."""

    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != V2_TEMPORAL_PLAN_SCHEMA_VERSION:
        raise ValueError("Temporal plan has an unsupported schema version")
    if payload.get("source_stage2_sha256") != source_stage2_sha256:
        raise ValueError("Temporal plan belongs to another Stage 2 artifact")
    artifact_sha256 = payload.get("artifact_sha256")
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if artifact_sha256 != canonical_json_sha256(body):
        raise ValueError("Temporal plan artifact hash is invalid")
    transitions: dict[str, dict[int, dict[str, Any]]] = {}
    for raw in payload.get("temporal_transitions", []):
        transition = V2TemporalTransition.model_validate(raw)
        by_update = transitions.setdefault(transition.source_event_id, {})
        if transition.source_update_index in by_update:
            raise ValueError("Temporal plan contains a duplicate transition")
        by_update[transition.source_update_index] = transition.model_dump(mode="json")
    return transitions


def extract_dialogue_temporal_cue(action: str, text: str) -> str:
    """Return an exact current-dialogue cue required by a temporal action."""

    pattern = {
        "temporary_override": _DIALOGUE_TEMPORARY_CUE,
        "end_temporary": _DIALOGUE_END_CUE,
    }.get(action)
    if pattern is None:
        return ""
    match = pattern.search(text)
    return match.group(0).strip() if match else ""


def normalize_temporal_hybrid_alignment(
    event: V1EventRecord,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    generated: V2GeneratedHybridEventAlignment,
    transitions: Mapping[int, Mapping[str, Any]] | None,
) -> V2GeneratedHybridEventAlignment:
    """Move scoped updates to the first self-contained training turn.

    The generic alignment may correctly use a cue from the preceding prefix.
    Turn-wise SFT, however, receives only the current turn and approved memory,
    so temporary updates must be placed on a turn that restates both scope and
    value. An end marker needs only its explicit scope-ending cue because the
    durable baseline is already present in approved memory.
    """

    if not transitions:
        return generated
    normalized = []
    notes = list(generated.normalization_notes)
    for alignment in generated.payload.alignments:
        transition = transitions.get(alignment.source_update_index)
        action = str((transition or {}).get("temporal_action", ""))
        if action not in {"temporary_override", "end_temporary"}:
            normalized.append(alignment)
            continue
        update = event.preference_updates[alignment.source_update_index]
        participant_speakers = {turn.speaker_id for turn in dialogue_turns}
        selected = _find_self_contained_temporal_turn(
            action,
            dialogue_turns,
            start_index=(
                0 if action == "end_temporary" else alignment.evidence_turn_index
            ),
            new_value=update.new_value,
            subject_id=(
                update.subject_id
                if action == "temporary_override"
                and update.subject_id in participant_speakers
                else None
            ),
        )
        if selected is None:
            raise ValueError(
                f"Temporal alignment lacks a self-contained {action} turn"
            )
        if selected == alignment.evidence_turn_index:
            normalized.append(alignment)
            continue
        turn = dialogue_turns[selected]
        normalized.append(
            alignment.model_copy(
                update={
                    "evidence_turn_index": selected,
                    "evidence_quote": turn.text,
                    "reason": (
                        "The first self-contained turn states the temporal "
                        "scope and the memory change needed for turn-wise training."
                    ),
                }
            )
        )
        notes.append(
            f"source_update_index={alignment.source_update_index}:"
            f"temporal_turn={alignment.evidence_turn_index}->{selected}"
        )
    return generated.model_copy(
        update={
            "payload": V2HybridEventAlignmentPayload(
                alignments=tuple(normalized)
            ),
            "normalization_notes": tuple(notes),
        }
    )


def _find_self_contained_temporal_turn(
    action: str,
    dialogue_turns: Sequence[V1DialogueTurnRecord],
    *,
    start_index: int,
    new_value: Any,
    subject_id: str | None = None,
) -> int | None:
    for index in range(start_index, len(dialogue_turns)):
        if subject_id and dialogue_turns[index].speaker_id != subject_id:
            continue
        text = dialogue_turns[index].text
        if not extract_dialogue_temporal_cue(action, text):
            continue
        if action == "temporary_override" and not _text_mentions_value(
            text, new_value
        ):
            continue
        return index
    return None


def _text_mentions_value(text: str, value: Any) -> bool:
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    if isinstance(value, bool):
        candidates = ("true", "on", "enabled") if value else (
            "false",
            "off",
            "disabled",
        )
    elif isinstance(value, int) and not isinstance(value, bool):
        number_words = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        candidates = (str(value), number_words.get(value, ""))
    else:
        candidates = (" ".join(re.findall(r"[a-z0-9]+", str(value).casefold())),)
    return any(
        candidate
        and re.search(rf"(?:^|\s){re.escape(candidate)}(?:$|\s)", normalized)
        for candidate in candidates
    )


def validate_temporal_dialogue_cues(
    transitions: Mapping[int, Mapping[str, Any]] | None,
    turn_texts: Sequence[str],
    *,
    event: V1EventRecord | None = None,
    speaker_ids: Sequence[str] | None = None,
) -> None:
    """Ensure generated dialogue retains explicit temporary/end semantics."""

    if not transitions:
        return
    dialogue = "\n".join(turn_texts)
    for transition in transitions.values():
        action = str(transition.get("temporal_action", ""))
        if action in {"temporary_override", "end_temporary"} and not (
            extract_dialogue_temporal_cue(action, dialogue)
        ):
            raise ValueError(f"Generated dialogue lost explicit {action} cue")
        if action == "temporary_override" and event is not None:
            if speaker_ids is None or len(speaker_ids) != len(turn_texts):
                raise ValueError("Temporal dialogue speaker IDs are incomplete")
            turns = tuple(
                V1DialogueTurnRecord(
                    turn_id=f"{event.event_id}-validation-{index:03d}",
                    source_event_id=event.event_id,
                    timestamp=event.timestamp,
                    speaker_id=speaker_id,
                    speaker_name=speaker_id,
                    text=text,
                )
                for index, (speaker_id, text) in enumerate(
                    zip(speaker_ids, turn_texts, strict=True)
                )
            )
            update_index = int(transition["source_update_index"])
            update = event.preference_updates[update_index]
            required_speaker = (
                update.subject_id
                if update.subject_id in set(speaker_ids)
                else None
            )
            if _find_self_contained_temporal_turn(
                action,
                turns,
                start_index=0,
                new_value=update.new_value,
                subject_id=required_speaker,
            ) is None:
                raise ValueError(
                    "Generated dialogue lacks a self-contained temporary update turn"
                )


def _focus_count(
    focus: TemporalFocus,
    transitions: Sequence[V2TemporalTransition],
    events: Mapping[str, Any],
) -> int:
    counts = Counter(item.temporal_action for item in transitions)
    if focus == "current_durable":
        return counts["current_upsert"]
    if focus == "temporary_restore":
        return min(counts["temporary_override"], counts["end_temporary"])
    if focus == "conditional":
        return counts["conditional_upsert"]
    return _durable_correction_count(transitions, events)


def _durable_correction_count(
    transitions: Sequence[V2TemporalTransition],
    events: Mapping[str, Any],
) -> int:
    count = 0
    for transition in transitions:
        if transition.temporal_action != "durable_upsert":
            continue
        update = events[transition.source_event_id].preference_updates[
            transition.source_update_index
        ]
        count += bool(update.supersedes_event_id and update.previous_value is not None)
    return count


def _referenced_update(
    annotation: V2TemporalTransition,
    events: Mapping[str, Any],
) -> V1PreferenceUpdate:
    event_id = annotation.baseline_event_id
    update_index = annotation.baseline_source_update_index
    if event_id is None or update_index is None or event_id not in events:
        raise ValueError("Temporal baseline reference is invalid")
    event = events[event_id]
    if update_index >= len(event.preference_updates):
        raise ValueError("Temporal baseline update index is invalid")
    return event.preference_updates[update_index]


def _require_same_setting(
    left: V1PreferenceUpdate,
    right: V1PreferenceUpdate,
    *,
    require_value: bool,
) -> None:
    if (
        left.subject_id != right.subject_id
        or left.attribute_path != right.attribute_path
        or left.context_arguments != right.context_arguments
    ):
        raise ValueError("Temporal transition changed baseline setting identity")
    if require_value and left.new_value != right.new_value:
        raise ValueError("Temporal transition did not restore baseline value")


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0),
    }
