"""Paper-first Persona/Event generation for VehicleMemBench V1 reproduction."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from palmclaw_ubuntu.vehicle_bench.dataset import REASONING_TYPES
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    V1EventChainRecord,
    V1EventRecord,
    V1PersonaRecord,
    canonical_json_sha256,
)

V1_STAGE2_SCHEMA_VERSION = "vehiclemembench-v1-stage2-schema-v1"
V1_PERSONA_PROMPT_VERSION = "vehiclemembench-v1-persona-paper-stage1-rebuilt-v2"
V1_EVENT_CHAIN_PROMPT_VERSION = (
    "vehiclemembench-v1-event-chain-paper-appendix-g-rebuilt-v6"
)
V1_EVENT_CHAIN_STATE_EVOLUTION_PROMPT_VERSION = (
    "vehiclemembench-v1-event-chain-paper-appendix-g-state-evolution-v1"
)

# Appendix B, Table 4. These are used only to create a deterministic target mix.
V1_PUBLISHED_REASONING_COUNTS = {
    "preference_conflict": 149,
    "conditional_constraint": 102,
    "coreference_resolution": 97,
    "state_shift": 90,
    "error_correction": 62,
}
V1_DEFAULT_GENERATION_MODEL = "gpt-5.6-terra"
V1_SHARED_VEHICLE_SUBJECT_ID = "shared-vehicle"

V1_PERSONA_ENRICHMENT_INSTRUCTIONS = """
Create one coherent group of exactly three recurring vehicle occupants from the
three supplied Persona-Hub seed profiles.

Paper-specified objectives:
- Preserve each seed person's identity and enrich basic profile information,
  cultural interests, lifestyle habits, and vehicle-related preferences.
- For each basic_profile, include explicit keys for age, education, occupation,
  and MBTI. The person's name is stored in the top-level name field.
- Keep the three people diverse but plausible as occupants who share a vehicle.
- Include useful preference tendencies and potential conflicts without forcing
  every person to disagree.

Reconstructed output constraints:
- Return only the requested structured record.
- Keep persona_id unchanged and use aliases only for natural name variants.
- Do not generate event chains, dialogue, queries, answers, or future facts.
""".strip()

V1_EVENT_CHAIN_INSTRUCTIONS = """
Generate structured event chains for a three-person in-vehicle memory scenario.

Paper-specified objectives:
- Produce exactly 20 background chains unrelated to vehicle preferences and 10
  executable vehicle-preference chains.
- Vehicle chains require multi-hop reasoning, span weeks or months, and end in a
  delayed query seed whose answer is deterministic and executable.
- Use only the supplied reasoning-type assignment for the 10 vehicle chains.
- Every vehicle preference attribute must use a path from the supplied vehicle
  attribute catalog.
- Every event must involve at least two supplied occupants so it can be realized
  as human-to-human dialogue. Its description must contain enough concrete
  who/what/where/how/why/result detail for a natural conversation, especially
  for background events that later require at least 40 dialogue turns.
- Events from different chains must have timestamps suitable for one temporally
  interleaved timeline.
- Generate at least 80 historical events in total and stay close to the paper's
  reported average of 81.78 events per scenario.

Reconstructed output constraints:
- Return only the requested structured record with stable unique IDs.
- preference_updates.subject_id must be an exact supplied persona_id. Use the
  reserved value shared-vehicle only for an explicitly negotiated multi-occupant
  compromise or a vehicle-wide safety rule, never for an individual's setting.
- Background chains must not contain vehicle preference updates.
- Each vehicle chain must contain at least two events and at least one preference
  update. Values and required context arguments must satisfy the Tool schema.
- In each preference_update, context_arguments contains selector arguments only
  (for example zone, seat, light, side, or wiper). Do not repeat the terminal
  value argument from attribute_path there: new_value is its single source.
- Treat Tool descriptions and validation hints as executable constraints. Use
  exact simulator values rather than natural-language aliases. In particular,
  vehicle scopes use driver/passenger/rear_left/rear_right/front/rear/all;
  circulation uses inside/outside; do not emit cabin, fresh_air, or
  front_passenger as executable arguments.
- Do not generate natural dialogue or final benchmark answers in this stage.
""".strip()

V1_EVENT_CHAIN_STATE_EVOLUTION_INSTRUCTIONS = f"""
{V1_EVENT_CHAIN_INSTRUCTIONS}

Project-defined V2 state-evolution extension (required for S21 and later):
- Keep one initial preference update in each of the 10 vehicle chains. Across
  the scenario these initial updates must have distinct memory identities
  (subject, attribute, selector context, and condition), yielding at least 10
  deterministic ADD operations.
- Add at least two later updates that change the value of an existing identity.
  Preserve subject_id, attribute_path, context_arguments, and condition; set
  previous_value to the prior value and supersedes_event_id to the prior source
  event. These updates must yield deterministic REPLACE operations.
- Add at least one explicit withdrawal or correction of an earlier scoped
  preference. Point supersedes_event_id to the withdrawn source event, retain
  its previous_value, and use a genuinely corrected context or condition. The
  schema represents this as deterministic DELETE followed by ADD.
- State changes must be motivated in the event descriptions and later dialogue,
  not attached as artificial metadata. The delayed query must remain answerable
  from the latest valid state after all replacements and withdrawals.
- Never mark an initial fact as superseding another event. Never reference a
  future or unrelated event in supersedes_event_id.
""".strip()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V1PersonaSeed(_StrictModel):
    seed_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    source: str = Field(min_length=1)
    profile: dict[str, JsonValue] = Field(min_length=1)


class V1PersonaGroupPayload(_StrictModel):
    personas: tuple[V1PersonaRecord, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_unique_personas(self) -> V1PersonaGroupPayload:
        identifiers = [persona.persona_id for persona in self.personas]
        if len(set(identifiers)) != 3:
            raise ValueError("persona IDs must be unique")
        names = [persona.name.casefold() for persona in self.personas]
        if len(set(names)) != 3:
            raise ValueError("persona names must be unique")
        if any(not persona.vehicle_preferences for persona in self.personas):
            raise ValueError("every enriched persona requires vehicle preferences")
        required_basic = {"age", "education", "occupation", "mbti"}
        for persona in self.personas:
            basic_keys = {
                item.key.strip().casefold().replace(" ", "_")
                for item in persona.basic_profile
            }
            if not required_basic.issubset(basic_keys):
                missing = sorted(required_basic - basic_keys)
                raise ValueError(
                    f"persona {persona.persona_id} misses basic fields: {missing}"
                )
        return self


class V1VehicleAttribute(_StrictModel):
    path: str = Field(min_length=1)
    module: str = Field(min_length=1)
    tool_name: str = Field(pattern=r"^carcontrol_[A-Za-z0-9_]+$")
    tool_description: str = ""
    argument_name: str = Field(min_length=1)
    argument_description: str = ""
    validation_hints: tuple[str, ...] = ()
    required_arguments: tuple[str, ...]
    argument_schema: dict[str, JsonValue]
    tool_parameters_schema: dict[str, JsonValue]


class V1EventChainPayload(_StrictModel):
    background_chains: tuple[V1EventChainRecord, ...] = Field(
        min_length=20,
        max_length=20,
    )
    vehicle_chains: tuple[V1EventChainRecord, ...] = Field(
        min_length=10,
        max_length=10,
    )

    @model_validator(mode="after")
    def validate_chain_kinds(self) -> V1EventChainPayload:
        if any(chain.kind != "background" for chain in self.background_chains):
            raise ValueError("background_chains contains a vehicle chain")
        if any(chain.kind != "vehicle" for chain in self.vehicle_chains):
            raise ValueError("vehicle_chains contains a background chain")
        return self

    @property
    def all_chains(self) -> tuple[V1EventChainRecord, ...]:
        return (*self.background_chains, *self.vehicle_chains)


class V1GeneratedPersonaGroup(_StrictModel):
    candidate_group_id: str = Field(min_length=1)
    seed_persona_ids: tuple[str, str, str]
    seed_personas: tuple[V1PersonaSeed, V1PersonaSeed, V1PersonaSeed]
    payload: V1PersonaGroupPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]


class V1GeneratedEventChains(_StrictModel):
    scenario_candidate_id: str = Field(min_length=1)
    planned_reasoning_types: tuple[str, ...] = Field(min_length=10, max_length=10)
    payload: V1EventChainPayload
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vehicle_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_id: str | None = None
    usage: dict[str, int]


class V1InterleavedEvent(_StrictModel):
    timeline_index: int = Field(ge=0)
    chain_id: str = Field(min_length=1)
    chain_kind: str
    reasoning_type: str | None
    event: V1EventRecord


class V1Stage2Audit(_StrictModel):
    persona_count: int
    background_chain_count: int
    vehicle_chain_count: int
    event_count: int
    vehicle_event_count: int
    background_event_count: int
    timestamp_span_days: float
    reasoning_type_counts: dict[str, int]
    attribute_update_count: int
    passed: bool


class V1StateEvolutionCoverage(_StrictModel):
    initial_identity_count: int = Field(ge=0)
    add_count: int = Field(ge=0)
    replace_count: int = Field(ge=0)
    delete_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    explicit_supersession_count: int = Field(ge=0)
    passed: bool


class V1Stage2Artifact(_StrictModel):
    schema_version: str = V1_STAGE2_SCHEMA_VERSION
    scenario_candidate_id: str = Field(min_length=1)
    persona_group: V1GeneratedPersonaGroup
    event_chains: V1GeneratedEventChains
    interleaved_timeline: tuple[V1InterleavedEvent, ...] = Field(min_length=1)
    audit: V1Stage2Audit
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifact_hash(self) -> V1Stage2Artifact:
        payload = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if canonical_json_sha256(payload) != self.artifact_sha256:
            raise ValueError("artifact_sha256 does not match the Stage 2 artifact")
        return self


class V1PersonaGenerationModel(Protocol):
    def generate(
        self,
        seeds: Sequence[V1PersonaSeed],
        *,
        candidate_group_id: str,
    ) -> V1GeneratedPersonaGroup: ...


class V1EventChainGenerationModel(Protocol):
    def generate(
        self,
        personas: Sequence[V1PersonaRecord],
        *,
        scenario_candidate_id: str,
        reasoning_types: Sequence[str],
        vehicle_attributes: Sequence[V1VehicleAttribute],
    ) -> V1GeneratedEventChains: ...


class OpenAIV1PersonaGenerationModel:
    """Structured Persona-Hub enrichment using the reconstructed Stage 1 prompt."""

    def __init__(
        self,
        model_id: str,
        *,
        timeout_seconds: float,
        temperature: float | None = None,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int = 8_192,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("persona generation model ID is required")
        self.model_id = model_id
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def generate(
        self,
        seeds: Sequence[V1PersonaSeed],
        *,
        candidate_group_id: str,
    ) -> V1GeneratedPersonaGroup:
        provider_input = render_persona_generation_input(
            seeds,
            candidate_group_id=candidate_group_id,
        )
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": V1_PERSONA_ENRICHMENT_INSTRUCTIONS,
            "input": provider_input,
            "text_format": V1PersonaGroupPayload,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        response = self._client.responses.parse(
            **request,
        )
        _raise_for_bad_response(response, stage="persona generation")
        payload = V1PersonaGroupPayload.model_validate(response.output_parsed)
        seed_ids = tuple(seed.seed_id for seed in seeds)
        if len(seed_ids) != 3:
            raise ValueError("persona generation requires exactly three seeds")
        if {persona.persona_id for persona in payload.personas} != set(seed_ids):
            raise ValueError("generated persona IDs must match the seed IDs")
        return V1GeneratedPersonaGroup(
            candidate_group_id=candidate_group_id,
            seed_persona_ids=seed_ids,
            seed_personas=tuple(seeds),
            payload=payload,
            model_id=self.model_id,
            prompt_version=V1_PERSONA_PROMPT_VERSION,
            input_sha256=canonical_json_sha256(json.loads(provider_input)),
            response_id=getattr(response, "id", None),
            usage=_response_usage(response),
        )


class OpenAIV1EventChainGenerationModel:
    """Structured event-chain generation following Appendix G."""

    def __init__(
        self,
        model_id: str,
        *,
        timeout_seconds: float,
        temperature: float | None = None,
        reasoning_effort: str | None = "medium",
        max_output_tokens: int = 32_768,
        state_evolution: bool = False,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("event generation model ID is required")
        self.model_id = model_id
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.state_evolution = state_evolution
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
    ) -> V1GeneratedEventChains:
        provider_input = render_event_chain_generation_input(
            personas,
            scenario_candidate_id=scenario_candidate_id,
            reasoning_types=reasoning_types,
            vehicle_attributes=vehicle_attributes,
        )
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": (
                V1_EVENT_CHAIN_STATE_EVOLUTION_INSTRUCTIONS
                if self.state_evolution
                else V1_EVENT_CHAIN_INSTRUCTIONS
            ),
            "input": provider_input,
            "text_format": V1EventChainPayload,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        response = self._client.responses.parse(
            **request,
        )
        _raise_for_bad_response(response, stage="event-chain generation")
        payload = V1EventChainPayload.model_validate(response.output_parsed)
        catalog_payload = [item.model_dump(mode="json") for item in vehicle_attributes]
        return V1GeneratedEventChains(
            scenario_candidate_id=scenario_candidate_id,
            planned_reasoning_types=tuple(reasoning_types),
            payload=payload,
            model_id=self.model_id,
            prompt_version=(
                V1_EVENT_CHAIN_STATE_EVOLUTION_PROMPT_VERSION
                if self.state_evolution
                else V1_EVENT_CHAIN_PROMPT_VERSION
            ),
            input_sha256=canonical_json_sha256(json.loads(provider_input)),
            vehicle_catalog_sha256=canonical_json_sha256(catalog_payload),
            response_id=getattr(response, "id", None),
            usage=_response_usage(response),
        )


def build_vehicle_attribute_catalog(
    tool_schemas: Sequence[Mapping[str, Any]],
) -> tuple[V1VehicleAttribute, ...]:
    catalog: list[V1VehicleAttribute] = []
    seen_paths: set[str] = set()
    for tool in tool_schemas:
        name = tool.get("name")
        parameters = tool.get("parameters")
        if not isinstance(name, str) or not name.startswith("carcontrol_"):
            raise ValueError(f"invalid vehicle tool name: {name!r}")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"invalid parameter schema for {name}")
        properties = parameters.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError(f"vehicle tool has invalid parameter properties: {name}")
        if not properties:
            # Command-only actions such as navigation_stop do not expose a
            # preference value and therefore are not attribute candidates.
            continue
        required_raw = parameters.get("required", [])
        if not isinstance(required_raw, list) or not all(
            isinstance(item, str) for item in required_raw
        ):
            raise ValueError(f"invalid required arguments for {name}")
        module = name.removeprefix("carcontrol_").split("_", 1)[0]
        for argument_name, argument_schema in sorted(properties.items()):
            if not isinstance(argument_name, str) or not isinstance(
                argument_schema, Mapping
            ):
                raise ValueError(f"invalid argument schema for {name}")
            path = f"{name}.{argument_name}"
            if path in seen_paths:
                raise ValueError(f"duplicate vehicle attribute path: {path}")
            seen_paths.add(path)
            catalog.append(
                V1VehicleAttribute(
                    path=path,
                    module=module,
                    tool_name=name,
                    tool_description=str(tool.get("description", "")),
                    argument_name=argument_name,
                    argument_description=str(argument_schema.get("description", "")),
                    validation_hints=_vehicle_argument_validation_hints(
                        name,
                        argument_name,
                    ),
                    required_arguments=tuple(required_raw),
                    argument_schema=dict(argument_schema),
                    tool_parameters_schema=dict(parameters),
                )
            )
    return tuple(sorted(catalog, key=lambda item: item.path))


def _vehicle_argument_validation_hints(
    tool_name: str,
    argument_name: str,
) -> tuple[str, ...]:
    """Expose simulator selector conventions omitted by generated JSON schemas."""

    exact_domains = {
        (
            "carcontrol_airConditioner_set_air_direction",
            "direction",
        ): "face|feet|window|face_feet|face_window|feet_window|face_feet_window",
    }
    scopes = {
        "zone": "driver|passenger|rear_left|rear_right|front|rear|all",
        "seat": "driver|passenger|rear_left|rear_right",
        "light": "driver|passenger|rear_left|rear_right|front|rear|all",
        "window": "driver|passenger|rear_left|rear_right|front|rear|all",
        "door": "driver|passenger|rear_left|rear_right|front|rear|all",
        "circulation": "inside|outside",
    }
    hint = exact_domains.get((tool_name, argument_name), scopes.get(argument_name))
    return (f"allowed values: {hint}",) if hint is not None else ()


def build_reasoning_type_plan(
    scenario_count: int,
    *,
    seed: int = 260323840,
    source_counts: Mapping[str, int] = V1_PUBLISHED_REASONING_COUNTS,
) -> tuple[tuple[str, ...], ...]:
    """Scale the published distribution with deterministic largest remainder."""

    if scenario_count < 1:
        raise ValueError("scenario_count must be positive")
    if set(source_counts) != REASONING_TYPES or any(
        value < 0 for value in source_counts.values()
    ):
        raise ValueError("source_counts must cover the five official reasoning types")
    source_total = sum(source_counts.values())
    if source_total < 1:
        raise ValueError("source reasoning count total must be positive")
    target_total = scenario_count * 10
    raw = {
        name: target_total * count / source_total
        for name, count in source_counts.items()
    }
    allocations = {name: int(value) for name, value in raw.items()}
    remaining = target_total - sum(allocations.values())
    remainder_order = sorted(
        raw,
        key=lambda name: (-(raw[name] - allocations[name]), name),
    )
    for name in remainder_order[:remaining]:
        allocations[name] += 1

    flat = [
        name
        for name in sorted(allocations)
        for _ in range(allocations[name])
    ]
    random.Random(seed).shuffle(flat)
    return tuple(
        tuple(flat[index : index + 10])
        for index in range(0, len(flat), 10)
    )


def load_persona_seeds(path: Path | str) -> tuple[V1PersonaSeed, ...]:
    """Load the normalized, versioned Persona-Hub seed JSONL boundary."""

    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"persona seed path is not a file: {source}")
    seeds: list[V1PersonaSeed] = []
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
            seeds.append(V1PersonaSeed.model_validate(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid persona seed JSONL at {source.name}:{line_number}: {exc}"
            ) from exc
    if not seeds:
        raise ValueError(f"persona seed file is empty: {source}")
    identifiers = [seed.seed_id for seed in seeds]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("persona seed IDs must be unique")
    return tuple(seeds)


def build_persona_seed_groups(
    seeds: Sequence[V1PersonaSeed],
    *,
    candidate_group_count: int,
    seed: int = 260323840,
) -> tuple[tuple[V1PersonaSeed, V1PersonaSeed, V1PersonaSeed], ...]:
    """Randomly group distinct seed personas exactly once, as reported in Stage 1."""

    if candidate_group_count < 1:
        raise ValueError("candidate_group_count must be positive")
    required = candidate_group_count * 3
    if len(seeds) < required:
        raise ValueError(
            f"need at least {required} distinct persona seeds, found {len(seeds)}"
        )
    identifiers = [item.seed_id for item in seeds]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("persona seed IDs must be unique")
    selected = list(seeds)
    random.Random(seed).shuffle(selected)
    selected = selected[:required]
    return tuple(
        (selected[index], selected[index + 1], selected[index + 2])
        for index in range(0, required, 3)
    )


def render_persona_generation_input(
    seeds: Sequence[V1PersonaSeed],
    *,
    candidate_group_id: str,
) -> str:
    if len(seeds) != 3 or len({seed.seed_id for seed in seeds}) != 3:
        raise ValueError("persona generation requires three unique seeds")
    payload = {
        "candidate_group_id": candidate_group_id,
        "seed_personas": [seed.model_dump(mode="json") for seed in seeds],
    }
    return _canonical_json(payload)


def render_event_chain_generation_input(
    personas: Sequence[V1PersonaRecord],
    *,
    scenario_candidate_id: str,
    reasoning_types: Sequence[str],
    vehicle_attributes: Sequence[V1VehicleAttribute],
) -> str:
    if len(personas) != 3 or len({persona.persona_id for persona in personas}) != 3:
        raise ValueError("event generation requires three unique personas")
    if len(reasoning_types) != 10 or any(
        reasoning_type not in REASONING_TYPES for reasoning_type in reasoning_types
    ):
        raise ValueError("event generation requires 10 official reasoning types")
    if not vehicle_attributes:
        raise ValueError("vehicle attribute catalog cannot be empty")
    payload = {
        "scenario_candidate_id": scenario_candidate_id,
        "personas": [persona.model_dump(mode="json") for persona in personas],
        "vehicle_chain_reasoning_types_in_order": list(reasoning_types),
        "vehicle_attribute_catalog": [
            attribute.model_dump(mode="json") for attribute in vehicle_attributes
        ],
    }
    return _canonical_json(payload)


def interleave_event_chains(
    chains: Sequence[V1EventChainRecord],
) -> tuple[V1InterleavedEvent, ...]:
    ordered = sorted(
        (
            event.timestamp,
            chain.chain_id,
            event.event_id,
            chain,
            event,
        )
        for chain in chains
        for event in chain.events
    )
    return tuple(
        V1InterleavedEvent(
            timeline_index=index,
            chain_id=chain.chain_id,
            chain_kind=chain.kind,
            reasoning_type=chain.reasoning_type,
            event=event,
        )
        for index, (_, _, _, chain, event) in enumerate(ordered)
    )


def validate_stage2_contract(
    *,
    personas: Sequence[V1PersonaRecord],
    event_payload: V1EventChainPayload,
    planned_reasoning_types: Sequence[str],
    vehicle_attributes: Sequence[V1VehicleAttribute],
    minimum_event_count: int = 80,
    minimum_span_days: int = 14,
) -> V1Stage2Audit:
    if len(personas) != 3:
        raise ValueError("Stage 2 requires exactly three personas")
    persona_ids = {persona.persona_id for persona in personas}
    if len(persona_ids) != 3:
        raise ValueError("Stage 2 persona IDs must be unique")
    chains = event_payload.all_chains
    chain_ids = [chain.chain_id for chain in chains]
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError("event chain IDs must be unique")
    event_ids = [event.event_id for chain in chains for event in chain.events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event IDs must be unique")
    if len(event_ids) < minimum_event_count:
        raise ValueError(
            f"event count {len(event_ids)} is below the paper-based minimum "
            f"{minimum_event_count}"
        )

    planned = Counter(planned_reasoning_types)
    actual = Counter(chain.reasoning_type for chain in event_payload.vehicle_chains)
    if actual != planned:
        raise ValueError(
            "vehicle reasoning mix differs from plan: "
            f"{dict(actual)} != {dict(planned)}"
        )
    catalog = {attribute.path: attribute for attribute in vehicle_attributes}
    update_count = 0
    timestamps: list[datetime] = []
    for chain in chains:
        chain_times = [_parse_timestamp(event.timestamp) for event in chain.events]
        timestamps.extend(chain_times)
        if chain_times != sorted(chain_times):
            raise ValueError(f"events are not chronological in chain {chain.chain_id}")
        if chain.kind == "background" and any(
            event.preference_updates for event in chain.events
        ):
            raise ValueError("background chains cannot contain vehicle preferences")
        if chain.kind == "vehicle" and len(chain.events) < 2:
            raise ValueError("vehicle chains must require at least two events")
        for event in chain.events:
            if len(set(event.participant_ids)) < 2:
                raise ValueError(
                    f"event {event.event_id} requires at least two participants"
                )
            if not set(event.participant_ids).issubset(persona_ids):
                raise ValueError(
                    f"event {event.event_id} references an unknown persona"
                )
            for update in event.preference_updates:
                update_count += 1
                if update.subject_id not in persona_ids and (
                    update.subject_id != V1_SHARED_VEHICLE_SUBJECT_ID
                ):
                    raise ValueError("preference update references an unknown persona")
                if (
                    update.subject_id != V1_SHARED_VEHICLE_SUBJECT_ID
                    and update.subject_id not in event.participant_ids
                ):
                    raise ValueError(
                        "preference subject must participate in the source event"
                    )
                attribute = catalog.get(update.attribute_path)
                if attribute is None:
                    raise ValueError(
                        f"unknown vehicle attribute path: {update.attribute_path}"
                    )
                arguments = update.context_argument_map()
                arguments[attribute.argument_name] = update.new_value
                try:
                    Draft202012Validator(
                        attribute.tool_parameters_schema
                    ).validate(arguments)
                except JsonSchemaValidationError as exc:
                    raise ValueError(
                        f"invalid preference value for {update.attribute_path}: "
                        f"{exc.message}"
                    ) from exc

    span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86_400
    if span_days < minimum_span_days:
        raise ValueError(
            f"event span {span_days:.2f} days is shorter than {minimum_span_days}"
        )
    background_events = sum(
        len(chain.events) for chain in event_payload.background_chains
    )
    vehicle_events = sum(len(chain.events) for chain in event_payload.vehicle_chains)
    return V1Stage2Audit(
        persona_count=len(personas),
        background_chain_count=len(event_payload.background_chains),
        vehicle_chain_count=len(event_payload.vehicle_chains),
        event_count=len(event_ids),
        vehicle_event_count=vehicle_events,
        background_event_count=background_events,
        timestamp_span_days=round(span_days, 3),
        reasoning_type_counts=dict(sorted(actual.items())),
        attribute_update_count=update_count,
        passed=True,
    )


def validate_state_evolution_coverage(
    event_payload: V1EventChainPayload,
    *,
    minimum_add_count: int = 10,
    minimum_replace_count: int = 2,
    minimum_delete_count: int = 1,
) -> V1StateEvolutionCoverage:
    """Validate S21+ state transitions using the Hybrid identity semantics."""

    if min(minimum_add_count, minimum_replace_count, minimum_delete_count) < 0:
        raise ValueError("state-evolution minimums cannot be negative")

    initial_updates = []
    for chain in event_payload.vehicle_chains:
        first = next(
            (
                update
                for event in chain.events
                for update in event.preference_updates
            ),
            None,
        )
        if first is None:
            raise ValueError(f"state evolution chain {chain.chain_id} has no update")
        if first.previous_value is not None or first.supersedes_event_id is not None:
            raise ValueError(
                f"state evolution chain {chain.chain_id} must begin with a new fact"
            )
        initial_updates.append(first)
    initial_identity_count = len(
        {_state_evolution_identity(update) for update in initial_updates}
    )
    if initial_identity_count != len(event_payload.vehicle_chains):
        raise ValueError(
            "state-evolution vehicle chains need distinct initial identities"
        )

    # identity -> (current value, source event ID)
    state: dict[str, tuple[JsonValue, str]] = {}
    add_count = replace_count = delete_count = duplicate_count = 0
    supersession_count = 0
    timeline = interleave_event_chains(event_payload.all_chains)
    for item in timeline:
        event = item.event
        for update in event.preference_updates:
            identity = _state_evolution_identity(update)
            superseded: list[tuple[str, JsonValue]] = []
            if update.supersedes_event_id is not None:
                supersession_count += 1
                superseded = [
                    (key, value)
                    for key, (value, source_event_id) in state.items()
                    if source_event_id == update.supersedes_event_id
                ]
                if not superseded:
                    raise ValueError(
                        f"state evolution {event.event_id} supersedes an inactive "
                        f"or unknown event: {update.supersedes_event_id}"
                    )
                prior_values = {value for _, value in superseded}
                if update.previous_value not in prior_values:
                    raise ValueError(
                        f"state evolution {event.event_id} previous_value does not "
                        "match its superseded state"
                    )
                for obsolete_identity, _ in superseded:
                    if obsolete_identity != identity:
                        del state[obsolete_identity]
                        delete_count += 1

            existing = state.get(identity)
            if existing is None:
                if (
                    update.supersedes_event_id is None
                    and update.previous_value is not None
                ):
                    raise ValueError(
                        f"state evolution {event.event_id} has a dangling "
                        "previous_value"
                    )
                state[identity] = (update.new_value, event.event_id)
                add_count += 1
            elif existing[0] == update.new_value:
                # This semantic audit treats a repeated value as duplicate even
                # though the rendered memory line includes a new timestamp.
                duplicate_count += 1
            else:
                if update.supersedes_event_id is None:
                    raise ValueError(
                        f"state evolution {event.event_id} changes an existing "
                        "identity without supersedes_event_id"
                    )
                if update.previous_value != existing[0]:
                    raise ValueError(
                        f"state evolution {event.event_id} previous_value does not "
                        "match the current identity value"
                    )
                state[identity] = (update.new_value, event.event_id)
                replace_count += 1

    coverage = V1StateEvolutionCoverage(
        initial_identity_count=initial_identity_count,
        add_count=add_count,
        replace_count=replace_count,
        delete_count=delete_count,
        duplicate_count=duplicate_count,
        explicit_supersession_count=supersession_count,
        passed=(
            add_count >= minimum_add_count
            and replace_count >= minimum_replace_count
            and delete_count >= minimum_delete_count
        ),
    )
    if not coverage.passed:
        raise ValueError(
            "state-evolution coverage is below minimum: "
            f"add={add_count}/{minimum_add_count}, "
            f"replace={replace_count}/{minimum_replace_count}, "
            f"delete={delete_count}/{minimum_delete_count}"
        )
    return coverage


def _state_evolution_identity(update: Any) -> str:
    payload = {
        "subject_id": update.subject_id,
        "attribute_path": update.attribute_path,
        "condition": update.condition,
        "context_arguments": [
            argument.model_dump(mode="json")
            for argument in update.context_arguments
            if not update.attribute_path.endswith(f".{argument.name}")
        ],
    }
    return canonical_json_sha256(payload)


def validate_stage2_simulator_arguments(
    event_payload: V1EventChainPayload,
    *,
    runtime: Any,
) -> int:
    """Execute every structured preference update against VehicleWorld ranges."""

    executed = 0
    for chain in event_payload.vehicle_chains:
        for event in chain.events:
            for update in event.preference_updates:
                tool_name, argument_name = update.attribute_path.rsplit(".", 1)
                arguments = update.context_argument_map()
                arguments[argument_name] = update.new_value
                result = runtime.execute(
                    runtime.create_world(),
                    tool_name,
                    arguments,
                )
                if not isinstance(result, Mapping) or result.get("success") is not True:
                    message = (
                        result.get("message")
                        if isinstance(result, Mapping)
                        else result
                    )
                    raise ValueError(
                        f"simulator rejected {chain.chain_id}/{event.event_id} "
                        f"{tool_name}: {message}"
                    )
                executed += 1
    return executed


def generate_stage2_artifact(
    *,
    candidate_group_id: str,
    scenario_candidate_id: str,
    seeds: Sequence[V1PersonaSeed],
    reasoning_types: Sequence[str],
    vehicle_attributes: Sequence[V1VehicleAttribute],
    persona_model: V1PersonaGenerationModel,
    event_model: V1EventChainGenerationModel,
    require_state_evolution: bool = False,
) -> V1Stage2Artifact:
    persona_group = persona_model.generate(
        seeds,
        candidate_group_id=candidate_group_id,
    )
    event_chains = event_model.generate(
        persona_group.payload.personas,
        scenario_candidate_id=scenario_candidate_id,
        reasoning_types=reasoning_types,
        vehicle_attributes=vehicle_attributes,
    )
    audit = validate_stage2_contract(
        personas=persona_group.payload.personas,
        event_payload=event_chains.payload,
        planned_reasoning_types=reasoning_types,
        vehicle_attributes=vehicle_attributes,
    )
    if require_state_evolution:
        validate_state_evolution_coverage(event_chains.payload)
    timeline = interleave_event_chains(event_chains.payload.all_chains)
    body = {
        "schema_version": V1_STAGE2_SCHEMA_VERSION,
        "scenario_candidate_id": scenario_candidate_id,
        "persona_group": persona_group.model_dump(mode="json"),
        "event_chains": event_chains.model_dump(mode="json"),
        "interleaved_timeline": [item.model_dump(mode="json") for item in timeline],
        "audit": audit.model_dump(mode="json"),
    }
    return V1Stage2Artifact(
        **body,
        artifact_sha256=canonical_json_sha256(body),
    )


def write_stage2_artifact(
    output_dir: Path | str,
    artifact: V1Stage2Artifact,
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "stage2.json"
    temporary = root / ".stage2.json.tmp"
    temporary.write_text(
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _raise_for_bad_response(response: Any, *, stage: str) -> None:
    status = getattr(response, "status", None)
    if status in {"failed", "incomplete", "cancelled"}:
        raise RuntimeError(f"V1 {stage} provider returned status {status}")


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
    }
