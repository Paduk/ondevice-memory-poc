from __future__ import annotations

import json
from importlib.resources import files

import pytest

from palmclaw_ubuntu.models import MemoryMessage
from palmclaw_ubuntu.providers import FakeEmbeddingModel
from palmclaw_ubuntu.vehicle_bench.vehicle_fact_ontology import (
    VEHICLE_FACT_ONTOLOGY_RESOURCE,
    VEHICLE_FACT_ONTOLOGY_VERSION,
    VehicleFactOntologyError,
    VehicleFactOntologyMatcher,
    build_vehicle_fact_ontology,
    vehicle_tool_schema_sha256,
)


def _tool(
    name: str,
    description: str,
    properties: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
        },
    }


def _definition(
    schemas: list[dict[str, object]],
    tools: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": VEHICLE_FACT_ONTOLOGY_VERSION,
        "source_policy": "Tool interface only; excludes QA, History, and Gold",
        "source_schema_sha256": vehicle_tool_schema_sha256(schemas),
        "tools": tools,
    }


def test_frozen_vehicle_fact_ontology_v1_mapping_is_complete():
    resource = files("palmclaw_ubuntu.vehicle_bench").joinpath(
        VEHICLE_FACT_ONTOLOGY_RESOURCE
    )
    definition = json.loads(resource.read_text(encoding="utf-8"))

    assert definition["schema_version"] == VEHICLE_FACT_ONTOLOGY_VERSION
    assert len(definition["tools"]) == 111
    assert sum(
        len(binding["arguments"])
        for binding in definition["tools"].values()
    ) == 155
    assert {
        role
        for binding in definition["tools"].values()
        for role in binding["arguments"].values()
    } == {"content", "selector", "value"}
    assert "Gold" in definition["source_policy"]
    assert "excludes" in definition["source_policy"]


def test_build_vehicle_fact_ontology_groups_targets_and_constraints():
    schemas = [
        _tool(
            "carcontrol_radio_set_volume",
            "Set radio volume (0-100).",
            {"volume": {"type": "integer"}},
        ),
        _tool(
            "carcontrol_video_set_volume",
            "Set video volume (0-100).",
            {"volume": {"type": "integer"}},
        ),
    ]
    definition = _definition(
        schemas,
        {
            "carcontrol_radio_set_volume": {
                "capability": "audio.volume",
                "target": "radio",
                "arguments": {"volume": "value"},
            },
            "carcontrol_video_set_volume": {
                "capability": "audio.volume",
                "target": "video",
                "arguments": {"volume": "value"},
            },
        },
    )

    ontology = build_vehicle_fact_ontology(schemas, definition)

    capability = ontology.capability("audio.volume")
    assert capability is not None
    assert capability.allowed_targets == (
        "other",
        "radio",
        "unspecified",
        "video",
    )
    assert capability.value_types == ("integer",)
    radio = ontology.binding("carcontrol_radio_set_volume")
    assert radio is not None
    assert radio.arguments[0].description_constraints == {
        "maximum": 100,
        "minimum": 0,
    }


def test_target_resolution_uses_explicit_or_unique_supported_target():
    schemas = [
        _tool(
            "carcontrol_radio_set_volume",
            "Set radio volume.",
            {"volume": {"type": "integer"}},
        ),
        _tool(
            "carcontrol_video_set_volume",
            "Set video volume.",
            {"volume": {"type": "integer"}},
        ),
    ]
    definition = _definition(
        schemas,
        {
            "carcontrol_radio_set_volume": {
                "capability": "audio.volume",
                "target": "radio",
                "arguments": {"volume": "value"},
            },
            "carcontrol_video_set_volume": {
                "capability": "audio.volume",
                "target": "video",
                "arguments": {"volume": "value"},
            },
        },
    )
    ontology = build_vehicle_fact_ontology(schemas, definition)

    assert ontology.resolve_target(
        "audio.volume", explicit_target="Radio"
    ) == "radio"
    assert ontology.resolve_target(
        "audio.volume", supported_targets=("radio",)
    ) == "radio"
    assert ontology.resolve_target(
        "audio.volume", supported_targets=("radio", "video")
    ) == "unspecified"
    assert ontology.resolve_target(
        "audio.volume", explicit_target="cassette"
    ) == "other"
    assert ontology.resolve_target(
        "audio.volume", explicit_target="unspecified"
    ) == "unspecified"
    assert ontology.resolve_target(
        "audio.volume", explicit_target="other"
    ) == "other"


def test_ontology_matcher_retrieves_a_small_cached_capability_slice():
    schemas = [
        _tool(
            "carcontrol_radio_set_volume",
            "Set radio audio volume (0-100).",
            {"volume": {"type": "integer"}},
        ),
        _tool(
            "carcontrol_seat_set_heating_level",
            "Set seat heating level (1-3).",
            {
                "seat": {"type": "string"},
                "level": {"type": "integer"},
            },
        ),
    ]
    definition = _definition(
        schemas,
        {
            "carcontrol_radio_set_volume": {
                "capability": "audio.volume",
                "target": "radio",
                "arguments": {"volume": "value"},
            },
            "carcontrol_seat_set_heating_level": {
                "capability": "heating.level",
                "target": "seat",
                "arguments": {"seat": "selector", "level": "value"},
            },
        },
    )
    ontology = build_vehicle_fact_ontology(schemas, definition)
    embedding = FakeEmbeddingModel()
    matcher = VehicleFactOntologyMatcher(
        ontology,
        embedding_model=embedding,
        top_k_per_message=1,
        max_candidates=1,
    )
    messages = [
        MemoryMessage(
            id=1,
            role="user",
            content="Set the radio audio volume to 20.",
        )
    ]

    first = matcher.match(messages)
    second = matcher.match(messages)
    per_message = matcher.match_each(
        [
            MemoryMessage(
                id=2,
                role="user",
                content="Set the radio volume to 25.",
            ),
            MemoryMessage(
                id=3,
                role="user",
                content="Set the seat heating level to 2.",
            ),
        ]
    )

    assert first.matches[0].capability_id == "audio.volume"
    assert first.matches[0].storage_predicate == "audio_volume"
    assert first.matches[0].prompt_payload["allowed_targets"] == [
        "other",
        "radio",
        "unspecified",
    ]
    assert len(embedding.requests[0]) == 3
    assert embedding.requests[1] == (messages[0].content,)
    assert second.metadata["embedded_document_count"] == 0
    assert [items[0].capability_id for items in per_message.matches] == [
        "audio.volume",
        "heating.level",
    ]
    assert per_message.metadata["per_message"] is True
    assert len(embedding.requests[2]) == 2


def test_vehicle_fact_ontology_rejects_source_or_argument_drift():
    schemas = [
        _tool(
            "carcontrol_radio_set_volume",
            "Set radio volume.",
            {"volume": {"type": "integer"}},
        )
    ]
    definition = _definition(
        schemas,
        {
            "carcontrol_radio_set_volume": {
                "capability": "audio.volume",
                "target": "radio",
                "arguments": {},
            }
        },
    )

    with pytest.raises(VehicleFactOntologyError, match="Argument coverage"):
        build_vehicle_fact_ontology(schemas, definition)

    definition["source_schema_sha256"] = "0" * 64
    with pytest.raises(VehicleFactOntologyError, match="schema hash"):
        build_vehicle_fact_ontology(schemas, definition)
