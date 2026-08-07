from __future__ import annotations

from dataclasses import replace

import pytest

from palmclaw_ubuntu.models import (
    MemoryEvidence,
    ModelUsage,
    ToolDefinition,
    ToolMemoryIdentity,
    ToolResult,
)
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tool_memory_schema import (
    normalize_tool_memory_identity,
    split_tool_memory_conditions,
    tool_memory_identity_family_key,
    tool_memory_record_key,
)
from palmclaw_ubuntu.tools import ToolExecutionContext, ToolRegistry


class _SchemaOnlyTool:
    def __init__(self, definition: ToolDefinition):
        self.definition = definition

    def run(self, arguments, context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(tool_call_id="", content="unused")


def _definition(name: str, properties, required=()):
    return ToolDefinition(
        name=name,
        description="Schema-only test tool",
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
        timeout_seconds=1,
    )


def test_tool_registry_builds_memory_ontology_from_tool_schema():
    registry = ToolRegistry(
        [
            _SchemaOnlyTool(
                _definition(
                    "carcontrol_airConditioner_set_temperature",
                    {
                        "zone": {"type": "string"},
                        "temperature": {"type": "number"},
                    },
                    required=("zone", "temperature"),
                )
            ),
            _SchemaOnlyTool(
                _definition(
                    "file_read",
                    {"path": {"type": "string"}},
                    required=("path",),
                )
            ),
            _SchemaOnlyTool(
                _definition(
                    "carcontrol_navigation_navigate_to",
                    {
                        "destination": {"type": "string"},
                        "route_preference": {"type": "string"},
                    },
                    required=("destination",),
                )
            ),
        ]
    )

    ontology = registry.memory_ontology()
    climate = ontology.definition(
        "carcontrol_airConditioner_set_temperature"
    )
    file_read = ontology.definition("file_read")

    navigation = ontology.definition("carcontrol_navigation_navigate_to")
    assert ontology.domains == ("air_conditioner", "file", "navigation")
    assert climate is not None
    assert climate.namespace == "carcontrol"
    assert climate.domain == "air_conditioner"
    assert climate.action == "set"
    assert climate.topic == "temperature"
    assert [(slot.name, slot.path, slot.required) for slot in climate.slots] == [
        ("temperature", "air_conditioner.temperature.temperature", True),
        ("zone", "air_conditioner.temperature.zone", True),
    ]
    assert file_read is not None
    assert file_read.domain == "file"
    assert file_read.action == "read"
    assert file_read.topic == "read"
    assert navigation is not None
    assert navigation.domain == "navigation"
    assert navigation.action == "navigate"
    assert navigation.topic == "destination"


def test_record_key_is_stable_and_excludes_value():
    first = ToolMemoryIdentity(
        user_id=" Justin Martinez ",
        tool_domain="AirConditioner",
        topic="Ventilation Speed",
        scope="Conditional",
        scope_key="Vehicle 7",
        conditions={
            "Weather": " Hot ",
            "companions": ["Patricia", "Gary"],
        },
    )
    reordered = ToolMemoryIdentity(
        user_id="justin_martinez",
        tool_domain="air_conditioner",
        topic="ventilation_speed",
        scope="conditional",
        scope_key="vehicle_7",
        conditions={
            "companions": ["Gary", "Patricia"],
            "weather": "hot",
        },
    )

    assert tool_memory_record_key(first) == tool_memory_record_key(reordered)
    assert normalize_tool_memory_identity(first).conditions == {
        "companions": ["gary", "patricia"],
        "weather": "hot",
    }
    assert tool_memory_record_key(first) != tool_memory_record_key(
        replace(reordered, conditions={"weather": "cold"})
    )
    assert tool_memory_identity_family_key(
        first
    ) == tool_memory_identity_family_key(
        replace(
            reordered,
            conditions={
                "companions": ["Gary", "Patricia"],
                "weather": "cold",
            },
        )
    )
    assert split_tool_memory_conditions(first.conditions) == (
        {"companions": ["gary", "patricia"]},
        {"weather": "hot"},
    )


def test_record_identity_rejects_unsupported_scope():
    identity = ToolMemoryIdentity(
        user_id="user",
        tool_domain="seat",
        topic="height",
        scope="unknown",
        scope_key="user",
    )

    with pytest.raises(ValueError, match="Unsupported memory scope"):
        tool_memory_record_key(identity)


def test_record_identity_preserves_unicode_identifiers():
    identity = ToolMemoryIdentity(
        user_id="사용자 김",
        tool_domain="좌석",
        topic="통풍 단계",
        scope="global",
        scope_key="사용자 김",
    )

    normalized = normalize_tool_memory_identity(identity)

    assert normalized.user_id == "사용자_김"
    assert normalized.tool_domain == "좌석"
    assert normalized.topic == "통풍_단계"


def test_tool_memory_records_and_patch_audit_survive_restart(settings):
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    session = repository.create_session("Tool memory")
    turn_id = repository.create_turn(session.id)
    message_id = repository.append_message(
        session.id,
        "user",
        "On hot days use seat ventilation level 2",
        turn_id=turn_id,
    )
    repository.finish_turn(turn_id, "completed", 1, "seed")
    identity = ToolMemoryIdentity(
        user_id="Justin",
        tool_domain="seat",
        topic="ventilation_speed",
        scope="conditional",
        scope_key="Justin",
        conditions={"weather": "hot"},
    )
    record = repository.insert_tool_memory_record(
        session_id=session.id,
        identity=identity,
        value=2,
        memory_type="preference",
        confidence=0.96,
        version=1,
        evidence=(
            MemoryEvidence(
                message_id=message_id,
                quote="seat ventilation level 2",
            ),
        ),
    )
    run_id = repository.begin_memory_patch_run(
        session_id=session.id,
        source_turn_id=turn_id,
        backend="openai",
        model_id="memory-model",
        prompt_version="patch-v1",
        schema_version="tool-memory-patch-v1",
        status="running",
    )
    proposal_id = repository.record_memory_patch_proposal(
        run_id=run_id,
        operation="ADD",
        patch={"identity": {"topic": "ventilation_speed"}, "value": 2},
        evidence=({"message_id": message_id},),
        confidence=0.96,
        idempotency_key=f"{turn_id}:0",
        proposed_record_key=record.record_key,
    )
    repeated_id = repository.record_memory_patch_proposal(
        run_id=run_id,
        operation="ADD",
        patch={"identity": {"topic": "ventilation_speed"}, "value": 2},
        evidence=({"message_id": message_id},),
        confidence=0.96,
        idempotency_key=f"{turn_id}:0",
        proposed_record_key=record.record_key,
    )
    repository.resolve_memory_patch_proposal(
        proposal_id,
        status="applied",
        result_record_id=record.id,
    )
    repository.finish_memory_patch_run(
        run_id,
        status="completed",
        usage=ModelUsage(input_tokens=20, output_tokens=5, total_tokens=25),
    )
    repository.close()

    with SQLiteRepository(settings.database_path) as reopened:
        records = reopened.list_tool_memory_records(
            session.id,
            user_id="JUSTIN",
            tool_domain="Seat",
            topic="Ventilation Speed",
        )
        active = reopened.active_tool_memory_record(record.record_key)
        trace = reopened.memory_patch_trace(run_id)
        source = reopened._connection.execute(
            """
            SELECT evidence_text
            FROM tool_memory_record_sources
            WHERE record_id = ?
            """,
            (record.id,),
        ).fetchone()

    assert repeated_id == proposal_id
    assert records == [record]
    assert active == record
    assert source["evidence_text"] == "seat ventilation level 2"
    assert trace["run"]["status"] == "completed"
    assert trace["run"]["usage"]["input_tokens"] == 20
    assert trace["proposals"][0]["operation"] == "ADD"
    assert trace["proposals"][0]["status"] == "applied"
    assert trace["proposals"][0]["patch"]["value"] == 2


def test_tool_memory_active_record_is_unique_per_record_key(settings):
    settings.ensure_directories()
    identity = ToolMemoryIdentity(
        user_id="user",
        tool_domain="navigation",
        topic="volume",
        scope="global",
        scope_key="user",
    )
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Unique active")
        repository.insert_tool_memory_record(
            session_id=session.id,
            identity=identity,
            value=38,
            memory_type="preference",
            confidence=1,
            version=1,
        )

        with pytest.raises(Exception, match="UNIQUE constraint failed"):
            repository.insert_tool_memory_record(
                session_id=session.id,
                identity=identity,
                value=80,
                memory_type="preference",
                confidence=1,
                version=2,
            )
