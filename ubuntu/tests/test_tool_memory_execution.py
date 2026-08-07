from __future__ import annotations

from palmclaw_ubuntu.models import (
    ToolDefinition,
    ToolMemoryRecord,
)
from palmclaw_ubuntu.tool_memory_execution import (
    build_tool_memory_execution_hints,
    routed_tool_names,
)


def _definition(
    name: str,
    properties: dict[str, dict[str, object]],
    required: tuple[str, ...],
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="test",
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
        timeout_seconds=1,
    )


def _record(
    *,
    record_id: str = "record-1",
    domain: str,
    topic: str,
    value,
    conditions=None,
) -> ToolMemoryRecord:
    return ToolMemoryRecord(
        id=record_id,
        session_id="session",
        record_key=f"key-{record_id}",
        user_id="default_user",
        tool_domain=domain,
        topic=topic,
        scope="global",
        scope_key="default_user",
        conditions=conditions or {},
        value=value,
        memory_type="preference",
        status="active",
        confidence=0.95,
        version=1,
        supersedes_id=None,
        merged_into_id=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


def test_execution_hint_maps_record_to_one_schema_valid_call():
    definitions = (
        _definition(
            "carcontrol_light_set_reading_light_brightness",
            {
                "light": {"type": "string"},
                "brightness": {"type": "integer"},
            },
            ("light", "brightness"),
        ),
        _definition(
            "carcontrol_light_set_ambient_color",
            {"color": {"type": "string"}},
            ("color",),
        ),
    )
    record = _record(
        domain="light",
        topic="reading_light_brightness",
        value={
            "light": "driver",
            "brightness": 8,
            "ignored_note": "not a Tool argument",
        },
    )

    result = build_tool_memory_execution_hints((record,), definitions)

    assert len(result.hints) == 1
    assert result.rejections == ()
    assert result.hints[0].tool_name == (
        "carcontrol_light_set_reading_light_brightness"
    )
    assert result.hints[0].arguments == {
        "light": "driver",
        "brightness": 8,
    }


def test_execution_hint_uses_explicit_condition_for_required_target():
    definition = _definition(
        "carcontrol_seat_set_heating_level",
        {
            "seat": {"type": "string"},
            "level": {"type": "integer"},
        },
        ("seat", "level"),
    )
    record = _record(
        domain="seat",
        topic="heating_level",
        value={"level": 2},
        conditions={"seat": "driver", "person": "Gary"},
    )

    result = build_tool_memory_execution_hints((record,), (definition,))

    assert result.hints[0].arguments == {"level": 2, "seat": "driver"}


def test_execution_hint_rejects_missing_required_arguments():
    definition = _definition(
        "carcontrol_seat_set_heating_level",
        {
            "seat": {"type": "string"},
            "level": {"type": "integer"},
        },
        ("seat", "level"),
    )
    record = _record(
        domain="seat",
        topic="heating_level",
        value={"level": 2},
        conditions={"person": "Gary"},
    )

    result = build_tool_memory_execution_hints((record,), (definition,))

    assert result.hints == ()
    assert result.rejections[0].code == "incomplete_or_invalid_arguments"


def test_routed_tool_names_reads_only_retrieval_selector_metadata():
    names = routed_tool_names(
        {
            "tool_memory_route": {
                "routes": [
                    {"tool_names": ["tool_a", "tool_b"]},
                    {"tool_names": ["tool_b", "tool_c"]},
                ]
            },
            "reference_calls": [{"name": "gold_tool"}],
        }
    )

    assert names == ("tool_a", "tool_b", "tool_c")
