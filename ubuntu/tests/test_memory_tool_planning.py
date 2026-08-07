from __future__ import annotations

from palmclaw_ubuntu.memory_tool_planning import (
    build_memory_bundles,
    build_memory_tool_plan,
    choose_memory_tool_hint_result,
    render_joint_tool_candidate_rationale,
    select_joint_tool_candidates,
)
from palmclaw_ubuntu.models import (
    FactMemoryIdentity,
    FactMemoryRecord,
    FactQueryContext,
    MemoryEvidence,
    ToolDefinition,
)
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tool_memory_execution import (
    ToolMemoryExecutionHint,
    ToolMemoryHintResult,
)


def _record(
    record_id: str,
    predicate: str,
    value: object,
    *,
    bundle_id: str | None = "event:one",
    entity_id: str = "gary",
    applicability: dict[str, object] | None = None,
) -> FactMemoryRecord:
    return FactMemoryRecord(
        id=record_id,
        session_id="session",
        record_key=f"key:{record_id}",
        user_id="user",
        entity_id=entity_id,
        predicate=predicate,
        identity_conditions={},
        applicability=applicability or {},
        capability_hints=("media", predicate),
        value=value,
        memory_type="preference",
        status="active",
        confidence=0.9,
        version=1,
        supersedes_id=None,
        merged_into_id=None,
        created_at="2026-08-06T00:00:00+00:00",
        updated_at="2026-08-06T00:00:00+00:00",
        bundle_id=bundle_id,
    )


def _configure_media() -> ToolDefinition:
    return ToolDefinition(
        name="configure_media",
        description="Set the media source and media volume together.",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": ["bluetooth", "radio"]},
                "volume": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["source", "volume"],
            "additionalProperties": False,
        },
        timeout_seconds=1,
        side_effect="write",
    )


def test_bundle_keeps_entity_and_applicability_boundaries() -> None:
    bundles = build_memory_bundles(
        (
            _record("source", "source", "bluetooth"),
            _record("volume", "volume", 25),
            _record("night", "volume", 15, applicability={"time": "night"}),
            _record("other", "volume", 40, entity_id="patricia"),
            _record("legacy", "volume", 20, bundle_id=None),
        )
    )

    assert sorted(len(bundle.records) for bundle in bundles) == [1, 1, 1, 2]
    assert all(
        len({record.entity_id for record in bundle.records}) == 1
        for bundle in bundles
    )


def test_storage_assigns_same_bundle_to_same_event(tmp_path) -> None:
    with SQLiteRepository(tmp_path / "bundle.db") as repository:
        session = repository.create_session("bundle")
        turn_id = repository.create_turn(session.id)
        message_id = repository.append_message(
            session.id,
            "user",
            "Gary uses Bluetooth at volume 25.",
            turn_id=turn_id,
        )
        repository.finish_turn(turn_id, "completed", 1, "test")
        records = []
        for predicate, value in (("source", "bluetooth"), ("volume", 25)):
            records.append(
                repository.insert_fact_memory_record(
                    session_id=session.id,
                    identity=FactMemoryIdentity(
                        user_id="user",
                        entity_id="gary",
                        predicate=predicate,
                    ),
                    value=value,
                    memory_type="preference",
                    confidence=0.9,
                    version=1,
                    evidence=(
                        MemoryEvidence(
                            message_id=message_id,
                            quote="Gary uses Bluetooth at volume 25.",
                        ),
                    ),
                )
            )

    assert records[0].bundle_id == records[1].bundle_id
    assert records[0].bundle_id is not None


def test_joint_selector_uses_fact_and_route_with_bounded_schema() -> None:
    definition = _configure_media()
    candidates = select_joint_tool_candidates(
        query="Restore Gary's setup",
        bundles=build_memory_bundles(
            (
                _record("source", "source", "bluetooth"),
                _record("volume", "volume", 25),
            )
        ),
        definitions=(definition,),
        routed_tools=(definition.name,),
        max_tools=1,
        schema_token_budget=500,
    )

    assert [candidate.tool_name for candidate in candidates] == [
        "configure_media"
    ]
    assert set(candidates[0].sources) == {"fact", "route"}
    assert candidates[0].schema_tokens <= 500


def test_joint_selector_does_not_expand_from_fact_without_query_support() -> None:
    media = _configure_media()
    lighting = ToolDefinition(
        name="set_ambient_light_color",
        description="Set the ambient light color.",
        parameters={
            "type": "object",
            "properties": {"color": {"type": "string"}},
            "required": ["color"],
            "additionalProperties": False,
        },
        timeout_seconds=1,
    )

    candidates = select_joint_tool_candidates(
        query="Restore Gary's media setup",
        bundles=build_memory_bundles(
            (
                _record("source", "source", "bluetooth"),
                _record("color", "color", "blue"),
            )
        ),
        definitions=(media, lighting),
        routed_tools=(media.name,),
    )

    assert [candidate.tool_name for candidate in candidates] == [media.name]


def test_joint_selector_caps_adjacent_tools_per_domain() -> None:
    definitions = tuple(
        ToolDefinition(
            name=f"carcontrol_seat_set_{predicate}",
            description=f"Set seat {predicate.replace('_', ' ')}.",
            parameters={
                "type": "object",
                "properties": {predicate: {"type": "integer"}},
                "required": [predicate],
                "additionalProperties": False,
            },
            timeout_seconds=1,
        )
        for predicate in (
            "backrest_angle",
            "cushion_angle",
            "headrest_height",
            "leg_support_height",
            "vertical_position",
        )
    )
    candidates = select_joint_tool_candidates(
        query="Restore Gary's usual seat leg support height.",
        bundles=build_memory_bundles(
            tuple(
                _record(str(index), predicate, 50, bundle_id=f"event:{index}")
                for index, predicate in enumerate(
                    (
                        "backrest_angle",
                        "cushion_angle",
                        "headrest_height",
                        "leg_support_height",
                        "vertical_position",
                    )
                )
            )
        ),
        definitions=definitions,
        routed_tools=tuple(definition.name for definition in definitions[:2]),
        max_tools=12,
    )

    assert len(candidates) == 3
    assert candidates[0].tool_name == "carcontrol_seat_set_leg_support_height"
    rationale = render_joint_tool_candidate_rationale(candidates)
    assert "leg" in rationale
    assert "support" in rationale
    assert "leg_support_height" in rationale


def test_multi_fact_bundle_completes_one_schema_valid_call() -> None:
    plan = build_memory_tool_plan(
        query="Restore Gary's media setup",
        records=(
            _record("source", "source", "bluetooth"),
            _record("volume", "volume", 25),
        ),
        definitions=(_configure_media(),),
        query_context=FactQueryContext(entities=("gary",)),
    )

    assert plan.status == "ready"
    assert len(plan.calls) == 1
    assert plan.calls[0].arguments == {"source": "bluetooth", "volume": 25}
    assert set(plan.calls[0].record_ids) == {"source", "volume"}
    assert len(plan.desired_states) == 2


def test_hint_safety_gate_keeps_legacy_for_competing_single_fact_plan() -> None:
    fallback = ToolMemoryHintResult(
        hints=(
            ToolMemoryExecutionHint(
                record_id="volume",
                tool_name="set_media_volume",
                arguments={"volume": 25},
                confidence=0.9,
                tool_domain="media",
                topic="volume",
            ),
        ),
        rejections=(),
    )
    planned = ToolMemoryHintResult(
        hints=(
            ToolMemoryExecutionHint(
                record_id="volume",
                tool_name="configure_media",
                arguments={"volume": 25},
                confidence=0.9,
                tool_domain="joint_plan",
                topic="desired_state",
                record_ids=("volume",),
            ),
        ),
        rejections=(),
    )

    assert choose_memory_tool_hint_result(planned, fallback) is fallback


def test_hint_safety_gate_accepts_complete_multi_fact_aggregation() -> None:
    fallback = ToolMemoryHintResult(
        hints=tuple(
            ToolMemoryExecutionHint(
                record_id=record_id,
                tool_name=f"set_media_{record_id}",
                arguments={record_id: value},
                confidence=0.9,
                tool_domain="media",
                topic=record_id,
            )
            for record_id, value in (("source", "bluetooth"), ("volume", 25))
        ),
        rejections=(),
    )
    planned = ToolMemoryHintResult(
        hints=(
            ToolMemoryExecutionHint(
                record_id="source",
                tool_name="configure_media",
                arguments={"source": "bluetooth", "volume": 25},
                confidence=0.9,
                tool_domain="joint_plan",
                topic="desired_state",
                record_ids=("source", "volume"),
            ),
        ),
        rejections=(),
    )

    assert choose_memory_tool_hint_result(planned, fallback) is planned


def test_joint_plan_does_not_emit_competing_calls_for_claimed_facts() -> None:
    set_volume = ToolDefinition(
        name="set_media_volume",
        description="Set media volume.",
        parameters={
            "type": "object",
            "properties": {"volume": {"type": "integer"}},
            "required": ["volume"],
            "additionalProperties": False,
        },
        timeout_seconds=1,
    )
    plan = build_memory_tool_plan(
        query="Restore Gary's media source and volume",
        records=(
            _record("source", "source", "bluetooth"),
            _record("volume", "volume", 25),
        ),
        definitions=(_configure_media(), set_volume),
        query_context=FactQueryContext(entities=("gary",)),
    )

    assert [call.tool_name for call in plan.calls] == ["configure_media"]


def test_conflicting_bundle_arguments_abstain() -> None:
    plan = build_memory_tool_plan(
        query="Restore Gary's media setup",
        records=(
            _record("source-a", "source", "bluetooth"),
            _record("source-b", "source", "radio"),
            _record("volume", "volume", 25),
        ),
        definitions=(_configure_media(),),
        query_context=FactQueryContext(entities=("gary",)),
    )

    assert plan.status == "ambiguous"
    assert plan.calls == ()
    assert {item.code for item in plan.rejections} == {
        "conflicting_bundle_arguments"
    }


def test_relative_request_without_state_is_blocked() -> None:
    plan = build_memory_tool_plan(
        query="Raise Gary's media volume",
        records=(_record("volume", "volume", 25),),
        definitions=(_configure_media(),),
        query_context=FactQueryContext(entities=("gary",)),
    )

    assert plan.status == "blocked"
    assert plan.calls == ()
    assert "current_state" in plan.desired_states[0].unresolved_slots
