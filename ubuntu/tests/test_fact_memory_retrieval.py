from __future__ import annotations

from palmclaw_ubuntu.fact_memory_retrieval import (
    FactMemoryRetriever,
    interpret_fact_query,
)
from palmclaw_ubuntu.memory_router import ToolSchemaRouter
from palmclaw_ubuntu.memory_tool_planning import build_memory_bundles
from palmclaw_ubuntu.models import (
    FactMemoryIdentity,
    FactMemoryRecord,
    FactQueryContext,
    MemoryEvidence,
    ToolDefinition,
)
from palmclaw_ubuntu.providers import FakeEmbeddingModel
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tool_memory_execution import (
    build_fact_memory_execution_hints,
)
from palmclaw_ubuntu.tool_memory_schema import build_tool_memory_ontology


def _definition(
    name: str,
    properties: dict[str, dict[str, object]],
    required: tuple[str, ...],
    *,
    description: str,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
        timeout_seconds=1,
    )


def _definitions() -> tuple[ToolDefinition, ...]:
    return (
        _definition(
            "carcontrol_seat_set_headrest_height",
            {
                "seat": {"type": "string"},
                "value": {"type": "integer"},
            },
            ("seat", "value"),
            description="Set headrest height.",
        ),
        _definition(
            "carcontrol_light_set_reading_light_brightness",
            {
                "light": {"type": "string"},
                "brightness": {"type": "integer"},
            },
            ("light", "brightness"),
            description="Set reading light brightness.",
        ),
    )


def _insert(
    repository: SQLiteRepository,
    session_id: str,
    *,
    entity: str,
    predicate: str,
    value: object,
    applicability: dict[str, object] | None = None,
    hints: tuple[str, ...] = (),
    evidence: tuple[MemoryEvidence, ...] = (),
) -> FactMemoryRecord:
    return repository.insert_fact_memory_record(
        session_id=session_id,
        identity=FactMemoryIdentity(
            user_id="default_user",
            entity_id=entity,
            predicate=predicate,
            applicability=applicability or {},
        ),
        value=value,
        memory_type="preference",
        confidence=0.95,
        version=1,
        capability_hints=hints,
        evidence=evidence,
    )


def _retriever(repository: SQLiteRepository, *, top_k: int = 3):
    definitions = _definitions()
    embedding = FakeEmbeddingModel(dimensions=32)
    return FactMemoryRetriever(
        repository=repository,
        router=ToolSchemaRouter(
            build_tool_memory_ontology(definitions),
            embedding_model=embedding,
            model_timeout_seconds=1,
            repository=repository,
        ),
        embedding_model=embedding,
        user_id="default_user",
        mode="hybrid",
        top_k=top_k,
        token_budget=1_000,
        model_timeout_seconds=1,
    )


def test_memory_first_keeps_fact_when_tool_router_has_no_route(tmp_path):
    repository = SQLiteRepository(tmp_path / "memory.db")
    session = repository.create_session("Fact retrieval")
    record = _insert(
        repository,
        session.id,
        entity="Patricia",
        predicate="headrest_height",
        value=44,
        hints=("seat",),
    )
    retriever = _retriever(repository)

    result = retriever.retrieve(
        session.id,
        turn_id=None,
        query="Patricia wants her usual comfort setting restored.",
    )

    assert result.metadata["tool_memory_route"]["routes"] == []
    assert result.records[0].id == record.id
    assert result.metadata["route_is_hard_filter"] is False
    assert repository.fact_memory_retrieval_trace(result.run_id)["candidates"]
    repository.close()


def test_entity_role_and_time_condition_prioritize_compatible_fact(tmp_path):
    repository = SQLiteRepository(tmp_path / "memory.db")
    session = repository.create_session("Fact roles")
    patricia = _insert(
        repository,
        session.id,
        entity="Patricia Garcia",
        predicate="reading_light_brightness",
        value=8,
        applicability={"time": "after dusk"},
        hints=("light",),
    )
    _insert(
        repository,
        session.id,
        entity="Nicole Adams",
        predicate="reading_light_brightness",
        value=5,
        applicability={"time": "morning"},
        hints=("light",),
    )

    result = _retriever(repository, top_k=1).retrieve(
        session.id,
        turn_id=None,
        query=(
            "At 8:00 PM Patricia Garcia sat in the driver's seat while "
            "Nicole Adams rode along. Restore Patricia's reading light."
        ),
    )

    assert result.records == (patricia,)
    assert result.query_context.entity_roles["patricia_garcia"] == "driver"
    assert result.query_context.time_of_day == "evening"
    repository.close()


def test_retrieval_attaches_unselected_records_from_same_source_event(tmp_path):
    repository = SQLiteRepository(tmp_path / "memory.db")
    session = repository.create_session("Fact bundles")
    turn_id = repository.create_turn(session.id)
    message_id = repository.append_message(
        session.id,
        "user",
        "Gary uses Bluetooth at volume 25.",
        turn_id=turn_id,
    )
    repository.finish_turn(turn_id, "completed", 1, "test")
    evidence = (
        MemoryEvidence(
            message_id=message_id,
            quote="Gary uses Bluetooth at volume 25.",
        ),
    )
    source = _insert(
        repository,
        session.id,
        entity="Gary",
        predicate="source",
        value="bluetooth",
        hints=("media",),
        evidence=evidence,
    )
    volume = _insert(
        repository,
        session.id,
        entity="Gary",
        predicate="volume",
        value=25,
        hints=("media",),
        evidence=evidence,
    )

    result = _retriever(repository, top_k=1).retrieve(
        session.id,
        turn_id=None,
        query="Restore Gary's usual volume.",
    )

    assert result.records == (volume,)
    assert result.related_records == (source,)
    assert result.metadata["fact_memory_related_record_count"] == 1
    assert len(build_memory_bundles((*result.records, *result.related_records))) == 1
    repository.close()


def test_oracle_retrieval_prepends_reviewed_record_without_changing_database(
    tmp_path,
):
    repository = SQLiteRepository(tmp_path / "memory.db")
    session = repository.create_session("Oracle retrieval")
    patricia = _insert(
        repository,
        session.id,
        entity="Patricia",
        predicate="reading_light_brightness",
        value=8,
        hints=("light",),
    )
    nicole = _insert(
        repository,
        session.id,
        entity="Nicole",
        predicate="reading_light_brightness",
        value=5,
        hints=("light",),
    )
    retriever = _retriever(repository, top_k=1)
    baseline = retriever.retrieve(
        session.id,
        turn_id=None,
        query="Restore Patricia's reading light.",
    )

    oracle = retriever.apply_oracle_selection(
        baseline,
        session_id=session.id,
        task_id="task-1",
        record_status="record_present",
        record_keys=(
            {
                "entity_id": "Nicole",
                "predicate": "reading_light_brightness",
                "value": 5,
            },
        ),
    )

    assert baseline.records == (patricia,)
    assert oracle.records == (nicole,)
    assert oracle.metadata["oracle_retrieval_forced_count"] == 1
    assert oracle.metadata["oracle_retrieval_baseline_selected_ids"] == [
        patricia.id
    ]
    assert {
        record.id
        for record in repository.list_fact_memory_records(session.id)
    } == {patricia.id, nicole.id}
    repository.close()


def test_late_binding_maps_fact_entity_role_to_vehicle_argument():
    record = FactMemoryRecord(
        id="fact-1",
        session_id="session",
        record_key="key",
        user_id="default_user",
        entity_id="patricia",
        predicate="headrest_height",
        identity_conditions={},
        applicability={},
        capability_hints=("seat",),
        value=44,
        memory_type="preference",
        status="active",
        confidence=0.95,
        version=1,
        supersedes_id=None,
        merged_into_id=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    result = build_fact_memory_execution_hints(
        (record,),
        _definitions(),
        FactQueryContext(
            entities=("patricia",),
            entity_roles={"patricia": "driver"},
        ),
        query="Patricia is driving; restore her headrest.",
    )

    assert result.rejections == ()
    assert result.hints[0].tool_name == "carcontrol_seat_set_headrest_height"
    assert result.hints[0].arguments == {"seat": "driver", "value": 44}


def test_query_parser_resolves_passenger_and_first_person_driver_roles():
    passenger = interpret_fact_query(
        "Nicole Adams was in the passenger seat while Thomas Carter drove.",
        known_entities=("nicole_adams", "thomas_carter"),
    )
    driver = interpret_fact_query(
        "Rachel and Brian got in. Rachel sat in the driver's seat. "
        "She said: 'I'm driving today.'",
        known_entities=("rachel", "brian"),
    )

    assert passenger.entity_roles == {
        "nicole_adams": "passenger",
        "thomas_carter": "driver",
    }
    assert passenger.requester_entity is None
    assert driver.entity_roles["rachel"] == "driver"


def test_query_parser_prioritizes_passenger_who_makes_request():
    context = interpret_fact_query(
        "Nicole Adams sat in the passenger seat while Thomas Carter drove. "
        "Nicole Adams told Thomas Carter: 'Adjust my usual reading light.'",
        known_entities=("nicole_adams", "thomas_carter"),
    )

    assert context.entity_roles["nicole_adams"] == "passenger"
    assert context.entity_roles["thomas_carter"] == "driver"
    assert context.requester_entity == "nicole_adams"


def test_query_parser_does_not_expand_ambiguous_partial_name_alias():
    context = interpret_fact_query(
        "Patricia asked the car to restore the usual setting.",
        known_entities=("patricia", "patricia_garcia"),
    )

    assert context.entities == ("patricia",)
