from __future__ import annotations

from dataclasses import replace

import pytest

from palmclaw_ubuntu.application import create_runtime
from palmclaw_ubuntu.memory_router import (
    SchemaRoute,
    ToolMemoryRetriever,
    ToolSchemaRouter,
)
from palmclaw_ubuntu.models import (
    AgentResponse,
    EmbeddingResponse,
    ModelUsage,
    ToolDefinition,
    ToolMemoryIdentity,
)
from palmclaw_ubuntu.providers import (
    FakeEmbeddingModel,
    ScriptedAgentModel,
)
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tool_memory_schema import build_tool_memory_ontology


class FailingEmbeddingModel:
    backend = "fake"
    model_id = "failing-embedding"
    dimensions = 16

    def embed(self, texts):
        del texts
        raise RuntimeError("embedding unavailable")


class ScriptedRouteClassifier:
    def __init__(self):
        self.requests = []

    def classify(self, query, ontology):
        self.requests.append((query, ontology))
        return (
            SchemaRoute(
                domain="file",
                topics=("write",),
                tool_names=("file_write",),
                score=1.0,
                matched_terms=(),
            ),
        )


class SemanticRouteEmbeddingModel:
    backend = "fake"
    model_id = "semantic-route"
    dimensions = 3

    def __init__(self):
        self.requests = []

    def embed(self, texts):
        self.requests.append(tuple(texts))
        vectors = []
        for text in texts:
            folded = text.casefold()
            if (
                "calming cabin" in folded
                or "soothing interior" in folded
                or "ambient color" in folded
                or "ambient light" in folded
            ):
                vectors.append((1.0, 0.0, 0.0))
            elif "ventilation" in folded:
                vectors.append((0.0, 1.0, 0.0))
            else:
                vectors.append((0.0, 0.0, 1.0))
        return EmbeddingResponse(
            vectors=tuple(vectors),
            usage=ModelUsage(input_tokens=len(texts)),
            response_id=f"semantic-{len(self.requests)}",
        )


def _definition(
    name: str,
    properties: dict[str, dict[str, object]],
    *,
    description: str = "test tool",
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        timeout_seconds=1,
    )


def _ontology():
    return build_tool_memory_ontology(
        [
            _definition(
                "carcontrol_seat_set_ventilation",
                {"level": {"type": "integer"}},
                description="Set seat ventilation airflow level.",
            ),
            _definition(
                "carcontrol_navigation_navigate_to",
                {"destination": {"type": "string"}},
                description="Navigate to a destination.",
            ),
            _definition(
                "carcontrol_light_set_ambient_color",
                {"color": {"type": "string"}},
                description="Set interior ambient light color.",
            ),
            _definition(
                "file_write",
                {"path": {"type": "string"}},
                description="Write content to a file.",
            ),
        ]
    )


def _retriever(
    repository: SQLiteRepository,
    *,
    mode: str = "bm25",
    top_k: int = 5,
    token_budget: int = 1_000,
    embedding_model=None,
) -> ToolMemoryRetriever:
    return ToolMemoryRetriever(
        repository=repository,
        router=ToolSchemaRouter(_ontology()),
        embedding_model=embedding_model or FakeEmbeddingModel(dimensions=16),
        user_id="default_user",
        vehicle_id="vehicle_1",
        mode=mode,
        top_k=top_k,
        token_budget=token_budget,
        model_timeout_seconds=1,
    )


def _record(
    repository: SQLiteRepository,
    session_id: str,
    *,
    domain: str,
    topic: str,
    value,
    scope: str = "global",
    scope_key: str = "default_user",
    conditions=None,
    version: int = 1,
):
    return repository.insert_tool_memory_record(
        session_id=session_id,
        identity=ToolMemoryIdentity(
            user_id="default_user",
            tool_domain=domain,
            topic=topic,
            scope=scope,
            scope_key=scope_key,
            conditions=conditions or {},
        ),
        value=value,
        memory_type="preference",
        confidence=0.95,
        version=version,
    )


def test_schema_router_uses_tool_domain_topic_and_safe_empty_fallback():
    router = ToolSchemaRouter(_ontology())

    seat = router.route("On hot days set seat ventilation level 2")
    empty = router.route("Tell me a joke about databases")

    assert seat.routes[0].domain == "seat"
    assert {"ventilation", "level"} <= set(seat.routes[0].topics)
    assert seat.classifier_used is False
    assert empty.routes == ()
    assert empty.reason == "no_schema_match"


def test_schema_router_uses_classifier_only_for_missing_or_ambiguous_route():
    classifier = ScriptedRouteClassifier()
    router = ToolSchemaRouter(_ontology(), classifier=classifier)

    lexical = router.route("set seat ventilation level")
    fallback = router.route("please remember my preference")

    assert lexical.classifier_used is False
    assert fallback.classifier_used is True
    assert fallback.routes[0].domain == "file"
    assert fallback.routes[0].topics == ("write",)
    assert len(classifier.requests) == 1


def test_schema_router_ignores_stopword_only_tool_name_matches():
    router = ToolSchemaRouter(_ontology())

    decision = router.route("Patricia drove to a site")

    assert decision.routes == ()
    assert decision.reason == "no_schema_match"


def test_semantic_router_prioritizes_direct_quoted_request_and_caches_tools():
    embedding = SemanticRouteEmbeddingModel()
    router = ToolSchemaRouter(
        _ontology(),
        embedding_model=embedding,
        model_timeout_seconds=1,
    )
    query = (
        "Gary got into the driver's seat with Patricia. "
        "Gary said: 'Restore that soothing interior feeling.'"
    )

    first = router.route(query)
    second = router.route("Please restore that soothing interior feeling")

    assert first.routes[0].domain == "light"
    assert first.routes[0].tool_names == (
        "carcontrol_light_set_ambient_color",
    )
    assert all(route.domain != "seat" for route in first.routes)
    assert first.routing_query == "Restore that soothing interior feeling."
    assert first.semantic_used is True
    assert first.reason == "hybrid_schema_match"
    assert len(embedding.requests[0]) == len(_ontology().tools) + 1
    assert len(embedding.requests[1]) == 1
    assert second.routes[0].domain == "light"


def test_semantic_route_trace_and_query_vector_are_reused_for_ranking(settings):
    settings.ensure_directories()
    embedding = SemanticRouteEmbeddingModel()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Semantic Tool route")
        record = _record(
            repository,
            session.id,
            domain="light",
            topic="ambient_color",
            value="green",
        )
        retriever = ToolMemoryRetriever(
            repository=repository,
            router=ToolSchemaRouter(
                _ontology(),
                embedding_model=embedding,
                model_timeout_seconds=1,
                repository=repository,
            ),
            embedding_model=embedding,
            user_id="default_user",
            vehicle_id="vehicle_1",
            mode="hybrid",
            top_k=5,
            token_budget=1_000,
            model_timeout_seconds=1,
        )

        result = retriever.retrieve(
            session.id,
            turn_id=None,
            query="Restore that soothing interior feeling",
        )
        trace = repository.tool_memory_retrieval_trace(result.run_id)
        restarted_router = ToolSchemaRouter(
            _ontology(),
            embedding_model=embedding,
            model_timeout_seconds=1,
            repository=repository,
        )
        restarted = restarted_router.route(
            "Restore that soothing interior feeling"
        )
        cached_schema_vectors = repository._connection.execute(
            "SELECT COUNT(*) FROM tool_schema_embeddings"
        ).fetchone()[0]
        roles = [
            row["role"]
            for row in repository._connection.execute(
                """
                SELECT role
                FROM model_calls
                ORDER BY created_at, id
                """
            ).fetchall()
        ]

    assert [item.id for item in result.records] == [record.id]
    assert trace["run"]["route"]["semantic"]["used"] is True
    assert roles == ["tool_schema_embedding", "tool_memory_embedding"]
    assert len(embedding.requests[0]) == len(_ontology().tools) + 1
    assert len(embedding.requests[1]) == 1
    assert len(embedding.requests[2]) == 1
    assert restarted.routes[0].domain == "light"
    assert cached_schema_vectors == len(_ontology().tools)


def test_schema_router_can_use_exact_tool_names_without_argument_leakage():
    router = ToolSchemaRouter(_ontology())

    decision = router.route_tool_names(
        ("carcontrol_navigation_navigate_to",)
    )

    assert decision.reason == "oracle_tool_names"
    assert decision.classifier_used is False
    assert decision.routes[0].domain == "navigation"
    assert decision.routes[0].tool_names == (
        "carcontrol_navigation_navigate_to",
    )
    assert set(decision.routes[0].topics) == {"destination"}
    assert decision.routes[0].matched_terms == ()


def test_schema_router_rejects_unknown_oracle_tool_name():
    router = ToolSchemaRouter(_ontology())

    with pytest.raises(ValueError, match="Unknown Oracle Tool names"):
        router.route_tool_names(("carcontrol_unknown",))


def test_retrieval_isolates_domain_scope_and_conditions(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Tool retrieval")
        hot = _record(
            repository,
            session.id,
            domain="seat",
            topic="ventilation",
            value=2,
            scope="conditional",
            conditions={"weather": "hot"},
        )
        cold = _record(
            repository,
            session.id,
            domain="seat",
            topic="ventilation",
            value=1,
            scope="conditional",
            conditions={"weather": "cold"},
        )
        wrong_session = _record(
            repository,
            session.id,
            domain="seat",
            topic="level",
            value=3,
            scope="session",
            scope_key="another_session",
        )
        wrong_global = _record(
            repository,
            session.id,
            domain="seat",
            topic="level",
            value=4,
            scope="global",
            scope_key="another_user",
        )
        navigation = _record(
            repository,
            session.id,
            domain="navigation",
            topic="destination",
            value="home",
        )
        repository.insert_tool_memory_record(
            session_id=session.id,
            identity=ToolMemoryIdentity(
                user_id="other_user",
                tool_domain="seat",
                topic="ventilation",
                scope="global",
                scope_key="other_user",
            ),
            value=9,
            memory_type="preference",
            confidence=1,
            version=1,
        )
        retriever = _retriever(repository)

        result = retriever.retrieve(
            session.id,
            turn_id=None,
            query="On hot days set seat ventilation level",
        )
        trace = repository.tool_memory_retrieval_trace(result.run_id)

    assert [record.id for record in result.records] == [hot.id]
    assert navigation.id not in {
        candidate["record_id"] for candidate in trace["candidates"]
    }
    exclusions = {
        candidate["record_id"]: candidate["exclusion_reason"]
        for candidate in trace["candidates"]
    }
    assert exclusions[cold.id] == "condition_mismatch"
    assert exclusions[wrong_session.id] == "scope_mismatch"
    assert exclusions[wrong_global.id] == "scope_mismatch"
    assert trace["run"]["candidate_count"] == 4
    assert trace["run"]["selected_tokens"] > 0


def test_condition_matching_supports_learned_name_aliases_and_time_buckets(
    settings,
):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Flexible conditions")
        patricia_day = _record(
            repository,
            session.id,
            domain="seat",
            topic="ventilation",
            value=44,
            scope="conditional",
            conditions={"person": "Patricia Garcia", "time": "daytime"},
        )
        patricia_night = _record(
            repository,
            session.id,
            domain="seat",
            topic="ventilation",
            value=3,
            scope="conditional",
            conditions={"person": "Patricia Garcia", "time": "night"},
        )
        retriever = _retriever(repository, mode="full")

        result = retriever.retrieve(
            session.id,
            turn_id=None,
            query=(
                "At 12:00 PM Patricia got into the driver's seat and needs "
                "the headrest adjusted."
            ),
            oracle_tool_names=("carcontrol_seat_set_ventilation",),
        )
        trace = repository.tool_memory_retrieval_trace(result.run_id)

    assert [record.id for record in result.records] == [patricia_day.id]
    exclusions = {
        candidate["record_id"]: candidate["exclusion_reason"]
        for candidate in trace["candidates"]
    }
    assert exclusions[patricia_night.id] == "condition_mismatch"


def test_free_text_context_is_ranked_instead_of_hard_filtered(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Semantic context")
        record = _record(
            repository,
            session.id,
            domain="seat",
            topic="ventilation",
            value=2,
            scope="conditional",
            conditions={
                "person": "Patricia Garcia",
                "context": "driving through the industrial district",
            },
        )
        retriever = _retriever(repository, mode="full")

        result = retriever.retrieve(
            session.id,
            turn_id=None,
            query="Patricia is approaching the smog zone.",
            oracle_tool_names=("carcontrol_seat_set_ventilation",),
        )

    assert [item.id for item in result.records] == [record.id]


def test_hybrid_retrieval_caches_record_embeddings_and_traces_usage(settings):
    settings.ensure_directories()
    embedding = FakeEmbeddingModel(dimensions=16)
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Tool hybrid")
        record = _record(
            repository,
            session.id,
            domain="file",
            topic="write",
            value="notes.txt",
        )
        retriever = _retriever(
            repository,
            mode="hybrid",
            embedding_model=embedding,
        )

        first = retriever.retrieve(
            session.id,
            turn_id=None,
            query="save the report file",
        )
        second = retriever.retrieve(
            session.id,
            turn_id=None,
            query="write another file report",
        )
        stored = repository.tool_memory_embeddings(
            [record.id],
            model_id=embedding.model_id,
            dimensions=embedding.dimensions,
        )
        calls = repository._connection.execute(
            """
            SELECT role, metadata_json
            FROM model_calls
            WHERE role = 'tool_memory_embedding'
            ORDER BY created_at
            """
        ).fetchall()

    assert first.records[0].id == second.records[0].id == record.id
    assert len(embedding.requests[0]) == 2
    assert len(embedding.requests[1]) == 1
    assert record.id in stored
    assert len(calls) == 2
    assert first.metadata["tool_memory_retrieval_latency_ms"] >= 0


def test_top_k_token_budget_and_no_route_are_hard_limits(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Tool limits")
        _record(
            repository,
            session.id,
            domain="seat",
            topic="ventilation",
            value=2,
        )
        _record(
            repository,
            session.id,
            domain="seat",
            topic="level",
            value=3,
        )
        top_one = _retriever(repository, mode="full", top_k=1)
        no_budget = _retriever(repository, mode="full", token_budget=1)

        limited = top_one.retrieve(
            session.id,
            turn_id=None,
            query="set seat ventilation level",
        )
        limited_trace = repository.tool_memory_retrieval_trace(limited.run_id)
        empty_budget = no_budget.retrieve(
            session.id,
            turn_id=None,
            query="set seat ventilation level",
        )
        no_route = top_one.retrieve(
            session.id,
            turn_id=None,
            query="tell me a joke",
        )

    assert len(limited.records) == 1
    assert any(
        candidate["exclusion_reason"] == "top_k"
        for candidate in limited_trace["candidates"]
    )
    assert empty_budget.records == ()
    assert no_route.records == ()
    assert no_route.metadata["tool_memory_retrieval_empty_reason"] == "no_route"


def test_retrieval_failure_uses_empty_memory_and_persists_failure_trace(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Tool failure")
        _record(
            repository,
            session.id,
            domain="file",
            topic="write",
            value="private-notes.txt",
        )
        retriever = _retriever(
            repository,
            mode="embedding",
            embedding_model=FailingEmbeddingModel(),
        )

        result = retriever.retrieve(
            session.id,
            turn_id=None,
            query="save report file",
        )
        trace = repository.tool_memory_retrieval_trace(result.run_id)

    assert result.content == ""
    assert result.records == ()
    assert result.metadata["tool_memory_retrieval_status"] == "failed"
    assert result.metadata["tool_memory_retrieval_selected_count"] == 0
    assert trace["run"]["status"] == "failed"
    assert trace["run"]["selected_count"] == 0
    assert "embedding unavailable" in trace["run"]["error"]


def test_interrupted_retrieval_is_marked_failed_on_recovery(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Tool recovery")
        run_id = repository.begin_tool_memory_retrieval(
            session_id=session.id,
            turn_id=None,
            query="save report file",
            user_id="default_user",
            mode="bm25",
            top_k=5,
            token_budget=500,
            route={"routes": []},
            classifier_used=False,
        )

        recovery = repository.recover_interrupted_execution()
        trace = repository.tool_memory_retrieval_trace(run_id)

    assert recovery["tool_memory_retrievals_interrupted"] == 1
    assert trace["run"]["status"] == "failed"
    assert trace["run"]["selected_count"] == 0
    assert "Process restarted" in trace["run"]["error"]


def test_agent_context_contains_only_compact_routed_tool_memory(settings):
    patched = replace(
        settings,
        tool_memory_retrieval_enabled=True,
        tool_memory_retrieval_mode="bm25",
        tool_memory_context_tokens=500,
        patch_memory_user_id="default_user",
        memory_trigger_messages=100,
    )
    model = ScriptedAgentModel([AgentResponse(content="done")])
    with create_runtime(patched, agent_model=model) as runtime:
        session = runtime.repository.create_session("Agent Tool memory")
        record = _record(
            runtime.repository,
            session.id,
            domain="file",
            topic="write",
            value="notes.txt",
        )
        _record(
            runtime.repository,
            session.id,
            domain="navigation",
            topic="destination",
            value="home",
        )

        result = runtime.turn_coordinator.run(
            session.id,
            "Save this report file",
        )
        request_messages = model.requests[0][0]
        model_call = runtime.repository._connection.execute(
            """
            SELECT metadata_json
            FROM model_calls
            WHERE turn_id = ? AND role = 'agent'
            """,
            (result.turn_id,),
        ).fetchone()

    system = request_messages[0].content
    assert "[Retrieved Tool Memory" in system
    assert "notes.txt" in system
    assert "destination" not in system
    assert "evidence" not in system
    assert record.id in system
    assert '"tool_memory_retrieval_selected_count":1' in model_call["metadata_json"]


def test_agent_context_contains_schema_validated_memory_tool_hint(settings):
    patched = replace(
        settings,
        tool_memory_retrieval_enabled=True,
        tool_memory_retrieval_mode="bm25",
        tool_memory_context_tokens=500,
        patch_memory_user_id="default_user",
        memory_trigger_messages=100,
    )
    model = ScriptedAgentModel([AgentResponse(content="done")])
    with create_runtime(patched, agent_model=model) as runtime:
        session = runtime.repository.create_session("Agent Tool hint")
        record = _record(
            runtime.repository,
            session.id,
            domain="file",
            topic="read",
            value={"path": "notes.txt"},
        )

        result = runtime.turn_coordinator.run(
            session.id,
            "Read the notes file",
        )
        system = model.requests[0][0][0].content
        model_call = runtime.repository._connection.execute(
            """
            SELECT metadata_json
            FROM model_calls
            WHERE turn_id = ? AND role = 'agent'
            """,
            (result.turn_id,),
        ).fetchone()

    assert "[Schema-validated candidate Tool calls]" in system
    assert '"tool_name":"file_read"' in system
    assert '"arguments":{"path":"notes.txt"}' in system
    assert record.id in system
    assert '"tool_memory_execution_hint_count":1' in model_call["metadata_json"]
