from __future__ import annotations

import math
from dataclasses import replace

from palmclaw_ubuntu.application import create_runtime
from palmclaw_ubuntu.models import (
    AgentResponse,
    EmbeddingResponse,
    MemoryCandidate,
    MemoryEvidence,
    StructuredMemoryResponse,
)
from palmclaw_ubuntu.providers import (
    FakeMemoryModel,
    ScriptedAgentModel,
)


class _PreferenceStructuredModel:
    backend = "fake"
    model_id = "preference-memory"
    prompt_version = "memory-structured-v1"
    schema_version = "structured-v1"

    def __init__(self, *, scope: str = "session", invalid_quote: bool = False):
        self.scope = scope
        self.invalid_quote = invalid_quote
        self.existing_memory: list[str] = []

    def extract(self, messages, existing_memory):
        self.existing_memory.append(existing_memory)
        latest = next(
            message for message in reversed(messages) if message.role == "user"
        )
        value = "Korean" if "Korean" in latest.content else "English"
        return StructuredMemoryResponse(
            candidates=(
                MemoryCandidate(
                    subject="user",
                    predicate="preferred_language",
                    value=value,
                    scope=self.scope,
                    memory_type="preference",
                    confidence=0.95,
                    sensitivity="low",
                    evidence=(
                        MemoryEvidence(
                            message_id=latest.id,
                            quote=(
                                "not present" if self.invalid_quote else latest.content
                            ),
                        ),
                    ),
                ),
            )
        )


class _KeywordEmbeddingModel:
    backend = "fake"
    model_id = "keyword-embedding-v1"
    dimensions = 3

    def __init__(self):
        self.requests: list[tuple[str, ...]] = []

    def embed(self, texts):
        self.requests.append(tuple(texts))
        vectors = []
        for text in texts:
            normalized = text.casefold()
            if "all topics" in normalized:
                scale = 1 / math.sqrt(3)
                vector = (scale, scale, scale)
            elif "software" in normalized or "python" in normalized:
                vector = (1.0, 0.0, 0.0)
            elif "food" in normalized or "sushi" in normalized:
                vector = (0.0, 1.0, 0.0)
            elif "outdoor" in normalized or "hiking" in normalized:
                vector = (0.0, 0.0, 1.0)
            else:
                vector = (0.0, 0.0, 0.0)
            vectors.append(vector)
        return EmbeddingResponse(vectors=tuple(vectors))


class _FailingEmbeddingModel:
    backend = "fake"
    model_id = "failing-embedding"
    dimensions = 3

    def embed(self, texts):
        del texts
        raise RuntimeError("embedding unavailable")


def test_structured_memory_tracks_evidence_and_versions(settings):
    structured_model = _PreferenceStructuredModel()
    phase3_settings = replace(
        settings,
        memory_strategy="structured",
        retrieval_mode="bm25",
        memory_trigger_messages=2,
    )
    with create_runtime(
        phase3_settings,
        structured_memory_model=structured_model,
    ) as runtime:
        session = runtime.repository.create_session("Versions")
        first = runtime.turn_coordinator.run(
            session.id,
            "My preferred language is Korean",
        )
        second = runtime.turn_coordinator.run(
            session.id,
            "I changed my preferred language to English",
        )
        active = runtime.repository.active_structured_memories(session.id)
        versions = runtime.repository.memory_versions(active[0].fact_key)
        first_detail = runtime.repository.memory_detail(versions[0]["id"])
        second_detail = runtime.repository.memory_detail(versions[1]["id"])

    assert first.consolidation_run_id is not None
    assert second.consolidation_run_id is not None
    assert [(item["version"], item["status"], item["value"]) for item in versions] == [
        (1, "superseded", "Korean"),
        (2, "verified", "English"),
    ]
    assert versions[1]["supersedes_id"] == versions[0]["id"]
    assert active[0].value == "English"
    assert first_detail["sources"][0]["start_char"] == 0
    assert first_detail["sources"][0]["end_char"] == len(
        "My preferred language is Korean"
    )
    assert first_detail["embeddings"]
    assert second_detail["embeddings"]
    assert [event["status"] for event in first_detail["status_events"]] == [
        "proposed",
        "verified",
        "superseded",
    ]
    assert [event["status"] for event in second_detail["status_events"]] == [
        "proposed",
        "verified",
    ]
    assert "preferred_language: Korean" in structured_model.existing_memory[1]


def test_unsupported_evidence_is_rejected_but_watermark_completes(settings):
    phase3_settings = replace(
        settings,
        memory_strategy="structured",
        memory_trigger_messages=2,
    )
    with create_runtime(
        phase3_settings,
        structured_memory_model=_PreferenceStructuredModel(invalid_quote=True),
    ) as runtime:
        session = runtime.repository.create_session("Rejected")
        result = runtime.turn_coordinator.run(
            session.id,
            "My preferred language is Korean",
        )
        all_memories = runtime.repository.list_memories(
            session.id,
            include_superseded=True,
        )
        active = runtime.repository.active_structured_memories(session.id)
        consolidation = runtime.repository._connection.execute(
            "SELECT status FROM consolidation_runs WHERE id = ?",
            (result.consolidation_run_id,),
        ).fetchone()

    assert active == []
    assert all_memories[0]["status"] == "rejected"
    assert "unsupported_evidence_quote" in all_memories[0]["rejection_reason"]
    assert consolidation["status"] == "completed"


def test_duplicate_candidate_is_rejected_without_replacing_active_memory(
    settings,
):
    phase3_settings = replace(
        settings,
        memory_strategy="structured",
        memory_trigger_messages=2,
    )
    with create_runtime(
        phase3_settings,
        structured_memory_model=_PreferenceStructuredModel(),
    ) as runtime:
        session = runtime.repository.create_session("Duplicate")
        runtime.turn_coordinator.run(
            session.id,
            "My preferred language is Korean",
        )
        runtime.turn_coordinator.run(
            session.id,
            "My preferred language is Korean",
        )
        active = runtime.repository.active_structured_memories(session.id)
        versions = runtime.repository.memory_versions(active[0].fact_key)

    assert len(active) == 1
    assert [(item["version"], item["status"]) for item in versions] == [
        (1, "verified"),
        (2, "rejected"),
    ]
    assert versions[1]["rejection_reason"] == "duplicate_active_value"


def test_fact_key_normalizes_spaces_and_underscores(settings):
    phase3_settings = replace(settings, memory_strategy="structured")
    with create_runtime(phase3_settings) as runtime:
        session = runtime.repository.create_session("Fact key")
        spaced = MemoryCandidate(
            subject="User Profile",
            predicate="preferred editor",
            value="Neovim",
            scope="session",
            memory_type="preference",
            confidence=1.0,
            sensitivity="low",
            evidence=(),
        )
        underscored = replace(
            spaced,
            subject="user_profile",
            predicate="preferred_editor",
        )

        first_key = runtime.memory_engine._fact_key(session.id, spaced)
        second_key = runtime.memory_engine._fact_key(
            session.id,
            underscored,
        )

    assert first_key == second_key


def test_global_memory_is_visible_across_sessions(settings):
    phase3_settings = replace(
        settings,
        memory_strategy="structured",
        memory_trigger_messages=2,
    )
    with create_runtime(
        phase3_settings,
        structured_memory_model=_PreferenceStructuredModel(scope="global"),
    ) as runtime:
        first_session = runtime.repository.create_session("Origin")
        second_session = runtime.repository.create_session("Consumer")
        runtime.turn_coordinator.run(
            first_session.id,
            "My preferred language is Korean",
        )
        visible = runtime.repository.active_structured_memories(second_session.id)

    assert len(visible) == 1
    assert visible[0].scope == "global"
    assert visible[0].value == "Korean"


def _seed_retrieval_memories(runtime, session_id):
    repository = runtime.repository
    turn_id = repository.create_turn(session_id)
    message_id = repository.append_message(
        session_id,
        "user",
        "Python coding, sushi food, and hiking outdoor",
        turn_id=turn_id,
    )
    repository.finish_turn(turn_id, "completed", 1, "seed")
    run_id = repository.begin_consolidation(
        session_id,
        message_id,
        message_id,
        backend="fake",
        model_id="seed",
        prompt_version="seed-v1",
        schema_version="structured-v1",
    )
    candidates = [
        MemoryCandidate(
            subject="user",
            predicate="coding_language",
            value="Python coding",
            scope="session",
            memory_type="preference",
            confidence=0.9,
            sensitivity="low",
            evidence=(
                MemoryEvidence(
                    message_id=message_id,
                    quote="Python coding",
                ),
            ),
        ),
        MemoryCandidate(
            subject="user",
            predicate="favorite_food",
            value="sushi food",
            scope="session",
            memory_type="preference",
            confidence=0.9,
            sensitivity="low",
            evidence=(
                MemoryEvidence(
                    message_id=message_id,
                    quote="sushi food",
                ),
            ),
        ),
        MemoryCandidate(
            subject="user",
            predicate="outdoor_activity",
            value="hiking outdoor",
            scope="session",
            memory_type="preference",
            confidence=0.9,
            sensitivity="low",
            evidence=(
                MemoryEvidence(
                    message_id=message_id,
                    quote="hiking outdoor",
                ),
            ),
        ),
    ]
    records = []
    for candidate in candidates:
        fact_key = runtime.memory_engine._fact_key(session_id, candidate)
        record, _ = repository.apply_structured_memory(
            session_id=session_id,
            candidate=candidate,
            fact_key=fact_key,
            consolidation_run_id=run_id,
            model_id="seed",
            schema_version="structured-v1",
        )
        records.append(record)
    repository.complete_structured_consolidation(
        run_id,
        output={"seeded": len(records)},
        usage=EmbeddingResponse(vectors=()).usage,
    )
    return records


def test_retrieval_modes_top_k_budget_and_trace(settings):
    model = _KeywordEmbeddingModel()
    phase3_settings = replace(
        settings,
        memory_strategy="structured",
        retrieval_mode="hybrid",
        memory_trigger_messages=100,
    )
    with create_runtime(
        phase3_settings,
        embedding_model=model,
    ) as runtime:
        session = runtime.repository.create_session("Retrieval")
        records = _seed_retrieval_memories(runtime, session.id)

        full = runtime.memory_engine.retrieve(
            session.id,
            turn_id=None,
            query="anything",
            mode="full",
            top_k=1,
        )
        lexical_miss = runtime.memory_engine.retrieve(
            session.id,
            turn_id=None,
            query="software",
            mode="bm25",
        )
        semantic = runtime.memory_engine.retrieve(
            session.id,
            turn_id=None,
            query="software",
            mode="embedding",
            top_k=1,
        )
        hybrid = runtime.memory_engine.retrieve(
            session.id,
            turn_id=None,
            query="software",
            mode="hybrid",
            top_k=1,
        )
        limited = runtime.memory_engine.retrieve(
            session.id,
            turn_id=None,
            query="all topics",
            mode="embedding",
            top_k=1,
        )
        limited_trace = runtime.repository.retrieval_trace(limited.run_id)
        runtime.memory_engine.token_budget = 1
        over_budget = runtime.memory_engine.retrieve(
            session.id,
            turn_id=None,
            query="anything",
            mode="full",
        )

    python_id = next(record.id for record in records if record.value == "Python coding")
    assert len(full.memories) == 3
    assert lexical_miss.memories == ()
    assert semantic.memories[0].id == python_id
    assert hybrid.memories[0].id == python_id
    assert len(limited.memories) == 1
    assert (
        sum(
            candidate["exclusion_reason"] == "top_k"
            for candidate in limited_trace["candidates"]
        )
        == 2
    )
    assert over_budget.memories == ()
    assert over_budget.metadata["retrieval_tokens"] == 0


def test_agent_context_contains_only_retrieved_memory(settings):
    embedding_model = _KeywordEmbeddingModel()
    agent_model = ScriptedAgentModel([AgentResponse(content="context checked")])
    phase3_settings = replace(
        settings,
        memory_strategy="structured",
        retrieval_mode="embedding",
        memory_top_k=1,
        memory_trigger_messages=100,
    )
    with create_runtime(
        phase3_settings,
        agent_model=agent_model,
        memory_model=FakeMemoryModel(),
        embedding_model=embedding_model,
    ) as runtime:
        session = runtime.repository.create_session("Context retrieval")
        _seed_retrieval_memories(runtime, session.id)
        result = runtime.turn_coordinator.run(
            session.id,
            "software recommendation",
        )
        system_prompt = agent_model.requests[0][0][0].content
        trace = runtime.repository.get_trace(result.turn_id)

    agent_trace = next(call for call in trace["model_calls"] if call["role"] == "agent")
    assert "Python coding" in system_prompt
    assert "sushi food" not in system_prompt
    assert "hiking outdoor" not in system_prompt
    assert agent_trace["metadata"]["retrieval_selected_count"] == 1
    assert agent_trace["metadata"]["retrieval_run_id"]


def test_embedding_failure_is_traced_without_blocking_retrieval(settings):
    phase3_settings = replace(
        settings,
        memory_strategy="structured",
        retrieval_mode="embedding",
        memory_trigger_messages=100,
    )
    with create_runtime(
        phase3_settings,
        embedding_model=_FailingEmbeddingModel(),
    ) as runtime:
        session = runtime.repository.create_session("Embedding failure")
        _seed_retrieval_memories(runtime, session.id)
        result = runtime.memory_engine.retrieve(
            session.id,
            turn_id=None,
            query="software",
        )
        trace = runtime.repository.retrieval_trace(result.run_id)

    assert result.content == ""
    assert result.metadata["retrieval_status"] == "failed"
    assert "embedding unavailable" in result.metadata["retrieval_error"]
    assert trace["run"]["status"] == "failed"
