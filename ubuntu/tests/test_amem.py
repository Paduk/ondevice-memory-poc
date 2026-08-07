from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from palmclaw_ubuntu.amem import (
    AMEM_LINK_CANDIDATE_LIMIT,
    AMEM_MAX_METADATA_ITEMS,
    AMEM_METADATA_EMBEDDING_FORMAT_VERSION,
    AMemEngine,
    AMemHistoryEntry,
    format_amem_metadata_embedding_text,
    format_amem_note_content,
)
from palmclaw_ubuntu.models import (
    AMemConstructionResponse,
    AMemEvolutionDecision,
    AMemNeighborUpdate,
    ModelUsage,
)
from palmclaw_ubuntu.providers import (
    FakeAMemModel,
    FakeEmbeddingModel,
    OpenAIAMemModel,
)
from palmclaw_ubuntu.storage import SQLiteRepository


def _insert_note(
    repository: SQLiteRepository,
    session_id: str,
    source_message_id: int,
    *,
    note_id: str | None = None,
):
    return repository.insert_amem_note(
        session_id=session_id,
        source_message_id=source_message_id,
        timestamp=f"2026-01-{source_message_id:02d}T00:00:00Z",
        speaker="user",
        content=f"source {source_message_id}",
        note_id=note_id,
    )


def _entry(source_message_id: int, content: str | None = None) -> AMemHistoryEntry:
    return AMemHistoryEntry(
        source_message_id=source_message_id,
        timestamp=f"2026-02-{source_message_id:02d}T00:00:00Z",
        speaker="user",
        content=content or f"history {source_message_id}",
    )


def _construction(label: str) -> AMemConstructionResponse:
    return AMemConstructionResponse(
        context=f"context {label}",
        keywords=(f"keyword-{label}",),
        tags=("profile",),
        usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
    )


def _seed_engine_note(
    repository: SQLiteRepository,
    session_id: str,
    source_message_id: int,
    *,
    dimensions: int,
):
    note = repository.insert_amem_note(
        session_id=session_id,
        source_message_id=source_message_id,
        timestamp=f"2026-01-{source_message_id:02d}T00:00:00Z",
        speaker="user",
        content=f"seed {source_message_id}",
        note_id=f"seed-{source_message_id}",
    )
    version = repository.insert_amem_note_version(
        note_id=note.id,
        context=f"seed context {source_message_id}",
        keywords=(f"seed-{source_message_id}",),
        tags=("seed",),
    )
    repository.store_amem_note_embedding(
        note_version_id=version.id,
        model_id="fake-hash-embedding-v1",
        dimensions=dimensions,
        vector=(0.0,) * dimensions,
    )
    return note, version


def test_amem_graph_survives_restart_clone_and_backup(settings):
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    session = repository.create_session("A-MEM graph")
    first = _insert_note(repository, session.id, 1, note_id="note-b")
    second = _insert_note(repository, session.id, 2, note_id="note-a")
    first_version = repository.insert_amem_note_version(
        note_id=first.id,
        context="first context",
        keywords=("first",),
        tags=("profile",),
        version_id="version-1",
    )
    repository.insert_amem_note_version(
        note_id=second.id,
        context="second context",
        keywords=("second",),
        tags=("preference",),
    )
    repository.store_amem_note_embedding(
        note_version_id=first_version.id,
        model_id="fake-embedding",
        dimensions=3,
        vector=(0.1, 0.2, 0.3),
    )
    link = repository.insert_amem_link(
        session_id=session.id,
        note_a_id=first.id,
        note_b_id=second.id,
        created_by_note_id=second.id,
        decision="semantic_relation",
        similarity_score=0.75,
        link_id="link-1",
    )
    assert (link.left_note_id, link.right_note_id) == ("note-a", "note-b")

    clone = repository.clone_in_memory()
    assert [note.id for note in clone.list_amem_notes(session.id)] == [
        "note-b",
        "note-a",
    ]
    assert clone.amem_note_embeddings(
        (first_version.id,),
        model_id="fake-embedding",
        dimensions=3,
    ) == {first_version.id: (0.1, 0.2, 0.3)}
    assert clone.list_amem_links(session.id) == (link,)
    clone.close()

    backup_path = settings.data_dir / "amem-backup.db"
    repository.backup_to(backup_path)
    repository.close()

    with SQLiteRepository(settings.database_path) as reopened:
        assert reopened.amem_note_by_source(session.id, 1) == first
        assert reopened.latest_amem_note_version(first.id) == first_version
        assert reopened.list_amem_links(session.id) == (link,)
    with SQLiteRepository(backup_path) as backup:
        assert backup.get_amem_note(second.id) == second
        assert backup.list_amem_links(session.id) == (link,)


def test_amem_graph_fingerprint_is_sensitive_to_every_graph_layer(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM fingerprint matrix")
        empty = repository.amem_graph_fingerprint(session.id)
        first = _insert_note(repository, session.id, 1, note_id="first")
        note_fingerprint = repository.amem_graph_fingerprint(session.id)
        version = repository.insert_amem_note_version(
            note_id=first.id,
            context="context",
            keywords=("keyword",),
            tags=("tag",),
        )
        version_fingerprint = repository.amem_graph_fingerprint(session.id)
        repository.store_amem_note_embedding(
            note_version_id=version.id,
            model_id="embedding",
            dimensions=2,
            vector=(1.0, 0.0),
        )
        embedding_fingerprint = repository.amem_graph_fingerprint(session.id)
        second = _insert_note(repository, session.id, 2, note_id="second")
        repository.insert_amem_note_version(
            note_id=second.id,
            context="second context",
            keywords=("second",),
            tags=("tag",),
        )
        before_link = repository.amem_graph_fingerprint(session.id)
        repository.insert_amem_link(
            session_id=session.id,
            note_a_id=first.id,
            note_b_id=second.id,
            created_by_note_id=second.id,
            decision="test",
        )
        linked_fingerprint = repository.amem_graph_fingerprint(session.id)

        assert len(
            {
                empty,
                note_fingerprint,
                version_fingerprint,
                embedding_fingerprint,
                before_link,
                linked_fingerprint,
            }
        ) == 6


def test_amem_versions_are_immutable_and_latest_is_deterministic(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM versions")
        note = _insert_note(repository, session.id, 1)
        first = repository.insert_amem_note_version(
            note_id=note.id,
            context="initial",
            keywords=("old",),
            tags=("profile",),
        )
        second = repository.insert_amem_note_version(
            note_id=note.id,
            context="evolved",
            keywords=("new",),
            tags=("profile", "updated"),
        )

        versions = repository.amem_note_versions(note.id)
        assert [version.version for version in versions] == [1, 2]
        assert [version.status for version in versions] == [
            "superseded",
            "active",
        ]
        assert second.supersedes_id == first.id
        assert repository.latest_amem_note_version(note.id) == second
        assert repository.get_amem_note(note.id) == note


def test_amem_duplicate_sources_and_invalid_links_are_rejected(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        first_session = repository.create_session("first")
        second_session = repository.create_session("second")
        first = _insert_note(repository, first_session.id, 1, note_id="n-1")
        second = _insert_note(repository, first_session.id, 2, note_id="n-2")
        other = _insert_note(repository, second_session.id, 1, note_id="n-3")

        with pytest.raises(sqlite3.IntegrityError):
            _insert_note(repository, first_session.id, 1)
        with pytest.raises(ValueError, match="self-links"):
            repository.insert_amem_link(
                session_id=first_session.id,
                note_a_id=first.id,
                note_b_id=first.id,
                created_by_note_id=first.id,
                decision="invalid",
            )
        with pytest.raises(ValueError, match="cross sessions"):
            repository.insert_amem_link(
                session_id=first_session.id,
                note_a_id=first.id,
                note_b_id=other.id,
                created_by_note_id=first.id,
                decision="invalid",
            )

        repository.insert_amem_link(
            session_id=first_session.id,
            note_a_id=second.id,
            note_b_id=first.id,
            created_by_note_id=second.id,
            decision="accepted",
        )
        with pytest.raises(sqlite3.IntegrityError):
            repository.insert_amem_link(
                session_id=first_session.id,
                note_a_id=first.id,
                note_b_id=second.id,
                created_by_note_id=first.id,
                decision="duplicate",
            )


def test_amem_execution_traces_and_interrupted_recovery(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM traces")
        note = _insert_note(repository, session.id, 1)
        version = repository.insert_amem_note_version(
            note_id=note.id,
            context="context",
            keywords=("keyword",),
            tags=("tag",),
        )
        construction = repository.begin_amem_construction(
            session_id=session.id,
            source_message_id=1,
            backend="fake",
            model_id="fake-amem",
            prompt_version="construct-v1",
            schema_version="amem-v1",
            idempotency_key="construct-1",
        )
        repository.complete_amem_construction(
            construction,
            note_id=note.id,
            output={"context": "context", "keywords": ["keyword"]},
            usage=ModelUsage(input_tokens=2, output_tokens=3, total_tokens=5),
        )
        assert repository.amem_construction_trace(construction)["output"] == {
            "context": "context",
            "keywords": ["keyword"],
        }

        evolution = repository.begin_amem_evolution(
            session_id=session.id,
            new_note_id=note.id,
            backend="fake",
            model_id="fake-amem",
            prompt_version="evolve-v1",
            schema_version="amem-v1",
            idempotency_key="evolve-1",
        )
        repository.complete_amem_evolution(
            evolution,
            decision={"should_evolve": False},
            usage=ModelUsage(),
        )
        assert repository.amem_evolution_trace(evolution)["decision"] == {
            "should_evolve": False
        }

        retrieval = repository.begin_amem_retrieval(
            session_id=session.id,
            query="what does the user prefer?",
            mode="embedding_linked",
            top_k=5,
            token_budget=200,
            embedding_model_id="fake-embedding",
            embedding_dimensions=3,
        )
        repository.complete_amem_retrieval(
            retrieval,
            (
                {
                    "note_id": note.id,
                    "note_version_id": version.id,
                    "source": "seed",
                    "rank": 1,
                    "embedding_score": 0.9,
                    "combined_score": 0.9,
                    "selected": True,
                    "token_count": 12,
                },
            ),
            selected_count=1,
            selected_tokens=12,
            latency_ms=4,
        )
        trace = repository.amem_retrieval_trace(retrieval)
        assert trace["run"]["status"] == "completed"
        assert trace["candidates"][0]["keywords"] == ["keyword"]

        repository.begin_amem_construction(
            session_id=session.id,
            source_message_id=2,
            backend="fake",
            model_id="fake-amem",
            prompt_version="construct-v1",
            schema_version="amem-v1",
            idempotency_key="construct-2",
        )
        repository.begin_amem_evolution(
            session_id=session.id,
            new_note_id=note.id,
            backend="fake",
            model_id="fake-amem",
            prompt_version="evolve-v1",
            schema_version="amem-v1",
            idempotency_key="evolve-2",
        )
        repository.begin_amem_retrieval(
            session_id=session.id,
            query="interrupted query",
            mode="embedding_linked",
            top_k=5,
            token_budget=200,
        )
        recovered = repository.recover_interrupted_execution()

        assert recovered["amem_constructions_interrupted"] == 1
        assert recovered["amem_evolutions_interrupted"] == 1
        assert recovered["amem_retrievals_interrupted"] == 1
        statuses = {
            row["status"]
            for table in (
                "amem_construction_runs",
                "amem_evolution_events",
                "amem_retrieval_runs",
            )
            for row in repository._connection.execute(
                f"SELECT status FROM {table} WHERE status = 'interrupted'"
            )
        }
        assert statuses == {"interrupted"}


def test_amem_engine_first_note_skips_evolution_and_rerun_is_idempotent(
    settings,
):
    settings.ensure_directories()
    model = FakeAMemModel((_construction("first"),))
    embedding = FakeEmbeddingModel(dimensions=8)
    entry = _entry(1)
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM engine")
        engine = AMemEngine(repository, model, embedding)

        first = engine.ingest(session.id, (entry,))
        request_count = len(embedding.requests)
        second = engine.ingest(session.id, (entry,))

        assert first.processed_source_ids == (1,)
        assert first.skipped_source_ids == ()
        assert second.processed_source_ids == ()
        assert second.skipped_source_ids == (1,)
        assert len(model.construction_requests) == 1
        assert model.evolution_requests == []
        assert len(embedding.requests) == request_count == 1
        assert model.construction_requests[0]["formatted_content"] == (
            format_amem_note_content(entry)
        )
        note = first.notes[0]
        version = repository.latest_amem_note_version(note.id)
        assert version is not None
        assert embedding.requests[0] == (
            format_amem_metadata_embedding_text(
                note,
                context=version.context,
                keywords=version.keywords,
                tags=version.tags,
            ),
        )
        assert AMEM_METADATA_EMBEDDING_FORMAT_VERSION in embedding.requests[0][0]


def test_amem_engine_exposes_only_top_five_and_commits_link_and_update(
    settings,
):
    settings.ensure_directories()
    embedding = FakeEmbeddingModel(dimensions=8)
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM top five")
        seeded = [
            _seed_engine_note(
                repository,
                session.id,
                source_id,
                dimensions=embedding.dimensions,
            )
            for source_id in range(1, 7)
        ]
        target = seeded[0][0]
        model = FakeAMemModel(
            (_construction("new"),),
            (
                AMemEvolutionDecision(
                    should_evolve=True,
                    linked_note_ids=(target.id,),
                    new_note_keywords=("enriched-new",),
                    new_note_tags=("profile", "linked"),
                    neighbor_updates=(
                        AMemNeighborUpdate(
                            note_id=target.id,
                            context="corrected neighbor context",
                            keywords=("corrected",),
                            tags=("seed", "corrected"),
                        ),
                    ),
                ),
            ),
        )
        result = AMemEngine(repository, model, embedding).ingest(
            session.id,
            (_entry(7),),
        )

        _, neighbors = model.evolution_requests[0]
        assert len(neighbors) == AMEM_LINK_CANDIDATE_LIMIT
        assert [neighbor.note.id for neighbor in neighbors] == [
            f"seed-{source_id}" for source_id in range(1, 6)
        ]
        new_note = result.notes[-1]
        new_version = repository.latest_amem_note_version(new_note.id)
        assert new_version is not None
        assert new_version.keywords == ("enriched-new",)
        assert new_version.tags == ("profile", "linked")
        target_versions = repository.amem_note_versions(target.id)
        assert [version.status for version in target_versions] == [
            "superseded",
            "active",
        ]
        assert target_versions[-1].context == "corrected neighbor context"
        assert repository.get_amem_note(target.id).content == "seed 1"
        links = repository.list_amem_links(session.id)
        assert len(links) == 1
        assert {links[0].left_note_id, links[0].right_note_id} == {
            new_note.id,
            target.id,
        }
        latest_embedding = repository.amem_note_embeddings(
            (target_versions[-1].id,),
            model_id=embedding.model_id,
            dimensions=embedding.dimensions,
        )
        assert target_versions[-1].id in latest_embedding


def test_amem_style_skips_evolution_below_similarity_threshold(settings):
    settings.ensure_directories()
    embedding = FakeEmbeddingModel(dimensions=8)
    model = FakeAMemModel(
        (_construction("new"),),
        (AMemEvolutionDecision(should_evolve=False),),
    )
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM style gate")
        _seed_engine_note(
            repository,
            session.id,
            1,
            dimensions=embedding.dimensions,
        )

        AMemEngine(
            repository,
            model,
            embedding,
            evolution_similarity_threshold=0.75,
        ).ingest(session.id, (_entry(2),))

        assert model.evolution_requests == []
        metrics = repository.amem_metrics(session.id)
        assert metrics["generation"]["call_count"] == 1
        assert metrics["generation"]["evolution"] == {}


def test_amem_style_validates_similarity_threshold(settings):
    settings.ensure_directories()
    with (
        SQLiteRepository(settings.database_path) as repository,
        pytest.raises(ValueError, match="between -1 and 1"),
    ):
        AMemEngine(
            repository,
            FakeAMemModel(),
            FakeEmbeddingModel(),
            evolution_similarity_threshold=1.01,
        )


def test_amem_style_recovers_non_candidate_provider_response(settings):
    class InvalidCandidateModel(FakeAMemModel):
        def evolve(self, new_note, neighbors):
            self.evolution_requests.append((new_note, tuple(neighbors)))
            raise ValueError("A-MEM evolution returned a non-candidate note ID")

    settings.ensure_directories()
    embedding = FakeEmbeddingModel(dimensions=8)
    model = InvalidCandidateModel((_construction("new"),))
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM style provider recovery")
        _seed_engine_note(
            repository,
            session.id,
            1,
            dimensions=embedding.dimensions,
        )

        result = AMemEngine(
            repository,
            model,
            embedding,
            evolution_similarity_threshold=-1.0,
        ).ingest(session.id, (_entry(2),))

        assert len(result.notes) == 2
        assert len(model.evolution_requests) == 1
        metrics = repository.amem_metrics(session.id)
        assert metrics["generation"]["evolution"] == {}
        assert metrics["provider"]["failed_call_count"] == 1


@pytest.mark.parametrize(
    ("provider_error", "expected_reason"),
    (
        (
            lambda: TypeAdapter(dict[str, object]).validate_json(
                '{"should_evolve":true,"linked_note_ids":["truncated'
            ),
            "invalid_structured_output",
        ),
        (
            lambda: (_ for _ in ()).throw(
                RuntimeError(
                    "A-MEM evolution provider returned status incomplete"
                )
            ),
            "incomplete_structured_output",
        ),
    ),
)
def test_amem_style_recovers_truncated_evolution_output(
    settings,
    provider_error,
    expected_reason,
):
    class TruncatedEvolutionModel(FakeAMemModel):
        def evolve(self, new_note, neighbors):
            self.evolution_requests.append((new_note, tuple(neighbors)))
            provider_error()
            raise AssertionError("provider_error must raise")

    settings.ensure_directories()
    embedding = FakeEmbeddingModel(dimensions=8)
    model = TruncatedEvolutionModel((_construction("new"),))
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM style truncated output")
        _seed_engine_note(
            repository,
            session.id,
            1,
            dimensions=embedding.dimensions,
        )

        result = AMemEngine(
            repository,
            model,
            embedding,
            evolution_similarity_threshold=-1.0,
        ).ingest(session.id, (_entry(2),))

        assert len(result.notes) == 2
        calls = repository.model_calls_for_sessions((session.id,))
        failed = next(call for call in calls if call["error"] is not None)
        assert failed["metadata"]["recovered_as_no_evolution"] is True
        assert failed["metadata"]["recovery_reason"] == expected_reason


@pytest.mark.parametrize(
    ("link", "update"),
    ((True, False), (False, True)),
    ids=("link-only", "neighbor-update-only"),
)
def test_amem_engine_accepts_independent_evolution_operations(
    settings,
    link,
    update,
):
    settings.ensure_directories()
    embedding = FakeEmbeddingModel(dimensions=8)
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM independent operations")
        target, _ = _seed_engine_note(
            repository,
            session.id,
            1,
            dimensions=embedding.dimensions,
        )
        model = FakeAMemModel(
            (_construction("new"),),
            (
                AMemEvolutionDecision(
                    should_evolve=True,
                    linked_note_ids=(target.id,) if link else (),
                    neighbor_updates=(
                        (
                            AMemNeighborUpdate(
                                note_id=target.id,
                                context="updated context",
                                keywords=("updated",),
                                tags=("updated",),
                            ),
                        )
                        if update
                        else ()
                    ),
                ),
            ),
        )

        AMemEngine(repository, model, embedding).ingest(session.id, (_entry(2),))

        assert bool(repository.list_amem_links(session.id)) is link
        assert len(repository.amem_note_versions(target.id)) == (2 if update else 1)


@pytest.mark.parametrize("invalid_kind", ("link", "update"))
def test_amem_engine_rejects_non_candidate_decisions_atomically(
    settings,
    invalid_kind,
):
    settings.ensure_directories()
    embedding = FakeEmbeddingModel(dimensions=8)
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM invalid decision")
        target, _ = _seed_engine_note(
            repository,
            session.id,
            1,
            dimensions=embedding.dimensions,
        )
        decision = AMemEvolutionDecision(
            should_evolve=True,
            linked_note_ids=("outside-candidates",) if invalid_kind == "link" else (),
            neighbor_updates=(
                (
                    AMemNeighborUpdate(
                        note_id="outside-candidates",
                        context="invalid",
                        keywords=("invalid",),
                        tags=("invalid",),
                    ),
                )
                if invalid_kind == "update"
                else ()
            ),
        )
        model = FakeAMemModel((_construction("invalid"),), (decision,))

        with pytest.raises(ValueError, match="non-candidate"):
            AMemEngine(repository, model, embedding).ingest(
                session.id,
                (_entry(2),),
            )

        assert repository.amem_note_by_source(session.id, 2) is None
        assert repository.amem_note_versions(target.id)[0].status == "active"
        assert repository.list_amem_links(session.id) == ()
        run = repository._connection.execute(
            """
            SELECT status, error FROM amem_construction_runs
            WHERE session_id = ? AND source_message_id = 2
            """,
            (session.id,),
        ).fetchone()
        assert run["status"] == "failed"
        assert "non-candidate" in run["error"]


class _SelfLinkAMemModel(FakeAMemModel):
    def evolve(self, new_note, neighbors):
        self.evolution_requests.append((new_note, tuple(neighbors)))
        return AMemEvolutionDecision(
            should_evolve=True,
            linked_note_ids=(new_note.note.id,),
        )


def test_amem_engine_rejects_self_duplicate_and_oversized_metadata(settings):
    settings.ensure_directories()
    embedding = FakeEmbeddingModel(dimensions=8)
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM validation")
        target, _ = _seed_engine_note(
            repository,
            session.id,
            1,
            dimensions=embedding.dimensions,
        )
        self_link_model = _SelfLinkAMemModel((_construction("self"),))
        with pytest.raises(ValueError, match="new note"):
            AMemEngine(repository, self_link_model, embedding).ingest(
                session.id,
                (_entry(2),),
            )

        duplicate_model = FakeAMemModel(
            (_construction("duplicate"),),
            (
                AMemEvolutionDecision(
                    should_evolve=True,
                    linked_note_ids=(target.id, target.id),
                ),
            ),
        )
        with pytest.raises(ValueError, match="duplicate note IDs"):
            AMemEngine(repository, duplicate_model, embedding).ingest(
                session.id,
                (_entry(2),),
            )

        oversized_model = FakeAMemModel(
            (
                AMemConstructionResponse(
                    context="valid context",
                    keywords=tuple(
                        f"keyword-{index}"
                        for index in range(AMEM_MAX_METADATA_ITEMS + 1)
                    ),
                    tags=("profile",),
                ),
            )
        )
        with pytest.raises(ValueError, match="too many items"):
            AMemEngine(repository, oversized_model, embedding).ingest(
                session.id,
                (_entry(2),),
            )
        assert repository.amem_note_by_source(session.id, 2) is None


class _FailOnceEmbeddingModel:
    backend = "fake"
    model_id = "fail-once-embedding"
    dimensions = 8

    def __init__(self):
        self.failed = False
        self.delegate = FakeEmbeddingModel(self.dimensions)

    def embed(self, texts):
        if not self.failed:
            self.failed = True
            raise RuntimeError("planned embedding failure")
        return self.delegate.embed(texts)


class _FailOnSecondEmbeddingModel:
    backend = "fake"
    model_id = "fake-hash-embedding-v1"
    dimensions = 8

    def __init__(self):
        self.calls = 0
        self.delegate = FakeEmbeddingModel(self.dimensions)

    def embed(self, texts):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("planned neighbor embedding failure")
        return self.delegate.embed(texts)


def test_amem_engine_construct_and_embedding_failures_rollback_and_resume(
    settings,
):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM failures")
        model = FakeAMemModel()
        embedding = FakeEmbeddingModel(dimensions=8)
        engine = AMemEngine(repository, model, embedding)

        with pytest.raises(RuntimeError, match="no construction response"):
            engine.ingest(session.id, (_entry(1),))
        assert repository.list_amem_notes(session.id) == ()
        model.queue_construction(_construction("retry-construct"))
        assert engine.ingest(session.id, (_entry(1),)).processed_source_ids == (1,)

        failing_embedding = _FailOnceEmbeddingModel()
        embed_model = FakeAMemModel((_construction("retry-embed"),))
        embed_engine = AMemEngine(repository, embed_model, failing_embedding)
        with pytest.raises(RuntimeError, match="planned embedding failure"):
            embed_engine.ingest(session.id, (_entry(2),))
        assert repository.amem_note_by_source(session.id, 2) is None
        embed_model.queue_construction(_construction("retry-embed"))
        embed_model.queue_evolution(AMemEvolutionDecision(should_evolve=False))
        assert embed_engine.ingest(
            session.id,
            (_entry(2),),
        ).processed_source_ids == (2,)

        statuses = [
            row["status"]
            for row in repository._connection.execute(
                """
                SELECT status FROM amem_construction_runs
                ORDER BY started_at, rowid
                """
            )
        ]
        assert statuses == ["failed", "completed", "failed", "completed"]


def test_amem_engine_evolution_failure_resumes_after_reopen(settings):
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    session = repository.create_session("A-MEM persistent resume")
    embedding = FakeEmbeddingModel(dimensions=8)
    first_model = FakeAMemModel((_construction("first"),))
    AMemEngine(repository, first_model, embedding).ingest(session.id, (_entry(1),))
    failing_model = FakeAMemModel((_construction("second"),))
    with pytest.raises(RuntimeError, match="no evolution response"):
        AMemEngine(repository, failing_model, embedding).ingest(
            session.id,
            (_entry(2),),
        )
    assert repository.amem_note_by_source(session.id, 2) is None
    repository.close()

    with SQLiteRepository(settings.database_path) as reopened:
        retry_model = FakeAMemModel(
            (_construction("second"),),
            (AMemEvolutionDecision(should_evolve=False),),
        )
        result = AMemEngine(
            reopened,
            retry_model,
            FakeEmbeddingModel(dimensions=8),
        ).ingest(session.id, (_entry(1), _entry(2)))
        assert result.processed_source_ids == (2,)
        assert result.skipped_source_ids == (1,)
        assert len(reopened.list_amem_notes(session.id)) == 2


def test_amem_neighbor_embedding_failure_rolls_back_and_resumes(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM neighbor embedding")
        target, _ = _seed_engine_note(
            repository,
            session.id,
            1,
            dimensions=8,
        )
        decision = AMemEvolutionDecision(
            should_evolve=True,
            neighbor_updates=(
                AMemNeighborUpdate(
                    note_id=target.id,
                    context="updated neighbor",
                    keywords=("updated",),
                    tags=("preference",),
                ),
            ),
        )
        failing = _FailOnSecondEmbeddingModel()
        model = FakeAMemModel((_construction("new"),), (decision,))
        before = repository.amem_graph_fingerprint(session.id)

        with pytest.raises(
            RuntimeError,
            match="planned neighbor embedding failure",
        ):
            AMemEngine(repository, model, failing).ingest(
                session.id,
                (_entry(2),),
            )

        assert repository.amem_graph_fingerprint(session.id) == before
        assert repository.amem_note_by_source(session.id, 2) is None
        assert len(repository.amem_note_versions(target.id)) == 1
        retry = FakeAMemModel((_construction("new"),), (decision,))
        result = AMemEngine(
            repository,
            retry,
            FakeEmbeddingModel(8),
        ).ingest(session.id, (_entry(2),))
        assert result.processed_source_ids == (2,)
        assert len(repository.amem_note_versions(target.id)) == 2


def test_amem_commit_failure_is_atomic_and_retryable(settings, monkeypatch):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM commit failure")
        _seed_engine_note(repository, session.id, 1, dimensions=8)
        model = FakeAMemModel(
            (_construction("second"),),
            (AMemEvolutionDecision(should_evolve=False),),
        )
        original = repository.complete_amem_construction
        failed = False

        def fail_once(*args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("planned pre-commit failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(
            repository,
            "complete_amem_construction",
            fail_once,
        )
        before = repository.amem_graph_fingerprint(session.id)
        with pytest.raises(RuntimeError, match="planned pre-commit failure"):
            AMemEngine(
                repository,
                model,
                FakeEmbeddingModel(8),
            ).ingest(session.id, (_entry(2),))

        assert repository.amem_graph_fingerprint(session.id) == before
        assert repository.amem_note_by_source(session.id, 2) is None
        assert repository.list_amem_links(session.id) == ()
        retry = FakeAMemModel(
            (_construction("second"),),
            (AMemEvolutionDecision(should_evolve=False),),
        )
        result = AMemEngine(
            repository,
            retry,
            FakeEmbeddingModel(8),
        ).ingest(session.id, (_entry(2),))
        assert result.processed_source_ids == (2,)
        statuses = [
            row["status"]
            for row in repository._connection.execute(
                """
                SELECT status FROM amem_construction_runs
                WHERE source_message_id = 2 ORDER BY rowid
                """
            )
        ]
        assert statuses == ["failed", "completed"]


def test_amem_engine_requires_chronology_and_rejects_historical_backfill(
    settings,
):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM chronology")
        model = FakeAMemModel((_construction("later"),))
        engine = AMemEngine(repository, model, FakeEmbeddingModel(dimensions=8))

        with pytest.raises(ValueError, match="chronological"):
            engine.ingest(session.id, (_entry(2), _entry(1)))
        assert model.construction_requests == []
        engine.ingest(session.id, (_entry(2),))
        model.queue_construction(_construction("earlier"))
        with pytest.raises(ValueError, match="backfill"):
            engine.ingest(session.id, (_entry(1),))
        assert len(model.construction_requests) == 1


def test_openai_amem_provider_failure_uses_engine_resume_path(settings):
    class _SequenceResponses:
        def __init__(self):
            self.responses = [
                SimpleNamespace(
                    id="failed-response",
                    status="incomplete",
                    output_parsed=None,
                    usage=None,
                ),
                SimpleNamespace(
                    id="retry-response",
                    status="completed",
                    output_parsed={
                        "contextual_description": "resumed context",
                        "keywords": ["resumed"],
                        "tags": ["profile"],
                    },
                    usage=None,
                ),
            ]

        def parse(self, **kwargs):
            del kwargs
            return self.responses.pop(0)

    settings.ensure_directories()
    client = SimpleNamespace(responses=_SequenceResponses())
    provider = OpenAIAMemModel(
        "amem-model",
        timeout_seconds=1,
        client=client,
    )
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM OpenAI resume")
        engine = AMemEngine(
            repository,
            provider,
            FakeEmbeddingModel(dimensions=8),
        )

        with pytest.raises(RuntimeError, match="status incomplete"):
            engine.ingest(session.id, (_entry(1),))
        assert repository.list_amem_notes(session.id) == ()
        assert engine.ingest(
            session.id,
            (_entry(1),),
        ).processed_source_ids == (1,)
        statuses = [
            row["status"]
            for row in repository._connection.execute(
                """
                SELECT status FROM amem_construction_runs
                ORDER BY started_at, rowid
                """
            )
        ]
        assert statuses == ["failed", "completed"]
