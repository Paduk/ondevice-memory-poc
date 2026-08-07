from __future__ import annotations

from palmclaw_ubuntu.amem_retrieval import AMemRetriever
from palmclaw_ubuntu.models import EmbeddingResponse
from palmclaw_ubuntu.storage import SQLiteRepository


class _QueryEmbeddingModel:
    backend = "fake"
    model_id = "query-embedding"
    dimensions = 2

    def __init__(self, vector=(1.0, 0.0), *, error: Exception | None = None):
        self.vector = tuple(vector)
        self.error = error
        self.requests: list[tuple[str, ...]] = []

    def embed(self, texts):
        self.requests.append(tuple(texts))
        if self.error is not None:
            raise self.error
        return EmbeddingResponse(vectors=(self.vector,))


class _WordTokenCounter:
    @staticmethod
    def count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def truncate(text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def _store_note(
    repository: SQLiteRepository,
    session_id: str,
    *,
    note_id: str,
    source_message_id: int,
    timestamp: str,
    vector: tuple[float, float],
    content: str | None = None,
    context: str | None = None,
):
    note = repository.insert_amem_note(
        session_id=session_id,
        source_message_id=source_message_id,
        timestamp=timestamp,
        speaker="driver",
        content=content or f"content {note_id}",
        note_id=note_id,
    )
    version = repository.insert_amem_note_version(
        note_id=note.id,
        context=context or f"context {note_id}",
        keywords=(f"keyword-{note_id}",),
        tags=("preference",),
    )
    repository.store_amem_note_embedding(
        note_version_id=version.id,
        model_id="query-embedding",
        dimensions=2,
        vector=vector,
    )
    return note, version


def test_amem_retrieval_empty_graph_skips_embedding(settings):
    settings.ensure_directories()
    embedding = _QueryEmbeddingModel()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM empty retrieval")

        result = AMemRetriever(
            repository,
            embedding,
            top_k=3,
            token_budget=100,
            token_counter=_WordTokenCounter(),
        ).retrieve(session.id, "quiet cabin")

        assert result.content == ""
        assert result.notes == ()
        assert result.metadata["retrieval_status"] == "completed"
        assert embedding.requests == []
        trace = repository.amem_retrieval_trace(result.run_id)
        assert trace["run"]["candidate_count"] == 0
        assert trace["candidates"] == []


def test_amem_retrieval_embedding_failure_records_failed_trace(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM retrieval failure")
        _store_note(
            repository,
            session.id,
            note_id="note-1",
            source_message_id=1,
            timestamp="2026-01-01T00:00:00Z",
            vector=(1.0, 0.0),
        )
        result = AMemRetriever(
            repository,
            _QueryEmbeddingModel(error=RuntimeError("planned failure")),
            top_k=1,
            token_budget=100,
            token_counter=_WordTokenCounter(),
        ).retrieve(session.id, "quiet cabin")

        assert result.content == ""
        assert result.metadata["retrieval_status"] == "failed"
        assert "planned failure" in result.metadata["retrieval_error"]
        trace = repository.amem_retrieval_trace(result.run_id)
        assert trace["run"]["status"] == "failed"
        assert "planned failure" in trace["run"]["error"]


def test_amem_retrieval_ranks_only_latest_metadata_embedding(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM latest metadata")
        old_note, old_version = _store_note(
            repository,
            session.id,
            note_id="evolved-note",
            source_message_id=1,
            timestamp="2026-01-01T00:00:00Z",
            vector=(1.0, 0.0),
            context="obsolete context",
        )
        current_version = repository.insert_amem_note_version(
            note_id=old_note.id,
            context="current evolved context",
            keywords=("current",),
            tags=("updated",),
        )
        repository.store_amem_note_embedding(
            note_version_id=current_version.id,
            model_id="query-embedding",
            dimensions=2,
            vector=(0.0, 1.0),
        )
        winner, winner_version = _store_note(
            repository,
            session.id,
            note_id="winner",
            source_message_id=2,
            timestamp="2026-01-02T00:00:00Z",
            vector=(0.8, 0.6),
        )

        result = AMemRetriever(
            repository,
            _QueryEmbeddingModel(),
            top_k=1,
            token_budget=100,
            token_counter=_WordTokenCounter(),
        ).retrieve(session.id, "query")

        assert [note.id for note in result.notes] == [winner.id]
        trace = repository.amem_retrieval_trace(result.run_id)
        candidates = {item["note_id"]: item for item in trace["candidates"]}
        assert candidates[old_note.id]["note_version_id"] == current_version.id
        assert candidates[old_note.id]["embedding_score"] == 0.0
        assert candidates[winner.id]["note_version_id"] == winner_version.id
        assert old_version.id not in {
            item["note_version_id"] for item in trace["candidates"]
        }


def test_amem_retrieval_tie_breaks_by_timestamp_then_note_id(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM tie break")
        for note_id, source_id, timestamp in (
            ("older", 1, "2026-01-01T00:00:00Z"),
            ("newer-b", 2, "2026-01-02T00:00:00Z"),
            ("newer-a", 3, "2026-01-02T00:00:00Z"),
        ):
            _store_note(
                repository,
                session.id,
                note_id=note_id,
                source_message_id=source_id,
                timestamp=timestamp,
                vector=(1.0, 0.0),
            )

        result = AMemRetriever(
            repository,
            _QueryEmbeddingModel(),
            top_k=3,
            token_budget=200,
            token_counter=_WordTokenCounter(),
        ).retrieve(session.id, "query")

        assert [note.id for note in result.notes] == [
            "newer-a",
            "newer-b",
            "older",
        ]


def test_amem_retrieval_expands_undirected_links_and_deduplicates(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM linked retrieval")
        first, _ = _store_note(
            repository,
            session.id,
            note_id="seed-1",
            source_message_id=1,
            timestamp="2026-01-03T00:00:00Z",
            vector=(1.0, 0.0),
        )
        second, _ = _store_note(
            repository,
            session.id,
            note_id="seed-2",
            source_message_id=2,
            timestamp="2026-01-02T00:00:00Z",
            vector=(0.9, 0.435889894),
        )
        shared, _ = _store_note(
            repository,
            session.id,
            note_id="shared",
            source_message_id=3,
            timestamp="2026-01-01T00:00:00Z",
            vector=(0.0, 1.0),
        )
        repository.insert_amem_link(
            session_id=session.id,
            note_a_id=shared.id,
            note_b_id=first.id,
            created_by_note_id=first.id,
            decision="test",
        )
        repository.insert_amem_link(
            session_id=session.id,
            note_a_id=second.id,
            note_b_id=shared.id,
            created_by_note_id=second.id,
            decision="test",
        )

        result = AMemRetriever(
            repository,
            _QueryEmbeddingModel(),
            top_k=2,
            token_budget=200,
            token_counter=_WordTokenCounter(),
        ).retrieve(session.id, "query")

        assert [note.id for note in result.notes] == [
            first.id,
            second.id,
            shared.id,
        ]
        assert result.content.count("content shared") == 1
        assert result.content.index("[A-MEM seed]") < result.content.index(
            "[A-MEM linked]"
        )
        trace = repository.amem_retrieval_trace(result.run_id)
        shared_trace = next(
            item for item in trace["candidates"] if item["note_id"] == shared.id
        )
        assert shared_trace["source"] == "linked"
        assert shared_trace["selected"] == 1


def test_amem_retrieval_bounds_oversized_note_and_prioritizes_seed(settings):
    settings.ensure_directories()
    counter = _WordTokenCounter()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("A-MEM bounded retrieval")
        seed, _ = _store_note(
            repository,
            session.id,
            note_id="seed",
            source_message_id=1,
            timestamp="2026-01-02T00:00:00Z",
            vector=(1.0, 0.0),
            content=" ".join(f"seedword-{index}" for index in range(100)),
        )
        linked, _ = _store_note(
            repository,
            session.id,
            note_id="linked",
            source_message_id=2,
            timestamp="2026-01-01T00:00:00Z",
            vector=(0.0, 1.0),
        )
        repository.insert_amem_link(
            session_id=session.id,
            note_a_id=seed.id,
            note_b_id=linked.id,
            created_by_note_id=seed.id,
            decision="test",
        )

        result = AMemRetriever(
            repository,
            _QueryEmbeddingModel(),
            top_k=1,
            token_budget=12,
            token_counter=counter,
        ).retrieve(session.id, "query")

        assert [note.id for note in result.notes] == [seed.id]
        assert counter.count(result.content) <= 12
        assert "[A-MEM seed]" in result.content
        assert "similarity_score" not in result.content
        trace = repository.amem_retrieval_trace(result.run_id)
        linked_trace = next(
            item for item in trace["candidates"] if item["note_id"] == linked.id
        )
        assert linked_trace["exclusion_reason"] == "token_budget"


def test_amem_retrieval_is_byte_stable_and_graph_read_only_after_reopen(
    settings,
):
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    session = repository.create_session("A-MEM read only")
    _store_note(
        repository,
        session.id,
        note_id="stable-note",
        source_message_id=1,
        timestamp="2026-01-01T00:00:00Z",
        vector=(1.0, 0.0),
        context="stable evolved context",
    )
    before = repository.amem_graph_fingerprint(session.id)
    first = AMemRetriever(
        repository,
        _QueryEmbeddingModel(),
        top_k=1,
        token_budget=100,
        token_counter=_WordTokenCounter(),
    ).retrieve(session.id, "query")
    after = repository.amem_graph_fingerprint(session.id)
    retrieval_count = repository._connection.execute(
        "SELECT COUNT(*) FROM amem_retrieval_runs"
    ).fetchone()[0]
    repository.close()

    with SQLiteRepository(settings.database_path) as reopened:
        reopened_before = reopened.amem_graph_fingerprint(session.id)
        second = AMemRetriever(
            reopened,
            _QueryEmbeddingModel(),
            top_k=1,
            token_budget=100,
            token_counter=_WordTokenCounter(),
        ).retrieve(session.id, "query")
        reopened_after = reopened.amem_graph_fingerprint(session.id)
        assert reopened._connection.execute(
            "SELECT COUNT(*) FROM amem_retrieval_runs"
        ).fetchone()[0] == retrieval_count + 1

    assert first.content == second.content
    assert "stable evolved context" in first.content
    assert before == after == reopened_before == reopened_after
