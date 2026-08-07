from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from palmclaw_ubuntu.amem import AMemHistoryEntry
from palmclaw_ubuntu.compact_amem import (
    COMPACT_AMEM_EPISODE_FORMAT_VERSION,
    CompactAMemGraphEngine,
    build_compact_amem_episodes,
)
from palmclaw_ubuntu.models import (
    AMemEvolutionDecision,
    CompactAMemNoteDraft,
    CompactAMemResponse,
)
from palmclaw_ubuntu.providers import (
    FakeAMemModel,
    FakeCompactAMemModel,
    FakeEmbeddingModel,
)
from palmclaw_ubuntu.storage import SQLiteRepository


def _entry(source_id: int, *, minutes: int, content: str | None = None):
    timestamp = datetime(2026, 3, 1, tzinfo=UTC) + timedelta(minutes=minutes)
    return AMemHistoryEntry(
        source_message_id=source_id,
        timestamp=timestamp.isoformat(),
        speaker="user",
        content=content or f"history {source_id}",
    )


def test_compact_amem_groups_chronological_history_by_entry_bound():
    episodes = build_compact_amem_episodes(
        tuple(_entry(index, minutes=index) for index in range(1, 6)),
        max_entries=2,
        max_chars=1_000,
        max_gap_seconds=None,
    )

    assert [episode.index for episode in episodes] == [0, 1, 2]
    assert [episode.source_message_ids for episode in episodes] == [
        (1, 2),
        (3, 4),
        (5,),
    ]
    assert episodes[0].start_timestamp == episodes[0].entries[0].timestamp
    assert episodes[0].end_timestamp == episodes[0].entries[-1].timestamp
    assert (
        f"format={COMPACT_AMEM_EPISODE_FORMAT_VERSION}"
        in episodes[0].rendered_content
    )
    assert "source_message_id=1" in episodes[0].rendered_content


def test_compact_amem_splits_on_time_gap_and_char_bound():
    gap_split = build_compact_amem_episodes(
        (_entry(1, minutes=0), _entry(2, minutes=61)),
        max_entries=10,
        max_chars=1_000,
        max_gap_seconds=3_600,
    )
    char_split = build_compact_amem_episodes(
        (
            _entry(1, minutes=0, content="a" * 40),
            _entry(2, minutes=1, content="b" * 40),
        ),
        max_entries=10,
        max_chars=150,
        max_gap_seconds=None,
    )

    assert [episode.source_message_ids for episode in gap_split] == [(1,), (2,)]
    assert [episode.source_message_ids for episode in char_split] == [(1,), (2,)]
    assert all(len(episode.rendered_content) <= 150 for episode in char_split)


@pytest.mark.parametrize(
    "entries",
    (
        (_entry(2, minutes=2), _entry(1, minutes=1)),
        (_entry(1, minutes=1), _entry(1, minutes=2)),
    ),
    ids=("non-chronological", "duplicate-source"),
)
def test_compact_amem_rejects_invalid_history(entries):
    with pytest.raises(ValueError):
        build_compact_amem_episodes(entries)


def test_fake_compact_amem_model_records_episode_and_allows_empty_notes():
    episode = build_compact_amem_episodes((_entry(1, minutes=0),))[0]
    response = CompactAMemResponse(notes=())
    model = FakeCompactAMemModel((response,))

    assert model.compact_episode(episode) == response
    assert model.requests == [episode]
    with pytest.raises(RuntimeError, match="no response left"):
        model.compact_episode(episode)


def _draft(
    content: str,
    source_ids: tuple[int, ...],
    *,
    kind: str = "preference",
) -> CompactAMemNoteDraft:
    return CompactAMemNoteDraft(
        content=content,
        context=f"Durable context for {content}",
        keywords=("shared", content),
        tags=(kind,),
        memory_kind=kind,
        source_message_ids=source_ids,
    )


def test_compact_graph_persists_sources_filters_empty_and_resumes(settings):
    settings.ensure_directories()
    episodes = build_compact_amem_episodes(
        (_entry(1, minutes=0), _entry(2, minutes=1)),
        max_entries=1,
    )
    compact = FakeCompactAMemModel(
        (
            CompactAMemResponse(notes=()),
            CompactAMemResponse(notes=(_draft("quiet cabin", (2,)),)),
        )
    )
    embedding = FakeEmbeddingModel(8)
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Compact A-MEM graph")
        engine = CompactAMemGraphEngine(
            repository,
            compact,
            FakeAMemModel(),
            embedding,
        )

        first = engine.ingest(session.id, episodes)
        fingerprint = first.graph_fingerprint
        resumed_model = FakeCompactAMemModel()
        resumed = CompactAMemGraphEngine(
            repository,
            resumed_model,
            FakeAMemModel(),
            embedding,
        ).ingest(session.id, episodes)

        assert first.processed_episode_indices == (0, 1)
        assert len(first.notes) == 1
        assert first.notes[0].source_message_id == 16
        assert repository.compact_amem_note_sources(first.notes[0].id) == (2,)
        assert resumed.processed_episode_indices == ()
        assert resumed.skipped_episode_indices == (0, 1)
        assert resumed.graph_fingerprint == fingerprint
        assert resumed_model.requests == []
        assert len(compact.requests) == 2


def test_compact_graph_links_deterministically_and_evolves_only_correction(
    settings,
):
    settings.ensure_directories()
    episodes = build_compact_amem_episodes(
        tuple(_entry(index, minutes=index) for index in range(1, 4)),
        max_entries=1,
    )
    compact = FakeCompactAMemModel(
        (
            CompactAMemResponse(notes=(_draft("quiet", (1,)),)),
            CompactAMemResponse(
                notes=(_draft("night driving", (2,), kind="stable_context"),)
            ),
            CompactAMemResponse(
                notes=(_draft("not quiet", (3,), kind="correction"),)
            ),
        )
    )
    evolution = FakeAMemModel(
        evolution_responses=(
            AMemEvolutionDecision(
                should_evolve=True,
                new_note_keywords=("corrected",),
                new_note_tags=("correction", "active"),
            ),
        )
    )
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Compact A-MEM link gate")
        result = CompactAMemGraphEngine(
            repository,
            compact,
            evolution,
            FakeEmbeddingModel(8),
            link_similarity_threshold=-1.0,
        ).ingest(session.id, episodes)

        links = repository.list_amem_links(session.id)
        correction = result.notes[-1]
        correction_version = repository.latest_amem_note_version(correction.id)
        assert len(result.notes) == 3
        assert len(links) == 3
        assert all(link.decision == "compact_embedding" for link in links)
        assert len(evolution.evolution_requests) == 1
        assert evolution.evolution_requests[0][0].note.id == correction.id
        assert correction_version is not None
        assert correction_version.keywords == ("corrected",)
        assert correction_version.tags == ("correction", "active")
        metrics = repository.amem_metrics(session.id)
        assert metrics["generation"]["compaction"] == {"completed": 3}
        assert metrics["generation"]["evolution"] == {"completed": 1}
        assert metrics["generation"]["call_count"] == 4


def test_compact_graph_failure_is_atomic_and_episode_can_retry(settings):
    settings.ensure_directories()
    episode = build_compact_amem_episodes((_entry(1, minutes=0),))[0]
    invalid = FakeCompactAMemModel(
        (CompactAMemResponse(notes=(_draft("invalid", (999,)),)),)
    )
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Compact A-MEM retry")
        with pytest.raises(ValueError, match="outside source"):
            CompactAMemGraphEngine(
                repository,
                invalid,
                FakeAMemModel(),
                FakeEmbeddingModel(8),
            ).ingest(session.id, (episode,))

        assert repository.list_amem_notes(session.id) == ()
        failed_run = repository._connection.execute(
            """
            SELECT id, status FROM compact_amem_episode_runs
            WHERE session_id = ? ORDER BY started_at DESC LIMIT 1
            """,
            (session.id,),
        ).fetchone()
        assert failed_run["status"] == "failed"

        retry = CompactAMemGraphEngine(
            repository,
            FakeCompactAMemModel(
                (CompactAMemResponse(notes=(_draft("valid", (1,)),)),)
            ),
            FakeAMemModel(),
            FakeEmbeddingModel(8),
        ).ingest(session.id, (episode,))

        assert retry.processed_episode_indices == (0,)
        assert len(retry.notes) == 1
        assert repository.compact_amem_episode_trace(failed_run["id"])[
            "status"
        ] == "failed"


def test_compact_graph_requires_dedicated_session(settings):
    settings.ensure_directories()
    episode = build_compact_amem_episodes((_entry(1, minutes=0),))[0]
    compact = FakeCompactAMemModel()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Mixed A-MEM graph")
        repository.insert_amem_note(
            session_id=session.id,
            source_message_id=1,
            timestamp="2026-03-01T00:00:00+00:00",
            speaker="user",
            content="regular A-MEM note",
        )

        with pytest.raises(ValueError, match="dedicated session"):
            CompactAMemGraphEngine(
                repository,
                compact,
                FakeAMemModel(),
                FakeEmbeddingModel(8),
            ).ingest(session.id, (episode,))

        assert compact.requests == []
