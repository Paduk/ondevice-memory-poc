from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from palmclaw_ubuntu.amem import (
    AMEM_LINK_CANDIDATE_LIMIT,
    AMemEngine,
    AMemHistoryEntry,
)
from palmclaw_ubuntu.contracts import AMemModel, CompactAMemModel, EmbeddingModel
from palmclaw_ubuntu.models import (
    AMemEvolutionDecision,
    AMemNeighbor,
    AMemNote,
    AMemNoteVersion,
    CompactAMemEpisode,
    CompactAMemEpisodeEntry,
    CompactAMemNoteDraft,
    CompactAMemResponse,
    ModelUsage,
)
from palmclaw_ubuntu.privacy import redact_secrets
from palmclaw_ubuntu.storage import SQLiteRepository

COMPACT_AMEM_EPISODE_FORMAT_VERSION = "compact-amem-episode-v1"
DEFAULT_COMPACT_AMEM_EPISODE_MAX_ENTRIES = 16
DEFAULT_COMPACT_AMEM_EPISODE_MAX_CHARS = 8_000
DEFAULT_COMPACT_AMEM_EPISODE_MAX_GAP_SECONDS = 6 * 60 * 60
DEFAULT_COMPACT_AMEM_LINK_THRESHOLD = 0.75
COMPACT_AMEM_GRAPH_POLICY_VERSION = "compact-amem-graph-source-order-v2"
COMPACT_AMEM_NOTE_SEQUENCE_WIDTH = 16
_COMPACT_AMEM_MEMORY_KINDS = {
    "preference",
    "constraint",
    "correction",
    "commitment",
    "profile",
    "stable_context",
}


@dataclass(frozen=True)
class CompactAMemIngestionResult:
    notes: tuple[AMemNote, ...]
    processed_episode_indices: tuple[int, ...]
    skipped_episode_indices: tuple[int, ...]
    graph_fingerprint: str


@dataclass(frozen=True)
class _CompactNotePlan:
    note: AMemNote
    draft: CompactAMemNoteDraft
    final_keywords: tuple[str, ...]
    final_tags: tuple[str, ...]
    final_vector: tuple[float, ...]
    linked_neighbors: tuple[AMemNeighbor, ...]
    evolution: AMemEvolutionDecision | None
    neighbor_vectors: dict[str, tuple[float, ...]]


def build_compact_amem_episodes(
    entries: Sequence[AMemHistoryEntry],
    *,
    max_entries: int = DEFAULT_COMPACT_AMEM_EPISODE_MAX_ENTRIES,
    max_chars: int = DEFAULT_COMPACT_AMEM_EPISODE_MAX_CHARS,
    max_gap_seconds: int | None = DEFAULT_COMPACT_AMEM_EPISODE_MAX_GAP_SECONDS,
) -> tuple[CompactAMemEpisode, ...]:
    """Group chronological history into deterministic, resource-bounded episodes."""

    if max_entries < 1:
        raise ValueError("Compact A-MEM episode max entries must be positive")
    if max_chars < 1:
        raise ValueError("Compact A-MEM episode max chars must be positive")
    if max_gap_seconds is not None and max_gap_seconds < 0:
        raise ValueError("Compact A-MEM episode max gap cannot be negative")

    normalized = tuple(_validate_entry(entry) for entry in entries)
    source_ids = [entry.source_message_id for entry in normalized]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Compact A-MEM history contains duplicate source IDs")
    parsed_timestamps = tuple(_parse_timestamp(entry.timestamp) for entry in normalized)
    if any(
        current < previous
        for previous, current in zip(
            parsed_timestamps,
            parsed_timestamps[1:],
            strict=False,
        )
    ):
        raise ValueError("Compact A-MEM history must be chronological")

    groups: list[list[CompactAMemEpisodeEntry]] = []
    current: list[CompactAMemEpisodeEntry] = []
    header_chars = len(f"format={COMPACT_AMEM_EPISODE_FORMAT_VERSION}\n")
    current_chars = header_chars
    previous_timestamp: datetime | None = None
    for entry, timestamp in zip(normalized, parsed_timestamps, strict=True):
        rendered_entry = _render_entry(entry)
        if header_chars + len(rendered_entry) > max_chars:
            raise ValueError(
                "Compact A-MEM history entry exceeds the episode char limit: "
                f"source_message_id={entry.source_message_id}"
            )
        separator_chars = 2 if current else 0
        gap_exceeded = (
            bool(current)
            and max_gap_seconds is not None
            and previous_timestamp is not None
            and (timestamp - previous_timestamp).total_seconds() > max_gap_seconds
        )
        limit_exceeded = bool(current) and (
            len(current) >= max_entries
            or current_chars + separator_chars + len(rendered_entry) > max_chars
        )
        if gap_exceeded or limit_exceeded:
            groups.append(current)
            current = []
            current_chars = header_chars
            separator_chars = 0
        current.append(entry)
        current_chars += separator_chars + len(rendered_entry)
        previous_timestamp = timestamp
    if current:
        groups.append(current)

    return tuple(
        CompactAMemEpisode(
            index=index,
            entries=tuple(group),
            start_timestamp=group[0].timestamp,
            end_timestamp=group[-1].timestamp,
            rendered_content=format_compact_amem_episode(group),
        )
        for index, group in enumerate(groups)
    )


def format_compact_amem_episode(
    entries: Sequence[CompactAMemEpisodeEntry],
) -> str:
    normalized = tuple(entries)
    if not normalized:
        raise ValueError("Compact A-MEM episode cannot be empty")
    return (
        f"format={COMPACT_AMEM_EPISODE_FORMAT_VERSION}\n"
        + "\n\n".join(_render_entry(entry) for entry in normalized)
    )


class CompactAMemGraphEngine(AMemEngine):
    """Build an A-MEM-compatible graph from compact, evidence-linked notes."""

    def __init__(
        self,
        repository: SQLiteRepository,
        compact_model: CompactAMemModel,
        evolution_model: AMemModel,
        embedding_model: EmbeddingModel,
        *,
        candidate_limit: int = AMEM_LINK_CANDIDATE_LIMIT,
        link_similarity_threshold: float = DEFAULT_COMPACT_AMEM_LINK_THRESHOLD,
        memory_input_cost_per_million: float = 0.0,
        memory_output_cost_per_million: float = 0.0,
        embedding_input_cost_per_million: float = 0.0,
    ):
        if not -1.0 <= link_similarity_threshold <= 1.0:
            raise ValueError(
                "Compact A-MEM link similarity threshold must be between -1 and 1"
            )
        super().__init__(
            repository,
            evolution_model,
            embedding_model,
            candidate_limit=candidate_limit,
            evolution_similarity_threshold=None,
            memory_input_cost_per_million=memory_input_cost_per_million,
            memory_output_cost_per_million=memory_output_cost_per_million,
            embedding_input_cost_per_million=embedding_input_cost_per_million,
        )
        self.compact_model = compact_model
        self.link_similarity_threshold = link_similarity_threshold

    def ingest(
        self,
        session_id: str,
        episodes: Sequence[CompactAMemEpisode],
    ) -> CompactAMemIngestionResult:
        self.repository.require_session(session_id)
        ordered = self._validate_episodes(episodes)
        non_compact_notes = [
            note.id
            for note in self.repository.list_amem_notes(session_id)
            if not self.repository.compact_amem_note_sources(note.id)
        ]
        if non_compact_notes:
            raise ValueError(
                "Compact A-MEM requires a dedicated session without regular "
                "A-MEM notes"
            )
        completed = self.repository.completed_compact_amem_episode_sources(
            session_id
        )
        processed: list[int] = []
        skipped: list[int] = []
        for episode in ordered:
            existing_sources = completed.get(episode.index)
            if existing_sources is not None:
                if existing_sources != episode.source_message_ids:
                    raise ValueError(
                        "Compact A-MEM episode index was completed with "
                        "different sources"
                    )
                skipped.append(episode.index)
                continue
            self._ingest_episode(session_id, episode)
            processed.append(episode.index)
        return CompactAMemIngestionResult(
            notes=self.repository.list_amem_notes(session_id),
            processed_episode_indices=tuple(processed),
            skipped_episode_indices=tuple(skipped),
            graph_fingerprint=(
                self.repository.compact_amem_graph_fingerprint(session_id)
            ),
        )

    def _ingest_episode(
        self,
        session_id: str,
        episode: CompactAMemEpisode,
    ) -> None:
        run_id = self.repository.begin_compact_amem_episode(
            session_id=session_id,
            episode_index=episode.index,
            start_timestamp=episode.start_timestamp,
            end_timestamp=episode.end_timestamp,
            source_message_ids=episode.source_message_ids,
            backend=self.compact_model.backend,
            model_id=self.compact_model.model_id,
            prompt_version=self.compact_model.prompt_version,
            schema_version=self.compact_model.schema_version,
            idempotency_key=(
                f"compact-amem:{session_id}:{episode.index}:{uuid.uuid4()}"
            ),
        )
        try:
            response = self._compact_episode(
                session_id,
                episode,
                run_id=run_id,
            )
            plans = tuple(
                self._plan_note(
                    session_id,
                    episode,
                    draft,
                    ordinal=ordinal,
                    run_id=run_id,
                )
                for ordinal, draft in enumerate(response.notes)
            )
            self._commit_episode(run_id, episode, response, plans)
        except BaseException as exc:
            self.repository.fail_compact_amem_episode(
                run_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def _compact_episode(
        self,
        session_id: str,
        episode: CompactAMemEpisode,
        *,
        run_id: str,
    ) -> CompactAMemResponse:
        started = time.monotonic()
        try:
            response = self.compact_model.compact_episode(episode)
        except BaseException as exc:
            self._record_provider_call(
                session_id=session_id,
                role="amem_compaction",
                backend=self.compact_model.backend,
                model_id=self.compact_model.model_id,
                prompt_version=self.compact_model.prompt_version,
                schema_version=self.compact_model.schema_version,
                started=started,
                usage=ModelUsage(),
                response_id=None,
                metadata={"compact_amem_episode_run_id": run_id},
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        if not isinstance(response, CompactAMemResponse):
            raise TypeError("Compact A-MEM provider returned an invalid response")
        self._record_provider_call(
            session_id=session_id,
            role="amem_compaction",
            backend=self.compact_model.backend,
            model_id=self.compact_model.model_id,
            prompt_version=self.compact_model.prompt_version,
            schema_version=self.compact_model.schema_version,
            started=started,
            usage=response.usage,
            response_id=response.response_id,
            metadata={
                **dict(response.metadata),
                "compact_amem_episode_run_id": run_id,
                "source_count": len(episode.entries),
                "note_count": len(response.notes),
            },
        )
        return replace(
            response,
            notes=self._validate_drafts(episode, response.notes),
        )

    def _plan_note(
        self,
        session_id: str,
        episode: CompactAMemEpisode,
        draft: CompactAMemNoteDraft,
        *,
        ordinal: int,
        run_id: str,
    ) -> _CompactNotePlan:
        note = AMemNote(
            id=str(uuid.uuid4()),
            session_id=session_id,
            source_message_id=(
                episode.index * COMPACT_AMEM_NOTE_SEQUENCE_WIDTH + ordinal
            ),
            timestamp=episode.end_timestamp,
            speaker="compact_memory",
            content=redact_secrets(draft.content),
            status="active",
            created_at="",
        )
        initial_version = AMemNoteVersion(
            id=str(uuid.uuid4()),
            note_id=note.id,
            version=1,
            context=draft.context,
            keywords=draft.keywords,
            tags=draft.tags,
            status="active",
            supersedes_id=None,
            created_by_event_id=None,
            created_at="",
        )
        initial_vector = self._embed_metadata(
            note,
            context=draft.context,
            keywords=draft.keywords,
            tags=draft.tags,
            run_id=run_id,
        )
        candidates = self._link_candidates(session_id, initial_vector)
        linked = tuple(
            candidate
            for candidate in candidates
            if candidate.similarity_score >= self.link_similarity_threshold
        )
        evolution = None
        if draft.memory_kind == "correction" and linked:
            evolution = self._evolve(
                AMemNeighbor(
                    note=note,
                    version=initial_version,
                    similarity_score=1.0,
                ),
                linked,
                session_id=session_id,
                run_id=run_id,
            )
            if evolution is not None:
                evolution = replace(evolution, linked_note_ids=())
        final_keywords = (
            evolution.new_note_keywords
            if evolution is not None and evolution.new_note_keywords
            else draft.keywords
        )
        final_tags = (
            evolution.new_note_tags
            if evolution is not None and evolution.new_note_tags
            else draft.tags
        )
        final_vector = initial_vector
        if final_keywords != draft.keywords or final_tags != draft.tags:
            final_vector = self._embed_metadata(
                note,
                context=draft.context,
                keywords=final_keywords,
                tags=final_tags,
                run_id=run_id,
            )
        return _CompactNotePlan(
            note=note,
            draft=draft,
            final_keywords=final_keywords,
            final_tags=final_tags,
            final_vector=final_vector,
            linked_neighbors=linked,
            evolution=evolution,
            neighbor_vectors=self._updated_neighbor_vectors(
                evolution,
                linked,
                run_id=run_id,
            ),
        )

    def _commit_episode(
        self,
        run_id: str,
        episode: CompactAMemEpisode,
        response: CompactAMemResponse,
        plans: Sequence[_CompactNotePlan],
    ) -> None:
        with self.repository.amem_transaction():
            for ordinal, plan in enumerate(plans):
                note = self.repository.insert_amem_note(
                    session_id=plan.note.session_id,
                    source_message_id=plan.note.source_message_id,
                    timestamp=plan.note.timestamp,
                    speaker=plan.note.speaker,
                    content=plan.note.content,
                    note_id=plan.note.id,
                )
                self.repository.insert_compact_amem_note_provenance(
                    note_id=note.id,
                    episode_run_id=run_id,
                    note_ordinal=ordinal,
                    memory_kind=plan.draft.memory_kind,
                    source_message_ids=plan.draft.source_message_ids,
                )
                event_id = None
                if plan.evolution is not None:
                    event_id = self.repository.begin_amem_evolution(
                        session_id=note.session_id,
                        new_note_id=note.id,
                        backend=self.model.backend,
                        model_id=self.model.model_id,
                        prompt_version=self.model.evolution_prompt_version,
                        schema_version=self.model.evolution_schema_version,
                        idempotency_key=f"compact-amem-evolution:{run_id}:{ordinal}",
                    )
                version = self.repository.insert_amem_note_version(
                    note_id=note.id,
                    context=plan.draft.context,
                    keywords=plan.final_keywords,
                    tags=plan.final_tags,
                    created_by_event_id=event_id,
                )
                self.repository.store_amem_note_embedding(
                    note_version_id=version.id,
                    model_id=self.embedding_model.model_id,
                    dimensions=self.embedding_model.dimensions,
                    vector=plan.final_vector,
                )
                if plan.evolution is not None and event_id is not None:
                    for update in plan.evolution.neighbor_updates:
                        updated = self.repository.insert_amem_note_version(
                            note_id=update.note_id,
                            context=update.context,
                            keywords=update.keywords,
                            tags=update.tags,
                            created_by_event_id=event_id,
                        )
                        self.repository.store_amem_note_embedding(
                            note_version_id=updated.id,
                            model_id=self.embedding_model.model_id,
                            dimensions=self.embedding_model.dimensions,
                            vector=plan.neighbor_vectors[update.note_id],
                        )
                for neighbor in plan.linked_neighbors:
                    self.repository.insert_amem_link(
                        session_id=note.session_id,
                        note_a_id=note.id,
                        note_b_id=neighbor.note.id,
                        created_by_note_id=note.id,
                        decision="compact_embedding",
                        similarity_score=neighbor.similarity_score,
                    )
                if plan.evolution is not None and event_id is not None:
                    self.repository.complete_amem_evolution(
                        event_id,
                        decision=self._decision_trace(plan.evolution),
                        usage=plan.evolution.usage,
                    )
            self.repository.complete_compact_amem_episode(
                run_id,
                output={
                    "episode_index": episode.index,
                    "note_ids": [plan.note.id for plan in plans],
                    "source_message_ids": list(episode.source_message_ids),
                    "response_id": response.response_id,
                    "metadata": dict(response.metadata),
                },
                usage=response.usage,
            )

    @classmethod
    def _validate_drafts(
        cls,
        episode: CompactAMemEpisode,
        drafts: Sequence[CompactAMemNoteDraft],
    ) -> tuple[CompactAMemNoteDraft, ...]:
        normalized = tuple(drafts)
        if len(normalized) > COMPACT_AMEM_NOTE_SEQUENCE_WIDTH:
            raise ValueError("Compact A-MEM episode returned too many notes")
        source_order = {
            source_id: index
            for index, source_id in enumerate(episode.source_message_ids)
        }
        contents: set[str] = set()
        validated = []
        for draft in normalized:
            if not isinstance(draft, CompactAMemNoteDraft):
                raise TypeError("Compact A-MEM response contains an invalid note")
            if draft.memory_kind not in _COMPACT_AMEM_MEMORY_KINDS:
                raise ValueError("Compact A-MEM note has an invalid memory kind")
            if not draft.content.strip() or not draft.source_message_ids:
                raise ValueError("Compact A-MEM note content and sources are required")
            if draft.content.strip() in contents:
                raise ValueError("Compact A-MEM response contains duplicate notes")
            contents.add(draft.content.strip())
            if len(set(draft.source_message_ids)) != len(draft.source_message_ids):
                raise ValueError("Compact A-MEM note contains duplicate sources")
            if set(draft.source_message_ids) - set(source_order):
                raise ValueError("Compact A-MEM note references an outside source")
            chronological_sources = tuple(
                sorted(draft.source_message_ids, key=source_order.__getitem__)
            )
            context, keywords, tags = cls._validate_metadata(
                context=draft.context,
                keywords=draft.keywords,
                tags=draft.tags,
            )
            validated.append(
                replace(
                    draft,
                    content=draft.content.strip(),
                    context=context,
                    keywords=keywords,
                    tags=tags,
                    source_message_ids=chronological_sources,
                )
            )
        return tuple(validated)

    @staticmethod
    def _validate_episodes(
        episodes: Sequence[CompactAMemEpisode],
    ) -> tuple[CompactAMemEpisode, ...]:
        normalized = tuple(episodes)
        if any(not isinstance(episode, CompactAMemEpisode) for episode in normalized):
            raise TypeError("Compact A-MEM ingest contains an invalid episode")
        indices = [episode.index for episode in normalized]
        if len(set(indices)) != len(indices):
            raise ValueError("Compact A-MEM ingest contains duplicate episode indices")
        if indices != sorted(indices):
            raise ValueError("Compact A-MEM episodes must be ordered by index")
        if any(not episode.entries for episode in normalized):
            raise ValueError("Compact A-MEM episode cannot be empty")
        return normalized


def _validate_entry(entry: AMemHistoryEntry) -> CompactAMemEpisodeEntry:
    if not isinstance(entry, AMemHistoryEntry):
        raise TypeError("Compact A-MEM history contains an invalid entry")
    if entry.source_message_id < 0:
        raise ValueError("Compact A-MEM source_message_id cannot be negative")
    if (
        not entry.timestamp.strip()
        or not entry.speaker.strip()
        or not entry.content.strip()
    ):
        raise ValueError("Compact A-MEM history fields cannot be empty")
    return CompactAMemEpisodeEntry(
        source_message_id=entry.source_message_id,
        timestamp=entry.timestamp,
        speaker=entry.speaker,
        content=entry.content,
    )


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid Compact A-MEM timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("Compact A-MEM timestamp must include a timezone")
    return parsed


def _render_entry(entry: CompactAMemEpisodeEntry) -> str:
    return (
        f"source_message_id={entry.source_message_id}\n"
        f"timestamp={entry.timestamp}\n"
        f"speaker={entry.speaker}\n"
        f"content={entry.content}"
    )
