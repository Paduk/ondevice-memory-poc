from __future__ import annotations

import math
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any

from palmclaw_ubuntu.contracts import AMemModel, EmbeddingModel
from palmclaw_ubuntu.memory import cosine_similarity
from palmclaw_ubuntu.models import (
    AMemConstructionResponse,
    AMemEvolutionDecision,
    AMemNeighbor,
    AMemNeighborUpdate,
    AMemNote,
    AMemNoteVersion,
    EmbeddingResponse,
    ModelUsage,
)
from palmclaw_ubuntu.privacy import redact_secrets
from palmclaw_ubuntu.storage import SQLiteRepository

AMEM_NOTE_CONTENT_FORMAT_VERSION = "amem-note-content-v1"
AMEM_METADATA_EMBEDDING_FORMAT_VERSION = "amem-metadata-embedding-v1"
AMEM_LINK_CANDIDATE_LIMIT = 5
AMEM_MAX_CONTEXT_CHARS = 4_000
AMEM_MAX_METADATA_ITEMS = 32
AMEM_MAX_METADATA_ITEM_CHARS = 128


@dataclass(frozen=True)
class AMemHistoryEntry:
    source_message_id: int
    timestamp: str
    speaker: str
    content: str


@dataclass(frozen=True)
class AMemIngestionResult:
    notes: tuple[AMemNote, ...]
    processed_source_ids: tuple[int, ...]
    skipped_source_ids: tuple[int, ...]


def format_amem_note_content(entry: AMemHistoryEntry) -> str:
    """Render immutable source content for a construction prompt."""

    return (
        f"format={AMEM_NOTE_CONTENT_FORMAT_VERSION}\n"
        f"timestamp={entry.timestamp}\n"
        f"speaker={entry.speaker}\n"
        f"content={entry.content}"
    )


def format_amem_metadata_embedding_text(
    note: AMemNote,
    *,
    context: str,
    keywords: Sequence[str],
    tags: Sequence[str],
) -> str:
    """Render source plus evolvable metadata for embedding."""

    return (
        f"format={AMEM_METADATA_EMBEDDING_FORMAT_VERSION}\n"
        f"timestamp={note.timestamp}\n"
        f"speaker={note.speaker}\n"
        f"content={note.content}\n"
        f"context={context}\n"
        f"keywords={', '.join(keywords)}\n"
        f"tags={', '.join(tags)}"
    )


class AMemEngine:
    def __init__(
        self,
        repository: SQLiteRepository,
        model: AMemModel,
        embedding_model: EmbeddingModel,
        *,
        candidate_limit: int = AMEM_LINK_CANDIDATE_LIMIT,
        evolution_similarity_threshold: float | None = None,
        memory_input_cost_per_million: float = 0.0,
        memory_output_cost_per_million: float = 0.0,
        embedding_input_cost_per_million: float = 0.0,
    ):
        if candidate_limit < 1:
            raise ValueError("A-MEM candidate limit must be positive")
        if (
            evolution_similarity_threshold is not None
            and not -1.0 <= evolution_similarity_threshold <= 1.0
        ):
            raise ValueError(
                "A-MEM evolution similarity threshold must be between -1 and 1"
            )
        self.repository = repository
        self.model = model
        self.embedding_model = embedding_model
        self.candidate_limit = min(candidate_limit, AMEM_LINK_CANDIDATE_LIMIT)
        self.evolution_similarity_threshold = evolution_similarity_threshold
        self.memory_input_cost_per_million = memory_input_cost_per_million
        self.memory_output_cost_per_million = memory_output_cost_per_million
        self.embedding_input_cost_per_million = (
            embedding_input_cost_per_million
        )

    def ingest(
        self,
        session_id: str,
        entries: Sequence[AMemHistoryEntry],
    ) -> AMemIngestionResult:
        self.repository.require_session(session_id)
        ordered = self._validate_chronology(entries)
        existing = self.repository.list_amem_notes(session_id)
        existing_by_source = {note.source_message_id: note for note in existing}
        existing_keys = {
            (note.timestamp, note.source_message_id) for note in existing
        }
        latest_existing_key = max(existing_keys, default=None)
        processed: list[int] = []
        skipped: list[int] = []

        for entry in ordered:
            if entry.source_message_id in existing_by_source:
                note = existing_by_source[entry.source_message_id]
                if self._source_identity(note) != self._entry_identity(entry):
                    raise ValueError(
                        "A-MEM source ID was already stored with different content"
                    )
                skipped.append(entry.source_message_id)
                continue
            entry_key = (entry.timestamp, entry.source_message_id)
            if latest_existing_key is not None and entry_key <= latest_existing_key:
                raise ValueError(
                    "A-MEM cannot backfill a source before the latest stored note"
                )
            note = self._ingest_entry(session_id, entry)
            existing_by_source[entry.source_message_id] = note
            latest_existing_key = entry_key
            processed.append(entry.source_message_id)

        return AMemIngestionResult(
            notes=self.repository.list_amem_notes(session_id),
            processed_source_ids=tuple(processed),
            skipped_source_ids=tuple(skipped),
        )

    def _ingest_entry(
        self,
        session_id: str,
        entry: AMemHistoryEntry,
    ) -> AMemNote:
        run_id = self.repository.begin_amem_construction(
            session_id=session_id,
            source_message_id=entry.source_message_id,
            backend=self.model.backend,
            model_id=self.model.model_id,
            prompt_version=self.model.construction_prompt_version,
            schema_version=self.model.construction_schema_version,
            idempotency_key=(
                f"amem-construction:{session_id}:{entry.source_message_id}:"
                f"{uuid.uuid4()}"
            ),
        )
        try:
            construction = self._construct(
                entry,
                session_id=session_id,
                run_id=run_id,
            )
            prospective_note = AMemNote(
                id=str(uuid.uuid4()),
                session_id=session_id,
                source_message_id=entry.source_message_id,
                timestamp=entry.timestamp,
                speaker=entry.speaker,
                content=redact_secrets(entry.content),
                status="active",
                created_at="",
            )
            prospective_version = AMemNoteVersion(
                id=str(uuid.uuid4()),
                note_id=prospective_note.id,
                version=1,
                context=construction.context,
                keywords=construction.keywords,
                tags=construction.tags,
                status="active",
                supersedes_id=None,
                created_by_event_id=None,
                created_at="",
            )
            initial_vector = self._embed_metadata(
                prospective_note,
                context=construction.context,
                keywords=construction.keywords,
                tags=construction.tags,
                run_id=run_id,
            )
            neighbors = self._link_candidates(session_id, initial_vector)
            decision = self._evolve(
                AMemNeighbor(
                    note=prospective_note,
                    version=prospective_version,
                    similarity_score=1.0,
                ),
                neighbors,
                session_id=session_id,
                run_id=run_id,
            )
            final_keywords = (
                decision.new_note_keywords
                if decision and decision.new_note_keywords
                else construction.keywords
            )
            final_tags = (
                decision.new_note_tags
                if decision and decision.new_note_tags
                else construction.tags
            )
            metadata_changed = (
                final_keywords != construction.keywords
                or final_tags != construction.tags
            )
            if metadata_changed:
                final_vector = self._embed_metadata(
                    prospective_note,
                    context=construction.context,
                    keywords=final_keywords,
                    tags=final_tags,
                    run_id=run_id,
                )
            else:
                final_vector = initial_vector
            neighbor_vectors = self._updated_neighbor_vectors(
                decision,
                neighbors,
                run_id=run_id,
            )
            return self._commit_entry(
                run_id=run_id,
                entry=entry,
                prospective_note=prospective_note,
                construction=construction,
                final_keywords=final_keywords,
                final_tags=final_tags,
                final_vector=final_vector,
                neighbors=neighbors,
                decision=decision,
                neighbor_vectors=neighbor_vectors,
            )
        except BaseException as exc:
            self._record_failure(run_id, exc)
            raise

    def _construct(
        self,
        entry: AMemHistoryEntry,
        *,
        session_id: str,
        run_id: str,
    ) -> AMemConstructionResponse:
        started = time.monotonic()
        try:
            response = self.model.construct(
                {
                    "source_message_id": entry.source_message_id,
                    "timestamp": entry.timestamp,
                    "speaker": entry.speaker,
                    "content": entry.content,
                    "formatted_content": format_amem_note_content(entry),
                    "format_version": AMEM_NOTE_CONTENT_FORMAT_VERSION,
                }
            )
        except BaseException as exc:
            self._record_provider_call(
                session_id=session_id,
                role="amem_construction",
                backend=self.model.backend,
                model_id=self.model.model_id,
                prompt_version=self.model.construction_prompt_version,
                schema_version=self.model.construction_schema_version,
                started=started,
                usage=ModelUsage(),
                response_id=None,
                metadata={"amem_construction_run_id": run_id},
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        if not isinstance(response, AMemConstructionResponse):
            raise TypeError("A-MEM construction returned an invalid response")
        self._record_provider_call(
            session_id=session_id,
            role="amem_construction",
            backend=self.model.backend,
            model_id=self.model.model_id,
            prompt_version=self.model.construction_prompt_version,
            schema_version=self.model.construction_schema_version,
            started=started,
            usage=response.usage,
            response_id=response.response_id,
            metadata={
                **dict(response.metadata),
                "amem_construction_run_id": run_id,
            },
        )
        context, keywords, tags = self._validate_metadata(
            context=response.context,
            keywords=response.keywords,
            tags=response.tags,
        )
        return replace(
            response,
            context=context,
            keywords=keywords,
            tags=tags,
        )

    def _evolve(
        self,
        new_note: AMemNeighbor,
        neighbors: tuple[AMemNeighbor, ...],
        *,
        session_id: str,
        run_id: str,
    ) -> AMemEvolutionDecision | None:
        if not neighbors:
            return None
        if (
            self.evolution_similarity_threshold is not None
            and neighbors[0].similarity_score
            < self.evolution_similarity_threshold
        ):
            return None
        started = time.monotonic()
        try:
            decision = self.model.evolve(new_note, neighbors)
        except BaseException as exc:
            recovery_reason = self._style_evolution_recovery_reason(exc)
            self._record_provider_call(
                session_id=session_id,
                role="amem_evolution",
                backend=self.model.backend,
                model_id=self.model.model_id,
                prompt_version=self.model.evolution_prompt_version,
                schema_version=self.model.evolution_schema_version,
                started=started,
                usage=ModelUsage(),
                response_id=None,
                metadata={
                    "amem_construction_run_id": run_id,
                    "recovered_as_no_evolution": recovery_reason is not None,
                    "recovery_reason": recovery_reason,
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            if recovery_reason is not None:
                return None
            raise
        if not isinstance(decision, AMemEvolutionDecision):
            raise TypeError("A-MEM evolution returned an invalid response")
        self._record_provider_call(
            session_id=session_id,
            role="amem_evolution",
            backend=self.model.backend,
            model_id=self.model.model_id,
            prompt_version=self.model.evolution_prompt_version,
            schema_version=self.model.evolution_schema_version,
            started=started,
            usage=decision.usage,
            response_id=decision.response_id,
            metadata={
                **dict(decision.metadata),
                "amem_construction_run_id": run_id,
                "candidate_count": len(neighbors),
            },
        )
        candidate_ids = {neighbor.note.id for neighbor in neighbors}
        linked_ids = self._validate_candidate_ids(
            "linked_note_ids",
            decision.linked_note_ids,
            candidate_ids,
            new_note.note.id,
        )
        update_ids = self._validate_candidate_ids(
            "neighbor_updates",
            tuple(update.note_id for update in decision.neighbor_updates),
            candidate_ids,
            new_note.note.id,
        )
        updates: list[AMemNeighborUpdate] = []
        for update in decision.neighbor_updates:
            context, keywords, tags = self._validate_metadata(
                context=update.context,
                keywords=update.keywords,
                tags=update.tags,
            )
            updates.append(
                replace(
                    update,
                    context=context,
                    keywords=keywords,
                    tags=tags,
                )
            )
        new_keywords = self._validate_items(
            "new_note_keywords",
            decision.new_note_keywords,
        )
        new_tags = self._validate_items("new_note_tags", decision.new_note_tags)
        has_payload = bool(linked_ids or update_ids or new_keywords or new_tags)
        if not decision.should_evolve and has_payload:
            raise ValueError(
                "A-MEM decision cannot include updates when should_evolve is false"
            )
        return replace(
            decision,
            linked_note_ids=linked_ids,
            new_note_keywords=new_keywords,
            new_note_tags=new_tags,
            neighbor_updates=tuple(updates),
        )

    def _style_evolution_recovery_reason(
        self,
        exc: BaseException,
    ) -> str | None:
        if self.evolution_similarity_threshold is None:
            return None
        if (
            isinstance(exc, ValueError)
            and "non-candidate note ID" in str(exc)
        ):
            return "non_candidate_note_id"
        if type(exc).__name__ == "ValidationError":
            return "invalid_structured_output"
        if isinstance(exc, RuntimeError) and (
            "provider returned status incomplete" in str(exc)
            or "returned no parsed output" in str(exc)
        ):
            return "incomplete_structured_output"
        return None

    def _link_candidates(
        self,
        session_id: str,
        new_vector: tuple[float, ...],
    ) -> tuple[AMemNeighbor, ...]:
        note_versions = []
        for note in self.repository.list_amem_notes(session_id):
            version = self.repository.latest_amem_note_version(note.id)
            if version is not None:
                note_versions.append((note, version))
        embeddings = self.repository.amem_note_embeddings(
            tuple(version.id for _, version in note_versions),
            model_id=self.embedding_model.model_id,
            dimensions=self.embedding_model.dimensions,
        )
        candidates = [
            AMemNeighbor(
                note=note,
                version=version,
                similarity_score=cosine_similarity(
                    new_vector,
                    embeddings[version.id],
                ),
            )
            for note, version in note_versions
            if version.id in embeddings
        ]
        candidates.sort(
            key=lambda item: (
                -item.similarity_score,
                item.note.timestamp,
                item.note.source_message_id,
                item.note.id,
            )
        )
        return tuple(candidates[: self.candidate_limit])

    def _updated_neighbor_vectors(
        self,
        decision: AMemEvolutionDecision | None,
        neighbors: tuple[AMemNeighbor, ...],
        *,
        run_id: str,
    ) -> dict[str, tuple[float, ...]]:
        if decision is None:
            return {}
        notes = {neighbor.note.id: neighbor.note for neighbor in neighbors}
        return {
            update.note_id: self._embed_metadata(
                notes[update.note_id],
                context=update.context,
                keywords=update.keywords,
                tags=update.tags,
                run_id=run_id,
            )
            for update in decision.neighbor_updates
        }

    def _commit_entry(
        self,
        *,
        run_id: str,
        entry: AMemHistoryEntry,
        prospective_note: AMemNote,
        construction: AMemConstructionResponse,
        final_keywords: tuple[str, ...],
        final_tags: tuple[str, ...],
        final_vector: tuple[float, ...],
        neighbors: tuple[AMemNeighbor, ...],
        decision: AMemEvolutionDecision | None,
        neighbor_vectors: Mapping[str, tuple[float, ...]],
    ) -> AMemNote:
        with self.repository.amem_transaction():
            note = self.repository.insert_amem_note(
                session_id=prospective_note.session_id,
                source_message_id=entry.source_message_id,
                timestamp=entry.timestamp,
                speaker=entry.speaker,
                content=entry.content,
                note_id=prospective_note.id,
            )
            event_id = None
            if decision is not None:
                event_id = self.repository.begin_amem_evolution(
                    session_id=note.session_id,
                    new_note_id=note.id,
                    backend=self.model.backend,
                    model_id=self.model.model_id,
                    prompt_version=self.model.evolution_prompt_version,
                    schema_version=self.model.evolution_schema_version,
                    idempotency_key=f"amem-evolution:{run_id}",
                )
            version = self.repository.insert_amem_note_version(
                note_id=note.id,
                context=construction.context,
                keywords=final_keywords,
                tags=final_tags,
                created_by_event_id=event_id,
            )
            self.repository.store_amem_note_embedding(
                note_version_id=version.id,
                model_id=self.embedding_model.model_id,
                dimensions=self.embedding_model.dimensions,
                vector=final_vector,
            )
            neighbor_by_id = {neighbor.note.id: neighbor for neighbor in neighbors}
            if decision is not None and event_id is not None:
                for update in decision.neighbor_updates:
                    updated_version = self.repository.insert_amem_note_version(
                        note_id=update.note_id,
                        context=update.context,
                        keywords=update.keywords,
                        tags=update.tags,
                        created_by_event_id=event_id,
                    )
                    self.repository.store_amem_note_embedding(
                        note_version_id=updated_version.id,
                        model_id=self.embedding_model.model_id,
                        dimensions=self.embedding_model.dimensions,
                        vector=neighbor_vectors[update.note_id],
                    )
                for linked_id in decision.linked_note_ids:
                    self.repository.insert_amem_link(
                        session_id=note.session_id,
                        note_a_id=note.id,
                        note_b_id=linked_id,
                        created_by_note_id=note.id,
                        evolution_event_id=event_id,
                        similarity_score=neighbor_by_id[
                            linked_id
                        ].similarity_score,
                        decision="evolution",
                    )
                self.repository.complete_amem_evolution(
                    event_id,
                    decision=self._decision_trace(decision),
                    usage=decision.usage,
                )
            self.repository.complete_amem_construction(
                run_id,
                note_id=note.id,
                output={
                    "context": construction.context,
                    "keywords": list(construction.keywords),
                    "tags": list(construction.tags),
                    "response_id": construction.response_id,
                    "metadata": dict(construction.metadata),
                },
                usage=construction.usage,
            )
        return note

    def _embed_metadata(
        self,
        note: AMemNote,
        *,
        context: str,
        keywords: Sequence[str],
        tags: Sequence[str],
        run_id: str | None = None,
    ) -> tuple[float, ...]:
        started = time.monotonic()
        try:
            response = self.embedding_model.embed(
                (
                    format_amem_metadata_embedding_text(
                        note,
                        context=context,
                        keywords=keywords,
                        tags=tags,
                    ),
                )
            )
        except BaseException as exc:
            self._record_provider_call(
                session_id=note.session_id,
                role="amem_embedding",
                backend=self.embedding_model.backend,
                model_id=self.embedding_model.model_id,
                prompt_version=AMEM_METADATA_EMBEDDING_FORMAT_VERSION,
                schema_version=None,
                started=started,
                usage=ModelUsage(),
                response_id=None,
                metadata={"amem_construction_run_id": run_id},
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._record_provider_call(
            session_id=note.session_id,
            role="amem_embedding",
            backend=self.embedding_model.backend,
            model_id=self.embedding_model.model_id,
            prompt_version=AMEM_METADATA_EMBEDDING_FORMAT_VERSION,
            schema_version=None,
            started=started,
            usage=response.usage,
            response_id=response.response_id,
            metadata={
                **dict(response.metadata),
                "amem_construction_run_id": run_id,
            },
        )
        return self._validate_embedding_response(response)

    def _record_provider_call(
        self,
        *,
        session_id: str,
        role: str,
        backend: str,
        model_id: str,
        prompt_version: str,
        schema_version: str | None,
        started: float,
        usage: ModelUsage,
        response_id: str | None,
        metadata: Mapping[str, Any],
        error: str | None = None,
    ) -> None:
        if backend != "openai":
            estimated_cost = 0.0
        elif role == "amem_embedding":
            estimated_cost = (
                usage.input_tokens
                * self.embedding_input_cost_per_million
                / 1_000_000
            )
        else:
            estimated_cost = (
                usage.input_tokens * self.memory_input_cost_per_million
                + usage.output_tokens * self.memory_output_cost_per_million
            ) / 1_000_000
        self.repository.record_model_call(
            session_id=session_id,
            role=role,
            backend=backend,
            model_id=model_id,
            prompt_version=prompt_version,
            schema_version=schema_version,
            latency_ms=round((time.monotonic() - started) * 1_000),
            usage=usage,
            response_id=response_id,
            metadata={
                **dict(metadata),
                "estimated_cost_usd": estimated_cost,
            },
            error=error,
        )

    def _validate_embedding_response(
        self,
        response: EmbeddingResponse,
    ) -> tuple[float, ...]:
        if not isinstance(response, EmbeddingResponse) or len(response.vectors) != 1:
            raise ValueError("A-MEM embedding must return exactly one vector")
        vector = tuple(float(value) for value in response.vectors[0])
        if len(vector) != self.embedding_model.dimensions:
            raise ValueError("A-MEM embedding has unexpected dimensions")
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("A-MEM embedding contains a non-finite value")
        return vector

    @classmethod
    def _validate_metadata(
        cls,
        *,
        context: str,
        keywords: Sequence[str],
        tags: Sequence[str],
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        if not isinstance(context, str) or not context.strip():
            raise ValueError("A-MEM context cannot be empty")
        normalized_context = context.strip()
        if len(normalized_context) > AMEM_MAX_CONTEXT_CHARS:
            raise ValueError("A-MEM context is too long")
        return (
            normalized_context,
            cls._validate_items("keywords", keywords),
            cls._validate_items("tags", tags),
        )

    @staticmethod
    def _validate_items(
        name: str,
        values: Sequence[str],
    ) -> tuple[str, ...]:
        if isinstance(values, str):
            raise ValueError(f"A-MEM {name} must be an array")
        if len(values) > AMEM_MAX_METADATA_ITEMS:
            raise ValueError(f"A-MEM {name} has too many items")
        normalized = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"A-MEM {name} contains an empty item")
            item = value.strip()
            if len(item) > AMEM_MAX_METADATA_ITEM_CHARS:
                raise ValueError(f"A-MEM {name} item is too long")
            normalized.append(item)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"A-MEM {name} contains duplicate items")
        return tuple(normalized)

    @staticmethod
    def _validate_candidate_ids(
        name: str,
        values: Sequence[str],
        candidate_ids: set[str],
        new_note_id: str,
    ) -> tuple[str, ...]:
        normalized = tuple(values)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"A-MEM {name} contains duplicate note IDs")
        if new_note_id in normalized:
            raise ValueError(f"A-MEM {name} cannot reference the new note")
        outside = set(normalized) - candidate_ids
        if outside:
            raise ValueError(f"A-MEM {name} references a non-candidate note")
        return normalized

    @staticmethod
    def _validate_chronology(
        entries: Sequence[AMemHistoryEntry],
    ) -> tuple[AMemHistoryEntry, ...]:
        normalized = tuple(entries)
        if any(not isinstance(entry, AMemHistoryEntry) for entry in normalized):
            raise TypeError("A-MEM history contains an invalid entry")
        if any(entry.source_message_id < 0 for entry in normalized):
            raise ValueError("A-MEM source_message_id cannot be negative")
        if any(
            not entry.timestamp.strip()
            or not entry.speaker.strip()
            or not entry.content.strip()
            for entry in normalized
        ):
            raise ValueError("A-MEM history fields cannot be empty")
        source_ids = [entry.source_message_id for entry in normalized]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("A-MEM history contains duplicate source IDs")
        ordered = tuple(
            sorted(
                normalized,
                key=lambda entry: (entry.timestamp, entry.source_message_id),
            )
        )
        if ordered != normalized:
            raise ValueError("A-MEM history must be in chronological order")
        return ordered

    @staticmethod
    def _entry_identity(entry: AMemHistoryEntry) -> tuple[str, str, str]:
        return (entry.timestamp, entry.speaker, redact_secrets(entry.content))

    @staticmethod
    def _source_identity(note: AMemNote) -> tuple[str, str, str]:
        return (note.timestamp, note.speaker, note.content)

    @staticmethod
    def _decision_trace(decision: AMemEvolutionDecision) -> dict[str, Any]:
        return {
            "should_evolve": decision.should_evolve,
            "linked_note_ids": list(decision.linked_note_ids),
            "new_note_keywords": list(decision.new_note_keywords),
            "new_note_tags": list(decision.new_note_tags),
            "neighbor_updates": [
                {
                    "note_id": update.note_id,
                    "context": update.context,
                    "keywords": list(update.keywords),
                    "tags": list(update.tags),
                }
                for update in decision.neighbor_updates
            ],
            "response_id": decision.response_id,
            "metadata": dict(decision.metadata),
        }

    def _record_failure(self, run_id: str, error: BaseException) -> None:
        with suppress(KeyError):
            self.repository.fail_amem_construction(
                run_id,
                error=f"{type(error).__name__}: {error}",
            )
