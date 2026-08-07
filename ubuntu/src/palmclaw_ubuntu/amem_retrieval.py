from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from palmclaw_ubuntu.contracts import EmbeddingModel
from palmclaw_ubuntu.memory import cosine_similarity
from palmclaw_ubuntu.models import (
    AMemNote,
    AMemNoteVersion,
    AMemRetrievalResult,
    EmbeddingResponse,
    ModelUsage,
)
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tokens import TokenCounter

AMEM_RETRIEVAL_MODE = "embedding_linked"
AMEM_MAX_SEEDS = 10


@dataclass
class _RetrievalCandidate:
    note: AMemNote
    version: AMemNoteVersion
    source: str
    embedding_score: float
    query_rank: int | None
    eligible: bool
    selected: bool = False
    exclusion_reason: str | None = None
    rendered: str = ""
    rendered_tokens: int = 0


def render_amem_note(
    note: AMemNote,
    version: AMemNoteVersion,
    *,
    source: str,
) -> str:
    """Render one current note without internal IDs, scores, or schema fields."""

    if source not in {"seed", "linked"}:
        raise ValueError(f"Unsupported A-MEM retrieval source: {source}")
    keywords = ", ".join(version.keywords) or "(none)"
    tags = ", ".join(version.tags) or "(none)"
    return (
        f"[A-MEM {source}]\n"
        f"Timestamp: {note.timestamp}\n"
        f"Speaker: {note.speaker}\n"
        f"Original: {note.content}\n"
        f"Current context: {version.context}\n"
        f"Keywords: {keywords}\n"
        f"Tags: {tags}"
    )


class AMemRetriever:
    def __init__(
        self,
        repository: SQLiteRepository,
        embedding_model: EmbeddingModel,
        *,
        top_k: int = AMEM_MAX_SEEDS,
        token_budget: int = 2_000,
        token_counter: TokenCounter | None = None,
        embedding_input_cost_per_million: float = 0.0,
    ):
        self._validate_limits(top_k, token_budget)
        self.repository = repository
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.token_budget = token_budget
        self.token_counter = token_counter or TokenCounter()
        self.embedding_input_cost_per_million = (
            embedding_input_cost_per_million
        )

    def retrieve(
        self,
        session_id: str,
        query: str,
        *,
        top_k: int | None = None,
    ) -> AMemRetrievalResult:
        selected_top_k = self.top_k if top_k is None else top_k
        self._validate_limits(selected_top_k, self.token_budget)
        if not query.strip():
            raise ValueError("A-MEM retrieval query cannot be empty")
        started = time.monotonic()
        run_id = self.repository.begin_amem_retrieval(
            session_id=session_id,
            query=query,
            mode=AMEM_RETRIEVAL_MODE,
            top_k=selected_top_k,
            token_budget=self.token_budget,
            embedding_model_id=self.embedding_model.model_id,
            embedding_dimensions=self.embedding_model.dimensions,
        )
        try:
            return self._retrieve(
                run_id=run_id,
                session_id=session_id,
                query=query,
                top_k=selected_top_k,
                started=started,
            )
        except Exception as exc:
            latency_ms = self._elapsed_ms(started)
            self.repository.fail_amem_retrieval(
                run_id,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=latency_ms,
            )
            return AMemRetrievalResult(
                content="",
                run_id=run_id,
                metadata={
                    "memory_strategy": "amem",
                    "retrieval_run_id": run_id,
                    "retrieval_mode": AMEM_RETRIEVAL_MODE,
                    "retrieval_status": "failed",
                    "retrieval_error": f"{type(exc).__name__}: {exc}",
                    "retrieval_top_k": selected_top_k,
                    "retrieval_token_budget": self.token_budget,
                    "retrieval_selected_count": 0,
                    "retrieval_latency_ms": latency_ms,
                },
            )

    def _retrieve(
        self,
        *,
        run_id: str,
        session_id: str,
        query: str,
        top_k: int,
        started: float,
    ) -> AMemRetrievalResult:
        note_versions = self._latest_note_versions(session_id)
        if not note_versions:
            latency_ms = self._elapsed_ms(started)
            self.repository.complete_amem_retrieval(
                run_id,
                (),
                selected_count=0,
                selected_tokens=0,
                latency_ms=latency_ms,
            )
            return self._result(
                run_id=run_id,
                content="",
                selected=(),
                top_k=top_k,
                candidate_count=0,
                seed_count=0,
                linked_count=0,
                selected_tokens=0,
                latency_ms=latency_ms,
            )

        embeddings = self.repository.amem_note_embeddings(
            tuple(version.id for _, version in note_versions.values()),
            model_id=self.embedding_model.model_id,
            dimensions=self.embedding_model.dimensions,
        )
        query_vector = self._query_embedding(
            session_id,
            run_id,
            query,
        )
        scores = {
            note_id: cosine_similarity(query_vector, embeddings[version.id])
            for note_id, (_, version) in note_versions.items()
            if version.id in embeddings
        }
        ranked_ids = self._rank_ids(note_versions, scores)
        seed_ids = tuple(ranked_ids[:top_k])
        seed_id_set = set(seed_ids)
        candidates = self._initial_candidates(
            note_versions,
            scores,
            ranked_ids,
            seed_id_set,
        )
        linked_ids = self._expand_links(
            session_id,
            seed_ids,
            seed_id_set,
            candidates,
            note_versions,
            scores,
        )
        allocation_ids = (*seed_ids, *linked_ids)
        selected, rendered_parts, selected_tokens = self._allocate(
            allocation_ids,
            candidates,
        )
        trace_order = self._trace_order(allocation_ids, candidates)
        trace_candidates = [
            {
                "note_id": candidate.note.id,
                "note_version_id": candidate.version.id,
                "source": candidate.source,
                "rank": rank,
                "embedding_score": candidate.embedding_score,
                "combined_score": candidate.embedding_score,
                "selected": candidate.selected,
                "exclusion_reason": candidate.exclusion_reason,
                "token_count": candidate.rendered_tokens,
            }
            for rank, candidate in enumerate(trace_order, start=1)
        ]
        latency_ms = self._elapsed_ms(started)
        self.repository.complete_amem_retrieval(
            run_id,
            trace_candidates,
            selected_count=len(selected),
            selected_tokens=selected_tokens,
            latency_ms=latency_ms,
        )
        return self._result(
            run_id=run_id,
            content="\n\n".join(rendered_parts),
            selected=tuple(selected),
            top_k=top_k,
            candidate_count=len(candidates),
            seed_count=len(seed_ids),
            linked_count=len(linked_ids),
            selected_tokens=selected_tokens,
            latency_ms=latency_ms,
        )

    def _latest_note_versions(
        self,
        session_id: str,
    ) -> dict[str, tuple[AMemNote, AMemNoteVersion]]:
        result = {}
        for note in self.repository.list_amem_notes(session_id):
            version = self.repository.latest_amem_note_version(note.id)
            if version is not None:
                result[note.id] = (note, version)
        return result

    def _query_embedding(
        self,
        session_id: str,
        run_id: str,
        query: str,
    ) -> tuple[float, ...]:
        started = time.monotonic()
        try:
            response = self.embedding_model.embed((query,))
        except BaseException as exc:
            self._record_embedding_call(
                session_id=session_id,
                run_id=run_id,
                started=started,
                usage=ModelUsage(),
                response_id=None,
                metadata={},
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._record_embedding_call(
            session_id=session_id,
            run_id=run_id,
            started=started,
            usage=response.usage,
            response_id=response.response_id,
            metadata=response.metadata,
        )
        return self._validate_query_embedding(response)

    def _record_embedding_call(
        self,
        *,
        session_id: str,
        run_id: str,
        started: float,
        usage: ModelUsage,
        response_id: str | None,
        metadata: Mapping[str, Any],
        error: str | None = None,
    ) -> None:
        self.repository.record_model_call(
            session_id=session_id,
            role="amem_retrieval_embedding",
            backend=self.embedding_model.backend,
            model_id=self.embedding_model.model_id,
            prompt_version="amem-retrieval-query-v1",
            schema_version=None,
            latency_ms=self._elapsed_ms(started),
            usage=usage,
            response_id=response_id,
            metadata={
                **dict(metadata),
                "amem_retrieval_run_id": run_id,
                "estimated_cost_usd": (
                    usage.input_tokens
                    * self.embedding_input_cost_per_million
                    / 1_000_000
                    if self.embedding_model.backend == "openai"
                    else 0.0
                ),
            },
            error=error,
        )

    def _validate_query_embedding(
        self,
        response: EmbeddingResponse,
    ) -> tuple[float, ...]:
        if not isinstance(response, EmbeddingResponse) or len(response.vectors) != 1:
            raise ValueError("A-MEM query embedding must return one vector")
        vector = tuple(float(value) for value in response.vectors[0])
        if len(vector) != self.embedding_model.dimensions:
            raise ValueError("A-MEM query embedding has unexpected dimensions")
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("A-MEM query embedding contains a non-finite value")
        return vector

    @staticmethod
    def _rank_ids(
        note_versions: dict[str, tuple[AMemNote, AMemNoteVersion]],
        scores: dict[str, float],
    ) -> list[str]:
        ranked = list(scores)
        ranked.sort()
        ranked.sort(
            key=lambda note_id: note_versions[note_id][0].timestamp,
            reverse=True,
        )
        ranked.sort(key=lambda note_id: scores[note_id], reverse=True)
        return ranked

    @staticmethod
    def _initial_candidates(
        note_versions: dict[str, tuple[AMemNote, AMemNoteVersion]],
        scores: dict[str, float],
        ranked_ids: Sequence[str],
        seed_ids: set[str],
    ) -> dict[str, _RetrievalCandidate]:
        query_ranks = {
            note_id: rank for rank, note_id in enumerate(ranked_ids, start=1)
        }
        return {
            note_id: _RetrievalCandidate(
                note=note,
                version=version,
                source="seed",
                embedding_score=scores.get(note_id, 0.0),
                query_rank=query_ranks.get(note_id),
                eligible=note_id in seed_ids,
                exclusion_reason=(
                    None
                    if note_id in seed_ids
                    else "top_k"
                    if note_id in scores
                    else "missing_embedding"
                ),
            )
            for note_id, (note, version) in note_versions.items()
        }

    def _expand_links(
        self,
        session_id: str,
        seed_ids: Sequence[str],
        seed_id_set: set[str],
        candidates: dict[str, _RetrievalCandidate],
        note_versions: dict[str, tuple[AMemNote, AMemNoteVersion]],
        scores: dict[str, float],
    ) -> tuple[str, ...]:
        linked: list[str] = []
        seen = set(seed_id_set)
        for seed_id in seed_ids:
            neighbor_ids = {
                link.right_note_id
                if link.left_note_id == seed_id
                else link.left_note_id
                for link in self.repository.list_amem_links(
                    session_id,
                    note_id=seed_id,
                )
            }
            ordered = self._rank_linked_ids(neighbor_ids, note_versions, scores)
            for note_id in ordered:
                if note_id in seen or note_id not in candidates:
                    continue
                seen.add(note_id)
                linked.append(note_id)
                candidate = candidates[note_id]
                candidate.source = "linked"
                candidate.eligible = True
                candidate.exclusion_reason = None
        return tuple(linked)

    @staticmethod
    def _rank_linked_ids(
        note_ids: set[str],
        note_versions: dict[str, tuple[AMemNote, AMemNoteVersion]],
        scores: dict[str, float],
    ) -> list[str]:
        ranked = [note_id for note_id in note_ids if note_id in note_versions]
        ranked.sort()
        ranked.sort(
            key=lambda note_id: note_versions[note_id][0].timestamp,
            reverse=True,
        )
        ranked.sort(key=lambda note_id: scores.get(note_id, 0.0), reverse=True)
        return ranked

    def _allocate(
        self,
        allocation_ids: Sequence[str],
        candidates: dict[str, _RetrievalCandidate],
    ) -> tuple[
        list[_RetrievalCandidate],
        list[str],
        int,
    ]:
        selected: list[_RetrievalCandidate] = []
        rendered_parts: list[str] = []
        used_tokens = 0
        for note_id in allocation_ids:
            candidate = candidates[note_id]
            rendered = render_amem_note(
                candidate.note,
                candidate.version,
                source=candidate.source,
            )
            full_tokens = self.token_counter.count(rendered)
            separator_tokens = (
                self.token_counter.count("\n\n") if rendered_parts else 0
            )
            remaining = self.token_budget - used_tokens - separator_tokens
            if remaining <= 0:
                candidate.exclusion_reason = "token_budget"
                candidate.rendered_tokens = full_tokens
                continue
            bounded = self.token_counter.truncate(rendered, remaining)
            bounded_tokens = self.token_counter.count(bounded)
            if not bounded or bounded_tokens < 1:
                candidate.exclusion_reason = "token_budget"
                candidate.rendered_tokens = full_tokens
                continue
            candidate.selected = True
            candidate.rendered = bounded
            candidate.rendered_tokens = bounded_tokens
            candidate.exclusion_reason = None
            selected.append(candidate)
            rendered_parts.append(bounded)
            used_tokens += separator_tokens + bounded_tokens
        return selected, rendered_parts, used_tokens

    @staticmethod
    def _trace_order(
        allocation_ids: Sequence[str],
        candidates: dict[str, _RetrievalCandidate],
    ) -> list[_RetrievalCandidate]:
        allocated = set(allocation_ids)
        remaining = [
            candidate
            for note_id, candidate in candidates.items()
            if note_id not in allocated
        ]
        remaining.sort(key=lambda candidate: candidate.note.id)
        remaining.sort(
            key=lambda candidate: candidate.note.timestamp,
            reverse=True,
        )
        remaining.sort(
            key=lambda candidate: (
                candidate.query_rank is None,
                candidate.query_rank or 0,
            )
        )
        return [*(candidates[note_id] for note_id in allocation_ids), *remaining]

    @staticmethod
    def _validate_limits(top_k: int, token_budget: int) -> None:
        if not 1 <= top_k <= AMEM_MAX_SEEDS:
            raise ValueError("A-MEM retrieval top_k must be between 1 and 10")
        if token_budget < 0:
            raise ValueError("A-MEM retrieval token budget cannot be negative")

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return round((time.monotonic() - started) * 1_000)

    def _result(
        self,
        *,
        run_id: str,
        content: str,
        selected: Sequence[_RetrievalCandidate],
        top_k: int,
        candidate_count: int,
        seed_count: int,
        linked_count: int,
        selected_tokens: int,
        latency_ms: int,
    ) -> AMemRetrievalResult:
        return AMemRetrievalResult(
            content=content,
            notes=tuple(candidate.note for candidate in selected),
            versions=tuple(candidate.version for candidate in selected),
            run_id=run_id,
            metadata={
                "memory_strategy": "amem",
                "retrieval_run_id": run_id,
                "retrieval_mode": AMEM_RETRIEVAL_MODE,
                "retrieval_status": "completed",
                "retrieval_top_k": top_k,
                "retrieval_token_budget": self.token_budget,
                "retrieval_candidate_count": candidate_count,
                "retrieval_seed_count": seed_count,
                "retrieval_linked_count": linked_count,
                "retrieval_selected_count": len(selected),
                "retrieval_selected_tokens": selected_tokens,
                "retrieval_latency_ms": latency_ms,
                "query_dependent": True,
            },
        )
