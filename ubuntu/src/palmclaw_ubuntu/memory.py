from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import replace
from typing import Any

from palmclaw_ubuntu.contracts import (
    EmbeddingModel,
    MemoryModel,
    RecursiveSummaryMemoryModel,
    StructuredMemoryModel,
)
from palmclaw_ubuntu.models import (
    ChatMessage,
    EmbeddingResponse,
    MemoryCandidate,
    MemoryEvidence,
    MemoryMessage,
    MemoryRecord,
    MemoryRetrievalResult,
    ModelUsage,
)
from palmclaw_ubuntu.privacy import redact_for_cloud
from palmclaw_ubuntu.storage import SQLiteRepository, StoredMessage
from palmclaw_ubuntu.tokens import TokenCounter
from palmclaw_ubuntu.validation import MemoryValidationGate

_TOKEN_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)
_ALLOWED_SCOPES = {"session", "global"}
_ALLOWED_MEMORY_TYPES = {
    "fact",
    "preference",
    "decision",
    "commitment",
    "profile",
}
_ALLOWED_SENSITIVITY = {"low", "medium", "high"}


class MemoryEngine:
    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        strategy: str,
        summary_model: MemoryModel,
        structured_model: StructuredMemoryModel,
        embedding_model: EmbeddingModel,
        trigger_messages: int,
        retrieval_mode: str,
        top_k: int,
        token_budget: int,
        model_timeout_seconds: float,
        validation_gate: MemoryValidationGate,
        local_pii_storage: str,
        pii_allowlist: tuple[str, ...],
        memory_input_cost_per_million: float = 0.0,
        memory_output_cost_per_million: float = 0.0,
        embedding_input_cost_per_million: float = 0.0,
        token_counter: TokenCounter | None = None,
        recursive_summary_model: RecursiveSummaryMemoryModel | None = None,
        recursive_summary_max_attempts: int = 2,
    ):
        self.repository = repository
        self.strategy = strategy
        self.summary_model = summary_model
        self.structured_model = structured_model
        self.embedding_model = embedding_model
        self.trigger_messages = trigger_messages
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.token_budget = token_budget
        self.model_timeout_seconds = model_timeout_seconds
        self.validation_gate = validation_gate
        self.local_pii_storage = local_pii_storage
        self.pii_allowlist = pii_allowlist
        self.memory_input_cost_per_million = memory_input_cost_per_million
        self.memory_output_cost_per_million = memory_output_cost_per_million
        self.embedding_input_cost_per_million = embedding_input_cost_per_million
        self.token_counter = token_counter or TokenCounter()
        self.recursive_summary_model = recursive_summary_model
        self.recursive_summary_max_attempts = recursive_summary_max_attempts
        if recursive_summary_max_attempts < 1:
            raise ValueError(
                "Recursive summary max attempts must be positive"
            )

    def retrieve(
        self,
        session_id: str,
        *,
        turn_id: str | None,
        query: str,
        mode: str | None = None,
        top_k: int | None = None,
    ) -> MemoryRetrievalResult:
        if self.strategy in {"summary", "recursive_summary"}:
            content = self.repository.latest_memory(session_id)
            recursive = self.strategy == "recursive_summary"
            return MemoryRetrievalResult(
                content=content,
                metadata={
                    "memory_strategy": self.strategy,
                    "retrieval_mode": (
                        "full_recursive_summary"
                        if recursive
                        else "full_summary"
                    ),
                    "retrieval_selected_count": int(bool(content)),
                    "query_dependent": False,
                    "recall_at_k_applicable": not recursive,
                },
            )

        selected_mode = mode or self.retrieval_mode
        selected_top_k = self.top_k if top_k is None else top_k
        if selected_top_k < 1:
            raise ValueError("top_k must be at least 1")
        embedding_enabled = selected_mode in {"embedding", "hybrid"}
        run_id = self.repository.begin_retrieval(
            session_id=session_id,
            turn_id=turn_id,
            query=query,
            mode=selected_mode,
            top_k=selected_top_k,
            token_budget=self.token_budget,
            embedding_backend=(
                self.embedding_model.backend if embedding_enabled else None
            ),
            embedding_model_id=(
                self.embedding_model.model_id if embedding_enabled else None
            ),
        )
        try:
            memories = self.repository.active_structured_memories(session_id)
            result = self._rank(
                session_id=session_id,
                turn_id=turn_id,
                query=query,
                mode=selected_mode,
                top_k=selected_top_k,
                memories=memories,
                run_id=run_id,
            )
            return result
        except Exception as exc:
            self.repository.fail_retrieval(
                run_id,
                f"{type(exc).__name__}: {exc}",
            )
            return MemoryRetrievalResult(
                content="",
                run_id=run_id,
                metadata={
                    "memory_strategy": "structured",
                    "retrieval_run_id": run_id,
                    "retrieval_mode": selected_mode,
                    "retrieval_status": "failed",
                    "retrieval_error": f"{type(exc).__name__}: {exc}",
                    "retrieval_selected_count": 0,
                },
            )

    def _rank(
        self,
        *,
        session_id: str,
        turn_id: str | None,
        query: str,
        mode: str,
        top_k: int,
        memories: Sequence[MemoryRecord],
        run_id: str,
    ) -> MemoryRetrievalResult:
        states = {
            memory.id: {
                "memory": memory,
                "bm25_score": 0.0,
                "embedding_score": 0.0,
                "combined_score": 0.0,
            }
            for memory in memories
        }
        if mode in {"bm25", "hybrid"}:
            bm25_scores = bm25(query, [memory.content for memory in memories])
            for memory, score in zip(memories, bm25_scores, strict=True):
                states[memory.id]["bm25_score"] = score
        if mode in {"embedding", "hybrid"} and memories:
            embedding_scores = self._embedding_scores(
                session_id=session_id,
                turn_id=turn_id,
                query=query,
                memories=memories,
            )
            for memory in memories:
                states[memory.id]["embedding_score"] = embedding_scores.get(
                    memory.id,
                    0.0,
                )

        if mode == "full":
            for index, memory in enumerate(memories):
                states[memory.id]["combined_score"] = 1.0 / (index + 1)
        elif mode == "bm25":
            for state in states.values():
                state["combined_score"] = state["bm25_score"]
        elif mode == "embedding":
            for state in states.values():
                state["combined_score"] = state["embedding_score"]
        elif mode == "hybrid":
            max_bm25 = max(
                (float(state["bm25_score"]) for state in states.values()),
                default=0.0,
            )
            positive_embeddings = [
                max(0.0, float(state["embedding_score"])) for state in states.values()
            ]
            max_embedding = max(positive_embeddings, default=0.0)
            for state in states.values():
                normalized_bm25 = (
                    float(state["bm25_score"]) / max_bm25 if max_bm25 else 0.0
                )
                normalized_embedding = (
                    max(0.0, float(state["embedding_score"])) / max_embedding
                    if max_embedding
                    else 0.0
                )
                state["combined_score"] = (normalized_bm25 + normalized_embedding) / 2
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        ranked = sorted(
            states.values(),
            key=lambda state: (
                -float(state["combined_score"]),
                -float(state["memory"].confidence),
                state["memory"].id,
            ),
        )
        selected: list[MemoryRecord] = []
        trace_candidates: list[dict[str, Any]] = []
        used_tokens = 0
        for rank, state in enumerate(ranked, start=1):
            memory = state["memory"]
            token_count = self.token_counter.count(self._render_memory(memory))
            exclusion_reason = None
            is_selected = False
            if mode == "none":
                exclusion_reason = "mode_none"
            elif mode != "full" and float(state["combined_score"]) <= 0:
                exclusion_reason = "below_score"
            elif mode != "full" and len(selected) >= top_k:
                exclusion_reason = "top_k"
            elif used_tokens + token_count > self.token_budget:
                exclusion_reason = "token_budget"
            else:
                selected.append(memory)
                used_tokens += token_count
                is_selected = True
            trace_candidates.append(
                {
                    "memory_id": memory.id,
                    "rank": rank,
                    "bm25_score": state["bm25_score"],
                    "embedding_score": state["embedding_score"],
                    "combined_score": state["combined_score"],
                    "selected": is_selected,
                    "exclusion_reason": exclusion_reason,
                    "token_count": token_count,
                }
            )
        self.repository.complete_retrieval(
            run_id,
            trace_candidates,
            selected_count=len(selected),
        )
        content = "\n".join(self._render_memory(memory) for memory in selected)
        return MemoryRetrievalResult(
            content=content,
            memories=tuple(selected),
            run_id=run_id,
            metadata={
                "memory_strategy": "structured",
                "retrieval_run_id": run_id,
                "retrieval_mode": mode,
                "retrieval_status": "completed",
                "retrieval_candidate_count": len(memories),
                "retrieval_selected_count": len(selected),
                "retrieval_selected_ids": [memory.id for memory in selected],
                "retrieval_tokens": used_tokens,
            },
        )

    def _embedding_scores(
        self,
        *,
        session_id: str,
        turn_id: str | None,
        query: str,
        memories: Sequence[MemoryRecord],
    ) -> dict[str, float]:
        memory_ids = [memory.id for memory in memories]
        stored = self.repository.memory_embeddings(
            memory_ids,
            model_id=self.embedding_model.model_id,
            dimensions=self.embedding_model.dimensions,
        )
        missing = [memory for memory in memories if memory.id not in stored]
        texts = [query, *(memory.content for memory in missing)]
        response = self._embed(
            texts,
            session_id=session_id,
            turn_id=turn_id,
        )
        if len(response.vectors) != len(texts):
            raise RuntimeError("EmbeddingModel returned an unexpected vector count")
        query_vector = response.vectors[0]
        for memory, vector in zip(
            missing,
            response.vectors[1:],
            strict=True,
        ):
            self._validate_vector(vector)
            self.repository.store_memory_embedding(
                memory_id=memory.id,
                model_id=self.embedding_model.model_id,
                dimensions=self.embedding_model.dimensions,
                vector=vector,
            )
            stored[memory.id] = vector
        self._validate_vector(query_vector)
        return {
            memory.id: cosine_similarity(
                query_vector,
                stored[memory.id],
            )
            for memory in memories
        }

    def consolidate(
        self,
        session_id: str,
        *,
        date: str | None = None,
    ) -> str | None:
        window = self.repository.consolidation_window(
            session_id,
            self.trigger_messages,
        )
        if not window:
            return None
        if self.strategy == "summary":
            return self._consolidate_summary(session_id, window)
        if self.strategy == "recursive_summary":
            if self.recursive_summary_model is None:
                raise RuntimeError(
                    "recursive_summary requires a RecursiveSummaryMemoryModel"
                )
            if not date:
                raise ValueError(
                    "recursive_summary consolidation requires a date"
                )
            return self._consolidate_recursive_summary(
                session_id,
                window,
                date=date,
            )
        return self._consolidate_structured(session_id, window)

    def _consolidate_summary(
        self,
        session_id: str,
        window: Sequence[StoredMessage],
    ) -> str:
        run_id = self.repository.begin_consolidation(
            session_id,
            window[0].id,
            window[-1].id,
            backend=self.summary_model.backend,
            model_id=self.summary_model.model_id,
            prompt_version=self.summary_model.prompt_version,
            schema_version=self.summary_model.schema_version,
        )
        started = time.monotonic()
        try:
            response = self._invoke_with_timeout(
                lambda: self.summary_model.consolidate(
                    self._summary_messages(window),
                    self.repository.latest_memory(session_id),
                ),
                label="MemoryModel",
            )
            self.repository.record_model_call(
                session_id=session_id,
                consolidation_run_id=run_id,
                role="memory",
                backend=self.summary_model.backend,
                model_id=self.summary_model.model_id,
                prompt_version=self.summary_model.prompt_version,
                schema_version=self.summary_model.schema_version,
                latency_ms=self._elapsed_ms(started),
                usage=response.usage,
                response_id=response.response_id,
                metadata=self._memory_call_metadata(
                    response.metadata,
                    response.usage,
                    backend=self.summary_model.backend,
                ),
            )
            content = response.content.strip()
            if not content:
                self.repository.complete_empty_consolidation(
                    run_id,
                    response.usage,
                )
                return run_id
            self.repository.complete_consolidation(
                run_id,
                session_id,
                self._redact_storage_text(content),
                response.usage,
            )
        except Exception as exc:
            self._record_consolidation_failure(
                session_id=session_id,
                run_id=run_id,
                model=self.summary_model,
                started=started,
                exc=exc,
            )
        return run_id

    def _consolidate_recursive_summary(
        self,
        session_id: str,
        window: Sequence[StoredMessage],
        *,
        date: str,
    ) -> str:
        model = self.recursive_summary_model
        assert model is not None
        run_id = self.repository.begin_consolidation(
            session_id,
            window[0].id,
            window[-1].id,
            backend=model.backend,
            model_id=model.model_id,
            prompt_version=model.prompt_version,
            schema_version=model.schema_version,
        )
        previous_memory = self.repository.latest_memory(session_id)
        daily_history = "\n".join(
            message.content
            for message in window
            if message.role in {"user", "assistant"}
        )
        final_error: Exception | None = None
        for attempt in range(1, self.recursive_summary_max_attempts + 1):
            started = time.monotonic()
            try:
                response = self._invoke_with_timeout(
                    lambda: model.update(
                        previous_memory=previous_memory,
                        date=date,
                        daily_history=daily_history,
                    ),
                    label="RecursiveSummaryMemoryModel",
                )
                update_status = str(
                    response.metadata.get("update_status", "")
                )
                if update_status not in {"updated", "noop"}:
                    raise RuntimeError(
                        "Recursive summary response is missing a valid "
                        "update_status"
                    )
                content = response.content.strip()
                if update_status == "updated" and not content:
                    raise RuntimeError(
                        "Recursive summary update returned empty memory"
                    )
                if update_status == "noop" and content:
                    raise RuntimeError(
                        "Recursive summary no-op returned memory content"
                    )
                stored_content = ""
                storage_truncated = False
                if update_status == "updated":
                    stored_content, storage_truncated = (
                        _truncate_complete_summary_lines(
                            self._redact_storage_text(content),
                            max_chars=model.max_memory_chars,
                        )
                    )
                    if not stored_content:
                        raise RuntimeError(
                            "Recursive summary update became empty after "
                            "storage normalization"
                        )
                self.repository.record_model_call(
                    session_id=session_id,
                    consolidation_run_id=run_id,
                    role="memory",
                    backend=model.backend,
                    model_id=model.model_id,
                    prompt_version=model.prompt_version,
                    schema_version=model.schema_version,
                    latency_ms=self._elapsed_ms(started),
                    usage=response.usage,
                    response_id=response.response_id,
                    metadata={
                        **self._memory_call_metadata(
                            response.metadata,
                            response.usage,
                            backend=model.backend,
                        ),
                        "truncated": bool(
                            response.metadata.get("truncated", False)
                            or storage_truncated
                        ),
                        "storage_truncated": storage_truncated,
                        "memory_chars": (
                            len(stored_content)
                            if update_status == "updated"
                            else len(previous_memory)
                        ),
                        "attempt": attempt,
                        "max_attempts": self.recursive_summary_max_attempts,
                    },
                )
                if update_status == "noop":
                    self.repository.complete_empty_consolidation(
                        run_id,
                        response.usage,
                    )
                else:
                    self.repository.complete_consolidation(
                        run_id,
                        session_id,
                        stored_content,
                        response.usage,
                    )
                return run_id
            except Exception as exc:
                final_error = exc
                self.repository.record_model_call(
                    session_id=session_id,
                    consolidation_run_id=run_id,
                    role="memory",
                    backend=model.backend,
                    model_id=model.model_id,
                    prompt_version=model.prompt_version,
                    schema_version=model.schema_version,
                    latency_ms=self._elapsed_ms(started),
                    usage=ModelUsage(),
                    response_id=None,
                    metadata={
                        "date": date,
                        "attempt": attempt,
                        "max_attempts": self.recursive_summary_max_attempts,
                        "update_status": "failed",
                    },
                    error=f"{type(exc).__name__}: {exc}",
                )
        assert final_error is not None
        self.repository.fail_consolidation(
            run_id,
            f"{type(final_error).__name__}: {final_error}",
        )
        return run_id

    def _consolidate_structured(
        self,
        session_id: str,
        window: Sequence[StoredMessage],
    ) -> str:
        run_id = self.repository.begin_consolidation(
            session_id,
            window[0].id,
            window[-1].id,
            backend=self.structured_model.backend,
            model_id=self.structured_model.model_id,
            prompt_version=self.structured_model.prompt_version,
            schema_version=self.structured_model.schema_version,
        )
        started = time.monotonic()
        try:
            active = self.repository.active_structured_memories(session_id)
            response = self._invoke_with_timeout(
                lambda: self.structured_model.extract(
                    self._structured_messages(window),
                    "\n".join(self._render_memory(memory) for memory in active),
                ),
                label="StructuredMemoryModel",
            )
            self.repository.record_model_call(
                session_id=session_id,
                consolidation_run_id=run_id,
                role="memory",
                backend=self.structured_model.backend,
                model_id=self.structured_model.model_id,
                prompt_version=self.structured_model.prompt_version,
                schema_version=self.structured_model.schema_version,
                latency_ms=self._elapsed_ms(started),
                usage=response.usage,
                response_id=response.response_id,
                metadata={
                    **self._memory_call_metadata(
                        response.metadata,
                        response.usage,
                        backend=self.structured_model.backend,
                    ),
                    "candidate_count": len(response.candidates),
                },
            )
            message_by_id = {message.id: message for message in window}
            outcomes = []
            accepted: list[MemoryRecord] = []
            for candidate in response.candidates:
                normalized, rejection_reason = self._validate_candidate(
                    candidate,
                    message_by_id,
                )
                fact_key = self._fact_key(session_id, normalized)
                active_memory = self.repository.active_fact_memory(fact_key)
                gate_decision = self.validation_gate.evaluate(
                    normalized,
                    structural_error=rejection_reason,
                    active_memory=active_memory,
                )
                storage_candidate = self._candidate_for_storage(normalized)
                record, action = self.repository.apply_structured_memory(
                    session_id=session_id,
                    candidate=storage_candidate,
                    fact_key=fact_key,
                    consolidation_run_id=run_id,
                    model_id=self.structured_model.model_id,
                    schema_version=self.structured_model.schema_version,
                    gate_decision=gate_decision,
                )
                if action in {"created", "superseded"}:
                    accepted.append(record)
                outcomes.append(
                    {
                        "memory_id": record.id,
                        "action": action,
                        "status": record.status,
                        "fact_key": record.fact_key,
                        "version": record.version,
                        "gate_decision": gate_decision.decision,
                        "evidence_relation": gate_decision.evidence_relation,
                        "conflict_relation": gate_decision.conflict_relation,
                        "reason": gate_decision.reason,
                    }
                )
            embedding_error = self._embed_new_memories(
                session_id=session_id,
                consolidation_run_id=run_id,
                memories=accepted,
            )
            self.repository.complete_structured_consolidation(
                run_id,
                output={
                    "strategy": "structured",
                    "outcomes": outcomes,
                    "accepted_count": len(accepted),
                    "embedding_error": embedding_error,
                },
                usage=response.usage,
            )
        except Exception as exc:
            self._record_consolidation_failure(
                session_id=session_id,
                run_id=run_id,
                model=self.structured_model,
                started=started,
                exc=exc,
            )
        return run_id

    def _embed_new_memories(
        self,
        *,
        session_id: str,
        consolidation_run_id: str,
        memories: Sequence[MemoryRecord],
    ) -> str | None:
        if not memories:
            return None
        try:
            response = self._embed(
                [memory.content for memory in memories],
                session_id=session_id,
                consolidation_run_id=consolidation_run_id,
            )
            if len(response.vectors) != len(memories):
                raise RuntimeError("EmbeddingModel returned an unexpected vector count")
            for memory, vector in zip(
                memories,
                response.vectors,
                strict=True,
            ):
                self._validate_vector(vector)
                self.repository.store_memory_embedding(
                    memory_id=memory.id,
                    model_id=self.embedding_model.model_id,
                    dimensions=self.embedding_model.dimensions,
                    vector=vector,
                )
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return None

    def _embed(
        self,
        texts: Sequence[str],
        *,
        session_id: str,
        turn_id: str | None = None,
        consolidation_run_id: str | None = None,
    ) -> EmbeddingResponse:
        started = time.monotonic()
        try:
            response = self._invoke_with_timeout(
                lambda: self.embedding_model.embed(texts),
                label="EmbeddingModel",
            )
        except Exception as exc:
            self.repository.record_model_call(
                session_id=session_id,
                turn_id=turn_id,
                consolidation_run_id=consolidation_run_id,
                role="embedding",
                backend=self.embedding_model.backend,
                model_id=self.embedding_model.model_id,
                prompt_version="embedding-v1",
                schema_version=None,
                latency_ms=self._elapsed_ms(started),
                usage=ModelUsage(),
                response_id=None,
                metadata={
                    "input_count": len(texts),
                    "dimensions": self.embedding_model.dimensions,
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self.repository.record_model_call(
            session_id=session_id,
            turn_id=turn_id,
            consolidation_run_id=consolidation_run_id,
            role="embedding",
            backend=self.embedding_model.backend,
            model_id=self.embedding_model.model_id,
            prompt_version="embedding-v1",
            schema_version=None,
            latency_ms=self._elapsed_ms(started),
            usage=response.usage,
            response_id=response.response_id,
            metadata={
                **dict(response.metadata),
                "input_count": len(texts),
                "dimensions": self.embedding_model.dimensions,
                "estimated_cost_usd": (
                    response.usage.input_tokens
                    * self.embedding_input_cost_per_million
                    / 1_000_000
                    if self.embedding_model.backend == "openai"
                    else 0.0
                ),
            },
        )
        return response

    def _validate_candidate(
        self,
        candidate: MemoryCandidate,
        message_by_id: dict[int, StoredMessage],
    ) -> tuple[MemoryCandidate, str | None]:
        reasons: list[str] = []
        subject = candidate.subject.strip()
        predicate = candidate.predicate.strip()
        value = candidate.value.strip()
        if not subject:
            reasons.append("empty_subject")
            subject = "invalid"
        if not predicate:
            reasons.append("empty_predicate")
            predicate = "invalid"
        if not value:
            reasons.append("empty_value")
        if candidate.scope not in _ALLOWED_SCOPES:
            reasons.append("invalid_scope")
        if candidate.memory_type not in _ALLOWED_MEMORY_TYPES:
            reasons.append("invalid_memory_type")
        if candidate.sensitivity not in _ALLOWED_SENSITIVITY:
            reasons.append("invalid_sensitivity")
        if not 0 <= candidate.confidence <= 1:
            reasons.append("invalid_confidence")

        valid_evidence: list[MemoryEvidence] = []
        for evidence in candidate.evidence:
            source = message_by_id.get(evidence.message_id)
            quote = evidence.quote.strip()
            if source is None:
                reasons.append("missing_evidence_message")
                continue
            if not quote:
                reasons.append("empty_evidence_quote")
                continue
            start = source.content.find(quote)
            if start < 0:
                reasons.append("unsupported_evidence_quote")
                valid_evidence.append(
                    MemoryEvidence(
                        message_id=evidence.message_id,
                        quote=quote,
                    )
                )
                continue
            valid_evidence.append(
                MemoryEvidence(
                    message_id=evidence.message_id,
                    quote=quote,
                    start_char=start,
                    end_char=start + len(quote),
                )
            )
        if not valid_evidence:
            reasons.append("no_valid_evidence")
        normalized = replace(
            candidate,
            subject=subject,
            predicate=predicate,
            value=value,
            evidence=tuple(valid_evidence),
        )
        return normalized, ",".join(dict.fromkeys(reasons)) or None

    def _fact_key(
        self,
        session_id: str,
        candidate: MemoryCandidate,
    ) -> str:
        owner = "global" if candidate.scope == "global" else session_id
        canonical = "|".join(
            (
                owner,
                candidate.scope,
                self._canonical_fact_field(candidate.subject),
                self._canonical_fact_field(candidate.predicate),
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _canonical_fact_field(value: str) -> str:
        return re.sub(
            r"[_\W]+",
            "_",
            value.casefold().strip(),
            flags=re.UNICODE,
        ).strip("_")

    def _record_consolidation_failure(
        self,
        *,
        session_id: str,
        run_id: str,
        model: Any,
        started: float,
        exc: Exception,
    ) -> None:
        self.repository.record_model_call(
            session_id=session_id,
            consolidation_run_id=run_id,
            role="memory",
            backend=model.backend,
            model_id=model.model_id,
            prompt_version=model.prompt_version,
            schema_version=model.schema_version,
            latency_ms=self._elapsed_ms(started),
            usage=ModelUsage(),
            response_id=None,
            metadata={},
            error=f"{type(exc).__name__}: {exc}",
        )
        self.repository.fail_consolidation(
            run_id,
            f"{type(exc).__name__}: {exc}",
            retryable=model.backend == "local",
        )

    def _candidate_for_storage(
        self,
        candidate: MemoryCandidate,
    ) -> MemoryCandidate:
        if self.local_pii_storage == "raw":
            return candidate
        subject = self._redact_storage_text(candidate.subject)
        predicate = self._redact_storage_text(candidate.predicate)
        value = self._redact_storage_text(candidate.value)
        evidence = tuple(
            replace(
                item,
                quote=self._redact_storage_text(item.quote),
            )
            for item in candidate.evidence
        )
        return replace(
            candidate,
            subject=subject,
            predicate=predicate,
            value=value,
            evidence=evidence,
        )

    def _redact_storage_text(self, value: str) -> str:
        if self.local_pii_storage == "raw":
            return value
        redacted, _ = redact_for_cloud(
            value,
            allowlist=self.pii_allowlist,
        )
        return redacted

    def _memory_call_metadata(
        self,
        metadata: Any,
        usage: ModelUsage,
        *,
        backend: str,
    ) -> dict[str, Any]:
        estimated_cost = 0.0
        if backend == "openai":
            estimated_cost = (
                usage.input_tokens * self.memory_input_cost_per_million
                + usage.output_tokens * self.memory_output_cost_per_million
            ) / 1_000_000
        return {
            **dict(metadata),
            "estimated_cost_usd": estimated_cost,
        }

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.embedding_model.dimensions:
            raise ValueError(
                "Embedding vector dimension does not match configured dimensions"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding vector contains a non-finite value")

    def _invoke_with_timeout(self, operation, *, label: str):
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=label.lower(),
        )
        future = executor.submit(operation)
        try:
            return future.result(timeout=self.model_timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(
                f"{label} timed out after {self.model_timeout_seconds:g}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _render_memory(memory: MemoryRecord) -> str:
        return (
            f"- [memory_id={memory.id} scope={memory.scope} "
            f"version={memory.version} confidence={memory.confidence:.2f}] "
            f"{memory.content}"
        )

    @staticmethod
    def _summary_messages(
        window: Sequence[StoredMessage],
    ) -> list[ChatMessage]:
        return [
            message.as_chat_message()
            for message in window
            if message.role in {"user", "assistant"}
        ]

    @staticmethod
    def _structured_messages(
        window: Sequence[StoredMessage],
    ) -> list[MemoryMessage]:
        return [
            MemoryMessage(
                id=message.id,
                role=message.role,
                content=message.content,
            )
            for message in window
            if message.role in {"user", "assistant"}
        ]

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, round((time.monotonic() - started) * 1000))


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold().replace("_", " "))


def bm25(query: str, documents: Sequence[str]) -> list[float]:
    if not documents:
        return []
    query_terms = tokenize(query)
    if not query_terms:
        return [0.0] * len(documents)
    document_terms = [tokenize(document) for document in documents]
    average_length = (
        sum(len(terms) for terms in document_terms) / len(document_terms)
    ) or 1.0
    document_frequency = Counter(
        term
        for term in set(query_terms)
        for terms in document_terms
        if term in set(terms)
    )
    total_documents = len(documents)
    k1 = 1.5
    b = 0.75
    scores = []
    for terms in document_terms:
        frequencies = Counter(terms)
        score = 0.0
        for term in set(query_terms):
            frequency = frequencies[term]
            if not frequency:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1
                + (total_documents - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * len(terms) / average_length)
            score += inverse_document_frequency * (frequency * (k1 + 1) / denominator)
        scores.append(score)
    return scores


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("Cannot compare vectors with different dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    ) / (left_norm * right_norm)


def _truncate_complete_summary_lines(
    content: str,
    *,
    max_chars: int,
) -> tuple[str, bool]:
    normalized = "\n".join(
        line.rstrip() for line in content.replace("\r\n", "\n").splitlines()
    ).strip()
    if len(normalized) <= max_chars:
        return normalized, False
    boundary = normalized.rfind("\n", 0, max_chars + 1)
    if boundary > 0:
        return normalized[:boundary].rstrip(), True
    return normalized[:max_chars].rstrip(), True
