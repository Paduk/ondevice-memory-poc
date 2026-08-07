from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from palmclaw_ubuntu.contracts import FactMemoryModel, PatchMemoryModel
from palmclaw_ubuntu.fact_memory import (
    FactMemoryLinker,
    select_relevant_fact_records,
)
from palmclaw_ubuntu.memory_patch import ToolMemoryPatchEngine
from palmclaw_ubuntu.models import (
    FactMemoryRecord,
    FactMemorySemanticDecision,
    FactMemorySemanticReviewCase,
    MemoryMessage,
    ModelUsage,
    ToolMemoryPatchRejection,
)
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tokens import TokenCounter
from palmclaw_ubuntu.tool_memory_schema import ToolMemoryOntology
from palmclaw_ubuntu.tools import ToolRegistry
from palmclaw_ubuntu.validation import (
    FactMemoryValidationPolicy,
    ToolMemoryPatchPolicy,
)


@dataclass(frozen=True)
class PatchWorkerResult:
    claimed_count: int
    completed_count: int
    retryable_count: int
    failed_count: int
    run_ids: tuple[str, ...]
    errors: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "claimed_count": self.claimed_count,
            "completed_count": self.completed_count,
            "retryable_count": self.retryable_count,
            "failed_count": self.failed_count,
            "run_ids": list(self.run_ids),
            "errors": list(self.errors),
        }


class BackgroundMemoryPatchWorker:
    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        model: PatchMemoryModel,
        tool_registry: ToolRegistry | None,
        user_id: str,
        model_timeout_seconds: float,
        max_attempts: int,
        retry_delay_seconds: float,
        lease_seconds: float,
        memory_input_cost_per_million: float = 0.0,
        memory_output_cost_per_million: float = 0.0,
        ontology: ToolMemoryOntology | None = None,
        batch_turn_limit: int = 32,
        batch_token_limit: int = 4_096,
        token_counter: TokenCounter | None = None,
    ):
        if model_timeout_seconds <= 0:
            raise ValueError("Patch model timeout must be positive")
        if max_attempts < 1:
            raise ValueError("Patch max attempts must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("Patch retry delay cannot be negative")
        if lease_seconds <= model_timeout_seconds:
            raise ValueError("Patch job lease must exceed the model timeout")
        if batch_turn_limit < 1:
            raise ValueError("Patch batch turn limit must be positive")
        if batch_token_limit < 256:
            raise ValueError("Patch batch token limit must be at least 256")
        self.repository = repository
        self.model = model
        self.tool_registry = tool_registry
        if ontology is None and tool_registry is None:
            raise ValueError("Patch worker requires a Tool registry or ontology")
        self.ontology = (
            ontology
            if ontology is not None
            else tool_registry.memory_ontology()
        )
        self.user_id = user_id
        self.model_timeout_seconds = model_timeout_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.lease_seconds = lease_seconds
        self.memory_input_cost_per_million = memory_input_cost_per_million
        self.memory_output_cost_per_million = memory_output_cost_per_million
        self.batch_turn_limit = batch_turn_limit
        self.batch_token_limit = batch_token_limit
        self.token_counter = token_counter or TokenCounter()
        allowed_domain_topics = tuple(
            sorted(
                {
                    (tool.domain, tool.topic)
                    for tool in self.ontology.tools
                }
                | {
                    (tool.domain, slot.name)
                    for tool in self.ontology.tools
                    for slot in tool.slots
                }
            )
        )
        self.patch_engine = ToolMemoryPatchEngine(
            repository,
            policy=ToolMemoryPatchPolicy(
                pii_allowlist=(),
                expected_user_id=user_id,
                allowed_domain_topics=allowed_domain_topics,
                policy_version="tool-memory-patch-gate-v3",
            ),
        )

    def enqueue_turn(self, session_id: str, turn_id: str) -> str:
        return self.repository.enqueue_memory_patch_job(
            session_id=session_id,
            source_turn_id=turn_id,
            max_attempts=self.max_attempts,
        )

    def run_pending(
        self,
        *,
        limit: int,
        session_id: str | None = None,
    ) -> PatchWorkerResult:
        jobs = self.repository.claim_memory_patch_jobs(
            limit=limit,
            lease_seconds=self.lease_seconds,
            session_id=session_id,
        )
        status_counts: Counter[str] = Counter()
        run_ids: list[str] = []
        errors: list[dict[str, str]] = []
        for batch in self._batch_jobs(jobs):
            if len(batch) == 1:
                status, run_id, error = self._process_job(batch[0])
                statuses = (status,)
                job_errors = (
                    {str(batch[0]["id"]): error}
                    if error is not None
                    else {}
                )
            else:
                statuses, run_id, job_errors = self._process_batch(batch)
            if run_id is not None:
                run_ids.append(run_id)
            status_counts.update(statuses)
            errors.extend(
                {
                    "job_id": job_id,
                    "run_id": run_id or "",
                    "error": error,
                }
                for job_id, error in job_errors.items()
            )
        return PatchWorkerResult(
            claimed_count=len(jobs),
            completed_count=status_counts["completed"],
            retryable_count=status_counts["retryable"],
            failed_count=status_counts["failed"],
            run_ids=tuple(run_ids),
            errors=tuple(errors),
        )

    def _batch_jobs(
        self,
        jobs: list[dict[str, Any]],
    ) -> tuple[tuple[dict[str, Any], ...], ...]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for job in jobs:
            grouped.setdefault(str(job["session_id"]), []).append(job)
        batches: list[tuple[dict[str, Any], ...]] = []
        for session_jobs in grouped.values():
            current: list[dict[str, Any]] = []
            current_tokens = 0
            for job in session_jobs:
                messages = self._job_messages(job)
                token_count = sum(
                    self.token_counter.count(message.content) + 8
                    for message in messages
                )
                if current and (
                    len(current) >= self.batch_turn_limit
                    or current_tokens + token_count > self.batch_token_limit
                ):
                    batches.append(tuple(current))
                    current = []
                    current_tokens = 0
                current.append(job)
                current_tokens += token_count
            if current:
                batches.append(tuple(current))
        return tuple(batches)

    def _process_batch(
        self,
        jobs: tuple[dict[str, Any], ...],
    ) -> tuple[
        tuple[str, ...],
        str | None,
        dict[str, str],
    ]:
        session_ids = {str(job["session_id"]) for job in jobs}
        if len(session_ids) != 1:
            raise ValueError("A patch batch cannot cross sessions")
        session_id = session_ids.pop()
        job_ids = tuple(str(job["id"]) for job in jobs)
        turn_ids = tuple(str(job["source_turn_id"]) for job in jobs)
        run_id = self.repository.begin_memory_patch_run(
            session_id=session_id,
            source_turn_id=None,
            backend=self.model.backend,
            model_id=self.model.model_id,
            prompt_version=self.model.prompt_version,
            schema_version=self.model.schema_version,
            status="running",
        )
        self.repository.link_memory_patch_jobs_to_batch_run(
            job_ids,
            run_id=run_id,
            session_id=session_id,
        )
        messages = [
            message
            for job in jobs
            for message in self._job_messages(job)
        ]
        allowed_message_ids = tuple(message.id for message in messages)
        active_records = self.repository.list_tool_memory_records(
            session_id,
            user_id=self.user_id,
        )
        started = time.monotonic()
        model_call_recorded = False
        try:
            response = self._invoke_with_timeout(
                lambda: self.model.propose(
                    messages,
                    self.ontology,
                    active_records,
                    user_id=self.user_id,
                    session_id=session_id,
                )
            )
            ordered_patches = tuple(
                patch
                for _, patch in sorted(
                    enumerate(response.patches),
                    key=lambda item: (
                        min(
                            (
                                evidence.message_id
                                for evidence in item[1].evidence
                            ),
                            default=2**63 - 1,
                        ),
                        item[0],
                    ),
                )
            )
            self.repository.record_model_call(
                session_id=session_id,
                turn_id=None,
                memory_patch_run_id=run_id,
                role="memory_patch",
                backend=self.model.backend,
                model_id=self.model.model_id,
                prompt_version=self.model.prompt_version,
                schema_version=self.model.schema_version,
                latency_ms=self._elapsed_ms(started),
                usage=response.usage,
                response_id=response.response_id,
                metadata={
                    **dict(response.metadata),
                    "job_ids": list(job_ids),
                    "source_turn_ids": list(turn_ids),
                    "source_message_cursors": [
                        int(job["source_message_cursor"]) for job in jobs
                    ],
                    "batch_turn_count": len(jobs),
                    "batch_message_count": len(messages),
                    "batch_token_limit": self.batch_token_limit,
                    "patch_count": len(ordered_patches),
                    "estimated_cost_usd": self._estimated_cost(response.usage),
                },
            )
            model_call_recorded = True
            result = None
            if ordered_patches:
                result = self.patch_engine.apply_batch(
                    run_id,
                    ordered_patches,
                    usage=response.usage,
                    allowed_evidence_message_ids=allowed_message_ids,
                    isolate_invalid_patches=True,
                    finish_run=False,
                )
            rejections = result.rejections if result is not None else ()
            affected_job_ids = self._jobs_for_rejections(jobs, rejections)
            completed_job_ids = tuple(
                job_id
                for job_id in job_ids
                if job_id not in affected_job_ids
            )
            status_by_job = {
                job_id: "completed" for job_id in completed_job_ids
            }
            job_errors: dict[str, str] = {}
            with self.repository.memory_patch_transaction():
                if completed_job_ids:
                    self.repository.complete_memory_patch_jobs(
                        completed_job_ids
                    )
                for job_id in job_ids:
                    if job_id not in affected_job_ids:
                        continue
                    rejection_error = self._rejection_error(
                        rejections,
                        affected_job_ids[job_id],
                    )
                    status_by_job[job_id] = (
                        self.repository.retry_memory_patch_job(
                            job_id,
                            error=rejection_error,
                            retry_delay_seconds=self.retry_delay_seconds,
                        )
                    )
                    job_errors[job_id] = rejection_error
                self.repository.finish_memory_patch_run(
                    run_id,
                    status="completed",
                    usage=response.usage,
                )
            return (
                tuple(status_by_job[job_id] for job_id in job_ids),
                run_id,
                job_errors,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if not model_call_recorded:
                self.repository.record_model_call(
                    session_id=session_id,
                    turn_id=None,
                    memory_patch_run_id=run_id,
                    role="memory_patch",
                    backend=self.model.backend,
                    model_id=self.model.model_id,
                    prompt_version=self.model.prompt_version,
                    schema_version=self.model.schema_version,
                    latency_ms=self._elapsed_ms(started),
                    usage=ModelUsage(),
                    response_id=None,
                    metadata={
                        "job_ids": list(job_ids),
                        "source_turn_ids": list(turn_ids),
                        "batch_turn_count": len(jobs),
                    },
                    error=message,
                )
            run = self.repository.memory_patch_run(run_id)
            if run["status"] in {"queued", "running"}:
                self.repository.finish_memory_patch_run(
                    run_id,
                    status="failed",
                    error=message,
                )
            status_by_job: dict[str, str] = {}
            job_errors: dict[str, str] = {}
            for job_id in job_ids:
                job = self.repository.memory_patch_job(job_id)
                if job["status"] == "running":
                    status_by_job[job_id] = (
                        self.repository.retry_memory_patch_job(
                            job_id,
                            error=message,
                            retry_delay_seconds=self.retry_delay_seconds,
                        )
                    )
                    job_errors[job_id] = message
                else:
                    status_by_job[job_id] = str(job["status"])
            return (
                tuple(status_by_job[job_id] for job_id in job_ids),
                run_id,
                job_errors,
            )

    def _process_job(
        self,
        job: dict[str, object],
    ) -> tuple[str, str | None, str | None]:
        job_id = str(job["id"])
        session_id = str(job["session_id"])
        turn_id = str(job["source_turn_id"])
        run_id = self.repository.begin_memory_patch_run(
            session_id=session_id,
            source_turn_id=turn_id,
            backend=self.model.backend,
            model_id=self.model.model_id,
            prompt_version=self.model.prompt_version,
            schema_version=self.model.schema_version,
            status="running",
            job_id=job_id,
        )
        messages = [
            MemoryMessage(
                id=message.id,
                role=message.role,
                content=message.content,
            )
            for message in self.repository.turn_memory_messages(turn_id)
            if message.role in {"user", "assistant"}
        ]
        active_records = self.repository.list_tool_memory_records(
            session_id,
            user_id=self.user_id,
        )
        started = time.monotonic()
        model_call_recorded = False
        try:
            response = self._invoke_with_timeout(
                lambda: self.model.propose(
                    messages,
                    self.ontology,
                    active_records,
                    user_id=self.user_id,
                    session_id=session_id,
                )
            )
            self.repository.record_model_call(
                session_id=session_id,
                turn_id=turn_id,
                memory_patch_run_id=run_id,
                role="memory_patch",
                backend=self.model.backend,
                model_id=self.model.model_id,
                prompt_version=self.model.prompt_version,
                schema_version=self.model.schema_version,
                latency_ms=self._elapsed_ms(started),
                usage=response.usage,
                response_id=response.response_id,
                metadata={
                    **dict(response.metadata),
                    "job_id": job_id,
                    "source_turn_id": turn_id,
                    "patch_count": len(response.patches),
                    "estimated_cost_usd": self._estimated_cost(response.usage),
                },
            )
            model_call_recorded = True
            result = None
            if response.patches:
                result = self.patch_engine.apply_batch(
                    run_id,
                    response.patches,
                    usage=response.usage,
                    isolate_invalid_patches=True,
                    finish_run=False,
                )
            if result is not None and result.rejections:
                message = self._rejection_error(
                    result.rejections,
                    set(range(len(result.rejections))),
                )
                with self.repository.memory_patch_transaction():
                    status = self.repository.retry_memory_patch_job(
                        job_id,
                        error=message,
                        retry_delay_seconds=self.retry_delay_seconds,
                    )
                    self.repository.finish_memory_patch_run(
                        run_id,
                        status="completed",
                        usage=response.usage,
                    )
                return status, run_id, message
            with self.repository.memory_patch_transaction():
                self.repository.complete_memory_patch_job(job_id)
                self.repository.finish_memory_patch_run(
                    run_id,
                    status="completed",
                    usage=response.usage,
                )
            return "completed", run_id, None
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if not model_call_recorded:
                self.repository.record_model_call(
                    session_id=session_id,
                    turn_id=turn_id,
                    memory_patch_run_id=run_id,
                    role="memory_patch",
                    backend=self.model.backend,
                    model_id=self.model.model_id,
                    prompt_version=self.model.prompt_version,
                    schema_version=self.model.schema_version,
                    latency_ms=self._elapsed_ms(started),
                    usage=ModelUsage(),
                    response_id=None,
                    metadata={
                        "job_id": job_id,
                        "source_turn_id": turn_id,
                    },
                    error=message,
                )
            run = self.repository.memory_patch_run(run_id)
            if run["status"] in {"queued", "running"}:
                self.repository.finish_memory_patch_run(
                    run_id,
                    status="failed",
                    error=message,
                )
            status = self.repository.retry_memory_patch_job(
                job_id,
                error=message,
                retry_delay_seconds=self.retry_delay_seconds,
            )
            return status, run_id, message

    def _job_messages(self, job: dict[str, Any]) -> list[MemoryMessage]:
        turn_id = str(job["source_turn_id"])
        return [
            MemoryMessage(
                id=message.id,
                role=message.role,
                content=message.content,
            )
            for message in self.repository.turn_memory_messages(turn_id)
            if message.role in {"user", "assistant"}
        ]

    def _jobs_for_rejections(
        self,
        jobs: tuple[dict[str, Any], ...],
        rejections: Sequence[ToolMemoryPatchRejection],
    ) -> dict[str, set[int]]:
        message_jobs = {
            message.id: str(job["id"])
            for job in jobs
            for message in self._job_messages(job)
        }
        affected: dict[str, set[int]] = {}
        for rejection_index, rejection in enumerate(rejections):
            for message_id in rejection.evidence_message_ids:
                job_id = message_jobs.get(message_id)
                if job_id is not None:
                    affected.setdefault(job_id, set()).add(rejection_index)
        return affected

    @staticmethod
    def _rejection_error(
        rejections: Sequence[ToolMemoryPatchRejection],
        indexes: set[int],
    ) -> str:
        details = [
            f"{rejections[index].code}: {rejections[index].message}"
            for index in sorted(indexes)
        ]
        return "Rejected memory patch subset: " + "; ".join(details)

    def _invoke_with_timeout(self, operation):
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="patch-memory-model",
        )
        future = executor.submit(operation)
        try:
            return future.result(timeout=self.model_timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(
                "PatchMemoryModel timed out after "
                f"{self.model_timeout_seconds:g}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _estimated_cost(self, usage: ModelUsage) -> float:
        if self.model.backend != "openai":
            return 0.0
        return (
            usage.input_tokens * self.memory_input_cost_per_million
            + usage.output_tokens * self.memory_output_cost_per_million
        ) / 1_000_000

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, round((time.monotonic() - started) * 1000))


class BackgroundFactMemoryWorker:
    """Idle-time, batched Fact extraction without sending a Tool ontology."""

    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        model: FactMemoryModel,
        user_id: str,
        model_timeout_seconds: float,
        max_attempts: int,
        retry_delay_seconds: float,
        lease_seconds: float,
        memory_input_cost_per_million: float = 0.0,
        memory_output_cost_per_million: float = 0.0,
        batch_turn_limit: int = 32,
        batch_token_limit: int = 4_096,
        linking_context_limit: int = 24,
        minimum_confidence: float = 0.5,
        pii_allowlist: tuple[str, ...] = (),
        token_counter: TokenCounter | None = None,
    ):
        if model_timeout_seconds <= 0:
            raise ValueError("Fact model timeout must be positive")
        if max_attempts < 1:
            raise ValueError("Fact max attempts must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("Fact retry delay cannot be negative")
        if lease_seconds <= model_timeout_seconds:
            raise ValueError("Fact job lease must exceed the model timeout")
        if batch_turn_limit < 1:
            raise ValueError("Fact batch turn limit must be positive")
        if batch_token_limit < 256:
            raise ValueError("Fact batch token limit must be at least 256")
        if linking_context_limit < 1:
            raise ValueError("Fact linking context limit must be positive")
        self.repository = repository
        self.model = model
        self.user_id = user_id
        self.model_timeout_seconds = model_timeout_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.lease_seconds = lease_seconds
        self.memory_input_cost_per_million = memory_input_cost_per_million
        self.memory_output_cost_per_million = memory_output_cost_per_million
        self.batch_turn_limit = batch_turn_limit
        self.batch_token_limit = batch_token_limit
        self.linking_context_limit = linking_context_limit
        self.token_counter = token_counter or TokenCounter()
        self.linker = FactMemoryLinker(
            repository,
            user_id=user_id,
            require_exact_predicate=bool(
                getattr(model, "canonical_predicates", False)
            ),
            exact_predicates=tuple(
                getattr(model, "canonical_predicate_allowlist", ())
            ),
            validation_policy=FactMemoryValidationPolicy(
                minimum_confidence=minimum_confidence,
                reject_pii=True,
                pii_allowlist=pii_allowlist,
                policy_version="fact-memory-gate-v1",
            ),
        )

    def enqueue_turn(self, session_id: str, turn_id: str) -> str:
        return self.repository.enqueue_memory_patch_job(
            session_id=session_id,
            source_turn_id=turn_id,
            max_attempts=self.max_attempts,
        )

    def run_pending(
        self,
        *,
        limit: int,
        session_id: str | None = None,
    ) -> PatchWorkerResult:
        jobs = self.repository.claim_memory_patch_jobs(
            limit=limit,
            lease_seconds=self.lease_seconds,
            session_id=session_id,
        )
        status_counts: Counter[str] = Counter()
        run_ids = []
        errors = []
        for batch in self._batch_jobs(jobs):
            statuses, run_id, batch_errors = self._process_batch(batch)
            run_ids.append(run_id)
            status_counts.update(statuses)
            errors.extend(
                {
                    "job_id": job_id,
                    "run_id": run_id,
                    "error": error,
                }
                for job_id, error in batch_errors.items()
            )
        return PatchWorkerResult(
            claimed_count=len(jobs),
            completed_count=status_counts["completed"],
            retryable_count=status_counts["retryable"],
            failed_count=status_counts["failed"],
            run_ids=tuple(run_ids),
            errors=tuple(errors),
        )

    def _batch_jobs(
        self,
        jobs: Sequence[dict[str, Any]],
    ) -> tuple[tuple[dict[str, Any], ...], ...]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for job in jobs:
            grouped.setdefault(str(job["session_id"]), []).append(job)
        batches = []
        for session_jobs in grouped.values():
            current: list[dict[str, Any]] = []
            current_tokens = 0
            for job in session_jobs:
                messages = self._job_messages(job)
                token_count = sum(
                    self.token_counter.count(message.content) + 8
                    for message in messages
                )
                if current and (
                    len(current) >= self.batch_turn_limit
                    or current_tokens + token_count > self.batch_token_limit
                ):
                    batches.append(tuple(current))
                    current = []
                    current_tokens = 0
                current.append(job)
                current_tokens += token_count
            if current:
                batches.append(tuple(current))
        return tuple(batches)

    def _process_batch(
        self,
        jobs: tuple[dict[str, Any], ...],
    ) -> tuple[tuple[str, ...], str, dict[str, str]]:
        session_ids = {str(job["session_id"]) for job in jobs}
        if len(session_ids) != 1:
            raise ValueError("A Fact batch cannot cross sessions")
        session_id = session_ids.pop()
        job_ids = tuple(str(job["id"]) for job in jobs)
        turn_ids = tuple(str(job["source_turn_id"]) for job in jobs)
        run_id = self.repository.begin_memory_patch_run(
            session_id=session_id,
            source_turn_id=turn_ids[0] if len(jobs) == 1 else None,
            backend=self.model.backend,
            model_id=self.model.model_id,
            prompt_version=self.model.prompt_version,
            schema_version=self.model.schema_version,
            status="running",
            job_id=job_ids[0] if len(jobs) == 1 else None,
        )
        if len(jobs) > 1:
            self.repository.link_memory_patch_jobs_to_batch_run(
                job_ids,
                run_id=run_id,
                session_id=session_id,
            )
        messages = [
            message
            for job in jobs
            for message in self._job_messages(job)
        ]
        all_active = self.repository.list_fact_memory_records(
            session_id,
            user_id=self.user_id,
        )
        linking_context = select_relevant_fact_records(
            messages,
            all_active,
            limit=self.linking_context_limit,
        )
        started = time.monotonic()
        model_call_recorded = False
        try:
            response = self._invoke_with_timeout(
                lambda: self.model.extract(
                    messages,
                    linking_context,
                    user_id=self.user_id,
                    session_id=session_id,
                )
            )
            extraction_latency_ms = self._elapsed_ms(started)
            allowed_message_ids = tuple(message.id for message in messages)
            review_cases = self.linker.semantic_review_cases(
                session_id=session_id,
                candidates=response.candidates,
                allowed_evidence_message_ids=allowed_message_ids,
            )
            semantic_context = list(linking_context)
            semantic_context_ids = {record.id for record in semantic_context}
            active_by_id = {record.id: record for record in all_active}
            for record_id in (
                record_id
                for case in review_cases
                for record_id in case.related_record_ids
            ):
                record = active_by_id.get(record_id)
                if record is not None and record.id not in semantic_context_ids:
                    semantic_context.append(record)
                    semantic_context_ids.add(record.id)
            (
                semantic_decisions,
                semantic_usage,
                semantic_error,
            ) = self._review_uncertain(
                cases=review_cases,
                messages=messages,
                active_records=semantic_context,
                session_id=session_id,
                run_id=run_id,
                turn_id=turn_ids[0] if len(jobs) == 1 else None,
            )
            link_result = self.linker.apply_candidates(
                run_id=run_id,
                session_id=session_id,
                candidates=response.candidates,
                allowed_evidence_message_ids=allowed_message_ids,
                semantic_decisions=semantic_decisions,
            )
            decision_counts = Counter(
                outcome.decision for outcome in link_result.outcomes
            )
            self.repository.record_model_call(
                session_id=session_id,
                turn_id=turn_ids[0] if len(jobs) == 1 else None,
                memory_patch_run_id=run_id,
                role="fact_memory_extraction",
                backend=self.model.backend,
                model_id=self.model.model_id,
                prompt_version=self.model.prompt_version,
                schema_version=self.model.schema_version,
                latency_ms=extraction_latency_ms,
                usage=response.usage,
                response_id=response.response_id,
                metadata={
                    **dict(response.metadata),
                    "job_ids": list(job_ids),
                    "source_turn_ids": list(turn_ids),
                    "batch_turn_count": len(jobs),
                    "batch_message_count": len(messages),
                    "batch_token_limit": self.batch_token_limit,
                    "active_record_count": len(all_active),
                    "linking_context_count": len(linking_context),
                    "candidate_count": len(response.candidates),
                    "semantic_review_case_count": len(review_cases),
                    "semantic_review_called": bool(review_cases)
                    and semantic_error != "semantic_reviewer_unavailable",
                    "semantic_review_error": semantic_error,
                    "decision_counts": dict(decision_counts),
                    "tool_ontology_included": False,
                    "estimated_cost_usd": self._estimated_cost(response.usage),
                },
            )
            model_call_recorded = True
            run_usage = self._sum_usage(response.usage, semantic_usage)
            with self.repository.memory_patch_transaction():
                self.repository.complete_memory_patch_jobs(job_ids)
                self.repository.finish_memory_patch_run(
                    run_id,
                    status="completed",
                    usage=run_usage,
                )
            return tuple("completed" for _ in jobs), run_id, {}
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if not model_call_recorded:
                self.repository.record_model_call(
                    session_id=session_id,
                    turn_id=turn_ids[0] if len(jobs) == 1 else None,
                    memory_patch_run_id=run_id,
                    role="fact_memory_extraction",
                    backend=self.model.backend,
                    model_id=self.model.model_id,
                    prompt_version=self.model.prompt_version,
                    schema_version=self.model.schema_version,
                    latency_ms=self._elapsed_ms(started),
                    usage=ModelUsage(),
                    response_id=None,
                    metadata={
                        "job_ids": list(job_ids),
                        "source_turn_ids": list(turn_ids),
                        "batch_turn_count": len(jobs),
                        "tool_ontology_included": False,
                    },
                    error=message,
                )
            run = self.repository.memory_patch_run(run_id)
            if run["status"] in {"queued", "running"}:
                self.repository.finish_memory_patch_run(
                    run_id,
                    status="failed",
                    error=message,
                )
            statuses = []
            errors = {}
            for job_id in job_ids:
                job = self.repository.memory_patch_job(job_id)
                if job["status"] == "running":
                    status = self.repository.retry_memory_patch_job(
                        job_id,
                        error=message,
                        retry_delay_seconds=self.retry_delay_seconds,
                    )
                    errors[job_id] = message
                else:
                    status = str(job["status"])
                statuses.append(status)
            return tuple(statuses), run_id, errors

    def _review_uncertain(
        self,
        *,
        cases: Sequence[FactMemorySemanticReviewCase],
        messages: Sequence[MemoryMessage],
        active_records: Sequence[FactMemoryRecord],
        session_id: str,
        run_id: str,
        turn_id: str | None,
    ) -> tuple[dict[int, FactMemorySemanticDecision], ModelUsage, str | None]:
        if not cases:
            return {}, ModelUsage(), None
        reviewer = getattr(self.model, "review_uncertain", None)
        if not callable(reviewer):
            return {}, ModelUsage(), "semantic_reviewer_unavailable"
        started = time.monotonic()
        response = None
        try:
            response = self._invoke_with_timeout(
                lambda: reviewer(
                    cases,
                    messages,
                    active_records,
                    user_id=self.user_id,
                    session_id=session_id,
                )
            )
            expected = {case.sequence_index for case in cases}
            decisions: dict[int, FactMemorySemanticDecision] = {}
            for decision in response.decisions:
                if decision.sequence_index in decisions:
                    raise RuntimeError(
                        "Fact semantic review returned a duplicate sequence_index"
                    )
                decisions[decision.sequence_index] = decision
            if set(decisions) != expected:
                raise RuntimeError(
                    "Fact semantic review must return exactly one decision "
                    "for every uncertain candidate"
                )
            decision_counts = Counter(
                decision.decision.strip().upper()
                for decision in decisions.values()
            )
            self.repository.record_model_call(
                session_id=session_id,
                turn_id=turn_id,
                memory_patch_run_id=run_id,
                role="fact_memory_semantic_validation",
                backend=self.model.backend,
                model_id=self.model.model_id,
                prompt_version=getattr(
                    self.model,
                    "semantic_prompt_version",
                    "fact-memory-semantic-review-v1",
                ),
                schema_version=getattr(
                    self.model,
                    "semantic_schema_version",
                    "fact-memory-semantic-decision-v1",
                ),
                latency_ms=self._elapsed_ms(started),
                usage=response.usage,
                response_id=response.response_id,
                metadata={
                    **dict(response.metadata),
                    "case_count": len(cases),
                    "sequence_indices": sorted(expected),
                    "decision_counts": dict(decision_counts),
                    "estimated_cost_usd": self._estimated_cost(response.usage),
                },
            )
            return decisions, response.usage, None
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failed_usage = (
                response.usage if response is not None else ModelUsage()
            )
            failed_metadata = (
                dict(response.metadata) if response is not None else {}
            )
            self.repository.record_model_call(
                session_id=session_id,
                turn_id=turn_id,
                memory_patch_run_id=run_id,
                role="fact_memory_semantic_validation",
                backend=self.model.backend,
                model_id=self.model.model_id,
                prompt_version=getattr(
                    self.model,
                    "semantic_prompt_version",
                    "fact-memory-semantic-review-v1",
                ),
                schema_version=getattr(
                    self.model,
                    "semantic_schema_version",
                    "fact-memory-semantic-decision-v1",
                ),
                latency_ms=self._elapsed_ms(started),
                usage=failed_usage,
                response_id=(
                    response.response_id if response is not None else None
                ),
                metadata={
                    **failed_metadata,
                    "case_count": len(cases),
                    "sequence_indices": sorted(
                        case.sequence_index for case in cases
                    ),
                },
                error=message,
            )
            return {}, failed_usage, type(exc).__name__

    def _job_messages(self, job: Mapping[str, Any]) -> list[MemoryMessage]:
        return [
            MemoryMessage(
                id=message.id,
                role=message.role,
                content=message.content,
            )
            for message in self.repository.turn_memory_messages(
                str(job["source_turn_id"])
            )
            if message.role in {"user", "assistant"}
        ]

    def _invoke_with_timeout(self, operation):
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fact-memory-model",
        )
        future = executor.submit(operation)
        try:
            return future.result(timeout=self.model_timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(
                "FactMemoryModel timed out after "
                f"{self.model_timeout_seconds:g}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _estimated_cost(self, usage: ModelUsage) -> float:
        if self.model.backend != "openai":
            return 0.0
        return (
            usage.input_tokens * self.memory_input_cost_per_million
            + usage.output_tokens * self.memory_output_cost_per_million
        ) / 1_000_000

    @staticmethod
    def _sum_usage(first: ModelUsage, second: ModelUsage) -> ModelUsage:
        return ModelUsage(
            input_tokens=first.input_tokens + second.input_tokens,
            output_tokens=first.output_tokens + second.output_tokens,
            total_tokens=first.total_tokens + second.total_tokens,
            cached_tokens=first.cached_tokens + second.cached_tokens,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, round((time.monotonic() - started) * 1000))
