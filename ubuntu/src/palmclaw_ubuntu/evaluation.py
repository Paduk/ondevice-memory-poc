from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any, Literal
from xml.sax.saxutils import escape

from pydantic import BaseModel, ConfigDict, Field

from palmclaw_ubuntu.application import create_runtime
from palmclaw_ubuntu.config import Settings
from palmclaw_ubuntu.models import (
    ChatMessage,
    MemoryCandidate,
    MemoryEvidence,
    MemoryMessage,
    MemoryResponse,
    ModelUsage,
    StructuredMemoryResponse,
)
from palmclaw_ubuntu.privacy import (
    detect_sensitive_spans,
    redact_data_for_cloud,
)
from palmclaw_ubuntu.providers import FakeEmbeddingModel
from palmclaw_ubuntu.storage import SQLiteRepository


class _EvalMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class _EvalEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_index: int = Field(ge=0)
    quote: str


class _EvalFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    value: str
    scope: Literal["session", "global"]


class _EvalMemory(_EvalFact):
    memory_type: Literal[
        "fact",
        "preference",
        "decision",
        "commitment",
        "profile",
    ] = "fact"
    confidence: float = Field(default=0.95, ge=0, le=1)
    sensitivity: Literal["low", "medium", "high"] = "low"


class _EvalCandidate(_EvalMemory):
    evidence: list[_EvalEvidence]


class _EvalGateExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    decision: Literal["accept", "reject", "review", "supersede"]


class _EvalFactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str


class _EvalPrivacySpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_index: int = Field(ge=0)
    category: str
    text: str


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tags: list[str]
    transcript: list[_EvalMessage]
    candidates: list[_EvalCandidate]
    initial_memories: list[_EvalMemory]
    gold_memories: list[_EvalFact]
    candidate_expectations: list[_EvalGateExpectation]
    query: str
    relevant_facts: list[_EvalFactRef]
    privacy_spans: list[_EvalPrivacySpan]


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str
    cases: list[EvaluationCase]


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    memory_enabled: bool
    memory_strategy: str
    logical_backend: str
    retrieval_mode: str
    gate_enabled: bool
    redaction_enabled: bool
    description: str


@dataclass(frozen=True)
class EvaluationResult:
    run_id: str
    status: str
    artifact_dir: Path
    metrics: dict[str, Any]


BUILTIN_PROFILES: tuple[EvaluationProfile, ...] = (
    EvaluationProfile(
        "no_memory",
        False,
        "structured",
        "none",
        "none",
        True,
        True,
        "No memory baseline",
    ),
    EvaluationProfile(
        "cloud_summary",
        True,
        "summary",
        "openai",
        "full",
        True,
        True,
        "Cloud summary baseline",
    ),
    EvaluationProfile(
        "cloud_structured_bm25",
        True,
        "structured",
        "openai",
        "bm25",
        True,
        True,
        "Cloud structured memory with BM25",
    ),
    EvaluationProfile(
        "cloud_structured_embedding",
        True,
        "structured",
        "openai",
        "embedding",
        True,
        True,
        "Cloud structured memory with deterministic comparison embedding",
    ),
    EvaluationProfile(
        "local_structured_bm25",
        True,
        "structured",
        "local",
        "bm25",
        True,
        True,
        "Local structured memory with BM25",
    ),
    EvaluationProfile(
        "local_structured_embedding",
        True,
        "structured",
        "local",
        "embedding",
        True,
        True,
        "Local structured memory with deterministic comparison embedding",
    ),
    EvaluationProfile(
        "local_full",
        True,
        "structured",
        "local",
        "hybrid",
        True,
        True,
        "Local memory with retrieval, gate, and redaction",
    ),
    EvaluationProfile(
        "ablation_no_retrieval",
        True,
        "structured",
        "local",
        "full",
        True,
        True,
        "Local full-memory injection without top-k retrieval",
    ),
    EvaluationProfile(
        "ablation_no_gate",
        True,
        "structured",
        "local",
        "hybrid",
        False,
        True,
        "Local memory without validation gate",
    ),
    EvaluationProfile(
        "ablation_no_redaction",
        True,
        "structured",
        "openai",
        "hybrid",
        True,
        False,
        "Cloud memory without PII redaction",
    ),
)


class FixtureStructuredMemoryModel:
    prompt_version = "memory-structured-v1"
    schema_version = "structured-v1"

    def __init__(self, profile: EvaluationProfile, case: EvaluationCase):
        self.backend = f"fixture-{profile.logical_backend}"
        self.model_id = f"fixture-{profile.name}"
        self.case = case

    def extract(
        self,
        messages: Sequence[MemoryMessage],
        existing_memory: str,
    ) -> StructuredMemoryResponse:
        del existing_memory
        candidates = []
        for spec in self.case.candidates:
            evidence = []
            for item in spec.evidence:
                if item.message_index >= len(messages):
                    message_id = -1
                else:
                    message_id = messages[item.message_index].id
                evidence.append(
                    MemoryEvidence(
                        message_id=message_id,
                        quote=item.quote,
                    )
                )
            candidates.append(
                MemoryCandidate(
                    subject=spec.subject,
                    predicate=spec.predicate,
                    value=spec.value,
                    scope=spec.scope,
                    memory_type=spec.memory_type,
                    confidence=spec.confidence,
                    sensitivity=spec.sensitivity,
                    evidence=tuple(evidence),
                )
            )
        token_count = sum(len(message.content.split()) for message in messages)
        return StructuredMemoryResponse(
            candidates=tuple(candidates),
            usage=ModelUsage(
                input_tokens=token_count,
                output_tokens=len(candidates) * 20,
                total_tokens=token_count + len(candidates) * 20,
            ),
            metadata={
                "fixture": True,
                "logical_backend": self.backend,
            },
        )


class FixtureSummaryMemoryModel:
    prompt_version = "memory-summary-v1"
    schema_version = "summary-v1"

    def __init__(self, profile: EvaluationProfile, case: EvaluationCase):
        self.backend = f"fixture-{profile.logical_backend}"
        self.model_id = f"fixture-{profile.name}"
        self.profile = profile
        self.case = case

    def consolidate(
        self,
        messages: Sequence[ChatMessage],
        previous_memory: str,
    ) -> MemoryResponse:
        lines = [line for line in previous_memory.splitlines() if line.strip()]
        lines.extend(
            f"- {candidate.subject}.{candidate.predicate}: {candidate.value}"
            for candidate in self.case.candidates
        )
        content = "\n".join(dict.fromkeys(lines))
        privacy_metadata: dict[str, Any] = {
            "destination": "local",
            "detected_count": 0,
            "redacted_count": 0,
            "sensitive_chars_before": 0,
            "sensitive_chars_after": 0,
        }
        if self.profile.logical_backend == "openai":
            sanitized, report = redact_data_for_cloud(
                content,
                include_pii=self.profile.redaction_enabled,
            )
            content = str(sanitized)
            privacy_metadata = report.as_dict(destination="cloud")
        token_count = sum(len(message.content.split()) for message in messages)
        return MemoryResponse(
            content=content,
            usage=ModelUsage(
                input_tokens=token_count,
                output_tokens=len(content.split()),
                total_tokens=token_count + len(content.split()),
            ),
            metadata={
                "fixture": True,
                "logical_backend": self.backend,
                "privacy": privacy_metadata,
            },
        )


def load_evaluation_dataset(
    path: Path | None = None,
) -> tuple[EvaluationDataset, str]:
    if path is None:
        resource = resources.files("palmclaw_ubuntu.evaluation_datasets").joinpath(
            "synthetic_memory_v1.json"
        )
        raw = resource.read_bytes()
    else:
        raw = Path(path).expanduser().resolve().read_bytes()
    dataset = EvaluationDataset.model_validate_json(raw)
    if len({case.id for case in dataset.cases}) != len(dataset.cases):
        raise ValueError("Evaluation case IDs must be unique")
    return dataset, hashlib.sha256(raw).hexdigest()


def evaluation_profiles() -> dict[str, EvaluationProfile]:
    return {profile.name: profile for profile in BUILTIN_PROFILES}


class EvaluationRunner:
    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        settings: Settings,
    ):
        self.repository = repository
        self.settings = settings

    def run(
        self,
        *,
        dataset_path: Path | None = None,
        profile_names: Sequence[str] = (),
        execution_mode: str = "offline",
        repetitions: int = 1,
        seed: int = 17,
        output_root: Path | None = None,
        name: str = "Phase 5 Evaluation",
        case_limit: int | None = None,
    ) -> EvaluationResult:
        if execution_mode not in {"offline", "live"}:
            raise ValueError("execution_mode must be 'offline' or 'live'")
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if case_limit is not None and case_limit < 1:
            raise ValueError("case_limit must be at least 1")
        dataset, dataset_sha256 = load_evaluation_dataset(dataset_path)
        profiles = self._resolve_profiles(profile_names)
        cases = list(dataset.cases)
        if case_limit is not None:
            cases = cases[:case_limit]
        run_id = self.repository.begin_evaluation(
            name=name,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            dataset_sha256=dataset_sha256,
            execution_mode=execution_mode,
            profiles=[profile.name for profile in profiles],
            seed=seed,
            repetitions=repetitions,
        )
        artifact_root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else self.settings.data_dir / "evaluations"
        )
        artifact_dir = artifact_root / run_id
        case_results: list[dict[str, Any]] = []
        randomizer = random.Random(seed)
        try:
            for repetition in range(repetitions):
                scheduled = [(profile, case) for profile in profiles for case in cases]
                randomizer.shuffle(scheduled)
                for profile, case in scheduled:
                    started = time.monotonic()
                    try:
                        result = self._evaluate_case(
                            profile,
                            case,
                            execution_mode=execution_mode,
                        )
                        status = "completed"
                        error = None
                    except Exception as exc:
                        result = self._failed_case_metrics(profile, case)
                        status = "failed"
                        error = f"{type(exc).__name__}: {exc}"
                    latency_ms = max(
                        0,
                        round((time.monotonic() - started) * 1000),
                    )
                    result["metrics"]["case_latency_ms"] = latency_ms
                    record = {
                        "profile": profile.name,
                        "case_id": case.id,
                        "repetition": repetition,
                        "status": status,
                        "latency_ms": latency_ms,
                        "expected": result["expected"],
                        "actual": result["actual"],
                        "metrics": result["metrics"],
                        "error": error,
                    }
                    case_results.append(record)
                    self.repository.record_evaluation_case(
                        evaluation_run_id=run_id,
                        profile=profile.name,
                        case_id=case.id,
                        repetition=repetition,
                        status=status,
                        latency_ms=latency_ms,
                        expected=result["expected"],
                        actual=result["actual"],
                        metrics=result["metrics"],
                        error=error,
                    )
            metrics = self._aggregate(
                dataset=dataset,
                dataset_sha256=dataset_sha256,
                evaluated_case_count=len(cases),
                profiles=profiles,
                execution_mode=execution_mode,
                repetitions=repetitions,
                seed=seed,
                case_results=case_results,
            )
            self._write_artifacts(
                artifact_dir,
                metrics=metrics,
                case_results=case_results,
            )
            with_errors = any(result["status"] == "failed" for result in case_results)
            self.repository.complete_evaluation(
                run_id,
                metrics=metrics,
                artifact_dir=str(artifact_dir),
                with_errors=with_errors,
            )
            return EvaluationResult(
                run_id=run_id,
                status=("completed_with_errors" if with_errors else "completed"),
                artifact_dir=artifact_dir,
                metrics=metrics,
            )
        except Exception as exc:
            self.repository.fail_evaluation(
                run_id,
                f"{type(exc).__name__}: {exc}",
            )
            raise

    @staticmethod
    def _resolve_profiles(
        names: Sequence[str],
    ) -> list[EvaluationProfile]:
        available = evaluation_profiles()
        selected = list(names) or list(available)
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"Unknown evaluation profiles: {', '.join(unknown)}")
        return [available[name] for name in selected]

    def _evaluate_case(
        self,
        profile: EvaluationProfile,
        case: EvaluationCase,
        *,
        execution_mode: str,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(
            prefix=f"palmclaw-eval-{profile.name}-{case.id}-"
        ) as temporary_directory:
            runtime_settings = self._runtime_settings(
                Path(temporary_directory),
                profile,
                execution_mode=execution_mode,
            )
            runtime_kwargs: dict[str, Any] = {
                "embedding_model": FakeEmbeddingModel(
                    runtime_settings.embedding_dimensions
                )
            }
            if execution_mode == "offline":
                runtime_kwargs.update(
                    {
                        "memory_model": FixtureSummaryMemoryModel(
                            profile,
                            case,
                        ),
                        "structured_memory_model": (
                            FixtureStructuredMemoryModel(profile, case)
                        ),
                    }
                )
            with create_runtime(runtime_settings, **runtime_kwargs) as runtime:
                session = runtime.repository.create_session(f"Evaluation {case.id}")
                if profile.memory_enabled:
                    self._seed_initial_memory(
                        runtime,
                        session.id,
                        profile,
                        case,
                    )
                for message in case.transcript:
                    runtime.repository.append_message(
                        session.id,
                        message.role,
                        message.content,
                    )
                consolidation_started = time.monotonic()
                consolidation_run_id = (
                    runtime.memory_engine.consolidate(session.id)
                    if profile.memory_enabled
                    else None
                )
                consolidation_latency_ms = max(
                    0,
                    round((time.monotonic() - consolidation_started) * 1000),
                )
                retrieval_started = time.monotonic()
                retrieval = runtime.memory_engine.retrieve(
                    session.id,
                    turn_id=None,
                    query=case.query,
                    mode=(profile.retrieval_mode if profile.memory_enabled else "none"),
                )
                retrieval_latency_ms = max(
                    0,
                    round((time.monotonic() - retrieval_started) * 1000),
                )
                return self._score_case(
                    runtime=runtime,
                    session_id=session.id,
                    profile=profile,
                    case=case,
                    consolidation_run_id=consolidation_run_id,
                    retrieval=retrieval,
                    consolidation_latency_ms=consolidation_latency_ms,
                    retrieval_latency_ms=retrieval_latency_ms,
                )

    def _runtime_settings(
        self,
        data_dir: Path,
        profile: EvaluationProfile,
        *,
        execution_mode: str,
    ) -> Settings:
        memory_backend = (
            profile.logical_backend
            if execution_mode == "live" and profile.memory_enabled
            else "fake"
        )
        if memory_backend == "none":
            memory_backend = "fake"
        return replace(
            self.settings,
            data_dir=data_dir,
            database_path=data_dir / "palmclaw.db",
            workspace_root=data_dir / "workspaces",
            shared_workspace_root=data_dir / "shared",
            workspace_skills_root=data_dir / "skills",
            backend="fake",
            memory_backend=memory_backend,
            memory_strategy=profile.memory_strategy,
            retrieval_mode=profile.retrieval_mode,
            memory_top_k=2,
            memory_trigger_messages=1,
            memory_gate_enabled=profile.gate_enabled,
            cloud_pii_redaction=profile.redaction_enabled,
            local_pii_storage=("redacted" if profile.redaction_enabled else "raw"),
        )

    @staticmethod
    def _seed_initial_memory(runtime, session_id, profile, case) -> None:
        if not case.initial_memories:
            return
        evidence_content = " | ".join(memory.value for memory in case.initial_memories)
        message_id = runtime.repository.append_message(
            session_id,
            "user",
            evidence_content,
        )
        run_id = runtime.repository.begin_consolidation(
            session_id,
            message_id,
            message_id,
            backend="evaluation-seed",
            model_id="evaluation-seed",
            prompt_version="evaluation-seed-v1",
            schema_version=(
                "summary-v1"
                if profile.memory_strategy == "summary"
                else "structured-v1"
            ),
        )
        if profile.memory_strategy == "summary":
            content = "\n".join(
                f"- {memory.subject}.{memory.predicate}: {memory.value}"
                for memory in case.initial_memories
            )
            runtime.repository.complete_consolidation(
                run_id,
                session_id,
                content,
                ModelUsage(),
            )
            return
        for spec in case.initial_memories:
            candidate = MemoryCandidate(
                subject=spec.subject,
                predicate=spec.predicate,
                value=spec.value,
                scope=spec.scope,
                memory_type=spec.memory_type,
                confidence=spec.confidence,
                sensitivity=spec.sensitivity,
                evidence=(
                    MemoryEvidence(
                        message_id=message_id,
                        quote=spec.value,
                        start_char=evidence_content.find(spec.value),
                        end_char=(evidence_content.find(spec.value) + len(spec.value)),
                    ),
                ),
            )
            runtime.repository.apply_structured_memory(
                session_id=session_id,
                candidate=candidate,
                fact_key=runtime.memory_engine._fact_key(
                    session_id,
                    candidate,
                ),
                consolidation_run_id=run_id,
                model_id="evaluation-seed",
                schema_version="structured-v1",
            )
        runtime.repository.complete_structured_consolidation(
            run_id,
            output={"evaluation_seed": len(case.initial_memories)},
            usage=ModelUsage(),
        )

    def _score_case(
        self,
        *,
        runtime,
        session_id: str,
        profile: EvaluationProfile,
        case: EvaluationCase,
        consolidation_run_id: str | None,
        retrieval,
        consolidation_latency_ms: int,
        retrieval_latency_ms: int,
    ) -> dict[str, Any]:
        all_memories = runtime.repository.list_memories(
            session_id,
            include_superseded=True,
        )
        active_memories = runtime.repository.active_structured_memories(session_id)
        summary_content = (
            runtime.repository.latest_memory(session_id)
            if profile.memory_strategy == "summary"
            else ""
        )
        gold = {self._fact_tuple(memory) for memory in case.gold_memories}
        gold_values = {
            (memory.value.casefold(), memory.scope) for memory in case.gold_memories
        }
        if not profile.memory_enabled:
            actual = set()
            actual_count = 0
        elif profile.memory_strategy == "summary":
            matched = {
                fact
                for fact in gold
                if fact[2].casefold() in summary_content.casefold()
            }
            line_count = len(
                [line for line in summary_content.splitlines() if line.strip()]
            )
            actual = matched
            actual_count = line_count
        else:
            actual = {
                (
                    self._canonical_name(memory.subject),
                    self._canonical_name(memory.predicate),
                    memory.value.casefold(),
                    memory.scope,
                )
                for memory in active_memories
            }
            actual_count = len(actual)
        memory_tp = len(gold & actual)
        memory_fp = max(0, actual_count - memory_tp)
        memory_fn = len(gold - actual)
        if profile.memory_strategy == "summary":
            actual_values = {(value, scope) for _, _, value, scope in actual}
        else:
            actual_values = {
                (memory.value.casefold(), memory.scope) for memory in active_memories
            }
        memory_value_tp = len(gold_values & actual_values)
        memory_value_fp = max(0, actual_count - memory_value_tp)
        memory_value_fn = len(gold_values - actual_values)

        candidate_records = [
            memory
            for memory in all_memories
            if memory.get("consolidation_run_id") == consolidation_run_id
        ]
        gate_by_ref: dict[tuple[str, str], dict[str, Any]] = {}
        for memory in candidate_records:
            detail = runtime.repository.memory_detail(memory["id"])
            gate = detail.get("gate_decision")
            if gate:
                gate_by_ref[
                    (
                        self._canonical_name(str(memory["subject"])),
                        self._canonical_name(str(memory["predicate"])),
                    )
                ] = gate
        expected_gate = {
            (
                self._canonical_name(item.subject),
                self._canonical_name(item.predicate),
            ): item.decision
            for item in case.candidate_expectations
        }
        gate_exact = sum(
            int(gate_by_ref.get(key, {}).get("decision") == decision)
            for key, decision in expected_gate.items()
        )
        initial_refs = {
            (
                self._canonical_name(item.subject),
                self._canonical_name(item.predicate),
            )
            for item in case.initial_memories
        }
        conflict_expected = {
            key
            for key, decision in expected_gate.items()
            if decision in {"review", "supersede"} and key in initial_refs
        }
        conflict_predicted = {
            key
            for key, gate in gate_by_ref.items()
            if gate.get("conflict_relation") == "contradiction"
        }

        relevant = {
            (
                self._canonical_name(item.subject),
                self._canonical_name(item.predicate),
            )
            for item in case.relevant_facts
        }
        selected_refs = [
            (
                self._canonical_name(memory.subject),
                self._canonical_name(memory.predicate),
            )
            for memory in retrieval.memories
        ]
        selected_values = {memory.value.casefold() for memory in retrieval.memories}
        if profile.memory_strategy == "summary" and relevant:
            relevant_values = {
                fact[2] for fact in gold if (fact[0], fact[1]) in relevant
            }
            summary_hit = any(
                value in retrieval.content.casefold() for value in relevant_values
            )
            selected_refs = [next(iter(relevant))] if summary_hit else []
            selected_values = set(relevant_values) if summary_hit else set()
        relevant_hits = [ref for ref in selected_refs if ref in relevant]
        retrieval_recall = len(set(relevant_hits)) / len(relevant) if relevant else None
        reciprocal_rank = 0.0
        for rank, ref in enumerate(selected_refs, start=1):
            if ref in relevant:
                reciprocal_rank = 1.0 / rank
                break
        ndcg = self._ndcg(selected_refs, relevant) if relevant else None
        task_eligible = bool(relevant)
        task_values = {fact[2] for fact in gold if (fact[0], fact[1]) in relevant}
        task_success = int(task_eligible and task_values.issubset(selected_values))
        irrelevant_selected = sum(ref not in relevant for ref in selected_refs)

        privacy = self._privacy_metrics(profile, case)
        backend_report = runtime.repository.memory_backend_report(session_id)
        profile_backend = next(
            (
                backend
                for backend in backend_report["backends"]
                if backend["backend"] != "evaluation-seed"
            ),
            {},
        )
        expected_payload = {
            "gold_memories": [
                memory.model_dump(mode="json") for memory in case.gold_memories
            ],
            "gate_decisions": [
                item.model_dump(mode="json") for item in case.candidate_expectations
            ],
            "relevant_facts": [
                item.model_dump(mode="json") for item in case.relevant_facts
            ],
        }
        actual_payload = {
            "active_memories": [
                {
                    "subject": memory.subject,
                    "predicate": memory.predicate,
                    "value": memory.value,
                    "scope": memory.scope,
                }
                for memory in active_memories
            ],
            "summary": summary_content,
            "gate_decisions": {
                f"{key[0]}.{key[1]}": value for key, value in gate_by_ref.items()
            },
            "retrieved_memory_ids": [memory.id for memory in retrieval.memories],
            "retrieved_facts": [
                f"{subject}.{predicate}" for subject, predicate in selected_refs
            ],
            "consolidation_run_id": consolidation_run_id,
            "retrieval_run_id": retrieval.run_id,
        }
        metrics = {
            "memory_tp": memory_tp,
            "memory_fp": memory_fp,
            "memory_fn": memory_fn,
            "memory_value_tp": memory_value_tp,
            "memory_value_fp": memory_value_fp,
            "memory_value_fn": memory_value_fn,
            # Predicate naming can drift while the stored value remains
            # semantically correct, so unsupported-memory uses value-level
            # false positives. Strict schema drift is still visible in
            # memory_f1 versus memory_value_f1.
            "unsupported_verified": memory_value_fp,
            "verified_total": actual_count,
            "gate_exact": gate_exact,
            "gate_total": len(expected_gate),
            "conflict_tp": len(conflict_expected & conflict_predicted),
            "conflict_fp": len(conflict_predicted - conflict_expected),
            "conflict_fn": len(conflict_expected - conflict_predicted),
            "wrong_auto_overwrite": self._wrong_auto_overwrite(
                case,
                active_memories,
            ),
            "task_eligible": int(task_eligible),
            "task_success": task_success,
            "retrieval_eligible": int(bool(relevant)),
            "retrieval_recall": retrieval_recall,
            "reciprocal_rank": reciprocal_rank,
            "ndcg": ndcg,
            "selected_count": len(selected_refs),
            "irrelevant_selected": irrelevant_selected,
            "consolidation_latency_ms": consolidation_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "input_tokens": int(profile_backend.get("input_tokens", 0)),
            "output_tokens": int(profile_backend.get("output_tokens", 0)),
            "estimated_cost_usd": float(profile_backend.get("estimated_cost_usd", 0.0)),
            **privacy,
        }
        return {
            "expected": expected_payload,
            "actual": actual_payload,
            "metrics": metrics,
        }

    @staticmethod
    def _fact_tuple(memory: _EvalFact) -> tuple[str, str, str, str]:
        return (
            EvaluationRunner._canonical_name(memory.subject),
            EvaluationRunner._canonical_name(memory.predicate),
            memory.value.casefold(),
            memory.scope,
        )

    @staticmethod
    def _canonical_name(value: str) -> str:
        return re.sub(
            r"[_\W]+",
            "_",
            value.casefold().strip(),
            flags=re.UNICODE,
        ).strip("_")

    @staticmethod
    def _wrong_auto_overwrite(case, active_memories) -> int:
        expected_review = {
            (
                EvaluationRunner._canonical_name(item.subject),
                EvaluationRunner._canonical_name(item.predicate),
            )
            for item in case.candidate_expectations
            if item.decision == "review"
        }
        candidate_values = {
            (
                EvaluationRunner._canonical_name(item.subject),
                EvaluationRunner._canonical_name(item.predicate),
            ): (item.value.casefold())
            for item in case.candidates
        }
        return sum(
            1
            for memory in active_memories
            if (
                EvaluationRunner._canonical_name(memory.subject),
                EvaluationRunner._canonical_name(memory.predicate),
            )
            in expected_review
            and memory.value.casefold()
            == candidate_values.get(
                (
                    EvaluationRunner._canonical_name(memory.subject),
                    EvaluationRunner._canonical_name(memory.predicate),
                )
            )
        )

    @staticmethod
    def _ndcg(
        selected: Sequence[tuple[str, str]],
        relevant: set[tuple[str, str]],
    ) -> float:
        dcg = sum(
            (1.0 / math.log2(rank + 1)) if item in relevant else 0.0
            for rank, item in enumerate(selected, start=1)
        )
        ideal_count = min(len(relevant), len(selected))
        if ideal_count == 0:
            return 0.0
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        return dcg / ideal if ideal else 0.0

    @staticmethod
    def _privacy_metrics(
        profile: EvaluationProfile,
        case: EvaluationCase,
    ) -> dict[str, Any]:
        gold: set[tuple[int, str, int, int]] = set()
        for span in case.privacy_spans:
            content = case.transcript[span.message_index].content
            start = content.find(span.text)
            if start < 0:
                raise ValueError(f"Privacy fixture text not found in case {case.id}")
            gold.add(
                (
                    span.message_index,
                    span.category,
                    start,
                    start + len(span.text),
                )
            )
        predicted: set[tuple[int, str, int, int]] = set()
        total_input_chars = 0
        for index, message in enumerate(case.transcript):
            total_input_chars += len(message.content)
            for span in detect_sensitive_spans(message.content):
                predicted.add((index, span.category, span.start, span.end))
        sensitive_characters = sum(end - start for _, _, start, end in gold)
        cloud_memory = profile.memory_enabled and profile.logical_backend == "openai"
        exposed_spans = (
            len(gold) if cloud_memory and not profile.redaction_enabled else 0
        )
        exposed_characters = (
            sensitive_characters
            if cloud_memory and not profile.redaction_enabled
            else 0
        )
        false_positive_characters = sum(
            end - start
            for item_index, category, start, end in predicted - gold
            if (item_index, category, start, end) not in gold
        )
        return {
            "pii_tp": len(gold & predicted),
            "pii_fp": len(predicted - gold),
            "pii_fn": len(gold - predicted),
            "sensitive_spans": len(gold),
            "sensitive_characters": sensitive_characters,
            "cloud_exposed_sensitive_spans": exposed_spans,
            "cloud_exposed_sensitive_characters": exposed_characters,
            "total_input_characters": total_input_chars,
            "over_redacted_characters": false_positive_characters,
        }

    @staticmethod
    def _failed_case_metrics(
        profile: EvaluationProfile,
        case: EvaluationCase,
    ) -> dict[str, Any]:
        del profile
        return {
            "expected": {
                "gold_memories": [
                    memory.model_dump(mode="json") for memory in case.gold_memories
                ]
            },
            "actual": {},
            "metrics": {
                "memory_tp": 0,
                "memory_fp": 0,
                "memory_fn": len(case.gold_memories),
                "memory_value_tp": 0,
                "memory_value_fp": 0,
                "memory_value_fn": len(case.gold_memories),
                "task_eligible": int(bool(case.relevant_facts)),
                "task_success": 0,
                "retrieval_eligible": int(bool(case.relevant_facts)),
                "retrieval_recall": 0.0 if case.relevant_facts else None,
                "reciprocal_rank": 0.0,
                "ndcg": 0.0 if case.relevant_facts else None,
            },
        }

    def _aggregate(
        self,
        *,
        dataset: EvaluationDataset,
        dataset_sha256: str,
        evaluated_case_count: int,
        profiles: Sequence[EvaluationProfile],
        execution_mode: str,
        repetitions: int,
        seed: int,
        case_results: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        profile_metrics = {}
        for profile in profiles:
            records = [
                result for result in case_results if result["profile"] == profile.name
            ]
            counts: Counter[str] = Counter()
            retrieval_recalls = []
            reciprocal_ranks = []
            ndcg_scores = []
            latency_values = []
            for record in records:
                metrics = record["metrics"]
                for key, value in metrics.items():
                    if isinstance(value, int):
                        counts[key] += value
                if metrics.get("retrieval_recall") is not None:
                    retrieval_recalls.append(metrics["retrieval_recall"])
                if metrics.get("retrieval_eligible"):
                    reciprocal_ranks.append(metrics.get("reciprocal_rank", 0.0))
                if metrics.get("ndcg") is not None:
                    ndcg_scores.append(metrics["ndcg"])
                latency_values.append(metrics.get("case_latency_ms", 0))
            memory_precision, memory_recall, memory_f1 = self._prf(
                counts["memory_tp"],
                counts["memory_fp"],
                counts["memory_fn"],
            )
            (
                memory_value_precision,
                memory_value_recall,
                memory_value_f1,
            ) = self._prf(
                counts["memory_value_tp"],
                counts["memory_value_fp"],
                counts["memory_value_fn"],
            )
            conflict_precision, conflict_recall, conflict_f1 = self._prf(
                counts["conflict_tp"],
                counts["conflict_fp"],
                counts["conflict_fn"],
            )
            pii_precision, pii_recall, pii_f1 = self._prf(
                counts["pii_tp"],
                counts["pii_fp"],
                counts["pii_fn"],
            )
            profile_metrics[profile.name] = {
                "description": profile.description,
                "logical_backend": profile.logical_backend,
                "memory_strategy": profile.memory_strategy,
                "retrieval_mode": profile.retrieval_mode,
                "gate_enabled": profile.gate_enabled,
                "redaction_enabled": profile.redaction_enabled,
                "cases": len(records),
                "failed_cases": sum(record["status"] == "failed" for record in records),
                "task_success_rate": self._ratio(
                    counts["task_success"],
                    counts["task_eligible"],
                ),
                "memory_precision": memory_precision,
                "memory_recall": memory_recall,
                "memory_f1": memory_f1,
                "memory_value_precision": memory_value_precision,
                "memory_value_recall": memory_value_recall,
                "memory_value_f1": memory_value_f1,
                "unsupported_memory_rate": self._ratio(
                    counts["unsupported_verified"],
                    counts["verified_total"],
                ),
                "gate_decision_accuracy": self._ratio(
                    counts["gate_exact"],
                    counts["gate_total"],
                ),
                "conflict_precision": conflict_precision,
                "conflict_recall": conflict_recall,
                "conflict_f1": conflict_f1,
                "wrong_auto_overwrite_rate": self._ratio(
                    counts["wrong_auto_overwrite"],
                    counts["conflict_tp"] + counts["conflict_fn"],
                ),
                "retrieval_recall_at_k": self._mean(retrieval_recalls),
                "retrieval_mrr": self._mean(reciprocal_ranks),
                "retrieval_ndcg": self._mean(ndcg_scores),
                "irrelevant_memory_injection_rate": self._ratio(
                    counts["irrelevant_selected"],
                    counts["selected_count"],
                ),
                "pii_precision": pii_precision,
                "pii_recall": pii_recall,
                "pii_f1": pii_f1,
                "cloud_exposed_span_rate": self._ratio(
                    counts["cloud_exposed_sensitive_spans"],
                    counts["sensitive_spans"],
                ),
                "cloud_exposed_character_rate": self._ratio(
                    counts["cloud_exposed_sensitive_characters"],
                    counts["sensitive_characters"],
                ),
                "over_redaction_rate": self._ratio(
                    counts["over_redacted_characters"],
                    counts["total_input_characters"],
                ),
                "latency_ms_mean": self._mean(latency_values),
                "consolidation_latency_ms_total": counts["consolidation_latency_ms"],
                "retrieval_latency_ms_total": counts["retrieval_latency_ms"],
                "input_tokens": counts["input_tokens"],
                "output_tokens": counts["output_tokens"],
                "estimated_cost_usd": round(
                    sum(
                        float(record["metrics"].get("estimated_cost_usd", 0.0))
                        for record in records
                    ),
                    8,
                ),
            }
        return {
            "dataset": {
                "name": dataset.name,
                "version": dataset.version,
                "sha256": dataset_sha256,
                "total_case_count": len(dataset.cases),
                "evaluated_case_count": evaluated_case_count,
            },
            "execution": {
                "mode": execution_mode,
                "seed": seed,
                "repetitions": repetitions,
                "prompt_version": "memory-structured-v1",
                "schema_version": "structured-v1",
                "gate_policy_version": "memory-gate-v1",
                "profiles": [profile.name for profile in profiles],
            },
            "profiles": profile_metrics,
        }

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = EvaluationRunner._ratio(tp, tp + fp, empty=1.0)
        recall = EvaluationRunner._ratio(tp, tp + fn, empty=1.0)
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return precision, recall, f1

    @staticmethod
    def _ratio(
        numerator: int,
        denominator: int,
        *,
        empty: float = 0.0,
    ) -> float:
        return numerator / denominator if denominator else empty

    @staticmethod
    def _mean(values: Sequence[float | int]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _write_artifacts(
        self,
        artifact_dir: Path,
        *,
        metrics: dict[str, Any],
        case_results: Sequence[dict[str, Any]],
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=False)
        safe_metrics, _ = redact_data_for_cloud(metrics)
        safe_cases, _ = redact_data_for_cloud(list(case_results))
        (artifact_dir / "metrics.json").write_text(
            json.dumps(safe_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (artifact_dir / "cases.jsonl").open(
            "w",
            encoding="utf-8",
        ) as stream:
            for result in safe_cases:
                stream.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        self._write_tsv(artifact_dir / "results.tsv", safe_metrics)
        self._write_markdown(artifact_dir / "results.md", safe_metrics)
        for metric, title in (
            ("task_success_rate", "Task success rate"),
            ("memory_f1", "Memory F1"),
            ("retrieval_recall_at_k", "Retrieval Recall@k"),
            ("cloud_exposed_span_rate", "Cloud sensitive-span exposure"),
        ):
            self._write_bar_svg(
                artifact_dir / f"{metric}.svg",
                title,
                [
                    (profile, float(values.get(metric, 0.0)))
                    for profile, values in safe_metrics["profiles"].items()
                ],
            )

    @staticmethod
    def _write_tsv(path: Path, metrics: dict[str, Any]) -> None:
        profiles = metrics["profiles"]
        keys = [
            "task_success_rate",
            "memory_precision",
            "memory_recall",
            "memory_f1",
            "memory_value_f1",
            "unsupported_memory_rate",
            "conflict_f1",
            "wrong_auto_overwrite_rate",
            "retrieval_recall_at_k",
            "retrieval_mrr",
            "retrieval_ndcg",
            "pii_f1",
            "cloud_exposed_span_rate",
            "cloud_exposed_character_rate",
            "over_redaction_rate",
            "latency_ms_mean",
            "input_tokens",
            "output_tokens",
            "estimated_cost_usd",
            "failed_cases",
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["profile", *keys])
            for profile, values in profiles.items():
                writer.writerow([profile, *(values.get(key, "") for key in keys)])

    @staticmethod
    def _write_markdown(path: Path, metrics: dict[str, Any]) -> None:
        lines = [
            "# PalmClaw Phase 5 Evaluation",
            "",
            (
                f"Dataset: `{metrics['dataset']['name']}` "
                f"`{metrics['dataset']['version']}`"
            ),
            "",
            "| Profile | Task success | Exact F1 | Value F1 | Recall@k | "
            "Conflict F1 | PII F1 | Cloud exposure |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for profile, values in metrics["profiles"].items():
            lines.append(
                f"| {profile} "
                f"| {values['task_success_rate']:.3f} "
                f"| {values['memory_f1']:.3f} "
                f"| {values['memory_value_f1']:.3f} "
                f"| {values['retrieval_recall_at_k']:.3f} "
                f"| {values['conflict_f1']:.3f} "
                f"| {values['pii_f1']:.3f} "
                f"| {values['cloud_exposed_span_rate']:.3f} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_bar_svg(
        path: Path,
        title: str,
        values: Sequence[tuple[str, float]],
    ) -> None:
        width = 960
        row_height = 34
        left = 270
        chart_width = 620
        height = 80 + len(values) * row_height
        rows = [
            (
                '<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            (
                f'<text x="20" y="32" font-family="sans-serif" '
                f'font-size="20" font-weight="600">{escape(title)}</text>'
            ),
        ]
        for index, (label, raw_value) in enumerate(values):
            value = min(1.0, max(0.0, raw_value))
            y = 58 + index * row_height
            bar_width = chart_width * value
            rows.extend(
                [
                    (
                        f'<text x="20" y="{y + 18}" font-family="monospace" '
                        f'font-size="13">{escape(label)}</text>'
                    ),
                    (
                        f'<rect x="{left}" y="{y}" width="{chart_width}" '
                        'height="20" rx="3" fill="#e5e7eb"/>'
                    ),
                    (
                        f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" '
                        'height="20" rx="3" fill="#2563eb"/>'
                    ),
                    (
                        f'<text x="{left + chart_width + 10}" y="{y + 16}" '
                        f'font-family="sans-serif" font-size="12">{value:.3f}</text>'
                    ),
                ]
            )
        rows.append("</svg>")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
