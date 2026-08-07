from __future__ import annotations

from collections import deque
from dataclasses import replace

from palmclaw_ubuntu.application import create_runtime
from palmclaw_ubuntu.models import (
    MemoryCandidate,
    MemoryEvidence,
    StructuredMemoryResponse,
)


class _CandidateSequenceModel:
    backend = "fake"
    model_id = "phase4-candidates"
    prompt_version = "memory-structured-v1"
    schema_version = "structured-v1"

    def __init__(self, values, *, confidence=0.95, sensitivity="low"):
        self.values = deque(values)
        self.confidence = confidence
        self.sensitivity = sensitivity

    def extract(self, messages, existing_memory):
        del existing_memory
        latest = next(
            message for message in reversed(messages) if message.role == "user"
        )
        value = self.values.popleft()
        return StructuredMemoryResponse(
            candidates=(
                MemoryCandidate(
                    subject="user",
                    predicate="preferred_editor",
                    value=value,
                    scope="global",
                    memory_type="preference",
                    confidence=self.confidence,
                    sensitivity=self.sensitivity,
                    evidence=(
                        MemoryEvidence(
                            message_id=latest.id,
                            quote=latest.content,
                        ),
                    ),
                ),
            )
        )


class _UnknownEvidenceModel(_CandidateSequenceModel):
    def extract(self, messages, existing_memory):
        response = super().extract(messages, existing_memory)
        candidate = response.candidates[0]
        return StructuredMemoryResponse(
            candidates=(replace(candidate, value="Neovim"),)
        )


class _FailingLocalMemoryModel:
    backend = "local"
    model_id = "offline-local-model"
    prompt_version = "memory-structured-v1"
    schema_version = "structured-v1"

    def extract(self, messages, existing_memory):
        del messages, existing_memory
        raise RuntimeError("local endpoint unavailable")


def _phase4_settings(settings):
    return replace(
        settings,
        memory_strategy="structured",
        retrieval_mode="bm25",
        memory_trigger_messages=2,
        memory_gate_enabled=True,
    )


def test_unresolved_conflict_enters_review_and_manual_accept_supersedes(settings):
    model = _CandidateSequenceModel(["Neovim", "VS Code"])
    with create_runtime(
        _phase4_settings(settings),
        structured_memory_model=model,
    ) as runtime:
        session = runtime.repository.create_session("Conflict review")
        runtime.turn_coordinator.run(session.id, "I prefer Neovim")
        runtime.turn_coordinator.run(session.id, "I prefer VS Code")

        active_before = runtime.repository.active_structured_memories(session.id)
        reviews = runtime.repository.list_memory_reviews(session.id)
        reviewed = runtime.repository.resolve_memory_review(
            reviews[0]["id"],
            decision="accept",
            reason="user_confirmed",
        )
        active_after = runtime.repository.active_structured_memories(session.id)
        versions = runtime.repository.memory_versions(active_after[0].fact_key)

    assert [memory.value for memory in active_before] == ["Neovim"]
    assert reviews[0]["gate_reason"] == "unresolved_conflict"
    assert reviews[0]["conflict_relation"] == "contradiction"
    assert reviewed["status"] == "verified"
    assert [memory.value for memory in active_after] == ["VS Code"]
    assert [item["status"] for item in versions] == [
        "superseded",
        "verified",
    ]


def test_semantically_unsupported_candidate_is_rejected(settings):
    model = _UnknownEvidenceModel(["unused"])
    with create_runtime(
        _phase4_settings(settings),
        structured_memory_model=model,
    ) as runtime:
        session = runtime.repository.create_session("Unknown evidence")
        runtime.turn_coordinator.run(session.id, "I enjoy editing text")
        memories = runtime.repository.list_memories(
            session.id,
            include_superseded=True,
        )
        detail = runtime.repository.memory_detail(memories[0]["id"])

    assert memories[0]["status"] == "rejected"
    assert memories[0]["rejection_reason"] == "evidence_unknown"
    assert detail["gate_decision"]["evidence_relation"] == "unknown"


def test_pii_candidate_is_redacted_and_queued_for_review(settings):
    model = _CandidateSequenceModel(
        ["alice@example.com"],
        sensitivity="high",
    )
    with create_runtime(
        _phase4_settings(settings),
        structured_memory_model=model,
    ) as runtime:
        session = runtime.repository.create_session("PII")
        runtime.turn_coordinator.run(
            session.id,
            "My preferred contact is alice@example.com",
        )
        reviews = runtime.repository.list_memory_reviews(session.id)
        detail = runtime.repository.memory_detail(reviews[0]["id"])

    assert reviews[0]["value"] == "[REDACTED_EMAIL]"
    assert reviews[0]["gate_reason"] == "pii_candidate"
    assert reviews[0]["pii_categories"] == ["email"]
    assert "alice@example.com" not in detail["sources"][0]["evidence_text"]


def test_local_memory_failure_is_retryable_and_does_not_fail_agent_turn(settings):
    with create_runtime(
        _phase4_settings(settings),
        structured_memory_model=_FailingLocalMemoryModel(),
    ) as runtime:
        session = runtime.repository.create_session("Local failure")
        result = runtime.turn_coordinator.run(session.id, "Remember this")
        run = runtime.repository._connection.execute(
            "SELECT * FROM consolidation_runs WHERE id = ?",
            (result.consolidation_run_id,),
        ).fetchone()
        report = runtime.repository.memory_backend_report(session.id)

    assert result.status == "completed"
    assert run["status"] == "retryable"
    assert "local endpoint unavailable" in run["error"]
    assert report["backends"][0]["backend"] == "local"
    assert report["backends"][0]["errors"] == 1
    assert report["backends"][0]["retryable_runs"] == 1


def test_backend_report_includes_gate_and_privacy_proxies(settings):
    model = _CandidateSequenceModel(["Neovim"])
    with create_runtime(
        _phase4_settings(settings),
        structured_memory_model=model,
    ) as runtime:
        session = runtime.repository.create_session("Report")
        runtime.turn_coordinator.run(session.id, "I prefer Neovim")
        report = runtime.repository.memory_backend_report(session.id)

    fake = report["backends"][0]
    assert fake["calls"] == 1
    assert fake["gate_outcomes"] == {"accept": 1}
    assert fake["input_tokens"] == 0
