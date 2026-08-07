from __future__ import annotations

from palmclaw_ubuntu.background_memory import BackgroundFactMemoryWorker
from palmclaw_ubuntu.fact_memory import FactMemoryLinker
from palmclaw_ubuntu.models import (
    FactMemoryCandidate,
    FactMemoryExtractionResponse,
    FactMemorySemanticDecision,
    FactMemorySemanticReviewResponse,
    MemoryEvidence,
    ModelUsage,
)
from palmclaw_ubuntu.providers import ScriptedFactMemoryModel
from palmclaw_ubuntu.storage import SQLiteRepository


def _turn(
    repository: SQLiteRepository,
    session_id: str,
    content: str,
) -> tuple[str, int]:
    turn_id = repository.create_turn(session_id)
    message_id = repository.append_message(
        session_id,
        "user",
        content,
        turn_id=turn_id,
    )
    repository.finish_turn(turn_id, "completed", 1, "test")
    return turn_id, message_id


def _run(repository: SQLiteRepository, session_id: str) -> str:
    return repository.begin_memory_patch_run(
        session_id=session_id,
        source_turn_id=None,
        backend="fake",
        model_id="fact-model",
        prompt_version="fact-memory-extraction-v1",
        schema_version="fact-memory-candidate-v1",
        status="running",
    )


def _candidate(
    message_id: int,
    quote: str,
    value,
    *,
    entity_id: str,
    predicate: str,
) -> FactMemoryCandidate:
    return FactMemoryCandidate(
        entity_id=entity_id,
        predicate=predicate,
        value=value,
        identity_conditions={},
        applicability={},
        capability_hints=("vehicle",),
        memory_type="preference",
        confidence=0.95,
        evidence=(MemoryEvidence(message_id=message_id, quote=quote),),
    )


def _worker(
    repository: SQLiteRepository,
    model: ScriptedFactMemoryModel,
) -> BackgroundFactMemoryWorker:
    return BackgroundFactMemoryWorker(
        repository=repository,
        model=model,
        user_id="Driver One",
        model_timeout_seconds=2,
        max_attempts=2,
        retry_delay_seconds=0,
        lease_seconds=3,
        batch_turn_limit=32,
        batch_token_limit=4_096,
        memory_input_cost_per_million=1.0,
        memory_output_cost_per_million=2.0,
    )


def test_r31_reviews_only_uncertain_candidates_in_one_batch(settings) -> None:
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("R3.1 semantic batch")
        initial = "Patricia prefers headrest height 44."
        _, initial_id = _turn(repository, session.id, initial)
        FactMemoryLinker(repository, user_id="Driver One").apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(
                _candidate(
                    initial_id,
                    initial,
                    44,
                    entity_id="Patricia",
                    predicate="headrest height",
                ),
            ),
        )

        update = "Patricia's headrest height is 48."
        glow = "Gary wants a calming cabin glow."
        invalid = "Dana prefers fan speed five."
        update_turn, update_id = _turn(repository, session.id, update)
        glow_turn, glow_id = _turn(repository, session.id, glow)
        invalid_turn, invalid_id = _turn(repository, session.id, invalid)
        candidates = (
            _candidate(
                update_id,
                update,
                48,
                entity_id="Patricia",
                predicate="headrest height",
            ),
            _candidate(
                glow_id,
                glow,
                "soft blue",
                entity_id="Gary",
                predicate="ambient color",
            ),
            _candidate(
                invalid_id,
                "quote absent from source",
                5,
                entity_id="Dana",
                predicate="fan speed",
            ),
        )
        model = ScriptedFactMemoryModel(
            [
                FactMemoryExtractionResponse(
                    candidates=candidates,
                    usage=ModelUsage(
                        input_tokens=100,
                        output_tokens=20,
                        total_tokens=120,
                    ),
                )
            ],
            semantic_responses=[
                FactMemorySemanticReviewResponse(
                    decisions=(
                        FactMemorySemanticDecision(
                            sequence_index=0,
                            decision="ACCEPT",
                            confidence=0.94,
                            reason="The statement semantically updates the value.",
                            evidence_relation="entailment",
                        ),
                        FactMemorySemanticDecision(
                            sequence_index=1,
                            decision="ACCEPT",
                            confidence=0.91,
                            reason="Calming glow supports the proposed preference.",
                            evidence_relation="entailment",
                        ),
                    ),
                    usage=ModelUsage(
                        input_tokens=30,
                        output_tokens=10,
                        total_tokens=40,
                    ),
                )
            ],
        )
        worker = _worker(repository, model)
        for turn_id in (update_turn, glow_turn, invalid_turn):
            worker.enqueue_turn(session.id, turn_id)

        result = worker.run_pending(limit=32, session_id=session.id)
        trace = repository.memory_patch_trace(result.run_ids[0])
        records = {
            (record.entity_id, record.predicate): record.value
            for record in repository.list_fact_memory_records(session.id)
        }
        metrics = repository.fact_memory_metrics(session.id)

        assert result.completed_count == 3
        assert len(model.requests) == 1
        assert len(model.semantic_requests) == 1
        review_cases = model.semantic_requests[0][0]
        assert [
            (case.sequence_index, case.review_codes, case.proposed_operation)
            for case in review_cases
        ] == [
            (0, ("update_intent_uncertain",), "UPDATE"),
            (1, ("value_entailment_uncertain",), "ADD"),
        ]
        assert len(review_cases[0].related_record_ids) == 1
        assert review_cases[1].related_record_ids == ()
        assert records[("patricia", "headrest_height")] == 48
        assert records[("gary", "ambient_color")] == "soft blue"
        assert [event["decision"] for event in trace["fact_candidates"]] == [
            "UPDATE",
            "ADD",
            "REJECT",
        ]
        assert [
            event["validation"]["code"]
            for event in trace["fact_candidates"]
        ] == [
            "semantic_accepted",
            "semantic_accepted",
            "unsupported_evidence",
        ]
        semantic_calls = [
            call
            for call in trace["model_calls"]
            if call["role"] == "fact_memory_semantic_validation"
        ]
        assert len(semantic_calls) == 1
        assert semantic_calls[0]["metadata"]["case_count"] == 2
        assert semantic_calls[0]["usage"]["total_tokens"] == 40
        assert trace["run"]["usage"]["total_tokens"] == 160
        assert metrics["semantic_model_call_count"] == 1
        assert metrics["semantic_model_usage"]["total_tokens"] == 40


def test_r31_semantic_failure_keeps_review_and_completes_batch(settings) -> None:
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("R3.1 fail safe")
        text = "Gary wants a calming cabin glow."
        turn_id, message_id = _turn(repository, session.id, text)
        model = ScriptedFactMemoryModel(
            [
                FactMemoryExtractionResponse(
                    candidates=(
                        _candidate(
                            message_id,
                            text,
                            "soft blue",
                            entity_id="Gary",
                            predicate="ambient color",
                        ),
                    )
                )
            ],
            semantic_responses=[RuntimeError("temporary semantic failure")],
        )
        worker = _worker(repository, model)
        job_id = worker.enqueue_turn(session.id, turn_id)

        result = worker.run_pending(limit=1, session_id=session.id)
        trace = repository.memory_patch_trace(result.run_ids[0])

        assert result.completed_count == 1
        assert result.retryable_count == 0
        assert repository.memory_patch_job(job_id)["attempt_count"] == 1
        assert trace["fact_candidates"][0]["decision"] == "REVIEW"
        assert (
            trace["fact_candidates"][0]["validation"]["code"]
            == "value_entailment_uncertain"
        )
        semantic_call = next(
            call
            for call in trace["model_calls"]
            if call["role"] == "fact_memory_semantic_validation"
        )
        assert semantic_call["error"] == "RuntimeError: temporary semantic failure"
        assert repository.list_fact_memory_records(session.id) == []
        assert len(model.requests) == 1
        assert len(model.semantic_requests) == 1
