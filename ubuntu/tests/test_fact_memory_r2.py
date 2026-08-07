from __future__ import annotations

from palmclaw_ubuntu.background_memory import BackgroundFactMemoryWorker
from palmclaw_ubuntu.fact_memory import FactMemoryLinker
from palmclaw_ubuntu.models import (
    FactMemoryCandidate,
    FactMemoryExtractionResponse,
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
    value: int,
    *,
    entity_id: str = "Patricia",
    predicate: str = "headrest height",
    directive: str = "UPSERT",
) -> FactMemoryCandidate:
    return FactMemoryCandidate(
        entity_id=entity_id,
        predicate=predicate,
        value=value,
        identity_conditions={},
        applicability={},
        capability_hints=("seat",),
        memory_type="preference",
        confidence=0.95,
        evidence=(MemoryEvidence(message_id=message_id, quote=quote),),
        directive=directive,
    )


def test_fact_linker_applies_ordered_add_update_and_noop(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Fact linker")
        first = "Patricia prefers headrest height 44."
        second = "Patricia now prefers headrest height 48."
        confirm = "Patricia still prefers headrest height 48."
        _, first_id = _turn(repository, session.id, first)
        _, second_id = _turn(repository, session.id, second)
        _, confirm_id = _turn(repository, session.id, confirm)
        run_id = _run(repository, session.id)
        linker = FactMemoryLinker(repository, user_id="Driver One")

        candidates = (
            _candidate(first_id, first, 44),
            _candidate(second_id, second, 48),
            _candidate(confirm_id, confirm, 48),
        )
        result = linker.apply_candidates(
            run_id=run_id,
            session_id=session.id,
            candidates=candidates,
        )
        replay = linker.apply_candidates(
            run_id=run_id,
            session_id=session.id,
            candidates=candidates,
        )
        active = repository.list_fact_memory_records(session.id)
        versions = repository.fact_memory_record_versions(active[0].record_key)
        events = repository.fact_memory_candidate_events(run_id)

        assert [outcome.decision for outcome in result.outcomes] == [
            "ADD",
            "UPDATE",
            "NOOP",
        ]
        assert [outcome.status for outcome in result.outcomes] == [
            "applied",
            "applied",
            "noop",
        ]
        assert len(active) == 1
        assert active[0].value == 48
        assert [(item.version, item.status) for item in versions] == [
            (1, "superseded"),
            (2, "active"),
        ]
        assert [event["decision"] for event in events] == [
            "ADD",
            "UPDATE",
            "NOOP",
        ]
        assert replay.outcomes == result.outcomes


def test_fact_linker_uses_alias_and_predicate_similarity(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Fact aliases")
        first = "Patricia Garcia prefers headrest height 44."
        second = "Patricia now wants headrest position 48."
        _, first_id = _turn(repository, session.id, first)
        _, second_id = _turn(repository, session.id, second)
        linker = FactMemoryLinker(repository, user_id="Driver One")

        first_result = linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(
                _candidate(
                    first_id,
                    first,
                    44,
                    entity_id="Patricia Garcia",
                ),
            ),
        )
        second_run = _run(repository, session.id)
        second_result = linker.apply_candidates(
            run_id=second_run,
            session_id=session.id,
            candidates=(
                _candidate(
                    second_id,
                    second,
                    48,
                    entity_id="Patricia",
                    predicate="headrest position",
                ),
            ),
        )
        updated = repository.fact_memory_record(
            second_result.outcomes[0].result_record_id
        )

        assert first_result.outcomes[0].decision == "ADD"
        assert second_result.outcomes[0].decision == "UPDATE"
        assert updated is not None
        assert updated.entity_id == "patricia_garcia"
        assert updated.predicate == "headrest_height"
        assert updated.value == 48
        assert (
            repository.fact_memory_candidate_events(second_run)[0]["match_score"]
            > 0.62
        )


def test_fact_worker_batches_turns_and_records_usage(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Fact worker")
        first = "Patricia prefers headrest height 44."
        second = "Patricia now prefers headrest height 48."
        first_turn, first_id = _turn(repository, session.id, first)
        second_turn, second_id = _turn(repository, session.id, second)
        model = ScriptedFactMemoryModel(
            [
                FactMemoryExtractionResponse(
                    candidates=(
                        _candidate(first_id, first, 44),
                        _candidate(second_id, second, 48),
                    ),
                    usage=ModelUsage(
                        input_tokens=120,
                        output_tokens=40,
                        total_tokens=160,
                    ),
                    metadata={"test": True},
                )
            ]
        )
        model.backend = "openai"
        worker = BackgroundFactMemoryWorker(
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
        worker.enqueue_turn(session.id, first_turn)
        worker.enqueue_turn(session.id, second_turn)

        result = worker.run_pending(limit=32, session_id=session.id)
        metrics = repository.fact_memory_metrics(session.id)
        trace = repository.memory_patch_trace(result.run_ids[0])
        request = model.requests[0]

        assert result.claimed_count == 2
        assert result.completed_count == 2
        assert result.retryable_count == 0
        assert len(model.requests) == 1
        assert len(request[0]) == 2
        assert request[1] == ()
        assert metrics["decision_counts"] == {"ADD": 1, "UPDATE": 1}
        assert metrics["active_record_count"] == 1
        assert metrics["model_call_count"] == 1
        assert metrics["model_usage"]["input_tokens"] == 120
        metadata = trace["model_calls"][0]["metadata"]
        assert metadata["tool_ontology_included"] is False
        assert metadata["batch_turn_count"] == 2
        assert metadata["estimated_cost_usd"] == 0.0002


def test_fact_worker_completes_review_without_repeated_model_retry(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Fact review")
        text = "Patricia prefers headrest height 44."
        turn_id, message_id = _turn(repository, session.id, text)
        model = ScriptedFactMemoryModel(
            [
                FactMemoryExtractionResponse(
                    candidates=(
                        _candidate(
                            message_id,
                            "quote not present in the source",
                            44,
                        ),
                    )
                )
            ]
        )
        worker = BackgroundFactMemoryWorker(
            repository=repository,
            model=model,
            user_id="Driver One",
            model_timeout_seconds=2,
            max_attempts=3,
            retry_delay_seconds=0,
            lease_seconds=3,
        )
        job_id = worker.enqueue_turn(session.id, turn_id)

        result = worker.run_pending(limit=1, session_id=session.id)
        events = repository.fact_memory_candidate_events(result.run_ids[0])

        assert result.completed_count == 1
        assert result.retryable_count == 0
        assert repository.memory_patch_job(job_id)["attempt_count"] == 1
        assert events[0]["decision"] == "REJECT"
        assert events[0]["status"] == "rejected"
        assert events[0]["validation"]["code"] == "unsupported_evidence"
        assert repository.list_fact_memory_records(session.id) == []
        assert len(model.requests) == 1
