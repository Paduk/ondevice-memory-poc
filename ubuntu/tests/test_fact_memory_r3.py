from __future__ import annotations

from typing import Any

from palmclaw_ubuntu.fact_memory import FactMemoryLinker
from palmclaw_ubuntu.models import FactMemoryCandidate, MemoryEvidence
from palmclaw_ubuntu.storage import SQLiteRepository


def _message(
    repository: SQLiteRepository,
    session_id: str,
    content: str,
    *,
    role: str = "user",
) -> int:
    turn_id = repository.create_turn(session_id)
    message_id = repository.append_message(
        session_id,
        role,
        content,
        turn_id=turn_id,
    )
    repository.finish_turn(turn_id, "completed", 1, "test")
    return message_id


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
    value: Any,
    *,
    entity_id: str = "Patricia",
    predicate: str = "headrest height",
    identity_conditions: dict[str, Any] | None = None,
    applicability: dict[str, Any] | None = None,
    confidence: float = 0.95,
    directive: str = "UPSERT",
    memory_type: str = "preference",
    start_char: int | None = None,
    end_char: int | None = None,
) -> FactMemoryCandidate:
    return FactMemoryCandidate(
        entity_id=entity_id,
        predicate=predicate,
        value=value,
        identity_conditions=identity_conditions or {},
        applicability=applicability or {},
        capability_hints=("vehicle",),
        memory_type=memory_type,
        confidence=confidence,
        evidence=(
            MemoryEvidence(
                message_id=message_id,
                quote=quote,
                start_char=start_char,
                end_char=end_char,
            ),
        ),
        directive=directive,
    )


def test_r3_normalizes_number_boolean_and_enum_aliases(settings) -> None:
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("R3 normalization")
        number = "Patricia prefers fan speed level three."
        compound_number = "Patricia prefers cabin temperature twenty one."
        boolean = "Patricia wants ambient lighting turned on."
        enum = "Patricia prefers outside air for circulation."
        number_id = _message(repository, session.id, number)
        compound_number_id = _message(
            repository,
            session.id,
            compound_number,
        )
        boolean_id = _message(repository, session.id, boolean)
        enum_id = _message(repository, session.id, enum)
        run_id = _run(repository, session.id)

        result = FactMemoryLinker(
            repository,
            user_id="Driver One",
        ).apply_candidates(
            run_id=run_id,
            session_id=session.id,
            candidates=(
                _candidate(number_id, number, "three", predicate="fan speed"),
                _candidate(
                    compound_number_id,
                    compound_number,
                    "twenty one",
                    predicate="cabin temperature",
                ),
                _candidate(
                    boolean_id,
                    boolean,
                    True,
                    predicate="ambient lighting",
                ),
                _candidate(
                    enum_id,
                    enum,
                    "outside_air",
                    predicate="air circulation",
                ),
            ),
        )

        records = {
            record.predicate: record.value
            for record in repository.list_fact_memory_records(session.id)
        }
        events = repository.fact_memory_candidate_events(run_id)
        assert [outcome.decision for outcome in result.outcomes] == [
            "ADD",
            "ADD",
            "ADD",
            "ADD",
        ]
        assert records == {
            "air_circulation": "outside",
            "ambient_lighting": True,
            "cabin_temperature": 21,
            "fan_speed": 3,
        }
        assert {
            event["validation"]["evidence_relation"] for event in events
        } == {"normalized_entailment"}


def test_r3_removes_runtime_binding_before_storage(settings) -> None:
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("R3 runtime binding")
        text = "Gary prefers fan level three in comfort mode."
        message_id = _message(repository, session.id, text)
        run_id = _run(repository, session.id)

        result = FactMemoryLinker(
            repository,
            user_id="Driver One",
        ).apply_candidates(
            run_id=run_id,
            session_id=session.id,
            candidates=(
                _candidate(
                    message_id,
                    text,
                    {"seat": "Gary", "level": "three"},
                    entity_id="Gary",
                    predicate="fan speed",
                    identity_conditions={
                        "person": "Gary",
                        "profile": "comfort",
                    },
                ),
            ),
        )

        record = repository.list_fact_memory_records(session.id)[0]
        event = repository.fact_memory_candidate_events(run_id)[0]
        assert result.outcomes[0].status == "applied"
        assert record.value == {"level": 3}
        assert record.identity_conditions == {"profile": "comfort"}
        assert set(event["validation"]["ignored_runtime_fields"]) == {
            "identity.person",
            "value.seat",
        }


def test_r3_exact_predicate_policy_prevents_cross_capability_update(
    settings,
) -> None:
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Canonical predicates")
        circulation = "Michael prefers outside circulation."
        fan = "Michael prefers fan speed three."
        circulation_id = _message(repository, session.id, circulation)
        fan_id = _message(repository, session.id, fan)
        linker = FactMemoryLinker(
            repository,
            user_id="Driver One",
            require_exact_predicate=True,
        )

        first = linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(
                _candidate(
                    circulation_id,
                    circulation,
                    {"circulation": "outside"},
                    entity_id="Michael Davis",
                    predicate="climate_circulation",
                    identity_conditions={"target": "climate"},
                ),
            ),
        )
        second = linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(
                _candidate(
                    fan_id,
                    fan,
                    {"speed": 3},
                    entity_id="Michael Davis",
                    predicate="climate_fan_speed",
                    identity_conditions={"target": "climate"},
                ),
            ),
        )

        records = {
            record.predicate: record.value
            for record in repository.list_fact_memory_records(session.id)
        }
        assert first.outcomes[0].decision == "ADD"
        assert second.outcomes[0].decision == "ADD"
        assert records == {
            "climate_circulation": {"circulation": "outside"},
            "climate_fan_speed": {"speed": 3},
        }


def test_r3_mixed_predicate_policy_is_exact_only_for_canonical_facts(
    settings,
) -> None:
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Mixed canonical predicates")
        canonical_text = "Michael prefers outside circulation."
        raw_first = "Patricia prefers headrest height four."
        raw_update = "Patricia now prefers headrest height five."
        canonical_id = _message(repository, session.id, canonical_text)
        raw_first_id = _message(repository, session.id, raw_first)
        raw_update_id = _message(repository, session.id, raw_update)
        linker = FactMemoryLinker(
            repository,
            user_id="Driver One",
            exact_predicates=("climate_circulation", "climate_fan_speed"),
        )

        canonical = linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(
                _candidate(
                    canonical_id,
                    canonical_text,
                    {"circulation": "outside"},
                    entity_id="Michael Davis",
                    predicate="climate_circulation",
                    identity_conditions={"target": "climate"},
                ),
            ),
        )
        raw_add = linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(
                _candidate(
                    raw_first_id,
                    raw_first,
                    4,
                    predicate="headrest_height",
                ),
            ),
        )
        raw_change = linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(
                _candidate(
                    raw_update_id,
                    raw_update,
                    5,
                    predicate="preferred_headrest_height",
                ),
            ),
        )

        assert canonical.outcomes[0].decision == "ADD"
        assert raw_add.outcomes[0].decision == "ADD"
        assert raw_change.outcomes[0].decision == "UPDATE"
        active = repository.list_fact_memory_records(session.id)
        assert {record.predicate: record.value for record in active} == {
            "climate_circulation": {"circulation": "outside"},
            "headrest_height": 5,
        }


def test_r3_uncertain_update_is_reviewed_then_explicit_update_applies(
    settings,
) -> None:
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("R3 update intent")
        initial = "Patricia's headrest height is 44."
        uncertain = "Patricia's headrest height is 48."
        explicit = "Patricia now prefers headrest height 48."
        initial_id = _message(repository, session.id, initial)
        uncertain_id = _message(repository, session.id, uncertain)
        explicit_id = _message(repository, session.id, explicit)
        linker = FactMemoryLinker(repository, user_id="Driver One")

        linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(_candidate(initial_id, initial, 44),),
        )
        review_run = _run(repository, session.id)
        review = linker.apply_candidates(
            run_id=review_run,
            session_id=session.id,
            candidates=(_candidate(uncertain_id, uncertain, 48),),
        )
        assert review.outcomes[0].decision == "REVIEW"
        assert (
            repository.fact_memory_candidate_events(review_run)[0]["validation"][
                "code"
            ]
            == "update_intent_uncertain"
        )
        assert repository.list_fact_memory_records(session.id)[0].value == 44

        updated = linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(_candidate(explicit_id, explicit, 48),),
        )
        assert updated.outcomes[0].decision == "UPDATE"
        assert repository.list_fact_memory_records(session.id)[0].value == 48


def test_r3_isolates_strict_evidence_and_privacy_rejections(settings) -> None:
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("R3 strict validation")
        valid = "Patricia prefers headrest height 44."
        invalid_source = "Gary prefers fan speed four."
        pii = "Contact me at alice@example.com."
        outside_batch = "Dana prefers temperature 21."
        assistant = "Sam prefers fan speed five."
        valid_id = _message(repository, session.id, valid)
        invalid_id = _message(repository, session.id, invalid_source)
        pii_id = _message(repository, session.id, pii)
        outside_id = _message(repository, session.id, outside_batch)
        assistant_id = _message(
            repository,
            session.id,
            assistant,
            role="assistant",
        )
        run_id = _run(repository, session.id)

        result = FactMemoryLinker(
            repository,
            user_id="Driver One",
        ).apply_candidates(
            run_id=run_id,
            session_id=session.id,
            candidates=(
                _candidate(valid_id, valid, 44),
                _candidate(
                    invalid_id,
                    "This quote is absent.",
                    4,
                    entity_id="Gary",
                    predicate="fan speed",
                ),
                _candidate(
                    pii_id,
                    pii,
                    "alice@example.com",
                    entity_id="owner",
                    predicate="contact email",
                ),
                _candidate(
                    outside_id,
                    outside_batch,
                    21,
                    entity_id="Dana",
                    predicate="temperature",
                ),
                _candidate(
                    assistant_id,
                    assistant,
                    5,
                    entity_id="Sam",
                    predicate="fan speed",
                ),
            ),
            allowed_evidence_message_ids=(
                valid_id,
                invalid_id,
                pii_id,
                assistant_id,
            ),
        )

        events = repository.fact_memory_candidate_events(run_id)
        assert [outcome.status for outcome in result.outcomes] == [
            "applied",
            "rejected",
            "rejected",
            "rejected",
            "rejected",
        ]
        assert [event["validation"]["code"] for event in events] == [
            "accepted",
            "unsupported_evidence",
            "pii_candidate",
            "out_of_batch_evidence",
            "non_user_evidence",
        ]
        assert len(repository.list_fact_memory_records(session.id)) == 1


def test_r3_repairs_unique_span_and_reviews_low_confidence(settings) -> None:
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("R3 repair and review")
        repair_text = "Patricia prefers headrest height 44."
        review_text = "Gary prefers cabin temperature 21."
        repair_id = _message(repository, session.id, repair_text)
        review_id = _message(repository, session.id, review_text)
        run_id = _run(repository, session.id)

        result = FactMemoryLinker(
            repository,
            user_id="Driver One",
        ).apply_candidates(
            run_id=run_id,
            session_id=session.id,
            candidates=(
                _candidate(
                    repair_id,
                    repair_text,
                    44,
                    start_char=7,
                    end_char=12,
                ),
                _candidate(
                    review_id,
                    review_text,
                    21,
                    entity_id="Gary",
                    predicate="cabin temperature",
                    confidence=0.2,
                ),
            ),
        )

        events = repository.fact_memory_candidate_events(run_id)
        assert [outcome.status for outcome in result.outcomes] == [
            "applied",
            "review",
        ]
        assert events[0]["validation"]["evidence_span_corrections"] == [
            {
                "corrected_end_char": len(repair_text),
                "corrected_start_char": 0,
                "message_id": repair_id,
                "provided_end_char": 12,
                "provided_start_char": 7,
                "reason": "unique_exact_quote",
            }
        ]
        assert events[1]["validation"]["code"] == "confidence_below_minimum"
        metrics = repository.fact_memory_metrics(session.id)
        assert metrics["candidate_status_counts"] == {
            "applied": 1,
            "review": 1,
        }
        assert metrics["validation_code_counts"] == {
            "accepted": 1,
            "confidence_below_minimum": 1,
        }


def test_r3_delete_requires_explicit_invalidation(settings) -> None:
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("R3 delete intent")
        initial = "Patricia prefers headrest height 44."
        uncertain = "Patricia has a headrest preference."
        explicit = "Forget Patricia's headrest preference."
        initial_id = _message(repository, session.id, initial)
        uncertain_id = _message(repository, session.id, uncertain)
        explicit_id = _message(repository, session.id, explicit)
        linker = FactMemoryLinker(repository, user_id="Driver One")

        linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(_candidate(initial_id, initial, 44),),
        )
        review = linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(
                _candidate(
                    uncertain_id,
                    uncertain,
                    None,
                    directive="DELETE",
                ),
            ),
        )
        deleted = linker.apply_candidates(
            run_id=_run(repository, session.id),
            session_id=session.id,
            candidates=(
                _candidate(
                    explicit_id,
                    explicit,
                    None,
                    directive="DELETE",
                ),
            ),
        )

        assert review.outcomes[0].decision == "REVIEW"
        assert deleted.outcomes[0].decision == "DELETE"
        assert repository.list_fact_memory_records(session.id) == []
