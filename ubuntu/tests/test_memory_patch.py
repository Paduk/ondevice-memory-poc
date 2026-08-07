from __future__ import annotations

from typing import Any

import pytest

from palmclaw_ubuntu.memory_patch import (
    ToolMemoryPatchEngine,
    ToolMemoryPatchSchemaError,
    parse_tool_memory_patch,
)
from palmclaw_ubuntu.models import ToolMemoryIdentity
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tool_memory_schema import tool_memory_record_key
from palmclaw_ubuntu.validation import ToolMemoryPatchValidationError


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


def _run(
    repository: SQLiteRepository,
    session_id: str,
    turn_id: str,
) -> str:
    return repository.begin_memory_patch_run(
        session_id=session_id,
        source_turn_id=turn_id,
        backend="test",
        model_id="patch-model",
        prompt_version="patch-v1",
        schema_version="tool-memory-patch-v1",
    )


def _identity(
    *,
    conditions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "user_id": "Driver One",
        "tool_domain": "Seat",
        "topic": "Ventilation Speed",
        "scope": "conditional",
        "scope_key": "Driver One",
        "conditions": conditions or {"weather": "hot"},
    }


def _add_patch(message_id: int, quote: str, value: int = 2) -> dict[str, Any]:
    return {
        "operation": "ADD",
        "identity": _identity(),
        "value": value,
        "memory_type": "preference",
        "confidence": 0.96,
        "evidence": [{"message_id": message_id, "quote": quote}],
        "reason": "new supported preference",
    }


def _add_record(
    repository: SQLiteRepository,
    engine: ToolMemoryPatchEngine,
    session_id: str,
    *,
    conditions: dict[str, Any] | None = None,
    content: str = "On hot days, set seat ventilation to level 2.",
) -> tuple[str, int, str]:
    turn_id, message_id = _turn(repository, session_id, content)
    patch = _add_patch(message_id, content)
    patch["identity"] = _identity(conditions=conditions)
    run_id = _run(repository, session_id, turn_id)
    result = engine.apply_batch(run_id, [patch])
    return result.outcomes[0].result_record_id, message_id, run_id


def test_patch_json_schema_rejects_unknown_fields():
    payload = _add_patch(1, "Set level 2")
    payload["unexpected"] = True

    with pytest.raises(ToolMemoryPatchSchemaError, match="Additional properties"):
        parse_tool_memory_patch(payload)

    payload.pop("unexpected")
    payload["operation"] = "UPSERT"
    with pytest.raises(ToolMemoryPatchSchemaError, match="is not one of"):
        parse_tool_memory_patch(payload)


def test_add_is_audited_and_completed_run_replays_idempotently(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Patch ADD")
        content = "On hot days, set seat ventilation to level 2."
        turn_id, message_id = _turn(repository, session.id, content)
        run_id = _run(repository, session.id, turn_id)
        engine = ToolMemoryPatchEngine(repository)
        patch = _add_patch(message_id, content)

        first = engine.apply_batch(run_id, [patch])
        replay = engine.apply_batch(run_id, [patch])
        record = repository.tool_memory_record(
            first.outcomes[0].result_record_id
        )
        detail = repository.tool_memory_record_detail(record.id)
        trace = repository.memory_patch_trace(run_id)

        assert replay.replayed is True
        assert replay.outcomes == first.outcomes
        assert len(repository.list_tool_memory_records(session.id)) == 1
        assert record.status == "active"
        assert record.version == 1
        assert detail["sources"][0].start_char == 0
        assert detail["sources"][0].end_char == len(content)
        assert [event["status"] for event in detail["status_events"]] == [
            "active"
        ]
        assert trace["run"]["status"] == "completed"
        assert trace["proposals"][0]["status"] == "applied"
        assert trace["proposals"][0]["validation"]["accepted"] is True


def test_unique_exact_quote_repairs_incorrect_evidence_offsets(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Patch evidence repair")
        content = (
            "[2025-03-15 09:00] Gary Allen: "
            "Set seat ventilation to level 2."
        )
        quote = "Set seat ventilation to level 2."
        turn_id, message_id = _turn(repository, session.id, content)
        run_id = _run(repository, session.id, turn_id)
        engine = ToolMemoryPatchEngine(repository)
        patch = _add_patch(message_id, quote)
        patch["evidence"][0].update(
            {
                "start_char": 4,
                "end_char": 12,
            }
        )

        outcome = engine.apply_batch(run_id, [patch]).outcomes[0]
        detail = repository.tool_memory_record_detail(
            outcome.result_record_id
        )
        trace = repository.memory_patch_trace(run_id)

    expected_start = content.index(quote)
    source = detail["sources"][0]
    correction = trace["proposals"][0]["validation"][
        "evidence_span_corrections"
    ][0]
    assert source.start_char == expected_start
    assert source.end_char == expected_start + len(quote)
    assert correction == {
        "message_id": message_id,
        "provided_start_char": 4,
        "provided_end_char": 12,
        "corrected_start_char": expected_start,
        "corrected_end_char": expected_start + len(quote),
        "reason": "unique_exact_quote",
    }


def test_patch_identity_reuses_unambiguous_learned_entity_alias(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Patch aliases")
        repository.insert_tool_memory_record(
            session_id=session.id,
            identity=ToolMemoryIdentity(
                user_id="Driver One",
                tool_domain="Seat",
                topic="heating_level",
                scope="conditional",
                scope_key="Driver One",
                conditions={"person": "Patricia Garcia"},
            ),
            value=1,
            memory_type="preference",
            confidence=0.96,
            version=1,
        )
        content = "Patricia prefers ventilation level 2."
        turn_id, message_id = _turn(repository, session.id, content)
        run_id = _run(repository, session.id, turn_id)
        engine = ToolMemoryPatchEngine(repository)
        patch = _add_patch(message_id, content)
        patch["identity"] = _identity(conditions={"person": "Patricia"})

        result = engine.apply_batch(run_id, [patch])
        record = repository.tool_memory_record(
            result.outcomes[0].result_record_id
        )

    assert record.conditions == {"person": "patricia garcia"}
    assert record.entity_id == "patricia_garcia"
    assert record.identity_conditions == {"person": "patricia garcia"}
    assert record.applicability == {}
    assert record.identity_family_key


def test_update_supersedes_old_record_and_increments_version(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Patch UPDATE")
        engine = ToolMemoryPatchEngine(repository)
        old_id, _, _ = _add_record(repository, engine, session.id)
        content = (
            "I changed it: on hot days, set seat ventilation to level 3 instead."
        )
        turn_id, message_id = _turn(repository, session.id, content)
        run_id = _run(repository, session.id, turn_id)
        update = {
            "operation": "UPDATE",
            "target_record_id": old_id,
            "identity": _identity(),
            "value": 3,
            "memory_type": "preference",
            "confidence": 0.97,
            "evidence": [{"message_id": message_id, "quote": content}],
            "reason": "explicit user change",
        }

        outcome = engine.apply_batch(run_id, [update]).outcomes[0]
        old = repository.tool_memory_record(old_id)
        new = repository.tool_memory_record(outcome.result_record_id)

        assert old.status == "superseded"
        assert new.status == "active"
        assert new.value == 3
        assert new.version == 2
        assert new.supersedes_id == old.id
        assert (
            repository.active_tool_memory_record(new.record_key).id == new.id
        )


def test_update_requires_valid_target_and_explicit_change_evidence(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Patch UPDATE validation")
        engine = ToolMemoryPatchEngine(repository)
        old_id, _, _ = _add_record(repository, engine, session.id)
        content = "On hot days, set seat ventilation to level 3."
        turn_id, message_id = _turn(repository, session.id, content)
        run_id = _run(repository, session.id, turn_id)
        update = {
            "operation": "UPDATE",
            "target_record_id": old_id,
            "identity": _identity(),
            "value": 3,
            "memory_type": "preference",
            "confidence": 0.97,
            "evidence": [{"message_id": message_id, "quote": content}],
        }

        with pytest.raises(
            ToolMemoryPatchValidationError,
            match="explicit change language",
        ):
            engine.apply_batch(run_id, [update])

        assert repository.tool_memory_record(old_id).status == "active"
        assert repository.memory_patch_trace(run_id)["run"]["status"] == "failed"

        missing_target_run = _run(repository, session.id, turn_id)
        update["target_record_id"] = "missing-record"
        with pytest.raises(
            ToolMemoryPatchValidationError,
            match="Unknown target record",
        ):
            engine.apply_batch(missing_target_run, [update])
        assert repository.list_tool_memory_records(session.id) == [
            repository.tool_memory_record(old_id)
        ]


def test_delete_keeps_tombstone_and_evidence(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Patch DELETE")
        engine = ToolMemoryPatchEngine(repository)
        record_id, _, _ = _add_record(repository, engine, session.id)
        content = "Forget my hot-day seat ventilation setting."
        turn_id, message_id = _turn(repository, session.id, content)
        run_id = _run(repository, session.id, turn_id)
        delete = {
            "operation": "DELETE",
            "target_record_id": record_id,
            "confidence": 0.99,
            "evidence": [{"message_id": message_id, "quote": content}],
            "reason": "explicit user deletion",
        }

        outcome = engine.apply_batch(run_id, [delete]).outcomes[0]
        record = repository.tool_memory_record(record_id)
        detail = repository.tool_memory_record_detail(record_id)

        assert outcome.result_record_id == record_id
        assert record.status == "deleted"
        assert repository.active_tool_memory_record(record.record_key) is None
        assert len(detail["sources"]) == 1
        assert [event["status"] for event in detail["status_events"]] == [
            "active",
            "deleted",
        ]


def test_merge_replaces_duplicate_records_and_combines_evidence(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Patch MERGE")
        engine = ToolMemoryPatchEngine(repository)
        hot_id, _, _ = _add_record(repository, engine, session.id)
        warm_id, _, _ = _add_record(
            repository,
            engine,
            session.id,
            conditions={"weather": "warm"},
            content="On warm days, set seat ventilation to level 2.",
        )
        content = (
            "Merge my hot and warm weather seat ventilation preferences at level 2."
        )
        turn_id, message_id = _turn(repository, session.id, content)
        run_id = _run(repository, session.id, turn_id)
        merge = {
            "operation": "MERGE",
            "target_record_id": hot_id,
            "merge_record_ids": [warm_id],
            "identity": _identity(conditions={"weather": ["hot", "warm"]}),
            "value": 2,
            "memory_type": "preference",
            "confidence": 0.98,
            "evidence": [{"message_id": message_id, "quote": content}],
            "reason": "consolidate equivalent preferences",
        }

        outcome = engine.apply_batch(run_id, [merge]).outcomes[0]
        result = repository.tool_memory_record(outcome.result_record_id)
        hot = repository.tool_memory_record(hot_id)
        warm = repository.tool_memory_record(warm_id)
        sources = repository.tool_memory_record_sources(result.id)

        assert result.status == "active"
        assert result.version == 2
        assert hot.status == warm.status == "merged"
        assert hot.merged_into_id == warm.merged_into_id == result.id
        assert {item.message_id for item in sources} == {
            message_id,
            *(
                item.message_id
                for record_id in (hot_id, warm_id)
                for item in repository.tool_memory_record_sources(record_id)
            ),
        }


def test_batch_failure_rolls_back_prior_patch_and_records_rejection(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Patch rollback")
        content = "Set seat ventilation to level 2, then set it to level 3."
        turn_id, message_id = _turn(repository, session.id, content)
        run_id = _run(repository, session.id, turn_id)
        engine = ToolMemoryPatchEngine(repository)
        first = _add_patch(message_id, content, value=2)
        second = _add_patch(message_id, content, value=3)

        with pytest.raises(
            ToolMemoryPatchValidationError,
            match="ADD cannot replace",
        ):
            engine.apply_batch(run_id, [first, second])

        trace = repository.memory_patch_trace(run_id)
        identity = parse_tool_memory_patch(first).identity
        assert repository.list_tool_memory_records(session.id) == []
        assert repository.active_tool_memory_record(
            tool_memory_record_key(identity)
        ) is None
        assert trace["run"]["status"] == "failed"
        assert len(trace["proposals"]) == 1
        assert trace["proposals"][0]["idempotency_key"] == f"{run_id}:1"
        assert trace["proposals"][0]["status"] == "rejected"
        assert trace["proposals"][0]["validation"]["code"] == "add_contradiction"


def test_isolated_batch_keeps_valid_patch_and_rejects_only_invalid_patch(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Patch isolation")
        content = "Set seat ventilation to level 2, then set it to level 3."
        turn_id, message_id = _turn(repository, session.id, content)
        run_id = _run(repository, session.id, turn_id)
        engine = ToolMemoryPatchEngine(repository)
        first = _add_patch(message_id, content, value=2)
        second = _add_patch(message_id, content, value=3)

        result = engine.apply_batch(
            run_id,
            [first, second],
            isolate_invalid_patches=True,
        )
        trace = repository.memory_patch_trace(run_id)
        records = repository.list_tool_memory_records(session.id)

    assert trace["run"]["status"] == "completed"
    assert len(result.outcomes) == 1
    assert len(result.rejections) == 1
    assert result.rejections[0].sequence_index == 1
    assert result.rejections[0].code == "add_contradiction"
    assert len(records) == 1
    assert records[0].value == 2
    assert [proposal["status"] for proposal in trace["proposals"]] == [
        "applied",
        "rejected",
    ]


def test_unsupported_evidence_and_pii_are_rejected_without_memory(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Patch validation")
        engine = ToolMemoryPatchEngine(repository)
        content = "Set level 2 and contact driver@example.com."
        turn_id, message_id = _turn(repository, session.id, content)
        run_id = _run(repository, session.id, turn_id)
        patch = _add_patch(message_id, content)

        with pytest.raises(
            ToolMemoryPatchValidationError,
            match="contains PII",
        ):
            engine.apply_batch(run_id, [patch])

        trace = repository.memory_patch_trace(run_id)
        proposal = trace["proposals"][0]
        assert repository.list_tool_memory_records(session.id) == []
        assert proposal["validation"]["code"] == "pii_candidate"
        assert "driver@example.com" not in str(proposal)
        assert "[REDACTED_EMAIL]" in str(proposal)

        other_turn, other_message = _turn(
            repository,
            session.id,
            "Set seat ventilation to level 4.",
        )
        other_run = _run(repository, session.id, other_turn)
        unsupported = _add_patch(
            other_message,
            "This quote is absent from the source message.",
            value=4,
        )
        with pytest.raises(
            ToolMemoryPatchValidationError,
            match="does not occur",
        ):
            engine.apply_batch(other_run, [unsupported])
        assert repository.list_tool_memory_records(session.id) == []
