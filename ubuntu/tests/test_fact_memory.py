from __future__ import annotations

import pytest

from palmclaw_ubuntu.fact_memory import (
    FactMemoryPatchEngine,
    FactMemoryPatchError,
    parse_fact_memory_patch,
)
from palmclaw_ubuntu.storage import SQLiteRepository


def _message(
    repository: SQLiteRepository,
    session_id: str,
    content: str,
) -> int:
    turn_id = repository.create_turn(session_id)
    message_id = repository.append_message(
        session_id,
        "user",
        content,
        turn_id=turn_id,
    )
    repository.finish_turn(turn_id, "completed", 1, "test")
    return message_id


def _identity(
    *,
    entity_id: str = "Patricia",
    applicability: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "user_id": "Driver One",
        "entity_id": entity_id,
        "predicate": "Headrest Height",
        "identity_conditions": {"occupant_role": "passenger"},
        "applicability": applicability or {},
    }


def _add_patch(
    message_id: int,
    quote: str,
    *,
    entity_id: str = "Patricia",
    value: int = 44,
) -> dict[str, object]:
    return {
        "operation": "ADD",
        "identity": _identity(entity_id=entity_id),
        "value": value,
        "memory_type": "preference",
        "capability_hints": ["Seat Adjustment"],
        "confidence": 0.94,
        "evidence": [{"message_id": message_id, "quote": quote}],
        "reason": "new supported fact",
    }


def test_fact_add_and_update_create_immutable_versions(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Fact versions")
        engine = FactMemoryPatchEngine(repository)
        first_text = "Patricia likes the headrest at 44."
        first_message = _message(repository, session.id, first_text)

        added = engine.apply(
            session_id=session.id,
            patch=_add_patch(first_message, first_text),
            idempotency_key="fact:add:patricia",
        )
        first = repository.fact_memory_record(added.result_record_id)

        assert first is not None
        assert first.entity_id == "patricia"
        assert first.predicate == "headrest_height"
        assert first.value == 44
        assert first.version == 1
        assert first.status == "active"
        assert first.capability_hints == ("seat_adjustment",)
        assert "seat" not in first.identity_conditions
        assert "driver" not in first.identity_conditions

        second_text = "Patricia now wants the headrest at 48 instead."
        second_message = _message(repository, session.id, second_text)
        updated = engine.apply(
            session_id=session.id,
            patch={
                "operation": "UPDATE",
                "identity": _identity(),
                "value": 48,
                "memory_type": "preference",
                "confidence": 0.97,
                "evidence": [
                    {"message_id": second_message, "quote": second_text}
                ],
                "target_record_id": first.id,
                "reason": "explicit preference change",
            },
            idempotency_key="fact:update:patricia:48",
        )
        second = repository.fact_memory_record(updated.result_record_id)
        versions = repository.fact_memory_record_versions(first.record_key)

        assert second is not None
        assert second.value == 48
        assert second.version == 2
        assert second.supersedes_id == first.id
        assert second.capability_hints == ("seat_adjustment",)
        assert [(item.version, item.status) for item in versions] == [
            (1, "superseded"),
            (2, "active"),
        ]
        assert repository.fact_memory_record(first.id).value == 44
        assert repository.fact_memory_record_sources(second.id)[0].message_id == (
            second_message
        )

        replay = engine.apply(
            session_id=session.id,
            patch={
                "operation": "UPDATE",
                "identity": _identity(),
                "value": 48,
                "memory_type": "preference",
                "confidence": 0.97,
                "evidence": [
                    {"message_id": second_message, "quote": second_text}
                ],
                "target_record_id": first.id,
                "reason": "explicit preference change",
            },
            idempotency_key="fact:update:patricia:48",
        )
        assert replay.replayed is True
        assert replay.result_record_id == second.id
        assert len(repository.fact_memory_record_versions(first.record_key)) == 2


def test_fact_identity_separates_applicability_but_not_value(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Conditional facts")
        hot_text = "When it is hot, Patricia wants the headrest at 44."
        cold_text = "When it is cold, Patricia wants the headrest at 40."
        hot_message = _message(repository, session.id, hot_text)
        cold_message = _message(repository, session.id, cold_text)
        engine = FactMemoryPatchEngine(repository)

        hot_patch = _add_patch(hot_message, hot_text, value=44)
        hot_patch["identity"] = _identity(applicability={"weather": "Hot"})
        cold_patch = _add_patch(cold_message, cold_text, value=40)
        cold_patch["identity"] = _identity(applicability={"weather": "Cold"})
        hot = engine.apply(
            session_id=session.id,
            patch=hot_patch,
            idempotency_key="fact:hot",
        )
        cold = engine.apply(
            session_id=session.id,
            patch=cold_patch,
            idempotency_key="fact:cold",
        )
        records = repository.list_fact_memory_records(session.id)

        assert len(records) == 2
        assert hot.result_record_id != cold.result_record_id
        assert len({record.record_key for record in records}) == 2
        assert {record.applicability["weather"] for record in records} == {
            "hot",
            "cold",
        }


def test_fact_merge_and_delete_preserve_lineage(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Fact merge delete")
        short_text = "Patricia likes the headrest at 44."
        full_text = "Patricia Garcia also prefers headrest position 44."
        merge_text = "Patricia and Patricia Garcia refer to the same person."
        delete_text = "Forget Patricia's saved headrest preference."
        short_message = _message(repository, session.id, short_text)
        full_message = _message(repository, session.id, full_text)
        merge_message = _message(repository, session.id, merge_text)
        delete_message = _message(repository, session.id, delete_text)
        engine = FactMemoryPatchEngine(repository)

        short = engine.apply(
            session_id=session.id,
            patch=_add_patch(short_message, short_text),
            idempotency_key="fact:short",
        )
        full = engine.apply(
            session_id=session.id,
            patch=_add_patch(
                full_message,
                full_text,
                entity_id="Patricia Garcia",
            ),
            idempotency_key="fact:full",
        )
        merged = engine.apply(
            session_id=session.id,
            patch={
                "operation": "MERGE",
                "identity": _identity(entity_id="Patricia Garcia"),
                "value": 44,
                "memory_type": "preference",
                "capability_hints": ["seat"],
                "confidence": 0.99,
                "evidence": [
                    {"message_id": merge_message, "quote": merge_text}
                ],
                "target_record_id": full.result_record_id,
                "merge_record_ids": [short.result_record_id],
            },
            idempotency_key="fact:merge",
        )
        merged_record = repository.fact_memory_record(merged.result_record_id)

        assert merged_record is not None
        assert merged_record.status == "active"
        assert merged_record.version == 2
        assert len(repository.fact_memory_record_sources(merged_record.id)) == 3
        assert (
            repository.fact_memory_record(short.result_record_id).merged_into_id
            == merged_record.id
        )
        assert (
            repository.fact_memory_record(full.result_record_id).merged_into_id
            == merged_record.id
        )

        deleted = engine.apply(
            session_id=session.id,
            patch={
                "operation": "DELETE",
                "confidence": 0.99,
                "evidence": [
                    {"message_id": delete_message, "quote": delete_text}
                ],
                "target_record_id": merged_record.id,
                "reason": "explicit deletion",
            },
            idempotency_key="fact:delete",
        )
        tombstone = repository.fact_memory_record(deleted.result_record_id)
        versions = repository.fact_memory_record_versions(merged_record.record_key)

        assert tombstone is not None
        assert tombstone.status == "deleted"
        assert tombstone.version == 3
        assert tombstone.supersedes_id == merged_record.id
        assert repository.list_fact_memory_records(session.id) == []
        assert [(item.version, item.status) for item in versions] == [
            (1, "merged"),
            (2, "superseded"),
            (3, "deleted"),
        ]


def test_fact_patch_rejects_bad_evidence_and_identity_change(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Fact validation")
        other = repository.create_session("Other")
        text = "Patricia likes the headrest at 44."
        message = _message(repository, session.id, text)
        other_message = _message(repository, other.id, text)
        engine = FactMemoryPatchEngine(repository)
        added = engine.apply(
            session_id=session.id,
            patch=_add_patch(message, text),
            idempotency_key="fact:valid",
        )

        with pytest.raises(FactMemoryPatchError, match="not in this session"):
            engine.apply(
                session_id=session.id,
                patch=_add_patch(other_message, text, entity_id="Alex"),
                idempotency_key="fact:cross-session",
            )

        with pytest.raises(FactMemoryPatchError, match="preserve"):
            engine.apply(
                session_id=session.id,
                patch={
                    "operation": "UPDATE",
                    "identity": _identity(entity_id="Someone Else"),
                    "value": 48,
                    "memory_type": "preference",
                    "confidence": 0.9,
                    "evidence": [{"message_id": message, "quote": text}],
                    "target_record_id": added.result_record_id,
                },
                idempotency_key="fact:identity-change",
            )

        assert len(repository.list_fact_memory_records(session.id)) == 1


def test_fact_patch_schema_rejects_tool_bound_identity():
    with pytest.raises(FactMemoryPatchError, match="Additional properties"):
        parse_fact_memory_patch(
            {
                "operation": "ADD",
                "identity": {
                    **_identity(),
                    "tool_domain": "seat",
                },
                "value": 44,
                "memory_type": "preference",
                "confidence": 0.9,
                "evidence": [{"message_id": 1, "quote": "headrest at 44"}],
            }
        )
