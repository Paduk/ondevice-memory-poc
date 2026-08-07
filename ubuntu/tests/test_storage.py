from __future__ import annotations

import sqlite3
from importlib import resources

from palmclaw_ubuntu.models import ModelUsage, ToolCall
from palmclaw_ubuntu.storage import SQLiteRepository


def test_session_messages_and_active_session_survive_restart(settings):
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    session = repository.create_session("Research")
    repository.set_active_session(session.id)
    turn_id = repository.create_turn(session.id)
    repository.append_message(
        session.id,
        "user",
        "hello",
        turn_id=turn_id,
    )
    repository.finish_turn(
        turn_id,
        "completed",
        1,
        "assistant_response",
    )
    repository.close()

    reopened = SQLiteRepository(settings.database_path)
    assert reopened.get_active_session_id() == session.id
    assert reopened.require_session(session.id).title == "Research"
    assert [item.content for item in reopened.list_messages(session.id)] == ["hello"]
    assert reopened.list_turns(session.id)[0]["status"] == "completed"
    reopened.close()


def test_storage_redacts_secrets(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Secret test")
        repository.append_message(
            session.id,
            "user",
            "api_key=sk-abcdefghijklmnop1234",
        )
        content = repository.list_messages(session.id)[0].content
    assert "sk-" not in content
    assert "[REDACTED]" in content


def test_migrates_001_database_through_runtime_safety(settings):
    settings.ensure_directories()
    connection = sqlite3.connect(settings.database_path)
    connection.execute(
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    initial_sql = (
        resources.files("palmclaw_ubuntu.migrations")
        .joinpath("001_initial.sql")
        .read_text(encoding="utf-8")
    )
    connection.executescript(initial_sql)
    connection.execute(
        """
        INSERT INTO schema_migrations(version, name, applied_at)
        VALUES (1, '001_initial.sql', '2026-07-24T00:00:00+00:00')
        """
    )
    connection.commit()
    connection.close()

    with SQLiteRepository(settings.database_path) as repository:
        message_columns = {
            row["name"]
            for row in repository._connection.execute("PRAGMA table_info(messages)")
        }
        tool_columns = {
            row["name"]
            for row in repository._connection.execute("PRAGMA table_info(tool_calls)")
        }
    assert "provider_items_json" in message_columns
    assert {"side_effect", "retry_safety", "fingerprint"} <= tool_columns
    with SQLiteRepository(settings.database_path) as repository:
        tables = {
            row["name"]
            for row in repository._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        memory_columns = {
            row["name"]
            for row in repository._connection.execute("PRAGMA table_info(memories)")
        }
        patch_columns = {
            row["name"]
            for row in repository._connection.execute(
                "PRAGMA table_info(memory_patch_proposals)"
            )
        }
        patch_run_columns = {
            row["name"]
            for row in repository._connection.execute(
                "PRAGMA table_info(memory_patch_runs)"
            )
        }
        model_call_columns = {
            row["name"]
            for row in repository._connection.execute(
                "PRAGMA table_info(model_calls)"
            )
        }
        patch_job_columns = {
            row["name"]
            for row in repository._connection.execute(
                "PRAGMA table_info(memory_patch_jobs)"
            )
        }
        tool_memory_columns = {
            row["name"]
            for row in repository._connection.execute(
                "PRAGMA table_info(tool_memory_records)"
            )
        }
    assert {
        "memory_sources",
        "memory_status_events",
        "memory_embeddings",
        "retrieval_runs",
        "retrieval_candidates",
        "tool_memory_records",
        "tool_memory_record_sources",
        "memory_patch_runs",
        "memory_patch_proposals",
        "memory_patch_jobs",
        "tool_memory_record_status_events",
        "tool_memory_embeddings",
        "tool_memory_retrieval_runs",
        "tool_memory_retrieval_candidates",
        "amem_notes",
        "amem_note_versions",
        "amem_note_embeddings",
        "amem_links",
        "amem_construction_runs",
        "amem_evolution_events",
        "amem_retrieval_runs",
        "amem_retrieval_candidates",
    } <= tables
    assert {
        "fact_key",
        "supersedes_id",
        "confidence",
        "rejection_reason",
    } <= memory_columns
    assert {"sequence_index", "validation_json"} <= patch_columns
    assert "job_id" in patch_run_columns
    assert "batch_run_id" in patch_job_columns
    assert {
        "entity_id",
        "identity_conditions_json",
        "applicability_json",
        "identity_family_key",
    } <= tool_memory_columns
    assert "memory_patch_run_id" in model_call_columns


def test_tool_and_model_trace_redacts_content_and_sensitive_metadata(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Trace redaction")
        turn_id = repository.create_turn(session.id)
        call = ToolCall(
            id="secret-call",
            name="file_write",
            arguments={
                "path": "secret.txt",
                "content": "password=hunter2",
                "api_key": "plain-secret-value",
            },
        )
        repository.record_tool_call(
            turn_id,
            1,
            call,
            side_effect="workspace_write",
            retry_safety="unsafe",
            fingerprint="safe-fingerprint",
        )
        repository.mark_tool_call_executing(call.id)
        repository.record_tool_result(
            call.id,
            content="token=abcdefghijklmnop",
            is_error=False,
            metadata={
                "authorization": "plain-authorization-value",
                "safe": "visible",
            },
            duration_ms=1,
        )
        repository.record_model_call(
            session_id=session.id,
            turn_id=turn_id,
            role="agent",
            backend="fake",
            model_id="fake",
            prompt_version="v1",
            schema_version=None,
            latency_ms=1,
            usage=ModelUsage(),
            response_id=None,
            metadata={"cookie": "plain-cookie-value"},
        )
        trace = repository.get_trace(turn_id)

    serialized = str(trace)
    assert "hunter2" not in serialized
    assert "plain-secret-value" not in serialized
    assert "abcdefghijklmnop" not in serialized
    assert "plain-authorization-value" not in serialized
    assert "plain-cookie-value" not in serialized
    assert trace["tool_results"][0]["metadata"]["safe"] == "visible"
