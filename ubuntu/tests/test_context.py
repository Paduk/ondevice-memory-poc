from __future__ import annotations

import pytest

from palmclaw_ubuntu.context import ContextBudgetError, ContextBuilder
from palmclaw_ubuntu.models import ToolCall
from palmclaw_ubuntu.skills import SkillsLoader
from palmclaw_ubuntu.storage import SQLiteRepository


def _builder(
    settings,
    *,
    max_history_messages: int = 40,
    max_context_tokens: int = 1_000,
    memory_token_budget: int = 100,
    skill_token_budget: int = 100,
) -> ContextBuilder:
    return ContextBuilder(
        SkillsLoader(
            settings.builtin_skills_root,
            settings.workspace_skills_root,
        ),
        max_history_messages=max_history_messages,
        max_context_tokens=max_context_tokens,
        memory_token_budget=memory_token_budget,
        skill_token_budget=skill_token_budget,
    )


def test_incomplete_and_orphan_tool_messages_are_excluded(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Context")
        turn_id = repository.create_turn(session.id)
        repository.append_message(
            session.id,
            "tool",
            "orphan",
            turn_id=turn_id,
            tool_call_id="missing",
        )
        repository.append_message(
            session.id,
            "user",
            "current request",
            turn_id=turn_id,
        )
        repository.append_message(
            session.id,
            "assistant",
            "",
            turn_id=turn_id,
            tool_calls=(
                ToolCall(
                    id="unfinished",
                    name="file_read",
                    arguments={"path": "missing.txt"},
                ),
            ),
        )
        built = _builder(settings).build(
            session_id=session.id,
            messages=repository.list_messages(session.id),
            memory="",
        )

    assert [message.role for message in built.messages] == ["system", "user"]
    assert built.metadata["dropped_incomplete_tool_messages"] == 2


def test_history_selects_complete_turns_atomically(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Atomic context")
        old_turn = repository.create_turn(session.id)
        repository.append_message(
            session.id,
            "user",
            "old request",
            turn_id=old_turn,
        )
        repository.append_message(
            session.id,
            "assistant",
            "",
            turn_id=old_turn,
            tool_calls=(
                ToolCall(
                    id="old-call",
                    name="file_read",
                    arguments={"path": "old.txt"},
                ),
            ),
        )
        repository.append_message(
            session.id,
            "tool",
            "old result",
            turn_id=old_turn,
            tool_call_id="old-call",
        )
        repository.append_message(
            session.id,
            "assistant",
            "old final",
            turn_id=old_turn,
        )
        repository.finish_turn(
            old_turn,
            "completed",
            2,
            "assistant_response",
        )
        current_turn = repository.create_turn(session.id)
        repository.append_message(
            session.id,
            "user",
            "current request",
            turn_id=current_turn,
        )
        built = _builder(
            settings,
            max_history_messages=4,
        ).build(
            session_id=session.id,
            messages=repository.list_messages(session.id),
            memory="",
        )

    assert [message.content for message in built.messages[1:]] == ["current request"]
    assert not any(message.tool_calls for message in built.messages)


def test_component_and_total_token_budgets_are_enforced(settings):
    settings.ensure_directories()
    skill_file = settings.workspace_skills_root / "large" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: large\ndescription: large\nalways: true\n---\n"
        + ("skill-data " * 1_000),
        encoding="utf-8",
    )
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Budget")
        for index in range(3):
            turn_id = repository.create_turn(session.id)
            repository.append_message(
                session.id,
                "user",
                f"old-{index} " + ("history " * 1_000),
                turn_id=turn_id,
            )
            repository.append_message(
                session.id,
                "assistant",
                f"answer-{index} " + ("history " * 1_000),
                turn_id=turn_id,
            )
            repository.finish_turn(
                turn_id,
                "completed",
                1,
                "assistant_response",
            )
        current_turn = repository.create_turn(session.id)
        repository.append_message(
            session.id,
            "user",
            "current request",
            turn_id=current_turn,
        )
        built = _builder(
            settings,
            max_context_tokens=500,
            memory_token_budget=20,
            skill_token_budget=20,
        ).build(
            session_id=session.id,
            messages=repository.list_messages(session.id),
            memory="memory-data " * 1_000,
        )

    assert built.metadata["total_context_tokens"] <= 500
    assert built.metadata["memory_tokens"] <= 20
    assert built.metadata["skill_tokens"] <= 20
    assert built.messages[-1].content == "current request"
    assert all("old-" not in message.content for message in built.messages)


def test_current_turn_over_budget_fails_explicitly(settings):
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path) as repository:
        session = repository.create_session("Oversized")
        turn_id = repository.create_turn(session.id)
        repository.append_message(
            session.id,
            "user",
            "oversized " * 5_000,
            turn_id=turn_id,
        )
        messages = repository.list_messages(session.id)
        with pytest.raises(ContextBudgetError, match="current turn"):
            _builder(
                settings,
                max_context_tokens=300,
                memory_token_budget=0,
                skill_token_budget=0,
            ).build(
                session_id=session.id,
                messages=messages,
                memory="",
            )
