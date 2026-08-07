from __future__ import annotations

from dataclasses import replace

from palmclaw_ubuntu.application import create_runtime
from palmclaw_ubuntu.models import AgentResponse, ToolCall
from palmclaw_ubuntu.providers import FakeMemoryModel, ScriptedAgentModel
from palmclaw_ubuntu.storage import SQLiteRepository


def test_agent_tool_memory_and_restart_end_to_end(settings):
    settings = replace(settings, memory_strategy="summary")
    agent_model = ScriptedAgentModel(
        [
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="call-write",
                        name="file_write",
                        arguments={
                            "path": "notes/hello.txt",
                            "content": "안녕하세요",
                            "overwrite": False,
                        },
                    ),
                ),
            ),
            AgentResponse(content="파일을 작성했습니다."),
        ]
    )
    memory_model = FakeMemoryModel()
    with create_runtime(
        settings,
        agent_model=agent_model,
        memory_model=memory_model,
    ) as runtime:
        session = runtime.repository.create_session("E2E")
        runtime.repository.set_active_session(session.id)
        result = runtime.agent_loop.run(
            session.id,
            "hello.txt에 인사말을 작성해줘",
        )
        trace = runtime.repository.get_trace(result.turn_id)
        messages = runtime.repository.list_messages(session.id)
        memory = runtime.repository.latest_memory(session.id)

    assert result.status == "completed"
    assert result.rounds == 2
    assert result.consolidation_run_id is not None
    assert trace["tool_calls"][0]["status"] == "succeeded"
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2].tool_call_id == "call-write"
    assert "hello.txt에 인사말" in memory
    assert (settings.workspace_root / session.id / "notes" / "hello.txt").read_text(
        encoding="utf-8"
    ) == "안녕하세요"
    second_request_messages = agent_model.requests[1][0]
    assert any(
        message.role == "tool" and message.tool_call_id == "call-write"
        for message in second_request_messages
    )

    with SQLiteRepository(settings.database_path) as reopened:
        assert reopened.get_active_session_id() == session.id
        assert len(reopened.list_messages(session.id)) == 4
        assert reopened.latest_memory(session.id) == memory
