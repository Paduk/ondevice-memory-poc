from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from palmclaw_ubuntu.agent import RunResult
from palmclaw_ubuntu.application import create_runtime
from palmclaw_ubuntu.coordinator import SessionTurnCoordinator
from palmclaw_ubuntu.models import AgentResponse, ToolCall
from palmclaw_ubuntu.providers import FakeMemoryModel, ScriptedAgentModel
from palmclaw_ubuntu.storage import SQLiteRepository


class _SlowAgentModel:
    backend = "fake"
    model_id = "slow-agent"
    prompt_version = "agent-v1"

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def complete(self, messages, tools):
        del messages, tools
        self.started.set()
        self.release.wait(timeout=5)
        return AgentResponse(content="late")


class _BarrierEchoAgentModel:
    backend = "fake"
    model_id = "barrier-agent"
    prompt_version = "agent-v1"

    def __init__(self):
        self.barrier = threading.Barrier(2)

    def complete(self, messages, tools):
        del tools
        latest_user = next(
            message.content for message in reversed(messages) if message.role == "user"
        )
        self.barrier.wait(timeout=2)
        return AgentResponse(content=f"reply:{latest_user}")


class _CancelledAgentModel:
    backend = "fake"
    model_id = "cancelled-agent"
    prompt_version = "agent-v1"

    def complete(self, messages, tools):
        del messages, tools
        raise KeyboardInterrupt


class _GateLoop:
    def __init__(self, target_active: int):
        self.target_active = target_active
        self.ready = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.entered = 0
        self.active = 0
        self.max_active = 0

    def run(self, session_id: str, user_text: str) -> RunResult:
        del user_text
        with self.lock:
            self.entered += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= self.target_active:
                self.ready.set()
        self.release.wait(timeout=3)
        with self.lock:
            self.active -= 1
        return RunResult(
            turn_id=session_id,
            content="done",
            rounds=1,
            status="completed",
            terminal_reason="assistant_response",
        )


def test_restart_marks_inflight_work_interrupted(settings):
    settings.ensure_directories()
    repository = SQLiteRepository(settings.database_path)
    session = repository.create_session("Recovery")
    turn_id = repository.create_turn(session.id)
    message_id = repository.append_message(
        session.id,
        "user",
        "unfinished",
        turn_id=turn_id,
    )
    call = ToolCall(
        id="inflight-call",
        name="file_write",
        arguments={
            "path": "result.txt",
            "content": "partial",
            "overwrite": False,
        },
    )
    repository.record_tool_call(
        turn_id,
        1,
        call,
        side_effect="workspace_write",
        retry_safety="unsafe",
        fingerprint="fingerprint",
    )
    repository.mark_tool_call_executing(call.id)
    completed_call = ToolCall(
        id="completed-call",
        name="file_write",
        arguments={
            "path": "completed.txt",
            "content": "completed",
            "overwrite": False,
        },
    )
    repository.record_tool_call(
        turn_id,
        1,
        completed_call,
        side_effect="workspace_write",
        retry_safety="unsafe",
        fingerprint="completed-fingerprint",
    )
    repository.mark_tool_call_executing(completed_call.id)
    repository.record_tool_result(
        completed_call.id,
        content='{"written":true}',
        is_error=False,
        metadata={},
        duration_ms=1,
    )
    consolidation_id = repository.begin_consolidation(
        session.id,
        message_id,
        message_id,
        backend="fake",
        model_id="fake-memory",
        prompt_version="memory-v1",
        schema_version="summary-v1",
    )
    repository.close()

    with create_runtime(settings) as runtime:
        trace = runtime.repository.get_trace(turn_id)
        consolidation = runtime.repository._connection.execute(
            "SELECT * FROM consolidation_runs WHERE id = ?",
            (consolidation_id,),
        ).fetchone()
        recovery = runtime.recovery

    assert recovery == {"turns": 1, "tools": 1, "consolidations": 1}
    assert trace["turn"]["status"] == "interrupted"
    assert trace["turn"]["terminal_reason"] == "process_restart"
    statuses = {call["id"]: call["status"] for call in trace["tool_calls"]}
    assert statuses == {
        "inflight-call": "interrupted",
        "completed-call": "succeeded",
    }
    assert consolidation["status"] == "interrupted"


def test_second_runtime_cannot_corrupt_active_turn_recovery(settings):
    with create_runtime(settings) as first_runtime:
        session = first_runtime.repository.create_session("Exclusive")
        turn_id = first_runtime.repository.create_turn(session.id)
        with pytest.raises(RuntimeError, match="Another PalmClaw runtime"):
            create_runtime(settings)
        assert (
            first_runtime.repository.get_trace(turn_id)["turn"]["status"] == "running"
        )

    with create_runtime(settings) as restarted:
        assert restarted.recovery["turns"] == 1
        assert (
            restarted.repository.get_trace(turn_id)["turn"]["status"] == "interrupted"
        )


def test_duplicate_unsafe_side_effect_is_not_reexecuted(settings):
    safe_settings = replace(settings, memory_trigger_messages=100)
    arguments = {
        "path": "once.txt",
        "content": "written once",
        "overwrite": True,
    }
    model = ScriptedAgentModel(
        [
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="write-once",
                        name="file_write",
                        arguments=arguments,
                    ),
                ),
            ),
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="write-duplicate",
                        name="file_write",
                        arguments=arguments,
                    ),
                ),
            ),
            AgentResponse(content="done"),
        ]
    )
    with create_runtime(
        safe_settings,
        agent_model=model,
        memory_model=FakeMemoryModel(),
    ) as runtime:
        session = runtime.repository.create_session("Retry safety")
        result = runtime.turn_coordinator.run(session.id, "write once")
        trace = runtime.repository.get_trace(result.turn_id)
        content = (safe_settings.workspace_root / session.id / "once.txt").read_text(
            encoding="utf-8"
        )

    assert content == "written once"
    assert [call["status"] for call in trace["tool_calls"]] == [
        "succeeded",
        "denied",
    ]
    assert trace["tool_calls"][0]["side_effect"] == "workspace_write"
    assert trace["tool_calls"][0]["retry_safety"] == "unsafe"
    assert (
        trace["tool_calls"][0]["fingerprint"] == trace["tool_calls"][1]["fingerprint"]
    )
    assert trace["tool_results"][1]["metadata"]["error_code"] == "duplicate_side_effect"


def test_provider_timeout_leaves_terminal_turn(settings):
    model = _SlowAgentModel()
    timeout_settings = replace(
        settings,
        model_timeout_seconds=0.05,
        memory_trigger_messages=100,
    )
    with create_runtime(
        timeout_settings,
        agent_model=model,
        memory_model=FakeMemoryModel(),
    ) as runtime:
        session = runtime.repository.create_session("Timeout")
        with pytest.raises(TimeoutError, match="AgentModel timed out"):
            runtime.turn_coordinator.run(session.id, "wait")
        model.release.set()
        turn = runtime.repository.list_turns(session.id)[0]
        trace = runtime.repository.get_trace(turn["id"])

    assert turn["status"] == "timed_out"
    assert turn["terminal_reason"] == "provider_timeout"
    assert "TimeoutError" in trace["model_calls"][0]["error"]


def test_user_cancellation_leaves_terminal_turn(settings):
    cancel_settings = replace(settings, memory_trigger_messages=100)
    with create_runtime(
        cancel_settings,
        agent_model=_CancelledAgentModel(),
        memory_model=FakeMemoryModel(),
    ) as runtime:
        session = runtime.repository.create_session("Cancel")
        with pytest.raises(KeyboardInterrupt):
            runtime.turn_coordinator.run(session.id, "cancel")
        turn = runtime.repository.list_turns(session.id)[0]

    assert turn["status"] == "cancelled"
    assert turn["terminal_reason"] == "user_cancelled"


def test_max_rounds_is_a_terminal_failure(settings):
    bounded_settings = replace(
        settings,
        max_tool_rounds=2,
        memory_trigger_messages=100,
    )
    model = ScriptedAgentModel(
        [
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id=f"read-{index}",
                        name="file_read",
                        arguments={"path": "missing.txt"},
                    ),
                ),
            )
            for index in range(2)
        ]
    )
    with create_runtime(
        bounded_settings,
        agent_model=model,
        memory_model=FakeMemoryModel(),
    ) as runtime:
        session = runtime.repository.create_session("Max rounds")
        result = runtime.turn_coordinator.run(session.id, "loop")
        trace = runtime.repository.get_trace(result.turn_id)

    assert result.status == "failed"
    assert result.terminal_reason == "max_rounds"
    assert trace["turn"]["round_count"] == 2


def test_same_session_turns_are_serialized():
    loop = _GateLoop(target_active=2)
    coordinator = SessionTurnCoordinator(
        loop,
        max_concurrent_sessions=2,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(coordinator.run, "same", "one")
        assert loop.ready.wait(timeout=0.1) is False
        second = executor.submit(coordinator.run, "same", "two")
        assert loop.ready.wait(timeout=0.1) is False
        assert loop.entered == 1
        loop.release.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert loop.max_active == 1
    assert loop.entered == 2


def test_global_session_concurrency_is_bounded():
    loop = _GateLoop(target_active=2)
    coordinator = SessionTurnCoordinator(
        loop,
        max_concurrent_sessions=2,
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(coordinator.run, f"session-{index}", "request")
            for index in range(3)
        ]
        assert loop.ready.wait(timeout=2)
        assert loop.active == 2
        assert loop.entered == 2
        loop.release.set()
        for future in futures:
            future.result(timeout=2)

    assert loop.max_active == 2
    assert loop.entered == 3


def test_parallel_sessions_keep_history_isolated(settings):
    concurrent_settings = replace(
        settings,
        max_concurrent_sessions=2,
        memory_trigger_messages=100,
    )
    with create_runtime(
        concurrent_settings,
        agent_model=_BarrierEchoAgentModel(),
        memory_model=FakeMemoryModel(),
    ) as runtime:
        first_session = runtime.repository.create_session("First")
        second_session = runtime.repository.create_session("Second")
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                runtime.turn_coordinator.run,
                first_session.id,
                "alpha",
            )
            second = executor.submit(
                runtime.turn_coordinator.run,
                second_session.id,
                "beta",
            )
            assert first.result(timeout=3).content == "reply:alpha"
            assert second.result(timeout=3).content == "reply:beta"
        first_history = runtime.repository.list_messages(first_session.id)
        second_history = runtime.repository.list_messages(second_session.id)

    assert [message.content for message in first_history] == [
        "alpha",
        "reply:alpha",
    ]
    assert [message.content for message in second_history] == [
        "beta",
        "reply:beta",
    ]
