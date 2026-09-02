from __future__ import annotations

from types import SimpleNamespace

import pytest

from palmclaw_ubuntu.models import MemoryResponse, ModelUsage
from palmclaw_ubuntu.vehicle_bench.v2_cloud_memory_gap import (
    build_hybrid_cloud_replay_plan,
    execute_cloud_memory_replay,
)


class _FinalQuiz(SimpleNamespace):
    def model_dump(self, *, mode: str) -> dict[str, str]:
        assert mode == "json"
        return {"quiz_id": self.quiz_id, "query": self.query}


class _ReplayModel:
    backend = "fake"
    model_id = "fake-memory"
    prompt_version = "fake-prompt-v1"
    schema_version = "fake-schema-v1"
    max_memory_chars = 8_192

    def __init__(self, cadence: str) -> None:
        self.update_cadence = cadence
        self.calls = 0

    def update(
        self,
        *,
        previous_memory: str,
        date: str,
        daily_history: str,
    ) -> MemoryResponse:
        del date, daily_history
        self.calls += 1
        return MemoryResponse(
            content=f"{previous_memory}|m{self.calls}".lstrip("|"),
            usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            response_id=f"response-{self.calls}",
            metadata={"update_status": "updated"},
        )


def _components() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    turns = tuple(
        SimpleNamespace(
            turn_id=f"ev{index // 2 + 1}-turn-{index % 2 + 1:03d}",
            source_event_id=f"ev{index // 2 + 1}",
            timestamp=(
                f"2026-01-0{index // 2 + 1}T0{index % 2 + 8}:00"
            ),
            speaker_name="Driver",
            text=f"turn {index}",
        )
        for index in range(4)
    )
    stage3 = SimpleNamespace(
        source_stage2_sha256="a" * 64,
        scenario=SimpleNamespace(
            dialogue_turns=turns,
            final_quizzes=tuple(
                _FinalQuiz(
                    quiz_id=f"q{index}",
                    query=f"final query {index}",
                    reasoning_type="preference_conflict",
                )
                for index in range(10)
            ),
        ),
    )
    labels = tuple(
        SimpleNamespace(
            global_turn_index=index,
            turn_id=turn.turn_id,
            source_event_id=turn.source_event_id,
            event_turn_index=index % 2,
        )
        for index, turn in enumerate(turns)
    )
    hybrid = SimpleNamespace(
        source_stage2_sha256="a" * 64,
        final_memory_sha256="b" * 64,
        event_checkpoints=(
            SimpleNamespace(timeline_index=0, turn_labels=labels[:2]),
            SimpleNamespace(timeline_index=1, turn_labels=labels[2:]),
        ),
    )
    quizzes = tuple(
        SimpleNamespace(
            quiz_id=f"turn-quiz-ev{index % 2 + 1}-{index:03d}",
            global_turn_index=index % 4,
            source_event_id=turns[index % 4].source_event_id,
            event_turn_index=index % 2,
            query=f"turn query {index}",
            reasoning_type="preference_conflict",
            quiz_sha256=f"{index:064x}",
        )
        for index in range(30)
    )
    turn_quiz = SimpleNamespace(
        source_stage2_sha256="a" * 64,
        quizzes=quizzes,
    )
    return stage3, hybrid, turn_quiz


def test_builds_causal_daily_and_turnwise_cutoffs() -> None:
    stage3, hybrid, turn_quiz = _components()
    plan = build_hybrid_cloud_replay_plan(
        scenario_index=1,
        stage3=stage3,
        hybrid=hybrid,
        turn_quiz=turn_quiz,
    )

    assert plan.dialogue_turn_count == 4
    assert len(plan.daily_batches) == 2
    assert len(plan.turn_batches) == 4
    assert len(plan.quiz_checkpoints) == 40
    assert plan.batches_through("calendar_day", 0) == ()
    assert len(plan.batches_through("calendar_day", 1)) == 1
    assert len(plan.batches_through("history_entry", 0)) == 1
    assert len(plan.batches_through("history_entry", 1)) == 2


def test_rejects_mixed_frozen_stage2_sources() -> None:
    stage3, hybrid, turn_quiz = _components()
    turn_quiz.source_stage2_sha256 = "c" * 64

    with pytest.raises(ValueError, match="frozen Stage 2"):
        build_hybrid_cloud_replay_plan(
            scenario_index=1,
            stage3=stage3,
            hybrid=hybrid,
            turn_quiz=turn_quiz,
        )


def test_rejects_misaligned_dialogue_turns() -> None:
    stage3, hybrid, turn_quiz = _components()
    hybrid.event_checkpoints[0].turn_labels[0].turn_id = "wrong-turn"

    with pytest.raises(ValueError, match="turn alignment mismatch"):
        build_hybrid_cloud_replay_plan(
            scenario_index=1,
            stage3=stage3,
            hybrid=hybrid,
            turn_quiz=turn_quiz,
        )


@pytest.mark.parametrize(
    ("arm", "cadence", "expected_calls", "early_memory"),
    (
        ("turnwise_summary", "history_entry", 4, "m1"),
        ("turnwise_combined", "history_entry", 4, "m1"),
    ),
)
def test_executes_and_resumes_replay(
    tmp_path, arm: str, cadence: str, expected_calls: int, early_memory: str
) -> None:
    stage3, hybrid, turn_quiz = _components()
    plan = build_hybrid_cloud_replay_plan(
        scenario_index=1,
        stage3=stage3,
        hybrid=hybrid,
        turn_quiz=turn_quiz,
    )
    model = _ReplayModel(cadence)
    artifact = execute_cloud_memory_replay(
        plan,
        arm=arm,
        model=model,
        output_dir=tmp_path / arm,
    )

    assert model.calls == expected_calls
    assert artifact["input_tokens"] == expected_calls * 10
    early = next(
        snapshot
        for snapshot in artifact["quiz_snapshots"]
        if snapshot["cutoff_turn_index"] == 0
    )
    assert early["memory"] == early_memory

    resumed_model = _ReplayModel(cadence)
    resumed = execute_cloud_memory_replay(
        plan,
        arm=arm,
        model=resumed_model,
        output_dir=tmp_path / arm,
    )
    assert resumed_model.calls == 0
    assert resumed["artifact_sha256"] == artifact["artifact_sha256"]
