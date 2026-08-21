from __future__ import annotations

from pathlib import Path

import pytest

from palmclaw_ubuntu.models import ModelUsage
from palmclaw_ubuntu.vehicle_bench.critic import (
    VehicleMemoryCriticResponse,
    validate_vehicle_memory_critic_response,
)
from palmclaw_ubuntu.vehicle_bench.critic_metrics import build_critic_metrics
from palmclaw_ubuntu.vehicle_bench.critic_pipeline import CriticCheckpointStore
from palmclaw_ubuntu.vehicle_bench.critic_trace import CriticSourceTurn


def _turn(index: int) -> CriticSourceTurn:
    return CriticSourceTurn(
        method="summary",
        scenario_index=1,
        turn_index=index,
        message_id=index + 1,
        role="user",
        content=f"[VehicleMemBench history turn={index}]\nSmall talk.",
        date="2025-01-01",
        consolidation_run_id=f"run-{index}",
        model_call_id=f"call-{index}",
        original_decision="NO_OP",
        original_before_memory="",
        original_after_memory="",
        original_before_sha256=f"before-{index}",
        original_after_sha256=f"after-{index}",
        recorded_provider_after_sha256=None,
        storage_transformed=False,
        model_id="source-model",
        prompt_version="source-prompt",
        schema_version="source-schema",
        usage={},
        latency_ms=1,
        metadata={},
    )


def test_metrics_separate_cached_cost_and_extrapolate_partial_run(
    tmp_path: Path,
) -> None:
    turns = (_turn(0), _turn(1))
    store = CriticCheckpointStore(tmp_path / "critic.sqlite", turns=turns)
    response = VehicleMemoryCriticResponse(
        method="summary",
        verdict="ACCEPT",
        decision="NO_OP",
        reason_code="UNRELATED_CONVERSATION",
        reason="The turn adds no durable vehicle information.",
        usage=ModelUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            cached_tokens=10,
        ),
        metadata={"latency_ms": 500, "semantic_retry_count": 1},
    )
    result = validate_vehicle_memory_critic_response(
        response,
        source_turn=turns[0],
        approved_memory="",
        max_memory_chars=8_192,
    )
    store.record_attempt(
        turn=turns[0],
        replay_generation=0,
        stage="luna",
        payload={
            "response": response.as_dict(),
            "approved_memory_before_sha256": (
                result.approved_memory_before_sha256
            ),
            "approved_memory_after_sha256": (
                result.approved_memory_after_sha256
            ),
            "patch_stats": {},
            "commit_ready": True,
        },
    )
    store.commit_turn(turn=turns[0], result=result, stage="luna")

    metrics = build_critic_metrics(store, turns=turns)

    assert metrics["status"] == "RUNNING"
    assert metrics["processed_turns"] == 1
    assert metrics["labels"]["verdicts"] == {"ACCEPT": 1}
    assert metrics["labels"]["original_to_final"] == {"NO_OP->NO_OP": 1}
    luna = metrics["models"]["luna"]
    assert luna["calls"] == 2
    assert luna["logical_attempts"] == 1
    assert luna["input_tokens"] == 100
    assert luna["cached_input_tokens"] == 10
    assert luna["output_tokens"] == 20
    assert luna["latency_seconds"] == 0.5
    assert luna["estimated_cost_usd"] == pytest.approx(0.000042)
    extrapolation = metrics["full_run_extrapolation"]
    assert extrapolation["estimated_full_calls"] == 4
    assert extrapolation["estimated_full_input_tokens"] == 200
    assert extrapolation["estimated_full_output_tokens"] == 40
    assert extrapolation["estimated_full_provider_latency_seconds"] == 1.0
