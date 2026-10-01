from __future__ import annotations

import pytest

from memory_training.scripts.aggregate_section5_update_cycle_efficiency import (
    aggregate_cycles,
    aggregate_turns,
    applied_update_cycles,
)


def _record(
    scenario: int,
    sequence: int,
    *,
    applied: bool = False,
    repetition: int = 1,
    decode: int = 0,
    logical: int = 0,
    evaluated: int = 0,
) -> dict:
    return {
        "scenario_index": scenario,
        "sequence_index": sequence,
        "repetition": repetition,
        "applied_update": applied,
        "decode_tokens": decode,
        "logical_prompt_tokens": logical,
        "evaluated_prefill_tokens": evaluated,
    }


def test_applied_update_cycles_require_same_trajectory_and_next_record() -> None:
    records = [
        _record(1, 0, applied=True),
        _record(1, 1),
        _record(1, 2, applied=True),
        _record(2, 0),  # Scenario boundary: turn 1:2 has no eligible next turn.
        _record(2, 1, applied=True),  # Last record has no next turn.
    ]

    pairs = applied_update_cycles(records)

    assert [(a["sequence_index"], b["sequence_index"]) for a, b in pairs] == [(0, 1)]


def test_applied_update_cycles_do_not_use_gold_decision() -> None:
    records = [
        {
            **_record(1, 0, applied=False),
            "gold_decision": "UPDATE",
        },
        _record(1, 1),
        {
            **_record(1, 2, applied=True),
            "gold_decision": "NO_OP",
        },
        _record(1, 3),
    ]

    pairs = applied_update_cycles(records)

    assert len(pairs) == 1
    assert pairs[0][0]["sequence_index"] == 2


def test_aggregate_cycles_uses_update_decode_and_next_turn_prefill() -> None:
    pairs = [
        (
            _record(1, 0, applied=True, decode=20),
            _record(1, 1, logical=100, evaluated=40),
        ),
        (
            _record(1, 2, applied=True, decode=40),
            _record(1, 3, logical=200, evaluated=60),
        ),
    ]

    metrics = aggregate_cycles(
        pairs,
        prefill_tokens_per_second=10.0,
        decode_tokens_per_second=5.0,
    )

    assert metrics["cycles"] == 2
    assert metrics["update_decode_tokens_mean"] == 30.0
    assert metrics["next_logical_prefill_tokens_mean"] == 150.0
    assert metrics["next_evaluated_prefill_tokens_mean"] == 50.0
    assert metrics["projected_cache_on_latency_seconds_mean"] == 11.0
    assert metrics["projected_cache_off_latency_seconds_mean"] == 21.0


def test_aggregate_cycles_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No applied UPDATE cycles"):
        aggregate_cycles(
            [],
            prefill_tokens_per_second=10.0,
            decode_tokens_per_second=5.0,
        )


def test_aggregate_turns_reports_amortized_model_and_retrieval_cost() -> None:
    records = [
        {
            **_record(1, 0, decode=20, logical=100, evaluated=40),
            "retrieval_seconds": 0.2,
        },
        {
            **_record(1, 1, decode=40, logical=200, evaluated=60),
            "retrieval_seconds": 0.4,
        },
    ]

    metrics = aggregate_turns(
        records,
        prefill_tokens_per_second=10.0,
        decode_tokens_per_second=5.0,
    )

    assert metrics["all_turns"] == 2
    assert metrics["all_turn_model_tokens_mean"] == 80.0
    assert metrics["all_turn_projected_latency_seconds_mean"] == 11.0
    assert metrics["all_turn_retrieval_seconds_mean"] == pytest.approx(0.3)
    assert metrics["generation_errors"] == 0
    assert metrics["manager_invocations"] == 0
    assert metrics["manager_invocation_rate"] == 0.0
