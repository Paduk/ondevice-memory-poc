from __future__ import annotations

import pytest

from palmclaw_ubuntu.vehicle_bench.critic import (
    VehicleCriticContextRequestPayload,
)
from palmclaw_ubuntu.vehicle_bench.critic_retrieval import CausalTurnRetriever
from palmclaw_ubuntu.vehicle_bench.critic_trace import CriticSourceTurn


def _turn(index: int, content: str, *, scenario_index: int = 1) -> CriticSourceTurn:
    return CriticSourceTurn(
        method="summary",
        scenario_index=scenario_index,
        turn_index=index,
        message_id=index + 1,
        role="user",
        content=f"[VehicleMemBench history turn={index}]\n{content}",
        date="2025-01-01",
        consolidation_run_id=f"run-{index}",
        model_call_id=f"call-{index}",
        original_decision="NO_OP",
        original_before_memory="",
        original_after_memory="",
        original_before_sha256="before",
        original_after_sha256="after",
        recorded_provider_after_sha256=None,
        storage_transformed=False,
        model_id="gpt-5.6-luna",
        prompt_version="source-prompt",
        schema_version="source-schema",
        usage={},
        latency_ms=1,
        metadata={},
    )


def _request(*, entity: str, setting: str) -> VehicleCriticContextRequestPayload:
    return VehicleCriticContextRequestPayload(
        verdict="CONTEXT_REQUIRED",
        reason="Earlier context is required.",
        entity_hints=[entity],
        setting_hints=[setting],
        temporal_hints=[],
    )


def _turns() -> tuple[CriticSourceTurn, ...]:
    return (
        _turn(0, "Gary always prefers green ambient lighting."),
        _turn(1, "The weather is sunny."),
        _turn(2, "Thomas prefers a reclined seat."),
        _turn(3, "Gary likes audio volume four."),
        _turn(4, "The trip begins tomorrow."),
        _turn(5, "Use that setting."),
    )


def test_causal_retriever_finds_hint_match_and_includes_neighbor() -> None:
    turns = _turns()
    retriever = CausalTurnRetriever(turns)

    result = retriever.retrieve(
        turns[-1],
        _request(entity="Gary", setting="ambient lighting"),
        top_k=1,
        neighbor_window=1,
    )

    assert result.primary_message_ids == (1,)
    assert [hit.turn.message_id for hit in result.hits] == [1, 2]
    assert result.hits[0].primary is True
    assert result.hits[0].matched_hints == ("gary", "ambient lighting")
    assert result.hits[1].primary is False
    assert result.fallback_to_recent is False


def test_causal_retriever_never_returns_current_or_future_turn() -> None:
    turns = _turns()
    retriever = CausalTurnRetriever(turns)
    current = turns[3]

    result = retriever.retrieve(
        current,
        _request(entity="Gary", setting="audio"),
        top_k=8,
        neighbor_window=2,
    )

    assert all(hit.turn.turn_index < current.turn_index for hit in result.hits)
    assert all(hit.turn.message_id < current.message_id for hit in result.hits)


def test_causal_retriever_falls_back_to_recent_prefix_when_no_hint_matches() -> None:
    turns = _turns()
    retriever = CausalTurnRetriever(turns)

    result = retriever.retrieve(
        turns[-1],
        _request(entity="Unknown", setting="moonroof opacity"),
        top_k=2,
        neighbor_window=0,
    )

    assert result.fallback_to_recent is True
    assert result.primary_message_ids == (4, 5)
    assert [hit.turn.message_id for hit in result.hits] == [4, 5]


def test_causal_retriever_is_deterministic_and_deduplicates_windows() -> None:
    turns = _turns()
    retriever = CausalTurnRetriever(turns)
    request = _request(entity="Gary", setting="prefers")

    first = retriever.retrieve(turns[-1], request, top_k=2, neighbor_window=2)
    second = retriever.retrieve(turns[-1], request, top_k=2, neighbor_window=2)

    assert first.as_dict() == second.as_dict()
    ids = [hit.turn.message_id for hit in first.hits]
    assert ids == sorted(set(ids))


def test_causal_retriever_rejects_cross_scenario_or_unknown_current_turn() -> None:
    turns = _turns()
    with pytest.raises(ValueError, match="share method and scenario"):
        CausalTurnRetriever((*turns[:-1], _turn(5, "Other", scenario_index=2)))

    retriever = CausalTurnRetriever(turns)
    with pytest.raises(ValueError, match="outside"):
        retriever.retrieve(
            _turn(8, "Unknown"),
            _request(entity="Gary", setting="ambient"),
        )


@pytest.mark.parametrize(("top_k", "window"), [(0, 1), (1, -1)])
def test_causal_retriever_validates_limits(top_k: int, window: int) -> None:
    turns = _turns()
    retriever = CausalTurnRetriever(turns)

    with pytest.raises(ValueError):
        retriever.retrieve(
            turns[-1],
            _request(entity="Gary", setting="ambient"),
            top_k=top_k,
            neighbor_window=window,
        )
