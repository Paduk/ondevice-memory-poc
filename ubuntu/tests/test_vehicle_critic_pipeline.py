from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from palmclaw_ubuntu.vehicle_bench.critic import (
    CriticEvidence,
    ValidatedVehicleMemoryCriticResult,
    VehicleCriticContextRequestPayload,
    VehicleCriticRollbackRequestPayload,
    VehicleMemoryAdjudication,
    VehicleMemoryCriticResponse,
    validate_vehicle_memory_critic_response,
)
from palmclaw_ubuntu.vehicle_bench.critic_pipeline import (
    CriticCheckpointStore,
    VehicleMemoryCriticPipeline,
)
from palmclaw_ubuntu.vehicle_bench.critic_trace import CriticSourceTurn

ResultFactory = Callable[
    [CriticSourceTurn, str],
    ValidatedVehicleMemoryCriticResult,
]


def _turn(index: int, content: str) -> CriticSourceTurn:
    return CriticSourceTurn(
        method="summary",
        scenario_index=1,
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
        original_before_sha256=f"before-{index}",
        original_after_sha256=f"after-{index}",
        recorded_provider_after_sha256=None,
        storage_transformed=False,
        model_id="gpt-5.6-luna",
        prompt_version="source-prompt",
        schema_version="source-schema",
        usage={},
        latency_ms=1,
        metadata={},
    )


def _accept_noop(
    turn: CriticSourceTurn,
    approved_memory: str,
) -> ValidatedVehicleMemoryCriticResult:
    return validate_vehicle_memory_critic_response(
        VehicleMemoryCriticResponse(
            method=turn.method,
            verdict="ACCEPT",
            decision="NO_OP",
            reason_code="UNRELATED_CONVERSATION",
            reason="The turn adds no durable vehicle information.",
        ),
        source_turn=turn,
        approved_memory=approved_memory,
        max_memory_chars=8_192,
    )


def _reject(
    turn: CriticSourceTurn,
    approved_memory: str,
) -> ValidatedVehicleMemoryCriticResult:
    return validate_vehicle_memory_critic_response(
        VehicleMemoryCriticResponse(
            method=turn.method,
            verdict="REJECT",
            decision=None,
            reason_code="NEEDS_CAUSAL_CONTEXT",
            reason="The reference needs an earlier causal turn.",
        ),
        source_turn=turn,
        approved_memory=approved_memory,
        max_memory_chars=8_192,
    )


def _correct_green(
    turn: CriticSourceTurn,
    approved_memory: str,
) -> ValidatedVehicleMemoryCriticResult:
    quote = "Gary always prefers green ambient lighting"
    return validate_vehicle_memory_critic_response(
        VehicleMemoryCriticResponse(
            method=turn.method,
            verdict="CORRECT",
            decision="UPDATE",
            reason_code="MISSED_DURABLE_INFORMATION",
            reason="The source turn explicitly states a durable preference.",
            evidence=(CriticEvidence(turn.message_id, quote),),
            next_memory="### Gary\n- Ambient color: green",
        ),
        source_turn=turn,
        approved_memory=approved_memory,
        max_memory_chars=8_192,
    )


class ScriptedLuna:
    def __init__(self, script: Sequence[tuple[int, ResultFactory]]) -> None:
        self.script = list(script)
        self.calls: list[tuple[int, str]] = []

    def review(
        self,
        *,
        source_turn: CriticSourceTurn,
        approved_memory: str,
        max_memory_chars: int,
    ) -> ValidatedVehicleMemoryCriticResult:
        assert max_memory_chars == 8_192
        expected_turn, factory = self.script.pop(0)
        assert source_turn.turn_index == expected_turn
        self.calls.append((source_turn.turn_index, approved_memory))
        return factory(source_turn, approved_memory)


class RollbackTerra:
    def __init__(self) -> None:
        self.adjudication_calls: list[tuple[int, tuple[int, ...]]] = []
        self.repair_calls: list[tuple[int, str]] = []

    def adjudicate(
        self,
        *,
        source_turn: CriticSourceTurn,
        approved_memory: str,
        luna_reject: ValidatedVehicleMemoryCriticResult,
        recovered_context: Sequence[CriticSourceTurn] = (),
        max_memory_chars: int,
    ) -> VehicleMemoryAdjudication:
        assert not luna_reject.commit_ready
        assert max_memory_chars == 8_192
        ids = tuple(turn.message_id for turn in recovered_context)
        self.adjudication_calls.append((source_turn.turn_index, ids))
        if not recovered_context:
            return VehicleMemoryAdjudication(
                resolution="CONTEXT_REQUIRED",
                context_request=VehicleCriticContextRequestPayload(
                    verdict="CONTEXT_REQUIRED",
                    reason="The usual setting needs earlier context.",
                    entity_hints=["Gary"],
                    setting_hints=["ambient lighting"],
                    temporal_hints=[],
                ),
            )
        return VehicleMemoryAdjudication(
            resolution="ROLLBACK_REQUIRED",
            rollback_request=VehicleCriticRollbackRequestPayload(
                source_message_id=1,
                quote="Gary always prefers green ambient lighting",
                reason="The durable preference was omitted at its source turn.",
            ),
        )

    def repair_omission(
        self,
        *,
        source_turn: CriticSourceTurn,
        approved_memory: str,
        rollback_request: VehicleCriticRollbackRequestPayload,
        trigger_message_id: int,
        max_memory_chars: int,
    ) -> ValidatedVehicleMemoryCriticResult:
        assert rollback_request.source_message_id == source_turn.message_id
        assert trigger_message_id == 3
        assert max_memory_chars == 8_192
        self.repair_calls.append((source_turn.turn_index, approved_memory))
        return _correct_green(source_turn, approved_memory)


class FailingTerra:
    def adjudicate(self, **_: object) -> VehicleMemoryAdjudication:
        raise AssertionError("Terra must not be called")

    def repair_omission(self, **_: object) -> ValidatedVehicleMemoryCriticResult:
        raise AssertionError("Terra repair must not be called")


class FinalRejectTerra:
    def adjudicate(
        self,
        *,
        source_turn: CriticSourceTurn,
        approved_memory: str,
        **_: object,
    ) -> VehicleMemoryAdjudication:
        return VehicleMemoryAdjudication(
            resolution="FINAL",
            final_result=_reject(source_turn, approved_memory),
        )

    def repair_omission(self, **_: object) -> ValidatedVehicleMemoryCriticResult:
        raise AssertionError("Terra repair must not be called")


class ContextFinalTerra:
    def adjudicate(
        self,
        *,
        source_turn: CriticSourceTurn,
        approved_memory: str,
        recovered_context: Sequence[CriticSourceTurn] = (),
        **_: object,
    ) -> VehicleMemoryAdjudication:
        if not recovered_context:
            return VehicleMemoryAdjudication(
                resolution="CONTEXT_REQUIRED",
                context_request=VehicleCriticContextRequestPayload(
                    verdict="CONTEXT_REQUIRED",
                    reason="The entity needs earlier context.",
                    entity_hints=["Gary"],
                    setting_hints=["ambient"],
                    temporal_hints=[],
                ),
            )
        response = VehicleMemoryCriticResponse(
            method=source_turn.method,
            verdict="CORRECT",
            decision="UPDATE",
            reason_code="MISSED_DURABLE_INFORMATION",
            reason="The current turn explicitly changes the referenced setting.",
            evidence=(CriticEvidence(source_turn.message_id, "Make it green"),),
            next_memory="### Gary\n- Ambient color: green",
        )
        return VehicleMemoryAdjudication(
            resolution="FINAL",
            final_result=validate_vehicle_memory_critic_response(
                response,
                source_turn=source_turn,
                approved_memory=approved_memory,
                max_memory_chars=8_192,
            ),
        )

    def repair_omission(self, **_: object) -> ValidatedVehicleMemoryCriticResult:
        raise AssertionError("Terra repair must not be called")


def test_pipeline_rolls_back_repairs_and_replays_causal_trajectory(
    tmp_path: Path,
) -> None:
    turns = (
        _turn(0, "Gary always prefers green ambient lighting."),
        _turn(1, "The weather is pleasant."),
        _turn(2, "Use my usual ambient setting."),
    )
    store = CriticCheckpointStore(tmp_path / "critic_state.sqlite", turns=turns)
    luna = ScriptedLuna(
        (
            (0, _accept_noop),
            (1, _accept_noop),
            (2, _reject),
            (1, _accept_noop),
            (2, _accept_noop),
        )
    )
    terra = RollbackTerra()
    pipeline = VehicleMemoryCriticPipeline(
        turns=turns,
        store=store,
        luna=luna,
        terra=terra,
        max_memory_chars=8_192,
        retrieval_top_k=1,
        retrieval_neighbor_window=0,
    )

    result = pipeline.run()

    assert result.status == "COMPLETED"
    assert result.processed_turns == 3
    assert result.replay_generation == 1
    assert result.rollback_count == 1
    assert result.attempt_count == 8
    assert result.final_memory == "### Gary\n- Ambient color: green"
    assert terra.adjudication_calls == [(2, ()), (2, (1,))]
    assert terra.repair_calls == [(0, "")]
    assert luna.calls[-2:] == [
        (1, "### Gary\n- Ambient color: green"),
        (2, "### Gary\n- Ambient color: green"),
    ]
    active = store.active_results()
    assert [item["turn_index"] for item in active] == [0, 1, 2]
    assert [item["replay_generation"] for item in active] == [1, 1, 1]
    assert active[0]["stage"] == "terra_rollback_repair"
    with sqlite3.connect(store.path) as connection:
        snapshot_count = connection.execute(
            "SELECT COUNT(*) FROM memory_snapshots"
        ).fetchone()[0]
    assert snapshot_count == 2

    labels_path = tmp_path / "labels.jsonl"
    store.export_labels(labels_path)
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    assert len(labels) == 3
    assert labels[0]["result"]["response"]["decision"] == "UPDATE"


def test_pipeline_resumes_from_transactional_checkpoint(tmp_path: Path) -> None:
    turns = (_turn(0, "Small talk."), _turn(1, "More small talk."))
    state_path = tmp_path / "critic_state.sqlite"
    first_store = CriticCheckpointStore(state_path, turns=turns)
    first_luna = ScriptedLuna(((0, _accept_noop),))
    first = VehicleMemoryCriticPipeline(
        turns=turns,
        store=first_store,
        luna=first_luna,
        terra=FailingTerra(),
        max_memory_chars=8_192,
    ).run(max_commits=1)

    assert first.status == "RUNNING"
    assert first.processed_turns == 1

    resumed_store = CriticCheckpointStore(state_path, turns=turns)
    second_luna = ScriptedLuna(((1, _accept_noop),))
    second = VehicleMemoryCriticPipeline(
        turns=turns,
        store=resumed_store,
        luna=second_luna,
        terra=FailingTerra(),
        max_memory_chars=8_192,
    ).run()

    assert second.status == "COMPLETED"
    assert second.processed_turns == 2
    assert second_luna.calls == [(1, "")]

    no_call_luna = ScriptedLuna(())
    completed = VehicleMemoryCriticPipeline(
        turns=turns,
        store=resumed_store,
        luna=no_call_luna,
        terra=FailingTerra(),
        max_memory_chars=8_192,
    ).run()
    assert completed.status == "COMPLETED"
    assert no_call_luna.calls == []

    with pytest.raises(ValueError, match="configuration does not match"):
        VehicleMemoryCriticPipeline(
            turns=turns,
            store=resumed_store,
            luna=ScriptedLuna(()),
            terra=FailingTerra(),
            max_memory_chars=8_192,
            retrieval_top_k=7,
        )


def test_pipeline_commits_current_turn_after_context_recovery(tmp_path: Path) -> None:
    turns = (
        _turn(0, "Gary discussed ambient lighting."),
        _turn(1, "Make it green."),
    )
    store = CriticCheckpointStore(tmp_path / "critic_state.sqlite", turns=turns)
    pipeline = VehicleMemoryCriticPipeline(
        turns=turns,
        store=store,
        luna=ScriptedLuna(((0, _accept_noop), (1, _reject))),
        terra=ContextFinalTerra(),
        max_memory_chars=8_192,
        retrieval_top_k=1,
        retrieval_neighbor_window=0,
    )

    result = pipeline.run()

    assert result.status == "COMPLETED"
    assert result.rollback_count == 0
    assert result.final_memory.endswith("Ambient color: green")
    assert store.active_results()[1]["stage"] == "terra_recovered"


def test_pipeline_commits_terra_reject_as_defer_keep(
    tmp_path: Path,
) -> None:
    turns = (_turn(0, "Use that setting."),)
    store = CriticCheckpointStore(tmp_path / "critic_state.sqlite", turns=turns)
    pipeline = VehicleMemoryCriticPipeline(
        turns=turns,
        store=store,
        luna=ScriptedLuna(((0, _reject),)),
        terra=FinalRejectTerra(),
        max_memory_chars=8_192,
    )

    result = pipeline.run()

    assert result.status == "COMPLETED"
    assert result.processed_turns == 1
    assert result.final_memory == ""
    label = store.active_results()[0]
    assert label["stage"] == "terra_initial_defer_keep"
    assert label["result"]["response"]["verdict"] == "DEFER_KEEP"
    assert label["result"]["response"]["decision"] == "NO_OP"
    assert label["resolution"]["defer_code"] == "TERRA_FINAL_REJECT"
    assert store.unresolved_events() == ()
    review_path = tmp_path / "review_queue.jsonl"
    store.export_review_queue(review_path)
    assert review_path.read_text() == ""


def test_pipeline_resumes_existing_terra_reject_as_defer_keep(
    tmp_path: Path,
) -> None:
    turns = (_turn(0, "Use that setting."),)
    store = CriticCheckpointStore(tmp_path / "critic_state.sqlite", turns=turns)
    store.mark_unresolved(
        turn=turns[0],
        code="TERRA_FINAL_REJECT",
        detail={
            "final": {
                "response": {
                    "verdict": "REJECT",
                    "reason_code": "AMBIGUOUS_EVIDENCE",
                    "reason": "The setting cannot be resolved safely.",
                }
            }
        },
    )
    pipeline = VehicleMemoryCriticPipeline(
        turns=turns,
        store=store,
        luna=ScriptedLuna(()),
        terra=FailingTerra(),
        max_memory_chars=8_192,
    )

    state = pipeline.resume_unresolved_as_defer_keep()

    assert state.status == "COMPLETED"
    assert state.next_turn_index == 1
    assert store.unresolved_events() == ()
    history = store.unresolved_history()
    assert history[0]["resolution_code"] == "DEFER_KEEP"
    assert history[0]["resolved_at"] is not None
    assert store.active_results()[0]["result"]["response"]["verdict"] == (
        "DEFER_KEEP"
    )


def test_checkpoint_rejects_changed_source_trace(tmp_path: Path) -> None:
    state_path = tmp_path / "critic_state.sqlite"
    CriticCheckpointStore(state_path, turns=(_turn(0, "Original."),))

    with pytest.raises(ValueError, match="does not match"):
        CriticCheckpointStore(state_path, turns=(_turn(0, "Changed."),))
