from __future__ import annotations

from memory_training.capture_hf_memory_snapshots import _serialize_state
from memory_training.methods.delta_v2 import DeltaV2State
from memory_training.methods.patch import PatchMethod


def test_patch_snapshot_has_materialized_state_without_pending_updates() -> None:
    snapshot = _serialize_state(PatchMethod(), "### Ada\n- remembered fact")
    assert snapshot["materialized_memory"] == "### Ada\n- remembered fact"
    assert snapshot["base_memory"] == snapshot["materialized_memory"]
    assert snapshot["pending_updates"] == []
    assert snapshot["pending_depth"] == 0


def test_delta_snapshot_preserves_base_and_pending_state() -> None:
    state = DeltaV2State(base_summary="### Ada\n- old")

    class Materializer:
        @staticmethod
        def materialize_memory(value: DeltaV2State) -> str:
            return value.base_summary

    snapshot = _serialize_state(Materializer(), state)
    assert snapshot["materialized_memory"] == "### Ada\n- old"
    assert snapshot["base_memory"] == "### Ada\n- old"
    assert snapshot["pending_updates"] == ()
    assert snapshot["pending_depth"] == 0
