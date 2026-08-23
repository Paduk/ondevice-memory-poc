from __future__ import annotations

import json

import pytest

from memory_training.methods import (
    DeltaMethod,
    PatchMethod,
    SummaryMethod,
    SummaryReasonMethod,
)


def _row(
    *,
    view: str,
    previous: str,
    decision: str,
    operations: list[dict[str, str]] | None = None,
    next_memory: str | None = None,
) -> dict:
    memory_input = (
        {"previous_memory": previous}
        if view != "delta"
        else {
            "base_summary": previous,
            "pending_deltas": [],
            "updates_since_compaction": 0,
        }
    )
    return {
        "turn_id": "turn-1",
        "timestamp": "2026-01-01T00:00",
        "current_turn": {
            "speaker_id": "p1",
            "speaker_name": "Alex",
            "text": "Set the preferred value.",
        },
        "input": memory_input,
        "target": {
            "decision": decision,
            "reason_code": (
                "NO_NEW_VEHICLE_FACT" if decision == "NO_OP" else "NEW_VEHICLE_MEMORY"
            ),
            "reason": (
                "No durable vehicle fact changes."
                if decision == "NO_OP"
                else "This turn adds a durable vehicle preference."
            ),
            "operations": operations or [],
            "next_memory": previous if next_memory is None else next_memory,
        },
    }


def test_summary_drops_reason_but_summary_reason_keeps_it() -> None:
    row = _row(view="summary", previous="", decision="NO_OP")
    summary = SummaryMethod()
    reason = SummaryReasonMethod()

    assert summary.format_target(row) == '{"decision":"NO_OP"}'
    reason_target = json.loads(reason.format_target(row), object_pairs_hook=dict)
    assert list(reason_target) == ["reason_code", "reason", "decision"]
    parsed = reason.parse_output(reason.format_target(row))
    assert parsed.reason_code == "NO_NEW_VEHICLE_FACT"
    assert reason.apply_output("old", parsed) == "old"


def test_summary_update_replaces_whole_memory() -> None:
    method = SummaryMethod()
    row = _row(
        view="summary",
        previous="- old",
        decision="UPDATE",
        next_memory="- old\n\n- new",
    )
    output = method.parse_output(method.format_target(row))
    assert method.apply_output("- old", output) == "- old\n\n- new"


def test_patch_add_replace_delete_and_strict_schema() -> None:
    method = PatchMethod()
    state = "- a\n\n- b"
    operations = [
        {"op": "replace", "target": "- a", "content": "- aa"},
        {"op": "delete", "target": "- b", "content": ""},
        {"op": "add", "target": "", "content": "- c"},
    ]
    row = _row(view="patch", previous=state, decision="UPDATE", operations=operations)
    output = method.parse_output(method.format_target(row))
    assert method.apply_output(state, output) == "- aa\n\n- c"
    with pytest.raises(ValueError, match="fields differ"):
        method.parse_output('{"decision":"NO_OP","operations":[]}')
    with pytest.raises(ValueError, match="strict JSON"):
        method.parse_output('```json\n{"decision":"NO_OP"}\n```')
    with pytest.raises(ValueError, match="append a complete new user block"):
        method.parse_output(
            '{"decision":"UPDATE","operations":['
            '{"op":"add","target":"- aa","content":"### New user"}]}'
        )


def test_delta_compacts_after_interval() -> None:
    method = DeltaMethod(compaction_interval=2)
    state = method.initial_state()
    first = method.parse_output(
        '{"decision":"UPDATE","operations":[{"op":"add","target":"","content":"- a"}]}'
    )
    state = method.apply_output(state, first, turn_id="t1")
    assert state.updates_since_compaction == 1
    assert method.materialize_memory(state) == "- a"

    second = method.parse_output(
        '{"decision":"UPDATE","operations":['
        '{"op":"replace","target":"- a","content":"- b"}]}'
    )
    state = method.apply_output(state, second, turn_id="t2")
    assert state.updates_since_compaction == 0
    assert state.pending_deltas == ()
    assert state.base_summary == "- b"
