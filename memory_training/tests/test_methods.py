from __future__ import annotations

import json

import pytest

from memory_training.methods import (
    DeltaMethod,
    DeltaV2Method,
    DeltaV3AppendMethod,
    DeltaV3CompactK2Method,
    DeltaV3CompactK5Method,
    DeltaV3CompactK10Method,
    PatchMethod,
    SummaryBatchMethod,
    SummaryMethod,
    SummaryReasonMethod,
)
from memory_training.validation import (
    row_with_runtime_state,
    state_from_input,
    state_to_input,
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


def test_summary_batch_formats_all_turns_and_runtime_state_preserves_them() -> None:
    method = SummaryBatchMethod()
    row = {
        "scenario_index": 81,
        "turn_id": "batch:2026-01-01",
        "batch_date": "2026-01-01",
        "input": {
            "previous_memory": "- old",
            "turns": [
                {
                    "turn_id": "turn-1",
                    "timestamp": "2026-01-01T00:00",
                    "speaker_name": "Alex",
                    "text": "Set it to 3.",
                }
            ],
        },
        "target": {"decision": "NO_OP", "next_memory": "- old"},
    }
    assert '"turns"' in method.format_input(row)
    replay = row_with_runtime_state(row, method, "- corrected")
    assert replay["input"]["previous_memory"] == "- corrected"
    assert replay["input"]["turns"] == row["input"]["turns"]


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


def test_delta_v2_compact_operations_and_compaction() -> None:
    method = DeltaV2Method(compaction_interval=2)
    row = {
        **_row(view="patch", previous="", decision="UPDATE"),
        "input": {"base_summary": "", "pending_updates": []},
        "target": {
            "decision": "UPDATE",
            "operations": [["add", "- a"]],
        },
    }
    assert "\n" not in method.format_input(row)
    assert method.format_target(row) == (
        '{"decision":"UPDATE","operations":[["add","- a"]]}'
    )
    state = method.apply_output(
        method.initial_state(), method.parse_output(method.format_target(row))
    )
    assert len(state.pending_updates) == 1
    assert method.materialize_memory(state) == "- a"

    positioned = method.parse_output(
        '{"decision":"UPDATE","operations":'
        '[["add","- a","- b"],["replace","- b","- bb"]]}'
    )
    state = method.apply_output(state, positioned)
    assert state.pending_updates == ()
    assert state.base_summary == "- a\n- bb"


def test_delta_v3_append_uses_base_only_at_cache_epoch_start() -> None:
    method = DeltaV3AppendMethod()
    row = {
        **_row(view="patch", previous="", decision="NO_OP"),
        "input": {
            "base_summary": "- old",
            "pending_updates": [[["add", "- new"]]],
        },
    }

    epoch = json.loads(method.format_input(row))
    followup = json.loads(method.format_followup_turn(row))

    assert epoch["base_summary"] == "- old\n\n- new"
    assert "pending_updates" not in epoch
    assert set(followup) == {"current_turn"}
    assert "base_summary" not in followup


@pytest.mark.parametrize(
    ("method_type", "interval"),
    (
        (DeltaV3CompactK2Method, 2),
        (DeltaV3CompactK5Method, 5),
        (DeltaV3CompactK10Method, 10),
    ),
)
def test_delta_v3_compact_profiles_preserve_state_and_turn_fields(
    method_type: type[DeltaV2Method], interval: int
) -> None:
    method = method_type()
    row = {
        **_row(view="patch", previous="", decision="UPDATE"),
        "global_turn_index": 7,
        "input": {
            "base_summary": "- old",
            "pending_updates": [[["add", "- new"]]],
        },
        "target": {
            "decision": "UPDATE",
            "operations": [["replace", "- old", "- current"]],
        },
    }

    prompt = method.format_input(row)

    assert method.compaction_interval == interval
    assert prompt == (
        "B:\n- old\nP:\n[[[\"add\",\"- new\"]]]\n"
        "T:\nturn-1|2026-01-01T00:00|p1|Alex\nSet the preferred value."
    )
    assert "applying pending patches P to base B in order" in method.system_prompt
    assert method.format_target(row) == (
        '{"decision":"UPDATE","operations":'
        '[["replace","- old","- current"]]}'
    )


def test_delta_v2_delete_noop_and_strict_compact_schema() -> None:
    method = DeltaV2Method()
    deleted = method.parse_output(
        '{"decision":"UPDATE","operations":[["delete","- old"]]}'
    )
    assert deleted.payload["operations"] == (
        {"op": "delete", "target": "- old", "content": ""},
    )
    assert method.parse_output('{"decision":"NO_OP"}').decision == "NO_OP"
    with pytest.raises(ValueError, match="invalid arity"):
        method.parse_output(
            '{"decision":"UPDATE","operations":[["delete","- old","extra"]]}'
        )
    with pytest.raises(ValueError, match="fields differ"):
        method.parse_output('{"decision":"NO_OP","operations":[]}')


def test_delta_v2_rejects_five_pending_updates_in_input() -> None:
    method = DeltaV2Method()
    row = {
        **_row(view="patch", previous="", decision="NO_OP"),
        "input": {
            "base_summary": "",
            "pending_updates": [[["add", f"- fact {index}"]] for index in range(5)],
        },
    }
    with pytest.raises(ValueError, match="compacted before the next turn"):
        method.format_input(row)


def test_delta_v2_runtime_state_round_trip_preserves_pending_updates() -> None:
    method = DeltaV2Method()
    canonical = {
        **_row(view="patch", previous="", decision="NO_OP"),
        "input": {
            "base_summary": "- old  ",
            "pending_updates": [
                [["add", "- new"]],
                [["replace", "- old", "- current"]],
            ],
        },
    }

    state = state_from_input(method, canonical["input"])
    assert state.base_summary == "- old"
    assert method.materialize_memory(state) == "- current\n\n- new"

    replay = row_with_runtime_state(canonical, method, state)
    assert replay["input"] == {
        "base_summary": "- old",
        "pending_updates": [
            [["add", "- new"]],
            [["replace", "- old", "- current"]],
        ],
    }
    assert state_from_input(method, state_to_input(method, state)) == state

    updated = method.apply_output(
        state,
        method.parse_output('{"decision":"UPDATE","operations":[["delete","- new"]]}'),
    )
    next_row = row_with_runtime_state(canonical, method, updated)
    assert next_row["input"]["pending_updates"][-1] == [["delete", "- new"]]


def test_delta_v2_rejects_invalid_target_before_deferring_update() -> None:
    method = DeltaV2Method()
    state = method.apply_output(
        method.initial_state(),
        method.parse_output(
            '{"decision":"UPDATE","operations":[["add","- existing"]]}'
        ),
    )

    invalid = method.parse_output(
        '{"decision":"UPDATE","operations":'
        '[["replace","- missing","- replacement"]]}'
    )
    with pytest.raises(ValueError, match="target must occur exactly once"):
        method.apply_output(state, invalid)

    assert method.materialize_memory(state) == "- existing"
    assert len(state.pending_updates) == 1
