from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v2_hybrid import text_sha256
from palmclaw_ubuntu.vehicle_bench.v2_training_export import (
    CanonicalTrainingTurn,
    build_batch_training_views,
    build_training_views,
    canonical_hybrid_run_root,
)


def _turn(
    index: int,
    *,
    before: str,
    operation: dict[str, str] | None,
    timestamp: str = "2025-01-01T10:00",
) -> tuple[CanonicalTrainingTurn, str]:
    if operation is None:
        after = before
        decision = "NO_OP"
        operations = ()
        stored_after = None
        reason_code = "NO_NEW_VEHICLE_FACT"
    else:
        after = "\n\n".join(
            part for part in (before, operation["content"]) if part
        )
        decision = "UPDATE"
        operations = (operation,)
        stored_after = after
        reason_code = "NEW_VEHICLE_MEMORY"
    return (
        CanonicalTrainingTurn(
            global_turn_index=index,
            event_turn_index=index,
            turn_id=f"event-1-turn-{index + 1:03d}",
            event_id="event-1",
            timestamp=timestamp,
            speaker_id="p0",
            speaker_name="Person 0",
            text=f"Turn {index}",
            decision=decision,
            reason_code=reason_code,
            reason="One-line reason.",
            operations=operations,
            evidence=(),
            source_update_indexes=(0,) if operation else (),
            before_memory_sha256=text_sha256(before),
            after_memory_sha256=text_sha256(after),
            after_memory=stored_after,
        ),
        after,
    )


def test_build_training_views_compacts_every_five_updates_and_final_flush() -> None:
    turns = []
    memory = ""
    no_op, memory = _turn(0, before=memory, operation=None)
    turns.append(no_op)
    for index in range(1, 7):
        operation = {
            "op": "add",
            "target": "",
            "content": f"- memory line {index}",
        }
        turn, memory = _turn(index, before=memory, operation=operation)
        turns.append(turn)

    summary, patch, delta, compaction = build_training_views(
        scenario_index=1,
        run_id="run-1",
        source_hybrid_sha256="a" * 64,
        turns=turns,
        split="train",
        compaction_interval=5,
    )

    assert len(summary) == len(patch) == len(delta) == 7
    assert [row["trigger"] for row in compaction] == [
        "UPDATE_INTERVAL",
        "FINAL_FLUSH",
    ]
    assert len(compaction[0]["input"]["pending_deltas"]) == 5
    assert len(compaction[1]["input"]["pending_deltas"]) == 1
    assert delta[0]["target"]["operations"] == ()
    assert patch[0]["target"]["operations"] == ()
    assert patch[1]["input"]["previous_memory"] == ""
    assert patch[2]["input"]["previous_memory"] == summary[1]["target"]["next_memory"]
    assert patch[1]["target"]["operations"] == delta[1]["target"]["operations"]
    assert delta[5]["input"]["updates_since_compaction"] == 4
    assert delta[6]["input"]["updates_since_compaction"] == 0
    assert summary[-1]["target"]["next_memory"] == memory
    assert compaction[-1]["target"]["next_summary"] == memory


def test_canonical_run_resolution_preserves_legacy_r2_exceptions() -> None:
    root = Path("/data")

    assert canonical_hybrid_run_root(root, 1).name == "hybrid-full-terra-s1-r2"
    assert canonical_hybrid_run_root(root, 2).name.endswith("s02-r2")
    assert canonical_hybrid_run_root(root, 3).name.endswith("s03-r1")
    assert canonical_hybrid_run_root(root, 100).name.endswith("s100-r1")


def test_batch_views_preserve_ordered_updates_and_end_of_day_memory() -> None:
    turns = []
    memory = ""
    no_op, memory = _turn(
        0,
        before=memory,
        operation=None,
        timestamp="2025-01-01T08:00",
    )
    turns.append(no_op)
    for index in range(1, 3):
        update, memory = _turn(
            index,
            before=memory,
            operation={
                "op": "add",
                "target": "",
                "content": f"- memory line {index}",
            },
            timestamp=f"2025-01-01T0{8 + index}:00",
        )
        turns.append(update)
    next_day, memory = _turn(
        3,
        before=memory,
        operation=None,
        timestamp="2025-01-02T08:00",
    )
    turns.append(next_day)

    summary, patch = build_batch_training_views(
        scenario_index=1,
        run_id="run-1",
        source_hybrid_sha256="a" * 64,
        turns=turns,
        split="train",
    )

    assert len(summary) == len(patch) == 2
    assert patch[0]["target"]["decision"] == "UPDATE"
    assert patch[0]["target"]["update_count"] == 2
    assert len(patch[0]["target"]["operations"]) == 2
    assert len(patch[0]["provenance"]["update_sources"]) == 2
    assert summary[0]["target"]["next_memory"] == memory
    assert patch[1]["target"]["decision"] == "NO_OP"
    assert patch[1]["target"]["operations"] == ()
    assert patch[1]["input"]["previous_memory"] == memory
    assert summary[1]["target"]["next_memory"] == memory
