from __future__ import annotations

from datetime import datetime
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.dataset import (
    GoldToolCall,
    VehicleScenario,
    VehicleTask,
)
from palmclaw_ubuntu.vehicle_bench.memory import VehicleHistoryEntry
from palmclaw_ubuntu.vehicle_bench.turnwise_gold import (
    align_vehicle_oracle_events,
    apply_reviewed_alignment_overrides,
    build_turnwise_canonical_labels,
    build_turnwise_gold_smoke_audit,
    build_vehicle_oracle_ledger,
    write_turnwise_gold_smoke_artifacts,
)


def _scenario(tmp_path: Path, gold_memory: str) -> VehicleScenario:
    history_path = tmp_path / "history_1.txt"
    qa_path = tmp_path / "qa_1.json"
    history_path.write_text("history\n", encoding="utf-8")
    qa_path.write_text("{}\n", encoding="utf-8")
    return VehicleScenario(
        index=1,
        history_path=history_path,
        qa_path=qa_path,
        tasks=(
            VehicleTask(
                id="vehicle-01-00",
                scenario_index=1,
                event_index=0,
                query="Use the comfortable ventilation level.",
                gold_memory=gold_memory,
                reasoning_type="error_correction",
                gold_calls=(
                    GoldToolCall(
                        name="carcontrol_seat_set_ventilation_speed",
                        arguments={"seat": "driver", "speed": 2},
                        source=(
                            "carcontrol_seat_set_ventilation_speed("
                            'seat="driver", speed=2)'
                        ),
                    ),
                ),
            ),
        ),
    )


def _entry(line_number: int, timestamp: str, speaker: str, content: str):
    return VehicleHistoryEntry(
        line_number=line_number,
        timestamp=datetime.strptime(timestamp, "%Y-%m-%d %H:%M"),
        speaker=speaker,
        content=content,
        raw=f"[{timestamp}] {speaker}: {content}",
    )


def test_oracle_ledger_splits_multiple_events_and_preserves_contractions(tmp_path):
    scenario = _scenario(
        tmp_path,
        "[April 10, 2025] At 1:00 PM, Justin said: 'It's hot. Let's turn on "
        "the seat ventilation.' He set it to 3. At 1:05 PM, Justin said: "
        "'That's too strong.' He lowered it to 2.",
    )
    entries = (
        _entry(
            1,
            "2025-04-10 13:00",
            "Justin",
            "It's hot. Let's turn on the seat ventilation.",
        ),
        _entry(2, "2025-04-10 13:05", "Justin", "That's too strong."),
    )

    events = build_vehicle_oracle_ledger(scenario, entries)

    assert len(events) == 2
    assert events[0].evidence_hints == (
        "It's hot. Let's turn on the seat ventilation.",
    )
    assert events[1].evidence_hints == ("That's too strong.",)
    assert events[0].time == "13:00"
    assert events[1].time == "13:05"


def test_exact_oracle_alignment_and_artifact_write(tmp_path):
    scenario = _scenario(
        tmp_path,
        "[April 10, 2025] At 1:00 PM, Justin said: 'It's hot. Let's turn on "
        "the seat ventilation.' He set it to 3.",
    )
    entries = (
        _entry(
            1,
            "2025-04-10 13:00",
            "Justin",
            "It's hot. Let's turn on the seat ventilation.",
        ),
    )
    events = build_vehicle_oracle_ledger(scenario, entries)
    alignments = align_vehicle_oracle_events(events, entries)
    audit = build_turnwise_gold_smoke_audit(
        scenario=scenario,
        history_entries=entries,
        events=events,
        alignments=alignments,
    )

    assert alignments[0].status == "MATCHED_EXACT"
    assert alignments[0].turn_index == 0
    assert alignments[0].evidence_quote == (
        "It's hot. Let's turn on the seat ventilation."
    )
    assert audit["ready_for_canonical_replay"] is True

    output = write_turnwise_gold_smoke_artifacts(
        tmp_path / "output",
        scenario=scenario,
        history_entries=entries,
        events=events,
        alignments=alignments,
        audit=audit,
        source_metadata={"test": True},
    )
    assert (output / "oracle_ledger.json").is_file()
    assert (output / "turn_alignment.jsonl").is_file()
    assert (output / "review_queue.jsonl").read_text(encoding="utf-8") == ""


def test_reviewed_override_resolves_semantic_alignment(tmp_path):
    scenario = _scenario(
        tmp_path,
        "[April 10, 2025] At 1:00 PM, Justin drove and set seat ventilation "
        "to level 3.",
    )
    entries = (
        _entry(
            1,
            "2025-04-10 13:01",
            "Justin",
            "It's hot, so I set seat ventilation to level 3.",
        ),
    )
    events = build_vehicle_oracle_ledger(scenario, entries)
    alignments = align_vehicle_oracle_events(events, entries)
    assert alignments[0].status == "REVIEW_REQUIRED"

    resolved = apply_reviewed_alignment_overrides(
        alignments,
        {events[0].event_id: 0},
        entries,
    )

    assert resolved[0].status == "MATCHED_EXACT"
    assert resolved[0].match_basis.startswith("reviewed_override:")


def test_canonical_replay_requires_reviewed_events(tmp_path):
    scenario = _scenario(
        tmp_path,
        "[April 10, 2025] At 1:00 PM, Justin said: 'Set ventilation to 3.'",
    )
    entries = (
        _entry(1, "2025-04-10 13:00", "Justin", "Set ventilation to 3."),
    )
    events = build_vehicle_oracle_ledger(scenario, entries)
    alignments = align_vehicle_oracle_events(events, entries)
    trace = _trace_turn(entries[0])

    labels = build_turnwise_canonical_labels(
        method="combined",
        history_entries=entries,
        trace_turns=(trace,),
        events=events,
        alignments=alignments,
    )

    assert labels[0]["decision"] == "UPDATE"
    assert labels[0]["candidate_assessment"] == "CORRECT"
    assert labels[0]["operations"][0]["op"] == "add"
    assert labels[0]["after_memory"].startswith("- [2025-04-10 13:00]")


def _trace_turn(entry: VehicleHistoryEntry):
    from palmclaw_ubuntu.vehicle_bench.critic_trace import CriticSourceTurn

    return CriticSourceTurn(
        method="combined",
        scenario_index=1,
        turn_index=0,
        message_id=1,
        role="user",
        content=f"[VehicleMemBench history turn=0]\n{entry.raw}",
        date=entry.date,
        consolidation_run_id="run",
        model_call_id="call",
        original_decision="NO_OP",
        original_before_memory="",
        original_after_memory="",
        original_before_sha256="",
        original_after_sha256="",
        recorded_provider_after_sha256=None,
        storage_transformed=False,
        model_id="test",
        prompt_version="test",
        schema_version="test",
        usage={},
        latency_ms=0,
        metadata={},
    )
