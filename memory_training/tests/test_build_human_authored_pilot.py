from __future__ import annotations

import json
from pathlib import Path

from memory_training.scripts.build_human_authored_pilot import build

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    REPOSITORY_ROOT
    / "evaluation"
    / "human-authored-vehicle-memory"
    / "pilot-hv01"
    / "scenario-source.json"
)
SOURCE_HVP02 = (
    REPOSITORY_ROOT
    / "evaluation"
    / "human-authored-vehicle-memory"
    / "pilot-hv02"
    / "scenario-source.json"
)
SOURCE_HVP03 = (
    REPOSITORY_ROOT
    / "evaluation"
    / "human-authored-vehicle-memory"
    / "pilot-hv03"
    / "scenario-source.json"
)
SOURCE_HVP04 = (
    REPOSITORY_ROOT
    / "evaluation"
    / "human-authored-vehicle-memory"
    / "pilot-hv04"
    / "scenario-source.json"
)
SOURCE_HVP05 = (
    REPOSITORY_ROOT
    / "evaluation"
    / "human-authored-vehicle-memory"
    / "pilot-hvp05"
    / "scenario-source.json"
)
SOURCE_HVP06 = SOURCE_HVP05.parent.parent / "pilot-hvp06" / "scenario-source.json"
SOURCE_HVP07 = SOURCE_HVP05.parent.parent / "pilot-hvp07" / "scenario-source.json"
SOURCE_HVP08 = SOURCE_HVP05.parent.parent / "pilot-hvp08" / "scenario-source.json"
SOURCE_HVP09 = SOURCE_HVP05.parent.parent / "pilot-hvp09" / "scenario-source.json"
SOURCE_HVP10 = SOURCE_HVP05.parent.parent / "pilot-hvp10" / "scenario-source.json"
SOURCE_HVP11 = SOURCE_HVP05.parent.parent / "pilot-hvp11" / "scenario-source.json"
SOURCE_HVP12 = SOURCE_HVP05.parent.parent / "pilot-hvp12" / "scenario-source.json"
SOURCE_HVP13 = SOURCE_HVP05.parent.parent / "pilot-hvp13" / "scenario-source.json"
SOURCE_HVP14 = SOURCE_HVP05.parent.parent / "pilot-hvp14" / "scenario-source.json"
SOURCE_HVP15 = SOURCE_HVP05.parent.parent / "pilot-hvp15" / "scenario-source.json"
SOURCE_HVP16 = SOURCE_HVP05.parent.parent / "pilot-hvp16" / "scenario-source.json"
SOURCE_HVP17 = SOURCE_HVP05.parent.parent / "pilot-hvp17" / "scenario-source.json"
SOURCE_HVP18 = SOURCE_HVP05.parent.parent / "pilot-hvp18" / "scenario-source.json"
SOURCE_HVP19 = SOURCE_HVP05.parent.parent / "pilot-hvp19" / "scenario-source.json"
SOURCE_HVP20 = SOURCE_HVP05.parent.parent / "pilot-hvp20" / "scenario-source.json"
VEHICLEMEMBENCH_ROOT = REPOSITORY_ROOT.parent / "VehicleMemBench"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_hvp01_external_pilot_exports_and_replays(tmp_path: Path) -> None:
    manifest = build(SOURCE, tmp_path, VEHICLEMEMBENCH_ROOT)

    assert manifest["eligible_for_human_test"] is False
    assert manifest["statistics"] == {
        "speaker_count": 2,
        "session_count": 16,
        "turn_count": 50,
        "update_count": 10,
        "noop_count": 40,
        "add_count": 7,
        "replace_count": 3,
        "active_memory_slot_count": 7,
        "distinct_memory_tool_count": 7,
        "turn_quiz_count": 4,
        "final_quiz_count": 6,
        "quiz_tool_call_count": 12,
    }
    assert manifest["external_adaptation"]["origin_turn_counts"] == {
        "human_authored": 25,
        "synthetic": 25,
    }
    assert set(manifest["validation"].values()) == {"PASS"}

    patch_rows = load_jsonl(tmp_path / "patch.jsonl")
    snapshots = load_jsonl(tmp_path / "memory_snapshots.jsonl")
    assert len(patch_rows) == len(snapshots) == 50
    assert (
        patch_rows[-1]["provenance"]["after_memory_sha256"]
        == manifest["final_memory"]["sha256"]
    )
    assert snapshots[-1]["memory_snapshot_sha256"] == manifest["final_memory"]["sha256"]


def test_hvp01_preserves_historical_quiz_cutoffs(tmp_path: Path) -> None:
    build(SOURCE, tmp_path, VEHICLEMEMBENCH_ROOT)

    turn_quizzes = {
        row["quiz_id"]: row for row in load_jsonl(tmp_path / "turn_quiz.jsonl")
    }
    assert turn_quizzes["hvp01-turn-02"]["target"]["gold_calls"][0][
        "arguments"
    ]["volume"] == 11

    finals = {
        row["quiz_id"]: row for row in load_jsonl(tmp_path / "final_quiz.jsonl")
    }
    assert finals["hvp01-final-02"]["target"]["gold_calls"][0]["arguments"][
        "volume"
    ] == 58


def test_hvp01_final_quizzes_use_only_active_values(
    tmp_path: Path,
) -> None:
    build(SOURCE, tmp_path, VEHICLEMEMBENCH_ROOT)

    finals = load_jsonl(tmp_path / "final_quiz.jsonl")
    gold_memory = "\n".join(row["evidence"]["gold_memory"] for row in finals)
    assert "value=58" in gold_memory
    assert 'value="defrost"' in gold_memory
    assert 'value="outside"' in gold_memory
    assert "value=11" not in gold_memory
    assert 'value="auto"' not in gold_memory
    assert 'value="inside"' not in gold_memory


def test_hvp02_has_fifty_turns_and_valid_update_chain(tmp_path: Path) -> None:
    manifest = build(SOURCE_HVP02, tmp_path, VEHICLEMEMBENCH_ROOT)

    assert manifest["eligible_for_human_test"] is False
    assert manifest["statistics"] == {
        "speaker_count": 2,
        "session_count": 16,
        "turn_count": 50,
        "update_count": 10,
        "noop_count": 40,
        "add_count": 7,
        "replace_count": 3,
        "active_memory_slot_count": 7,
        "distinct_memory_tool_count": 7,
        "turn_quiz_count": 4,
        "final_quiz_count": 6,
        "quiz_tool_call_count": 12,
    }
    assert set(manifest["validation"].values()) == {"PASS"}
    assert manifest["final_memory"]["turn_id"] == "hvp02-s16-t01"

    patch_rows = load_jsonl(tmp_path / "patch.jsonl")
    assert [row["target"]["decision"] for row in patch_rows[-3:]] == [
        "UPDATE",
        "UPDATE",
        "UPDATE",
    ]
    finals = load_jsonl(tmp_path / "final_quiz.jsonl")
    assert finals[0]["target"]["gold_calls"][0]["arguments"]["volume"] == 5


def test_hvp03_preserves_external_source_provenance(tmp_path: Path) -> None:
    manifest = build(SOURCE_HVP03, tmp_path, VEHICLEMEMBENCH_ROOT)

    assert manifest["statistics"] == {
        "speaker_count": 2,
        "session_count": 9,
        "turn_count": 50,
        "update_count": 10,
        "noop_count": 40,
        "add_count": 7,
        "replace_count": 3,
        "active_memory_slot_count": 7,
        "distinct_memory_tool_count": 7,
        "turn_quiz_count": 4,
        "final_quiz_count": 6,
        "quiz_tool_call_count": 11,
    }
    assert manifest["external_adaptation"] == {
        "externally_sourced_turn_count": 50,
        "externally_sourced_turn_rate": 1.0,
        "dataset_turn_counts": {
            "Audio2Tool": 25,
            "Envisioned Voice Assistant Dialogues": 25,
        },
        "origin_turn_counts": {"human_authored": 25, "synthetic": 25},
        "reuse_mode_turn_counts": {
            "minimal_persistence_adaptation": 10,
            "verbatim_text_role_normalized": 40,
        },
        "source_catalog": json.loads(SOURCE_HVP03.read_text(encoding="utf-8"))[
            "external_sources"
        ],
    }
    dialogue_rows = load_jsonl(tmp_path / "dialogue.jsonl")
    patch_rows = load_jsonl(tmp_path / "patch.jsonl")
    assert all("source_trace" in row for row in dialogue_rows)
    assert all("external_source_trace" in row["provenance"] for row in patch_rows)
    assert set(manifest["validation"].values()) == {"PASS"}


def test_hvp04_uses_disjoint_external_records_and_valid_replacements(
    tmp_path: Path,
) -> None:
    manifest = build(SOURCE_HVP04, tmp_path, VEHICLEMEMBENCH_ROOT)

    assert manifest["statistics"] == {
        "speaker_count": 2,
        "session_count": 9,
        "turn_count": 50,
        "update_count": 10,
        "noop_count": 40,
        "add_count": 7,
        "replace_count": 3,
        "active_memory_slot_count": 7,
        "distinct_memory_tool_count": 7,
        "turn_quiz_count": 4,
        "final_quiz_count": 6,
        "quiz_tool_call_count": 11,
    }
    assert manifest["external_adaptation"]["origin_turn_counts"] == {
        "human_authored": 25,
        "synthetic": 25,
    }
    assert manifest["external_adaptation"]["reuse_mode_turn_counts"] == {
        "minimal_persistence_adaptation": 10,
        "verbatim_text_role_normalized": 40,
    }

    hvp03 = json.loads(SOURCE_HVP03.read_text(encoding="utf-8"))
    hvp04 = json.loads(SOURCE_HVP04.read_text(encoding="utf-8"))
    hvp03_records = {
        (trace["dataset"], trace["record_id"])
        for session in hvp03["sessions"]
        for turn in session["turns"]
        for trace in [turn["source_trace"]]
    }
    hvp04_records = {
        (trace["dataset"], trace["record_id"])
        for session in hvp04["sessions"]
        for turn in session["turns"]
        for trace in [turn["source_trace"]]
    }
    assert hvp03_records.isdisjoint(hvp04_records)
    assert set(manifest["validation"].values()) == {"PASS"}


def test_hvp01_hvp20_are_fifty_turn_disjoint_external_pilots(
    tmp_path: Path,
) -> None:
    generated_sources = [
        SOURCE,
        SOURCE_HVP02,
        SOURCE_HVP05,
        SOURCE_HVP06,
        SOURCE_HVP07,
        SOURCE_HVP08,
        SOURCE_HVP09,
        SOURCE_HVP10,
        SOURCE_HVP11,
        SOURCE_HVP12,
        SOURCE_HVP13,
        SOURCE_HVP14,
        SOURCE_HVP15,
        SOURCE_HVP16,
        SOURCE_HVP17,
        SOURCE_HVP18,
        SOURCE_HVP19,
        SOURCE_HVP20,
    ]
    sources = [SOURCE_HVP03, SOURCE_HVP04, *generated_sources]
    record_sets = []
    expected = {
        "HVP01": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 12},
        "HVP02": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 12},
        "HVP05": {"sessions": 13, "updates": 7, "adds": 5, "replaces": 2, "slots": 5, "tools": 5, "calls": 11},
        "HVP06": {"sessions": 13, "updates": 7, "adds": 4, "replaces": 3, "slots": 4, "tools": 4, "calls": 12},
        "HVP07": {"sessions": 12, "updates": 7, "adds": 5, "replaces": 2, "slots": 5, "tools": 5, "calls": 11},
        "HVP08": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP09": {"sessions": 17, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP10": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 6, "calls": 11},
        "HVP11": {"sessions": 17, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP12": {"sessions": 17, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP13": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP14": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP15": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP16": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP17": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP18": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP19": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
        "HVP20": {"sessions": 16, "updates": 10, "adds": 7, "replaces": 3, "slots": 7, "tools": 7, "calls": 11},
    }

    for source_path in sources:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        record_sets.append(
            {
                (turn["source_trace"]["dataset"], turn["source_trace"]["record_id"])
                for session in source["sessions"]
                for turn in session["turns"]
            }
        )

    for left_index, left in enumerate(record_sets):
        for right in record_sets[left_index + 1 :]:
            assert left.isdisjoint(right)

    for source_path in generated_sources:
        output = tmp_path / source_path.parent.name
        manifest = build(source_path, output, VEHICLEMEMBENCH_ROOT)
        values = expected[manifest["scenario_id"]]
        assert manifest["statistics"] == {
            "speaker_count": 2,
            "session_count": values["sessions"],
            "turn_count": 50,
            "update_count": values["updates"],
            "noop_count": 50 - values["updates"],
            "add_count": values["adds"],
            "replace_count": values["replaces"],
            "active_memory_slot_count": values["slots"],
            "distinct_memory_tool_count": values["tools"],
            "turn_quiz_count": 4,
            "final_quiz_count": 6,
            "quiz_tool_call_count": values["calls"],
        }
        assert manifest["external_adaptation"]["origin_turn_counts"] == {
            "human_authored": 25,
            "synthetic": 25,
        }
        assert manifest["external_adaptation"]["reuse_mode_turn_counts"] == {
            "minimal_persistence_adaptation": values["updates"],
            "verbatim_text_role_normalized": 50 - values["updates"],
        }
        assert set(manifest["validation"].values()) == {"PASS"}
