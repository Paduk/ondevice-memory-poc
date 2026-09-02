from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memory_training.dataset import build_catalog
from memory_training.quiz_sft import (
    IndexedQuizSFTDataset,
    VehicleToolSchemaStore,
    _convert_row,
    sft_split_for_scenario,
)


def _summary_row(scenario: int, turn: int, memory: str) -> dict:
    return {
        "sample_id": f"s{scenario:03d}:summary:{turn:05d}",
        "scenario_index": scenario,
        "split": "train",
        "global_turn_index": turn,
        "turn_id": f"turn-{turn}",
        "target": {"decision": "UPDATE", "next_memory": memory},
    }


def _write_catalog_data(root: Path, memory: str = "- preferred brightness=3") -> None:
    summary = _summary_row(15, 3, memory)
    for view in ("summary", "patch", "delta"):
        row = dict(summary)
        row["sample_id"] = f"s015:{view}:00003"
        if view != "summary":
            row["target"] = {"decision": "UPDATE", "operations": []}
        (root / f"{view}.jsonl").write_text(json.dumps(row) + "\n")
    (root / "manifest.json").write_text("{}")
    (root / "quiz_manifest.json").write_text("{}")


def test_sft_split_keeps_pilot_scenarios_out_of_training() -> None:
    assert sft_split_for_scenario(14) == "excluded"
    assert sft_split_for_scenario(15) == "train"
    assert sft_split_for_scenario(81) == "validation"
    assert sft_split_for_scenario(86) == "test"
    assert sft_split_for_scenario(101) == "train"
    assert sft_split_for_scenario(110) == "train"
    assert sft_split_for_scenario(111) == "validation"
    assert sft_split_for_scenario(112) == "test"
    assert sft_split_for_scenario(120) == "test"
    assert sft_split_for_scenario(201) == "train"
    assert sft_split_for_scenario(250) == "train"
    assert sft_split_for_scenario(301) == "train"
    assert sft_split_for_scenario(320) == "train"


def test_convert_quiz_resolves_gold_memory_and_hf_tool_calls(tmp_path: Path) -> None:
    from memory_training.dataset import DatasetCatalog, IndexedMemoryDataset

    data_root = tmp_path / "data"
    data_root.mkdir()
    memory = "- preferred brightness=3"
    _write_catalog_data(data_root, memory)
    catalog_path = tmp_path / "catalog.sqlite"
    build_catalog(data_root, catalog_path)
    summaries = IndexedMemoryDataset(DatasetCatalog(catalog_path, data_root), "summary")
    parameters = {
        "type": "object",
        "properties": {"brightness": {"type": "integer"}},
        "required": ["brightness"],
        "additionalProperties": False,
    }
    from jsonschema import Draft202012Validator

    row = {
        "sample_id": "s015:turn_quiz:q1",
        "scenario_index": 15,
        "split": "train",
        "quiz_type": "TURN",
        "quiz_id": "q1",
        "memory_ref": {
            "global_turn_index": 3,
            "summary_sample_id": "s015:summary:00003",
            "patch_sample_id": "s015:patch:00003",
            "delta_sample_id": "s015:delta:00003",
            "memory_snapshot_sha256": hashlib.sha256(memory.encode()).hexdigest(),
        },
        "input": {
            "query": "Set my preferred brightness.",
            "reasoning_type": "preference_conflict",
        },
        "target": {
            "gold_calls": [{"name": "set_brightness", "arguments": {"brightness": 3}}]
        },
    }
    converted = _convert_row(
        row,
        source_name="turn_quiz.jsonl",
        line_number=1,
        summaries=summaries,
        validators={"set_brightness": Draft202012Validator(parameters)},
        tool_modules={
            "get_brightness": "display",
            "set_brightness": "display",
            "set_temperature": "climate",
        },
        module_tools={
            "display": ["get_brightness", "set_brightness"],
            "climate": ["set_temperature"],
        },
        tools_sha256="abc",
    )
    assert converted["sft_split"] == "train"
    assert memory in converted["messages"][1]["content"]
    assert converted["messages"][2]["tool_calls"][0]["function"] == {
        "name": "set_brightness",
        "arguments": {"brightness": 3},
    }
    assert converted["tool_schema_ref"]["tool_names"] == [
        "get_brightness",
        "set_brightness",
        "set_temperature",
    ]


def test_convert_quiz_rejects_invalid_tool_arguments(tmp_path: Path) -> None:
    from jsonschema import Draft202012Validator

    from memory_training.dataset import DatasetCatalog, IndexedMemoryDataset

    data_root = tmp_path / "data"
    data_root.mkdir()
    memory = "memory"
    _write_catalog_data(data_root, memory)
    catalog_path = tmp_path / "catalog.sqlite"
    build_catalog(data_root, catalog_path)
    summaries = IndexedMemoryDataset(DatasetCatalog(catalog_path, data_root), "summary")
    row = {
        "sample_id": "s015:turn_quiz:q1",
        "scenario_index": 15,
        "split": "train",
        "quiz_type": "TURN",
        "quiz_id": "q1",
        "memory_ref": {
            "global_turn_index": 3,
            "summary_sample_id": "s015:summary:00003",
            "memory_snapshot_sha256": hashlib.sha256(memory.encode()).hexdigest(),
        },
        "input": {"query": "Set it.", "reasoning_type": "state_shift"},
        "target": {
            "gold_calls": [
                {"name": "set_brightness", "arguments": {"brightness": "high"}}
            ]
        },
    }
    validator = Draft202012Validator(
        {
            "type": "object",
            "properties": {"brightness": {"type": "integer"}},
            "required": ["brightness"],
        }
    )
    with pytest.raises(ValueError, match="Invalid Gold arguments"):
        _convert_row(
            row,
            source_name="turn_quiz.jsonl",
            line_number=1,
            summaries=summaries,
            validators={"set_brightness": validator},
            tool_modules={
                "set_brightness": "display",
                "set_temperature": "climate",
                "open_window": "window",
            },
            module_tools={
                "display": ["set_brightness"],
                "climate": ["set_temperature"],
                "window": ["open_window"],
            },
            tools_sha256="abc",
        )


def test_indexed_quiz_and_tool_store_resolve_compact_references(tmp_path: Path) -> None:
    tools_path = tmp_path / "vehicle_tools.json"
    tools_payload = {
        "schema_version": "vehiclemembench-v1-official-tools-v1",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "set_brightness",
                    "description": "Set brightness",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }
    tools_path.write_text(json.dumps(tools_payload))
    tools = VehicleToolSchemaStore(tools_path)
    quiz_path = tmp_path / "quiz_sft.jsonl"
    rows = [
        {
            "schema_version": "vehiclemembench-v2-quiz-tool-sft-v1",
            "sample_id": f"q-{split}",
            "scenario_index": scenario,
            "sft_split": split,
            "tool_schema_ref": {
                "sha256": tools.sha256,
                "tool_names": ["set_brightness"],
            },
        }
        for split, scenario in (("train", 15), ("validation", 81))
    ]
    quiz_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    train = IndexedQuizSFTDataset(quiz_path, split="train")
    assert len(train) == 1
    assert train[0]["sample_id"] == "q-train"
    assert tools.tools_for(train[0])[0]["function"]["name"] == "set_brightness"
