from __future__ import annotations

import json
from pathlib import Path

from memory_training.dataset import DatasetCatalog, IndexedMemoryDataset
from memory_training.quiz_sft import IndexedQuizSFTDataset, VehicleToolSchemaStore
from memory_training.scripts.build_human_authored_pilot import build as build_pilot
from memory_training.scripts.prepare_human_authored_pilot_evaluation import (
    build as build_evaluation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    REPOSITORY_ROOT
    / "evaluation/human-authored-vehicle-memory/pilot-hv01/scenario-source.json"
)
VEHICLEMEMBENCH_ROOT = REPOSITORY_ROOT.parent / "VehicleMemBench"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepares_isolated_v2_evaluation_root(tmp_path: Path) -> None:
    pilot_root = tmp_path / "pilot"
    evaluation_root = tmp_path / "evaluation"
    build_pilot(SOURCE, pilot_root, VEHICLEMEMBENCH_ROOT)
    manifest = build_evaluation(pilot_root, evaluation_root, VEHICLEMEMBENCH_ROOT)

    assert manifest["source_scenario_index"] == 901
    assert manifest["evaluation_alias_scenario_index"] == 120
    assert manifest["eligible_for_training"] is False
    assert manifest["counts"] == {
        "turns": 50,
        "updates": 10,
        "noops": 40,
        "turn_quizzes": 4,
        "final_quizzes": 6,
    }
    for view in ("summary", "patch", "delta"):
        rows = load_jsonl(evaluation_root / f"{view}.jsonl")
        assert len(rows) == 50
        assert {row["scenario_index"] for row in rows} == {120}
        assert {row["split"] for row in rows} == {"test"}
        assert not any(row["train_eligible"] for row in rows)
        assert not any(
            " | " in json.dumps(row.get("input", {}), ensure_ascii=False)
            or " | " in json.dumps(row.get("target", {}), ensure_ascii=False)
            for row in rows
        )

    catalog = DatasetCatalog(evaluation_root / "catalog.sqlite", evaluation_root)
    assert catalog.scenarios(split="test") == [120]
    assert len(IndexedMemoryDataset(catalog, "summary", split="test")) == 50
    quiz = IndexedQuizSFTDataset(evaluation_root / "quiz_sft.jsonl", split="test")
    assert len(quiz) == 10
    assert {quiz[index]["quiz_type"] for index in range(len(quiz))} == {"TURN", "FINAL"}
    tools = VehicleToolSchemaStore(evaluation_root / "vehicle_tools.json")
    assert all(tools.tools_for(quiz[index]) for index in range(len(quiz)))
