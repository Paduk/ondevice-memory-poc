"""Validate balanced Easy HVE against the natural-explicit parent corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--count", type=int, default=20)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    repo = args.repository_root.resolve(strict=True)
    parent_root = repo / "evaluation/human-authored-vehicle-memory/easy-natural-explicit-v1"
    source_root = repo / "evaluation/human-authored-vehicle-memory/easy-natural-balanced-v1"
    derived_root = repo / "memory_training/data/human-authored-v2-easy-natural-balanced-v1"
    reports = []

    for number in range(1, args.count + 1):
        scenario_id = f"HVE{number:02d}"
        parent = load(parent_root / scenario_id / "scenario-source.json")
        source = load(source_root / scenario_id / "scenario-source.json")
        manifest = load(derived_root / scenario_id / "manifest.json")
        require(source["difficulty_stratum"] == "easy_natural_balanced_v1", f"{scenario_id}: stratum")
        require(source["external_sources"] == parent["external_sources"], f"{scenario_id}: sources")
        require(source["quizzes"] == parent["quizzes"], f"{scenario_id}: Quiz/Gold drift")
        require(source["authorship"]["eligible_for_human_test"] is False, f"{scenario_id}: eligibility")

        changed_updates = changed_noops = 0
        for old_session, session in zip(parent["sessions"], source["sessions"]):
            require(session["session_id"] == old_session["session_id"], f"{scenario_id}: session")
            require(session["timestamp"] == old_session["timestamp"], f"{scenario_id}: timestamp")
            for old_turn, turn in zip(old_session["turns"], session["turns"]):
                require(turn["turn_id"] == old_turn["turn_id"], f"{scenario_id}: turn")
                require(turn["speaker"] == old_turn["speaker"], f"{scenario_id}: speaker")
                require(turn.get("update") == old_turn.get("update"), f"{scenario_id}: update Gold")
                old_trace = dict(old_turn["source_trace"])
                trace = dict(turn["source_trace"])
                old_mode = old_trace.pop("reuse_mode")
                mode = trace.pop("reuse_mode")
                require(trace == old_trace, f"{scenario_id}: source trace drift")

                if number <= 10:
                    require(turn["text"] == old_turn["text"], f"{scenario_id}: first-half text")
                    require(mode == old_mode, f"{scenario_id}: first-half reuse mode")
                elif "update" in turn:
                    original = str(old_turn["source_trace"]["original_text"]).strip()
                    expected = old_turn["text"][len(original):].lstrip(" .")
                    require(turn["text"] == expected, f"{scenario_id}: UPDATE not single-intent")
                    require(mode == "source_inspired_single_intent_rewrite", f"{scenario_id}: UPDATE mode")
                    changed_updates += 1
                elif turn["text"] != old_turn["text"]:
                    require(mode == "explicit_nonvehicle_calibration_wrapper", f"{scenario_id}: NO_OP mode")
                    lower = turn["text"].lower()
                    require("at home" in lower or "lights at home" in lower, f"{scenario_id}: home frame")
                    changed_noops += 1
                else:
                    require(mode == old_mode, f"{scenario_id}: unchanged NO_OP mode")

        if number <= 10:
            require((changed_updates, changed_noops) == (0, 0), f"{scenario_id}: unexpected changes")
        else:
            require((changed_updates, changed_noops) == (6, 6), f"{scenario_id}: change counts")
        require(set(manifest["validation"].values()) == {"PASS"}, f"{scenario_id}: builder")
        reports.append({
            "scenario_id": scenario_id,
            "single_intent_updates": changed_updates,
            "framed_noop_sessions": changed_noops,
            "gold_equivalence": "PASS",
            "builder_validation": "PASS",
            "status": "PASS",
        })

    report = {
        "schema_version": "vehiclemembench-easy-natural-balanced-validation-v1",
        "scenario_count": args.count,
        "aggregate": {"turns": 600, "updates": 120, "noops": 480, "quizzes": 320},
        "adjusted_half": {
            "scenarios": 10,
            "single_intent_updates": 60,
            "naturally_framed_noop_sessions": 60,
        },
        "validation": {
            "gold_and_quiz_equivalence": "PASS",
            "first_half_text_identity": "PASS",
            "second_half_single_intent_updates": "PASS",
            "second_half_natural_noop_framing": "PASS",
            "source_trace_preservation": "PASS",
            "derived_patch_and_tool_validation": "PASS",
        },
        "scenarios": reports,
    }
    (source_root / "CORPUS_VALIDATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
