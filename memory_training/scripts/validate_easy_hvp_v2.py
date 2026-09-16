"""Validate that Easy-v2 only simplifies language while preserving v1 golds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memory_training.scripts.build_easy_hvp_v2_from_v1 import quiz_text, update_text


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve(strict=True)
    v1_root = root / "evaluation/human-authored-vehicle-memory/easy-explicit-v1"
    v2_root = root / "evaluation/human-authored-vehicle-memory/easy-explicit-v2"
    derived_root = root / "memory_training/data/human-authored-v2-easy-explicit-v2"
    reports = []

    for number in range(1, args.count + 1):
        scenario_id = f"HVE{number:02d}"
        v1 = json.loads((v1_root / scenario_id / "scenario-source.json").read_text(encoding="utf-8"))
        v2 = json.loads((v2_root / scenario_id / "scenario-source.json").read_text(encoding="utf-8"))
        require(v2["difficulty_stratum"] == "easy_explicit_v2", f"{scenario_id}: stratum")
        require(v2["authorship"]["eligible_for_human_test"] is False, f"{scenario_id}: eligibility")
        require(len(v1["sessions"]) == len(v2["sessions"]) == 12, f"{scenario_id}: sessions")

        people = {speaker["short_id"]: speaker["speaker_name"] for speaker in v2["speakers"]}
        v1_turns = [turn for session in v1["sessions"] for turn in session["turns"]]
        v2_turns = [turn for session in v2["sessions"] for turn in session["turns"]]
        require(len(v1_turns) == len(v2_turns) == 30, f"{scenario_id}: turns")
        updates = 0
        noops = 0
        for old, new in zip(v1_turns, v2_turns, strict=True):
            require(old["turn_id"] == new["turn_id"], f"{scenario_id}: turn identity")
            require(old["speaker"] == new["speaker"], f"{scenario_id}: speaker drift")
            require(old.get("update") == new.get("update"), f"{scenario_id}: gold update drift")
            for key in ("dataset", "record_id", "record_turn_index", "origin", "original_text"):
                require(
                    old["source_trace"][key] == new["source_trace"][key],
                    f"{scenario_id}: source trace drift at {new['turn_id']} ({key})",
                )
            if "update" in new:
                updates += 1
                require(
                    new["text"] == update_text(people[new["speaker"]], new["update"]),
                    f"{scenario_id}: UPDATE rewrite mismatch at {new['turn_id']}",
                )
                require(new["text"].count("vehicle-profile") == 1, f"{scenario_id}: compound UPDATE")
                require(new["source_trace"]["reuse_mode"] == "source_inspired_single_intent_rewrite", f"{scenario_id}: UPDATE provenance")
            else:
                noops += 1
                if new["speaker"] == "assistant":
                    require(new["text"] == old["text"], f"{scenario_id}: assistant NO_OP drift")
                else:
                    expected = (
                        "This is a one-time request outside the vehicle, not a saved "
                        f"vehicle preference: {old['source_trace']['original_text'].strip()}"
                    )
                    require(new["text"] == expected, f"{scenario_id}: human NO_OP wrapper")
                    require(new["source_trace"]["reuse_mode"] == "explicit_nonvehicle_calibration_wrapper", f"{scenario_id}: NO_OP provenance")
        require((updates, noops) == (6, 24), f"{scenario_id}: decision counts")

        require(len(v1["quizzes"]) == len(v2["quizzes"]) == 16, f"{scenario_id}: quizzes")
        owners = {"driver": people["user"], "codriver": people["co_user"]}
        for old, new in zip(v1["quizzes"], v2["quizzes"], strict=True):
            require(old["quiz_id"] == new["quiz_id"], f"{scenario_id}: quiz identity")
            for key in ("quiz_type", "cutoff_turn_id", "reasoning_type", "evidence_slot_ids", "gold_calls"):
                require(old[key] == new[key], f"{scenario_id}: quiz gold drift ({key})")
            slot_id = new["evidence_slot_ids"][0]
            owner = owners[slot_id.split(".", 1)[0]]
            require(new["query"] == quiz_text(owner, slot_id, new["quiz_id"]), f"{scenario_id}: quiz rewrite")
            answer_values = [
                value
                for key, value in new["gold_calls"][0]["arguments"].items()
                if key not in {"zone", "seat"}
            ]
            require(all(str(value).lower() not in new["query"].lower() for value in answer_values), f"{scenario_id}: answer leak")

        manifest = json.loads((derived_root / scenario_id / "manifest.json").read_text(encoding="utf-8"))
        require(set(manifest["validation"].values()) == {"PASS"}, f"{scenario_id}: builder")
        expected_rows = {"dialogue.jsonl": 30, "patch.jsonl": 30, "memory_snapshots.jsonl": 30, "turn_quiz.jsonl": 8, "final_quiz.jsonl": 8}
        for filename, count in expected_rows.items():
            require(len(load_jsonl(derived_root / scenario_id / filename)) == count, f"{scenario_id}: {filename}")
        reports.append({"scenario_id": scenario_id, "status": "PASS"})

    result = {
        "schema_version": "vehiclemembench-easy-explicit-v2-validation-v1",
        "scenario_count": args.count,
        "gold_equivalence_to_v1": "PASS",
        "single_intent_updates": "PASS",
        "explicit_nonvehicle_noops": "PASS",
        "quiz_answer_non_leakage": "PASS",
        "derived_v2_artifacts": "PASS",
        "scenarios": reports,
    }
    output = v2_root / "CORPUS_VALIDATION.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
