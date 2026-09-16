"""Validate natural-explicit HVE against its Easy-explicit v1 Gold parent."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PERSISTENCE_MARKERS = re.compile(
    r"\b(?:remember|remembered|save|usual|regular|normally|from now on|future|next time|profile|preference)\b",
    re.IGNORECASE,
)
REPLACEMENT_MARKERS = re.compile(
    r"\b(?:replace|changed|change|revise|old|previous|earlier|no longer|instead)\b",
    re.IGNORECASE,
)
BANNED_TEMPLATE_MARKERS = (
    "please save this exact setting to my permanent vehicle profile",
    "please replace my earlier saved setting in my permanent vehicle profile",
)


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


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    args = parse_args()
    repo = args.repository_root.resolve(strict=True)
    parent_root = repo / "evaluation/human-authored-vehicle-memory/easy-explicit-v1"
    source_root = repo / "evaluation/human-authored-vehicle-memory/easy-natural-explicit-v1"
    derived_root = repo / "memory_training/data/human-authored-v2-easy-natural-explicit-v1"
    reports = []
    all_clauses: set[str] = set()

    for number in range(1, args.count + 1):
        scenario_id = f"HVE{number:02d}"
        parent = load(parent_root / scenario_id / "scenario-source.json")
        source = load(source_root / scenario_id / "scenario-source.json")
        derived = derived_root / scenario_id
        manifest = load(derived / "manifest.json")

        require(source["scenario_id"] == scenario_id, f"{scenario_id}: ID")
        require(source["difficulty_stratum"] == "easy_natural_explicit_v1", f"{scenario_id}: stratum")
        require(source["authorship"]["eligible_for_human_test"] is False, f"{scenario_id}: eligibility")
        require(source["external_sources"] == parent["external_sources"], f"{scenario_id}: provenance drift")

        parent_speakers = {
            row["short_id"]: (row["speaker_id"], row["speaker_name"], row["role"])
            for row in parent["speakers"]
        }
        source_speakers = {
            row["short_id"]: (row["speaker_id"], row["speaker_name"], row["role"])
            for row in source["speakers"]
        }
        require(source_speakers == parent_speakers, f"{scenario_id}: speaker drift")
        names = {row["short_id"]: row["speaker_name"] for row in source["speakers"]}

        parent_sessions = parent["sessions"]
        source_sessions = source["sessions"]
        require(len(source_sessions) == len(parent_sessions) == 12, f"{scenario_id}: sessions")
        update_count = 0
        noop_count = 0
        for parent_session, session in zip(parent_sessions, source_sessions):
            require(session["session_id"] == parent_session["session_id"], f"{scenario_id}: session ID")
            require(session["timestamp"] == parent_session["timestamp"], f"{scenario_id}: timestamp")
            require(len(session["turns"]) == len(parent_session["turns"]), f"{scenario_id}: session turns")
            for old_turn, turn in zip(parent_session["turns"], session["turns"]):
                require(turn["turn_id"] == old_turn["turn_id"], f"{scenario_id}: turn ID")
                require(turn["speaker"] == old_turn["speaker"], f"{scenario_id}: speaker")
                require(turn.get("source_trace") == old_turn.get("source_trace"), f"{scenario_id}: source trace")
                require(turn.get("update") == old_turn.get("update"), f"{scenario_id}: Gold update drift")
                update = turn.get("update")
                if update is None:
                    noop_count += 1
                    require(turn["text"] == old_turn["text"], f"{scenario_id}: NO_OP text drift")
                    continue

                update_count += 1
                original = str(turn["source_trace"]["original_text"]).strip()
                require(turn["text"].startswith(original), f"{scenario_id}: source prefix drift")
                clause = turn["text"][len(original):].lstrip(" .")
                require(clause and clause not in all_clauses, f"{scenario_id}: duplicate/empty natural clause")
                all_clauses.add(clause)
                owner_name = names[turn["speaker"]]
                require(owner_name in clause, f"{scenario_id}: owner absent")
                require(str(update["value"]).lower() in clause.lower(), f"{scenario_id}: value absent")
                require(PERSISTENCE_MARKERS.search(clause) is not None, f"{scenario_id}: persistence unclear")
                require(
                    not any(marker in clause.lower() for marker in BANNED_TEMPLATE_MARKERS),
                    f"{scenario_id}: old rigid template remains",
                )
                if update["op"] == "replace":
                    require(REPLACEMENT_MARKERS.search(clause) is not None, f"{scenario_id}: replacement unclear")

        require((update_count, noop_count) == (6, 24), f"{scenario_id}: decision mix")

        require(len(source["quizzes"]) == len(parent["quizzes"]) == 16, f"{scenario_id}: quizzes")
        for old_quiz, quiz in zip(parent["quizzes"], source["quizzes"]):
            for key in old_quiz:
                if key != "query":
                    require(quiz[key] == old_quiz[key], f"{scenario_id}: Quiz Gold drift: {key}")
            require(quiz["query"] != old_quiz["query"], f"{scenario_id}: Quiz not naturalized")
            owner_names = [name for short, name in names.items() if short != "assistant"]
            require(sum(name in quiz["query"] for name in owner_names) == 1, f"{scenario_id}: Quiz owner")
            arguments = quiz["gold_calls"][0]["arguments"]
            answer = next(value for key, value in arguments.items() if key not in {"zone", "seat"})
            require(
                re.search(rf"\b{re.escape(str(answer).lower())}\b", quiz["query"].lower()) is None,
                f"{scenario_id}: Quiz answer leak",
            )

        require(set(manifest["validation"].values()) == {"PASS"}, f"{scenario_id}: builder")
        expected = {
            "dialogue.jsonl": 30,
            "patch.jsonl": 30,
            "memory_snapshots.jsonl": 30,
            "turn_quiz.jsonl": 8,
            "final_quiz.jsonl": 8,
        }
        for filename, count in expected.items():
            require(len(rows(derived / filename)) == count, f"{scenario_id}: {filename}")

        reports.append({
            "scenario_id": scenario_id,
            "updates": update_count,
            "noops": noop_count,
            "quizzes": len(source["quizzes"]),
            "gold_equivalence": "PASS",
            "natural_explicitness": "PASS",
            "builder_validation": "PASS",
            "status": "PASS",
        })

    require(len(all_clauses) == args.count * 6, "natural clauses are not globally unique")
    report = {
        "schema_version": "vehiclemembench-easy-natural-explicit-validation-v1",
        "scenario_count": args.count,
        "aggregate": {
            "turns": args.count * 30,
            "updates": args.count * 6,
            "noops": args.count * 24,
            "quizzes": args.count * 16,
            "unique_natural_update_clauses": len(all_clauses),
        },
        "validation": {
            "gold_equivalence_to_easy_explicit_v1": "PASS",
            "source_and_noop_preservation": "PASS",
            "owner_value_persistence_explicitness": "PASS",
            "replacement_explicitness": "PASS",
            "rigid_template_removal": "PASS",
            "quiz_answer_non_leakage": "PASS",
            "derived_patch_and_tool_validation": "PASS",
        },
        "scenarios": reports,
    }
    output = source_root / "CORPUS_VALIDATION.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
