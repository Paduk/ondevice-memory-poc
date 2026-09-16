"""Deterministically audit the Easy explicit HVP pilot corpus."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from memory_training.scripts.build_easy_hvp_pilots import (
    AMBIGUOUS_UPDATE_SOURCE_MARKERS,
    ENVISIONED_REL,
    LOW_QUALITY_SOURCE_MARKERS,
    TIER5_REL,
    TIER7_REL,
    human_turns,
    normalized,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(
            "/mnt/data/hj153lee/PalmClaw/external-sources/"
            "human-derived-vehicle-dialogue"
        ),
    )
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


def prior_source_records(repository_root: Path) -> set[tuple[str, str]]:
    root = repository_root / "evaluation/human-authored-vehicle-memory"
    paths = list(root.glob("pilot-hv*/scenario-source.json"))
    paths += list(root.glob("multiparty-quiz-expansion-v1/HVP*/scenario-source.json"))
    result: set[tuple[str, str]] = set()
    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        for entry in source.get("external_sources", []):
            for record_id in entry.get("records", []):
                result.add((entry["dataset"], str(record_id)))
        for record_id in source.get("multi_participant_extension", {}).get(
            "inserted_audio2tool_records", []
        ):
            result.add(("Audio2Tool", str(record_id)))
    return result


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    external_root = args.external_root.resolve(strict=True)
    source_root = repository_root / "evaluation/human-authored-vehicle-memory/easy-explicit-v1"
    derived_root = repository_root / "memory_training/data/human-authored-v2-easy-explicit-v1"

    with (external_root / ENVISIONED_REL).open(encoding="utf-8-sig", newline="") as handle:
        envisioned = {str(row["id"]): row for row in csv.DictReader(handle)}
    tier7 = {
        str(row["id"]): row
        for row in json.loads((external_root / TIER7_REL).read_text(encoding="utf-8"))
    }
    tier5 = {
        str(row["id"]): row
        for row in json.loads((external_root / TIER5_REL).read_text(encoding="utf-8"))
    }

    prior = prior_source_records(repository_root)
    corpus_records: set[tuple[str, str]] = set()
    corpus_queries: set[tuple[str, str]] = set()
    scenario_reports = []

    for number in range(1, args.count + 1):
        scenario_id = f"HVE{number:02d}"
        source_path = source_root / scenario_id / "scenario-source.json"
        derived = derived_root / scenario_id
        require(source_path.exists(), f"missing {source_path}")
        require((derived / "manifest.json").exists(), f"missing derived manifest for {scenario_id}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        manifest = json.loads((derived / "manifest.json").read_text(encoding="utf-8"))

        require(source["scenario_id"] == scenario_id, f"{scenario_id}: ID mismatch")
        require(source["difficulty_stratum"] == "easy_explicit_v1", f"{scenario_id}: stratum")
        require(source["authorship"]["eligible_for_human_test"] is False, f"{scenario_id}: eligibility")
        require(len(source["sessions"]) == 12, f"{scenario_id}: expected 12 sessions")
        require(len(source["speakers"]) == 3, f"{scenario_id}: expected 3 speakers")

        people = {entry["short_id"]: entry for entry in source["speakers"]}
        human_names = {people["user"]["speaker_name"], people["co_user"]["speaker_name"]}
        turns = [turn for session in source["sessions"] for turn in session["turns"]]
        updates = [turn for turn in turns if "update" in turn]
        require(len(turns) == 30, f"{scenario_id}: expected 30 turns")
        require(len(updates) == 6, f"{scenario_id}: expected 6 updates")
        require(len(turns) - len(updates) == 24, f"{scenario_id}: expected 24 NO_OP turns")
        require(Counter(turn["update"]["op"] for turn in updates) == {"add": 4, "replace": 2}, f"{scenario_id}: operation mix")
        require(Counter(turn["speaker"] for turn in updates) == {"user": 3, "co_user": 3}, f"{scenario_id}: owner balance")

        active: dict[str, Any] = {}
        update_cutoffs: set[str] = set()
        for turn in turns:
            trace = turn["source_trace"]
            record = (trace["dataset"], str(trace["record_id"]))
            require(record not in prior, f"{scenario_id}: record reused from prior HVP: {record}")
            corpus_records.add(record)
            index = int(trace["record_turn_index"])
            if trace["dataset"] == "Envisioned Voice Assistant Dialogues":
                row_id, field = str(trace["record_id"]).split(":", 1)
                source_turns = human_turns(envisioned[row_id][field])
                expected_role, expected_text = source_turns[index]
                require(turn["text"] == expected_text, f"{scenario_id}: Envisioned text drift")
                require(
                    turn["speaker"] == ("assistant" if expected_role == "assistant" else turn["speaker"]),
                    f"{scenario_id}: Envisioned role drift",
                )
                require(not LOW_QUALITY_SOURCE_MARKERS.search(turn["text"]), f"{scenario_id}: low-quality source marker")
            elif str(trace["record_id"]).startswith("tier7:"):
                row = tier7[str(trace["record_id"]).split(":", 1)[1]]
                expected = str(row["chat_history"][index]["content"]).strip()
                require(turn["text"] == expected, f"{scenario_id}: Tier-7 text drift")
            else:
                row = tier5[str(trace["record_id"]).split(":", 1)[1]]
                original = str(row["query"]).strip()
                require(trace["original_text"] == original, f"{scenario_id}: Tier-5 trace drift")
                require(turn["text"].startswith(original), f"{scenario_id}: adapted source prefix drift")
                require(not AMBIGUOUS_UPDATE_SOURCE_MARKERS.search(original), f"{scenario_id}: ambiguous update source")

            if "update" not in turn:
                continue
            update = turn["update"]
            name = people[turn["speaker"]]["speaker_name"]
            require(name in human_names and name in turn["text"], f"{scenario_id}: update owner not explicit")
            require("permanent vehicle profile" in turn["text"], f"{scenario_id}: persistence not explicit")
            slot_id = update["slot_id"]
            if update["op"] == "add":
                require(slot_id not in active, f"{scenario_id}: ADD overwrites active slot")
            else:
                require(slot_id in active, f"{scenario_id}: REPLACE has no prior slot")
                require("replace my earlier saved setting" in turn["text"], f"{scenario_id}: replacement not explicit")
            active[slot_id] = update["value"]
            update_cutoffs.add(turn["turn_id"])

        catalog_records = {
            (entry["dataset"], str(record_id))
            for entry in source["external_sources"]
            for record_id in entry["records"]
        }
        require(catalog_records <= corpus_records, f"{scenario_id}: source catalog not represented")
        require(len(catalog_records) == 12, f"{scenario_id}: expected 12 distinct source records")

        quizzes = source["quizzes"]
        require(Counter(quiz["quiz_type"] for quiz in quizzes) == {"TURN": 8, "FINAL": 8}, f"{scenario_id}: quiz mix")
        final_cutoff = turns[-1]["turn_id"]
        for quiz in quizzes:
            require(len(quiz["evidence_slot_ids"]) == 1, f"{scenario_id}: multi-fact quiz")
            require(len(quiz["gold_calls"]) == 1, f"{scenario_id}: multi-tool quiz")
            require(any(name in quiz["query"] for name in human_names), f"{scenario_id}: quiz owner absent")
            query_signature = normalized(quiz["query"])
            query_key = (quiz["cutoff_turn_id"], query_signature)
            require(query_key not in corpus_queries, f"{scenario_id}: duplicate quiz at one cutoff")
            corpus_queries.add(query_key)
            if quiz["quiz_type"] == "FINAL":
                require(quiz["cutoff_turn_id"] == final_cutoff, f"{scenario_id}: final cutoff")
            else:
                require(quiz["cutoff_turn_id"] in update_cutoffs, f"{scenario_id}: turn cutoff")
            arguments = quiz["gold_calls"][0]["arguments"]
            answer_value = next(
                value for key, value in arguments.items() if key not in {"zone", "seat"}
            )
            require(
                re.search(rf"\b{re.escape(str(answer_value).lower())}\b", quiz["query"].lower()) is None,
                f"{scenario_id}: answer leaked in quiz",
            )

        require(set(manifest["validation"].values()) == {"PASS"}, f"{scenario_id}: builder validation")
        expected_rows = {
            "dialogue.jsonl": 30,
            "patch.jsonl": 30,
            "memory_snapshots.jsonl": 30,
            "turn_quiz.jsonl": 8,
            "final_quiz.jsonl": 8,
        }
        for filename, expected in expected_rows.items():
            require(len(load_jsonl(derived / filename)) == expected, f"{scenario_id}: {filename} rows")

        scenario_reports.append(
            {
                "scenario_id": scenario_id,
                "turns": 30,
                "updates": 6,
                "noops": 24,
                "adds": 4,
                "replaces": 2,
                "turn_quizzes": 8,
                "final_quizzes": 8,
                "source_records": 12,
                "builder_checks": len(manifest["validation"]),
                "status": "PASS",
            }
        )

    require(len(corpus_records) == args.count * 12, "external records overlap across Easy scenarios")
    report = {
        "schema_version": "vehiclemembench-easy-explicit-corpus-validation-v1",
        "scenario_count": args.count,
        "aggregate": {
            "turns": args.count * 30,
            "updates": args.count * 6,
            "noops": args.count * 24,
            "adds": args.count * 4,
            "replaces": args.count * 2,
            "quizzes": args.count * 16,
            "distinct_external_records": len(corpus_records),
        },
        "scenarios": scenario_reports,
        "validation": {
            "easy_contract": "PASS",
            "external_source_exactness": "PASS",
            "prior_hvp_record_disjointness": "PASS",
            "cross_scenario_record_disjointness": "PASS",
            "update_state_machine": "PASS",
            "owner_and_persistence_explicitness": "PASS",
            "quiz_answer_non_leakage": "PASS",
            "derived_v2_artifacts": "PASS",
        },
    }
    report_path = source_root / "CORPUS_VALIDATION.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
