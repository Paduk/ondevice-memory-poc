"""Deterministically validate one versioned multi-participant HVP extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="HVP01")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--expanded-source-root",
        type=Path,
        default=Path(
            "evaluation/human-authored-vehicle-memory/"
            "multiparty-quiz-expansion-v1"
        ),
    )
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path(
            "memory_training/data/"
            "human-authored-v2-multiparty-quiz-expansion-v1"
        ),
    )
    parser.add_argument(
        "--quota-plan",
        type=Path,
        default=Path(
            "memory_training/data/human-authored-v2-quiz-expansion-v1/"
            "quota-plan.json"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def base_source_path(repository_root: Path, scenario_id: str) -> Path:
    index = int(scenario_id[-2:])
    directory = f"pilot-hv{index:02d}" if index <= 4 else f"pilot-hvp{index:02d}"
    return (
        repository_root
        / "evaluation"
        / "human-authored-vehicle-memory"
        / directory
        / "scenario-source.json"
    )


def normalized_query(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def memory_owners(memory: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    line_owner: dict[str, str] = {}
    owner_lines: dict[str, list[str]] = {}
    owner = ""
    for raw_line in memory.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            owner = line[4:]
            owner_lines.setdefault(owner, [])
        elif line.startswith("- ["):
            require(bool(owner), "memory fact appeared before an owner heading")
            line_owner[line] = owner
            owner_lines[owner].append(line)
    return line_owner, owner_lines


def tool_from_memory_line(line: str) -> str:
    match = re.search(r"\] ([^.]+)\.", line)
    require(match is not None, f"cannot parse Tool from memory line: {line}")
    return match.group(1)


def action_signature_from_memory_line(line: str) -> tuple[str, str, str, str]:
    """Return Tool, argument, value, and target scope, excluding owner/condition."""
    match = re.match(
        r"^- \[[^]]+\] ([^.]+)\.([^;]+); value=([^;]+)"
        r"(?:; context=\(([^)]*)\))?(?:; condition=.*)?$",
        line,
    )
    require(match is not None, f"cannot parse action signature from memory line: {line}")
    return match.group(1), match.group(2), match.group(3), match.group(4) or ""


def validate(args: argparse.Namespace) -> dict[str, Any]:
    scenario_id = args.scenario
    repository_root = args.repository_root.resolve(strict=True)
    base_path = base_source_path(repository_root, scenario_id).resolve(strict=True)
    expanded_path = (args.expanded_source_root / scenario_id / "scenario-source.json").resolve(
        strict=True
    )
    derived = (args.derived_root / scenario_id).resolve(strict=True)
    quota_path = args.quota_plan.resolve(strict=True)

    base = json.loads(base_path.read_text(encoding="utf-8"))
    expanded = json.loads(expanded_path.read_text(encoding="utf-8"))
    manifest = json.loads((derived / "manifest.json").read_text(encoding="utf-8"))
    quota = json.loads(quota_path.read_text(encoding="utf-8"))
    extension = expanded["multi_participant_extension"]

    checks: dict[str, str] = {}

    def check(name: str, condition: bool, failure: str) -> None:
        require(condition, failure)
        checks[name] = "PASS"

    check(
        "base_source_hash",
        extension["base_source_sha256"] == sha256(base_path),
        "base source hash changed",
    )
    base_sessions = {row["session_id"]: row for row in base["sessions"]}
    expanded_sessions = {row["session_id"]: row for row in expanded["sessions"]}
    check(
        "base_sessions_unchanged",
        all(expanded_sessions.get(key) == value for key, value in base_sessions.items()),
        "an original session was modified",
    )
    base_quizzes = {row["quiz_id"]: row for row in base["quizzes"]}
    expanded_quizzes = {row["quiz_id"]: row for row in expanded["quizzes"]}
    check(
        "base_quizzes_unchanged",
        all(expanded_quizzes.get(key) == value for key, value in base_quizzes.items()),
        "an original quiz was modified",
    )

    added_quizzes = [
        row for key, row in expanded_quizzes.items() if key not in base_quizzes
    ]
    quota_row = next(
        row
        for row in quota["per_scenario_addition_quota"]
        if row["scenario_id"] == scenario_id
    )
    actual_cross = Counter(
        (row["quiz_type"].lower(), row["reasoning_type"]) for row in added_quizzes
    )
    expected_cross = Counter(
        {
            (quiz_type, reasoning): count
            for quiz_type in ("turn", "final")
            for reasoning, count in quota_row[quiz_type].items()
        }
    )
    check(
        "scenario_quota",
        actual_cross == expected_cross and len(added_quizzes) == quota_row["total"],
        f"quiz quota mismatch: expected {expected_cross}, got {actual_cross}",
    )

    speaker_by_short_id = {
        row["short_id"]: row["speaker_id"] for row in expanded["speakers"]
    }
    human_speakers = {
        row["speaker_id"]
        for row in expanded["speakers"]
        if row["short_id"] != "assistant"
    }
    update_owners: Counter[str] = Counter()
    for session in expanded["sessions"]:
        for turn in session["turns"]:
            if "update" not in turn:
                continue
            owner = turn["update"].get("owner", speaker_by_short_id[turn["speaker"]])
            update_owners[owner] += 1
    check(
        "multiple_memory_owners",
        len(set(update_owners) & human_speakers) >= 2,
        f"fewer than two human memory owners: {update_owners}",
    )

    original_record_ids: set[str] = set()
    original_root = repository_root / "evaluation" / "human-authored-vehicle-memory"
    for path in original_root.glob("pilot-hv*/scenario-source.json"):
        source = json.loads(path.read_text(encoding="utf-8"))
        for catalog in source.get("external_sources", []):
            if catalog.get("dataset") == "Audio2Tool":
                original_record_ids.update(str(value) for value in catalog["records"])
    other_extension_record_ids: set[str] = set()
    expanded_root = args.expanded_source_root.resolve()
    for path in expanded_root.glob("HVP*/scenario-source.json"):
        if path.resolve() == expanded_path:
            continue
        source = json.loads(path.read_text(encoding="utf-8"))
        other_extension_record_ids.update(
            str(value)
            for value in source.get("multi_participant_extension", {}).get(
                "inserted_audio2tool_records", []
            )
        )
    inserted_records = extension["inserted_audio2tool_records"]
    check(
        "inserted_source_records_disjoint",
        not (
            set(inserted_records)
            & (original_record_ids | other_extension_record_ids)
        )
        and len(inserted_records) == len(set(inserted_records)),
        "inserted Audio2Tool records overlap an original or another extension",
    )

    check(
        "official_builder_validation",
        all(value == "PASS" for value in manifest["validation"].values()),
        "official builder validation is not all PASS",
    )
    check(
        "derived_source_hash",
        manifest["source_sha256"] == sha256(expanded_path),
        "derived manifest does not match expanded source",
    )
    check(
        "added_query_uniqueness",
        len({normalized_query(row["query"]) for row in added_quizzes})
        == len(added_quizzes)
        and not (
            {normalized_query(row["query"]) for row in added_quizzes}
            & {normalized_query(row["query"]) for row in base["quizzes"]}
        ),
        "an added query is duplicated within the addition or matches a baseline query",
    )

    derived_quizzes = {
        row["quiz_id"]: row
        for row in [
            *load_jsonl(derived / "turn_quiz.jsonl"),
            *load_jsonl(derived / "final_quiz.jsonl"),
        ]
    }
    snapshots = {
        row["global_turn_index"]: row
        for row in load_jsonl(derived / "memory_snapshots.jsonl")
    }
    conflict_audit = []
    for source_quiz in added_quizzes:
        if source_quiz["reasoning_type"] != "preference_conflict":
            continue
        row = derived_quizzes[source_quiz["quiz_id"]]
        memory = snapshots[row["memory_ref"]["global_turn_index"]]["memory"]
        line_owner, owner_lines = memory_owners(memory)
        evidence_lines = row["evidence"].get("memory_evidence_lines") or row["evidence"][
            "gold_memory"
        ].splitlines()
        alternatives = []
        different_action_alternatives = []
        for evidence in evidence_lines:
            evidence_owner = line_owner[evidence]
            tool = tool_from_memory_line(evidence)
            evidence_signature = action_signature_from_memory_line(evidence)
            for other_owner, lines in owner_lines.items():
                if other_owner == evidence_owner:
                    continue
                same_tool = [line for line in lines if tool_from_memory_line(line) == tool]
                alternatives.extend(same_tool)
                different_action_alternatives.extend(
                    line
                    for line in same_tool
                    if action_signature_from_memory_line(line)[:2]
                    == evidence_signature[:2]
                    and action_signature_from_memory_line(line)[2:]
                    != evidence_signature[2:]
                )
        check_name = f"preference_conflict::{source_quiz['quiz_id']}"
        check(check_name, bool(alternatives), f"{source_quiz['quiz_id']} lacks another-owner Tool conflict")
        difference_check_name = (
            f"preference_conflict_action_difference::{source_quiz['quiz_id']}"
        )
        check(
            difference_check_name,
            bool(different_action_alternatives),
            f"{source_quiz['quiz_id']} has no different value or target scope for another owner",
        )
        conflict_audit.append(
            {
                "quiz_id": source_quiz["quiz_id"],
                "gold_evidence": evidence_lines,
                "other_owner_alternatives": sorted(set(alternatives)),
                "other_owner_different_actions": sorted(
                    set(different_action_alternatives)
                ),
            }
        )

    added_sessions = [
        row for key, row in expanded_sessions.items() if key not in base_sessions
    ]
    added_update_count = sum(
        "update" in turn for session in added_sessions for turn in session["turns"]
    )
    check(
        "extension_counts",
        len(added_sessions) == extension["inserted_session_count"]
        and sum(len(row["turns"]) for row in added_sessions)
        == extension["inserted_turn_count"]
        and added_update_count == extension["inserted_update_count"],
        "extension metadata counts differ from source",
    )

    return {
        "schema_version": "vehiclemembench-hvp-multiparty-validation-v1",
        "scenario_id": scenario_id,
        "status": "PASS",
        "scope": "deterministic validation; semantic review remains separate",
        "checks": checks,
        "statistics": manifest["statistics"],
        "update_owners": dict(sorted(update_owners.items())),
        "added_quiz_distribution": {
            f"{quiz_type.upper()}::{reasoning}": count
            for (quiz_type, reasoning), count in sorted(actual_cross.items())
        },
        "preference_conflict_audit": conflict_audit,
        "semantic_review_status": "PENDING",
    }


def main() -> None:
    args = parse_args()
    report = validate(args)
    output = args.expanded_source_root / args.scenario / "deterministic-validation.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
