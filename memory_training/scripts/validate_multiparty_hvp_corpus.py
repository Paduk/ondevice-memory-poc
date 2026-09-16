"""Validate cross-scenario invariants for completed multi-participant HVP data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from memory_training.scripts.build_external_adapted_pilots import load_external_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "evaluation/human-authored-vehicle-memory/multiparty-quiz-expansion-v1"
        ),
    )
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path(
            "memory_training/data/human-authored-v2-multiparty-quiz-expansion-v1"
        ),
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(
            "/mnt/data/hj153lee/PalmClaw/external-sources/"
            "human-derived-vehicle-dialogue"
        ),
    )
    parser.add_argument("--expected-scenarios", type=int, default=20)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalized_query(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def owner_for_slot(slot_id: str) -> str:
    if slot_id.startswith("driver."):
        return "user"
    if slot_id.startswith("codriver."):
        return "co_user"
    return "unknown"


def add_statistics(total: Counter[str], manifest: dict[str, Any]) -> None:
    stats = manifest["statistics"]
    for source_key, target_key in (
        ("turn_count", "turns"),
        ("update_count", "updates"),
        ("noop_count", "noops"),
        ("turn_quiz_count", "turn_quizzes"),
        ("final_quiz_count", "final_quizzes"),
        ("quiz_tool_call_count", "quiz_calls"),
    ):
        total[target_key] += int(stats[source_key])


def validate(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve(strict=True)
    derived_root = args.derived_root.resolve(strict=True)
    _, audio_tiers = load_external_records(args.external_root.resolve(strict=True))
    source_paths = sorted(source_root.glob("HVP*/scenario-source.json"))
    require(
        len(source_paths) == args.expected_scenarios,
        f"expected {args.expected_scenarios} completed scenarios, found {len(source_paths)}",
    )

    inserted_record_owner: dict[str, str] = {}
    inserted_session_ids: set[str] = set()
    added_query_owner: dict[str, str] = {}
    base_query_owner: dict[str, str] = {}
    added_distribution: Counter[str] = Counter()
    statistics: Counter[str] = Counter()
    owner_sensitive_count = 0
    preference_count = 0
    source_trace_turn_count = 0
    source_reuse_mode_counts: Counter[str] = Counter()
    pilot_excluded_row_count = 0
    scenario_rows = []

    for source_path in source_paths:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        scenario_id = str(source["scenario_id"])
        extension = source["multi_participant_extension"]
        require(
            extension["human_review_status"] == "PENDING",
            f"{scenario_id} must not claim independent human review",
        )
        review_path = source_path.parent / "AUTHOR_REVIEW.md"
        require(
            review_path.is_file() and review_path.stat().st_size > 0,
            f"missing {review_path}",
        )

        report_path = source_path.parent / "deterministic-validation.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(
            report["status"] == "PASS", f"{scenario_id} deterministic report failed"
        )
        require(
            all(value == "PASS" for value in report["checks"].values()),
            f"{scenario_id} contains a failed deterministic check",
        )

        manifest_path = derived_root / scenario_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(
            all(value == "PASS" for value in manifest["validation"].values()),
            f"{scenario_id} official builder validation failed",
        )
        add_statistics(statistics, manifest)
        patch_rows = load_jsonl(derived_root / scenario_id / "patch.jsonl")
        quiz_rows = [
            *load_jsonl(derived_root / scenario_id / "turn_quiz.jsonl"),
            *load_jsonl(derived_root / scenario_id / "final_quiz.jsonl"),
        ]
        require(
            all(
                row["split"] == "pilot_excluded" and row["train_eligible"] is False
                for row in patch_rows
            ),
            f"{scenario_id} contains a train-eligible Patch row",
        )
        require(
            all(row["split"] == "pilot_excluded" for row in quiz_rows),
            f"{scenario_id} contains a non-excluded Quiz row",
        )
        pilot_excluded_row_count += len(patch_rows) + len(quiz_rows)

        for record_id in extension["inserted_audio2tool_records"]:
            record_id = str(record_id)
            require(
                record_id not in inserted_record_owner,
                f"source record {record_id} reused by {inserted_record_owner.get(record_id)} and {scenario_id}",
            )
            inserted_record_owner[record_id] = scenario_id

        base = json.loads(
            Path(extension["base_source_path"]).read_text(encoding="utf-8")
        )
        for base_quiz in base["quizzes"]:
            base_query_owner.setdefault(
                normalized_query(str(base_quiz["query"])),
                str(base_quiz["quiz_id"]),
            )
        base_ids = {row["quiz_id"] for row in base["quizzes"]}
        base_session_ids = {row["session_id"] for row in base["sessions"]}
        added_sessions = [
            row
            for row in source["sessions"]
            if row["session_id"] not in base_session_ids
        ]
        require(
            len(added_sessions) == extension["inserted_session_count"],
            f"{scenario_id} added-session count differs from extension metadata",
        )
        for session in added_sessions:
            session_id = str(session["session_id"])
            require(
                session_id not in inserted_session_ids,
                f"inserted session ID is duplicated: {session_id}",
            )
            inserted_session_ids.add(session_id)
            record_ids = {
                str(turn["source_trace"]["record_id"]) for turn in session["turns"]
            }
            require(
                len(record_ids) == 1, f"{session['session_id']} mixes source records"
            )
            record_id = next(iter(record_ids))
            require(
                record_id in extension["inserted_audio2tool_records"],
                f"{session['session_id']} traces an undeclared source record",
            )
            tier, numeric_id = record_id.split(":", 1)
            record = audio_tiers[tier][int(numeric_id)]
            if tier == "tier7":
                original_turns = [
                    str(turn["content"]).strip() for turn in record["chat_history"]
                ]
            else:
                original_turns = [str(record["query"]).strip()]
            require(
                len(original_turns) == len(session["turns"]),
                f"{session['session_id']} turn count differs from source record",
            )
            update_count = 0
            for index, (turn, original_text) in enumerate(
                zip(session["turns"], original_turns, strict=True)
            ):
                trace = turn["source_trace"]
                reuse_mode = str(trace["reuse_mode"])
                source_reuse_mode_counts[reuse_mode] += 1
                require(
                    int(trace["record_turn_index"]) == index
                    and trace["original_text"] == original_text,
                    f"{session['session_id']}:{index} does not match source text",
                )
                if "update" in turn:
                    update_count += 1
                    if reuse_mode == "minimal_persistence_adaptation":
                        require(
                            turn["text"].startswith(original_text)
                            and turn["text"] != original_text,
                            f"{session['session_id']} update does not append a "
                            "persistence edit",
                        )
                    elif reuse_mode == "vehicle_domain_rewrite_with_persistence":
                        require(
                            turn["text"] != original_text,
                            f"{session['session_id']} declared rewrite is unchanged",
                        )
                    else:
                        raise ValueError(
                            f"{session['session_id']} has invalid update reuse mode: "
                            f"{reuse_mode}"
                        )
                else:
                    if reuse_mode == "verbatim_text_role_normalized":
                        require(
                            turn["text"] == original_text,
                            f"{session['session_id']} changed a non-update source turn",
                        )
                    elif reuse_mode == "vehicle_domain_rewrite":
                        require(
                            turn["text"] != original_text,
                            f"{session['session_id']} declared rewrite is unchanged",
                        )
                    else:
                        raise ValueError(
                            f"{session['session_id']} has invalid non-update reuse mode: "
                            f"{reuse_mode}"
                        )
                source_trace_turn_count += 1
            require(
                update_count == 1,
                f"{session['session_id']} must add exactly one update",
            )

        names = {
            row["short_id"]: row["speaker_name"].split()[0]
            for row in source["speakers"]
            if row["short_id"] in {"user", "co_user"}
        }
        added = [row for row in source["quizzes"] if row["quiz_id"] not in base_ids]
        for quiz in added:
            key = normalized_query(str(quiz["query"]))
            require(
                key not in added_query_owner,
                f"added query duplicated by {added_query_owner.get(key)} and {quiz['quiz_id']}",
            )
            added_query_owner[key] = str(quiz["quiz_id"])
            added_distribution[f"{quiz['quiz_type']}::{quiz['reasoning_type']}"] += 1

            if quiz["reasoning_type"] in {
                "coreference_resolution",
                "preference_conflict",
            }:
                owner_sensitive_count += 1
                owners = {
                    owner_for_slot(str(value)) for value in quiz["evidence_slot_ids"]
                }
                require(
                    len(owners) == 1 and "unknown" not in owners,
                    f"{quiz['quiz_id']} mixes evidence owners",
                )
                owner = next(iter(owners))
                require(
                    names[owner].casefold() in str(quiz["query"]).casefold(),
                    f"{quiz['quiz_id']} does not name its Gold evidence owner",
                )
            if quiz["reasoning_type"] == "preference_conflict":
                preference_count += 1
                check_key = f"preference_conflict_action_difference::{quiz['quiz_id']}"
                require(
                    report["checks"].get(check_key) == "PASS",
                    f"{quiz['quiz_id']} is not a genuine action conflict",
                )

        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "added_quizzes": len(added),
                "inserted_records": len(extension["inserted_audio2tool_records"]),
                "status": "PASS",
            }
        )

    queue_path = source_root / "HUMAN_REVIEW_QUEUE.md"
    require(queue_path.is_file(), f"missing {queue_path}")
    queued_session_ids = set(
        re.findall(r"`(hvp\d{2}-mp\d{2})`", queue_path.read_text(encoding="utf-8"))
    )
    require(
        queued_session_ids == inserted_session_ids,
        "human-review queue does not exactly cover inserted sessions: "
        f"missing={sorted(inserted_session_ids - queued_session_ids)}, "
        f"extra={sorted(queued_session_ids - inserted_session_ids)}",
    )
    added_base_query_overlap = set(added_query_owner) & set(base_query_owner)
    require(
        not added_base_query_overlap,
        "added Quiz queries duplicate frozen-base queries: "
        f"{sorted(added_base_query_overlap)}",
    )

    return {
        "schema_version": "vehiclemembench-hvp-multiparty-corpus-validation-v1",
        "status": "PASS",
        "scenario_count": len(source_paths),
        "scenarios": scenario_rows,
        "statistics": dict(sorted(statistics.items())),
        "added_quiz_count": sum(added_distribution.values()),
        "added_query_unique_count": len(added_query_owner),
        "added_query_base_collision_count": len(added_base_query_overlap),
        "added_quiz_distribution": dict(sorted(added_distribution.items())),
        "inserted_record_count": len(inserted_record_owner),
        "human_review_queue_session_count": len(queued_session_ids),
        "source_trace_turn_count": source_trace_turn_count,
        "source_reuse_mode_counts": dict(sorted(source_reuse_mode_counts.items())),
        "owner_sensitive_quiz_count": owner_sensitive_count,
        "genuine_preference_conflict_count": preference_count,
        "pilot_excluded_patch_and_quiz_row_count": pilot_excluded_row_count,
        "independent_human_review_status": "PENDING",
    }


def main() -> None:
    args = parse_args()
    report = validate(args)
    output = args.source_root / "CORPUS_VALIDATION.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
