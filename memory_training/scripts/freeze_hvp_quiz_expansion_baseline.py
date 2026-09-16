"""Freeze the immutable HVP01-HVP20 baseline before adding new quizzes.

The source scenarios are never modified.  This script records their exact file
hashes and the aggregate dialogue, memory, and quiz statistics that subsequent
quiz-expansion stages must preserve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCENARIO_FILES = (
    "dialogue.jsonl",
    "memory_snapshots.jsonl",
    "patch.jsonl",
    "turn_quiz.jsonl",
    "final_quiz.jsonl",
    "manifest.json",
)
CORE_FILES = ("dialogue.jsonl", "memory_snapshots.jsonl", "patch.jsonl")
QUIZ_FILES = ("turn_quiz.jsonl", "final_quiz.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("memory_training/data/human-authored-v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "memory_training/data/human-authored-v2-quiz-expansion-v1/"
            "baseline-freeze.json"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(items: Iterable[tuple[str, str]]) -> str:
    payload = "".join(f"{name}\0{digest}\n" for name, digest in sorted(items))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def file_record(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if row_count is not None:
        record["row_count"] = row_count
    return record


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def count_cross(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(
        f"{row['quiz_type']}::{row['input']['reasoning_type']}" for row in rows
    )


def freeze(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    expected_ids = [f"HVP{index:02d}" for index in range(1, 21)]
    scenario_records: list[dict[str, Any]] = []
    all_file_hashes: list[tuple[str, str]] = []
    core_hashes: list[tuple[str, str]] = []
    quiz_hashes: list[tuple[str, str]] = []
    quiz_ids: list[str] = []
    totals: Counter[str] = Counter()
    reasoning_types: Counter[str] = Counter()
    quiz_types: Counter[str] = Counter()
    quiz_type_reasoning: Counter[str] = Counter()
    tool_calls: Counter[str] = Counter()

    for offset, scenario_id in enumerate(expected_ids, start=901):
        scenario_root = source_root / scenario_id
        require(scenario_root.is_dir(), f"Missing scenario directory: {scenario_root}")
        for name in SCENARIO_FILES:
            require((scenario_root / name).is_file(), f"Missing {scenario_id}/{name}")

        manifest = json.loads((scenario_root / "manifest.json").read_text(encoding="utf-8"))
        require(manifest["scenario_id"] == scenario_id, f"{scenario_id}: manifest ID mismatch")
        require(manifest["scenario_index"] == offset, f"{scenario_id}: index mismatch")
        require(
            all(value == "PASS" for value in manifest["validation"].values()),
            f"{scenario_id}: source validation is not all PASS",
        )

        dialogue = load_jsonl(scenario_root / "dialogue.jsonl")
        snapshots = load_jsonl(scenario_root / "memory_snapshots.jsonl")
        patches = load_jsonl(scenario_root / "patch.jsonl")
        turn_quizzes = load_jsonl(scenario_root / "turn_quiz.jsonl")
        final_quizzes = load_jsonl(scenario_root / "final_quiz.jsonl")
        quizzes = turn_quizzes + final_quizzes

        require(len(dialogue) == 50, f"{scenario_id}: expected 50 dialogue turns")
        require(len(snapshots) == len(dialogue), f"{scenario_id}: snapshot count mismatch")
        require(len(patches) == len(dialogue), f"{scenario_id}: patch count mismatch")
        require(
            [row["global_turn_index"] for row in dialogue] == list(range(len(dialogue))),
            f"{scenario_id}: dialogue indices are not contiguous",
        )
        require(
            [row["turn_id"] for row in snapshots] == [row["turn_id"] for row in dialogue],
            f"{scenario_id}: snapshot/dialogue turn IDs differ",
        )
        require(
            [row["turn_id"] for row in patches] == [row["turn_id"] for row in dialogue],
            f"{scenario_id}: patch/dialogue turn IDs differ",
        )
        require(
            [row["decision"] for row in snapshots]
            == [row["target"]["decision"] for row in patches],
            f"{scenario_id}: snapshot/patch decisions differ",
        )
        require(
            all(row["quiz_type"] == "TURN" for row in turn_quizzes),
            f"{scenario_id}: non-TURN row in turn_quiz.jsonl",
        )
        require(
            all(row["quiz_type"] == "FINAL" for row in final_quizzes),
            f"{scenario_id}: non-FINAL row in final_quiz.jsonl",
        )
        require(
            all(row["scenario_index"] == offset for row in quizzes),
            f"{scenario_id}: quiz scenario index mismatch",
        )
        require(
            all(0 <= row["memory_ref"]["global_turn_index"] < 50 for row in quizzes),
            f"{scenario_id}: quiz checkpoint outside dialogue",
        )

        decisions = Counter(row["decision"] for row in snapshots)
        require(set(decisions) <= {"UPDATE", "NO_OP"}, f"{scenario_id}: unknown decision")
        local_quiz_ids = [row["quiz_id"] for row in quizzes]
        require(len(local_quiz_ids) == len(set(local_quiz_ids)), f"{scenario_id}: duplicate quiz ID")
        quiz_ids.extend(local_quiz_ids)

        file_records: dict[str, dict[str, Any]] = {}
        row_counts = {
            "dialogue.jsonl": len(dialogue),
            "memory_snapshots.jsonl": len(snapshots),
            "patch.jsonl": len(patches),
            "turn_quiz.jsonl": len(turn_quizzes),
            "final_quiz.jsonl": len(final_quizzes),
        }
        for name in SCENARIO_FILES:
            record = file_record(scenario_root / name, row_count=row_counts.get(name))
            file_records[name] = record
            relative_name = f"{scenario_id}/{name}"
            all_file_hashes.append((relative_name, record["sha256"]))
            if name in CORE_FILES:
                core_hashes.append((relative_name, record["sha256"]))
            if name in QUIZ_FILES:
                quiz_hashes.append((relative_name, record["sha256"]))

            source_record = manifest.get("files", {}).get(name)
            if source_record:
                require(
                    source_record["sha256"] == record["sha256"],
                    f"{scenario_id}/{name}: manifest hash mismatch",
                )
                require(
                    source_record["row_count"] == record["row_count"],
                    f"{scenario_id}/{name}: manifest row count mismatch",
                )

        local_reasoning = Counter(row["input"]["reasoning_type"] for row in quizzes)
        local_quiz_types = Counter(row["quiz_type"] for row in quizzes)
        local_cross = count_cross(quizzes)
        local_tools = Counter(
            call["name"] for row in quizzes for call in row["target"]["gold_calls"]
        )
        reasoning_types.update(local_reasoning)
        quiz_types.update(local_quiz_types)
        quiz_type_reasoning.update(local_cross)
        tool_calls.update(local_tools)
        totals.update(
            {
                "scenarios": 1,
                "dialogue_turns": len(dialogue),
                "memory_snapshots": len(snapshots),
                "patch_rows": len(patches),
                "updates": decisions["UPDATE"],
                "noops": decisions["NO_OP"],
                "turn_quizzes": len(turn_quizzes),
                "final_quizzes": len(final_quizzes),
                "quizzes": len(quizzes),
                "gold_calls": sum(len(row["target"]["gold_calls"]) for row in quizzes),
            }
        )

        scenario_records.append(
            {
                "scenario_id": scenario_id,
                "scenario_index": offset,
                "files": file_records,
                "counts": {
                    "dialogue_turns": len(dialogue),
                    "updates": decisions["UPDATE"],
                    "noops": decisions["NO_OP"],
                    "turn_quizzes": len(turn_quizzes),
                    "final_quizzes": len(final_quizzes),
                },
                "reasoning_type_distribution": dict(sorted(local_reasoning.items())),
                "quiz_type_reasoning_distribution": dict(sorted(local_cross.items())),
            }
        )

    require(len(quiz_ids) == len(set(quiz_ids)), "Duplicate quiz IDs across scenarios")
    require(totals["dialogue_turns"] == 1000, "Expected exactly 1,000 turns")
    require(totals["quizzes"] == 200, "Expected exactly 200 baseline quizzes")

    return {
        "schema_version": "vehiclemembench-hvp-quiz-expansion-baseline-freeze-v1",
        "purpose": "immutable baseline for adding quizzes without changing dialogue or gold memory",
        "source_root": str(source_root),
        "scenario_ids": expected_ids,
        "immutability_contract": {
            "core_files": list(CORE_FILES),
            "baseline_quiz_files": list(QUIZ_FILES),
            "rule": "Do not modify source files; write expanded quizzes to a versioned sibling dataset.",
        },
        "digests": {
            "all_source_files_sha256": canonical_digest(all_file_hashes),
            "core_files_sha256": canonical_digest(core_hashes),
            "baseline_quiz_files_sha256": canonical_digest(quiz_hashes),
            "baseline_quiz_ids_sha256": hashlib.sha256(
                "\n".join(sorted(quiz_ids)).encode("utf-8")
            ).hexdigest(),
        },
        "totals": dict(sorted(totals.items())),
        "distributions": {
            "quiz_type": dict(sorted(quiz_types.items())),
            "reasoning_type": dict(sorted(reasoning_types.items())),
            "quiz_type_by_reasoning_type": dict(sorted(quiz_type_reasoning.items())),
            "gold_call_tool": dict(sorted(tool_calls.items())),
        },
        "scenarios": scenario_records,
    }


def main() -> None:
    args = parse_args()
    manifest = freeze(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **manifest["totals"]}, indent=2))


if __name__ == "__main__":
    main()
