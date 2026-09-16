#!/usr/bin/env python3
"""Audit the 24-scenario Base/U40/U60/U80 stress preparation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_training.cache_benchmark_manifest import validate_manifest
from memory_training.config import split_for_scenario
from memory_training.dataset import DatasetCatalog


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PARENT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training"
)
SOURCE_NAME = "grouped-v2-v1-10-eval-fixed-noop5-seed45-v1"
STRESS_PREFIX = "grouped-v2-v1-10-eval-stress-test24"
SCENARIOS = tuple(range(86, 101)) + tuple(range(112, 121))
LOADS = (40, 60, 80)
INTERVALS = (2, 5, 10)
SOURCE_TURNS = 2364
SOURCE_UPDATES = 394
EXPECTED_QUIZZES = 960


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-parent", type=Path, default=DATA_PARENT)
    result.add_argument("--manifest-root", type=Path, required=True)
    result.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "docs/engineering/results/full-test24-update-stress-preparation-audit.json"
        ),
    )
    return result


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_turn_manifest(root: Path, expected_updates: int, expected_turns: int) -> dict:
    catalog = DatasetCatalog(root / "catalog.sqlite", root)
    manifest = read_json(root / "turn_manifest.json")
    selected = validate_manifest(manifest, catalog)
    require(tuple(selected) == SCENARIOS, f"Scenario mismatch in {root}")
    require(
        int(manifest["totals"]["gold_updates"]) == expected_updates,
        f"UPDATE count mismatch in {root}",
    )
    require(
        int(manifest["totals"]["selected_turns"]) == expected_turns,
        f"Turn count mismatch in {root}",
    )
    return manifest


def audit_natural_manifests(data_parent: Path, manifest_root: Path) -> dict[str, Any]:
    source = data_parent / SOURCE_NAME
    result = {}
    for label in ("base", "k2", "k5", "k10"):
        root = source if label == "base" else source.with_name(f"{SOURCE_NAME}-delta-v3-compact-{label}-v1")
        catalog = DatasetCatalog(root / "catalog.sqlite", root)
        manifest = read_json(manifest_root / f"{label}.json")
        selected = validate_manifest(manifest, catalog)
        require(tuple(selected) == SCENARIOS, f"Natural manifest scenario mismatch: {label}")
        require(int(manifest["totals"]["gold_updates"]) == SOURCE_UPDATES, label)
        require(int(manifest["totals"]["selected_turns"]) == SOURCE_TURNS, label)
        result[label] = {
            "signature": manifest["signature"],
            "turns": manifest["totals"]["selected_turns"],
            "updates": manifest["totals"]["gold_updates"],
        }
    return result


def audit_stress_root(root: Path, load: int) -> dict[str, Any]:
    require((root / "COMPLETED").is_file(), f"Missing COMPLETED: {root}")
    manifest = read_json(root / "manifest.json")
    preparation = read_json(root / "preparation-summary.json")
    expected_updates = load * len(SCENARIOS)
    expected_turns = SOURCE_TURNS + expected_updates - SOURCE_UPDATES

    scenario_records = manifest["scenarios"]
    require(
        tuple(int(value["scenario_index"]) for value in scenario_records) == SCENARIOS,
        f"Suite scenario mismatch: {root}",
    )
    require(manifest["split"] == "mixed", f"Suite split is not mixed: {root}")
    require(preparation["quiz_split"] == "test", f"Quiz split mismatch: {root}")
    require(
        int(manifest["totals"]["updates"]) == expected_updates,
        f"Suite UPDATE total mismatch: {root}",
    )
    require(
        int(manifest["totals"]["turns"]) == expected_turns,
        f"Suite turn total mismatch: {root}",
    )
    require(
        int(manifest["totals"]["quiz_rows"]) == EXPECTED_QUIZZES,
        f"Suite Quiz total mismatch: {root}",
    )
    require(
        all(int(value["target_updates"]) == load for value in scenario_records),
        f"Per-scenario UPDATE target mismatch: {root}",
    )
    audit_turn_manifest(root, expected_updates, expected_turns)

    catalog = DatasetCatalog(root / "catalog.sqlite", root)
    for scenario in SCENARIOS:
        expected_split = split_for_scenario(scenario)
        require(
            all(
                catalog.record(row_id).split == expected_split
                for row_id in catalog.scenario_row_ids(scenario)
            ),
            f"Canonical memory split mismatch for S{scenario}: {root}",
        )

    quiz_counts: Counter[int] = Counter()
    with (root / "quiz_sft.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            require(row["sft_split"] == "test", f"Non-test Quiz row in {root}")
            quiz_counts[int(row["scenario_index"])] += 1
    require(tuple(sorted(quiz_counts)) == SCENARIOS, f"Quiz scenarios mismatch: {root}")
    require(all(quiz_counts[value] == 40 for value in SCENARIOS), f"Quiz count mismatch: {root}")

    base_hashes = {
        str(value["scenario_index"]): value["final_memory_sha256"]
        for value in scenario_records
    }
    compact = {}
    for interval in INTERVALS:
        compact_root = root.with_name(f"{root.name}-delta-v3-compact-k{interval}-v1")
        require((compact_root / "COMPLETED").is_file(), f"Missing COMPLETED: {compact_root}")
        compact_summary = read_json(compact_root / "preparation-summary.json")
        require(
            int(compact_summary["compaction_interval"]) == interval,
            f"Compaction interval mismatch: {compact_root}",
        )
        hashes = compact_summary["statistics"]["scenario_final_memory_sha256"]
        require(hashes == base_hashes, f"Final memory mismatch: {compact_root}")
        audit_turn_manifest(compact_root, expected_updates, expected_turns)
        compact[f"k{interval}"] = {
            "turns": expected_turns,
            "updates": expected_updates,
            "final_memories_match": True,
        }

    return {
        "data_root": str(root),
        "scenarios": len(SCENARIOS),
        "turns": expected_turns,
        "updates": expected_updates,
        "quizzes": EXPECTED_QUIZZES,
        "compact": compact,
    }


def main() -> None:
    args = parser().parse_args()
    data_parent = args.data_parent.expanduser().resolve()
    manifest_root = args.manifest_root.expanduser().resolve()
    report = {
        "schema_version": "palmclaw-full-test24-stress-preparation-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": list(SCENARIOS),
        "natural_manifests": audit_natural_manifests(data_parent, manifest_root),
        "stress": {},
    }
    for load in LOADS:
        root = data_parent / f"{STRESS_PREFIX}-update{load}-front-middle-v1"
        report["stress"][f"U{load}"] = audit_stress_root(root, load)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
