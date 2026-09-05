"""Freeze a deterministic turn subset for HF KV-cache benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_DATA_ROOT, DEFAULT_WORKSPACE_ROOT
from .dataset import (
    DatasetCatalog,
    IndexedMemoryDataset,
    default_catalog_path,
    ensure_catalog,
)
from .quiz_sft import IndexedQuizSFTDataset
from .validation import (
    closed_loop_quiz_snapshot_requests,
    quiz_indices_for_scenarios,
    ratio_closed_loop_row_ids,
)

SCHEMA_VERSION = "palmclaw-hf-cache-benchmark-turns-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--scenarios", nargs="+", type=int, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--noop-per-update", type=float, default=5.0)
    parser.add_argument("--sampling-seed", type=int, default=45)
    parser.add_argument(
        "--without-quiz-anchors",
        action="store_true",
        help="Do not force the Validation Quiz snapshot turns into the subset.",
    )
    return parser


def build_manifest(
    *,
    data_root: Path,
    catalog_path: Path,
    scenarios: Sequence[int],
    split: str,
    noop_per_update: float,
    sampling_seed: int,
    include_quiz_anchors: bool,
) -> dict[str, Any]:
    if noop_per_update < 0:
        raise ValueError("NO_OP per UPDATE must be non-negative")
    selected_scenarios = tuple(dict.fromkeys(int(value) for value in scenarios))
    if not selected_scenarios:
        raise ValueError("At least one scenario is required")
    data_root = data_root.resolve()
    catalog = ensure_catalog(data_root, catalog_path.resolve())
    available = set(catalog.scenarios(split=split))
    invalid = sorted(set(selected_scenarios) - available)
    if invalid:
        raise ValueError(f"Scenarios are outside the {split} split: {invalid}")

    source = IndexedMemoryDataset(catalog, "patch", split=split)
    required_by_scenario: Mapping[int, Mapping[int, Sequence[str]]] = {}
    if include_quiz_anchors:
        quiz_source = IndexedQuizSFTDataset(
            data_root / "quiz_sft.jsonl", split=split
        )
        quiz_indices = quiz_indices_for_scenarios(quiz_source, selected_scenarios)
        required_by_scenario = closed_loop_quiz_snapshot_requests(
            quiz_source, quiz_indices
        )

    scenario_rows = []
    totals: Counter[str] = Counter()
    original_turns = 0
    for scenario in selected_scenarios:
        original = [
            row_id
            for row_id in catalog.scenario_row_ids(scenario)
            if source.contains_row_id(row_id)
        ]
        required = {
            catalog.row_id_for_turn(scenario, int(turn))
            for turn in required_by_scenario.get(scenario, {})
        }
        selected = ratio_closed_loop_row_ids(
            source,
            original,
            noop_per_update=noop_per_update,
            seed=sampling_seed + scenario,
            required_row_ids=required,
        )
        rows = []
        decisions: Counter[str] = Counter()
        for row_id in selected:
            record = catalog.record(row_id)
            rows.append(
                {
                    "row_id": record.row_id,
                    "global_turn_index": record.global_turn_index,
                    "turn_id": record.turn_id,
                    "gold_decision": record.decision,
                }
            )
            decisions[record.decision] += 1
            totals[record.decision] += 1
        original_turns += len(original)
        scenario_rows.append(
            {
                "scenario_index": scenario,
                "original_turns": len(original),
                "selected_turns": len(rows),
                "gold_updates": decisions["UPDATE"],
                "gold_noops": decisions["NO_OP"],
                "required_quiz_anchor_turns": len(required),
                "rows": rows,
            }
        )

    stable = {
        "schema_version": SCHEMA_VERSION,
        "dataset_source_fingerprint": catalog.metadata()["source_fingerprint"],
        "split": split,
        "selection": {
            "scenarios": list(selected_scenarios),
            "noop_per_update": noop_per_update,
            "sampling_seed": sampling_seed,
            "include_quiz_anchors": include_quiz_anchors,
        },
        "totals": {
            "original_turns": original_turns,
            "selected_turns": totals["UPDATE"] + totals["NO_OP"],
            "gold_updates": totals["UPDATE"],
            "gold_noops": totals["NO_OP"],
        },
        "scenarios": scenario_rows,
    }
    signature = hashlib.sha256(
        json.dumps(
            stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        **stable,
        "signature": signature,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(
    value: Mapping[str, Any], catalog: DatasetCatalog
) -> dict[int, list[int]]:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported HF cache benchmark manifest schema")
    if value.get("dataset_source_fingerprint") != catalog.metadata().get(
        "source_fingerprint"
    ):
        raise ValueError("Benchmark manifest belongs to a different dataset")
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Benchmark manifest has no scenarios")
    selected: dict[int, list[int]] = {}
    for scenario_value in scenarios:
        if not isinstance(scenario_value, Mapping):
            raise TypeError("Benchmark scenario must be an object")
        scenario = int(scenario_value["scenario_index"])
        if scenario in selected:
            raise ValueError(f"Duplicate benchmark scenario: {scenario}")
        rows = scenario_value.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Benchmark scenario S{scenario} has no turns")
        row_ids = []
        previous_turn = -1
        for expected in rows:
            if not isinstance(expected, Mapping):
                raise TypeError("Benchmark turn must be an object")
            row_id = int(expected["row_id"])
            record = catalog.record(row_id)
            identity = (
                record.scenario_index,
                record.global_turn_index,
                record.turn_id,
                record.decision,
            )
            expected_identity = (
                scenario,
                int(expected["global_turn_index"]),
                str(expected["turn_id"]),
                str(expected["gold_decision"]),
            )
            if identity != expected_identity:
                raise ValueError(f"Benchmark turn identity changed for row {row_id}")
            if record.global_turn_index <= previous_turn:
                raise ValueError(f"Benchmark S{scenario} turns are not ordered")
            previous_turn = record.global_turn_index
            row_ids.append(row_id)
        selected[scenario] = row_ids
    return selected


def write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = build_parser().parse_args()
    data_root = args.data_root.resolve()
    workspace = args.workspace.resolve()
    catalog_path = args.catalog_path or (
        data_root / "catalog.sqlite"
        if (data_root / "catalog.sqlite").is_file()
        else default_catalog_path(workspace)
    )
    manifest = build_manifest(
        data_root=data_root,
        catalog_path=catalog_path,
        scenarios=args.scenarios,
        split=args.split,
        noop_per_update=args.noop_per_update,
        sampling_seed=args.sampling_seed,
        include_quiz_anchors=not args.without_quiz_anchors,
    )
    write_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "signature": manifest["signature"],
                "totals": manifest["totals"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
