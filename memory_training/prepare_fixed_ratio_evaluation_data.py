"""Build a fixed UPDATE:NO_OP V2 Validation/Test evaluation data root."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from .dataset import (
    MEMORY_VIEWS,
    DatasetCatalog,
    IndexedMemoryDataset,
    build_catalog,
)
from .validation import ratio_closed_loop_row_ids

DEFAULT_SOURCE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2"
)
DEFAULT_OUTPUT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-v2-v1-10-eval-fixed-noop5-seed45-v1"
)
VALIDATION_SCENARIOS = (81, 82, 83, 84, 85, 111)
TEST_SCENARIOS = (*range(86, 101), *range(112, 121))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--noop-per-update", type=int, default=5)
    parser.add_argument("--seed", type=int, default=45)
    return parser


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source.expanduser().resolve(strict=True)
    output_root = args.output.expanduser().resolve()
    if args.noop_per_update < 0:
        raise ValueError("noop_per_update must be non-negative")
    if output_root.exists():
        raise FileExistsError(output_root)
    _validate_source_root(source_root)

    protocol_split = {
        **{scenario: "validation" for scenario in VALIDATION_SCENARIOS},
        **{scenario: "test" for scenario in TEST_SCENARIOS},
    }
    scenarios = tuple(protocol_split)
    quiz_rows, anchors = _load_quiz_rows(source_root / "quiz_sft.jsonl", protocol_split)
    catalog = DatasetCatalog(source_root / "catalog.sqlite", source_root)
    source = IndexedMemoryDataset(catalog, "patch", row_ids=catalog.row_ids())

    selected_by_scenario: dict[int, list[int]] = {}
    selection_records = []
    scenario_records = []
    for scenario in scenarios:
        original = catalog.scenario_row_ids(scenario)
        if not original:
            raise ValueError(f"Memory rows are missing for S{scenario}")
        required = {
            catalog.row_id_for_turn(scenario, turn) for turn in anchors[scenario]
        }
        selected = ratio_closed_loop_row_ids(
            source,
            original,
            noop_per_update=args.noop_per_update,
            seed=args.seed + scenario,
            required_row_ids=required,
        )
        selected_by_scenario[scenario] = selected
        records, counts = _selection_records(
            catalog,
            original,
            selected,
            required,
            protocol_split=protocol_split[scenario],
            noop_per_update=args.noop_per_update,
        )
        selection_records.extend(records)
        scenario_records.append(
            {
                "scenario_index": scenario,
                "protocol_split": protocol_split[scenario],
                "source_memory_split": catalog.record(original[0]).split,
                "original_turns": len(original),
                **counts,
                "quiz_rows": sum(
                    int(row["scenario_index"]) == scenario for row in quiz_rows
                ),
                "quiz_anchor_turns": len(anchors[scenario]),
            }
        )

    selected_row_ids = sorted(
        row_id for selected in selected_by_scenario.values() for row_id in selected
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        for view in MEMORY_VIEWS:
            _copy_selected_view(
                source_root / f"{view}.jsonl",
                temporary / f"{view}.jsonl",
                catalog,
                view,
                selected_row_ids,
            )
        _write_jsonl(temporary / "quiz_sft.jsonl", quiz_rows)
        shutil.copy2(
            source_root / "vehicle_tools.json", temporary / "vehicle_tools.json"
        )
        _write_jsonl(temporary / "selection.jsonl", selection_records)

        split_counts = _aggregate_scenario_records(scenario_records)
        manifest = {
            "schema_version": "palmclaw-fixed-ratio-evaluation-data-v1",
            "created_at": _utc_now(),
            "source_data_root": str(source_root),
            "source_manifest_sha256": _sha256(source_root / "manifest.json"),
            "selection_policy": {
                "update_noop_ratio": f"1:{args.noop_per_update}",
                "noop_per_update": args.noop_per_update,
                "seed": args.seed,
                "priority": [
                    "all_update",
                    "all_quiz_anchor_noop",
                    "update_adjacent_noop",
                    "deterministic_random_noop",
                ],
                "trajectory_order": "preserved",
                "sampling_scope": "per_scenario",
            },
            "protocol_scenarios": {
                "validation": list(VALIDATION_SCENARIOS),
                "test": list(TEST_SCENARIOS),
            },
            "counts": split_counts,
            "scenarios": scenario_records,
        }
        quiz_manifest = {
            "schema_version": "palmclaw-fixed-ratio-evaluation-quiz-v1",
            "source_quiz_manifest_sha256": _sha256(source_root / "quiz_manifest.json"),
            "rows": len(quiz_rows),
            "split_counts": dict(Counter(str(row["sft_split"]) for row in quiz_rows)),
            "scenarios": {
                split: [
                    scenario
                    for scenario, scenario_split in protocol_split.items()
                    if scenario_split == split
                ]
                for split in ("validation", "test")
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        _write_json(temporary / "quiz_manifest.json", quiz_manifest)
        temporary.rename(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    catalog_result = build_catalog(output_root, output_root / "catalog.sqlite")
    result = {
        "schema_version": "palmclaw-fixed-ratio-evaluation-preparation-v1",
        "created_at": _utc_now(),
        "data_root": str(output_root),
        "catalog": str(output_root / "catalog.sqlite"),
        "catalog_result": catalog_result,
        "counts": _aggregate_scenario_records(scenario_records),
        "quiz_rows": len(quiz_rows),
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(output_root.iterdir())
            if path.is_file()
        },
    }
    _write_json(output_root / "preparation-summary.json", result)
    (output_root / "COMPLETED").write_text(_utc_now() + "\n", encoding="utf-8")
    return result


def _validate_source_root(root: Path) -> None:
    required = {
        "catalog.sqlite",
        "manifest.json",
        "quiz_manifest.json",
        "quiz_sft.jsonl",
        "vehicle_tools.json",
        *(f"{view}.jsonl" for view in MEMORY_VIEWS),
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"Source data root is incomplete: {missing}")


def _load_quiz_rows(
    path: Path,
    protocol_split: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[int, set[int]]]:
    rows = []
    anchors: dict[int, set[int]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            scenario = int(row.get("scenario_index", -1))
            if scenario not in protocol_split:
                continue
            expected_split = protocol_split[scenario]
            if row.get("sft_split") != expected_split:
                continue
            try:
                turn = int(row["memory_ref"]["global_turn_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid Quiz memory_ref at {path}:{line_number}"
                ) from exc
            rows.append(row)
            anchors[scenario].add(turn)
    missing = sorted(set(protocol_split) - set(anchors))
    if missing:
        raise ValueError(f"Quiz rows are missing for scenarios: {missing}")
    return rows, anchors


def _selection_records(
    catalog: DatasetCatalog,
    original: Sequence[int],
    selected: Sequence[int],
    required: set[int],
    *,
    protocol_split: str,
    noop_per_update: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected_set = set(selected)
    decisions = [catalog.record(row_id).decision for row_id in original]
    updates = {
        index for index, decision in enumerate(decisions) if decision == "UPDATE"
    }
    adjacent = {
        neighbor
        for index in updates
        for neighbor in (index - 1, index + 1)
        if 0 <= neighbor < len(original) and decisions[neighbor] == "NO_OP"
    }
    records = []
    counts: Counter[str] = Counter()
    for index, row_id in enumerate(original):
        if row_id not in selected_set:
            continue
        record = catalog.record(row_id)
        if record.decision == "UPDATE":
            reason = "UPDATE"
        elif row_id in required:
            reason = "QUIZ_ANCHOR_NO_OP"
        elif index in adjacent:
            reason = "ADJACENT_NO_OP"
        else:
            reason = "RANDOM_NO_OP"
        counts["selected_turns"] += 1
        counts["updates" if record.decision == "UPDATE" else "noops"] += 1
        counts[f"reason_{reason.lower()}"] += 1
        records.append(
            {
                "source_row_id": row_id,
                "scenario_index": record.scenario_index,
                "global_turn_index": record.global_turn_index,
                "turn_id": record.turn_id,
                "protocol_split": protocol_split,
                "source_memory_split": record.split,
                "decision": record.decision,
                "selection_reason": reason,
                "quiz_anchor": row_id in required,
            }
        )
    if counts["updates"] * noop_per_update != counts["noops"]:
        raise ValueError(
            f"S{catalog.record(original[0]).scenario_index} cannot satisfy exact "
            f"1:{noop_per_update}"
        )
    return records, dict(counts)


def _copy_selected_view(
    source: Path,
    output: Path,
    catalog: DatasetCatalog,
    view: str,
    selected_row_ids: Sequence[int],
) -> None:
    with source.open("rb") as input_handle, output.open("wb") as output_handle:
        for _row_id, offset, length, _scenario in catalog.locations(
            view, row_ids=selected_row_ids
        ):
            input_handle.seek(offset)
            _copy_exact(input_handle, output_handle, length)


def _copy_exact(source: BinaryIO, output: BinaryIO, length: int) -> None:
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise EOFError(f"Source JSONL ended with {remaining} bytes remaining")
        output.write(chunk)
        remaining -= len(chunk)


def _aggregate_scenario_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for split in ("validation", "test"):
        selected = [record for record in records if record["protocol_split"] == split]
        result[split] = {
            "scenarios": len(selected),
            "original_turns": sum(record["original_turns"] for record in selected),
            "selected_turns": sum(record["selected_turns"] for record in selected),
            "updates": sum(record["updates"] for record in selected),
            "noops": sum(record["noops"] for record in selected),
            "quiz_rows": sum(record["quiz_rows"] for record in selected),
            "quiz_anchor_turns": sum(
                record["quiz_anchor_turns"] for record in selected
            ),
        }
    result["total"] = {
        key: sum(result[split][key] for split in ("validation", "test"))
        for key in (
            "scenarios",
            "original_turns",
            "selected_turns",
            "updates",
            "noops",
            "quiz_rows",
            "quiz_anchor_turns",
        )
    }
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    result = prepare(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
