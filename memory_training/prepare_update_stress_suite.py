"""Build a deterministic multi-scenario UPDATE stress evaluation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from argparse import Namespace
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import MEMORY_VIEWS, DatasetCatalog, build_catalog
from .prepare_update_stress_scenario import prepare as prepare_scenario

DEFAULT_SOURCE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-v2-v1-10-eval-fixed-noop5-seed45-v1"
)
TURN_MANIFEST_SCHEMA_VERSION = "palmclaw-hf-cache-benchmark-turns-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", type=int, required=True)
    parser.add_argument("--target-updates", type=int, required=True)
    parser.add_argument("--compaction-interval", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    scenarios = tuple(dict.fromkeys(int(value) for value in args.scenarios))
    if not scenarios:
        raise ValueError("At least one scenario is required")
    if args.target_updates < 1:
        raise ValueError("target_updates must be positive")
    if output.exists():
        if not args.force:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.parts-", dir=output.parent))
    assembled = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        parts: list[tuple[int, Path, dict[str, Any]]] = []
        for scenario in scenarios:
            part = work / f"s{scenario:03d}"
            result = prepare_scenario(
                Namespace(
                    source=source,
                    output=part,
                    scenario=scenario,
                    target_updates=args.target_updates,
                    compaction_interval=args.compaction_interval,
                    force=False,
                )
            )
            parts.append((scenario, part, result))

        memory_split = _common_split(parts)
        quiz_split = _common_quiz_split(parts)
        _assemble_files(assembled, parts)
        shutil.copy2(source / "vehicle_tools.json", assembled / "vehicle_tools.json")
        manifest = _suite_manifest(
            source,
            assembled,
            parts=parts,
            split=memory_split,
            target_updates=args.target_updates,
            compaction_interval=args.compaction_interval,
        )
        _write_json(assembled / "manifest.json", manifest)
        _write_json(
            assembled / "quiz_manifest.json",
            {
                "schema_version": "palmclaw-update-stress-quiz-suite-v1",
                "source_quiz_sft": str(source / "quiz_sft.jsonl"),
                "memory_split": memory_split,
                "quiz_split": quiz_split,
                "scenarios": list(scenarios),
                "rows": sum(int(result["quiz"]["rows"]) for _, _, result in parts),
                "note": "Original labels with remapped stress-memory snapshots.",
            },
        )
        catalog_result = build_catalog(assembled, assembled / "catalog.sqlite")
        catalog_result["catalog_path"] = str(output / "catalog.sqlite")
        turn_manifest = _turn_manifest(
            assembled,
            scenarios=scenarios,
            split=memory_split,
            target_updates=args.target_updates,
        )
        _write_json(assembled / "turn_manifest.json", turn_manifest)
        summary = {
            "schema_version": "palmclaw-update-stress-suite-preparation-v1",
            "created_at": _utc_now(),
            "source": str(source),
            "data_root": str(output),
            "scenarios": list(scenarios),
            "memory_split": memory_split,
            "quiz_split": quiz_split,
            "target_updates_per_scenario": args.target_updates,
            "compaction_interval": args.compaction_interval,
            "totals": manifest["totals"],
            "catalog": catalog_result,
            "turn_manifest_signature": turn_manifest["signature"],
        }
        _write_json(assembled / "preparation-summary.json", summary)
        (assembled / "COMPLETED").write_text(_utc_now() + "\n", encoding="utf-8")
        assembled.rename(output)
    except BaseException:
        if assembled.exists():
            shutil.rmtree(assembled)
        raise
    finally:
        if work.exists():
            shutil.rmtree(work)
    return summary


def _common_split(parts: Sequence[tuple[int, Path, Mapping[str, Any]]]) -> str:
    splits = set()
    for _, part, _ in parts:
        with (part / "patch.jsonl").open(encoding="utf-8") as handle:
            splits.add(str(json.loads(next(handle))["split"]))
    if len(splits) != 1:
        raise ValueError(f"Stress suite cannot mix splits: {sorted(splits)}")
    return splits.pop()


def _common_quiz_split(
    parts: Sequence[tuple[int, Path, Mapping[str, Any]]]
) -> str:
    splits = {
        str(json.loads((part / "quiz_manifest.json").read_text())["quiz_split"])
        for _, part, _ in parts
    }
    if len(splits) != 1:
        raise ValueError(f"Stress suite cannot mix Quiz splits: {sorted(splits)}")
    return splits.pop()


def _assemble_files(
    output: Path, parts: Sequence[tuple[int, Path, Mapping[str, Any]]]
) -> None:
    for name in (*MEMORY_VIEWS, "compaction", "quiz_sft"):
        with (output / f"{name}.jsonl").open("wb") as target:
            for _, part, _ in parts:
                with (part / f"{name}.jsonl").open("rb") as source:
                    shutil.copyfileobj(source, target)


def _suite_manifest(
    source: Path,
    output: Path,
    *,
    parts: Sequence[tuple[int, Path, Mapping[str, Any]]],
    split: str,
    target_updates: int,
    compaction_interval: int,
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    scenario_records = []
    for scenario, _, result in parts:
        counts = result["counts"]
        record = {
            "scenario_index": scenario,
            "source_rows": int(result["construction"]["source_rows_preserved"]),
            "source_updates": int(result["construction"]["source_updates_preserved"]),
            "source_noops": int(result["construction"]["source_noops_preserved"]),
            "synthetic_updates": int(
                result["construction"]["synthetic_unique_add_updates"]
            ),
            "target_updates": target_updates,
            "turns": int(counts["UPDATE"]) + int(counts["NO_OP"]),
            "quiz_rows": int(result["quiz"]["rows"]),
            "final_memory_sha256": str(result["final_memory"]["sha256"]),
        }
        scenario_records.append(record)
        totals.update(
            {
                "source_rows": record["source_rows"],
                "source_updates": record["source_updates"],
                "source_noops": record["source_noops"],
                "synthetic_updates": record["synthetic_updates"],
                "updates": record["target_updates"],
                "turns": record["turns"],
                "quiz_rows": record["quiz_rows"],
            }
        )
    return {
        "schema_version": "palmclaw-update-stress-suite-v1",
        "created_at": _utc_now(),
        "source_data_root": str(source),
        "split": split,
        "scenarios": scenario_records,
        "construction": {
            "target_updates_per_scenario": target_updates,
            "synthetic_operation": "unambiguous unique ADD",
            "temporal_distribution": "evenly interleaved across retained source rows",
            "memory_positions": "alternating front and middle",
            "compaction_interval": compaction_interval,
            "quiz_policy": "original labels, remapped checkpoints and memory snapshots",
        },
        "totals": dict(totals),
        "files": {
            f"{name}.jsonl": _file_record(output / f"{name}.jsonl")
            for name in (*MEMORY_VIEWS, "compaction", "quiz_sft")
        },
    }


def _turn_manifest(
    data_root: Path,
    *,
    scenarios: Sequence[int],
    split: str,
    target_updates: int,
) -> dict[str, Any]:
    catalog = DatasetCatalog(data_root / "catalog.sqlite", data_root)
    quiz_anchors = _quiz_anchors(data_root / "quiz_sft.jsonl")
    scenario_values = []
    totals: Counter[str] = Counter()
    for scenario in scenarios:
        records = [catalog.record(row_id) for row_id in catalog.scenario_row_ids(scenario)]
        decisions = Counter(record.decision for record in records)
        totals.update(decisions)
        scenario_values.append(
            {
                "scenario_index": scenario,
                "original_turns": len(records),
                "selected_turns": len(records),
                "gold_updates": decisions["UPDATE"],
                "gold_noops": decisions["NO_OP"],
                "required_quiz_anchor_turns": len(quiz_anchors.get(scenario, set())),
                "rows": [
                    {
                        "row_id": record.row_id,
                        "global_turn_index": record.global_turn_index,
                        "turn_id": record.turn_id,
                        "gold_decision": record.decision,
                    }
                    for record in records
                ],
            }
        )
    stable = {
        "schema_version": TURN_MANIFEST_SCHEMA_VERSION,
        "dataset_source_fingerprint": catalog.metadata()["source_fingerprint"],
        "split": split,
        "selection": {
            "mode": "all_stress_scenario_rows",
            "scenarios": list(scenarios),
            "target_updates_per_scenario": target_updates,
        },
        "totals": {
            "original_turns": totals["UPDATE"] + totals["NO_OP"],
            "selected_turns": totals["UPDATE"] + totals["NO_OP"],
            "gold_updates": totals["UPDATE"],
            "gold_noops": totals["NO_OP"],
        },
        "scenarios": scenario_values,
    }
    signature = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**stable, "signature": signature, "created_at": _utc_now()}


def _quiz_anchors(path: Path) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result.setdefault(int(row["scenario_index"]), set()).add(
                int(row["memory_ref"]["global_turn_index"])
            )
    return result


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    result = prepare(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
