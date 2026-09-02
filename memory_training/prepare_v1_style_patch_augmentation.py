"""Append the audited V1-style S301-S320 clones to the grouped Patch corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import build_catalog
from .evaluate_hf_v1 import _build_v1_quiz_rows, _load_scenario_variants
from .methods.delta_v2 import compact_operations, expand_compact_operations
from .methods.operations import apply_operations, normalize_memory
from .quiz_sft import SCHEMA_VERSION, VehicleToolSchemaStore, sft_split_for_scenario
from .ubuntu_bridge import enable_ubuntu_runtime

DEFAULT_BASE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2"
)
DEFAULT_STYLE_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/"
    "vehiclemembench-v1-style-augmentation-v1"
)
DEFAULT_OUTPUT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-plus-v1-style-20-v1"
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--style-root", type=Path, default=DEFAULT_STYLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--vehiclemembench-root", type=Path, default=Path("/home/hj153lee/VehicleMemBench")
    )
    parser.add_argument("--force", action="store_true")
    return parser


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    enable_ubuntu_runtime()
    from palmclaw_ubuntu.vehicle_bench.dataset import load_vehicle_benchmark

    base = args.base_data.expanduser().resolve(strict=True)
    style_root = args.style_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    benchmark_root = args.vehiclemembench_root.expanduser().resolve(strict=True)
    splits = json.loads((style_root / "evaluation-splits.json").read_text(encoding="utf-8"))
    expected_sources = set(map(int, splits["v1_style_reference_20"]))
    styles = _discover_styles(style_root)
    if {item[0] for item in styles} != expected_sources:
        raise ValueError("Audited V1-style source scenarios differ from evaluation-splits.json")
    if {item[1] for item in styles} != set(range(301, 321)):
        raise ValueError("V1-style target scenarios must be exactly S301-S320")

    if output.exists():
        if not args.force:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for name in (
        "summary.jsonl",
        "patch.jsonl",
        "delta.jsonl",
        "compaction.jsonl",
        "turn_quiz.jsonl",
        "final_quiz.jsonl",
        "quiz_sft.jsonl",
        "vehicle_tools.json",
    ):
        source = base / name
        if source.exists():
            shutil.copy2(source, output / name)

    handles = {
        view: (output / f"{view}.jsonl").open("a", encoding="utf-8")
        for view in ("summary", "patch", "delta")
    }
    counts: Counter[str] = Counter()
    records = []
    try:
        for source_scenario, target_scenario, directory in styles:
            record = _append_scenario(
                directory,
                source_scenario=source_scenario,
                target_scenario=target_scenario,
                handles=handles,
            )
            records.append(record)
            counts.update(record["counts"])
    finally:
        for handle in handles.values():
            handle.close()

    benchmark = load_vehicle_benchmark(benchmark_root, strict=True)
    specs = [
        (
            target,
            source,
            directory / "benchmark" / "history" / f"history_{target}.txt",
            directory / "benchmark" / "qa_data" / f"qa_{target}.json",
        )
        for source, target, directory in styles
    ]
    variants = _load_scenario_variants(specs, benchmark)
    tools_path = output / "vehicle_tools.json"
    tools = VehicleToolSchemaStore(tools_path)
    quiz_rows = _build_v1_quiz_rows(
        [variants[target] for _, target, _ in styles], tools_path, tools.sha256
    )
    quiz_manifest = _append_quizzes(output / "quiz_sft.jsonl", quiz_rows, styles)

    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "palmclaw-v1-style-patch-augmentation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_data": str(base),
        "style_root": str(style_root),
        "method_scope": "patch_training; aligned summary/delta views retained for catalog integrity",
        "source_v1_scenarios": [source for source, _, _ in styles],
        "encoded_train_scenarios": [target for _, target, _ in styles],
        "split_policy": {
            "memory": "train only; S301-S320",
            "quiz": "train only; all 10/10 quizzes per style scenario",
            "validation_test": "unchanged from base data",
        },
        "sampling_policy": {
            "memory": "all UPDATE plus trainer noop5 sampling",
            "quiz": "all quizzes per style scenario",
        },
        "counts": dict(counts),
        "base_counts": base_manifest.get("counts", {}),
        "scenarios": records,
        "quiz": quiz_manifest,
        "files": {
            view: _file_record(output / f"{view}.jsonl")
            for view in ("summary", "patch", "delta")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "quiz_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "palmclaw-v1-style-quiz-augmentation-v1",
                "base_manifest": json.loads((base / "quiz_manifest.json").read_text()),
                "added_v1_style_quizzes": quiz_manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    catalog_result = build_catalog(output, output / "catalog.sqlite")
    result = {
        "data_root": str(output),
        "catalog": str(output / "catalog.sqlite"),
        "encoded_train_scenarios": [target for _, target, _ in styles],
        "added_memory_counts": dict(counts),
        "added_quiz_count": quiz_manifest["count"],
        "catalog_result": catalog_result,
    }
    (output / "preparation-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _discover_styles(root: Path) -> list[tuple[int, int, Path]]:
    found = []
    for directory in sorted(root.glob("*-style-v2")):
        audit_path = directory / "audit.json"
        if not (directory / "COMPLETED").is_file() or not audit_path.is_file():
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("passed") is not True:
            raise ValueError(f"V1-style audit did not pass: {directory}")
        first = json.loads(next((directory / "patch.jsonl").open(encoding="utf-8")))
        found.append((int(audit["source_scenario"]), int(first["scenario_index"]), directory))
    return sorted(found, key=lambda item: item[1])


def _append_scenario(
    directory: Path,
    *,
    source_scenario: int,
    target_scenario: int,
    handles: Mapping[str, Any],
) -> dict[str, Any]:
    summary_lines = (directory / "summary.jsonl").open(encoding="utf-8")
    patch_lines = (directory / "patch.jsonl").open(encoding="utf-8")
    turn_lines = (directory / "turnwise.jsonl").open(encoding="utf-8")
    base_summary = ""
    pending: list[list[list[str]]] = []
    counts: Counter[str] = Counter()
    final_memory = ""
    try:
        for position, lines in enumerate(zip(summary_lines, patch_lines, turn_lines, strict=True)):
            summary, patch, turn = map(json.loads, lines)
            identity = (
                int(summary["scenario_index"]),
                int(summary["global_turn_index"]),
                summary["turn_id"],
                summary["target"]["decision"],
            )
            if identity != (
                int(patch["scenario_index"]),
                int(patch["global_turn_index"]),
                patch["turn_id"],
                patch["target"]["decision"],
            ):
                raise ValueError(f"Summary/Patch mismatch: {directory}:{position}")
            if identity[0] != target_scenario or identity[1] != position:
                raise ValueError(f"Unexpected scenario/turn identity: {directory}:{position}")
            previous = normalize_memory(patch["input"]["previous_memory"])
            materialized = _materialize_delta(base_summary, pending)
            if materialized != previous:
                raise ValueError(f"Delta input replay mismatch: {directory}:{position}")
            operations = list(patch["target"].get("operations") or [])
            replayed = previous
            if operations:
                replayed, _ = apply_operations(previous, operations)
            next_memory = normalize_memory(summary["target"]["next_memory"])
            if replayed != next_memory or normalize_memory(turn["target"]["next_memory"]) != next_memory:
                raise ValueError(f"Patch replay mismatch: {directory}:{position}")

            for row in (summary, patch):
                row["train_eligible"] = True
            delta = {
                key: value
                for key, value in patch.items()
                if key not in {"schema_version", "sample_id", "input", "target"}
            }
            delta.update(
                {
                    "schema_version": "vehiclemembench-v2-grouped-delta-v2-sft-v2",
                    "sample_id": str(patch["sample_id"]).replace(":patch:", ":delta:"),
                    "train_eligible": True,
                    "input": {
                        "base_summary": base_summary,
                        "pending_updates": list(pending),
                    },
                    "target": {
                        "decision": patch["target"]["decision"],
                        "reason_code": patch["target"].get("reason_code", ""),
                        "reason": patch["target"].get("reason", ""),
                        "operations": compact_operations(operations) if operations else [],
                    },
                }
            )
            for view, row in (("summary", summary), ("patch", patch), ("delta", delta)):
                handles[view].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["turns"] += 1
            counts["updates" if operations else "no_ops"] += 1
            counts["operations"] += len(operations)
            counts.update(str(operation["op"]) for operation in operations)
            if operations:
                pending.append(compact_operations(operations))
                if len(pending) == 5:
                    base_summary = _materialize_delta(base_summary, pending)
                    pending = []
                    counts["compactions"] += 1
            if _materialize_delta(base_summary, pending) != next_memory:
                raise ValueError(f"Post-turn Delta replay mismatch: {directory}:{position}")
            final_memory = next_memory
    finally:
        summary_lines.close()
        patch_lines.close()
        turn_lines.close()
    return {
        "source_scenario_index": source_scenario,
        "encoded_scenario_index": target_scenario,
        "directory": str(directory),
        "audit_sha256": _sha256(directory / "audit.json"),
        "counts": dict(counts),
        "final_memory_sha256": hashlib.sha256(final_memory.encode()).hexdigest(),
    }


def _materialize_delta(base: str, pending: Sequence[Sequence[Sequence[str]]]) -> str:
    memory = normalize_memory(base)
    for compact in pending:
        memory, _ = apply_operations(memory, expand_compact_operations(compact))
    return memory


def _append_quizzes(
    path: Path,
    rows: Sequence[dict[str, Any]],
    styles: Sequence[tuple[int, int, Path]],
) -> dict[str, Any]:
    by_scenario = {target: [] for _, target, _ in styles}
    for row in rows:
        by_scenario[int(row["scenario_index"])].append(row)
    records = []
    with path.open("a", encoding="utf-8") as handle:
        for source, target, _ in styles:
            candidates = by_scenario[target]
            if len(candidates) != 10:
                raise ValueError(f"Expected 10 style quizzes for S{target}")
            selected = sorted(candidates, key=lambda row: row["quiz_id"])
            ids = []
            for row in selected:
                row = dict(row)
                row.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sample_id": f"v1-style:{row['sample_id']}",
                        "scenario_index": target,
                        "source_scenario_index": source,
                        "source_split": "V1_STYLE_AUXILIARY_TRAIN",
                        "sft_split": sft_split_for_scenario(target),
                    }
                )
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                ids.append(row["quiz_id"])
            records.append(
                {"source_scenario_index": source, "encoded_scenario_index": target, "quiz_ids": ids}
            )
    return {"count": len(records) * 10, "fraction": 1.0, "scenarios": records}


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    print(json.dumps(prepare(build_parser().parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
