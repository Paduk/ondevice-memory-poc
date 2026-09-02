"""Prepare S1-S100 + encoded T1-T20 Patch and Temporal-Patch datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .config import split_for_scenario
from .dataset import build_catalog
from .quiz_sft import build_quiz_sft
from .ubuntu_bridge import enable_ubuntu_runtime

DEFAULT_BASE_DATA = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "hybrid-s1-s100-summary-patch-delta-v2"
)
DEFAULT_TEMPORAL_SOURCE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-temporal"
)
DEFAULT_PATCH_OUTPUT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "hybrid-s1-s100-plus-temporal-t1-t20-patch-t1t10-v2"
)
DEFAULT_TEMPORAL_OUTPUT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "hybrid-s1-s100-plus-temporal-t1-t20-temporal-patch-t1t10-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--temporal-source", type=Path, default=DEFAULT_TEMPORAL_SOURCE)
    parser.add_argument("--patch-output", type=Path, default=DEFAULT_PATCH_OUTPUT)
    parser.add_argument("--temporal-output", type=Path, default=DEFAULT_TEMPORAL_OUTPUT)
    parser.add_argument(
        "--vehiclemembench-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def temporal_run_root(root: Path, temporal_index: int) -> Path:
    tag = f"{temporal_index:02d}" if temporal_index < 10 else str(temporal_index)
    revision = 2 if temporal_index == 1 else 1
    return root / f"hybrid-temporal-anchor-terra-t{tag}-r{revision}"


def prepare_mixed_datasets(
    *,
    base_data: Path,
    temporal_source: Path,
    patch_output: Path,
    temporal_output: Path,
    vehiclemembench_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    enable_ubuntu_runtime()
    from palmclaw_ubuntu.vehicle_bench.v2_quiz_export import (
        _export_scenario_quizzes,
    )
    from palmclaw_ubuntu.vehicle_bench.v2_temporal_training_export import (
        export_temporal_patch_dataset,
    )
    from palmclaw_ubuntu.vehicle_bench.v2_training_export import (
        build_training_views,
        load_canonical_training_turns,
    )

    base_data = base_data.expanduser().resolve(strict=True)
    temporal_source = temporal_source.expanduser().resolve(strict=True)
    vehiclemembench_root = vehiclemembench_root.expanduser().resolve(strict=True)
    patch_output = patch_output.expanduser().resolve()
    temporal_output = temporal_output.expanduser().resolve()
    for output in (patch_output, temporal_output):
        if output.exists():
            if not force:
                raise FileExistsError(f"Output already exists: {output}")
            shutil.rmtree(output)
        output.mkdir(parents=True)

    view_names = ("summary", "patch", "delta", "compaction")
    temporary_paths = {
        name: patch_output / f".{name}.jsonl.tmp" for name in view_names
    }
    final_paths = {name: patch_output / f"{name}.jsonl" for name in view_names}
    counts: Counter[str] = Counter()
    temporal_scenarios = []

    handles = {
        name: temporary_paths[name].open("wb") for name in view_names
    }
    try:
        for name, handle in handles.items():
            with (base_data / f"{name}.jsonl").open("rb") as source:
                shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
        for temporal_index in range(1, 21):
            encoded_scenario = 100 + temporal_index
            run_root = temporal_run_root(temporal_source, temporal_index)
            hybrid, turns = load_canonical_training_turns(run_root)
            summary, patch, delta, compaction = build_training_views(
                scenario_index=encoded_scenario,
                run_id=run_root.name,
                source_hybrid_sha256=hybrid.artifact_sha256,
                turns=turns,
                split=split_for_scenario(encoded_scenario),
                compaction_interval=5,
            )
            by_view = {
                "summary": summary,
                "patch": patch,
                "delta": delta,
                "compaction": compaction,
            }
            for name, rows in by_view.items():
                for row in rows:
                    handles[name].write(
                        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                    )
            update_count = sum(
                row["target"]["decision"] == "UPDATE" for row in patch
            )
            counts.update(
                turns=len(patch),
                updates=update_count,
                no_ops=len(patch) - update_count,
                compactions=len(compaction),
                operations=sum(
                    len(row["target"].get("operations", ())) for row in patch
                ),
            )
            temporal_scenarios.append(
                {
                    "temporal_id": f"T{temporal_index}",
                    "encoded_scenario_index": encoded_scenario,
                    "split": split_for_scenario(encoded_scenario),
                    "run_id": run_root.name,
                    "turn_count": len(patch),
                    "update_count": update_count,
                    "quiz_count": 40,
                }
            )
    finally:
        for handle in handles.values():
            handle.close()
    for name in view_names:
        temporary_paths[name].replace(final_paths[name])

    turn_path = patch_output / "turn_quiz.jsonl"
    final_quiz_path = patch_output / "final_quiz.jsonl"
    with turn_path.open("wb") as target, (base_data / "turn_quiz.jsonl").open(
        "rb"
    ) as source:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    with final_quiz_path.open("wb") as target, (base_data / "final_quiz.jsonl").open(
        "rb"
    ) as source:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    with turn_path.open("a", encoding="utf-8") as turn_handle, final_quiz_path.open(
        "a", encoding="utf-8"
    ) as final_handle:
        for temporal_index in range(1, 21):
            encoded_scenario = 100 + temporal_index
            _export_scenario_quizzes(
                scenario_index=encoded_scenario,
                split=split_for_scenario(encoded_scenario),
                run_root=temporal_run_root(temporal_source, temporal_index),
                turn_handle=turn_handle,
                final_handle=final_handle,
            )

    base_manifest = json.loads((base_data / "manifest.json").read_text())
    base_counts = Counter(base_manifest.get("counts", {}))
    merged_counts = dict(base_counts + counts)
    manifest = {
        "schema_version": "vehiclemembench-v2-s-plus-t-training-export-v1",
        "source_base_data": str(base_data),
        "source_temporal_root": str(temporal_source),
        "scenario_encoding": {"S1-S100": "1-100", "T1-T20": "101-120"},
        "split_policy": {
            "memory_train": "S15-S80 + T1-T10",
            "memory_validation": "S81-S85 + T11",
            "test": "S86-S100 + T12-T20",
        },
        "scenario_count": 120,
        "counts": merged_counts,
        "temporal_scenarios": temporal_scenarios,
        "files": {name: _file_record(path) for name, path in final_paths.items()},
    }
    (patch_output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    quiz_manifest = {
        "schema_version": "vehiclemembench-v2-s-plus-t-quiz-export-v1",
        "scenario_count": 120,
        "counts": {"turn_quizzes": 3600, "final_quizzes": 1200, "total": 4800},
        "files": {
            "turn_quiz": _file_record(turn_path),
            "final_quiz": _file_record(final_quiz_path),
        },
    }
    (patch_output / "quiz_manifest.json").write_text(
        json.dumps(quiz_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    patch_catalog = patch_output / "catalog.sqlite"
    build_catalog(patch_output, patch_catalog)
    quiz_sft_manifest = build_quiz_sft(
        patch_output,
        vehiclemembench_root,
        catalog_path=patch_catalog,
        expected_scenarios=tuple(range(1, 121)),
    )

    temporal_manifest = export_temporal_patch_dataset(
        source_root=patch_output,
        output_root=temporal_output,
    )
    quiz_manifest_link = temporal_output / "quiz_manifest.json"
    if not quiz_manifest_link.exists() and not quiz_manifest_link.is_symlink():
        quiz_manifest_link.symlink_to(patch_output / "quiz_manifest.json")
    temporal_catalog = temporal_output / "catalog.sqlite"
    build_catalog(temporal_output, temporal_catalog)
    result = {
        "patch_root": str(patch_output),
        "temporal_patch_root": str(temporal_output),
        "patch_catalog": str(patch_catalog),
        "temporal_patch_catalog": str(temporal_catalog),
        "scenario_encoding": manifest["scenario_encoding"],
        "split_policy": manifest["split_policy"],
        "temporal_counts": dict(counts),
        "quiz_sft_counts": quiz_sft_manifest["counts"],
        "temporal_action_counts": temporal_manifest["counts"].get(
            "temporal_actions", {}
        ),
    }
    (patch_output / "preparation-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        lines = sum(1 for _ in handle)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "line_count": lines,
        "sha256": _sha256(path),
    }


def main() -> int:
    args = build_parser().parse_args()
    result = prepare_mixed_datasets(
        base_data=args.base_data,
        temporal_source=args.temporal_source,
        patch_output=args.patch_output,
        temporal_output=args.temporal_output,
        vehiclemembench_root=args.vehiclemembench_root,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
