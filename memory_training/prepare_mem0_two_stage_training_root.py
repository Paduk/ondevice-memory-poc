"""Create a legacy-catalog-compatible EXTRACT+MANAGE training root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .prepare_mem0_one_pass_training_root import (
    DEFAULT_MEM0_ROOT,
    DEFAULT_SOURCE_ROOT,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "mem0-style-two-stage-training-s21t10-no-v1-v2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source_root: Path, mem0_root: Path, output_root: Path) -> dict:
    source_root = source_root.resolve(strict=True)
    mem0_root = mem0_root.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    output = temporary / "two_stage.jsonl"
    counts = {
        "extraction_rows": 0,
        "extraction_updates": 0,
        "extraction_no_ops": 0,
        "manager_rows": 0,
        "manager_updates": 0,
        "manager_none": 0,
    }
    try:
        manager_rows = [
            json.loads(line)
            for line in (mem0_root / "fact_manager.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        manager_sources = {
            row["provenance"]["source_sample_id"] for row in manager_rows
        }
        source_metadata: dict[str, dict] = {}
        with (source_root / "patch.jsonl").open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row["sample_id"] in manager_sources:
                    source_metadata[row["sample_id"]] = row
        if set(source_metadata) != manager_sources:
            raise ValueError("Some manager rows have no canonical source metadata")

        with (
            (source_root / "patch.jsonl").open(encoding="utf-8") as source,
            (mem0_root / "fact_extraction.jsonl").open(encoding="utf-8") as extraction,
            output.open("w", encoding="utf-8") as destination,
        ):
            for line_number, pair in enumerate(zip(source, extraction, strict=True), 1):
                source_row, row = (json.loads(raw) for raw in pair)
                if source_row["sample_id"] != row["sample_id"]:
                    raise ValueError(f"Extraction mismatch at row {line_number}")
                facts = row["target"]["facts"]
                row.update(
                    {
                        "task_type": "EXTRACT",
                        "turn_id": source_row["turn_id"],
                        "current_turn": source_row["current_turn"],
                        "split": source_row["split"],
                        "source_split": source_row["split"],
                        "target": {
                            "decision": "UPDATE" if facts else "NO_OP",
                            "facts": facts,
                        },
                    }
                )
                destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["extraction_rows"] += 1
                counts["extraction_updates" if facts else "extraction_no_ops"] += 1

            per_scenario: dict[int, int] = {}
            for row in manager_rows:
                source_row = source_metadata[row["provenance"]["source_sample_id"]]
                scenario = int(row["scenario_index"])
                ordinal = per_scenario.get(scenario, 0)
                per_scenario[scenario] = ordinal + 1
                events = row["target"]["memory"]
                update = any(event["event"] != "NONE" for event in events)
                row.update(
                    {
                        "task_type": "MANAGE",
                        "turn_id": f"{source_row['turn_id']}:manage:{ordinal:05d}",
                        "global_turn_index": 1_000_000 + ordinal,
                        "current_turn": source_row["current_turn"],
                        "split": source_row["split"],
                        "source_split": source_row["split"],
                        "target": {
                            "decision": "UPDATE" if update else "NO_OP",
                            "memory": events,
                        },
                    }
                )
                destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["manager_rows"] += 1
                counts["manager_updates" if update else "manager_none"] += 1

        for view in ("summary", "patch", "delta"):
            os.link(output, temporary / f"{view}.jsonl")
        for name in ("quiz_sft.jsonl", "vehicle_tools.json", "quiz_manifest.json"):
            os.link(source_root / name, temporary / name)
        manifest = {
            "schema_version": "palmclaw-mem0-two-stage-training-root-v1",
            "source_root": str(source_root),
            "mem0_root": str(mem0_root),
            "counts": counts,
            "extraction_row_boundary": counts["extraction_rows"],
            "two_stage_sha256": _sha256(output),
            "storage": "summary/patch/delta are hard links to two_stage.jsonl",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--mem0-root", type=Path, default=DEFAULT_MEM0_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_root, args.mem0_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
