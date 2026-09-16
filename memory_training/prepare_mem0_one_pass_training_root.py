"""Create a legacy-catalog-compatible root for Mem0 one-pass SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-s1-s100-plus-temporal-t1-t20-v2"
)
DEFAULT_MEM0_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "mem0-style-one-two-stage-s21t10-no-v1-v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "mem0-style-one-pass-training-s21t10-no-v1-v2"
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
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    output = temporary / "one_pass.jsonl"
    counts = {"rows": 0, "updates": 0, "no_ops": 0, "train_eligible": 0}
    split_counts: dict[str, int] = {}
    try:
        with (
            (source_root / "patch.jsonl").open(encoding="utf-8") as source,
            (mem0_root / "joint_fact_update.jsonl").open(encoding="utf-8") as joint,
            output.open("w", encoding="utf-8") as destination,
        ):
            for line_number, pair in enumerate(zip(source, joint, strict=True), 1):
                source_row, row = (json.loads(raw) for raw in pair)
                identity = ("scenario_index", "global_turn_index", "sample_id")
                if any(source_row.get(key) != row.get(key) for key in identity):
                    raise ValueError(f"Source/joint mismatch at row {line_number}")
                events = row["target"]["memory"]
                decision = "UPDATE" if events else "NO_OP"
                row["turn_id"] = source_row["turn_id"]
                row["current_turn"] = source_row["current_turn"]
                # Keep the canonical split used by the existing aligned catalog.
                row["split"] = source_row["split"]
                row["source_split"] = source_row["split"]
                row["target"] = {
                    "decision": decision,
                    "facts": row["target"]["facts"],
                    "memory": events,
                }
                destination.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                counts["rows"] += 1
                counts["updates" if events else "no_ops"] += 1
                counts["train_eligible"] += int(bool(row.get("train_eligible", True)))
                split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

        for view in ("summary", "patch", "delta"):
            os.link(output, temporary / f"{view}.jsonl")
        for name in ("quiz_sft.jsonl", "vehicle_tools.json", "quiz_manifest.json"):
            os.link(source_root / name, temporary / name)
        manifest = {
            "schema_version": "palmclaw-mem0-one-pass-training-root-v1",
            "source_root": str(source_root),
            "mem0_root": str(mem0_root),
            "counts": counts,
            "split_counts": split_counts,
            "one_pass_sha256": _sha256(output),
            "storage": "summary/patch/delta are hard links to one_pass.jsonl",
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
