"""Create an aligned root for predicted-retrieval Mem0 one-pass evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def prepare(one_pass_root: Path, source_root: Path, output_root: Path) -> dict:
    one_pass_root = one_pass_root.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent)
    )
    try:
        # The method reads joint fact-update rows through its legacy `patch`
        # view. Restore the source eligibility flag so the mixed views satisfy
        # the aligned-catalog identity contract; eligibility is irrelevant in
        # chronological Validation/Test replay.
        key_fields = ("scenario_index", "global_turn_index", "turn_id")
        with (source_root / "patch.jsonl").open(encoding="utf-8") as source:
            selected_rows = [json.loads(raw) for raw in source]
        selected_by_key = {
            tuple(row.get(key) for key in key_fields): row for row in selected_rows
        }
        if len(selected_by_key) != len(selected_rows):
            raise ValueError("Evaluation source contains duplicate turn identities")
        joint_by_key = {}
        with (one_pass_root / "one_pass.jsonl").open(encoding="utf-8") as joint:
            for raw in joint:
                row = json.loads(raw)
                key = tuple(row.get(field) for field in key_fields)
                if key in selected_by_key:
                    joint_by_key[key] = row
        missing = set(selected_by_key) - set(joint_by_key)
        if missing:
            raise ValueError(f"One-pass source is missing {len(missing)} selected turns")
        with (temporary / "patch.jsonl").open("w", encoding="utf-8") as output:
            for source_row in selected_rows:
                key = tuple(source_row.get(field) for field in key_fields)
                row = joint_by_key[key]
                if row["target"]["decision"] != source_row["target"]["decision"]:
                    raise ValueError(f"Decision mismatch at turn {key}")
                row["split"] = source_row["split"]
                row["train_eligible"] = bool(source_row.get("train_eligible", True))
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
        for view in ("summary", "delta"):
            os.link(source_root / f"{view}.jsonl", temporary / f"{view}.jsonl")
        for name in ("quiz_sft.jsonl", "vehicle_tools.json", "quiz_manifest.json"):
            os.link(source_root / name, temporary / name)
        manifest = {
            "schema_version": "palmclaw-mem0-one-pass-evaluation-root-v1",
            "one_pass_root": str(one_pass_root),
            "source_root": str(source_root),
            "retrieval": "predicted active facts; all-MiniLM-L6-v2 cosine top-20",
            "gold_view": "original grouped Summary",
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
    parser.add_argument("--one-pass-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.one_pass_root, args.source_root, args.output_root)))


if __name__ == "__main__":
    main()
