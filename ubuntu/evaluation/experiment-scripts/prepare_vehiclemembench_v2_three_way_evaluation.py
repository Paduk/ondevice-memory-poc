#!/usr/bin/env python3
"""Build a resumable readiness manifest for the V2 three-way evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v2_quality_evaluation import (
    build_readiness_manifest,
    default_quality_artifact_paths,
)

DEFAULT_EVALUATION_ROOT = Path("/mnt/data/hj153lee/PalmClaw/evaluation")
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_EVALUATION_ROOT / "vehiclemembench-v2-three-way-evaluation"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    artifacts = default_quality_artifact_paths(args.evaluation_root)
    manifest = build_readiness_manifest(artifacts)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "readiness-manifest.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return {**manifest, "manifest_path": str(destination)}


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
