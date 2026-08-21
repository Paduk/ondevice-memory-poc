#!/usr/bin/env python3
"""Build the deterministic UPDATE/NO_OP audit readiness manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v2_update_audit import (
    DEFAULT_NO_OP_EVENT_SAMPLE_RATE,
    DEFAULT_NO_OP_SAMPLE_SEED,
    build_update_audit_manifest,
    default_update_audit_artifact_paths,
)

DEFAULT_EVALUATION_ROOT = Path("/mnt/data/hj153lee/PalmClaw/evaluation")
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_EVALUATION_ROOT
    / "vehiclemembench-v2-three-way-evaluation"
    / "update-audit"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=DEFAULT_EVALUATION_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--no-op-event-sample-rate",
        type=float,
        default=DEFAULT_NO_OP_EVENT_SAMPLE_RATE,
    )
    parser.add_argument(
        "--no-op-sample-seed",
        default=DEFAULT_NO_OP_SAMPLE_SEED,
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    artifacts = default_update_audit_artifact_paths(args.evaluation_root)
    manifest = build_update_audit_manifest(
        artifacts,
        no_op_event_sample_rate=args.no_op_event_sample_rate,
        no_op_sample_seed=args.no_op_sample_seed,
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "manifest.json"
    _write_json(destination, manifest)
    return {**manifest, "manifest_path": str(destination)}


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
