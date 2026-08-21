#!/usr/bin/env python3
"""Extract the public V1 contract used by the reproduction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.dataset import (
    OFFICIAL_UPSTREAM_COMMIT,
    load_vehicle_benchmark,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import (
    build_v1_reference_profile,
    build_v1_reproduction_manifest,
    write_v1_reproduction_reference,
)

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "reference-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-scenarios", type=int, default=100)
    parser.add_argument(
        "--allow-unpinned-checkout",
        action="store_true",
        help="Analyze a checkout other than the official pinned commit.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_scenarios < 1:
        raise ValueError("--target-scenarios must be positive")
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=(
            None if args.allow_unpinned_checkout else OFFICIAL_UPSTREAM_COMMIT
        ),
        strict=True,
    )
    profile = build_v1_reference_profile(
        dataset,
        target_scenario_count=args.target_scenarios,
    )
    manifest = build_v1_reproduction_manifest(
        profile,
        target_scenario_count=args.target_scenarios,
    )
    paths = write_v1_reproduction_reference(
        args.output_root,
        profile=profile,
        manifest=manifest,
    )
    totals = profile.observed_contract["public_output_totals"]
    return {
        "status": "ok",
        "source_commit": dataset.manifest.upstream_commit,
        "source_dataset_sha256": dataset.manifest.dataset_sha256,
        "target_scenario_count": manifest.target_scenario_count,
        "candidate_persona_group_count": manifest.candidate_persona_group_count,
        "observed_totals": totals,
        "artifacts": {name: str(path) for name, path in sorted(paths.items())},
    }


def main() -> int:
    payload = run(build_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
