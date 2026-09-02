#!/usr/bin/env python3
"""Export the deterministic Hybrid S1-S100 Temporal-Patch dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v2_temporal_training_export import (
    export_temporal_patch_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
            "hybrid-s1-s100-summary-patch-delta-v2"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/nvme2/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
            "hybrid-s1-s100-temporal-patch-v1"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = export_temporal_patch_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
