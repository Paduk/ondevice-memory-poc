#!/usr/bin/env python3
"""Export deterministic Summary, Patch, and 5-UPDATE Delta training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v2_training_export import export_training_views


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path(
            "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
            "hybrid-s1-s100-summary-patch-delta-v2"
        ),
    )
    parser.add_argument("--scenario-start", type=int, default=1)
    parser.add_argument("--scenario-end", type=int, default=100)
    parser.add_argument("--compaction-interval", type=int, default=5)
    parser.add_argument("--train-end", type=int, default=80)
    parser.add_argument("--validation-end", type=int, default=90)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.scenario_start <= args.scenario_end <= 100:
        raise ValueError("Scenario range must be within 1..100")
    if not args.train_end <= args.validation_end:
        raise ValueError("Split boundaries are invalid")
    manifest = export_training_views(
        base_root=args.base_root.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().resolve(),
        scenarios=range(args.scenario_start, args.scenario_end + 1),
        compaction_interval=args.compaction_interval,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
