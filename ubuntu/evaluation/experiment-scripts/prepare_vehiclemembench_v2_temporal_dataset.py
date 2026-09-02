#!/usr/bin/env python3
"""Write the frozen 20-scenario V2 Temporal generation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v2_temporal_scenario import (
    build_temporal_dataset_plan,
    write_temporal_plan,
)

DEFAULT_OUTPUT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-temporal/"
    "dataset-plan.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_temporal_dataset_plan()
    write_temporal_plan(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
