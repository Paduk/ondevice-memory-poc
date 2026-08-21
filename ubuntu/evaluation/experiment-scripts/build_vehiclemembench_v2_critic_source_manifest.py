#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.critic_trace import (
    build_r2_source_inventory,
    write_source_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen VehicleMemBench R2 Summary/Combined caches and "
            "write the V2 critic source manifest."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Directory containing the existing VehicleMemBench experiment roots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination source_manifest.json path.",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        action="append",
        dest="scenarios",
        help="Scenario to inventory; repeat as needed. Defaults to S1-S50.",
    )
    args = parser.parse_args()
    scenarios = tuple(args.scenarios) if args.scenarios else tuple(range(1, 51))
    inventory = build_r2_source_inventory(
        args.artifact_root,
        scenario_indices=scenarios,
    )
    write_source_inventory(args.output, inventory)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "scenario_count": len(scenarios),
                "totals": inventory["totals"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
