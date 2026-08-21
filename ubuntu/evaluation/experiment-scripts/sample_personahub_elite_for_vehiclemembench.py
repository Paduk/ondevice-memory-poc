#!/usr/bin/env python3
"""Create a bounded, pinned Persona-Hub Elite seed sample for V1 reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v1_persona_source import (
    sample_persona_hub_elite,
)

DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "sources/personahub-elite-sample-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-count", type=int, default=600)
    parser.add_argument("--chunk-bytes", type=int, default=1_048_576)
    parser.add_argument("--sampling-seed", type=int, default=260323840)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seed_path, manifest_path = sample_persona_hub_elite(
        args.output_root,
        sample_count=args.sample_count,
        chunk_bytes=args.chunk_bytes,
        sampling_seed=args.sampling_seed,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "persona_seeds": str(seed_path),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
