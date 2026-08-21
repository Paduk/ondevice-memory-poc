#!/usr/bin/env python3
"""Create a causal V2 memory-anchor projection from one V1 Stage 2 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1Stage2Artifact,
)
from palmclaw_ubuntu.vehicle_bench.v2_memory_anchors import (
    OpenAIV2MemoryAnchorModel,
    V2MemoryAnchorArtifact,
    build_memory_anchor_artifact,
    project_anchored_stage2,
    write_memory_anchor_artifacts,
)

DEFAULT_STAGE2_PATH = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "stage2-smoke-terra-s1-v4/stage2.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid/"
    "hybrid-full-terra-s1-r1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-path", type=Path, default=DEFAULT_STAGE2_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--anchor-checkpoint",
        type=Path,
        help="Reuse a validated memory-anchor artifact without another API call.",
    )
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    return parser


def run(args: argparse.Namespace) -> dict:
    stage2 = V1Stage2Artifact.model_validate_json(
        args.stage2_path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    if args.anchor_checkpoint is not None:
        checkpoint = V2MemoryAnchorArtifact.model_validate_json(
            args.anchor_checkpoint.expanduser()
            .resolve(strict=True)
            .read_text(encoding="utf-8")
        )
        if checkpoint.source_stage2_sha256 != stage2.artifact_sha256:
            raise ValueError("Anchor checkpoint belongs to another Stage 2")
        generated = checkpoint.generated
    else:
        model = OpenAIV2MemoryAnchorModel(
            args.model,
            timeout_seconds=args.timeout_seconds,
        )
        generated = model.generate(stage2)
    anchored, audit = project_anchored_stage2(stage2, generated)
    artifact = build_memory_anchor_artifact(stage2, anchored, generated, audit)
    paths = write_memory_anchor_artifacts(args.output_root, artifact, anchored)
    return {
        "status": "completed",
        "paths": {name: str(path) for name, path in paths.items()},
        "artifact_sha256": artifact.artifact_sha256,
        "audit": audit.model_dump(mode="json"),
        "usage": generated.usage,
        "anchors": [
            item.model_dump(mode="json") for item in generated.payload.anchors
        ],
    }


def main() -> int:
    payload = run(build_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
