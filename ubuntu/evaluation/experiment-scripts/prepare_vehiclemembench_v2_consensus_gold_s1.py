#!/usr/bin/env python3
"""Preflight Scenario 1 for the VehicleMemBench V2 consensus-Gold pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.consensus_gold import (
    CONSENSUS_GOLD_PROMPT_VERSION,
    CONSENSUS_GOLD_SCHEMA_VERSION,
    build_gold_generation_input,
    text_sha256,
)
from palmclaw_ubuntu.vehicle_bench.dataset import load_vehicle_benchmark
from palmclaw_ubuntu.vehicle_bench.memory import parse_vehicle_history

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
PREFLIGHT_VERSION = "vehiclemembench-v2-consensus-gold-s1-preflight-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--scenario", type=int, choices=(1,), default=1)
    parser.add_argument(
        "--turn-limit",
        type=int,
        help="Limit the source prefix for a later canary; omit for all S1 turns.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.turn_limit is not None and args.turn_limit < 1:
        raise ValueError("--turn-limit must be positive")
    dataset = load_vehicle_benchmark(args.dataset_root, strict=True)
    scenario = dataset.scenario(args.scenario)
    all_entries = tuple(
        sorted(
            parse_vehicle_history(scenario.history_path),
            key=lambda entry: (entry.timestamp, entry.line_number),
        )
    )
    entries = all_entries[: args.turn_limit]
    first_input = build_gold_generation_input(
        scenario_index=args.scenario,
        turn_index=0,
        entry=entries[0],
        previous_memory="",
    )
    source_hashes = [text_sha256(entry.raw) for entry in entries]
    source_fingerprint = _json_sha256(
        {
            "dataset_sha256": dataset.manifest.dataset_sha256,
            "scenario_index": args.scenario,
            "source_hashes": source_hashes,
        }
    )
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "schema_version": CONSENSUS_GOLD_SCHEMA_VERSION,
        "prompt_version": CONSENSUS_GOLD_PROMPT_VERSION,
        "scenario_index": args.scenario,
        "dataset_root": str(dataset.root),
        "dataset_sha256": dataset.manifest.dataset_sha256,
        "history_path": str(scenario.history_path),
        "history_sha256": _file_sha256(scenario.history_path),
        "source_fingerprint": source_fingerprint,
        "full_turn_count": len(all_entries),
        "selected_turn_count": len(entries),
        "base_luna_calls": len(entries) * 3,
        "additional_calls": "3 per compaction plus retries and escalation",
        "models": {
            "luna": {"model": "gpt-5.6-luna", "reasoning_effort": "medium"},
            "terra": {"model": "gpt-5.6-terra", "reasoning_effort": "high"},
            "sol": {"model": "gpt-5.6-sol", "reasoning_effort": "high"},
        },
        "first_turn": {
            "message_id": first_input.message_id,
            "source_sha256": first_input.source_sha256,
            "input_sha256": first_input.input_sha256,
            "privacy": first_input.privacy,
        },
        "forbidden_generator_inputs": [
            "qa_gold_memory",
            "qa_query",
            "qa_gold_answer",
            "existing_trace",
            "future_turns",
        ],
    }


def main() -> int:
    payload = run(build_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text_sha256(encoded)


if __name__ == "__main__":
    raise SystemExit(main())
