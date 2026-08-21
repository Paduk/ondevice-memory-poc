#!/usr/bin/env python3
"""Prepare paper-first Persona/Event generation without calling an LLM API."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.dataset import (
    OFFICIAL_UPSTREAM_COMMIT,
    load_vehicle_benchmark,
)
from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1_EVENT_CHAIN_INSTRUCTIONS,
    V1_EVENT_CHAIN_PROMPT_VERSION,
    V1_PERSONA_ENRICHMENT_INSTRUCTIONS,
    V1_PERSONA_PROMPT_VERSION,
    V1_STAGE2_SCHEMA_VERSION,
    build_persona_seed_groups,
    build_reasoning_type_plan,
    build_vehicle_attribute_catalog,
    load_persona_seeds,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import canonical_json_sha256

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_OUTPUT_PATH = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "stage2-preflight.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-scenarios", type=int, default=100)
    parser.add_argument(
        "--persona-seeds",
        type=Path,
        help="Normalized Persona-Hub elite seed JSONL; no API is called.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_scenarios < 1:
        raise ValueError("--target-scenarios must be positive")
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    candidate_groups = args.target_scenarios * 2
    required_persona_seeds = candidate_groups * 3
    catalog = build_vehicle_attribute_catalog(dataset.tool_schemas)
    reasoning_plan = build_reasoning_type_plan(args.target_scenarios)
    reasoning_counts = Counter(
        item for scenario_plan in reasoning_plan for item in scenario_plan
    )
    seed_status: dict[str, Any]
    if args.persona_seeds is None:
        seed_status = {
            "status": "required",
            "required_seed_count": required_persona_seeds,
        }
    else:
        seeds = load_persona_seeds(args.persona_seeds)
        groups = build_persona_seed_groups(
            seeds,
            candidate_group_count=candidate_groups,
        )
        seed_status = {
            "status": "ready",
            "source_path": str(args.persona_seeds.resolve()),
            "available_seed_count": len(seeds),
            "selected_seed_count": sum(len(group) for group in groups),
            "group_plan_sha256": canonical_json_sha256(
                [[seed.seed_id for seed in group] for group in groups]
            ),
        }
    payload = {
        "schema_version": V1_STAGE2_SCHEMA_VERSION,
        "source": {
            "commit": dataset.manifest.upstream_commit,
            "dataset_sha256": dataset.manifest.dataset_sha256,
            "tool_schema_sha256": dataset.manifest.tool_schema_sha256,
        },
        "paper_contract": {
            "persona_group_size": 3,
            "background_chains_per_scenario": 20,
            "vehicle_chains_per_scenario": 10,
            "reported_average_events_per_scenario": 81.78,
            "event_time_span": "weeks_or_months",
        },
        "project_validation": {
            "minimum_events_per_scenario": 80,
            "minimum_time_span_days": 14,
            "provenance": "project_defined interpretation of paper constraints",
        },
        "project_run": {
            "target_scenarios": args.target_scenarios,
            "candidate_persona_groups": candidate_groups,
            "model": V1_DEFAULT_GENERATION_MODEL,
            "reasoning_effort": "medium",
            "temperature": None,
            "model_provenance": "project_defined substitution",
        },
        "prompts": {
            "persona": {
                "version": V1_PERSONA_PROMPT_VERSION,
                "sha256": canonical_json_sha256(
                    {"instructions": V1_PERSONA_ENRICHMENT_INSTRUCTIONS}
                ),
            },
            "event_chain": {
                "version": V1_EVENT_CHAIN_PROMPT_VERSION,
                "sha256": canonical_json_sha256(
                    {"instructions": V1_EVENT_CHAIN_INSTRUCTIONS}
                ),
            },
        },
        "vehicle_catalog": {
            "tool_count": len(dataset.tool_schemas),
            "attribute_path_count": len(catalog),
            "module_count": len({item.module for item in catalog}),
            "sha256": canonical_json_sha256(
                [item.model_dump(mode="json") for item in catalog]
            ),
        },
        "reasoning_plan": {
            "scenario_count": len(reasoning_plan),
            "counts": dict(sorted(reasoning_counts.items())),
            "sha256": canonical_json_sha256(reasoning_plan),
        },
        "persona_seeds": seed_status,
        "api_calls_made": 0,
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    payload = run(build_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
