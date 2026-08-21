#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

from palmclaw_ubuntu.vehicle_bench.critic_trace import (
    COMBINED_PROMPT_VERSION,
    COMBINED_SCHEMA_VERSION,
    SUMMARY_PROMPT_VERSION,
    SUMMARY_SCHEMA_VERSION,
    CriticTraceSource,
    discover_r2_trace_source,
    extract_source_turns,
)
from palmclaw_ubuntu.vehicle_bench.dataset import load_vehicle_benchmark


EXPECTED_COMMIT = "5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"
AUDIT_VERSION = "r2-final-memory-answerability-v1"
AUDIT_TOOL = {
    "type": "function",
    "name": "record_memory_answerability_audit",
    "description": "Record the answerability classification for all supplied tasks.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["SUPPORTED", "AMBIGUOUS", "MISSING"],
                        },
                        "evidence": {"type": "string"},
                        "missing_or_ambiguous_fact": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "task_id",
                        "verdict",
                        "evidence",
                        "missing_or_ambiguous_fact",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}


INSTRUCTIONS = """You audit whether a frozen vehicle preference memory contains enough information to answer benchmark queries.

For each task, treat only CANDIDATE FINAL MEMORY and QUERY as evidence available to the hypothetical solver. GOLD MEMORY and GOLD TOOL CALLS are answer keys supplied only so you can identify the facts required for the correct answer. Never count a fact as available merely because it appears in GOLD MEMORY or GOLD TOOL CALLS.

Classify conservatively:
- SUPPORTED: every preference fact needed to infer every gold tool argument is explicitly present or faithfully paraphrased in candidate memory or query. Known vehicle tool schemas may be assumed.
- AMBIGUOUS: relevant information exists, but identity, condition, temporal priority, correction, reference, or conflict resolution is insufficient to select the gold answer uniquely.
- MISSING: at least one indispensable preference value or rule is absent, contradicted without resolution, or would require guessing.

Information stated directly in the query is available. Do not require verbatim string matches. Do not grade whether an LLM would actually execute the tools correctly; grade information sufficiency only. Keep evidence and reason concise and never invent support."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=("summary", "combined"),
        default="combined",
    )
    parser.add_argument(
        "--source-repetition",
        type=int,
        choices=(1, 2),
        default=2,
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--consensus-root",
        type=Path,
        help="Use final_memory from consensus-gold-v2 scenario run_summary.json",
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scenario < 1 or args.scenario > 50:
        raise ValueError("Scenario must be between 1 and 50")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    dataset = load_vehicle_benchmark(
        args.benchmark_root,
        expected_commit=EXPECTED_COMMIT,
    )
    scenario = dataset.scenario(args.scenario)
    if args.consensus_root is not None:
        summary_path = (
            args.consensus_root.expanduser().resolve()
            / f"scenario-{args.scenario:02d}"
            / "run_summary.json"
        )
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if run_summary.get("status") != "COMPLETED":
            raise ValueError(
                f"Consensus scenario is not completed: {summary_path} "
                f"({run_summary.get('status')})"
            )
        final_memory = str(run_summary.get("final_memory") or "")
        if not final_memory:
            raise ValueError(f"Consensus final_memory is empty: {summary_path}")
        source_method = "consensus_gold_v2"
        source_repetition = None
        source_database = str(summary_path)
        source_final_memory_sha256 = str(
            run_summary.get("final_memory_sha256")
            or hashlib.sha256(final_memory.encode("utf-8")).hexdigest()
        )
    else:
        if args.artifact_root is None:
            raise ValueError("--artifact-root is required without --consensus-root")
        source = (
            discover_r2_trace_source(
                args.artifact_root,
                method=args.method,
                scenario_index=args.scenario,
            )
            if args.source_repetition == 2
            else discover_r1_trace_source(
                args.artifact_root,
                method=args.method,
                scenario_index=args.scenario,
            )
        )
        turns = extract_source_turns(source)
        final_memory = turns[-1].original_after_memory
        source_method = args.method
        source_repetition = args.source_repetition
        source_database = str(source.database_path)
        source_final_memory_sha256 = turns[-1].original_after_sha256
    task_payload = [
        {
            "task_id": task.id,
            "reasoning_type": task.reasoning_type,
            "query": task.query,
            "gold_memory": task.gold_memory,
            "gold_tool_calls": [call.as_official() for call in task.gold_calls],
        }
        for task in scenario.tasks
    ]
    provider_input = (
        f"SCENARIO: {args.scenario}\n\n"
        f"CANDIDATE FINAL MEMORY:\n{final_memory or '(empty)'}\n\n"
        "TASKS:\n"
        + json.dumps(task_payload, ensure_ascii=False, indent=2)
    )

    client = OpenAI(timeout=300)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            started = time.monotonic()
            response = client.responses.create(
                model=args.model,
                instructions=INSTRUCTIONS,
                input=provider_input,
                tools=[AUDIT_TOOL],
                tool_choice={
                    "type": "function",
                    "name": "record_memory_answerability_audit",
                },
                max_output_tokens=8192,
                reasoning={"effort": "medium"},
                store=False,
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            calls = [
                item
                for item in (getattr(response, "output", ()) or ())
                if getattr(item, "type", None) == "function_call"
                and getattr(item, "name", None)
                == "record_memory_answerability_audit"
            ]
            if len(calls) != 1:
                raise RuntimeError(f"Expected one audit tool call, got {len(calls)}")
            arguments = json.loads(getattr(calls[0], "arguments", "") or "")
            results = arguments.get("results")
            if not isinstance(results, list) or len(results) != len(task_payload):
                raise RuntimeError("Audit result count mismatch")
            expected_ids = [item["task_id"] for item in task_payload]
            actual_ids = [str(item.get("task_id", "")) for item in results]
            if actual_ids != expected_ids:
                raise RuntimeError(
                    f"Audit task order mismatch: {actual_ids} != {expected_ids}"
                )
            break
        except Exception as exc:  # provider retry boundary
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    else:  # pragma: no cover
        raise RuntimeError("Audit failed") from last_error

    usage = getattr(response, "usage", None)
    usage_payload = {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    counts = Counter(str(item["verdict"]) for item in results)
    output = {
        "audit_version": AUDIT_VERSION,
        "scenario_index": args.scenario,
        "method": source_method,
        "source_repetition": source_repetition,
        "source_database": source_database,
        "source_final_memory_sha256": source_final_memory_sha256,
        "model": args.model,
        "latency_ms": latency_ms,
        "usage": usage_payload,
        "counts": dict(counts),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"scenario-{args.scenario:02d}.json"
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(destination), **output["counts"]}))
    return 0


def discover_r1_trace_source(
    artifact_root: Path,
    *,
    method: str,
    scenario_index: int,
) -> CriticTraceSource:
    root = artifact_root.expanduser().resolve()
    if method == "summary":
        if scenario_index <= 5:
            experiment_root = root / "turnwise-summary-s1-s5-20260813"
        elif scenario_index == 6:
            experiment_root = root / "turnwise-s6-full-summary-20260813"
        elif scenario_index <= 10:
            experiment_root = root / "turnwise-summary-s7-s10-20260813"
        elif scenario_index <= 20:
            experiment_root = root / "turnwise-summary-s11-s20-20260813"
        else:
            experiment_root = (
                root
                / f"turnwise-summary-fresh-r1-s{scenario_index}-20260815"
            )
        expected_prompt = SUMMARY_PROMPT_VERSION
        expected_schema = SUMMARY_SCHEMA_VERSION
    elif method == "combined":
        if scenario_index <= 10:
            date_tag = "20260814" if scenario_index in {2, 6} else "20260815"
            experiment_root = (
                root
                / f"turnwise-temporal-compact-s{scenario_index}-v1-{date_tag}"
            )
        else:
            experiment_root = (
                root
                / f"turnwise-combined-fresh-r1-s{scenario_index}-20260815"
            )
        expected_prompt = COMBINED_PROMPT_VERSION
        expected_schema = COMBINED_SCHEMA_VERSION
    else:
        raise ValueError(f"Unsupported method: {method}")

    scenario_part = f"scenario-{scenario_index:02d}"
    database_paths = sorted(
        path
        for path in (experiment_root / "cache").rglob("memory.db")
        if scenario_part in path.parts
    )
    if len(database_paths) != 1:
        raise ValueError(
            f"Expected one R1 {method} database for S{scenario_index}, "
            f"found {len(database_paths)} under {experiment_root}"
        )
    database_path = database_paths[0].resolve()
    manifest_path = database_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    recursive = config["recursive_summary"]
    history = manifest["history"]
    actual_versions = (
        str(recursive["prompt_version"]),
        str(recursive["schema_version"]),
    )
    if actual_versions != (expected_prompt, expected_schema):
        raise ValueError(
            f"Unexpected R1 {method} versions for S{scenario_index}: "
            f"{actual_versions}"
        )
    if recursive.get("update_cadence") != "history_entry":
        raise ValueError(f"R1 {method} S{scenario_index} is not turn-wise")
    return CriticTraceSource(
        method=method,
        scenario_index=scenario_index,
        experiment_root=experiment_root.resolve(),
        database_path=database_path,
        cache_manifest_path=manifest_path.resolve(),
        cache_key=str(manifest["cache_key"]),
        dataset_sha256=str(config["dataset_sha256"]),
        history_sha256=str(config["history_sha256"]),
        history_line_count=int(history["line_count"]),
        model_id=str(recursive["model_id"]),
        prompt_version=str(recursive["prompt_version"]),
        schema_version=str(recursive["schema_version"]),
        max_memory_chars=int(recursive["max_memory_chars"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
