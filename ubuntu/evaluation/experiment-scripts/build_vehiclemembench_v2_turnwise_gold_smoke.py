#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.critic_trace import (
    discover_r2_trace_source,
    extract_source_turns,
)
from palmclaw_ubuntu.vehicle_bench.dataset import load_vehicle_benchmark
from palmclaw_ubuntu.vehicle_bench.memory import parse_vehicle_history
from palmclaw_ubuntu.vehicle_bench.turnwise_gold import (
    align_vehicle_oracle_events,
    apply_reviewed_alignment_overrides,
    build_turnwise_canonical_labels,
    build_turnwise_gold_smoke_audit,
    build_vehicle_oracle_ledger,
    write_turnwise_gold_smoke_artifacts,
)

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_ARTIFACT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2/"
    "turnwise-memory-gold-v1-smoke"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only VehicleMemBench turn-wise gold oracle/alignment "
            "smoke artifact."
        )
    )
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--alignment-overrides",
        type=Path,
        help="Reviewed event-to-turn alignment JSON for canonical replay.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = load_vehicle_benchmark(args.dataset_root, strict=True)
    scenario = dataset.scenario(args.scenario)
    entries = tuple(
        sorted(
            parse_vehicle_history(scenario.history_path),
            key=lambda entry: (entry.timestamp, entry.line_number),
        )
    )
    sources = {
        method: discover_r2_trace_source(
            args.artifact_root,
            method=method,
            scenario_index=args.scenario,
        )
        for method in ("summary", "combined")
    }
    trace_turns = {
        method: extract_source_turns(source) for method, source in sources.items()
    }
    _validate_trace_history(entries, trace_turns)
    trace_hashes = {source.dataset_sha256 for source in sources.values()}
    if trace_hashes != {dataset.manifest.dataset_sha256}:
        raise ValueError(
            "Dataset/source trace hash mismatch: "
            f"dataset={dataset.manifest.dataset_sha256} traces={sorted(trace_hashes)}"
        )

    events = build_vehicle_oracle_ledger(scenario, entries)
    alignments = align_vehicle_oracle_events(
        events,
        entries,
        trace_turns=trace_turns,
    )
    if args.alignment_overrides is not None:
        overrides = _load_alignment_overrides(
            args.alignment_overrides,
            scenario_index=args.scenario,
            dataset_sha256=dataset.manifest.dataset_sha256,
        )
        alignments = apply_reviewed_alignment_overrides(
            alignments,
            overrides,
            entries,
            trace_turns=trace_turns,
        )
    audit = build_turnwise_gold_smoke_audit(
        scenario=scenario,
        history_entries=entries,
        events=events,
        alignments=alignments,
    )
    summary_labels = None
    combined_labels = None
    if audit["ready_for_canonical_replay"]:
        summary_labels = build_turnwise_canonical_labels(
            method="summary",
            history_entries=entries,
            trace_turns=trace_turns["summary"],
            events=events,
            alignments=alignments,
        )
        combined_labels = build_turnwise_canonical_labels(
            method="combined",
            history_entries=entries,
            trace_turns=trace_turns["combined"],
            events=events,
            alignments=alignments,
        )
        if summary_labels[-1]["after_memory_sha256"] != combined_labels[-1][
            "after_memory_sha256"
        ]:
            raise RuntimeError("Summary/Combined canonical final memory differs")
    output_dir = (
        args.output_root.expanduser().resolve() / f"scenario-{args.scenario:02d}"
    )
    write_turnwise_gold_smoke_artifacts(
        output_dir,
        scenario=scenario,
        history_entries=entries,
        events=events,
        alignments=alignments,
        audit=audit,
        source_metadata={
            "dataset_sha256": dataset.manifest.dataset_sha256,
            "tool_schema_sha256": dataset.manifest.tool_schema_sha256,
            "traces": {
                method: {
                    "database_path": str(source.database_path),
                    "cache_key": source.cache_key,
                    "model_id": source.model_id,
                    "prompt_version": source.prompt_version,
                    "schema_version": source.schema_version,
                }
                for method, source in sources.items()
            },
        },
        summary_labels=summary_labels,
        combined_labels=combined_labels,
    )
    print(json.dumps({"output_dir": str(output_dir), **audit}, indent=2))
    return 0


def _load_alignment_overrides(
    path: Path,
    *,
    scenario_index: int,
    dataset_sha256: str,
) -> dict[str, int]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if payload.get("version") != "vehiclemembench-turnwise-gold-alignment-v1":
        raise ValueError("Unsupported alignment override version")
    if payload.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Alignment override dataset hash mismatch")
    if payload.get("scenario_index") != scenario_index:
        raise ValueError("Alignment override scenario mismatch")
    rows = payload.get("reviewed_alignments")
    if not isinstance(rows, list):
        raise ValueError("Alignment overrides must contain reviewed_alignments")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Alignment override row must be an object")
        event_id = row.get("event_id")
        turn_index = row.get("turn_index")
        if not isinstance(event_id, str) or not isinstance(turn_index, int):
            raise ValueError("Alignment override row is incomplete")
        if event_id in result:
            raise ValueError(f"Duplicate alignment override: {event_id}")
        result[event_id] = turn_index
    return result


def _validate_trace_history(entries, trace_turns) -> None:
    for method, turns in trace_turns.items():
        if len(turns) != len(entries):
            raise ValueError(
                f"{method} trace/history length mismatch: "
                f"{len(turns)} != {len(entries)}"
            )
        for turn_index, (entry, turn) in enumerate(zip(entries, turns, strict=True)):
            marker, separator, raw = turn.content.partition("\n")
            if not separator or raw != entry.raw:
                raise ValueError(
                    f"{method} trace/history content mismatch at turn {turn_index}: "
                    f"{marker}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
