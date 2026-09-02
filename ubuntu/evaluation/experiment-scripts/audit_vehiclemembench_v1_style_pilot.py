#!/usr/bin/env python3
"""Audit one V1-style augmentation pilot and estimate its Terra cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_training.methods.operations import apply_operations
from memory_training.prepare_grouped_memory_data import (
    group_memory,
    parse_flat_memory,
    transform_operations,
)

from palmclaw_ubuntu.vehicle_bench.dataset import (
    OFFICIAL_UPSTREAM_COMMIT,
    load_vehicle_benchmark,
)
from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime
from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1Stage2Artifact,
    build_vehicle_attribute_catalog,
    validate_stage2_contract,
    validate_stage2_simulator_arguments,
    validate_state_evolution_coverage,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import (
    V1Stage3Artifact,
    build_stage3_artifact,
)
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2HybridArtifact,
    build_hybrid_artifact,
)
from palmclaw_ubuntu.vehicle_bench.v2_training_export import (
    build_training_views,
    load_canonical_training_turns,
)

DEFAULT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/"
    "vehiclemembench-v1-style-augmentation-v1"
)
WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/hj153lee/VehicleMemBench"))
    parser.add_argument("--selection", type=Path, default=DEFAULT_ROOT / "source-selection.json")
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_ROOT / "pilot-s44-r1")
    parser.add_argument("--terra-input-per-million", type=float, default=2.0)
    parser.add_argument("--terra-cached-input-per-million", type=float, default=0.2)
    parser.add_argument("--terra-output-per-million", type=float, default=12.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    selection_path = args.selection.expanduser().resolve(strict=True)
    selection = _read_json(selection_path)
    blueprint = selection["pilot_blueprint"]
    source_index = int(blueprint["scenario"])
    source = dataset.scenarios[source_index - 1]
    pilot_root = args.pilot_root.expanduser().resolve(strict=True)
    stage2_path = pilot_root / "scenario" / "stage2-v2-anchored.json"
    hybrid_path = pilot_root / "scenario" / "hybrid.json"
    stage3_path = pilot_root / "scenario" / "final-v1" / "stage3.json"
    provenance_path = pilot_root / "stage2" / "style-provenance.json"

    stage2 = V1Stage2Artifact.model_validate_json(stage2_path.read_text(encoding="utf-8"))
    hybrid = V2HybridArtifact.model_validate_json(hybrid_path.read_text(encoding="utf-8"))
    stage3 = V1Stage3Artifact.model_validate_json(stage3_path.read_text(encoding="utf-8"))
    provenance = _read_json(provenance_path)
    runtime = VehicleWorldRuntime(dataset.root)
    catalog = tuple(
        item
        for item in build_vehicle_attribute_catalog(dataset.tool_schemas)
        if item.tool_name in set(blueprint["tool_counts"])
    )

    validate_stage2_contract(
        personas=stage2.persona_group.payload.personas,
        event_payload=stage2.event_chains.payload,
        planned_reasoning_types=tuple(blueprint["reasoning_types_in_quiz_order"]),
        vehicle_attributes=catalog,
    )
    validate_stage2_simulator_arguments(stage2.event_chains.payload, runtime=runtime)
    coverage = validate_state_evolution_coverage(stage2.event_chains.payload)
    rebuilt_hybrid = build_hybrid_artifact(stage2, hybrid.event_checkpoints)
    if rebuilt_hybrid.model_dump(mode="json") != hybrid.model_dump(mode="json"):
        raise ValueError("Hybrid artifact cannot be deterministically rebuilt")
    rebuilt_stage3 = build_stage3_artifact(
        stage2,
        stage3.generated_dialogues,
        stage3.generated_quizzes,
        scenario_id=stage3.scenario.scenario_id,
        tool_schemas=dataset.tool_schemas,
        runtime=runtime,
    )
    if rebuilt_stage3.model_dump(mode="json") != stage3.model_dump(mode="json"):
        raise ValueError("Stage 3 artifact cannot be deterministically rebuilt")

    expected_hashes = {
        "history": _sha256(source.history_path),
        "qa": _sha256(source.qa_path),
    }
    provenance_ok = (
        provenance["source_scenario"] == source_index
        and provenance["source_history_sha256"] == expected_hashes["history"]
        and provenance["source_qa_sha256"] == expected_hashes["qa"]
        and provenance["copied_source_content"] is False
    )
    if not provenance_ok:
        raise ValueError("Pilot provenance is inconsistent with the frozen source")

    source_lines = _dialogue_text_lines(source.history_path.read_text(encoding="utf-8"))
    generated_history = pilot_root / "scenario" / "final-v1" / "benchmark" / "history" / "history_301.txt"
    generated_lines = _dialogue_text_lines(generated_history.read_text(encoding="utf-8"))
    exact_overlap = len(set(source_lines) & set(generated_lines))
    ngram_overlap = _ngram_overlap(source_lines, generated_lines, n=8)
    if exact_overlap:
        raise ValueError(f"Generated dialogue copied {exact_overlap} exact source lines")
    if ngram_overlap > 0.01:
        raise ValueError(f"Generated/source 8-gram overlap is too high: {ngram_overlap:.4f}")

    source_turns = int(blueprint["turns"])
    generated_turns = stage3.audit.dialogue_turn_count
    turn_ratio = generated_turns / max(source_turns, 1)
    if not 0.75 <= turn_ratio <= 1.25:
        raise ValueError(
            f"Pilot turn count is outside the source-style tolerance: {turn_ratio:.3f}"
        )
    if not (hybrid.audit.completed and stage3.audit.passed):
        raise ValueError("Hybrid or final V1 audit did not complete")

    usage = _aggregate_usage(pilot_root, stage2, hybrid, stage3)
    uncached = max(usage["input_tokens"] - usage["cached_tokens"], 0)
    recorded_cost = (
        uncached / 1_000_000 * args.terra_input_per_million
        + usage["cached_tokens"] / 1_000_000 * args.terra_cached_input_per_million
        + usage["output_tokens"] / 1_000_000 * args.terra_output_per_million
    )
    estimated_uncached = max(
        usage["retry_adjusted_input_tokens"]
        - usage["retry_adjusted_cached_tokens"],
        0,
    )
    retry_adjusted_cost = (
        estimated_uncached / 1_000_000 * args.terra_input_per_million
        + usage["retry_adjusted_cached_tokens"]
        / 1_000_000
        * args.terra_cached_input_per_million
        + usage["retry_adjusted_output_tokens"]
        / 1_000_000
        * args.terra_output_per_million
    )
    operation_counts = Counter(
        operation.op
        for checkpoint in hybrid.event_checkpoints
        for label in checkpoint.turn_labels
        for operation in label.operations
    )
    loaded_hybrid, canonical_turns = load_canonical_training_turns(
        pilot_root / "scenario"
    )
    training_views = build_training_views(
        scenario_index=301,
        run_id=stage3.scenario.scenario_id,
        source_hybrid_sha256=loaded_hybrid.artifact_sha256,
        turns=canonical_turns,
        split="diagnostic",
        compaction_interval=5,
    )
    view_names = ("summary", "patch", "delta", "compaction")
    training_counts = {
        name: len(rows) for name, rows in zip(view_names, training_views, strict=True)
    }
    grouped_preflight = _grouped_preflight(
        training_views[0],
        training_views[1],
        stage2=stage2,
    )
    preflight_path = pilot_root / "training-jsonl-preflight.jsonl"
    with preflight_path.open("w", encoding="utf-8") as handle:
        for name, rows in zip(view_names, training_views, strict=True):
            if rows:
                handle.write(
                    json.dumps(
                        {"view": name, "sample": rows[0]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    report = {
        "schema_version": "vehiclemembench-v1-style-pilot-audit-v1",
        "passed": True,
        "source_scenario": source_index,
        "provenance": {
            "passed": provenance_ok,
            "source_history_sha256": expected_hashes["history"],
            "source_qa_sha256": expected_hashes["qa"],
            "exact_dialogue_line_overlap": exact_overlap,
            "source_generated_8gram_overlap": ngram_overlap,
        },
        "structure": {
            "source_turns": source_turns,
            "generated_turns": generated_turns,
            "turn_ratio": turn_ratio,
            "persona_count": len(stage2.persona_group.payload.personas),
            "event_count": len(stage2.interleaved_timeline),
            "quiz_count": stage3.audit.quiz_count,
        },
        "memory": {
            "turn_labels": hybrid.audit.turn_label_count,
            "no_op": hybrid.audit.no_op_count,
            "update": hybrid.audit.update_count,
            "operation_counts": dict(sorted(operation_counts.items())),
            "state_evolution_coverage": coverage.model_dump(mode="json"),
            "deterministic_rebuild_passed": True,
            "final_memory_sha256": hybrid.final_memory_sha256,
        },
        "quiz": {
            "count": stage3.audit.quiz_count,
            "simulator_executed_call_count": stage3.audit.simulator_executed_call_count,
            "simulator_passed_quiz_count": stage3.audit.simulator_passed_quiz_count,
            "deterministic_rebuild_passed": True,
        },
        "training_jsonl_preflight": {
            "passed": True,
            "counts": training_counts,
            "sample_path": str(preflight_path),
            "inline_json": True,
            "grouped_memory": grouped_preflight,
        },
        "usage": usage,
        "pricing": {
            "model": "gpt-5.6-terra",
            "input_per_1m": args.terra_input_per_million,
            "cached_input_per_1m": args.terra_cached_input_per_million,
            "output_per_1m": args.terra_output_per_million,
            "recorded_successful_cost_usd": recorded_cost,
            "retry_adjusted_cost_estimate_usd": retry_adjusted_cost,
            "note": (
                "Recorded cost includes successful responses. Retry-adjusted cost "
                "multiplies each saved response by generation_attempts because rejected "
                "responses do not retain exact usage. Input price applies only to "
                "non-cached tokens."
            ),
        },
    }
    _write_json(pilot_root / "pilot-audit.json", report)
    (pilot_root / "pilot-audit.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _aggregate_usage(
    pilot_root: Path,
    stage2: V1Stage2Artifact,
    hybrid: V2HybridArtifact,
    stage3: V1Stage3Artifact,
) -> dict[str, Any]:
    raw_stages: dict[str, list[dict[str, int]]] = {
        "stage2_persona": [stage2.persona_group.usage],
        "stage2_events": [stage2.event_chains.usage],
        "dialogues": [item.usage for item in stage3.generated_dialogues],
        "memory_alignment": [
            checkpoint.generated_alignment.usage for checkpoint in hybrid.event_checkpoints
        ],
        "final_quiz": [item.usage for item in stage3.generated_quizzes],
    }
    anchor_path = pilot_root / "scenario" / "memory-anchors.json"
    if anchor_path.exists():
        raw_stages["memory_anchors"] = [
            _read_json(anchor_path).get("generated", {}).get("usage", {})
        ]
    stages = {name: _sum_usage(rows) for name, rows in raw_stages.items()}
    retry_adjusted_stages = {
        name: _sum_usage(rows, apply_attempts=True)
        for name, rows in raw_stages.items()
    }
    keys = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "latency_ms")
    totals = {key: sum(stage.get(key, 0) for stage in stages.values()) for key in keys}
    adjusted = {
        f"retry_adjusted_{key}": sum(stage.get(key, 0) for stage in retry_adjusted_stages.values())
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens")
    }
    return {
        "by_stage": stages,
        "retry_adjusted_by_stage": retry_adjusted_stages,
        **totals,
        **adjusted,
    }


def _grouped_preflight(
    summary_rows: Iterable[dict[str, Any]],
    patch_rows: Iterable[dict[str, Any]],
    *,
    stage2: V1Stage2Artifact,
) -> dict[str, Any]:
    summaries = tuple(summary_rows)
    patches = tuple(patch_rows)
    if len(summaries) != len(patches):
        raise ValueError("Summary/Patch preflight row counts differ")
    subject_ids = {
        persona.name: persona.persona_id
        for persona in stage2.persona_group.payload.personas
    }
    subject_order = []
    seen = set()
    for row in summaries:
        for fact in parse_flat_memory(row["target"]["next_memory"]):
            if fact.subject not in seen:
                seen.add(fact.subject)
                subject_order.append(fact.subject)
    replayed_rows = 0
    example = None
    for summary, patch in zip(summaries, patches, strict=True):
        previous_flat = patch["input"]["previous_memory"]
        next_flat = summary["target"]["next_memory"]
        previous_grouped = group_memory(
            previous_flat,
            subject_ids=subject_ids,
            subject_order=subject_order,
        )
        next_grouped = group_memory(
            next_flat,
            subject_ids=subject_ids,
            subject_order=subject_order,
        )
        grouped_operations = transform_operations(
            previous_flat,
            patch["target"]["operations"],
            subject_ids=subject_ids,
            subject_order=subject_order,
        )
        if grouped_operations:
            replayed, _ = apply_operations(previous_grouped, grouped_operations)
        else:
            replayed = previous_grouped
        if replayed != next_grouped:
            raise ValueError(f"Grouped-memory replay failed at {patch['sample_id']}")
        replayed_rows += 1
        if example is None and grouped_operations:
            example = {
                "sample_id": patch["sample_id"],
                "previous_memory": previous_grouped,
                "operations": grouped_operations,
                "next_memory": next_grouped,
            }
    if not subject_order:
        raise ValueError("Grouped-memory preflight found no memory subjects")
    return {
        "passed": True,
        "replayed_rows": replayed_rows,
        "subject_order": subject_order,
        "speaker_id_in_memory": False,
        "shared_vehicle_separate_block": "shared-vehicle" in subject_order,
        "example": example,
    }


def _usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        key: int(value.get(key, 0) or 0)
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "latency_ms")
    }


def _sum_usage(
    values: Iterable[dict[str, int]], *, apply_attempts: bool = False
) -> dict[str, int]:
    raw_rows = list(values)
    rows = [_usage(value) for value in raw_rows]
    return {
        key: sum(
            row.get(key, 0)
            * (
                max(int(raw.get("generation_attempts", 1) or 1), 1)
                if apply_attempts and key != "latency_ms"
                else 1
            )
            for row, raw in zip(rows, raw_rows, strict=True)
        )
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "latency_ms")
    }


def _dialogue_text_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        value = line.split(": ", 1)[-1].strip().casefold()
        if value:
            lines.append(value)
    return lines


def _ngram_overlap(source_lines: list[str], generated_lines: list[str], *, n: int) -> float:
    source = _ngrams(source_lines, n=n)
    generated = _ngrams(generated_lines, n=n)
    return len(source & generated) / max(len(generated), 1)


def _ngrams(lines: list[str], *, n: int) -> set[tuple[str, ...]]:
    result = set()
    for line in lines:
        words = WORD_RE.findall(line)
        result.update(tuple(words[index : index + n]) for index in range(len(words) - n + 1))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    usage = report["usage"]
    memory = report["memory"]
    structure = report["structure"]
    return (
        "# V1-style augmentation pilot audit\n\n"
        f"- Result: **PASS**\n"
        f"- Source: S{report['source_scenario']}\n"
        f"- Turns: {structure['generated_turns']:,} (source {structure['source_turns']:,}, ratio {structure['turn_ratio']:.3f})\n"
        f"- Memory labels: UPDATE {memory['update']:,} / NO_OP {memory['no_op']:,}\n"
        f"- Operations: `{json.dumps(memory['operation_counts'], sort_keys=True)}`\n"
        f"- Final Quiz: {report['quiz']['count']} (simulator pass {report['quiz']['simulator_passed_quiz_count']})\n"
        f"- Tokens: input {usage['input_tokens']:,}, cached {usage['cached_tokens']:,}, output {usage['output_tokens']:,}\n"
        f"- Recorded API latency: {usage['latency_ms'] / 60000:.2f} min\n"
        f"- Recorded successful Terra cost: **${report['pricing']['recorded_successful_cost_usd']:.3f}**\n"
        f"- Retry-adjusted Terra estimate: **${report['pricing']['retry_adjusted_cost_estimate_usd']:.3f}**\n"
    )


def main() -> int:
    report = run(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
