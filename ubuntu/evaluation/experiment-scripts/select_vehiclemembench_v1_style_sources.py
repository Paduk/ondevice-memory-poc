#!/usr/bin/env python3
"""Select 20 V1-style augmentation references and one representative pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.dataset import (
    OFFICIAL_UPSTREAM_COMMIT,
    load_vehicle_benchmark,
)

DEFAULT_EVALUATION = Path(
    "/mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/"
    "qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-r1/"
    "test-hf-v1-s1-s50-best-epoch-04"
)
DEFAULT_TRACE_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
)
DEFAULT_OUTPUT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-style-augmentation-v1"
)
EXCLUDED_TRAIN = (2, 5, 6, 14, 17, 23, 31, 32, 33, 36)
TRACE_PATTERN = "turnwise-patch-soft30-fresh-r2-s{scenario}-20260816"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/hj153lee/VehicleMemBench")
    )
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=20)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    evaluation_root = args.evaluation_root.expanduser().resolve(strict=True)
    trace_root = args.trace_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    quiz_report = json.loads(
        (evaluation_root / "predicted-memory-quiz.json").read_text(encoding="utf-8")
    )
    quiz_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in quiz_report["records"]:
        quiz_records[int(record["scenario_index"])].append(record)

    rows = []
    for scenario in dataset.scenarios:
        report_path = evaluation_root / "scenarios" / f"s{scenario.index:03d}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        records = quiz_records[scenario.index]
        if len(records) != len(scenario.tasks):
            raise ValueError(f"S{scenario.index}: Quiz record count mismatch")
        reasoning = Counter(task.reasoning_type for task in scenario.tasks)
        tool_names = Counter(
            call.name for task in scenario.tasks for call in task.gold_calls
        )
        tool_modules = Counter(
            name.removeprefix("carcontrol_").split("_", 1)[0]
            for name in tool_names.elements()
        )
        turns = int(report["turns"])
        invalid = int(report.get("decision_counts", {}).get("INVALID", 0))
        esm = _mean(records, "exact_state_match")
        tool_f1 = _mean(records, "tool_f1")
        arg_exact = _mean(records, "arg_exact")
        trace = _trace_stats(trace_root, scenario.index)
        difficulty = (
            0.50 * (1.0 - esm)
            + 0.20 * (1.0 - tool_f1)
            + 0.20 * (1.0 - arg_exact)
            + 0.10 * min((invalid / max(turns, 1)) / 0.10, 1.0)
        )
        rows.append(
            {
                "scenario": scenario.index,
                "excluded_existing_train": scenario.index in EXCLUDED_TRAIN,
                "turns": turns,
                "esm": esm,
                "tool_f1": tool_f1,
                "arg_exact": arg_exact,
                "invalid_count": invalid,
                "invalid_rate": invalid / max(turns, 1),
                "cloud_trace_available": trace["available"],
                "cloud_trace_updates": trace["updates"],
                "cloud_trace_noops": trace["noops"],
                "reasoning_counts": dict(sorted(reasoning.items())),
                "reasoning_types_in_quiz_order": [
                    task.reasoning_type for task in scenario.tasks
                ],
                "tool_counts": dict(sorted(tool_names.items())),
                "tool_modules": dict(sorted(tool_modules.items())),
                "difficulty_score": difficulty,
                "history_sha256": _sha256(scenario.history_path),
                "qa_sha256": _sha256(scenario.qa_path),
            }
        )

    candidates = [row for row in rows if not row["excluded_existing_train"]]
    selection_pool = [row for row in candidates if row["cloud_trace_available"]]
    if not 1 <= args.count < len(selection_pool):
        raise ValueError("--count must leave at least one clean held-out scenario")
    selected = _balanced_hard_selection(selection_pool, args.count)
    selected_ids = {row["scenario"] for row in selected}
    heldout = [row for row in candidates if row["scenario"] not in selected_ids]
    pilot = _choose_pilot(selected)
    manifest = {
        "schema_version": "vehiclemembench-v1-style-source-selection-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.manifest.as_dict(),
        "evaluation_root": str(evaluation_root),
        "trace_root": str(trace_root),
        "existing_training_scenarios": list(EXCLUDED_TRAIN),
        "selection_policy": {
            "candidate_pool": "V1 S1-S50 excluding the existing V1-10 training set",
            "trusted_trace_gate": "only completed Cloud Patch R2 traces may become augmentation sources",
            "difficulty": "50% ESM + 20% Tool F1 + 20% Arg Exact error + 10% capped invalid rate",
            "diversity": "greedy deficits against 50% of candidate reasoning/tool-module totals",
            "pilot": "highest difficulty among non-pathological, IQR-update candidates",
        },
        "selected_source_scenarios": [row["scenario"] for row in selected],
        "clean_heldout_scenarios": [row["scenario"] for row in heldout],
        "pilot_source_scenario": pilot["scenario"],
        "pilot_blueprint": pilot,
        "selected": selected,
        "clean_heldout": heldout,
        "all_scenarios": rows,
    }
    output = output_root / "source-selection.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "source-selection.md").write_text(
        _markdown_summary(manifest), encoding="utf-8"
    )
    return manifest


def _balanced_hard_selection(
    candidates: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    reason_total = Counter()
    module_total = Counter()
    for row in candidates:
        reason_total.update(row["reasoning_counts"])
        module_total.update(row["tool_modules"])
    reason_target = {key: value * count / len(candidates) for key, value in reason_total.items()}
    module_target = {key: value * count / len(candidates) for key, value in module_total.items()}
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    reason_seen = Counter()
    module_seen = Counter()
    while len(selected) < count:
        best = None
        best_score = -math.inf
        for row in candidates:
            if row["scenario"] in selected_ids:
                continue
            reason_bonus = _deficit_bonus(row["reasoning_counts"], reason_seen, reason_target)
            module_bonus = _deficit_bonus(row["tool_modules"], module_seen, module_target)
            score = row["difficulty_score"] + 0.12 * reason_bonus + 0.08 * module_bonus
            key = (score, row["difficulty_score"], -row["scenario"])
            if best is None or key > best_score:
                best = row
                best_score = key
        assert best is not None
        selected.append(best)
        selected_ids.add(best["scenario"])
        reason_seen.update(best["reasoning_counts"])
        module_seen.update(best["tool_modules"])
    return sorted(selected, key=lambda row: row["scenario"])


def _deficit_bonus(counts: dict[str, int], seen: Counter, target: dict[str, float]) -> float:
    total = sum(counts.values()) or 1
    return sum(
        amount / total * max(target.get(key, 0.0) - seen[key], 0.0) / max(target.get(key, 1.0), 1.0)
        for key, amount in counts.items()
    )


def _choose_pilot(selected: list[dict[str, Any]]) -> dict[str, Any]:
    updates = sorted(row["cloud_trace_updates"] for row in selected)
    low = updates[len(updates) // 4]
    high = updates[(3 * len(updates)) // 4]
    eligible = [
        row
        for row in selected
        if low <= row["cloud_trace_updates"] <= high
        and row["invalid_rate"] <= 0.20
        and len(row["reasoning_counts"]) >= 3
        and len(row["tool_modules"]) >= 3
    ]
    if not eligible:
        eligible = selected
    return max(eligible, key=lambda row: (row["difficulty_score"], -row["scenario"]))


def _trace_stats(root: Path, scenario: int) -> dict[str, Any]:
    trace = root / TRACE_PATTERN.format(scenario=scenario)
    if not (trace / "COMPLETED").is_file():
        return {"available": False, "updates": None, "noops": None}
    databases = list((trace / "cache").rglob("memory.db"))
    if len(databases) != 1:
        raise ValueError(f"S{scenario}: expected one memory.db, found {len(databases)}")
    connection = sqlite3.connect(databases[0])
    try:
        statuses = Counter()
        for (metadata_json,) in connection.execute(
            "SELECT metadata_json FROM model_calls WHERE consolidation_run_id IS NOT NULL "
            "AND error IS NULL"
        ):
            status = json.loads(metadata_json).get("update_status")
            if status:
                statuses[status] += 1
    finally:
        connection.close()
    return {
        "available": True,
        "updates": statuses["updated"],
        "noops": statuses["noop"],
    }


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown_summary(manifest: dict[str, Any]) -> str:
    pilot = manifest["pilot_blueprint"]
    return (
        "# V1-style augmentation source selection\n\n"
        f"- Selected: `{manifest['selected_source_scenarios']}`\n"
        f"- Clean held-out: `{manifest['clean_heldout_scenarios']}`\n"
        f"- Existing V1-10 train: `{manifest['existing_training_scenarios']}`\n"
        f"- Pilot source: **S{manifest['pilot_source_scenario']}**\n"
        f"- Pilot ESM/Tool/Arg: `{pilot['esm']:.3f}` / `{pilot['tool_f1']:.3f}` / `{pilot['arg_exact']:.3f}`\n"
        f"- Pilot trace UPDATE/NO_OP: `{pilot['cloud_trace_updates']}` / `{pilot['cloud_trace_noops']}`\n"
    )


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps({
        "selected": result["selected_source_scenarios"],
        "heldout": result["clean_heldout_scenarios"],
        "pilot": result["pilot_source_scenario"],
    }, ensure_ascii=False, indent=2))
