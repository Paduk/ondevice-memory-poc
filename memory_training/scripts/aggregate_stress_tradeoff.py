#!/usr/bin/env python3
"""Aggregate natural and update-stress accuracy/cost results for four models."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_training.evaluate_hf_closed_loop import aggregate_results


WORKSPACE = Path("/mnt/data/hj153lee/PalmClaw/on-device-memory-training")
BENCHMARKS = WORKSPACE / "benchmarks"
RUNS = WORKSPACE / "runs"
OUTPUT_DIR = Path("/home/hj153lee/PalmClaw/docs/engineering/results")
SCENARIOS = tuple(range(86, 91))
FULL_TEST_SCENARIOS = tuple(range(86, 101)) + tuple(range(112, 121))
WORKLOADS = ("base", "20", "40", "60", "80")
METHODS = {
    "Patch": "patch",
    "Summary": "summary",
    "k=2": "delta_v3_compact_k2",
    "k=5": "delta_v3_compact_k5",
    "k=10": "delta_v3_compact_k10",
}

MODELS = {
    "Granite 350M": {
        "prefill_tps": 137.7,
        "decode_tps": 51.8,
        "stress": "granite4-350m-stress-test-once-20260905-v1",
        "natural": "granite4-350m-natural-s86-s90-cache-once-20260907-v1",
        "tests": {
            "Patch": ("granite4-350m-patch-multitask-noop5-full6val-grouped-v2-v1-10-e4-b8-trainseed45-r1", "eval-fixed-v2-test-best-epoch-04"),
            "Summary": ("granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1", "eval-fixed-v2-test-best-epoch-04"),
            "k=2": ("granite4-350m-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1", "eval-fixed-test-best-epoch-03"),
            "k=5": ("granite4-350m-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1", "eval-fixed-test-best-epoch-03"),
            "k=10": ("granite4-350m-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1", "eval-fixed-test-best-epoch-04"),
        },
    },
    "Qwen 0.8B": {
        "prefill_tps": 68.0,
        "decode_tps": 27.6,
        "stress": "qwen35-0.8b-stress-test-once-20260904-v1",
        "natural": "qwen35-0.8b-natural-s86-s90-cache-once-20260905-v1",
        "full_test_tests": {
            "Patch": ("qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed46-evalfixed-noop5-r1", "eval-fixed-composite-test-epoch-03"),
        },
        "tests": {
            "Patch": ("qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed45-evalfixed-noop5-r1", "eval-fixed-test-best-epoch-03"),
            "Summary": ("qwen35-0.8b-summary-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2", "eval-fixed-test-best-epoch-03"),
            "k=2": ("qwen35-0.8b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1", "eval-fixed-test-best-epoch-03"),
            "k=5": ("qwen35-0.8b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1", "eval-fixed-test-best-epoch-03"),
            "k=10": ("qwen35-0.8b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1", "eval-fixed-test-best-epoch-03"),
        },
    },
    "Granite 1B": {
        "prefill_tps": 25.7,
        "decode_tps": 16.4,
        "stress": "granite4-1b-stress-test-once-20260904-v1",
        "natural": "granite4-1b-natural-s86-s90-cache-once-20260905-v1",
        "tests": {
            "Patch": ("granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1", "eval-fixed-v2-test-best-epoch-03"),
            "Summary": ("granite4-1b-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1", "eval-fixed-v2-test-best-epoch-03"),
            "k=2": ("granite4-1b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1", "eval-fixed-test-best-epoch-03"),
            "k=5": ("granite4-1b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1", "eval-fixed-test-best-epoch-04"),
            "k=10": ("granite4-1b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1", "eval-fixed-test-best-epoch-04"),
        },
    },
    "Qwen 2B": {
        "prefill_tps": 24.7,
        "decode_tps": 14.5,
        "stress": "qwen35-2b-stress-test-once-20260905-v1",
        "natural": "qwen35-2b-natural-s86-s90-cache-once-20260907-v1",
        "full_test_tests": {
            "Patch": ("qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-trainseed46-r2", "eval-fixed-v2-test-best-epoch-03"),
        },
        "tests": {
            "Patch": ("qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-r1", "eval-fixed-v2-test-best-epoch-03"),
            "Summary": ("qwen35-2b-summary-multitask-noop5-grouped-v2-v1-10-e4-b2-r1", "eval-fixed-v2-test-best-epoch-04"),
            "k=2": ("qwen35-2b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1", "eval-fixed-test-best-epoch-03"),
            "k=5": ("qwen35-2b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1", "eval-fixed-test-best-epoch-04"),
            "k=10": ("qwen35-2b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b2-trainseed45-evalfixed-noop5-r1", "eval-fixed-test-best-epoch-04"),
        },
    },
}


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def composite(summary: dict) -> float:
    return 100 * (
        0.60 * float(summary["closed_loop_quiz"]["esm"])
        + 0.25 * float(summary["memory"]["final_state_f1"])
        + 0.15 * float(summary["memory"]["update_f1"])
    )


def scenario_composite(
    config: dict, method: str, scenarios: tuple[int, ...], *, full_test: bool = False
) -> dict:
    tests = config.get("full_test_tests", {}) if full_test else {}
    run, test = tests.get(method, config["tests"][method])
    scenario_dir = RUNS / run / test / "scenarios"
    results = [
        json.loads((scenario_dir / f"s{scenario:03d}.json").read_text())
        for scenario in scenarios
    ]
    return aggregate_results(results, scenarios)


def base_composite(config: dict, method: str) -> dict:
    return scenario_composite(config, method, SCENARIOS)


def cache_paths(config: dict, method_slug: str, workload: str) -> tuple[Path, Path]:
    if workload == "base":
        root = BENCHMARKS / config["natural"] / "cache" / method_slug / "on"
    else:
        root = BENCHMARKS / config["stress"] / "cache" / f"update{workload}" / method_slug / "on"
    return root / "summary.json", root / "turns.jsonl"


def stress_composite_path(config: dict, method_slug: str, workload: str) -> Path:
    return BENCHMARKS / config["stress"] / "composite" / f"update{workload}" / method_slug / "summary.json"


def slice_metrics(records: list[dict], prefill_tps: float, decode_tps: float) -> dict:
    logical = [float(row["logical_prompt_tokens"]) for row in records]
    prefill = [float(row["evaluated_prefill_tokens"]) for row in records]
    decode = [float(row["decode_tokens"]) for row in records]
    cache_on_prefill = [p / prefill_tps for p in prefill]
    cache_off_prefill = [p / prefill_tps for p in logical]
    decode_latency = [d / decode_tps for d in decode]
    projected = [p + d for p, d in zip(cache_on_prefill, decode_latency)]
    projected_off = [p + d for p, d in zip(cache_off_prefill, decode_latency)]
    return {
        "turns": len(records),
        "logical_prefill_tokens_total": sum(logical),
        "logical_prefill_tokens_mean": mean(logical) if logical else 0.0,
        "evaluated_prefill_tokens_total": sum(prefill),
        "evaluated_prefill_tokens_mean": mean(prefill) if prefill else 0.0,
        "decode_tokens_total": sum(decode),
        "decode_tokens_mean": mean(decode) if decode else 0.0,
        "cache_on_prefill_seconds_total": sum(cache_on_prefill),
        "cache_on_prefill_seconds_mean": mean(cache_on_prefill) if cache_on_prefill else 0.0,
        "cache_on_prefill_seconds_p95": percentile(cache_on_prefill, 0.95),
        "cache_off_prefill_seconds_total": sum(cache_off_prefill),
        "cache_off_prefill_seconds_mean": mean(cache_off_prefill) if cache_off_prefill else 0.0,
        "cache_off_prefill_seconds_p95": percentile(cache_off_prefill, 0.95),
        "decode_seconds_total": sum(decode_latency),
        "decode_seconds_mean": mean(decode_latency) if decode_latency else 0.0,
        "decode_seconds_p95": percentile(decode_latency, 0.95),
        "projected_latency_seconds_total": sum(projected),
        "projected_latency_seconds_mean": mean(projected) if projected else 0.0,
        "projected_latency_seconds_p95": percentile(projected, 0.95),
        "cache_off_projected_latency_seconds_total": sum(projected_off),
        "cache_off_projected_latency_seconds_mean": mean(projected_off) if projected_off else 0.0,
        "cache_off_projected_latency_seconds_p95": percentile(projected_off, 0.95),
    }


def update_cycle_metrics(
    pairs: list[tuple[dict, dict]], prefill_tps: float, decode_tps: float
) -> dict:
    update_decode = [float(update["decode_tokens"]) for update, _ in pairs]
    post_logical = [float(post["logical_prompt_tokens"]) for _, post in pairs]
    post_evaluated = [float(post["evaluated_prefill_tokens"]) for _, post in pairs]
    update_decode_seconds = [value / decode_tps for value in update_decode]
    post_off_seconds = [value / prefill_tps for value in post_logical]
    post_on_seconds = [value / prefill_tps for value in post_evaluated]
    cycle_off = [d + p for d, p in zip(update_decode_seconds, post_off_seconds)]
    cycle_on = [d + p for d, p in zip(update_decode_seconds, post_on_seconds)]
    return {
        "cycles": len(pairs),
        "update_decode_tokens_total": sum(update_decode),
        "update_decode_tokens_mean": mean(update_decode) if update_decode else 0.0,
        "update_decode_seconds_total": sum(update_decode_seconds),
        "update_decode_seconds_mean": mean(update_decode_seconds) if update_decode_seconds else 0.0,
        "update_decode_seconds_p95": percentile(update_decode_seconds, 0.95),
        "post_update_logical_prefill_tokens_total": sum(post_logical),
        "post_update_logical_prefill_tokens_mean": mean(post_logical) if post_logical else 0.0,
        "post_update_evaluated_prefill_tokens_total": sum(post_evaluated),
        "post_update_evaluated_prefill_tokens_mean": mean(post_evaluated) if post_evaluated else 0.0,
        "post_update_cache_off_prefill_seconds_total": sum(post_off_seconds),
        "post_update_cache_off_prefill_seconds_mean": mean(post_off_seconds) if post_off_seconds else 0.0,
        "post_update_cache_off_prefill_seconds_p95": percentile(post_off_seconds, 0.95),
        "post_update_cache_on_prefill_seconds_total": sum(post_on_seconds),
        "post_update_cache_on_prefill_seconds_mean": mean(post_on_seconds) if post_on_seconds else 0.0,
        "post_update_cache_on_prefill_seconds_p95": percentile(post_on_seconds, 0.95),
        "cache_off_seconds_total": sum(cycle_off),
        "cache_off_seconds_mean": mean(cycle_off) if cycle_off else 0.0,
        "cache_off_seconds_p95": percentile(cycle_off, 0.95),
        "cache_on_seconds_total": sum(cycle_on),
        "cache_on_seconds_mean": mean(cycle_on) if cycle_on else 0.0,
        "cache_on_seconds_p95": percentile(cycle_on, 0.95),
    }


def aggregate_cache(turns_path: Path, prefill_tps: float, decode_tps: float) -> dict:
    records = [json.loads(line) for line in turns_path.read_text().splitlines() if line]
    result = {"all": slice_metrics(records, prefill_tps, decode_tps)}
    for decision, label in (("UPDATE", "update"), ("NO_OP", "noop")):
        result[label] = slice_metrics(
            [row for row in records if row["gold_decision"] == decision],
            prefill_tps,
            decode_tps,
        )
    update_cycles = [
        (previous, current)
        for previous, current in zip(records, records[1:])
        if previous["scenario_index"] == current["scenario_index"]
        and previous["repetition"] == current["repetition"]
        and previous["gold_decision"] == "UPDATE"
    ]
    post_update = [current for _, current in update_cycles]
    result["post_update"] = slice_metrics(post_update, prefill_tps, decode_tps)
    result["update_cycle"] = update_cycle_metrics(update_cycles, prefill_tps, decode_tps)
    return result


def main() -> None:
    rows = []
    nested = {"models": {}, "workloads": list(WORKLOADS), "methods": list(METHODS)}
    for model, config in MODELS.items():
        model_result = {
            "prefill_tokens_per_second": config["prefill_tps"],
            "decode_tokens_per_second": config["decode_tps"],
            "methods": {},
        }
        nested["models"][model] = model_result
        for method, method_slug in METHODS.items():
            method_result = {}
            model_result["methods"][method] = method_result
            base_accuracy = base_composite(config, method)
            full_test_accuracy = scenario_composite(
                config, method, FULL_TEST_SCENARIOS, full_test=True
            )
            method_result["full_test"] = {
                "scenarios": len(FULL_TEST_SCENARIOS),
                "composite": composite(full_test_accuracy),
                "quiz_esm": 100 * float(full_test_accuracy["closed_loop_quiz"]["esm"]),
                "final_state_f1": 100 * float(full_test_accuracy["memory"]["final_state_f1"]),
                "update_f1": 100 * float(full_test_accuracy["memory"]["update_f1"]),
            }
            for workload in WORKLOADS:
                cache_summary_path, turns_path = cache_paths(config, method_slug, workload)
                if not cache_summary_path.exists() or not turns_path.exists():
                    raise FileNotFoundError(f"Missing cache result: {cache_summary_path.parent}")
                cache_summary = json.loads(cache_summary_path.read_text())
                accuracy = (
                    base_accuracy
                    if workload == "base"
                    else json.loads(stress_composite_path(config, method_slug, workload).read_text())
                )
                cache = aggregate_cache(turns_path, config["prefill_tps"], config["decode_tps"])
                value = {
                    "composite": composite(accuracy),
                    "quiz_esm": 100 * float(accuracy["closed_loop_quiz"]["esm"]),
                    "final_state_f1": 100 * float(accuracy["memory"]["final_state_f1"]),
                    "update_f1": 100 * float(accuracy["memory"]["update_f1"]),
                    "cache_reuse_ratio": float(cache_summary["metrics"]["all_turns"]["cache_reuse_ratio"]),
                    # The benchmark records model-output parse/application failures here;
                    # these are not KV-cache correctness failures.
                    "generation_errors": int(cache_summary["metrics"]["all_turns"]["errors"]),
                    **cache,
                }
                method_result[workload] = value
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "workload": workload,
                        "updates_per_scenario": 13.4 if workload == "base" else int(workload),
                        "prefill_tps": config["prefill_tps"],
                        "decode_tps": config["decode_tps"],
                        "composite": value["composite"],
                        "quiz_esm": value["quiz_esm"],
                        "final_state_f1": value["final_state_f1"],
                        "update_f1": value["update_f1"],
                        "cache_reuse_ratio": value["cache_reuse_ratio"],
                        "generation_errors": value["generation_errors"],
                        "full_test_composite": method_result["full_test"]["composite"],
                        **{
                            f"{scope}_{key}": metric
                            for scope in ("all", "update", "noop", "post_update", "update_cycle")
                            for key, metric in value[scope].items()
                        },
                    }
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "stress-tradeoff-four-models.json"
    csv_path = OUTPUT_DIR / "stress-tradeoff-four-models.csv"
    json_path.write_text(json.dumps(nested, ensure_ascii=False, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
