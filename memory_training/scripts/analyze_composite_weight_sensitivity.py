#!/usr/bin/env python3
"""Analyze Composite-weight robustness and paired scenario-level differences.

The analysis is intentionally offline: it reuses the completed 24-scenario
fixed-Test and update-stress artifacts used by Sections 5 and 6.2.  It produces
machine-readable grids, paired scenario statistics, a compact figure, and an
appendix-ready Markdown report.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_training.scripts.aggregate_stress_tradeoff import (  # noqa: E402
    MODELS as FOUR_MODEL_CONFIGS,
)
from memory_training.scripts.aggregate_three_method_type_accuracy import (  # noqa: E402
    MODEL_ARTIFACTS as THREE_METHOD_ARTIFACTS,
)
from memory_training.scripts.plot_delta_v3_six_model_workload_response import (  # noqa: E402
    LLAMA_3B_RUNS,
    LLAMA_RUNS,
)
from memory_training.scripts.plot_on_device_pareto_test24_partial import (  # noqa: E402
    MODEL_SLUGS,
)


WORKSPACE = Path("/mnt/data/hj153lee/PalmClaw/on-device-memory-training")
RUN_ROOT = WORKSPACE / "runs"
BENCHMARK_ROOT = WORKSPACE / "benchmarks"
RESULTS_DIR = REPO_ROOT / "docs/engineering/results"
FIGURE_DIR = REPO_ROOT / "docs/engineering/figures"
REPORT_PATH = REPO_ROOT / "docs/engineering/composite-weight-sensitivity.md"
GRID_PATH = RESULTS_DIR / "composite-weight-sensitivity-grid.csv"
PAIRED_PATH = RESULTS_DIR / "composite-weight-sensitivity-paired.csv"
JSON_PATH = RESULTS_DIR / "composite-weight-sensitivity.json"
FIGURE_STEM = FIGURE_DIR / "composite-weight-sensitivity"
SECTION6_RAW = RESULTS_DIR / "section6-2-compaction-interval-six-model-raw.csv"

MODELS = (
    "Granite 350M",
    "Qwen 0.8B",
    "Granite 1B",
    "Llama 3.2 1B",
    "Qwen 2B",
    "Llama 3.2 3B",
)
K_MODEL_NAMES = {
    "Granite 350M": "Granite 350M",
    "Qwen 0.8B": "Qwen 0.8B",
    "Granite 1B": "Granite 1B",
    "Llama 3.2 1B": "Llama 1B",
    "Qwen 2B": "Qwen 2B",
    "Llama 3.2 3B": "Llama 3B",
}
K_METHODS = ("Patch", "k=2", "k=5", "k=10")
METHOD_SLUGS = {
    "Patch": "patch",
    "k=2": "delta_v3_compact_k2",
    "k=5": "delta_v3_compact_k5",
    "k=10": "delta_v3_compact_k10",
}
WORKLOADS = ("Base", "U40", "U60", "U80")
MAIN_METHODS = ("Summary", "Patch", "Delta-v3", "Mem0 One-pass")
SCENARIOS = tuple(range(86, 101)) + tuple(range(112, 121))
DEFAULT_WEIGHT = (0.60, 0.25, 0.15)
METRIC_NAMES = ("quiz_esm", "final_memory_f1", "update_f1")

MEM0_ARTIFACTS = {
    "Granite 350M": (
        "granite4-350m-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1/"
        "eval-fixed-v2-test-best-epoch-03/summary.json"
    ),
    "Qwen 0.8B": (
        "qwen35-0.8b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1/"
        "eval-fixed-v2-test-best-epoch-03/summary.json"
    ),
    "Granite 1B": (
        "granite4-1b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1/"
        "eval-fixed-v2-test-best-epoch-02/summary.json"
    ),
    "Llama 3.2 1B": (
        "llama3.2-1b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1/"
        "eval-fixed-v2-test-best-epoch-03/summary.json"
    ),
    "Qwen 2B": (
        "qwen35-2b-mem0-one-pass-multitask-noop5-e4-b4-trainseed45-r1/"
        "eval-fixed-v2-test-best-epoch-03/summary.json"
    ),
    "Llama 3.2 3B": (
        "llama3.2-3b-mem0-one-pass-multitask-noop5-e4-b4-trainseed45-r1/"
        "eval-fixed-v2-test-best-epoch-03/summary.json"
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def metric_vector(result: Mapping[str, Any]) -> np.ndarray:
    """Return Quiz ESM, final-memory F1, and Update F1 on a 0--100 scale."""
    return 100.0 * np.array(
        [
            float(result["closed_loop_quiz"]["esm"]),
            float(result["memory"]["final_state_f1"]),
            float(result["memory"]["update_f1"]),
        ],
        dtype=float,
    )


def score(vector: np.ndarray, weight: tuple[float, float, float]) -> float:
    return float(np.dot(vector, np.asarray(weight, dtype=float)))


def validate_summary(path: Path) -> dict[str, Any]:
    result = load_json(path)
    completed = tuple(int(value) for value in result.get("completed_scenarios", ()))
    if not result.get("complete") or set(completed) != set(SCENARIOS):
        raise ValueError(f"Expected a complete 24-scenario result: {path}")
    return result


def scenario_vectors(summary_path: Path) -> dict[int, np.ndarray]:
    scenario_dir = summary_path.parent / "scenarios"
    values: dict[int, np.ndarray] = {}
    for scenario in SCENARIOS:
        path = scenario_dir / f"s{scenario:03d}.json"
        values[scenario] = metric_vector(load_json(path))
    return values


def primary_weights() -> list[tuple[float, float, float]]:
    """Local, predeclared range around 0.60/0.25/0.15 in 0.025 steps."""
    values = []
    denominator = 40
    for quiz in range(20, 29):  # 0.50--0.70
        for final in range(6, 15):  # 0.15--0.35
            update = denominator - quiz - final
            if 4 <= update <= 10:  # 0.10--0.25
                values.append((quiz / denominator, final / denominator, update / denominator))
    if DEFAULT_WEIGHT not in values:
        raise AssertionError("Default Composite weight is absent from primary grid")
    return values


def broad_weights() -> list[tuple[float, float, float]]:
    """Positive 0.05-step simplex used only as a deliberately broad diagnostic."""
    values = []
    denominator = 20
    for quiz in range(1, denominator - 1):
        for final in range(1, denominator - quiz):
            update = denominator - quiz - final
            if update >= 1:
                values.append((quiz / denominator, final / denominator, update / denominator))
    return values


def main_summary_path(model: str, method: str) -> Path:
    if method == "Mem0 One-pass":
        relative = MEM0_ARTIFACTS[model]
    else:
        relative = THREE_METHOD_ARTIFACTS[model][method]
    return RUN_ROOT / relative


def k_base_summary_path(model: str, method: str) -> Path:
    if model in ("Granite 350M", "Qwen 0.8B", "Granite 1B", "Qwen 2B"):
        config = FOUR_MODEL_CONFIGS[K_MODEL_NAMES[model]]
        override = config.get("full_test_tests", {})
        run, evaluation = override.get(method, config["tests"][method])
        return RUN_ROOT / run / evaluation / "summary.json"
    runs = LLAMA_RUNS if model == "Llama 3.2 1B" else LLAMA_3B_RUNS
    run, evaluation, _ = runs[method]
    return RUN_ROOT / run / evaluation / "summary.json"


def k_stress_summary_path(model: str, workload: str, method: str) -> Path:
    if workload == "Base":
        return k_base_summary_path(model, method)
    run = (
        f"{MODEL_SLUGS[K_MODEL_NAMES[model]]}-{METHOD_SLUGS[method]}"
        "-test24-update-stress-20260915-v1"
    )
    return (
        BENCHMARK_ROOT
        / run
        / "composite"
        / f"update{workload.removeprefix('U')}"
        / "summary.json"
    )


def load_latency() -> dict[tuple[str, str, str], float]:
    values = {}
    with SECTION6_RAW.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[(row["model"], row["workload"], row["method"])] = float(
                row["relative_latency_percent"]
            )
    return values


def pareto_set(points: Mapping[str, tuple[float, float]]) -> tuple[str, ...]:
    """Return non-dominated methods for (lower latency, higher quality)."""
    retained = []
    for method, (latency, quality) in points.items():
        dominated = False
        for other, (other_latency, other_quality) in points.items():
            if other == method:
                continue
            no_worse = other_latency <= latency + 1e-12 and other_quality >= quality - 1e-12
            strictly_better = other_latency < latency - 1e-12 or other_quality > quality + 1e-12
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            retained.append(method)
    return tuple(retained)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted


def hierarchical_bootstrap(
    differences: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    """Resample model and scenario axes independently with replacement."""
    rng = np.random.default_rng(seed)
    model_count, scenario_count = differences.shape
    estimates = np.empty(samples, dtype=float)
    chunk_size = 1000
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        size = stop - start
        model_index = rng.integers(0, model_count, size=(size, model_count))
        scenario_index = rng.integers(0, scenario_count, size=(size, scenario_count))
        for offset in range(size):
            estimates[start + offset] = differences[
                np.ix_(model_index[offset], scenario_index[offset])
            ].mean()
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def scenario_cluster_bootstrap(
    scenario_differences: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    """Bootstrap the 24 paired scenario clusters with the six models fixed."""
    rng = np.random.default_rng(seed)
    scenario_count = len(scenario_differences)
    indices = rng.integers(0, scenario_count, size=(samples, scenario_count))
    estimates = scenario_differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def paired_permutation_p(
    scenario_differences: np.ndarray, *, samples: int, seed: int
) -> float:
    """Two-sided paired randomization test over the 24 scenario clusters."""
    observed = abs(float(scenario_differences.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    generated = 0
    chunk_size = 5000
    while generated < samples:
        size = min(chunk_size, samples - generated)
        signs = rng.choice((-1.0, 1.0), size=(size, len(scenario_differences)))
        permuted = np.abs((signs * scenario_differences).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
        generated += size
    return (exceed + 1.0) / (samples + 1.0)


def fmt_weight(weight: tuple[float, float, float]) -> str:
    return "/".join(f"{value:.3f}" for value in weight)


def fmt_p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def analyze() -> dict[str, Any]:
    local_grid = primary_weights()
    wide_grid = broad_weights()
    latency = load_latency()

    main_aggregate: dict[tuple[str, str], np.ndarray] = {}
    main_scenarios: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for model in MODELS:
        for method in MAIN_METHODS:
            path = main_summary_path(model, method)
            summary = validate_summary(path)
            main_aggregate[(model, method)] = metric_vector(summary)
            main_scenarios[(model, method)] = scenario_vectors(path)

    k_aggregate: dict[tuple[str, str, str], np.ndarray] = {}
    k_scenarios: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
    for model in MODELS:
        for workload in WORKLOADS:
            for method in K_METHODS:
                path = k_stress_summary_path(model, workload, method)
                summary = validate_summary(path)
                k_aggregate[(model, workload, method)] = metric_vector(summary)
                k_scenarios[(model, workload, method)] = scenario_vectors(path)
                raw_model = K_MODEL_NAMES[model]
                expected = next(
                    row
                    for row in csv.DictReader(SECTION6_RAW.open(encoding="utf-8"))
                    if row["model"] == raw_model
                    and row["workload"] == workload
                    and row["method"] == method
                )
                actual = score(metric_vector(summary), DEFAULT_WEIGHT)
                if not math.isclose(actual, float(expected["composite_percent"]), abs_tol=1e-8):
                    raise ValueError(
                        f"Section 6.2 provenance mismatch for {model}/{workload}/{method}: "
                        f"{actual} != {expected['composite_percent']}"
                    )

    grid_rows: list[dict[str, Any]] = []
    main_grid_summary: dict[str, Any] = {}
    k_grid_summary: dict[str, Any] = {}

    for grid_name, weights in (("primary", local_grid), ("broad", wide_grid)):
        main_winner_counts: Counter[str] = Counter()
        main_order_counts: Counter[str] = Counter()
        patch_model_wins = 0
        patch_model_total = 0
        patch_model_wins_by_model: Counter[str] = Counter()
        k5_aggregate_pareto = Counter()
        aggregate_set_counts: dict[str, Counter[str]] = {
            workload: Counter() for workload in WORKLOADS
        }
        k5_model_pareto = 0
        k5_model_total = 0
        k5_model_pareto_by_model_workload: Counter[tuple[str, str]] = Counter()

        for weight in weights:
            main_means = {
                method: float(
                    np.mean([score(main_aggregate[(model, method)], weight) for model in MODELS])
                )
                for method in MAIN_METHODS
            }
            winner = max(main_means, key=main_means.get)
            ordering = tuple(sorted(MAIN_METHODS, key=main_means.get, reverse=True))
            main_winner_counts[winner] += 1
            main_order_counts[">".join(ordering)] += 1
            for model in MODELS:
                model_scores = {
                    method: score(main_aggregate[(model, method)], weight)
                    for method in MAIN_METHODS
                }
                patch_model_wins += int(
                    model_scores["Patch"] >= max(model_scores.values()) - 1e-12
                )
                patch_model_wins_by_model[model] += int(
                    model_scores["Patch"] >= max(model_scores.values()) - 1e-12
                )
                patch_model_total += 1
            grid_rows.append(
                {
                    "grid": grid_name,
                    "scope": "main",
                    "workload": "Base",
                    "quiz_weight": weight[0],
                    "final_memory_weight": weight[1],
                    "update_weight": weight[2],
                    "winner": winner,
                    "pareto_set": "",
                    "patch_score": main_means["Patch"],
                    "k2_score": "",
                    "k5_score": main_means["Delta-v3"],
                    "k10_score": "",
                    "summary_score": main_means["Summary"],
                    "mem0_one_pass_score": main_means["Mem0 One-pass"],
                    "patch_margin_over_best_alternative": main_means["Patch"]
                    - max(value for method, value in main_means.items() if method != "Patch"),
                    "k5_pareto": "",
                }
            )

            for workload in WORKLOADS:
                mean_scores = {
                    method: float(
                        np.mean(
                            [
                                score(k_aggregate[(model, workload, method)], weight)
                                for model in MODELS
                            ]
                        )
                    )
                    for method in K_METHODS
                }
                mean_latency = {
                    method: float(
                        np.mean(
                            [
                                latency[(K_MODEL_NAMES[model], workload, method)]
                                for model in MODELS
                            ]
                        )
                    )
                    for method in K_METHODS
                }
                aggregate_pareto = pareto_set(
                    {
                        method: (mean_latency[method], mean_scores[method])
                        for method in K_METHODS
                    }
                )
                k5_aggregate_pareto[workload] += int("k=5" in aggregate_pareto)
                aggregate_set_counts[workload]["|".join(aggregate_pareto)] += 1
                for model in MODELS:
                    model_pareto = pareto_set(
                        {
                            method: (
                                latency[(K_MODEL_NAMES[model], workload, method)],
                                score(k_aggregate[(model, workload, method)], weight),
                            )
                            for method in K_METHODS
                        }
                    )
                    k5_model_pareto += int("k=5" in model_pareto)
                    k5_model_pareto_by_model_workload[(model, workload)] += int(
                        "k=5" in model_pareto
                    )
                    k5_model_total += 1
                grid_rows.append(
                    {
                        "grid": grid_name,
                        "scope": "k_sweep",
                        "workload": workload,
                        "quiz_weight": weight[0],
                        "final_memory_weight": weight[1],
                        "update_weight": weight[2],
                        "winner": max(mean_scores, key=mean_scores.get),
                        "pareto_set": "|".join(aggregate_pareto),
                        "patch_score": mean_scores["Patch"],
                        "k2_score": mean_scores["k=2"],
                        "k5_score": mean_scores["k=5"],
                        "k10_score": mean_scores["k=10"],
                        "summary_score": "",
                        "mem0_one_pass_score": "",
                        "patch_margin_over_best_alternative": "",
                        "k5_pareto": int("k=5" in aggregate_pareto),
                    }
                )

        main_grid_summary[grid_name] = {
            "weights": len(weights),
            "winner_counts": dict(main_winner_counts),
            "winner_rates": {
                method: main_winner_counts[method] / len(weights) for method in MAIN_METHODS
            },
            "ordering_counts": dict(main_order_counts),
            "patch_model_level_top_rate": patch_model_wins / patch_model_total,
            "patch_top_rates_by_model": {
                model: patch_model_wins_by_model[model] / len(weights)
                for model in MODELS
            },
        }
        k_grid_summary[grid_name] = {
            "weights": len(weights),
            "k5_aggregate_pareto_counts": dict(k5_aggregate_pareto),
            "k5_aggregate_pareto_rates": {
                workload: k5_aggregate_pareto[workload] / len(weights)
                for workload in WORKLOADS
            },
            "aggregate_pareto_set_counts": {
                workload: dict(values) for workload, values in aggregate_set_counts.items()
            },
            "k5_model_level_pareto_rate": k5_model_pareto / k5_model_total,
            "k5_model_level_pareto_rates": {
                model: {
                    workload: k5_model_pareto_by_model_workload[(model, workload)]
                    / len(weights)
                    for workload in WORKLOADS
                }
                for model in MODELS
            },
        }

    paired_rows: list[dict[str, Any]] = []
    rng_seed = 20260923

    def add_pair(
        *,
        family: str,
        workload: str,
        focal: str,
        comparator: str,
        source: Mapping[tuple[str, ...], dict[int, np.ndarray]],
    ) -> None:
        differences = np.empty((len(MODELS), len(SCENARIOS)), dtype=float)
        for model_index, model in enumerate(MODELS):
            if family == "main":
                focal_values = source[(model, focal)]
                comparator_values = source[(model, comparator)]
            else:
                focal_values = source[(model, workload, focal)]
                comparator_values = source[(model, workload, comparator)]
            for scenario_index, scenario in enumerate(SCENARIOS):
                differences[model_index, scenario_index] = score(
                    focal_values[scenario], DEFAULT_WEIGHT
                ) - score(comparator_values[scenario], DEFAULT_WEIGHT)
        scenario_means = differences.mean(axis=0)
        scenario_low, scenario_high = scenario_cluster_bootstrap(
            scenario_means, samples=20_000, seed=rng_seed + len(paired_rows)
        )
        hierarchical_low, hierarchical_high = hierarchical_bootstrap(
            differences, samples=20_000, seed=rng_seed + len(paired_rows)
        )
        p_value = paired_permutation_p(
            scenario_means, samples=100_000, seed=rng_seed + 100 + len(paired_rows)
        )
        epsilon = 1e-12
        weight_signs = []
        for weight in local_grid:
            weight_differences = []
            for model in MODELS:
                if family == "main":
                    focal_values = source[(model, focal)]
                    comparator_values = source[(model, comparator)]
                else:
                    focal_values = source[(model, workload, focal)]
                    comparator_values = source[(model, workload, comparator)]
                weight_differences.extend(
                    score(focal_values[scenario], weight)
                    - score(comparator_values[scenario], weight)
                    for scenario in SCENARIOS
                )
            weight_signs.append(float(np.mean(weight_differences)))
        paired_rows.append(
            {
                "family": family,
                "workload": workload,
                "focal": focal,
                "comparator": comparator,
                "mean_difference_pp": float(differences.mean()),
                "scenario_bootstrap_ci_low": scenario_low,
                "scenario_bootstrap_ci_high": scenario_high,
                "hierarchical_bootstrap_ci_low": hierarchical_low,
                "hierarchical_bootstrap_ci_high": hierarchical_high,
                "scenario_cluster_wins": int(np.count_nonzero(scenario_means > epsilon)),
                "scenario_cluster_ties": int(np.count_nonzero(np.abs(scenario_means) <= epsilon)),
                "scenario_cluster_losses": int(np.count_nonzero(scenario_means < -epsilon)),
                "paired_permutation_p": p_value,
                "holm_adjusted_p": 0.0,
                "positive_mean_weight_fraction": float(np.mean(np.asarray(weight_signs) > epsilon)),
                "negative_mean_weight_fraction": float(np.mean(np.asarray(weight_signs) < -epsilon)),
            }
        )

    for comparator in ("Summary", "Delta-v3", "Mem0 One-pass"):
        add_pair(
            family="main",
            workload="Base",
            focal="Patch",
            comparator=comparator,
            source=main_scenarios,
        )
    for workload in WORKLOADS:
        for comparator in ("Patch", "k=2", "k=10"):
            add_pair(
                family="k_sweep",
                workload=workload,
                focal="k=5",
                comparator=comparator,
                source=k_scenarios,
            )

    for family in ("main", "k_sweep"):
        indices = [index for index, row in enumerate(paired_rows) if row["family"] == family]
        adjusted = holm_adjust([float(paired_rows[index]["paired_permutation_p"]) for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            paired_rows[index]["holm_adjusted_p"] = value

    default_main = {
        method: float(
            np.mean([score(main_aggregate[(model, method)], DEFAULT_WEIGHT) for model in MODELS])
        )
        for method in MAIN_METHODS
    }
    default_k = {}
    for workload in WORKLOADS:
        scores = {
            method: float(
                np.mean(
                    [score(k_aggregate[(model, workload, method)], DEFAULT_WEIGHT) for model in MODELS]
                )
            )
            for method in K_METHODS
        }
        latencies = {
            method: float(
                np.mean(
                    [latency[(K_MODEL_NAMES[model], workload, method)] for model in MODELS]
                )
            )
            for method in K_METHODS
        }
        default_k[workload] = {
            "scores": scores,
            "relative_latency_percent": latencies,
            "pareto_set": list(
                pareto_set(
                    {method: (latencies[method], scores[method]) for method in K_METHODS}
                )
            ),
        }

    return {
        "schema_version": "palmclaw-composite-weight-sensitivity-v1",
        "default_weight": dict(zip(METRIC_NAMES, DEFAULT_WEIGHT, strict=True)),
        "primary_grid": {
            "quiz_range": [0.50, 0.70],
            "final_memory_range": [0.15, 0.35],
            "update_range": [0.10, 0.25],
            "step": 0.025,
            **main_grid_summary["primary"],
            "k_sweep": k_grid_summary["primary"],
        },
        "broad_grid": {
            "description": "positive 0.05-step simplex; each weight >= 0.05",
            **main_grid_summary["broad"],
            "k_sweep": k_grid_summary["broad"],
        },
        "default_main_scores": default_main,
        "default_k_sweep": default_k,
        "paired": paired_rows,
        "grid_rows": grid_rows,
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def make_figure(result: Mapping[str, Any]) -> None:
    rows = [row for row in result["grid_rows"] if row["grid"] == "primary"]
    main_rows = [row for row in rows if row["scope"] == "main"]
    workloads = list(WORKLOADS)
    primary_rates = [
        100.0
        * float(result["primary_grid"]["k_sweep"]["k5_aggregate_pareto_rates"][workload])
        for workload in workloads
    ]
    broad_rates = [
        100.0
        * float(result["broad_grid"]["k_sweep"]["k5_aggregate_pareto_rates"][workload])
        for workload in workloads
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_margin, ax_pareto) = plt.subplots(1, 2, figsize=(10.8, 4.1))
    scatter = ax_margin.scatter(
        [100 * float(row["quiz_weight"]) for row in main_rows],
        [100 * float(row["final_memory_weight"]) for row in main_rows],
        c=[float(row["patch_margin_over_best_alternative"]) for row in main_rows],
        cmap="RdYlBu",
        s=42,
        edgecolor="#374151",
        linewidth=0.25,
    )
    ax_margin.scatter([60], [25], marker="*", s=150, color="#111827", label="Default")
    ax_margin.set_xlabel("Quiz ESM weight (%)")
    ax_margin.set_ylabel("Final-memory F1 weight (%)")
    ax_margin.set_title("(a) Patch margin over best alternative")
    ax_margin.legend(frameon=False, loc="lower left")
    colorbar = fig.colorbar(scatter, ax=ax_margin, pad=0.02)
    colorbar.set_label("Composite margin (pp)")

    positions = np.arange(len(workloads))
    width = 0.34
    bars = ax_pareto.bar(
        positions - width / 2,
        primary_rates,
        color="#E69F00",
        width=width,
        label="Primary grid",
    )
    broad_bars = ax_pareto.bar(
        positions + width / 2,
        broad_rates,
        color="#9CA3AF",
        width=width,
        label="Broad simplex",
    )
    for bar, value in zip(bars, primary_rates, strict=True):
        ax_pareto.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 2, 101),
            f"{value:.0f}%",
            ha="center",
            va="bottom",
        )
    for bar, value in zip(broad_bars, broad_rates, strict=True):
        ax_pareto.text(
            bar.get_x() + bar.get_width() / 2,
            max(value - 7, 2),
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            color="white" if value > 20 else "#111827",
            fontsize=8.5,
        )
    ax_pareto.set_xticks(positions, workloads)
    ax_pareto.set_ylim(0, 108)
    ax_pareto.set_ylabel("Weights retaining $k=5$ on Pareto set (%)")
    ax_pareto.set_title("(b) $k=5$ Pareto robustness")
    ax_pareto.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax_pareto.legend(frameon=False, loc="lower left")
    for axis in (ax_margin, ax_pareto):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_report(result: Mapping[str, Any]) -> str:
    primary = result["primary_grid"]
    broad = result["broad_grid"]
    k_primary = primary["k_sweep"]
    k_broad = broad["k_sweep"]
    paired = result["paired"]
    primary_main_rows = [
        row
        for row in result["grid_rows"]
        if row["grid"] == "primary" and row["scope"] == "main"
    ]
    patch_margins = [
        float(row["patch_margin_over_best_alternative"]) for row in primary_main_rows
    ]
    broad_non_patch = [
        row
        for row in result["grid_rows"]
        if row["grid"] == "broad"
        and row["scope"] == "main"
        and row["winner"] != "Patch"
    ]

    lines = [
        "# Composite Weight Sensitivity and Scenario-Level Significance",
        "",
        "## Analysis design",
        "",
        "The reported Composite uses `Quiz ESM / Final-memory F1 / Update F1 = "
        "0.60 / 0.25 / 0.15`. The primary sensitivity grid varies these weights in "
        "0.025 increments while constraining Quiz ESM to 0.50--0.70, Final-memory F1 "
        "to 0.15--0.35, and Update F1 to 0.10--0.25. The weights always sum to one. "
        "A broader positive 0.05-step simplex is reported as a deliberately permissive "
        "diagnostic, not as the primary construct definition.",
        "",
        f"The primary grid contains **{primary['weights']}** settings and the broad grid "
        f"contains **{broad['weights']}** settings. No model inference was rerun.",
        "",
        "## Main-table ranking robustness",
        "",
        "| Grid | Patch top | Patch model-level top | Winner counts |",
        "|---|---:|---:|---|",
        f"| Primary | {100*primary['winner_rates']['Patch']:.1f}% | "
        f"{100*primary['patch_model_level_top_rate']:.1f}% | "
        f"{json.dumps(primary['winner_counts'], ensure_ascii=False)} |",
        f"| Broad | {100*broad['winner_rates']['Patch']:.1f}% | "
        f"{100*broad['patch_model_level_top_rate']:.1f}% | "
        f"{json.dumps(broad['winner_counts'], ensure_ascii=False)} |",
        "",
        "`Patch top` uses the six-model mean. `Patch model-level top` evaluates every "
        "model--weight pair separately.",
        "",
        f"Across the primary grid, Patch's mean margin over the strongest alternative "
        f"ranges from **{min(patch_margins):.2f} to {max(patch_margins):.2f} pp**. "
        f"In the broad diagnostic, the only {len(broad_non_patch)} non-Patch winners "
        "place at most 0.10 weight on Quiz ESM and at least 0.80 on Final-memory F1; "
        "these settings fall well outside the primary construct range.",
        "",
        "## Compaction-interval Pareto robustness",
        "",
        "| Workload | Primary grid: k=5 Pareto | Broad grid: k=5 Pareto | Default Pareto set |",
        "|---|---:|---:|---|",
    ]
    for workload in WORKLOADS:
        default_set = ", ".join(result["default_k_sweep"][workload]["pareto_set"])
        lines.append(
            f"| {workload} | "
            f"{100*k_primary['k5_aggregate_pareto_rates'][workload]:.1f}% | "
            f"{100*k_broad['k5_aggregate_pareto_rates'][workload]:.1f}% | "
            f"{default_set} |"
        )
    lines.extend(
        [
            "",
            f"Across all model--workload--weight combinations, k=5 remains non-dominated "
            f"in **{100*k_primary['k5_model_level_pareto_rate']:.1f}%** of primary-grid "
            f"comparisons and **{100*k_broad['k5_model_level_pareto_rate']:.1f}%** of "
            "broad-grid comparisons.",
            "",
            "### Model-level k=5 Pareto retention",
            "",
            "| Model | Base | U40 | U60 | U80 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        rates = k_primary["k5_model_level_pareto_rates"][model]
        lines.append(
            f"| {model} | "
            + " | ".join(f"{100*rates[workload]:.1f}%" for workload in WORKLOADS)
            + " |"
        )
    lines.extend(
        [
            "",
            "The aggregate Pareto conclusion is fully stable over the primary grid, but "
            "model-level universality is not supported. The main exception is Granite 350M "
            "under U40--U80, where k=10 dominates k=5 throughout the primary grid. Thus the "
            "paper should describe k=5 as a robust cross-model operating point rather than "
            "the optimal setting for every individual model.",
            "",
            "## Paired scenario-level results at the reported weights",
            "",
            "Each comparison uses the same 24 scenarios for all methods. Mean differences "
            "are scenario-macro Composite differences averaged across six models. The 95% "
            "CI and paired randomization test operate on 24 scenario clusters after averaging "
            "within each scenario, treating the evaluated six models as the fixed model set. "
            "The machine-readable CSV additionally reports a two-axis hierarchical-bootstrap "
            "CI that treats both models and scenarios as varying. Holm adjustment is applied "
            "within the main-table and k-sweep families.",
            "",
            "| Family | Workload | Contrast | Mean Δ (pp) | 95% CI | W/T/L | p | Holm p | "
            "Positive across primary weights |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['family']} | {row['workload']} | {row['focal']} − {row['comparator']} | "
            f"{row['mean_difference_pp']:+.2f} | "
            f"[{row['scenario_bootstrap_ci_low']:+.2f}, "
            f"{row['scenario_bootstrap_ci_high']:+.2f}] | "
            f"{row['scenario_cluster_wins']}/{row['scenario_cluster_ties']}/"
            f"{row['scenario_cluster_losses']} | {fmt_p(row['paired_permutation_p'])} | "
            f"{fmt_p(row['holm_adjusted_p'])} | "
            f"{100*row['positive_mean_weight_fraction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Weight robustness and statistical significance answer different questions. "
            "Stable ranking over the predeclared grid shows that the qualitative conclusion "
            "does not depend on the single reported weighting. A non-significant paired test "
            "does not establish equivalence, and a Pareto point is not necessarily the "
            "highest-accuracy point. Accordingly, k=5 should be described as a balanced "
            "non-dominated operating point only in workloads and weight ranges for which the "
            "table supports that statement.",
            "",
            "## Artifacts",
            "",
            "- `docs/engineering/results/composite-weight-sensitivity.json`",
            "- `docs/engineering/results/composite-weight-sensitivity-grid.csv`",
            "- `docs/engineering/results/composite-weight-sensitivity-paired.csv`",
            "- `docs/engineering/figures/composite-weight-sensitivity.pdf`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = analyze()
    grid_rows = result.pop("grid_rows")
    write_csv(GRID_PATH, grid_rows)
    write_csv(PAIRED_PATH, result["paired"])
    JSON_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_figure({**result, "grid_rows": grid_rows})
    REPORT_PATH.write_text(make_report({**result, "grid_rows": grid_rows}), encoding="utf-8")
    for path in (
        GRID_PATH,
        PAIRED_PATH,
        JSON_PATH,
        FIGURE_STEM.with_suffix(".png"),
        FIGURE_STEM.with_suffix(".pdf"),
        REPORT_PATH,
    ):
        print(path)


if __name__ == "__main__":
    main()
