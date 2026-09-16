"""Derive and validate the fixed quota for the HVP quiz expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REASONING_TYPES = (
    "conditional_constraint",
    "coreference_resolution",
    "error_correction",
    "preference_conflict",
    "state_shift",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-freeze",
        type=Path,
        default=Path(
            "memory_training/data/human-authored-v2-quiz-expansion-v1/"
            "baseline-freeze.json"
        ),
    )
    parser.add_argument(
        "--reference-quiz-sft",
        type=Path,
        default=Path(
            "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
            "grouped-v2-v1-10-eval-fixed-noop5-seed45-v1/quiz_sft.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "memory_training/data/human-authored-v2-quiz-expansion-v1/"
            "quota-plan.json"
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apportion(total: int, weights: Counter[str]) -> dict[str, int]:
    denominator = sum(weights.values())
    raw = {key: total * value / denominator for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return dict(sorted(result.items()))


def minimum_compatible_total(
    baseline: Counter[str], reference: Counter[str]
) -> tuple[int, dict[str, int]]:
    reference_total = sum(reference.values())
    # Use the unrounded expectation as the lower-bound criterion.  This avoids
    # declaring a smaller set compatible only because largest-remainder rounding
    # happens to donate a marginal seat to an already overrepresented category.
    total = math.ceil(
        max(
            baseline[key] * reference_total / reference[key]
            for key in baseline
        )
    )
    target = apportion(total, reference)
    if not all(target.get(key, 0) >= value for key, value in baseline.items()):
        raise RuntimeError("Unrounded lower bound produced an incompatible allocation")
    return total, target


def best_final_addition(
    *,
    baseline_cross: Counter[str],
    additions_by_reasoning: dict[str, int],
    reference_cross: Counter[str],
    expanded_total: int,
    final_additions: int,
) -> dict[str, int]:
    """Minimize Pearson distance to the reference joint distribution.

    Reasoning-type and total TURN/FINAL margins remain fixed.  Exhaustive
    enumeration is tiny here because only 24 new FINAL rows are allocated.
    """

    reference_total = sum(reference_cross.values())
    expected = {
        key: value * expanded_total / reference_total
        for key, value in reference_cross.items()
    }
    best: tuple[float, tuple[int, ...]] | None = None

    def visit(index: int, remaining: int, values: list[int]) -> None:
        nonlocal best
        if index == len(REASONING_TYPES) - 1:
            value = remaining
            key = REASONING_TYPES[index]
            if value > additions_by_reasoning[key]:
                return
            candidate = (*values, value)
            score = 0.0
            for reasoning, final_added in zip(REASONING_TYPES, candidate):
                total_added = additions_by_reasoning[reasoning]
                observed_final = baseline_cross[f"FINAL::{reasoning}"] + final_added
                observed_turn = (
                    baseline_cross[f"TURN::{reasoning}"]
                    + total_added
                    - final_added
                )
                for quiz_type, observed in (
                    ("FINAL", observed_final),
                    ("TURN", observed_turn),
                ):
                    expected_cell = expected[f"{quiz_type}::{reasoning}"]
                    score += (observed - expected_cell) ** 2 / expected_cell
            ranked = (score, candidate)
            if best is None or ranked < best:
                best = ranked
            return

        reasoning = REASONING_TYPES[index]
        for value in range(min(remaining, additions_by_reasoning[reasoning]) + 1):
            visit(index + 1, remaining - value, [*values, value])

    visit(0, final_additions, [])
    if best is None:
        raise ValueError("No feasible TURN/FINAL allocation")
    return dict(zip(REASONING_TYPES, best[1]))


def scenario_quotas() -> list[dict[str, Any]]:
    """Balanced 18/19-row allocation with exact global cell margins."""

    core_extras = {1, 2, 3, 4, 5, 6}
    error_extras = {7, 8, 9, 10, 11, 12, 13}
    preference_extras = {14, 15, 16, 17, 18, 19, 20}
    state_extras = {1, 2, 3, 4, 12, 13, 14, 15, 16, 17}
    rows: list[dict[str, Any]] = []
    for index in range(1, 21):
        turn = {
            "conditional_constraint": 0,
            "coreference_resolution": 5 + (index in core_extras),
            "error_correction": 2 + (index in error_extras),
            "preference_conflict": 6 + (index in preference_extras),
            "state_shift": 3 + (index in state_extras),
        }
        final = {
            "conditional_constraint": 0,
            "coreference_resolution": int(index <= 11),
            "error_correction": 0,
            "preference_conflict": int(index >= 8),
            "state_shift": 0,
        }
        rows.append(
            {
                "scenario_id": f"HVP{index:02d}",
                "scenario_index": 900 + index,
                "turn": turn,
                "final": final,
                "total": sum(turn.values()) + sum(final.values()),
            }
        )
    return rows


def aggregate_scenario_cells(
    rows: Iterable[dict[str, Any]], quiz_type: str
) -> Counter[str]:
    key = quiz_type.lower()
    return Counter(
        {
            reasoning: sum(row[key][reasoning] for row in rows)
            for reasoning in REASONING_TYPES
        }
    )


def build_plan(baseline_path: Path, reference_path: Path) -> dict[str, Any]:
    baseline_path = baseline_path.resolve(strict=True)
    reference_path = reference_path.resolve(strict=True)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    reference_rows = [
        row for row in load_jsonl(reference_path) if row.get("sft_split") == "test"
    ]

    baseline_reasoning = Counter(baseline["distributions"]["reasoning_type"])
    baseline_quiz_type = Counter(baseline["distributions"]["quiz_type"])
    baseline_cross = Counter(
        baseline["distributions"]["quiz_type_by_reasoning_type"]
    )
    reference_reasoning = Counter(row["reasoning_type"] for row in reference_rows)
    reference_quiz_type = Counter(row["quiz_type"] for row in reference_rows)
    reference_cross = Counter(
        f"{row['quiz_type']}::{row['reasoning_type']}" for row in reference_rows
    )
    if set(reference_reasoning) != set(REASONING_TYPES):
        raise ValueError("Reference does not contain the five expected reasoning types")

    expanded_total, target_reasoning = minimum_compatible_total(
        baseline_reasoning, reference_reasoning
    )
    target_quiz_type = apportion(expanded_total, reference_quiz_type)
    additions_reasoning = {
        key: target_reasoning[key] - baseline_reasoning[key]
        for key in REASONING_TYPES
    }
    additions_quiz_type = {
        key: target_quiz_type[key] - baseline_quiz_type[key]
        for key in ("TURN", "FINAL")
    }
    if min(additions_quiz_type.values()) < 0:
        raise ValueError("Target total cannot be reached by quiz addition alone")

    final_by_reasoning = best_final_addition(
        baseline_cross=baseline_cross,
        additions_by_reasoning=additions_reasoning,
        reference_cross=reference_cross,
        expanded_total=expanded_total,
        final_additions=additions_quiz_type["FINAL"],
    )
    addition_cross: dict[str, int] = {}
    for reasoning in REASONING_TYPES:
        addition_cross[f"FINAL::{reasoning}"] = final_by_reasoning[reasoning]
        addition_cross[f"TURN::{reasoning}"] = (
            additions_reasoning[reasoning] - final_by_reasoning[reasoning]
        )

    per_scenario = scenario_quotas()
    planned_turn = aggregate_scenario_cells(per_scenario, "TURN")
    planned_final = aggregate_scenario_cells(per_scenario, "FINAL")
    for reasoning in REASONING_TYPES:
        if planned_turn[reasoning] != addition_cross[f"TURN::{reasoning}"]:
            raise ValueError(f"TURN scenario allocation mismatch for {reasoning}")
        if planned_final[reasoning] != addition_cross[f"FINAL::{reasoning}"]:
            raise ValueError(f"FINAL scenario allocation mismatch for {reasoning}")
    if sum(row["total"] for row in per_scenario) != expanded_total - sum(
        baseline_reasoning.values()
    ):
        raise ValueError("Per-scenario total does not match global addition total")
    if Counter(row["total"] for row in per_scenario) != Counter({19: 14, 18: 6}):
        raise ValueError("Per-scenario allocation is not balanced as designed")

    expanded_cross = {
        key: baseline_cross[key] + addition_cross[key]
        for key in sorted(reference_cross)
    }
    return {
        "schema_version": "vehiclemembench-hvp-quiz-expansion-quota-v1",
        "baseline": {
            "path": str(baseline_path),
            "core_files_sha256": baseline["digests"]["core_files_sha256"],
            "baseline_quiz_files_sha256": baseline["digests"][
                "baseline_quiz_files_sha256"
            ],
            "quiz_count": sum(baseline_reasoning.values()),
        },
        "reference": {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "filter": {"sft_split": "test"},
            "quiz_count": len(reference_rows),
            "reasoning_type_distribution": dict(sorted(reference_reasoning.items())),
            "quiz_type_distribution": dict(sorted(reference_quiz_type.items())),
            "quiz_type_by_reasoning_type": dict(sorted(reference_cross.items())),
        },
        "derivation": {
            "criterion": (
                "smallest integer total whose unrounded reference-proportional "
                "reasoning-type expectations never fall below a baseline count; "
                "integer targets then use largest-remainder apportionment"
            ),
            "expanded_quiz_count": expanded_total,
            "additional_quiz_count": expanded_total - sum(baseline_reasoning.values()),
            "reasoning_type_target": target_reasoning,
            "reasoning_type_additions": additions_reasoning,
            "quiz_type_target": target_quiz_type,
            "quiz_type_additions": additions_quiz_type,
            "addition_quiz_type_by_reasoning_type": dict(sorted(addition_cross.items())),
            "expanded_quiz_type_by_reasoning_type": expanded_cross,
            "joint_allocation_note": (
                "The pre-existing FINAL conditional/error excess cannot be removed by "
                "addition. New FINAL cells minimize Pearson distance to the reference "
                "subject to the fixed marginal quotas."
            ),
        },
        "selection_contract": {
            "method_blind": True,
            "forbidden_inputs": [
                "Summary predictions",
                "Patch predictions",
                "Delta-v3 predictions",
                "per-method error or disagreement results",
            ],
            "candidate_oversampling_minimum": 1.5,
            "deduplication_unit": "normalized query + gold_calls + memory checkpoint",
            "scenario_is_statistical_cluster": True,
        },
        "per_scenario_addition_quota": per_scenario,
    }


def main() -> None:
    args = parse_args()
    plan = build_plan(args.baseline_freeze, args.reference_quiz_sft)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                **{
                    key: plan["derivation"][key]
                    for key in (
                        "expanded_quiz_count",
                        "additional_quiz_count",
                        "reasoning_type_additions",
                        "quiz_type_additions",
                        "addition_quiz_type_by_reasoning_type",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
