"""Shared helpers for the Delta-v3 pending-depth sampling ablation."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .dataset import DatasetCatalog, IndexedMemoryDataset
from .sampling import EpochPlan


def pending_depth(row: Mapping[str, Any], *, compaction_interval: int) -> int:
    """Return the gold pending depth encoded in one Delta training row."""
    if compaction_interval < 1:
        raise ValueError("compaction_interval must be positive")
    memory_input = row.get("input")
    if not isinstance(memory_input, Mapping):
        raise TypeError("Delta row input must be an object")
    pending = memory_input.get("pending_updates")
    if not isinstance(pending, Sequence) or isinstance(pending, (str, bytes)):
        raise TypeError("Delta row pending_updates must be an array")
    depth = len(pending)
    if depth >= compaction_interval:
        raise ValueError(
            f"Pending depth {depth} must be below k={compaction_interval}"
        )
    return depth


def pending_depths_by_row_id(
    source: IndexedMemoryDataset,
    row_ids: Sequence[int],
    *,
    compaction_interval: int,
) -> dict[int, int]:
    """Read each selected row once and map its catalog ID to pending depth."""
    return {
        row_id: pending_depth(
            source[source.position_for_row_id(row_id)],
            compaction_interval=compaction_interval,
        )
        for row_id in row_ids
    }


def depth_balanced_diagnostic_rows(
    catalog: DatasetCatalog,
    source: IndexedMemoryDataset,
    row_ids: Sequence[int],
    *,
    compaction_interval: int,
    max_rows_per_depth_decision: int,
    seed: int,
) -> dict[int, tuple[int, ...]]:
    """Select the same number of gold UPDATE/NO_OP rows at every pending depth."""
    if max_rows_per_depth_decision < 1:
        raise ValueError("max_rows_per_depth_decision must be positive")
    depths = pending_depths_by_row_id(
        source, row_ids, compaction_interval=compaction_interval
    )
    grouped: dict[tuple[int, str], list[int]] = {
        (depth, decision): []
        for depth in range(compaction_interval)
        for decision in ("NO_OP", "UPDATE")
    }
    for row_id in row_ids:
        decision = catalog.record(row_id).decision
        grouped[(depths[row_id], decision)].append(row_id)
    missing = [key for key, values in grouped.items() if not values]
    if missing:
        rendered = ", ".join(f"d={depth}/{decision}" for depth, decision in missing)
        raise ValueError(f"Diagnostic source has empty strata: {rendered}")
    per_stratum = min(
        max_rows_per_depth_decision,
        *(len(values) for values in grouped.values()),
    )
    selected: dict[int, tuple[int, ...]] = {}
    for depth in range(compaction_interval):
        depth_rows = []
        for decision_index, decision in enumerate(("NO_OP", "UPDATE")):
            candidates = grouped[(depth, decision)]
            generator = random.Random(seed + depth * 1_000_003 + decision_index)
            depth_rows.extend(generator.sample(candidates, per_stratum))
        selected[depth] = tuple(sorted(depth_rows))
    return selected


def plan_exposure_counts(
    plan: EpochPlan,
    depths: Mapping[int, int],
    decisions: Mapping[int, str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Count depth/decision exposures, including duplicated trajectory turns."""
    counters: dict[str, Counter[tuple[str, int]]] = {
        "independent": Counter(),
        "trajectory": Counter(),
    }
    for row_id in plan.independent_row_ids:
        counters["independent"][(decisions[row_id], depths[row_id])] += 1
    for window in plan.trajectory_windows:
        for row_id in window.row_ids:
            counters["trajectory"][(decisions[row_id], depths[row_id])] += 1
    counters["total"] = counters["independent"] + counters["trajectory"]
    return {
        kind: {
            decision: {
                str(depth): int(counter[(decision, depth)])
                for depth in sorted(set(depths.values()))
            }
            for decision in ("NO_OP", "UPDATE")
        }
        for kind, counter in counters.items()
    }


def merge_exposure_counts(
    reports: Sequence[Mapping[str, Mapping[str, Mapping[str, int]]]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Sum exposure reports produced by :func:`plan_exposure_counts`."""
    merged: dict[str, dict[str, Counter[str]]] = {}
    for report in reports:
        for kind, by_decision in report.items():
            target = merged.setdefault(
                kind, {"NO_OP": Counter(), "UPDATE": Counter()}
            )
            for decision, by_depth in by_decision.items():
                target[decision].update(
                    {str(depth): int(count) for depth, count in by_depth.items()}
                )
    return {
        kind: {
            decision: dict(sorted(counter.items(), key=lambda item: int(item[0])))
            for decision, counter in by_decision.items()
        }
        for kind, by_decision in merged.items()
    }
