from __future__ import annotations

from types import SimpleNamespace

import pytest

from memory_training.aggregate_delta_pending_ablation import aggregate_reports
from memory_training.evaluate_delta_pending_depth import aggregate_depth_metrics
from memory_training.pending_depth_ablation import (
    depth_balanced_diagnostic_rows,
    pending_depth,
    plan_exposure_counts,
)
from memory_training.sampling import EpochPlan, TrainingUnit
from memory_training.train import build_parser


class _Rows:
    def __init__(self, rows):
        self.rows = rows
        self.positions = {
            row_id: position for position, (row_id, _) in enumerate(rows)
        }

    def position_for_row_id(self, row_id):
        return self.positions[row_id]

    def __getitem__(self, position):
        return self.rows[position][1]


class _Catalog:
    def __init__(self, decisions):
        self.decisions = decisions

    def record(self, row_id):
        return SimpleNamespace(decision=self.decisions[row_id])


def _row(depth: int) -> dict:
    return {"input": {"base_summary": "", "pending_updates": [[]] * depth}}


def test_train_parser_preserves_weighted_default_and_accepts_unstratified() -> None:
    parser = build_parser()
    required = ["--model", "granite4-1b", "--method", "delta_v3_compact_k5"]
    default = parser.parse_args(required)
    unstratified = parser.parse_args(
        [*required, "--delta-v3-noop-sampling", "unstratified"]
    )

    assert default.delta_v3_noop_sampling == "depth-weighted"
    assert unstratified.delta_v3_noop_sampling == "unstratified"


def test_pending_depth_validates_compaction_boundary() -> None:
    assert pending_depth(_row(4), compaction_interval=5) == 4
    with pytest.raises(ValueError, match="below k=5"):
        pending_depth(_row(5), compaction_interval=5)


def test_diagnostic_selection_balances_depth_and_decision() -> None:
    rows = []
    decisions = {}
    row_id = 0
    for depth in range(2):
        for decision in ("NO_OP", "UPDATE"):
            for _ in range(3):
                rows.append((row_id, _row(depth)))
                decisions[row_id] = decision
                row_id += 1
    source = _Rows(rows)
    selected = depth_balanced_diagnostic_rows(
        _Catalog(decisions),
        source,
        list(range(row_id)),
        compaction_interval=2,
        max_rows_per_depth_decision=2,
        seed=45,
    )

    assert {depth: len(values) for depth, values in selected.items()} == {0: 4, 1: 4}
    for values in selected.values():
        assert [decisions[value] for value in values].count("NO_OP") == 2
        assert [decisions[value] for value in values].count("UPDATE") == 2


def test_plan_exposure_counts_includes_trajectory_duplicates() -> None:
    plan = EpochPlan(
        epoch=0,
        seed=45,
        independent_row_ids=(0, 1),
        trajectory_windows=(TrainingUnit("trajectory", (1, 2), 21),),
        units=(),
        update_count=1,
        sampled_noop_count=1,
        sampled_adjacent_noop_count=0,
        sampled_random_noop_count=1,
        adjacent_noop_by_distance=(),
    )
    counts = plan_exposure_counts(
        plan,
        depths={0: 0, 1: 1, 2: 1},
        decisions={0: "UPDATE", 1: "NO_OP", 2: "UPDATE"},
    )

    assert counts["independent"]["UPDATE"] == {"0": 1, "1": 0}
    assert counts["total"]["NO_OP"] == {"0": 0, "1": 2}
    assert counts["total"]["UPDATE"] == {"0": 1, "1": 1}


def test_depth_metric_aggregation_reports_macro_and_micro_results() -> None:
    result = aggregate_depth_metrics(
        {
            "0": {
                "rows": 4,
                "state_f1": 0.5,
                "update_recall": 0.5,
                "false_update_rate": 0.0,
                "decision_counts": {
                    "true_update": 1,
                    "false_update": 0,
                    "missed_update": 1,
                    "true_noop": 2,
                    "invalid": 0,
                    "invalid_noop": 0,
                },
            },
            "1": {
                "rows": 4,
                "state_f1": 1.0,
                "update_recall": 1.0,
                "false_update_rate": 0.5,
                "decision_counts": {
                    "true_update": 2,
                    "false_update": 1,
                    "missed_update": 0,
                    "true_noop": 1,
                    "invalid": 0,
                    "invalid_noop": 0,
                },
            },
        }
    )

    assert result["update_recall"] == 0.75
    assert result["false_update_rate"] == 0.25
    assert result["macro_depth_update_recall"] == 0.75
    assert result["macro_depth_false_update_rate"] == 0.25
    assert result["state_f1"] == 0.75


def test_ablation_aggregation_uses_paired_seed_differences() -> None:
    metric_names = (
        "test_composite",
        "closed_loop_quiz_esm",
        "closed_loop_final_state_f1",
        "closed_loop_update_f1",
        "closed_loop_false_update_rate",
        "diagnostic_update_recall",
        "diagnostic_false_update_rate",
    )

    def report(seed: int, value: float) -> dict:
        return {"seed": seed, **{metric: value for metric in metric_names}}

    result = aggregate_reports(
        {
            "unstratified": [report(45, 0.4), report(46, 0.5)],
            "depth-weighted": [report(45, 0.5), report(46, 0.7)],
        }
    )

    paired = result["paired_difference_summary"]["test_composite"]
    assert paired["mean"] == pytest.approx(0.15)
    assert result["paired_differences"][0]["seed"] == 45
