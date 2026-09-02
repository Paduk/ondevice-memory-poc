from __future__ import annotations

import pytest

from memory_training.validation import _DecisionUsage, merge_decision_usage


def test_decision_usage_reports_gold_predicted_and_pair_totals() -> None:
    usage = _DecisionUsage()
    usage.add(
        "UPDATE",
        "UPDATE",
        prefill_tokens=100,
        decode_tokens=20,
        attributed_latency_seconds=2.0,
    )
    usage.add(
        "NO_OP",
        "UPDATE",
        prefill_tokens=80,
        decode_tokens=12,
        attributed_latency_seconds=1.0,
    )
    report = usage.as_dict()

    assert report["usage_by_gold_decision"]["UPDATE"]["turns"] == 1
    assert report["usage_by_predicted_decision"]["UPDATE"]["turns"] == 2
    assert report["usage_by_predicted_decision"]["UPDATE"]["total_tokens"] == 212
    assert report["usage_by_decision_pair"]["NO_OP->UPDATE"]["decode_tokens"] == 12
    assert report["usage_by_decision_pair"]["UPDATE->NO_OP"]["turns"] == 0
    assert report["decision_usage_latency_kind"] == "attributed_wall_clock"


def test_merge_decision_usage_rebuilds_marginal_totals_from_pairs() -> None:
    first = _DecisionUsage()
    first.add(
        "UPDATE",
        "NO_OP",
        prefill_tokens=70,
        decode_tokens=5,
        attributed_latency_seconds=0.5,
    )
    second = _DecisionUsage()
    second.add(
        "UPDATE",
        "UPDATE",
        prefill_tokens=90,
        decode_tokens=15,
        attributed_latency_seconds=1.5,
    )

    merged = merge_decision_usage([first.as_dict(), second.as_dict()])

    gold_update = merged["usage_by_gold_decision"]["UPDATE"]
    assert gold_update["turns"] == 2
    assert gold_update["prefill_tokens"] == 160
    assert gold_update["decode_tokens"] == 20
    assert gold_update["average_total_tokens"] == pytest.approx(90.0)
    assert merged["usage_by_predicted_decision"]["NO_OP"]["turns"] == 1
    assert merged["usage_by_decision_pair"]["UPDATE->UPDATE"][
        "attributed_latency_seconds"
    ] == pytest.approx(1.5)
