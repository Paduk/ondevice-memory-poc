from __future__ import annotations

from types import SimpleNamespace

import pytest

from palmclaw_ubuntu.vehicle_bench.dataset import GoldToolCall, VehicleTask
from palmclaw_ubuntu.vehicle_bench.qa_coverage import (
    OpenAIV2FinalMemoryCoverageModel,
)


class FakeResponses:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results
        self.requests: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            id="qa-coverage-response",
            status="completed",
            output_parsed={"results": self.results},
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
            ),
        )


def _tasks() -> tuple[VehicleTask, ...]:
    return tuple(
        VehicleTask(
            id=f"vehicle-01-{index:02d}",
            scenario_index=1,
            event_index=index,
            query=f"Question {index}",
            gold_memory="Gary prefers green ambient lighting.",
            reasoning_type="state_shift",
            gold_calls=(
                GoldToolCall(
                    name="carcontrol_ambient_light",
                    arguments={"color": "green"},
                    source="test",
                ),
            ),
        )
        for index in range(10)
    )


def _result(index: int, verdict: str) -> dict[str, object]:
    supported = verdict == "SUPPORTED"
    return {
        "task_id": f"vehicle-01-{index:02d}",
        "verdict": verdict,
        "evidence_line_ids": ["L0001"] if supported else [],
        "missing_or_ambiguous_fact": "" if supported else "Ambient preference",
        "reason": "The final memory supports the answer." if supported else "Missing.",
    }


def test_final_memory_coverage_passes_with_fewer_than_three_failures() -> None:
    results = [
        _result(index, "MISSING" if index < 2 else "SUPPORTED") for index in range(10)
    ]
    responses = FakeResponses(results)
    model = OpenAIV2FinalMemoryCoverageModel(
        "gpt-5.6-sol",
        timeout_seconds=1,
        client=SimpleNamespace(responses=responses),
    )

    report = model.audit(
        scenario_index=1,
        final_memory="Gary prefers green ambient lighting.",
        tasks=_tasks(),
        failure_threshold=3,
    )

    assert report.recommendation == "PASS"
    assert report.failed_task_count == 2
    assert report.counts == {"SUPPORTED": 8, "PARTIAL": 0, "MISSING": 2}
    assert report.results[-1].memory_evidence == (
        "Gary prefers green ambient lighting.",
    )
    assert responses.requests[0]["reasoning"] == {"effort": "high"}


def test_final_memory_coverage_partial_does_not_trigger_regeneration() -> None:
    results = [
        _result(index, "PARTIAL" if index < 3 else "SUPPORTED") for index in range(10)
    ]
    model = OpenAIV2FinalMemoryCoverageModel(
        "gpt-5.6-sol",
        timeout_seconds=1,
        client=SimpleNamespace(responses=FakeResponses(results)),
    )

    report = model.audit(
        scenario_index=1,
        final_memory="Gary prefers green ambient lighting.",
        tasks=_tasks(),
    )

    assert report.recommendation == "PASS"
    assert report.failed_task_count == 0


def test_final_memory_coverage_recommends_with_four_missing_tasks() -> None:
    results = [
        _result(index, "MISSING" if index < 4 else "SUPPORTED") for index in range(10)
    ]
    model = OpenAIV2FinalMemoryCoverageModel(
        "gpt-5.6-sol",
        timeout_seconds=1,
        client=SimpleNamespace(responses=FakeResponses(results)),
    )

    report = model.audit(
        scenario_index=1,
        final_memory="Gary prefers green ambient lighting.",
        tasks=_tasks(),
    )

    assert report.recommendation == "REGENERATION_RECOMMENDED"
    assert report.failed_task_count == 4


def test_final_memory_coverage_rejects_unknown_evidence_line() -> None:
    results = [_result(index, "SUPPORTED") for index in range(10)]
    results[0]["evidence_line_ids"] = ["L9999"]
    model = OpenAIV2FinalMemoryCoverageModel(
        "gpt-5.6-sol",
        timeout_seconds=1,
        client=SimpleNamespace(responses=FakeResponses(results)),
    )

    with pytest.raises(ValueError, match="line is invalid"):
        model.audit(
            scenario_index=1,
            final_memory="Gary prefers green ambient lighting.",
            tasks=_tasks(),
        )
