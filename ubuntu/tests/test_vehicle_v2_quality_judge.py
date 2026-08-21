from __future__ import annotations

from types import SimpleNamespace

import pytest

from palmclaw_ubuntu.vehicle_bench.v2_quality_evaluation import (
    QualityGoldCall,
    QualityQuiz,
)
from palmclaw_ubuntu.vehicle_bench.v2_quality_judge import (
    OpenAIV2AnswerabilityJudge,
)


def _quiz() -> QualityQuiz:
    return QualityQuiz(
        schema_version="test",
        method="post_hoc",
        scenario_index=1,
        quiz_id="q1",
        quiz_kind="immediate",
        reasoning_type="preference",
        query="Set my cabin temperature.",
        gold_calls=(
            QualityGoldCall(
                name="set_temperature",
                arguments={"temperature": 22},
            ),
        ),
        memory="The driver's preferred temperature is 22 C.",
        memory_sha256="memory-sha",
        source_stage2_sha256="stage2-sha",
        source_quiz_sha256="quiz-sha",
    )


class _FakeResponses:
    def __init__(self, parsed: dict[str, object]) -> None:
        self._parsed = parsed

    def parse(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            id="response-1",
            status="completed",
            output_parsed=self._parsed,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
            ),
        )


def _judge(parsed: dict[str, object]) -> OpenAIV2AnswerabilityJudge:
    client = SimpleNamespace(responses=_FakeResponses(parsed))
    return OpenAIV2AnswerabilityJudge(
        "test-model",
        timeout_seconds=1,
        client=client,
    )


def test_answerability_judge_derives_partial_support() -> None:
    report = _judge(
        {
            "results": [
                {
                    "quiz_id": "q1",
                    "fields": [
                        {
                            "call_index": 0,
                            "field_kind": "tool",
                            "field_name": "set_temperature",
                            "support": "SUPPORTED_BY_QUERY",
                            "evidence_line_ids": [],
                            "reason": "The action identifies the tool.",
                        },
                        {
                            "call_index": 0,
                            "field_kind": "argument",
                            "field_name": "temperature",
                            "support": "MISSING",
                            "evidence_line_ids": [],
                            "reason": "No temperature is recoverable.",
                        },
                    ],
                    "reason": "Only the tool is recoverable.",
                }
            ]
        }
    ).audit((_quiz(),))

    assert report["results"][0]["verdict"] == "PARTIAL_SUPPORT"
    assert report["results"][0]["support_coverage"] == 0.5
    assert report["usage"]["cached_tokens"] == 20


def test_answerability_judge_rejects_unknown_evidence_line() -> None:
    parsed = {
        "results": [
            {
                "quiz_id": "q1",
                "fields": [
                    {
                        "call_index": 0,
                        "field_kind": "tool",
                        "field_name": "set_temperature",
                        "support": "SUPPORTED_BY_MEMORY",
                        "evidence_line_ids": ["L9999"],
                        "reason": "Cited from memory.",
                    },
                    {
                        "call_index": 0,
                        "field_kind": "argument",
                        "field_name": "temperature",
                        "support": "SUPPORTED_BY_MEMORY",
                        "evidence_line_ids": ["L9999"],
                        "reason": "Cited from memory.",
                    },
                ],
                "reason": "Memory supports the call.",
            }
        ]
    }

    with pytest.raises(ValueError, match="Invalid evidence line"):
        _judge(parsed).audit((_quiz(),))
