from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from palmclaw_ubuntu.vehicle_bench.v2_quality_evaluation import QualityQuiz

ANSWERABILITY_PROMPT_VERSION = "vehiclemembench-v2-answerability-judge-v2"
ANSWERABILITY_SCHEMA_VERSION = "vehiclemembench-v2-answerability-result-v1"

type FieldSupport = Literal[
    "SUPPORTED_BY_MEMORY",
    "SUPPORTED_BY_QUERY",
    "SUPPORTED_BY_BOTH",
    "MISSING",
    "CONTRADICTED",
]
type AnswerabilityVerdict = Literal[
    "FULL_SUPPORT",
    "PARTIAL_SUPPORT",
    "UNSUPPORTED",
    "CONTRADICTED",
]

ANSWERABILITY_INSTRUCTIONS = """
Judge whether QUERY plus CUTOFF MEMORY contains enough evidence to reconstruct
each supplied GOLD CALL. GOLD CALLS define the fields to audit; they are never
evidence. Do not use world knowledge, likely defaults, or dialogue outside the
input.

Return one field decision for the Tool name and every Argument of every gold
call, in the supplied order. For a Tool decision, set field_name to the exact
gold Tool name; for an Argument decision, use the exact argument key:
- SUPPORTED_BY_MEMORY: explicitly stated or faithfully paraphrased in memory.
- SUPPORTED_BY_QUERY: explicitly fixed by the query, without relying on memory.
- SUPPORTED_BY_BOTH: independently supported by both.
- MISSING: not uniquely recoverable from query plus memory.
- CONTRADICTED: memory explicitly supports a conflicting Tool or value.

Memory is supplied as stable line IDs. Memory-based and contradicted decisions
must cite exact line IDs; query-only and missing decisions must cite none. A
condition merely selecting a remembered fact is not support for that fact's
hidden value. Keep every reason to one short sentence.
""".strip()


class AnswerabilityFieldDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_index: int = Field(ge=0)
    field_kind: Literal["tool", "argument"]
    field_name: str = Field(min_length=1, max_length=128)
    support: FieldSupport
    evidence_line_ids: list[str] = Field(max_length=8)
    reason: str = Field(min_length=1, max_length=512)

    @field_validator("field_name", "reason")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("evidence_line_ids")
    @classmethod
    def normalize_lines(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @model_validator(mode="after")
    def validate_evidence_shape(self) -> AnswerabilityFieldDecision:
        needs_memory = self.support in {
            "SUPPORTED_BY_MEMORY",
            "SUPPORTED_BY_BOTH",
            "CONTRADICTED",
        }
        if needs_memory and not self.evidence_line_ids:
            raise ValueError(f"{self.support} requires memory evidence")
        if not needs_memory and self.evidence_line_ids:
            raise ValueError(f"{self.support} cannot cite memory evidence")
        return self


class AnswerabilityQuizDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quiz_id: str = Field(min_length=1, max_length=160)
    fields: list[AnswerabilityFieldDecision] = Field(min_length=1, max_length=32)
    reason: str = Field(min_length=1, max_length=768)


class AnswerabilityBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[AnswerabilityQuizDecision] = Field(min_length=1, max_length=20)


class OpenAIV2AnswerabilityJudge:
    def __init__(
        self,
        model_id: str,
        *,
        timeout_seconds: float,
        max_output_tokens: int = 16_384,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Answerability Judge model ID is required")
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def audit(
        self,
        quizzes: Sequence[QualityQuiz],
    ) -> dict[str, Any]:
        if not quizzes:
            raise ValueError("Answerability batch cannot be empty")
        memory_sha256 = quizzes[0].memory_sha256
        memory = quizzes[0].memory
        if any(
            quiz.memory_sha256 != memory_sha256 or quiz.memory != memory
            for quiz in quizzes
        ):
            raise ValueError("Answerability batch must share one memory snapshot")
        memory_lines = {
            f"L{index:04d}": line
            for index, line in enumerate(memory.splitlines(), start=1)
            if line.strip()
        }
        provider_input = {
            "memory_sha256": memory_sha256,
            "cutoff_memory_lines": [
                {"line_id": line_id, "text": text}
                for line_id, text in memory_lines.items()
            ],
            "quizzes": [
                {
                    "quiz_id": quiz.quiz_id,
                    "query": quiz.query,
                    "gold_calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in quiz.gold_calls
                    ],
                }
                for quiz in quizzes
            ],
        }
        response = self._client.responses.parse(
            model=self.model_id,
            instructions=ANSWERABILITY_INSTRUCTIONS,
            input=json.dumps(
                provider_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            text_format=AnswerabilityBatchResponse,
            max_output_tokens=self.max_output_tokens,
            reasoning={"effort": "high"},
            store=False,
        )
        status = getattr(response, "status", None)
        if status in {"failed", "incomplete", "cancelled"}:
            raise RuntimeError(f"Answerability Judge returned status {status}")
        parsed = AnswerabilityBatchResponse.model_validate(
            getattr(response, "output_parsed", None)
        )
        results = _ground_answerability_results(
            quizzes,
            parsed,
            memory_lines=memory_lines,
        )
        return {
            "schema_version": ANSWERABILITY_SCHEMA_VERSION,
            "prompt_version": ANSWERABILITY_PROMPT_VERSION,
            "model_id": self.model_id,
            "response_id": getattr(response, "id", None),
            "memory_sha256": memory_sha256,
            "input_sha256": _canonical_sha256(provider_input),
            "results": results,
            "usage": _response_usage(response),
        }


def _ground_answerability_results(
    quizzes: Sequence[QualityQuiz],
    response: AnswerabilityBatchResponse,
    *,
    memory_lines: dict[str, str],
) -> list[dict[str, Any]]:
    expected_ids = [quiz.quiz_id for quiz in quizzes]
    actual_ids = [item.quiz_id for item in response.results]
    if actual_ids != expected_ids:
        raise ValueError(f"Answerability Quiz order mismatch: {actual_ids}")
    grounded: list[dict[str, Any]] = []
    for quiz, item in zip(quizzes, response.results, strict=True):
        expected_fields = []
        for call_index, call in enumerate(quiz.gold_calls):
            expected_fields.append((call_index, "tool", call.name))
            expected_fields.extend(
                (call_index, "argument", name) for name in call.arguments
            )
        actual_fields = [
            (field.call_index, field.field_kind, field.field_name)
            for field in item.fields
        ]
        if actual_fields != expected_fields:
            raise ValueError(
                f"Answerability field mismatch for {quiz.quiz_id}: "
                f"{actual_fields} != {expected_fields}"
            )
        field_results = []
        supported = 0
        contradicted = False
        for field in item.fields:
            evidence = []
            for line_id in field.evidence_line_ids:
                line = memory_lines.get(line_id)
                if line is None:
                    raise ValueError(
                        f"Invalid evidence line {line_id}: {quiz.quiz_id}"
                    )
                evidence.append({"line_id": line_id, "text": line})
            is_supported = field.support.startswith("SUPPORTED_BY_")
            supported += int(is_supported)
            contradicted = contradicted or field.support == "CONTRADICTED"
            field_results.append(
                {
                    **field.model_dump(mode="json"),
                    "memory_evidence": evidence,
                }
            )
        if contradicted:
            verdict: AnswerabilityVerdict = "CONTRADICTED"
        elif supported == len(item.fields):
            verdict = "FULL_SUPPORT"
        elif supported:
            verdict = "PARTIAL_SUPPORT"
        else:
            verdict = "UNSUPPORTED"
        grounded.append(
            {
                "quiz_id": quiz.quiz_id,
                "verdict": verdict,
                "supported_field_count": supported,
                "total_field_count": len(item.fields),
                "support_coverage": supported / len(item.fields),
                "fields": field_results,
                "reason": item.reason.strip(),
            }
        )
    return grounded


def _canonical_sha256(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
