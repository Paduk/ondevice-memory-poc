from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from palmclaw_ubuntu.vehicle_bench.dataset import VehicleTask

QA_COVERAGE_VERSION = "vehiclemembench-v2-final-memory-qa-coverage-v2"

type CoverageVerdict = Literal["SUPPORTED", "PARTIAL", "MISSING"]
type CoverageRecommendation = Literal["PASS", "REGENERATION_RECOMMENDED"]

QA_COVERAGE_INSTRUCTIONS = """
Audit whether a frozen final vehicle memory contains the facts required by each
benchmark answer key.

FINAL MEMORY is the only allowed evidence. QUERY only explains what the answer
key refers to. GOLD MEMORY and GOLD TOOL CALLS are answer keys; never treat facts
found only in those fields as support.

For every task return exactly one result in the supplied order:
- SUPPORTED: every fact required by the answer key is explicitly present or
  faithfully paraphrased in FINAL MEMORY.
- PARTIAL: FINAL MEMORY contains relevant support, but at least one required
  identity, value, condition, temporal priority, correction, or conflict
  resolution is missing or ambiguous.
- MISSING: the indispensable answer information is absent or contradicted.

FINAL MEMORY is supplied as stable line IDs and text. For SUPPORTED, return at
least one evidence_line_id copied exactly from that list. Do not copy or
paraphrase the line text into evidence_line_ids. Keep reasons and missing facts
concise. Do not solve the quiz, infer from world knowledge, or use history
outside the supplied final memory.
""".strip()


class QACoverageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=80)
    verdict: CoverageVerdict
    evidence_line_ids: list[str] = Field(max_length=6)
    missing_or_ambiguous_fact: str = Field(max_length=1_024)
    reason: str = Field(min_length=1, max_length=512)

    @field_validator("task_id", "missing_or_ambiguous_fact", "reason")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("evidence_line_ids")
    @classmethod
    def normalize_evidence(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @model_validator(mode="after")
    def validate_verdict_shape(self) -> QACoverageDecision:
        if self.verdict == "SUPPORTED":
            if not self.evidence_line_ids:
                raise ValueError("SUPPORTED requires final-memory evidence")
            if self.missing_or_ambiguous_fact:
                raise ValueError("SUPPORTED cannot report a missing fact")
        elif not self.missing_or_ambiguous_fact:
            raise ValueError("PARTIAL/MISSING requires the absent or ambiguous fact")
        return self


class QACoverageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[QACoverageDecision] = Field(min_length=1, max_length=50)


class QACoverageItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    verdict: CoverageVerdict
    memory_evidence: tuple[str, ...]
    missing_or_ambiguous_fact: str
    reason: str


class QACoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = QA_COVERAGE_VERSION
    scenario_index: int = Field(ge=1)
    final_memory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str
    response_id: str | None
    failure_threshold: int = Field(ge=1)
    recommendation: CoverageRecommendation
    counts: dict[str, int]
    failed_task_count: int = Field(ge=0)
    results: tuple[QACoverageItem, ...]
    usage: dict[str, int]


class OpenAIV2FinalMemoryCoverageModel:
    def __init__(
        self,
        model_id: str,
        *,
        timeout_seconds: float,
        max_output_tokens: int = 8_192,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("QA coverage model ID is required")
        self.model_id = model_id
        self.instructions = QA_COVERAGE_INSTRUCTIONS
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def audit(
        self,
        *,
        scenario_index: int,
        final_memory: str,
        tasks: Sequence[VehicleTask],
        failure_threshold: int = 3,
    ) -> QACoverageReport:
        if not tasks:
            raise ValueError("QA coverage requires benchmark tasks")
        if failure_threshold < 1:
            raise ValueError("QA failure threshold must be positive")
        task_payload = [
            {
                "task_id": task.id,
                "query": task.query,
                "gold_memory": task.gold_memory,
                "gold_tool_calls": [call.as_official() for call in task.gold_calls],
            }
            for task in tasks
        ]
        memory_lines = {
            f"L{index:04d}": line
            for index, line in enumerate(final_memory.splitlines(), start=1)
            if line.strip()
        }
        provider_input = {
            "scenario_index": scenario_index,
            "final_memory_lines": [
                {"line_id": line_id, "text": text}
                for line_id, text in memory_lines.items()
            ],
            "tasks": task_payload,
        }
        response = self._client.responses.parse(
            model=self.model_id,
            instructions=self.instructions,
            input=json.dumps(
                provider_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            text_format=QACoverageResponse,
            max_output_tokens=self.max_output_tokens,
            reasoning={"effort": "high"},
            store=False,
        )
        status = getattr(response, "status", None)
        if status in {"failed", "incomplete", "cancelled"}:
            raise RuntimeError(f"QA coverage provider returned status {status}")
        parsed = QACoverageResponse.model_validate(
            getattr(response, "output_parsed", None)
        )
        expected_ids = [task.id for task in tasks]
        actual_ids = [item.task_id for item in parsed.results]
        if actual_ids != expected_ids:
            raise ValueError(
                f"QA coverage task order mismatch: {actual_ids} != {expected_ids}"
            )
        grounded_results = []
        for item in parsed.results:
            evidence = []
            for line_id in item.evidence_line_ids:
                line = memory_lines.get(line_id)
                if line is None:
                    raise ValueError(
                        f"QA coverage evidence line is invalid: {item.task_id}"
                    )
                evidence.append(line)
            grounded_results.append(
                QACoverageItem(
                    task_id=item.task_id,
                    verdict=item.verdict,
                    memory_evidence=tuple(evidence),
                    missing_or_ambiguous_fact=item.missing_or_ambiguous_fact,
                    reason=item.reason,
                )
            )
        counts = Counter(item.verdict for item in grounded_results)
        failed = counts["MISSING"]
        return QACoverageReport(
            scenario_index=scenario_index,
            final_memory_sha256=hashlib.sha256(
                final_memory.encode("utf-8")
            ).hexdigest(),
            model_id=self.model_id,
            response_id=getattr(response, "id", None),
            failure_threshold=failure_threshold,
            recommendation=qa_coverage_recommendation(
                counts,
                failure_threshold=failure_threshold,
            ),
            counts={
                verdict: counts[verdict]
                for verdict in ("SUPPORTED", "PARTIAL", "MISSING")
            },
            failed_task_count=failed,
            results=tuple(grounded_results),
            usage=_response_usage(response),
        )


def qa_coverage_recommendation(
    counts: Mapping[str, int],
    *,
    failure_threshold: int = 3,
) -> CoverageRecommendation:
    if failure_threshold < 1:
        raise ValueError("QA failure threshold must be positive")
    failed = int(counts.get("MISSING", 0))
    return "REGENERATION_RECOMMENDED" if failed > failure_threshold else "PASS"


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
