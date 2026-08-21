from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from palmclaw_ubuntu.vehicle_bench.v2_update_audit import EventAuditCase
from palmclaw_ubuntu.vehicle_bench.v2_update_judge import (
    UPDATE_CONSENSUS_SCHEMA_VERSION,
    UpdateEventJudgePayload,
    ground_update_event_decision,
)

UPDATE_ADJUDICATOR_PROMPT_VERSION = "vehiclemembench-v2-update-sol-adjudicator-v1"
UPDATE_ADJUDICATOR_SCHEMA_VERSION = "vehiclemembench-v2-update-sol-result-v1"

Resolution = Literal["RESOLVE", "DEFER_HUMAN"]

UPDATE_ADJUDICATOR_INSTRUCTIONS = """
Resolve a disagreement between two independent vehicle-memory audit Judges.
Use the original event dialogue as the only factual evidence. expected_updates
define fields to audit but are not evidence. The prior Judge outputs are
arguments to inspect, not facts to trust.

Return RESOLVE only when the dialogue and memory transition support one complete
structured audit decision. Return DEFER_HUMAN when evidence is insufficient,
multiple interpretations remain plausible, or a grounded decision cannot be
made. Never force a resolution. Evidence quotes in a resolved decision must be
exact substrings of the cited event turns. Keep reasons short.
""".strip()


class SolUpdateAdjudicationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=320)
    resolution: Resolution
    decision: UpdateEventJudgePayload | None = None
    reason: str = Field(min_length=1, max_length=768)

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_resolution(self) -> SolUpdateAdjudicationPayload:
        if (self.resolution == "RESOLVE") != (self.decision is not None):
            raise ValueError("RESOLVE requires decision; DEFER_HUMAN forbids it")
        return self


class OpenAIV2UpdateAuditAdjudicator:
    def __init__(
        self,
        model_id: str,
        *,
        timeout_seconds: float,
        max_output_tokens: int = 16_384,
        client: Any | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Update audit adjudicator model ID is required")
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout_seconds)
        self._client = client

    def adjudicate(
        self,
        case: EventAuditCase,
        consensus: dict[str, Any],
    ) -> dict[str, Any]:
        if consensus.get("schema_version") != UPDATE_CONSENSUS_SCHEMA_VERSION:
            raise ValueError("Invalid UPDATE audit consensus schema")
        if consensus.get("case_id") != case.case_id:
            raise ValueError("UPDATE audit consensus case mismatch")
        if consensus.get("status") != "DISAGREED":
            raise ValueError("SOL adjudication requires DISAGREED consensus")
        provider_input = {
            "case": case.as_dict(),
            "luna_decision": consensus["judge_reports"]["luna"]["decision"],
            "terra_decision": consensus["judge_reports"]["terra"]["decision"],
            "material_diff": consensus["material_diff"],
        }
        response = self._client.responses.parse(
            model=self.model_id,
            instructions=UPDATE_ADJUDICATOR_INSTRUCTIONS,
            input=json.dumps(
                provider_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            text_format=SolUpdateAdjudicationPayload,
            max_output_tokens=self.max_output_tokens,
            reasoning={"effort": "high"},
            store=False,
        )
        status = getattr(response, "status", None)
        if status in {"failed", "incomplete", "cancelled"}:
            raise RuntimeError(f"Update audit SOL returned status {status}")
        parsed = SolUpdateAdjudicationPayload.model_validate(
            getattr(response, "output_parsed", None)
        )
        if parsed.case_id != case.case_id:
            raise ValueError("SOL adjudication changed case_id")
        grounded = (
            ground_update_event_decision(case, parsed.decision)
            if parsed.decision is not None
            else None
        )
        return {
            "schema_version": UPDATE_ADJUDICATOR_SCHEMA_VERSION,
            "prompt_version": UPDATE_ADJUDICATOR_PROMPT_VERSION,
            "model_id": self.model_id,
            "response_id": getattr(response, "id", None),
            "case_id": case.case_id,
            "case_input_sha256": case.input_sha256,
            "consensus_sha256": _canonical_sha256(consensus),
            "resolution": parsed.resolution,
            "decision": grounded,
            "reason": parsed.reason,
            "usage": _response_usage(response),
        }


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
