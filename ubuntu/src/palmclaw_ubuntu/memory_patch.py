from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from palmclaw_ubuntu.models import (
    MemoryEvidence,
    ModelUsage,
    ToolMemoryIdentity,
    ToolMemoryPatch,
    ToolMemoryPatchBatchResult,
    ToolMemoryPatchOutcome,
    ToolMemoryPatchRejection,
    ToolMemoryRecord,
)
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tool_memory_schema import (
    canonicalize_tool_memory_identity_aliases,
    normalize_tool_memory_identity,
    tool_memory_record_key,
)
from palmclaw_ubuntu.validation import (
    ToolMemoryPatchPolicy,
    ToolMemoryPatchValidationError,
    ToolMemoryPatchValidator,
)

TOOL_MEMORY_PATCH_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["ADD", "UPDATE", "MERGE", "DELETE"],
        },
        "identity": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "minLength": 1},
                "tool_domain": {"type": "string", "minLength": 1},
                "topic": {"type": "string", "minLength": 1},
                "scope": {
                    "type": "string",
                    "enum": ["global", "vehicle", "session", "conditional"],
                },
                "scope_key": {"type": "string", "minLength": 1},
                "conditions": {"type": "object"},
            },
            "required": [
                "user_id",
                "tool_domain",
                "topic",
                "scope",
                "scope_key",
            ],
            "additionalProperties": False,
        },
        "value": {},
        "memory_type": {
            "type": "string",
            "enum": [
                "constraint",
                "decision",
                "fact",
                "policy",
                "preference",
                "state",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string", "minLength": 1},
                    "start_char": {"type": "integer", "minimum": 0},
                    "end_char": {"type": "integer", "minimum": 0},
                },
                "required": ["message_id", "quote"],
                "additionalProperties": False,
            },
        },
        "target_record_id": {"type": "string", "minLength": 1},
        "merge_record_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "reason": {"type": "string"},
    },
    "required": ["operation", "confidence", "evidence"],
    "additionalProperties": False,
}

_PATCH_SCHEMA_VALIDATOR = Draft202012Validator(TOOL_MEMORY_PATCH_SCHEMA)


class ToolMemoryPatchSchemaError(ValueError):
    pass


def parse_tool_memory_patch(payload: Mapping[str, Any]) -> ToolMemoryPatch:
    try:
        _PATCH_SCHEMA_VALIDATOR.validate(dict(payload))
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        suffix = f" at {location}" if location else ""
        raise ToolMemoryPatchSchemaError(
            f"Invalid Tool memory patch{suffix}: {exc.message}"
        ) from exc
    identity_payload = payload.get("identity")
    identity = None
    if isinstance(identity_payload, Mapping):
        identity = ToolMemoryIdentity(
            user_id=str(identity_payload["user_id"]),
            tool_domain=str(identity_payload["tool_domain"]),
            topic=str(identity_payload["topic"]),
            scope=str(identity_payload["scope"]),
            scope_key=str(identity_payload["scope_key"]),
            conditions=dict(identity_payload.get("conditions", {})),
        )
    evidence = tuple(
        MemoryEvidence(
            message_id=int(item["message_id"]),
            quote=str(item["quote"]),
            start_char=item.get("start_char"),
            end_char=item.get("end_char"),
        )
        for item in payload["evidence"]
    )
    return ToolMemoryPatch(
        operation=str(payload["operation"]),
        confidence=float(payload["confidence"]),
        evidence=evidence,
        identity=identity,
        value=payload.get("value"),
        memory_type=(
            str(payload["memory_type"]) if "memory_type" in payload else None
        ),
        target_record_id=(
            str(payload["target_record_id"])
            if "target_record_id" in payload
            else None
        ),
        merge_record_ids=tuple(
            str(item) for item in payload.get("merge_record_ids", ())
        ),
        reason=str(payload.get("reason", "")),
    )


def tool_memory_patch_payload(
    patch: ToolMemoryPatch,
    *,
    include_evidence: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation": patch.operation,
        "confidence": patch.confidence,
    }
    if patch.identity is not None:
        payload["identity"] = {
            "user_id": patch.identity.user_id,
            "tool_domain": patch.identity.tool_domain,
            "topic": patch.identity.topic,
            "scope": patch.identity.scope,
            "scope_key": patch.identity.scope_key,
            "conditions": dict(patch.identity.conditions),
        }
    if patch.operation != "DELETE":
        payload["value"] = patch.value
    if patch.memory_type is not None:
        payload["memory_type"] = patch.memory_type
    if patch.target_record_id is not None:
        payload["target_record_id"] = patch.target_record_id
    if patch.merge_record_ids:
        payload["merge_record_ids"] = list(patch.merge_record_ids)
    if patch.reason:
        payload["reason"] = patch.reason
    if include_evidence:
        payload["evidence"] = [asdict(item) for item in patch.evidence]
    return payload


class ToolMemoryPatchEngine:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        policy: ToolMemoryPatchPolicy | None = None,
    ):
        self.repository = repository
        self.validator = ToolMemoryPatchValidator(policy)

    def apply_batch(
        self,
        run_id: str,
        patches: Sequence[ToolMemoryPatch | Mapping[str, Any]],
        *,
        usage: ModelUsage | None = None,
        allowed_evidence_message_ids: Sequence[int] = (),
        isolate_invalid_patches: bool = False,
        finish_run: bool = True,
    ) -> ToolMemoryPatchBatchResult:
        parsed = tuple(
            item if isinstance(item, ToolMemoryPatch) else parse_tool_memory_patch(item)
            for item in patches
        )
        if not parsed:
            raise ValueError("Tool memory patch batch cannot be empty")
        run = self.repository.memory_patch_run(run_id)
        if run["status"] == "completed":
            return self._replay_completed(run_id, parsed)
        if run["status"] not in {"queued", "running"}:
            raise ValueError(
                f"Memory patch run cannot be applied from status {run['status']}"
            )

        allowed_evidence_ids = set(allowed_evidence_message_ids)
        messages = {
            message.id: (message.session_id, message.turn_id, message.content)
            for message in self.repository.list_messages(run["session_id"])
            if (
                not allowed_evidence_ids
                or message.id in allowed_evidence_ids
            )
        }
        if isolate_invalid_patches:
            return self._apply_isolated_batch(
                run_id=run_id,
                run=run,
                parsed=parsed,
                messages=messages,
                usage=usage,
                finish_run=finish_run,
            )
        outcomes: list[ToolMemoryPatchOutcome] = []
        current_index = -1
        current_patch: ToolMemoryPatch | None = None
        try:
            with self.repository.memory_patch_transaction():
                if run["status"] == "queued":
                    self.repository.start_memory_patch_run(run_id)
                for current_index, current_patch in enumerate(parsed):
                    outcomes.append(
                        self._apply_one(
                            run_id=run_id,
                            index=current_index,
                            session_id=str(run["session_id"]),
                            source_turn_id=run["source_turn_id"],
                            messages=messages,
                            patch=current_patch,
                        )
                    )
                self.repository.finish_memory_patch_run(
                    run_id,
                    status="completed",
                    usage=usage,
                )
        except BaseException as exc:
            self._record_failed_batch(
                run_id=run_id,
                index=current_index,
                patch=current_patch,
                error=exc,
                usage=usage,
            )
            raise
        return ToolMemoryPatchBatchResult(
            run_id=run_id,
            outcomes=tuple(outcomes),
        )

    def _apply_isolated_batch(
        self,
        *,
        run_id: str,
        run: Mapping[str, Any],
        parsed: Sequence[ToolMemoryPatch],
        messages: Mapping[int, tuple[str, str | None, str]],
        usage: ModelUsage | None,
        finish_run: bool,
    ) -> ToolMemoryPatchBatchResult:
        outcomes: list[ToolMemoryPatchOutcome] = []
        rejections: list[ToolMemoryPatchRejection] = []
        current_index = -1
        current_patch: ToolMemoryPatch | None = None
        try:
            if run["status"] == "queued":
                self.repository.start_memory_patch_run(run_id)
            for current_index, current_patch in enumerate(parsed):
                try:
                    with self.repository.memory_patch_transaction():
                        outcomes.append(
                            self._apply_one(
                                run_id=run_id,
                                index=current_index,
                                session_id=str(run["session_id"]),
                                source_turn_id=run["source_turn_id"],
                                messages=messages,
                                patch=current_patch,
                            )
                        )
                except ToolMemoryPatchValidationError as exc:
                    rejections.append(
                        self._record_rejected_patch(
                            run_id=run_id,
                            index=current_index,
                            patch=current_patch,
                            error=exc,
                        )
                    )
            if finish_run:
                self.repository.finish_memory_patch_run(
                    run_id,
                    status="completed",
                    usage=usage,
                )
        except BaseException as exc:
            self._record_failed_batch(
                run_id=run_id,
                index=current_index,
                patch=current_patch,
                error=exc,
                usage=usage,
            )
            raise
        return ToolMemoryPatchBatchResult(
            run_id=run_id,
            outcomes=tuple(outcomes),
            rejections=tuple(rejections),
        )

    def _apply_one(
        self,
        *,
        run_id: str,
        index: int,
        session_id: str,
        source_turn_id: str | None,
        messages: Mapping[int, tuple[str, str | None, str]],
        patch: ToolMemoryPatch,
    ) -> ToolMemoryPatchOutcome:
        if patch.identity is not None:
            patch = replace(
                patch,
                identity=canonicalize_tool_memory_identity_aliases(
                    patch.identity,
                    self.repository.list_tool_memory_records(
                        session_id,
                        user_id=patch.identity.user_id,
                    ),
                ),
            )
        target = self._target(patch.target_record_id)
        merge_records = self._merge_records(patch.merge_record_ids)
        proposed_key = (
            tool_memory_record_key(patch.identity)
            if patch.identity is not None
            else None
        )
        active = (
            self.repository.active_tool_memory_record(proposed_key)
            if proposed_key is not None
            else None
        )
        validation = self.validator.validate(
            patch,
            session_id=session_id,
            source_turn_id=source_turn_id,
            messages=messages,
            target=target,
            merge_records=merge_records,
            active_for_identity=active,
        )
        normalized = validation.patch
        idempotency_key = f"{run_id}:{index}"
        proposal_id = self.repository.record_memory_patch_proposal(
            run_id=run_id,
            operation=normalized.operation,
            patch=tool_memory_patch_payload(
                normalized,
                include_evidence=False,
            ),
            evidence=[asdict(item) for item in normalized.evidence],
            confidence=normalized.confidence,
            idempotency_key=idempotency_key,
            target_record_id=normalized.target_record_id,
            proposed_record_key=proposed_key,
            validation=validation.as_dict(),
            sequence_index=index,
        )
        if normalized.operation == "ADD":
            result = self._create_record(
                session_id,
                normalized,
                proposal_id=proposal_id,
            )
        elif normalized.operation == "UPDATE":
            assert target is not None
            self.repository.transition_tool_memory_record(
                target.id,
                status="superseded",
                reason=normalized.reason or "patch_update",
                patch_proposal_id=proposal_id,
            )
            result = self._create_record(
                session_id,
                normalized,
                proposal_id=proposal_id,
                supersedes_id=target.id,
                version=max(
                    target.version + 1,
                    self.repository.next_tool_memory_version(
                        tool_memory_record_key(normalized.identity)
                    ),
                ),
            )
        elif normalized.operation == "MERGE":
            assert target is not None
            result_id = str(uuid.uuid4())
            merged = (target, *merge_records)
            for record in merged:
                self.repository.transition_tool_memory_record(
                    record.id,
                    status="merged",
                    reason=normalized.reason or "patch_merge",
                    patch_proposal_id=proposal_id,
                )
            result = self._create_record(
                session_id,
                normalized,
                proposal_id=proposal_id,
                supersedes_id=target.id,
                record_id=result_id,
                evidence=self._merged_evidence(normalized, merged),
                allow_cross_session_evidence=True,
                version=max(
                    max(record.version for record in merged) + 1,
                    self.repository.next_tool_memory_version(
                        tool_memory_record_key(normalized.identity)
                    ),
                ),
            )
            for record in merged:
                self.repository.link_merged_tool_memory_record(
                    record.id,
                    merged_into_id=result.id,
                )
        else:
            assert target is not None
            result = self.repository.transition_tool_memory_record(
                target.id,
                status="deleted",
                reason=normalized.reason or "patch_delete",
                patch_proposal_id=proposal_id,
            )
        self.repository.resolve_memory_patch_proposal(
            proposal_id,
            status="applied",
            result_record_id=result.id,
        )
        return ToolMemoryPatchOutcome(
            operation=normalized.operation,
            proposal_id=proposal_id,
            result_record_id=result.id,
            record_status=result.status,
        )

    def _create_record(
        self,
        session_id: str,
        patch: ToolMemoryPatch,
        *,
        proposal_id: str,
        supersedes_id: str | None = None,
        record_id: str | None = None,
        evidence: Sequence[MemoryEvidence] | None = None,
        allow_cross_session_evidence: bool = False,
        version: int | None = None,
    ) -> ToolMemoryRecord:
        assert patch.identity is not None
        assert patch.memory_type is not None
        record_key = tool_memory_record_key(patch.identity)
        return self.repository.insert_tool_memory_record(
            session_id=session_id,
            identity=patch.identity,
            value=patch.value,
            memory_type=patch.memory_type,
            confidence=patch.confidence,
            version=version or self.repository.next_tool_memory_version(record_key),
            supersedes_id=supersedes_id,
            evidence=tuple(evidence or patch.evidence),
            record_id=record_id,
            patch_proposal_id=proposal_id,
            status_reason=f"patch_{patch.operation.casefold()}",
            allow_cross_session_evidence=allow_cross_session_evidence,
        )

    def _target(self, record_id: str | None) -> ToolMemoryRecord | None:
        if record_id is None:
            return None
        record = self.repository.tool_memory_record(record_id)
        if record is None:
            raise ToolMemoryPatchValidationError(
                "missing_target",
                f"Unknown target record: {record_id}",
            )
        return record

    def _merge_records(
        self,
        record_ids: Sequence[str],
    ) -> tuple[ToolMemoryRecord, ...]:
        records = []
        for record_id in record_ids:
            record = self.repository.tool_memory_record(record_id)
            if record is None:
                raise ToolMemoryPatchValidationError(
                    "missing_merge_record",
                    f"Unknown merge record: {record_id}",
                )
            records.append(record)
        return tuple(records)

    def _merged_evidence(
        self,
        patch: ToolMemoryPatch,
        records: Sequence[ToolMemoryRecord],
    ) -> tuple[MemoryEvidence, ...]:
        evidence = [
            *patch.evidence,
            *(
                item
                for record in records
                for item in self.repository.tool_memory_record_sources(record.id)
            ),
        ]
        unique = {
            (
                item.message_id,
                item.quote,
                item.start_char,
                item.end_char,
            ): item
            for item in evidence
        }
        return tuple(unique.values())

    def _replay_completed(
        self,
        run_id: str,
        patches: Sequence[ToolMemoryPatch],
    ) -> ToolMemoryPatchBatchResult:
        trace = self.repository.memory_patch_trace(run_id)
        proposals = trace["proposals"]
        if len(proposals) != len(patches):
            raise ValueError("Completed patch run does not match the replayed batch")
        outcomes = []
        rejections = []
        for index, (proposal, patch) in enumerate(zip(proposals, patches, strict=True)):
            if proposal["idempotency_key"] != f"{run_id}:{index}":
                raise ValueError(
                    "Completed patch run has incompatible idempotency keys"
                )
            expected = self._canonical_replay_payload(patch)
            if proposal["patch"] != expected:
                raise ValueError(
                    "Completed patch run does not match the replayed patch"
                )
            if proposal["status"] == "rejected":
                validation = proposal["validation"]
                rejections.append(
                    ToolMemoryPatchRejection(
                        sequence_index=index,
                        proposal_id=str(proposal["id"]),
                        code=str(validation.get("code", "rejected")),
                        message=str(
                            validation.get(
                                "message",
                                proposal.get("rejection_reason", ""),
                            )
                        ),
                        evidence_message_ids=tuple(
                            dict.fromkeys(item.message_id for item in patch.evidence)
                        ),
                    )
                )
                continue
            if proposal["status"] != "applied":
                raise ValueError(
                    "Completed patch run has an unresolved proposal"
                )
            record = self.repository.tool_memory_record(
                str(proposal["result_record_id"])
            )
            if record is None:
                raise RuntimeError("Applied patch result record is missing")
            outcomes.append(
                ToolMemoryPatchOutcome(
                    operation=str(proposal["operation"]),
                    proposal_id=str(proposal["id"]),
                    result_record_id=record.id,
                    record_status=record.status,
                )
            )
        return ToolMemoryPatchBatchResult(
            run_id=run_id,
            outcomes=tuple(outcomes),
            rejections=tuple(rejections),
            replayed=True,
        )

    @staticmethod
    def _canonical_replay_payload(patch: ToolMemoryPatch) -> dict[str, Any]:
        normalized = ToolMemoryPatch(
            operation=patch.operation.strip().upper(),
            confidence=patch.confidence,
            evidence=patch.evidence,
            identity=(
                normalize_tool_memory_identity(patch.identity)
                if patch.identity is not None
                else None
            ),
            value=patch.value,
            memory_type=(
                patch.memory_type.strip()
                if patch.memory_type is not None
                else None
            ),
            target_record_id=patch.target_record_id,
            merge_record_ids=tuple(dict.fromkeys(patch.merge_record_ids)),
            reason=patch.reason.strip(),
        )
        return tool_memory_patch_payload(normalized, include_evidence=False)

    def _record_failed_batch(
        self,
        *,
        run_id: str,
        index: int,
        patch: ToolMemoryPatch | None,
        error: BaseException,
        usage: ModelUsage | None,
    ) -> None:
        code = (
            error.code
            if isinstance(error, ToolMemoryPatchValidationError)
            else type(error).__name__
        )
        message = f"{code}: {error}"
        try:
            self.repository.finish_memory_patch_run(
                run_id,
                status="failed",
                usage=usage,
                error=message,
            )
        except KeyError:
            return
        if patch is None or index < 0:
            return
        try:
            self._record_rejected_patch(
                run_id=run_id,
                index=index,
                patch=patch,
                error=error,
            )
        except Exception:
            # Preserve the original validation/application error if audit storage
            # itself is unavailable.
            return

    def _record_rejected_patch(
        self,
        *,
        run_id: str,
        index: int,
        patch: ToolMemoryPatch,
        error: BaseException,
    ) -> ToolMemoryPatchRejection:
        code = (
            error.code
            if isinstance(error, ToolMemoryPatchValidationError)
            else type(error).__name__
        )
        message = f"{code}: {error}"
        proposed_key = (
            tool_memory_record_key(patch.identity)
            if patch.identity is not None
            else None
        )
        proposal_id = self.repository.record_memory_patch_proposal(
            run_id=run_id,
            operation=patch.operation,
            patch=tool_memory_patch_payload(
                patch,
                include_evidence=False,
            ),
            evidence=[asdict(item) for item in patch.evidence],
            confidence=patch.confidence,
            idempotency_key=f"{run_id}:{index}",
            target_record_id=patch.target_record_id,
            proposed_record_key=proposed_key,
            validation={
                "accepted": False,
                "code": code,
                "message": str(error),
                "policy_version": self.validator.policy.policy_version,
            },
            sequence_index=index,
        )
        self.repository.resolve_memory_patch_proposal(
            proposal_id,
            status="rejected",
            rejection_reason=message,
        )
        return ToolMemoryPatchRejection(
            sequence_index=index,
            proposal_id=proposal_id,
            code=code,
            message=str(error),
            evidence_message_ids=tuple(
                dict.fromkeys(item.message_id for item in patch.evidence)
            ),
        )
