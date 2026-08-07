from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, replace
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from palmclaw_ubuntu.fact_memory_schema import (
    fact_memory_record_key,
    normalize_capability_hints,
    normalize_fact_memory_identity,
)
from palmclaw_ubuntu.models import (
    FactMemoryCandidate,
    FactMemoryIdentity,
    FactMemoryLinkBatchResult,
    FactMemoryLinkOutcome,
    FactMemoryPatch,
    FactMemoryPatchOutcome,
    FactMemoryRecord,
    FactMemorySemanticDecision,
    FactMemorySemanticReviewCase,
    MemoryEvidence,
)
from palmclaw_ubuntu.privacy import redact_data
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tool_memory_schema import (
    entity_alias_compatible,
    normalize_identifier,
)
from palmclaw_ubuntu.validation import (
    FactMemoryCandidateValidator,
    FactMemoryValidation,
    FactMemoryValidationPolicy,
)

_MEMORY_TYPES = frozenset(
    {"constraint", "decision", "fact", "policy", "preference", "state"}
)

FACT_MEMORY_PATCH_SCHEMA: dict[str, Any] = {
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
                "entity_id": {"type": "string", "minLength": 1},
                "predicate": {"type": "string", "minLength": 1},
                "identity_conditions": {"type": "object"},
                "applicability": {"type": "object"},
            },
            "required": ["user_id", "entity_id", "predicate"],
            "additionalProperties": False,
        },
        "value": {},
        "memory_type": {"type": "string", "enum": sorted(_MEMORY_TYPES)},
        "capability_hints": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
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

_PATCH_SCHEMA_VALIDATOR = Draft202012Validator(FACT_MEMORY_PATCH_SCHEMA)


class FactMemoryPatchError(ValueError):
    """Raised when a Fact-first patch is structurally invalid."""


def parse_fact_memory_patch(payload: Mapping[str, Any]) -> FactMemoryPatch:
    try:
        _PATCH_SCHEMA_VALIDATOR.validate(dict(payload))
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        suffix = f" at {location}" if location else ""
        raise FactMemoryPatchError(
            f"Invalid Fact memory patch{suffix}: {exc.message}"
        ) from exc
    identity_payload = payload.get("identity")
    identity = None
    if isinstance(identity_payload, Mapping):
        identity = FactMemoryIdentity(
            user_id=str(identity_payload["user_id"]),
            entity_id=str(identity_payload["entity_id"]),
            predicate=str(identity_payload["predicate"]),
            identity_conditions=dict(
                identity_payload.get("identity_conditions", {})
            ),
            applicability=dict(identity_payload.get("applicability", {})),
        )
    return FactMemoryPatch(
        operation=str(payload["operation"]),
        confidence=float(payload["confidence"]),
        evidence=tuple(
            MemoryEvidence(
                message_id=int(item["message_id"]),
                quote=str(item["quote"]),
                start_char=item.get("start_char"),
                end_char=item.get("end_char"),
            )
            for item in payload["evidence"]
        ),
        identity=identity,
        value=payload.get("value"),
        memory_type=(
            str(payload["memory_type"]) if "memory_type" in payload else None
        ),
        capability_hints=tuple(
            str(item) for item in payload.get("capability_hints", ())
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


def fact_memory_patch_payload(
    patch: FactMemoryPatch,
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
            "entity_id": patch.identity.entity_id,
            "predicate": patch.identity.predicate,
            "identity_conditions": dict(patch.identity.identity_conditions),
            "applicability": dict(patch.identity.applicability),
        }
    if patch.operation != "DELETE":
        payload["value"] = patch.value
    if patch.memory_type is not None:
        payload["memory_type"] = patch.memory_type
    if patch.capability_hints:
        payload["capability_hints"] = list(patch.capability_hints)
    if patch.target_record_id is not None:
        payload["target_record_id"] = patch.target_record_id
    if patch.merge_record_ids:
        payload["merge_record_ids"] = list(patch.merge_record_ids)
    if patch.reason:
        payload["reason"] = patch.reason
    if include_evidence:
        payload["evidence"] = [asdict(item) for item in patch.evidence]
    return payload


class FactMemoryPatchEngine:
    """Apply immutable Fact-first patches independently of Tool routing."""

    def __init__(self, repository: SQLiteRepository):
        self.repository = repository

    def apply(
        self,
        *,
        session_id: str,
        patch: FactMemoryPatch | Mapping[str, Any],
        idempotency_key: str,
    ) -> FactMemoryPatchOutcome:
        parsed = (
            patch
            if isinstance(patch, FactMemoryPatch)
            else parse_fact_memory_patch(patch)
        )
        normalized = self._validate_and_normalize(session_id, parsed)
        payload = redact_data(
            fact_memory_patch_payload(normalized, include_evidence=False)
        )
        evidence_payload = redact_data(
            [asdict(item) for item in normalized.evidence]
        )
        assert isinstance(payload, dict)
        assert isinstance(evidence_payload, list)
        replay = self.repository.fact_memory_patch_event_by_idempotency(
            idempotency_key
        )
        if replay is not None:
            return self._replay(replay, payload, evidence_payload)

        with self.repository.memory_patch_transaction():
            replay = self.repository.fact_memory_patch_event_by_idempotency(
                idempotency_key
            )
            if replay is not None:
                return self._replay(replay, payload, evidence_payload)
            self._validate_operation(session_id, normalized)
            event_id = str(uuid.uuid4())
            proposed_key = self._proposed_key(normalized)
            self.repository.begin_fact_memory_patch_event(
                event_id=event_id,
                session_id=session_id,
                operation=normalized.operation,
                patch=payload,
                evidence=evidence_payload,
                confidence=normalized.confidence,
                idempotency_key=idempotency_key,
                target_record_id=normalized.target_record_id,
                merge_record_ids=normalized.merge_record_ids,
                proposed_record_key=proposed_key,
            )
            result = self._apply_one(
                session_id=session_id,
                patch=normalized,
                event_id=event_id,
            )
            self.repository.resolve_fact_memory_patch_event(
                event_id,
                status="applied",
                result_record_id=result.id,
                reason=normalized.reason or f"patch_{normalized.operation.casefold()}",
            )
        return FactMemoryPatchOutcome(
            operation=normalized.operation,
            event_id=event_id,
            result_record_id=result.id,
            record_status=result.status,
        )

    def _validate_and_normalize(
        self,
        session_id: str,
        patch: FactMemoryPatch,
    ) -> FactMemoryPatch:
        self.repository.require_session(session_id)
        operation = patch.operation.strip().upper()
        if operation not in {"ADD", "UPDATE", "MERGE", "DELETE"}:
            raise FactMemoryPatchError(f"Unsupported operation: {patch.operation}")
        if not 0 <= patch.confidence <= 1:
            raise FactMemoryPatchError("confidence must be between 0 and 1")
        evidence = self._normalize_evidence(session_id, patch.evidence)
        identity = (
            normalize_fact_memory_identity(patch.identity)
            if patch.identity is not None
            else None
        )
        memory_type = (
            patch.memory_type.strip().casefold()
            if patch.memory_type is not None
            else None
        )
        if memory_type is not None and memory_type not in _MEMORY_TYPES:
            raise FactMemoryPatchError(
                f"Unsupported memory_type: {patch.memory_type}"
            )
        normalized = replace(
            patch,
            operation=operation,
            identity=identity,
            memory_type=memory_type,
            evidence=evidence,
            capability_hints=normalize_capability_hints(
                patch.capability_hints
            ),
            merge_record_ids=tuple(dict.fromkeys(patch.merge_record_ids)),
        )
        return normalized

    def _validate_operation(
        self,
        session_id: str,
        patch: FactMemoryPatch,
    ) -> None:
        if patch.operation == "ADD":
            self._require_fact_fields(patch)
            if patch.target_record_id or patch.merge_record_ids:
                raise FactMemoryPatchError("ADD cannot specify target records")
            assert patch.identity is not None
            if self.repository.active_versioned_fact_memory_record(
                fact_memory_record_key(patch.identity)
            ):
                raise FactMemoryPatchError("An active Fact already has this identity")
            return

        target = self._active_target(session_id, patch.target_record_id)
        if patch.operation == "DELETE":
            if patch.identity is not None or patch.merge_record_ids:
                raise FactMemoryPatchError(
                    "DELETE uses its target identity and cannot merge records"
                )
            return

        self._require_fact_fields(patch)
        assert patch.identity is not None
        proposed_key = fact_memory_record_key(patch.identity)
        if patch.operation == "UPDATE":
            if patch.merge_record_ids:
                raise FactMemoryPatchError("UPDATE cannot specify merge records")
            if proposed_key != target.record_key:
                raise FactMemoryPatchError(
                    "UPDATE must preserve the target Fact identity"
                )
            return

        if not patch.merge_record_ids:
            raise FactMemoryPatchError("MERGE requires merge_record_ids")
        record_ids = (target.id, *patch.merge_record_ids)
        if len(set(record_ids)) != len(record_ids):
            raise FactMemoryPatchError("MERGE records must be unique")
        for record_id in patch.merge_record_ids:
            self._active_target(session_id, record_id)
        active = self.repository.active_versioned_fact_memory_record(proposed_key)
        if active is not None and active.id not in record_ids:
            raise FactMemoryPatchError(
                "MERGE output identity already has another active Fact"
            )

    @staticmethod
    def _require_fact_fields(patch: FactMemoryPatch) -> None:
        if patch.identity is None:
            raise FactMemoryPatchError(f"{patch.operation} requires identity")
        if patch.memory_type is None:
            raise FactMemoryPatchError(f"{patch.operation} requires memory_type")

    def _normalize_evidence(
        self,
        session_id: str,
        evidence: Sequence[MemoryEvidence],
    ) -> tuple[MemoryEvidence, ...]:
        if not evidence:
            raise FactMemoryPatchError("At least one evidence quote is required")
        messages = {
            message.id: message
            for message in self.repository.list_messages(session_id)
        }
        normalized = []
        for source in evidence:
            message = messages.get(source.message_id)
            if message is None:
                raise FactMemoryPatchError(
                    f"Evidence message is not in this session: {source.message_id}"
                )
            quote = source.quote.strip()
            if not quote:
                raise FactMemoryPatchError("Evidence quote cannot be empty")
            if source.start_char is None and source.end_char is None:
                start = message.content.find(quote)
                if start < 0:
                    raise FactMemoryPatchError(
                        f"Evidence quote is absent from message {source.message_id}"
                    )
                end = start + len(quote)
            elif source.start_char is None or source.end_char is None:
                raise FactMemoryPatchError(
                    "Evidence start_char and end_char must be supplied together"
                )
            else:
                start = source.start_char
                end = source.end_char
                if (
                    start < 0
                    or end < start
                    or message.content[start:end] != quote
                ):
                    raise FactMemoryPatchError(
                        f"Evidence offsets do not match message {source.message_id}"
                    )
            normalized.append(
                MemoryEvidence(
                    message_id=source.message_id,
                    quote=quote,
                    start_char=start,
                    end_char=end,
                )
            )
        unique = {
            (item.message_id, item.quote, item.start_char, item.end_char): item
            for item in normalized
        }
        return tuple(unique.values())

    def _active_target(
        self,
        session_id: str,
        record_id: str | None,
    ) -> FactMemoryRecord:
        if record_id is None:
            raise FactMemoryPatchError("A target_record_id is required")
        record = self.repository.fact_memory_record(record_id)
        if record is None:
            raise FactMemoryPatchError(f"Unknown target record: {record_id}")
        if record.session_id != session_id:
            raise FactMemoryPatchError("Target record belongs to another session")
        if record.status != "active":
            raise FactMemoryPatchError(f"Target record is not active: {record_id}")
        return record

    def _apply_one(
        self,
        *,
        session_id: str,
        patch: FactMemoryPatch,
        event_id: str,
    ) -> FactMemoryRecord:
        if patch.operation == "ADD":
            return self._create_record(
                session_id=session_id,
                patch=patch,
                event_id=event_id,
            )
        target = self._active_target(session_id, patch.target_record_id)
        if patch.operation == "UPDATE":
            self.repository.transition_fact_memory_record(
                target.id,
                status="superseded",
                reason=patch.reason or "patch_update",
                patch_event_id=event_id,
            )
            return self._create_record(
                session_id=session_id,
                patch=patch,
                event_id=event_id,
                supersedes_id=target.id,
                version=target.version + 1,
                capability_hints=patch.capability_hints or target.capability_hints,
            )
        if patch.operation == "MERGE":
            records = (
                target,
                *(
                    self._active_target(session_id, record_id)
                    for record_id in patch.merge_record_ids
                ),
            )
            result_id = str(uuid.uuid4())
            for record in records:
                self.repository.transition_fact_memory_record(
                    record.id,
                    status="merged",
                    reason=patch.reason or "patch_merge",
                    patch_event_id=event_id,
                )
            result = self._create_record(
                session_id=session_id,
                patch=patch,
                event_id=event_id,
                record_id=result_id,
                supersedes_id=target.id,
                evidence=self._merged_evidence(patch, records),
                capability_hints=tuple(
                    hint
                    for record in records
                    for hint in record.capability_hints
                )
                + patch.capability_hints,
            )
            for record in records:
                self.repository.link_merged_fact_memory_record(
                    record.id,
                    merged_into_id=result.id,
                )
            return result

        identity = FactMemoryIdentity(
            user_id=target.user_id,
            entity_id=target.entity_id,
            predicate=target.predicate,
            identity_conditions=target.identity_conditions,
            applicability=target.applicability,
        )
        self.repository.transition_fact_memory_record(
            target.id,
            status="superseded",
            reason=patch.reason or "patch_delete",
            patch_event_id=event_id,
        )
        return self.repository.insert_fact_memory_record(
            session_id=session_id,
            identity=identity,
            value=target.value,
            memory_type=target.memory_type,
            confidence=patch.confidence,
            version=target.version + 1,
            capability_hints=target.capability_hints,
            status="deleted",
            supersedes_id=target.id,
            evidence=patch.evidence,
            patch_event_id=event_id,
            status_reason="patch_delete_tombstone",
        )

    def _create_record(
        self,
        *,
        session_id: str,
        patch: FactMemoryPatch,
        event_id: str,
        record_id: str | None = None,
        supersedes_id: str | None = None,
        version: int | None = None,
        evidence: Sequence[MemoryEvidence] | None = None,
        capability_hints: Sequence[str] | None = None,
    ) -> FactMemoryRecord:
        assert patch.identity is not None
        assert patch.memory_type is not None
        record_key = fact_memory_record_key(patch.identity)
        return self.repository.insert_fact_memory_record(
            session_id=session_id,
            identity=patch.identity,
            value=patch.value,
            memory_type=patch.memory_type,
            confidence=patch.confidence,
            version=version
            or self.repository.next_fact_memory_version(record_key),
            capability_hints=capability_hints or patch.capability_hints,
            supersedes_id=supersedes_id,
            evidence=evidence or patch.evidence,
            record_id=record_id,
            patch_event_id=event_id,
            status_reason=f"patch_{patch.operation.casefold()}",
        )

    def _merged_evidence(
        self,
        patch: FactMemoryPatch,
        records: Sequence[FactMemoryRecord],
    ) -> tuple[MemoryEvidence, ...]:
        evidence = [
            *patch.evidence,
            *(
                source
                for record in records
                for source in self.repository.fact_memory_record_sources(record.id)
            ),
        ]
        unique = {
            (item.message_id, item.quote, item.start_char, item.end_char): item
            for item in evidence
        }
        return tuple(unique.values())

    def _proposed_key(self, patch: FactMemoryPatch) -> str | None:
        if patch.identity is not None:
            return fact_memory_record_key(patch.identity)
        if patch.target_record_id is None:
            return None
        target = self.repository.fact_memory_record(patch.target_record_id)
        return target.record_key if target is not None else None

    def _replay(
        self,
        event: Mapping[str, Any],
        patch: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
    ) -> FactMemoryPatchOutcome:
        if event["patch"] != patch or event["evidence"] != list(evidence):
            raise FactMemoryPatchError(
                "Idempotency key was already used for a different Fact patch"
            )
        if event["status"] != "applied" or not event["result_record_id"]:
            raise FactMemoryPatchError(
                f"Fact patch event cannot be replayed from {event['status']}"
            )
        record = self.repository.fact_memory_record(
            str(event["result_record_id"])
        )
        if record is None:
            raise RuntimeError("Applied Fact patch result record is missing")
        return FactMemoryPatchOutcome(
            operation=str(event["operation"]),
            event_id=str(event["id"]),
            result_record_id=record.id,
            record_status=record.status,
            replayed=True,
        )


def fact_memory_candidate_payload(
    candidate: FactMemoryCandidate,
    *,
    include_evidence: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "entity_id": candidate.entity_id,
        "predicate": candidate.predicate,
        "value": candidate.value,
        "identity_conditions": dict(candidate.identity_conditions),
        "applicability": dict(candidate.applicability),
        "capability_hints": list(candidate.capability_hints),
        "memory_type": candidate.memory_type,
        "confidence": candidate.confidence,
        "directive": candidate.directive,
    }
    if candidate.reason:
        payload["reason"] = candidate.reason
    if include_evidence:
        payload["evidence"] = [asdict(item) for item in candidate.evidence]
    return payload


def select_relevant_fact_records(
    messages: Sequence[Any],
    active_records: Sequence[FactMemoryRecord],
    *,
    limit: int = 24,
    recent_fallback: int = 4,
) -> tuple[FactMemoryRecord, ...]:
    """Select compact linking context without Tool schemas or hard filtering."""
    if limit < 1:
        raise ValueError("Fact linking context limit must be positive")
    text = " ".join(
        str(getattr(message, "content", ""))
        for message in messages
        if getattr(message, "content", "")
    )
    query_tokens = set(normalize_identifier(text).split("_"))
    scored: list[tuple[float, int, FactMemoryRecord]] = []
    for index, record in enumerate(active_records):
        record_tokens = set(
            normalize_identifier(
                " ".join(
                    (
                        record.entity_id,
                        record.predicate,
                        *record.capability_hints,
                    )
                )
            ).split("_")
        )
        overlap = len(query_tokens & record_tokens)
        score = overlap / max(1, len(record_tokens))
        if score > 0:
            scored.append((score, -index, record))
    selected = [
        record
        for _, _, record in sorted(scored, key=lambda item: (-item[0], item[1]))
    ]
    selected_ids = {record.id for record in selected}
    for record in active_records[: max(0, recent_fallback)]:
        if record.id not in selected_ids:
            selected.append(record)
            selected_ids.add(record.id)
    return tuple(selected[:limit])


class FactMemoryLinker:
    """Link high-recall Fact candidates to immutable R1 Patch operations."""

    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        user_id: str,
        alias_match_threshold: float = 0.62,
        require_exact_predicate: bool = False,
        exact_predicates: Sequence[str] = (),
        validation_policy: FactMemoryValidationPolicy | None = None,
    ):
        if not user_id.strip():
            raise ValueError("Fact linker user_id cannot be empty")
        if not 0 <= alias_match_threshold <= 1:
            raise ValueError("alias_match_threshold must be between 0 and 1")
        self.repository = repository
        self.user_id = normalize_identifier(user_id)
        self.alias_match_threshold = alias_match_threshold
        self.require_exact_predicate = require_exact_predicate
        self.exact_predicates = frozenset(
            normalize_identifier(predicate)
            for predicate in exact_predicates
            if predicate.strip()
        )
        self.patch_engine = FactMemoryPatchEngine(repository)
        self.validator = FactMemoryCandidateValidator(validation_policy)

    def apply_candidates(
        self,
        *,
        run_id: str,
        session_id: str,
        candidates: Sequence[FactMemoryCandidate],
        allowed_evidence_message_ids: Sequence[int] = (),
        semantic_decisions: Mapping[int, FactMemorySemanticDecision] | None = None,
    ) -> FactMemoryLinkBatchResult:
        existing = {
            int(event["sequence_index"]): event
            for event in self.repository.fact_memory_candidate_events(run_id)
        }
        outcomes = []
        messages = {
            message.id: (
                message.session_id,
                message.role,
                message.content,
            )
            for message in self.repository.list_messages(session_id)
        }
        ordered = self._ordered_candidates(candidates)
        resolved_semantic = dict(semantic_decisions or {})
        for sequence_index, candidate in enumerate(ordered):
            prior = existing.get(sequence_index)
            if prior is not None:
                self._assert_replay_candidate(
                    prior,
                    candidate,
                    session_id=session_id,
                    messages=messages,
                    allowed_evidence_message_ids=allowed_evidence_message_ids,
                )
                outcomes.append(self._replay_link_outcome(prior))
                continue
            outcomes.append(
                self._link_one(
                    run_id=run_id,
                    session_id=session_id,
                    sequence_index=sequence_index,
                    candidate=candidate,
                    messages=messages,
                    allowed_evidence_message_ids=allowed_evidence_message_ids,
                    semantic_decision=resolved_semantic.get(sequence_index),
                )
            )
        return FactMemoryLinkBatchResult(
            run_id=run_id,
            outcomes=tuple(outcomes),
        )

    def semantic_review_cases(
        self,
        *,
        session_id: str,
        candidates: Sequence[FactMemoryCandidate],
        allowed_evidence_message_ids: Sequence[int] = (),
    ) -> tuple[FactMemorySemanticReviewCase, ...]:
        messages = {
            message.id: (
                message.session_id,
                message.role,
                message.content,
            )
            for message in self.repository.list_messages(session_id)
        }
        cases = []
        for sequence_index, candidate in enumerate(
            self._ordered_candidates(candidates)
        ):
            validation = self.validator.validate(
                candidate,
                session_id=session_id,
                messages=messages,
                allowed_evidence_message_ids=allowed_evidence_message_ids,
            )
            if validation.disposition == "rejected":
                continue
            if validation.disposition == "review":
                review_candidate = validation.candidate
                proposed_operation = (
                    "DELETE"
                    if validation.candidate.directive == "DELETE"
                    else "UPSERT"
                )
                related_record_ids: tuple[str, ...] = ()
                try:
                    review_candidate = self._normalize_candidate(
                        validation.candidate
                    )
                    review_decision = self._decide(
                        session_id,
                        review_candidate,
                    )
                    proposed_operation = str(review_decision["operation"])
                    target = review_decision.get("target")
                    related_record_ids = tuple(
                        dict.fromkeys(
                            (
                                *(
                                    (target.id,)
                                    if isinstance(target, FactMemoryRecord)
                                    else ()
                                ),
                                *(
                                    record.id
                                    for record in review_decision.get(
                                        "merge_records",
                                        (),
                                    )
                                ),
                            )
                        )
                    )
                except FactMemoryPatchError:
                    pass
                cases.append(
                    FactMemorySemanticReviewCase(
                        sequence_index=sequence_index,
                        candidate=review_candidate,
                        review_codes=(validation.code,),
                        proposed_operation=proposed_operation,
                        related_record_ids=related_record_ids,
                    )
                )
                continue
            try:
                normalized = self._normalize_candidate(validation.candidate)
                decision = self._decide(session_id, normalized)
            except FactMemoryPatchError:
                continue
            operation = str(decision["operation"])
            transition = self.validator.validate_transition(
                replace(validation, candidate=normalized),
                operation=operation,
            )
            if transition.disposition == "review":
                target = decision.get("target")
                cases.append(
                    FactMemorySemanticReviewCase(
                        sequence_index=sequence_index,
                        candidate=normalized,
                        review_codes=(transition.code,),
                        proposed_operation=operation,
                        related_record_ids=tuple(
                            dict.fromkeys(
                                (
                                    *(
                                        (target.id,)
                                        if isinstance(
                                            target,
                                            FactMemoryRecord,
                                        )
                                        else ()
                                    ),
                                    *(
                                        record.id
                                        for record in decision.get(
                                            "merge_records",
                                            (),
                                        )
                                    ),
                                )
                            )
                        ),
                    )
                )
        return tuple(cases)

    def _link_one(
        self,
        *,
        run_id: str,
        session_id: str,
        sequence_index: int,
        candidate: FactMemoryCandidate,
        messages: Mapping[int, tuple[str, str, str]],
        allowed_evidence_message_ids: Sequence[int],
        semantic_decision: FactMemorySemanticDecision | None,
    ) -> FactMemoryLinkOutcome:
        event_id = str(uuid.uuid4())
        idempotency_key = f"fact-candidate:{run_id}:{sequence_index}"
        validation: FactMemoryValidation | None = None
        try:
            validation = self.validator.validate(
                candidate,
                session_id=session_id,
                messages=messages,
                allowed_evidence_message_ids=allowed_evidence_message_ids,
            )
            if (
                semantic_decision is not None
                and validation.disposition != "rejected"
            ):
                validation = self.validator.apply_semantic_decision(
                    validation,
                    semantic_decision,
                )
            if not validation.accepted:
                return self._validation_outcome(
                    event_id=event_id,
                    run_id=run_id,
                    sequence_index=sequence_index,
                    validation=validation,
                    idempotency_key=idempotency_key,
                )
            normalized = self._normalize_candidate(validation.candidate)
            decision = self._decide(session_id, normalized)
            validation = self.validator.validate_transition(
                replace(validation, candidate=normalized),
                operation=str(decision["operation"]),
            )
            if (
                semantic_decision is not None
                and not validation.accepted
                and validation.disposition != "rejected"
            ):
                validation = self.validator.apply_semantic_decision(
                    validation,
                    semantic_decision,
                )
            if not validation.accepted:
                return self._validation_outcome(
                    event_id=event_id,
                    run_id=run_id,
                    sequence_index=sequence_index,
                    validation=validation,
                    idempotency_key=idempotency_key,
                    decision=decision,
                )
            if decision["operation"] in {"NOOP", "REVIEW"}:
                status = (
                    "noop"
                    if decision["operation"] == "NOOP"
                    else "review"
                )
                self._record_candidate_event(
                    event_id=event_id,
                    run_id=run_id,
                    sequence_index=sequence_index,
                    candidate=normalized,
                    decision=decision,
                    status=status,
                    idempotency_key=idempotency_key,
                    validation=(
                        self._link_review_validation(validation, decision)
                        if decision["operation"] == "REVIEW"
                        else validation
                    ),
                )
                return FactMemoryLinkOutcome(
                    sequence_index=sequence_index,
                    decision=str(decision["operation"]),
                    status=status,
                    candidate_event_id=event_id,
                    reason=str(decision["reason"]),
                )
            patch = self._patch_for_decision(normalized, decision)
            patch_outcome = self.patch_engine.apply(
                session_id=session_id,
                patch=patch,
                idempotency_key=f"fact-patch:{run_id}:{sequence_index}",
            )
            self._record_candidate_event(
                event_id=event_id,
                run_id=run_id,
                sequence_index=sequence_index,
                candidate=normalized,
                decision=decision,
                status="applied",
                result_record_id=patch_outcome.result_record_id,
                idempotency_key=idempotency_key,
                validation=validation,
            )
            return FactMemoryLinkOutcome(
                sequence_index=sequence_index,
                decision=str(decision["operation"]),
                status="applied",
                candidate_event_id=event_id,
                result_record_id=patch_outcome.result_record_id,
                reason=str(decision["reason"]),
            )
        except FactMemoryPatchError as exc:
            decision = {
                "operation": "REJECT",
                "target": None,
                "merge_records": (),
                "score": None,
                "reason": f"link_structure: {exc}",
            }
            failed_validation = (
                replace(
                    validation,
                    disposition="rejected",
                    code="link_structure",
                    message=str(exc),
                    evidence_relation="invalid",
                )
                if validation is not None
                else None
            )
            self._record_candidate_event(
                event_id=event_id,
                run_id=run_id,
                sequence_index=sequence_index,
                candidate=candidate,
                decision=decision,
                status="rejected",
                idempotency_key=idempotency_key,
                validation=failed_validation,
            )
            return FactMemoryLinkOutcome(
                sequence_index=sequence_index,
                decision="REJECT",
                status="rejected",
                candidate_event_id=event_id,
                reason=str(decision["reason"]),
            )

    @staticmethod
    def _ordered_candidates(
        candidates: Sequence[FactMemoryCandidate],
    ) -> tuple[FactMemoryCandidate, ...]:
        ordered = sorted(
            enumerate(candidates),
            key=lambda item: (
                min(
                    (source.message_id for source in item[1].evidence),
                    default=2**63 - 1,
                ),
                item[0],
            ),
        )
        return tuple(candidate for _, candidate in ordered)

    def _normalize_candidate(
        self,
        candidate: FactMemoryCandidate,
    ) -> FactMemoryCandidate:
        directive = candidate.directive.strip().upper()
        if directive not in {"UPSERT", "DELETE"}:
            raise FactMemoryPatchError(
                f"Unsupported Fact candidate directive: {candidate.directive}"
            )
        try:
            identity = normalize_fact_memory_identity(
                FactMemoryIdentity(
                    user_id=self.user_id,
                    entity_id=candidate.entity_id,
                    predicate=candidate.predicate,
                    identity_conditions=candidate.identity_conditions,
                    applicability=candidate.applicability,
                )
            )
        except ValueError as exc:
            raise FactMemoryPatchError(str(exc)) from exc
        memory_type = candidate.memory_type.strip().casefold()
        if memory_type not in _MEMORY_TYPES:
            raise FactMemoryPatchError(
                f"Unsupported memory_type: {candidate.memory_type}"
            )
        if not 0 <= candidate.confidence <= 1:
            raise FactMemoryPatchError(
                "Fact candidate confidence must be between 0 and 1"
            )
        if not candidate.evidence:
            raise FactMemoryPatchError("Fact candidate requires evidence")
        return replace(
            candidate,
            entity_id=identity.entity_id,
            predicate=identity.predicate,
            identity_conditions=identity.identity_conditions,
            applicability=identity.applicability,
            capability_hints=normalize_capability_hints(
                candidate.capability_hints
            ),
            memory_type=memory_type,
            directive=directive,
        )

    def _decide(
        self,
        session_id: str,
        candidate: FactMemoryCandidate,
    ) -> dict[str, Any]:
        identity = self._candidate_identity(candidate)
        exact = self.repository.active_versioned_fact_memory_record(
            fact_memory_record_key(identity)
        )
        active = self.repository.list_fact_memory_records(
            session_id,
            user_id=self.user_id,
        )
        compatible = [
            (score, record)
            for record in active
            if (score := self._match_score(identity, record))
            >= self.alias_match_threshold
        ]
        compatible.sort(key=lambda item: (-item[0], -item[1].version))

        if candidate.directive == "DELETE":
            targets = (
                [exact]
                if exact is not None
                else [record for _, record in compatible]
            )
            targets = list({record.id: record for record in targets}.values())
            if not targets:
                return self._decision(
                    "NOOP",
                    reason="delete_target_not_found",
                )
            if len(targets) > 1:
                return self._decision(
                    "REVIEW",
                    reason="ambiguous_delete_target",
                    score=compatible[0][0] if compatible else 1.0,
                )
            return self._decision(
                "DELETE",
                target=targets[0],
                reason="explicit_delete_target",
                score=1.0 if exact is not None else compatible[0][0],
            )

        equivalent = [
            record
            for _, record in compatible
            if self._values_equal(record.value, candidate.value)
        ]
        if exact is not None:
            merge_records = [
                record
                for record in equivalent
                if record.id != exact.id
            ]
            if merge_records:
                return self._decision(
                    "MERGE",
                    target=exact,
                    merge_records=tuple(merge_records),
                    reason="equivalent_alias_records",
                    score=1.0,
                )
            if self._values_equal(exact.value, candidate.value):
                return self._decision(
                    "NOOP",
                    target=exact,
                    reason="equivalent_active_fact",
                    score=1.0,
                )
            return self._decision(
                "UPDATE",
                target=exact,
                reason="same_identity_new_value",
                score=1.0,
            )

        if not compatible:
            return self._decision("ADD", reason="new_fact_identity")
        top_score = compatible[0][0]
        top = [
            record
            for score, record in compatible
            if top_score - score < 0.05
        ]
        if len(top) > 1:
            if all(
                self._values_equal(record.value, candidate.value)
                for record in top
            ):
                target = self._canonical_record(top)
                return self._decision(
                    "MERGE",
                    target=target,
                    merge_records=tuple(
                        record for record in top if record.id != target.id
                    ),
                    reason="equivalent_alias_records",
                    score=top_score,
                )
            return self._decision(
                "REVIEW",
                reason="ambiguous_alias_match",
                score=top_score,
            )
        target = top[0]
        if self._values_equal(target.value, candidate.value):
            return self._decision(
                "NOOP",
                target=target,
                reason="equivalent_alias_fact",
                score=top_score,
            )
        return self._decision(
            "UPDATE",
            target=target,
            reason="alias_identity_new_value",
            score=top_score,
        )

    def _patch_for_decision(
        self,
        candidate: FactMemoryCandidate,
        decision: Mapping[str, Any],
    ) -> FactMemoryPatch:
        operation = str(decision["operation"])
        target = decision.get("target")
        merge_records = tuple(decision.get("merge_records", ()))
        identity = self._candidate_identity(candidate)
        if isinstance(target, FactMemoryRecord) and operation in {
            "UPDATE",
            "MERGE",
        }:
            identity = self._record_identity(target)
        capability_hints = candidate.capability_hints
        if isinstance(target, FactMemoryRecord):
            capability_hints = normalize_capability_hints(
                (*target.capability_hints, *capability_hints)
            )
        return FactMemoryPatch(
            operation=operation,
            identity=None if operation == "DELETE" else identity,
            value=candidate.value,
            memory_type=(
                None if operation == "DELETE" else candidate.memory_type
            ),
            capability_hints=capability_hints,
            confidence=candidate.confidence,
            evidence=candidate.evidence,
            target_record_id=(
                target.id if isinstance(target, FactMemoryRecord) else None
            ),
            merge_record_ids=tuple(record.id for record in merge_records),
            reason=candidate.reason or str(decision["reason"]),
        )

    def _record_candidate_event(
        self,
        *,
        event_id: str,
        run_id: str,
        sequence_index: int,
        candidate: FactMemoryCandidate,
        decision: Mapping[str, Any],
        status: str,
        idempotency_key: str,
        result_record_id: str | None = None,
        validation: FactMemoryValidation | None = None,
    ) -> None:
        target = decision.get("target")
        merge_records = tuple(decision.get("merge_records", ()))
        self.repository.record_fact_memory_candidate_event(
            event_id=event_id,
            run_id=run_id,
            sequence_index=sequence_index,
            candidate=fact_memory_candidate_payload(
                candidate,
                include_evidence=False,
            ),
            evidence=[asdict(item) for item in candidate.evidence],
            decision=str(decision["operation"]),
            status=status,
            idempotency_key=idempotency_key,
            target_record_id=(
                target.id if isinstance(target, FactMemoryRecord) else None
            ),
            merge_record_ids=tuple(record.id for record in merge_records),
            result_record_id=result_record_id,
            match_score=decision.get("score"),
            reason=str(decision["reason"]),
            validation=validation.as_dict() if validation is not None else {},
        )

    def _validation_outcome(
        self,
        *,
        event_id: str,
        run_id: str,
        sequence_index: int,
        validation: FactMemoryValidation,
        idempotency_key: str,
        decision: Mapping[str, Any] | None = None,
    ) -> FactMemoryLinkOutcome:
        rejected = validation.disposition == "rejected"
        operation = "REJECT" if rejected else "REVIEW"
        status = "rejected" if rejected else "review"
        source_decision = dict(decision or {})
        event_decision = {
            "operation": operation,
            "target": source_decision.get("target"),
            "merge_records": source_decision.get("merge_records", ()),
            "score": source_decision.get("score"),
            "reason": f"{validation.code}: {validation.message}",
        }
        self._record_candidate_event(
            event_id=event_id,
            run_id=run_id,
            sequence_index=sequence_index,
            candidate=validation.candidate,
            decision=event_decision,
            status=status,
            idempotency_key=idempotency_key,
            validation=validation,
        )
        return FactMemoryLinkOutcome(
            sequence_index=sequence_index,
            decision=operation,
            status=status,
            candidate_event_id=event_id,
            reason=str(event_decision["reason"]),
        )

    @staticmethod
    def _link_review_validation(
        validation: FactMemoryValidation,
        decision: Mapping[str, Any],
    ) -> FactMemoryValidation:
        return replace(
            validation,
            disposition="review",
            code="ambiguous_link",
            message=str(decision["reason"]),
            evidence_relation="uncertain",
        )

    def _match_score(
        self,
        identity: FactMemoryIdentity,
        record: FactMemoryRecord,
    ) -> float:
        if identity.user_id != record.user_id:
            return 0.0
        if (
            self.require_exact_predicate
            and identity.predicate != record.predicate
        ):
            return 0.0
        if (
            identity.predicate != record.predicate
            and self.exact_predicates
            and (
                identity.predicate in self.exact_predicates
                or record.predicate in self.exact_predicates
            )
        ):
            return 0.0
        if (
            dict(identity.identity_conditions) != dict(record.identity_conditions)
            or dict(identity.applicability) != dict(record.applicability)
        ):
            return 0.0
        entity_score = self._identifier_similarity(
            identity.entity_id,
            record.entity_id,
            aliases=True,
        )
        predicate_score = self._identifier_similarity(
            identity.predicate,
            record.predicate,
            aliases=False,
        )
        return 0.5 * entity_score + 0.5 * predicate_score

    @staticmethod
    def _identifier_similarity(
        first: str,
        second: str,
        *,
        aliases: bool,
    ) -> float:
        if first == second:
            return 1.0
        if aliases and entity_alias_compatible(first, second):
            return 0.95
        first_tokens = set(normalize_identifier(first).split("_"))
        second_tokens = set(normalize_identifier(second).split("_"))
        if not first_tokens or not second_tokens:
            return 0.0
        return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)

    @staticmethod
    def _values_equal(first: Any, second: Any) -> bool:
        return json.dumps(
            first,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) == json.dumps(
            second,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _candidate_identity(
        self,
        candidate: FactMemoryCandidate,
    ) -> FactMemoryIdentity:
        return FactMemoryIdentity(
            user_id=self.user_id,
            entity_id=candidate.entity_id,
            predicate=candidate.predicate,
            identity_conditions=candidate.identity_conditions,
            applicability=candidate.applicability,
        )

    def _record_identity(self, record: FactMemoryRecord) -> FactMemoryIdentity:
        return FactMemoryIdentity(
            user_id=record.user_id,
            entity_id=record.entity_id,
            predicate=record.predicate,
            identity_conditions=record.identity_conditions,
            applicability=record.applicability,
        )

    @staticmethod
    def _canonical_record(
        records: Sequence[FactMemoryRecord],
    ) -> FactMemoryRecord:
        return max(
            records,
            key=lambda record: (
                len(record.entity_id.split("_")),
                len(record.entity_id),
                record.version,
            ),
        )

    @staticmethod
    def _decision(
        operation: str,
        *,
        target: FactMemoryRecord | None = None,
        merge_records: Sequence[FactMemoryRecord] = (),
        reason: str,
        score: float | None = None,
    ) -> dict[str, Any]:
        return {
            "operation": operation,
            "target": target,
            "merge_records": tuple(merge_records),
            "reason": reason,
            "score": score,
        }

    @staticmethod
    def _replay_link_outcome(event: Mapping[str, Any]) -> FactMemoryLinkOutcome:
        return FactMemoryLinkOutcome(
            sequence_index=int(event["sequence_index"]),
            decision=str(event["decision"]),
            status=str(event["status"]),
            candidate_event_id=str(event["id"]),
            result_record_id=event["result_record_id"],
            reason=str(event.get("reason") or ""),
        )

    def _assert_replay_candidate(
        self,
        event: Mapping[str, Any],
        candidate: FactMemoryCandidate,
        *,
        session_id: str,
        messages: Mapping[int, tuple[str, str, str]],
        allowed_evidence_message_ids: Sequence[int],
    ) -> None:
        validation = self.validator.validate(
            candidate,
            session_id=session_id,
            messages=messages,
            allowed_evidence_message_ids=allowed_evidence_message_ids,
        )
        candidate = validation.candidate
        with suppress(FactMemoryPatchError):
            candidate = self._normalize_candidate(candidate)
        payload = redact_data(
            fact_memory_candidate_payload(
                candidate,
                include_evidence=False,
            )
        )
        evidence = redact_data([asdict(item) for item in candidate.evidence])
        if event["candidate"] != payload or event["evidence"] != evidence:
            raise FactMemoryPatchError(
                "Fact candidate sequence was replayed with different content"
            )
