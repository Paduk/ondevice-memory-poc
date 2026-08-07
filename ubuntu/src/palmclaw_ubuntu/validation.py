from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from palmclaw_ubuntu.models import (
    FactMemoryCandidate,
    FactMemorySemanticDecision,
    MemoryCandidate,
    MemoryEvidence,
    MemoryGateDecision,
    MemoryRecord,
    ToolMemoryPatch,
    ToolMemoryRecord,
)
from palmclaw_ubuntu.normalization import (
    contains_delete_intent,
    contains_update_intent,
    normalize_memory_value,
    normalized_value_support,
    strip_runtime_identity_fields,
)
from palmclaw_ubuntu.privacy import detect_sensitive_spans
from palmclaw_ubuntu.tool_memory_schema import (
    normalize_identifier,
    normalize_tool_memory_identity,
    tool_memory_record_key,
)

_TOKEN_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)
_NEGATION_MARKERS = {
    "아니",
    "않",
    "안 ",
    "never",
    "no ",
    "not ",
}
_EXPLICIT_UPDATE_MARKERS = {
    "바꿨",
    "변경",
    "이제",
    "대신",
    "더 이상",
    "업데이트",
    "changed",
    "change my",
    "instead",
    "no longer",
    "now ",
    "update",
}
_EXPLICIT_DELETE_MARKERS = {
    "forget",
    "remove",
    "delete",
    "revoke",
    "do not remember",
    "don't remember",
    "no longer",
    "잊어",
    "삭제",
    "제거",
    "철회",
    "더 이상",
}
_TOOL_MEMORY_TYPES = {
    "constraint",
    "decision",
    "fact",
    "policy",
    "preference",
    "state",
}


@dataclass(frozen=True)
class MemoryGatePolicy:
    enabled: bool = True
    minimum_confidence: float = 0.65
    global_minimum_confidence: float = 0.85
    review_high_sensitivity: bool = True
    pii_allowlist: tuple[str, ...] = ()
    policy_version: str = "memory-gate-v1"


class MemoryValidationGate:
    def __init__(self, policy: MemoryGatePolicy):
        self.policy = policy

    def evaluate(
        self,
        candidate: MemoryCandidate,
        *,
        structural_error: str | None,
        active_memory: MemoryRecord | None,
    ) -> MemoryGateDecision:
        evidence_relation = self._evidence_relation(candidate)
        conflict_relation = self._conflict_relation(candidate, active_memory)
        categories = self._pii_categories(candidate)

        if structural_error:
            return self._decision(
                "reject",
                evidence_relation,
                conflict_relation,
                structural_error,
                categories,
            )
        if not self.policy.enabled:
            decision = "supersede" if conflict_relation == "contradiction" else "accept"
            return self._decision(
                decision,
                evidence_relation,
                conflict_relation,
                "gate_disabled",
                categories,
            )
        if evidence_relation != "entailment":
            return self._decision(
                "reject",
                evidence_relation,
                conflict_relation,
                f"evidence_{evidence_relation}",
                categories,
            )
        if candidate.confidence < self.policy.minimum_confidence:
            return self._decision(
                "reject",
                evidence_relation,
                conflict_relation,
                "confidence_below_minimum",
                categories,
            )
        if "secret" in categories:
            return self._decision(
                "reject",
                evidence_relation,
                conflict_relation,
                "secret_candidate",
                categories,
            )
        if categories:
            return self._decision(
                "review",
                evidence_relation,
                conflict_relation,
                "pii_candidate",
                categories,
            )
        if (
            candidate.scope == "global"
            and candidate.confidence < self.policy.global_minimum_confidence
        ):
            return self._decision(
                "review",
                evidence_relation,
                conflict_relation,
                "global_confidence_below_minimum",
                categories,
            )
        if self.policy.review_high_sensitivity and candidate.sensitivity == "high":
            return self._decision(
                "review",
                evidence_relation,
                conflict_relation,
                "high_sensitivity",
                categories,
            )
        if conflict_relation == "contradiction":
            if self._has_explicit_update(candidate):
                return self._decision(
                    "supersede",
                    evidence_relation,
                    conflict_relation,
                    "explicit_update",
                    categories,
                )
            return self._decision(
                "review",
                evidence_relation,
                conflict_relation,
                "unresolved_conflict",
                categories,
            )
        return self._decision(
            "accept",
            evidence_relation,
            conflict_relation,
            (
                "duplicate_candidate"
                if conflict_relation == "entailment"
                else "supported_candidate"
            ),
            categories,
        )

    def _decision(
        self,
        decision: str,
        evidence_relation: str,
        conflict_relation: str,
        reason: str,
        categories: tuple[str, ...],
    ) -> MemoryGateDecision:
        return MemoryGateDecision(
            decision=decision,
            evidence_relation=evidence_relation,
            conflict_relation=conflict_relation,
            reason=reason,
            policy_version=self.policy.policy_version,
            pii_categories=categories,
        )

    @staticmethod
    def _evidence_relation(candidate: MemoryCandidate) -> str:
        evidence = " ".join(item.quote for item in candidate.evidence).casefold()
        value = candidate.value.strip().casefold()
        if not evidence or not value:
            return "unknown"
        if value in evidence:
            value_position = evidence.find(value)
            prefix = evidence[max(0, value_position - 24) : value_position]
            if any(marker in prefix for marker in _NEGATION_MARKERS):
                return "contradiction"
            return "entailment"
        value_tokens = {
            token for token in _TOKEN_PATTERN.findall(value) if len(token) > 1
        }
        evidence_tokens = set(_TOKEN_PATTERN.findall(evidence))
        if value_tokens and value_tokens.issubset(evidence_tokens):
            return "entailment"
        return "unknown"

    @staticmethod
    def _conflict_relation(
        candidate: MemoryCandidate,
        active_memory: MemoryRecord | None,
    ) -> str:
        if active_memory is None:
            return "none"
        if candidate.value.strip().casefold() == active_memory.value.strip().casefold():
            return "entailment"
        return "contradiction"

    @staticmethod
    def _has_explicit_update(candidate: MemoryCandidate) -> bool:
        evidence = " ".join(item.quote for item in candidate.evidence).casefold()
        return any(marker in evidence for marker in _EXPLICIT_UPDATE_MARKERS)

    def _pii_categories(self, candidate: MemoryCandidate) -> tuple[str, ...]:
        text = "\n".join(
            (
                candidate.subject,
                candidate.predicate,
                candidate.value,
                *(item.quote for item in candidate.evidence),
            )
        )
        return tuple(
            sorted(
                {
                    span.category
                    for span in detect_sensitive_spans(
                        text,
                        allowlist=self.policy.pii_allowlist,
                    )
                }
            )
        )


class ToolMemoryPatchValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolMemoryPatchPolicy:
    minimum_confidence: float = 0.65
    global_minimum_confidence: float = 0.85
    reject_pii: bool = True
    pii_allowlist: tuple[str, ...] = ()
    expected_user_id: str | None = None
    allowed_domain_topics: tuple[tuple[str, str], ...] = ()
    policy_version: str = "tool-memory-patch-gate-v1"


@dataclass(frozen=True)
class ToolMemoryPatchValidation:
    patch: ToolMemoryPatch
    evidence_relation: str
    conflict_relation: str
    pii_categories: tuple[str, ...]
    policy_version: str
    evidence_span_corrections: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": True,
            "evidence_relation": self.evidence_relation,
            "conflict_relation": self.conflict_relation,
            "pii_categories": list(self.pii_categories),
            "policy_version": self.policy_version,
            "evidence_span_corrections": [
                dict(item) for item in self.evidence_span_corrections
            ],
        }


class ToolMemoryPatchValidator:
    def __init__(self, policy: ToolMemoryPatchPolicy | None = None):
        self.policy = policy or ToolMemoryPatchPolicy()

    def validate(
        self,
        patch: ToolMemoryPatch,
        *,
        session_id: str,
        source_turn_id: str | None,
        messages: Mapping[int, tuple[str, str | None, str]],
        target: ToolMemoryRecord | None,
        merge_records: Sequence[ToolMemoryRecord],
        active_for_identity: ToolMemoryRecord | None,
    ) -> ToolMemoryPatchValidation:
        operation = patch.operation.strip().upper()
        if operation not in {"ADD", "UPDATE", "MERGE", "DELETE"}:
            self._fail("invalid_operation", f"Unsupported operation: {operation}")
        if not 0 <= patch.confidence <= 1:
            self._fail("invalid_confidence", "confidence must be between 0 and 1")
        if patch.confidence < self.policy.minimum_confidence:
            self._fail(
                "confidence_below_minimum",
                "patch confidence is below the policy minimum",
            )
        identity = (
            normalize_tool_memory_identity(patch.identity)
            if patch.identity is not None
            else None
        )
        if (
            identity is not None
            and identity.scope == "global"
            and patch.confidence < self.policy.global_minimum_confidence
        ):
            self._fail(
                "global_confidence_below_minimum",
                "global memory confidence is below the policy minimum",
            )
        memory_type = patch.memory_type.strip() if patch.memory_type else None
        evidence, evidence_span_corrections = self._validate_evidence(
            patch.evidence,
            session_id=session_id,
            source_turn_id=source_turn_id,
            messages=messages,
        )
        normalized = replace(
            patch,
            operation=operation,
            identity=identity,
            memory_type=memory_type,
            evidence=evidence,
            merge_record_ids=tuple(dict.fromkeys(patch.merge_record_ids)),
            reason=patch.reason.strip(),
        )
        self._validate_partition(normalized, session_id=session_id)
        self._validate_shape(normalized)
        categories = self._pii_categories(normalized)
        if "secret" in categories:
            self._fail("secret_candidate", "patch contains a secret")
        if categories and self.policy.reject_pii:
            self._fail(
                "pii_candidate",
                f"patch contains PII: {', '.join(categories)}",
            )

        conflict_relation = self._validate_transition(
            normalized,
            session_id=session_id,
            target=target,
            merge_records=merge_records,
            active_for_identity=active_for_identity,
        )
        evidence_relation = "explicit_invalidation"
        if operation != "DELETE":
            if not self._value_supported(normalized.value, normalized.evidence):
                self._fail(
                    "unsupported_value",
                    "patch value is not supported by its evidence",
                )
            evidence_relation = "entailment"
        if operation == "UPDATE" and not self._has_marker(
            normalized.evidence,
            _EXPLICIT_UPDATE_MARKERS,
        ):
            self._fail(
                "missing_explicit_update",
                "UPDATE requires explicit change language in the evidence",
            )
        if operation == "DELETE" and not self._has_marker(
            normalized.evidence,
            _EXPLICIT_DELETE_MARKERS,
        ):
            self._fail(
                "missing_explicit_delete",
                "DELETE requires explicit invalidation language in the evidence",
            )
        return ToolMemoryPatchValidation(
            patch=normalized,
            evidence_relation=evidence_relation,
            conflict_relation=conflict_relation,
            pii_categories=categories,
            policy_version=self.policy.policy_version,
            evidence_span_corrections=evidence_span_corrections,
        )

    def _validate_partition(
        self,
        patch: ToolMemoryPatch,
        *,
        session_id: str,
    ) -> None:
        identity = patch.identity
        if identity is None:
            return
        expected_user = (
            normalize_identifier(self.policy.expected_user_id)
            if self.policy.expected_user_id
            else None
        )
        if expected_user is not None and identity.user_id != expected_user:
            self._fail(
                "fixed_user_mismatch",
                "patch user_id does not match the fixed runtime user",
            )
        allowed = {
            (
                normalize_identifier(domain),
                normalize_identifier(topic),
            )
            for domain, topic in self.policy.allowed_domain_topics
        }
        if allowed and (identity.tool_domain, identity.topic) not in allowed:
            self._fail(
                "unknown_tool_schema_partition",
                "patch domain/topic is not present in the Tool ontology",
            )
        if (
            identity.scope == "global"
            and expected_user is not None
            and identity.scope_key != expected_user
        ):
            self._fail(
                "global_scope_owner_mismatch",
                "global patch scope_key must match the fixed runtime user",
            )
        if (
            identity.scope == "session"
            and identity.scope_key != normalize_identifier(session_id)
        ):
            self._fail(
                "session_scope_owner_mismatch",
                "session patch scope_key must match the source session",
            )

    def _validate_shape(self, patch: ToolMemoryPatch) -> None:
        operation = patch.operation
        if operation == "ADD":
            if patch.target_record_id or patch.merge_record_ids:
                self._fail("invalid_add_target", "ADD cannot have target records")
            self._require_new_record_fields(patch)
            return
        if patch.target_record_id is None:
            self._fail(
                "missing_target",
                f"{operation} requires target_record_id",
            )
        if operation == "DELETE":
            if patch.identity is not None or patch.memory_type is not None:
                self._fail(
                    "invalid_delete_payload",
                    "DELETE cannot define a replacement record",
                )
            if patch.merge_record_ids:
                self._fail(
                    "invalid_delete_merge",
                    "DELETE cannot define merge_record_ids",
                )
            return
        self._require_new_record_fields(patch)
        if operation == "UPDATE" and patch.merge_record_ids:
            self._fail(
                "invalid_update_merge",
                "UPDATE cannot define merge_record_ids",
            )
        if operation == "MERGE" and not patch.merge_record_ids:
            self._fail(
                "missing_merge_records",
                "MERGE requires at least one merge_record_id",
            )

    def _require_new_record_fields(self, patch: ToolMemoryPatch) -> None:
        if patch.identity is None:
            self._fail("missing_identity", f"{patch.operation} requires identity")
        if patch.memory_type not in _TOOL_MEMORY_TYPES:
            self._fail(
                "invalid_memory_type",
                f"Unsupported memory type: {patch.memory_type}",
            )

    def _validate_transition(
        self,
        patch: ToolMemoryPatch,
        *,
        session_id: str,
        target: ToolMemoryRecord | None,
        merge_records: Sequence[ToolMemoryRecord],
        active_for_identity: ToolMemoryRecord | None,
    ) -> str:
        operation = patch.operation
        if operation == "ADD":
            if active_for_identity is not None:
                relation = (
                    "duplicate"
                    if self._same_value(active_for_identity.value, patch.value)
                    else "contradiction"
                )
                self._fail(
                    f"add_{relation}",
                    "ADD cannot replace an existing active record",
                )
            return "none"
        assert target is not None
        self._require_mutable_target(target, session_id)
        if operation == "DELETE":
            return "invalidation"

        assert patch.identity is not None
        self._require_same_owner_and_topic(target, patch.identity)
        proposed_key = tool_memory_record_key(patch.identity)
        if (
            active_for_identity is not None
            and active_for_identity.id != target.id
            and active_for_identity.id not in patch.merge_record_ids
        ):
            self._fail(
                "target_key_conflict",
                "another active record already owns the proposed record key",
            )
        if operation == "UPDATE":
            if (
                proposed_key == target.record_key
                and self._same_value(target.value, patch.value)
            ):
                self._fail(
                    "update_duplicate",
                    "UPDATE must change the value or record identity",
                )
            return (
                "contradiction"
                if proposed_key == target.record_key
                else "identity_change"
            )

        records = (target, *merge_records)
        if len({record.id for record in records}) != len(records):
            self._fail("duplicate_merge_target", "MERGE targets must be unique")
        for record in records:
            self._require_mutable_target(record, session_id)
            self._require_same_owner_and_topic(record, patch.identity)
            if not self._same_value(record.value, patch.value):
                self._fail(
                    "merge_value_conflict",
                    "MERGE only accepts records with equivalent values",
                )
        return "duplicate"

    def _validate_evidence(
        self,
        evidence_items: Sequence[MemoryEvidence],
        *,
        session_id: str,
        source_turn_id: str | None,
        messages: Mapping[int, tuple[str, str | None, str]],
    ) -> tuple[
        tuple[MemoryEvidence, ...],
        tuple[Mapping[str, Any], ...],
    ]:
        if not evidence_items:
            self._fail("missing_evidence", "patch requires evidence")
        normalized: list[MemoryEvidence] = []
        corrections: list[Mapping[str, Any]] = []
        for evidence in evidence_items:
            source = messages.get(evidence.message_id)
            if source is None:
                self._fail(
                    "missing_evidence_message",
                    f"Unknown evidence message: {evidence.message_id}",
                )
            message_session_id, message_turn_id, content = source
            if message_session_id != session_id:
                self._fail(
                    "cross_session_evidence",
                    "evidence must belong to the patch session",
                )
            if source_turn_id is not None and message_turn_id != source_turn_id:
                self._fail(
                    "cross_turn_evidence",
                    "evidence must belong to the patch source turn",
                )
            quote = evidence.quote.strip()
            if not quote:
                self._fail("empty_evidence", "evidence quote cannot be empty")
            provided_start = evidence.start_char
            provided_end = evidence.end_char
            if evidence.start_char is not None:
                start = evidence.start_char
                end = (
                    evidence.end_char
                    if evidence.end_char is not None
                    else start + len(quote)
                )
                if (
                    start < 0
                    or end != start + len(quote)
                    or content[start:end] != quote
                ):
                    corrected = self._unique_quote_span(content, quote)
                    if corrected is None:
                        self._fail(
                            "invalid_evidence_span",
                            "evidence span does not match the source message "
                            "and the exact quote is absent or ambiguous",
                        )
                    start, end = corrected
                    corrections.append(
                        {
                            "message_id": evidence.message_id,
                            "provided_start_char": provided_start,
                            "provided_end_char": provided_end,
                            "corrected_start_char": start,
                            "corrected_end_char": end,
                            "reason": "unique_exact_quote",
                        }
                    )
            else:
                start = content.find(quote)
                if start < 0:
                    self._fail(
                        "unsupported_evidence",
                        "evidence quote does not occur in its source message",
                    )
                end = start + len(quote)
                if (
                    evidence.end_char is not None
                    and evidence.end_char != end
                ):
                    corrected = self._unique_quote_span(content, quote)
                    if corrected is None:
                        self._fail(
                            "invalid_evidence_span",
                            "evidence span does not match the source message "
                            "and the exact quote is ambiguous",
                        )
                    start, end = corrected
                    corrections.append(
                        {
                            "message_id": evidence.message_id,
                            "provided_start_char": provided_start,
                            "provided_end_char": provided_end,
                            "corrected_start_char": start,
                            "corrected_end_char": end,
                            "reason": "unique_exact_quote",
                        }
                    )
            normalized.append(
                MemoryEvidence(
                    message_id=evidence.message_id,
                    quote=quote,
                    start_char=start,
                    end_char=end,
                )
            )
        return tuple(normalized), tuple(corrections)

    @staticmethod
    def _unique_quote_span(
        content: str,
        quote: str,
    ) -> tuple[int, int] | None:
        positions: list[int] = []
        offset = 0
        while True:
            position = content.find(quote, offset)
            if position < 0:
                break
            positions.append(position)
            if len(positions) > 1:
                return None
            offset = position + 1
        if not positions:
            return None
        start = positions[0]
        return start, start + len(quote)

    def _require_mutable_target(
        self,
        record: ToolMemoryRecord,
        session_id: str,
    ) -> None:
        if record.status != "active":
            self._fail(
                "inactive_target",
                f"Target record is not active: {record.id}",
            )
        if record.session_id != session_id:
            self._fail(
                "cross_session_target",
                "target record must belong to the patch session",
            )
        if self.policy.expected_user_id and record.user_id != normalize_identifier(
            self.policy.expected_user_id
        ):
            self._fail(
                "cross_user_target",
                "target record must belong to the fixed runtime user",
            )

    def _require_same_owner_and_topic(
        self,
        record: ToolMemoryRecord,
        identity: Any,
    ) -> None:
        if (
            record.user_id,
            record.tool_domain,
            record.topic,
            record.scope,
            record.scope_key,
        ) != (
            identity.user_id,
            identity.tool_domain,
            identity.topic,
            identity.scope,
            identity.scope_key,
        ):
            self._fail(
                "identity_mismatch",
                "patch cannot change user, Tool domain, topic, or scope owner",
            )

    def _pii_categories(self, patch: ToolMemoryPatch) -> tuple[str, ...]:
        identity = patch.identity
        payload = {
            "identity": (
                {
                    "user_id": identity.user_id,
                    "tool_domain": identity.tool_domain,
                    "topic": identity.topic,
                    "scope": identity.scope,
                    "scope_key": identity.scope_key,
                    "conditions": dict(identity.conditions),
                }
                if identity is not None
                else None
            ),
            "value": patch.value,
            "reason": patch.reason,
            "evidence": [item.quote for item in patch.evidence],
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return tuple(
            sorted(
                {
                    span.category
                    for span in detect_sensitive_spans(
                        text,
                        allowlist=self.policy.pii_allowlist,
                    )
                }
            )
        )

    @staticmethod
    def _value_supported(
        value: Any,
        evidence: Sequence[MemoryEvidence],
    ) -> bool:
        evidence_text = " ".join(item.quote for item in evidence).casefold()
        leaves = ToolMemoryPatchValidator._value_leaves(value)
        return bool(leaves) and all(
            str(leaf).casefold() in evidence_text for leaf in leaves
        )

    @staticmethod
    def _value_leaves(value: Any) -> tuple[str, ...]:
        if isinstance(value, Mapping):
            return tuple(
                leaf
                for item in value.values()
                for leaf in ToolMemoryPatchValidator._value_leaves(item)
            )
        if isinstance(value, (list, tuple)):
            return tuple(
                leaf
                for item in value
                for leaf in ToolMemoryPatchValidator._value_leaves(item)
            )
        if value is None:
            return ("null",)
        if isinstance(value, bool):
            return (str(value).casefold(),)
        if isinstance(value, (str, int, float)):
            normalized = str(value).strip()
            return (normalized,) if normalized else ()
        return ()

    @staticmethod
    def _same_value(first: Any, second: Any) -> bool:
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

    @staticmethod
    def _has_marker(
        evidence: Sequence[MemoryEvidence],
        markers: set[str],
    ) -> bool:
        text = " ".join(item.quote for item in evidence).casefold()
        return any(marker in text for marker in markers)

    @staticmethod
    def _fail(code: str, message: str) -> None:
        raise ToolMemoryPatchValidationError(code, message)


@dataclass(frozen=True)
class FactMemoryValidationPolicy:
    minimum_confidence: float = 0.5
    semantic_minimum_confidence: float = 0.7
    reject_pii: bool = True
    pii_allowlist: tuple[str, ...] = ()
    policy_version: str = "fact-memory-gate-v1"


@dataclass(frozen=True)
class FactMemoryValidation:
    candidate: FactMemoryCandidate
    disposition: str
    code: str
    message: str
    evidence_relation: str
    pii_categories: tuple[str, ...]
    policy_version: str
    evidence_span_corrections: tuple[Mapping[str, Any], ...] = ()
    ignored_runtime_fields: tuple[str, ...] = ()
    semantic_review: Mapping[str, Any] | None = None

    @property
    def accepted(self) -> bool:
        return self.disposition == "accepted"

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "disposition": self.disposition,
            "code": self.code,
            "message": self.message,
            "evidence_relation": self.evidence_relation,
            "pii_categories": list(self.pii_categories),
            "policy_version": self.policy_version,
            "evidence_span_corrections": [
                dict(item) for item in self.evidence_span_corrections
            ],
            "ignored_runtime_fields": list(self.ignored_runtime_fields),
            "semantic_review": (
                dict(self.semantic_review)
                if self.semantic_review is not None
                else None
            ),
        }


class FactMemoryCandidateValidator:
    """Separate structural rejection from uncertain semantic review."""

    def __init__(
        self,
        policy: FactMemoryValidationPolicy | None = None,
    ):
        self.policy = policy or FactMemoryValidationPolicy()

    def validate(
        self,
        candidate: FactMemoryCandidate,
        *,
        session_id: str,
        messages: Mapping[int, tuple[str, str, str]],
        allowed_evidence_message_ids: Sequence[int] = (),
    ) -> FactMemoryValidation:
        directive = candidate.directive.strip().upper()
        if directive not in {"UPSERT", "DELETE"}:
            return self._result(
                candidate,
                "rejected",
                "invalid_directive",
                f"Unsupported Fact directive: {candidate.directive}",
                "invalid",
            )
        if not candidate.entity_id.strip() or not candidate.predicate.strip():
            return self._result(
                candidate,
                "rejected",
                "missing_fact_identity",
                "Fact candidate requires entity_id and predicate",
                "invalid",
            )
        if not 0 <= candidate.confidence <= 1:
            return self._result(
                candidate,
                "rejected",
                "invalid_confidence",
                "Fact confidence must be between 0 and 1",
                "invalid",
            )
        evidence_result = self._validate_evidence(
            candidate.evidence,
            session_id=session_id,
            messages=messages,
            allowed_evidence_message_ids=allowed_evidence_message_ids,
        )
        if isinstance(evidence_result, FactMemoryValidation):
            return replace(
                evidence_result,
                candidate=candidate,
            )
        evidence, corrections = evidence_result
        normalized_value, ignored_value = normalize_memory_value(candidate.value)
        identity_conditions, ignored_identity = strip_runtime_identity_fields(
            candidate.identity_conditions
        )
        applicability, ignored_applicability = strip_runtime_identity_fields(
            candidate.applicability
        )
        ignored = tuple(
            dict.fromkeys(
                (
                    *(f"value.{item}" for item in ignored_value),
                    *(f"identity.{item}" for item in ignored_identity),
                    *(f"applicability.{item}" for item in ignored_applicability),
                )
            )
        )
        normalized = replace(
            candidate,
            value=normalized_value,
            identity_conditions=identity_conditions,
            applicability=applicability,
            evidence=evidence,
            directive=directive,
        )
        pii_categories = self._pii_categories(normalized)
        if "secret" in pii_categories:
            return self._result(
                normalized,
                "rejected",
                "secret_candidate",
                "Fact candidate contains a secret",
                "invalid",
                pii_categories=pii_categories,
                corrections=corrections,
                ignored=ignored,
            )
        if pii_categories and self.policy.reject_pii:
            return self._result(
                normalized,
                "rejected",
                "pii_candidate",
                f"Fact candidate contains PII: {', '.join(pii_categories)}",
                "invalid",
                pii_categories=pii_categories,
                corrections=corrections,
                ignored=ignored,
            )
        if candidate.confidence < self.policy.minimum_confidence:
            return self._result(
                normalized,
                "review",
                "confidence_below_minimum",
                "Fact confidence is below the automatic apply threshold",
                "uncertain",
                corrections=corrections,
                ignored=ignored,
            )
        evidence_text = " ".join(item.quote for item in evidence)
        if directive == "DELETE":
            if not contains_delete_intent(evidence_text):
                return self._result(
                    normalized,
                    "review",
                    "delete_intent_uncertain",
                    "DELETE requires explicit invalidation language",
                    "uncertain",
                    corrections=corrections,
                    ignored=ignored,
                )
            return self._result(
                normalized,
                "accepted",
                "accepted",
                "Fact candidate passed deterministic validation",
                "explicit_invalidation",
                corrections=corrections,
                ignored=ignored,
            )
        support = normalized_value_support(normalized.value, evidence_text)
        ignored = tuple(
            dict.fromkeys((*ignored, *support.ignored_runtime_fields))
        )
        if not support.supported:
            code = (
                "runtime_only_value"
                if support.checked_count == 0 and ignored
                else "value_entailment_uncertain"
            )
            details = ", ".join(support.unresolved) or "no durable value leaf"
            return self._result(
                normalized,
                "review",
                code,
                f"Fact value requires semantic review: {details}",
                "uncertain",
                corrections=corrections,
                ignored=ignored,
            )
        return self._result(
            normalized,
            "accepted",
            "accepted",
            "Fact candidate passed deterministic validation",
            "normalized_entailment",
            corrections=corrections,
            ignored=ignored,
        )

    def validate_transition(
        self,
        validation: FactMemoryValidation,
        *,
        operation: str,
    ) -> FactMemoryValidation:
        if not validation.accepted:
            return validation
        evidence_text = " ".join(
            item.quote for item in validation.candidate.evidence
        )
        if operation == "UPDATE" and not contains_update_intent(evidence_text):
            return replace(
                validation,
                disposition="review",
                code="update_intent_uncertain",
                message="UPDATE requires explicit change or preference language",
                evidence_relation="uncertain",
            )
        if operation == "DELETE" and not contains_delete_intent(evidence_text):
            return replace(
                validation,
                disposition="review",
                code="delete_intent_uncertain",
                message="DELETE requires explicit invalidation language",
                evidence_relation="uncertain",
            )
        return validation

    def apply_semantic_decision(
        self,
        validation: FactMemoryValidation,
        decision: FactMemorySemanticDecision,
    ) -> FactMemoryValidation:
        if validation.disposition == "rejected":
            return validation
        normalized = decision.decision.strip().upper()
        semantic_trace = {
            "decision": normalized,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "evidence_relation": decision.evidence_relation,
            "sequence_index": decision.sequence_index,
            "candidate_confidence": validation.candidate.confidence,
        }
        if not 0 <= decision.confidence <= 1:
            return replace(
                validation,
                disposition="review",
                code="semantic_invalid_confidence",
                message="Semantic review confidence is outside [0, 1]",
                evidence_relation="uncertain",
                semantic_review=semantic_trace,
            )
        if normalized not in {"ACCEPT", "REVIEW", "REJECT"}:
            return replace(
                validation,
                disposition="review",
                code="semantic_invalid_decision",
                message=f"Unsupported semantic decision: {normalized}",
                evidence_relation="uncertain",
                semantic_review=semantic_trace,
            )
        if (
            normalized in {"ACCEPT", "REJECT"}
            and decision.confidence
            < self.policy.semantic_minimum_confidence
        ):
            return replace(
                validation,
                disposition="review",
                code="semantic_confidence_below_minimum",
                message=decision.reason,
                evidence_relation="uncertain",
                semantic_review=semantic_trace,
            )
        if normalized == "ACCEPT":
            return replace(
                validation,
                candidate=replace(
                    validation.candidate,
                    confidence=decision.confidence,
                ),
                disposition="accepted",
                code="semantic_accepted",
                message=decision.reason,
                evidence_relation=decision.evidence_relation,
                semantic_review=semantic_trace,
            )
        if normalized == "REJECT":
            return replace(
                validation,
                disposition="rejected",
                code="semantic_rejected",
                message=decision.reason,
                evidence_relation=decision.evidence_relation,
                semantic_review=semantic_trace,
            )
        return replace(
            validation,
            disposition="review",
            code="semantic_review",
            message=decision.reason,
            evidence_relation="uncertain",
            semantic_review=semantic_trace,
        )

    def _validate_evidence(
        self,
        evidence_items: Sequence[MemoryEvidence],
        *,
        session_id: str,
        messages: Mapping[int, tuple[str, str, str]],
        allowed_evidence_message_ids: Sequence[int],
    ) -> (
        tuple[
            tuple[MemoryEvidence, ...],
            tuple[Mapping[str, Any], ...],
        ]
        | FactMemoryValidation
    ):
        placeholder = FactMemoryCandidate(
            entity_id="",
            predicate="",
            value=None,
            identity_conditions={},
            applicability={},
            capability_hints=(),
            memory_type="fact",
            confidence=0,
            evidence=(),
        )
        if not evidence_items:
            return self._result(
                placeholder,
                "rejected",
                "missing_evidence",
                "Fact candidate requires evidence",
                "invalid",
            )
        allowed = set(allowed_evidence_message_ids)
        normalized = []
        corrections = []
        for evidence in evidence_items:
            source = messages.get(evidence.message_id)
            if source is None:
                return self._result(
                    placeholder,
                    "rejected",
                    "missing_evidence_message",
                    f"Unknown evidence message: {evidence.message_id}",
                    "invalid",
                )
            message_session, role, content = source
            if message_session != session_id:
                return self._result(
                    placeholder,
                    "rejected",
                    "cross_session_evidence",
                    "Evidence belongs to another session",
                    "invalid",
                )
            if allowed and evidence.message_id not in allowed:
                return self._result(
                    placeholder,
                    "rejected",
                    "out_of_batch_evidence",
                    "Evidence is not part of the source batch",
                    "invalid",
                )
            if role != "user":
                return self._result(
                    placeholder,
                    "rejected",
                    "non_user_evidence",
                    "Fact evidence must come from a user message",
                    "invalid",
                )
            quote = evidence.quote.strip()
            if not quote:
                return self._result(
                    placeholder,
                    "rejected",
                    "empty_evidence",
                    "Evidence quote cannot be empty",
                    "invalid",
                )
            exact_start = content.find(quote)
            if exact_start < 0:
                return self._result(
                    placeholder,
                    "rejected",
                    "unsupported_evidence",
                    "Evidence quote does not occur in the source message",
                    "invalid",
                )
            exact_end = exact_start + len(quote)
            if (
                evidence.start_char != exact_start
                or evidence.end_char != exact_end
            ) and (
                evidence.start_char is not None
                or evidence.end_char is not None
            ):
                if content.find(quote, exact_start + 1) >= 0:
                    return self._result(
                        placeholder,
                        "rejected",
                        "ambiguous_evidence_span",
                        "Incorrect offsets cannot be repaired for a repeated quote",
                        "invalid",
                    )
                corrections.append(
                    {
                        "message_id": evidence.message_id,
                        "provided_start_char": evidence.start_char,
                        "provided_end_char": evidence.end_char,
                        "corrected_start_char": exact_start,
                        "corrected_end_char": exact_end,
                        "reason": "unique_exact_quote",
                    }
                )
            normalized.append(
                MemoryEvidence(
                    message_id=evidence.message_id,
                    quote=quote,
                    start_char=exact_start,
                    end_char=exact_end,
                )
            )
        return tuple(normalized), tuple(corrections)

    def _pii_categories(
        self,
        candidate: FactMemoryCandidate,
    ) -> tuple[str, ...]:
        payload = {
            "entity_id": candidate.entity_id,
            "predicate": candidate.predicate,
            "value": candidate.value,
            "identity_conditions": dict(candidate.identity_conditions),
            "applicability": dict(candidate.applicability),
            "reason": candidate.reason,
            "evidence": [item.quote for item in candidate.evidence],
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return tuple(
            sorted(
                {
                    span.category
                    for span in detect_sensitive_spans(
                        text,
                        allowlist=self.policy.pii_allowlist,
                    )
                }
            )
        )

    def _result(
        self,
        candidate: FactMemoryCandidate,
        disposition: str,
        code: str,
        message: str,
        evidence_relation: str,
        *,
        pii_categories: tuple[str, ...] = (),
        corrections: tuple[Mapping[str, Any], ...] = (),
        ignored: tuple[str, ...] = (),
    ) -> FactMemoryValidation:
        return FactMemoryValidation(
            candidate=candidate,
            disposition=disposition,
            code=code,
            message=message,
            evidence_relation=evidence_relation,
            pii_categories=pii_categories,
            policy_version=self.policy.policy_version,
            evidence_span_corrections=corrections,
            ignored_runtime_fields=ignored,
        )
