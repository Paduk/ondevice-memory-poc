from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from palmclaw_ubuntu.providers import apply_recursive_summary_patch
from palmclaw_ubuntu.vehicle_bench.v1_generation import V1Stage2Artifact
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import V1DialogueTurnRecord
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import V1GeneratedEventDialogue
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import (
    V2HybridArtifact,
    dialogue_sha256,
    event_dialogue_turns,
    event_sha256,
    text_sha256,
)
from palmclaw_ubuntu.vehicle_bench.v2_native_turnwise import (
    V2NativeScenarioArtifact,
)
from palmclaw_ubuntu.vehicle_bench.v2_quality_evaluation import (
    QualityMethod,
    default_quality_artifact_paths,
)

UPDATE_AUDIT_RECORD_VERSION = "vehiclemembench-v2-turn-audit-record-v1"
UPDATE_AUDIT_CASE_VERSION = "vehiclemembench-v2-event-audit-case-v1"
UPDATE_AUDIT_MANIFEST_VERSION = "vehiclemembench-v2-update-audit-manifest-v1"
DEFAULT_NO_OP_EVENT_SAMPLE_RATE = 0.10
DEFAULT_NO_OP_SAMPLE_SEED = "vehiclemembench-v2-update-audit-noop-v1"

AuditDecision = Literal["UPDATE", "NO_OP"]
AuditStatus = Literal["PASS", "FAIL"]


@dataclass(frozen=True)
class UpdateAuditArtifactPaths:
    method: QualityMethod
    scenario_index: int
    root: Path
    memory_artifact: Path
    stage2_artifact: Path | None
    dialogue_root: Path | None


@dataclass(frozen=True)
class AuditDialogueTurn:
    turn_id: str
    speaker_id: str
    speaker_name: str
    text: str

    @classmethod
    def from_record(cls, turn: V1DialogueTurnRecord) -> AuditDialogueTurn:
        return cls(
            turn_id=turn.turn_id,
            speaker_id=turn.speaker_id,
            speaker_name=turn.speaker_name,
            text=turn.text,
        )


@dataclass(frozen=True)
class TurnAuditRecord:
    schema_version: str
    record_id: str
    method: QualityMethod
    scenario_index: int
    source_stage2_sha256: str
    source_memory_artifact_sha256: str
    event_id: str
    timeline_index: int
    chain_kind: str
    reasoning_type: str | None
    global_turn_index: int
    event_turn_index: int
    current_turn: AuditDialogueTurn
    dialogue_prefix: tuple[AuditDialogueTurn, ...]
    before_memory: str
    decision: AuditDecision
    reason_code: str
    reason: str
    source_update_indexes: tuple[int, ...]
    committed_source_update_indexes_before: tuple[int, ...]
    committed_source_update_indexes_after: tuple[int, ...]
    expected_updates: tuple[dict[str, Any], ...]
    evidence: tuple[dict[str, Any], ...]
    operations: tuple[dict[str, Any], ...]
    after_memory: str
    before_memory_sha256: str
    after_memory_sha256: str
    input_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeterministicTurnAudit:
    record_id: str
    status: AuditStatus
    error_codes: tuple[str, ...]
    details: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UpdateAuditBundle:
    paths: UpdateAuditArtifactPaths
    records: tuple[TurnAuditRecord, ...]
    audits: tuple[DeterministicTurnAudit, ...]
    artifact_errors: tuple[str, ...]


@dataclass(frozen=True)
class EventAuditCase:
    schema_version: str
    case_id: str
    method: QualityMethod
    scenario_index: int
    source_stage2_sha256: str
    source_memory_artifact_sha256: str
    event_id: str
    timeline_index: int
    chain_kind: str
    reasoning_type: str | None
    selection_reason: Literal["EXPECTED_OR_GENERATED_UPDATE", "SAMPLED_NO_OP"]
    dialogue: tuple[AuditDialogueTurn, ...]
    event_before_memory: str
    event_after_memory: str
    expected_updates: tuple[dict[str, Any], ...]
    candidate_updates: tuple[dict[str, Any], ...]
    no_op_reason_codes: tuple[str, ...]
    input_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def event_audit_case_from_dict(value: dict[str, Any]) -> EventAuditCase:
    payload = dict(value)
    payload["dialogue"] = tuple(
        AuditDialogueTurn(**item) for item in payload.get("dialogue", ())
    )
    payload["expected_updates"] = tuple(payload.get("expected_updates", ()))
    payload["candidate_updates"] = tuple(payload.get("candidate_updates", ()))
    payload["no_op_reason_codes"] = tuple(payload.get("no_op_reason_codes", ()))
    return EventAuditCase(**payload)


def default_update_audit_artifact_paths(
    evaluation_root: Path,
) -> tuple[UpdateAuditArtifactPaths, ...]:
    evaluation_root = evaluation_root.expanduser().resolve()
    stage2_by_hash = _index_stage2_artifacts(evaluation_root)
    resolved: list[UpdateAuditArtifactPaths] = []
    for quality in default_quality_artifact_paths(evaluation_root):
        source_hash = _memory_source_hash(quality.memory_artifact)
        stage2 = _choose_stage2(stage2_by_hash.get(source_hash, ()))
        dialogue_root: Path | None = None
        if quality.method == "post_hoc":
            dialogue_root = quality.root / "dialogues"
        elif quality.method == "hybrid" and stage2 is not None:
            dialogue_root = stage2.parent / "dialogues"
        resolved.append(
            UpdateAuditArtifactPaths(
                method=quality.method,
                scenario_index=quality.scenario_index,
                root=quality.root,
                memory_artifact=quality.memory_artifact,
                stage2_artifact=stage2,
                dialogue_root=dialogue_root,
            )
        )
    return tuple(resolved)


def build_update_audit_bundle(
    paths: UpdateAuditArtifactPaths,
) -> UpdateAuditBundle:
    if paths.stage2_artifact is None or not paths.stage2_artifact.is_file():
        raise ValueError(f"Stage 2 artifact is unavailable: {paths.root}")
    stage2 = V1Stage2Artifact.model_validate_json(
        paths.stage2_artifact.read_text(encoding="utf-8")
    )
    if paths.method == "native_turnwise":
        artifact = V2NativeScenarioArtifact.model_validate_json(
            paths.memory_artifact.read_text(encoding="utf-8")
        )
        records, artifact_errors = _native_records(paths, stage2, artifact)
    else:
        artifact = V2HybridArtifact.model_validate_json(
            paths.memory_artifact.read_text(encoding="utf-8")
        )
        records, artifact_errors = _hybrid_records(paths, stage2, artifact)
    if artifact.source_stage2_sha256 != stage2.artifact_sha256:
        raise ValueError(f"Stage 2 source hash mismatch: {paths.root}")
    audits = tuple(audit_turn_record(record) for record in records)
    return UpdateAuditBundle(
        paths=paths,
        records=records,
        audits=audits,
        artifact_errors=artifact_errors,
    )


def audit_turn_record(record: TurnAuditRecord) -> DeterministicTurnAudit:
    errors: list[str] = []
    details: list[str] = []

    def fail(code: str, detail: str) -> None:
        errors.append(code)
        details.append(detail)

    if record.event_turn_index != len(record.dialogue_prefix) - 1:
        fail("PREFIX_INDEX_MISMATCH", "Current turn is not the end of its prefix")
    if not record.dialogue_prefix or record.dialogue_prefix[-1] != record.current_turn:
        fail("CURRENT_TURN_MISMATCH", "Current turn differs from prefix tail")
    if text_sha256(record.before_memory) != record.before_memory_sha256:
        fail("BEFORE_MEMORY_HASH_MISMATCH", "before_memory hash is invalid")
    if text_sha256(record.after_memory) != record.after_memory_sha256:
        fail("AFTER_MEMORY_HASH_MISMATCH", "after_memory hash is invalid")
    input_payload = record.as_dict()
    input_payload.pop("record_id")
    input_payload.pop("input_sha256")
    if _canonical_sha256(input_payload) != record.input_sha256:
        fail("RECORD_INPUT_HASH_MISMATCH", "audit record input hash is invalid")

    expected_indexes = set(range(len(record.expected_updates)))
    source_indexes = record.source_update_indexes
    if len(source_indexes) != len(set(source_indexes)):
        fail("DUPLICATE_SOURCE_UPDATE_INDEX", "source update indexes repeat")
    if set(source_indexes) - expected_indexes:
        fail("SOURCE_UPDATE_INDEX_OUT_OF_RANGE", "source update index is invalid")
    committed_before = set(record.committed_source_update_indexes_before)
    committed_after = set(record.committed_source_update_indexes_after)
    if committed_before - expected_indexes or committed_after - expected_indexes:
        fail("COMMITTED_UPDATE_INDEX_OUT_OF_RANGE", "committed index is invalid")
    if not committed_before.issubset(committed_after):
        fail("COMMITTED_UPDATE_ROLLBACK", "committed update disappeared")
    if committed_before.intersection(source_indexes):
        fail("DUPLICATE_UPDATE_COMMIT", "source update was committed twice")
    if committed_after != committed_before.union(source_indexes):
        fail("COMMITTED_UPDATE_TRANSITION", "committed indexes do not match label")

    if record.decision == "NO_OP":
        if record.operations or record.evidence or source_indexes:
            fail("NO_OP_HAS_UPDATE_PAYLOAD", "NO_OP contains UPDATE fields")
        if record.before_memory != record.after_memory:
            fail("NO_OP_CHANGED_MEMORY", "NO_OP changed memory")
    else:
        if not record.operations or not record.evidence or not source_indexes:
            fail("UPDATE_MISSING_PAYLOAD", "UPDATE is missing Patch/evidence/source")
        if len(record.evidence) != len(source_indexes):
            fail("UPDATE_EVIDENCE_COUNT", "UPDATE evidence/source counts differ")
        try:
            applied, _ = apply_recursive_summary_patch(
                record.before_memory,
                list(record.operations),
            )
        except Exception as exc:
            fail("PATCH_APPLY_FAILED", f"{type(exc).__name__}: {exc}")
        else:
            if applied != record.after_memory:
                fail("PATCH_RESULT_MISMATCH", "Patch does not reproduce after_memory")

    prefix_by_id = {turn.turn_id: turn for turn in record.dialogue_prefix}
    for evidence in record.evidence:
        turn_id = str(evidence.get("turn_id", ""))
        quote = str(evidence.get("quote", ""))
        evidence_index = evidence.get("event_turn_index")
        source_event = evidence.get("source_event_id")
        turn = prefix_by_id.get(turn_id)
        if source_event != record.event_id:
            fail("EVIDENCE_EVENT_MISMATCH", f"Evidence {turn_id} uses another event")
        if (
            not isinstance(evidence_index, int)
            or evidence_index > record.event_turn_index
        ):
            fail("FUTURE_EVIDENCE", f"Evidence {turn_id} is outside causal cutoff")
        elif evidence_index < 0 or evidence_index >= len(record.dialogue_prefix):
            fail("EVIDENCE_INDEX_OUT_OF_RANGE", f"Evidence {turn_id} index is invalid")
        elif record.dialogue_prefix[evidence_index].turn_id != turn_id:
            fail("EVIDENCE_INDEX_MISMATCH", f"Evidence {turn_id} index is inconsistent")
        if turn is None:
            fail("EVIDENCE_TURN_MISSING", f"Evidence turn {turn_id} is absent")
        elif not quote or quote not in turn.text:
            fail("EVIDENCE_QUOTE_MISSING", f"Evidence quote is absent from {turn_id}")

    return DeterministicTurnAudit(
        record_id=record.record_id,
        status="FAIL" if errors else "PASS",
        error_codes=tuple(errors),
        details=tuple(details),
    )


def select_event_audit_cases(
    bundle: UpdateAuditBundle,
    *,
    no_op_event_sample_rate: float = DEFAULT_NO_OP_EVENT_SAMPLE_RATE,
    no_op_sample_seed: str = DEFAULT_NO_OP_SAMPLE_SEED,
) -> tuple[EventAuditCase, ...]:
    if not 0 <= no_op_event_sample_rate <= 1:
        raise ValueError("NO_OP event sample rate must be between 0 and 1")
    records_by_event: dict[str, list[TurnAuditRecord]] = defaultdict(list)
    for record in bundle.records:
        records_by_event[record.event_id].append(record)
    mandatory = {
        event_id
        for event_id, records in records_by_event.items()
        if records[0].expected_updates
        or any(record.decision == "UPDATE" for record in records)
    }
    sampled = _sample_no_op_events(
        bundle,
        records_by_event,
        mandatory=mandatory,
        sample_rate=no_op_event_sample_rate,
        seed=no_op_sample_seed,
    )
    cases = []
    for event_id in sorted(mandatory.union(sampled)):
        records = records_by_event[event_id]
        first = records[0]
        last = records[-1]
        updates = tuple(
            {
                "record_id": record.record_id,
                "global_turn_index": record.global_turn_index,
                "event_turn_index": record.event_turn_index,
                "turn_id": record.current_turn.turn_id,
                "reason_code": record.reason_code,
                "reason": record.reason,
                "source_update_indexes": list(record.source_update_indexes),
                "evidence": list(record.evidence),
                "operations": list(record.operations),
                "before_memory": record.before_memory,
                "after_memory": record.after_memory,
                "before_memory_sha256": record.before_memory_sha256,
                "after_memory_sha256": record.after_memory_sha256,
            }
            for record in records
            if record.decision == "UPDATE"
        )
        body = {
            "schema_version": UPDATE_AUDIT_CASE_VERSION,
            "method": first.method,
            "scenario_index": first.scenario_index,
            "source_stage2_sha256": first.source_stage2_sha256,
            "source_memory_artifact_sha256": (
                first.source_memory_artifact_sha256
            ),
            "event_id": event_id,
            "timeline_index": first.timeline_index,
            "chain_kind": first.chain_kind,
            "reasoning_type": first.reasoning_type,
            "selection_reason": (
                "EXPECTED_OR_GENERATED_UPDATE"
                if event_id in mandatory
                else "SAMPLED_NO_OP"
            ),
            "dialogue": [asdict(turn) for turn in last.dialogue_prefix],
            "event_before_memory": first.before_memory,
            "event_after_memory": last.after_memory,
            "expected_updates": list(first.expected_updates),
            "candidate_updates": list(updates),
            "no_op_reason_codes": sorted(
                {record.reason_code for record in records if record.decision == "NO_OP"}
            ),
        }
        case_id = f"{first.method}:s{first.scenario_index:02d}:{event_id}"
        cases.append(
            EventAuditCase(
                case_id=case_id,
                input_sha256=_canonical_sha256(body),
                dialogue=last.dialogue_prefix,
                expected_updates=first.expected_updates,
                candidate_updates=updates,
                no_op_reason_codes=tuple(body["no_op_reason_codes"]),
                **{
                    key: value
                    for key, value in body.items()
                    if key
                    not in {
                        "dialogue",
                        "expected_updates",
                        "candidate_updates",
                        "no_op_reason_codes",
                    }
                },
            )
        )
    return tuple(cases)


def build_update_audit_manifest(
    artifacts: tuple[UpdateAuditArtifactPaths, ...],
    *,
    no_op_event_sample_rate: float = DEFAULT_NO_OP_EVENT_SAMPLE_RATE,
    no_op_sample_seed: str = DEFAULT_NO_OP_SAMPLE_SEED,
) -> dict[str, Any]:
    if not 0 <= no_op_event_sample_rate <= 1:
        raise ValueError("NO_OP event sample rate must be between 0 and 1")
    rows: list[dict[str, Any]] = []
    source_hashes: dict[int, set[str]] = defaultdict(set)
    for paths in artifacts:
        missing = [
            str(path)
            for path in (paths.memory_artifact, paths.stage2_artifact)
            if path is None or not path.is_file()
        ]
        if paths.method != "native_turnwise" and (
            paths.dialogue_root is None or not paths.dialogue_root.is_dir()
        ):
            missing.append(str(paths.dialogue_root or "(dialogue root unresolved)"))
        if missing:
            rows.append(
                {
                    "method": paths.method,
                    "scenario_index": paths.scenario_index,
                    "status": "PENDING",
                    "root": str(paths.root),
                    "missing_paths": missing,
                }
            )
            continue
        try:
            bundle = build_update_audit_bundle(paths)
        except Exception as exc:
            rows.append(
                {
                    "method": paths.method,
                    "scenario_index": paths.scenario_index,
                    "status": "INVALID",
                    "root": str(paths.root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        row = _bundle_manifest_row(
            bundle,
            no_op_event_sample_rate=no_op_event_sample_rate,
            no_op_sample_seed=no_op_sample_seed,
        )
        rows.append(row)
        source_hashes[paths.scenario_index].add(row["source_stage2_sha256"])

    mismatches = {
        str(index): sorted(values)
        for index, values in source_hashes.items()
        if len(values) > 1
    }
    totals = Counter()
    for row in rows:
        if row["status"] != "READY":
            continue
        for key in (
            "event_count",
            "turn_count",
            "update_turn_count",
            "no_op_turn_count",
            "expected_update_count",
            "judge_event_count",
            "initial_judge_call_count",
            "deterministic_failure_count",
        ):
            totals[key] += int(row[key])
    invalid_count = sum(row["status"] == "INVALID" for row in rows)
    return {
        "schema_version": UPDATE_AUDIT_MANIFEST_VERSION,
        "status": "INVALID" if invalid_count or mismatches else "READY_OR_PENDING",
        "no_op_event_sample_rate": no_op_event_sample_rate,
        "no_op_sample_seed": no_op_sample_seed,
        "artifact_count": len(rows),
        "ready_count": sum(row["status"] == "READY" for row in rows),
        "pending_count": sum(row["status"] == "PENDING" for row in rows),
        "invalid_count": invalid_count,
        "source_hash_mismatches": mismatches,
        "totals": dict(totals),
        "artifacts": rows,
    }


def _hybrid_records(
    paths: UpdateAuditArtifactPaths,
    stage2: V1Stage2Artifact,
    artifact: V2HybridArtifact,
) -> tuple[tuple[TurnAuditRecord, ...], tuple[str, ...]]:
    if paths.dialogue_root is None:
        raise ValueError("Hybrid/Post-hoc audit requires a dialogue root")
    timeline = stage2.interleaved_timeline
    errors: list[str] = []
    if len(artifact.event_checkpoints) != len(timeline):
        errors.append("EVENT_COUNT_MISMATCH")
    records: list[TurnAuditRecord] = []
    previous_memory = ""
    previous_global_index = -1
    for offset, checkpoint in enumerate(artifact.event_checkpoints):
        if offset >= len(timeline):
            errors.append(f"EXTRA_EVENT:{checkpoint.event_id}")
            break
        source = timeline[offset]
        if checkpoint.event_id != source.event.event_id:
            errors.append(f"EVENT_ORDER_MISMATCH:{checkpoint.event_id}")
        dialogue_path = paths.dialogue_root / f"{checkpoint.event_id}.json"
        generated = V1GeneratedEventDialogue.model_validate_json(
            dialogue_path.read_text(encoding="utf-8")
        )
        turns = event_dialogue_turns(stage2, generated)
        if checkpoint.source_event_sha256 != event_sha256(source.event):
            errors.append(f"SOURCE_EVENT_HASH_MISMATCH:{checkpoint.event_id}")
        if checkpoint.source_dialogue_sha256 != dialogue_sha256(turns):
            errors.append(f"SOURCE_DIALOGUE_HASH_MISMATCH:{checkpoint.event_id}")
        if checkpoint.before_memory_sha256 != text_sha256(previous_memory):
            errors.append(f"EVENT_BEFORE_MEMORY_MISMATCH:{checkpoint.event_id}")
        committed: tuple[int, ...] = ()
        expected = tuple(
            item.model_dump(mode="json") for item in source.event.preference_updates
        )
        for label, turn in zip(checkpoint.turn_labels, turns, strict=False):
            committed_after = tuple(
                sorted({*committed, *label.source_update_indexes})
            )
            after_memory = (
                label.after_memory
                if label.decision == "UPDATE" and label.after_memory is not None
                else previous_memory
            )
            record = _make_record(
                paths=paths,
                stage2=stage2,
                artifact_sha256=artifact.artifact_sha256,
                source=source,
                label=label,
                turn=turn,
                turns=turns,
                before_memory=previous_memory,
                after_memory=after_memory,
                committed_before=committed,
                committed_after=committed_after,
                expected_updates=expected,
            )
            records.append(record)
            if label.global_turn_index != previous_global_index + 1:
                errors.append(f"GLOBAL_TURN_GAP:{record.record_id}")
            previous_global_index = label.global_turn_index
            previous_memory = after_memory
            committed = committed_after
        if len(checkpoint.turn_labels) != len(turns):
            errors.append(f"TURN_COUNT_MISMATCH:{checkpoint.event_id}")
        if set(committed) != set(range(len(expected))):
            errors.append(f"EXPECTED_UPDATE_NOT_COMMITTED:{checkpoint.event_id}")
        if checkpoint.after_memory != previous_memory:
            errors.append(f"EVENT_AFTER_MEMORY_MISMATCH:{checkpoint.event_id}")
    if artifact.final_memory != previous_memory:
        errors.append("FINAL_MEMORY_MISMATCH")
    return tuple(records), tuple(errors)


def _native_records(
    paths: UpdateAuditArtifactPaths,
    stage2: V1Stage2Artifact,
    artifact: V2NativeScenarioArtifact,
) -> tuple[tuple[TurnAuditRecord, ...], tuple[str, ...]]:
    timeline = stage2.interleaved_timeline
    errors: list[str] = []
    if len(artifact.event_artifacts) != len(timeline):
        errors.append("EVENT_COUNT_MISMATCH")
    records: list[TurnAuditRecord] = []
    previous_memory = ""
    previous_global_index = -1
    for offset, event_artifact in enumerate(artifact.event_artifacts):
        if offset >= len(timeline):
            errors.append(f"EXTRA_EVENT:{event_artifact.event_id}")
            break
        source = timeline[offset]
        if event_artifact.event_id != source.event.event_id:
            errors.append(f"EVENT_ORDER_MISMATCH:{event_artifact.event_id}")
        expected = tuple(
            item.model_dump(mode="json") for item in source.event.preference_updates
        )
        turns = tuple(
            checkpoint.dialogue_turn for checkpoint in event_artifact.turn_checkpoints
        )
        committed: tuple[int, ...] = ()
        for checkpoint, turn in zip(
            event_artifact.turn_checkpoints,
            turns,
            strict=True,
        ):
            label = checkpoint.label
            record = _make_record(
                paths=paths,
                stage2=stage2,
                artifact_sha256=artifact.artifact_sha256,
                source=source,
                label=label,
                turn=turn,
                turns=turns,
                before_memory=previous_memory,
                after_memory=checkpoint.after_memory,
                committed_before=committed,
                committed_after=checkpoint.committed_source_update_indexes,
                expected_updates=expected,
            )
            records.append(record)
            if label.global_turn_index != previous_global_index + 1:
                errors.append(f"GLOBAL_TURN_GAP:{record.record_id}")
            previous_global_index = label.global_turn_index
            previous_memory = checkpoint.after_memory
            committed = checkpoint.committed_source_update_indexes
        if set(committed) != set(range(len(expected))):
            errors.append(f"EXPECTED_UPDATE_NOT_COMMITTED:{event_artifact.event_id}")
        if event_artifact.after_memory != previous_memory:
            errors.append(f"EVENT_AFTER_MEMORY_MISMATCH:{event_artifact.event_id}")
    if artifact.final_memory != previous_memory:
        errors.append("FINAL_MEMORY_MISMATCH")
    return tuple(records), tuple(errors)


def _make_record(
    *,
    paths: UpdateAuditArtifactPaths,
    stage2: V1Stage2Artifact,
    artifact_sha256: str,
    source: Any,
    label: Any,
    turn: V1DialogueTurnRecord,
    turns: tuple[V1DialogueTurnRecord, ...],
    before_memory: str,
    after_memory: str,
    committed_before: tuple[int, ...],
    committed_after: tuple[int, ...],
    expected_updates: tuple[dict[str, Any], ...],
) -> TurnAuditRecord:
    prefix = tuple(
        AuditDialogueTurn.from_record(item)
        for item in turns[: label.event_turn_index + 1]
    )
    current = AuditDialogueTurn.from_record(turn)
    base = {
        "schema_version": UPDATE_AUDIT_RECORD_VERSION,
        "method": paths.method,
        "scenario_index": paths.scenario_index,
        "source_stage2_sha256": stage2.artifact_sha256,
        "source_memory_artifact_sha256": artifact_sha256,
        "event_id": source.event.event_id,
        "timeline_index": source.timeline_index,
        "chain_kind": source.chain_kind,
        "reasoning_type": source.reasoning_type,
        "global_turn_index": label.global_turn_index,
        "event_turn_index": label.event_turn_index,
        "current_turn": asdict(current),
        "dialogue_prefix": [asdict(item) for item in prefix],
        "before_memory": before_memory,
        "decision": label.decision,
        "reason_code": label.reason_code,
        "reason": label.reason,
        "source_update_indexes": list(label.source_update_indexes),
        "committed_source_update_indexes_before": list(committed_before),
        "committed_source_update_indexes_after": list(committed_after),
        "expected_updates": list(expected_updates),
        "evidence": [item.model_dump(mode="json") for item in label.evidence],
        "operations": [item.model_dump(mode="json") for item in label.operations],
        "after_memory": after_memory,
        "before_memory_sha256": label.before_memory_sha256,
        "after_memory_sha256": label.after_memory_sha256,
    }
    record_id = (
        f"{paths.method}:s{paths.scenario_index:02d}:"
        f"{source.event.event_id}:{turn.turn_id}"
    )
    return TurnAuditRecord(
        record_id=record_id,
        input_sha256=_canonical_sha256(base),
        current_turn=current,
        dialogue_prefix=prefix,
        source_update_indexes=tuple(label.source_update_indexes),
        committed_source_update_indexes_before=committed_before,
        committed_source_update_indexes_after=committed_after,
        expected_updates=expected_updates,
        evidence=tuple(item.model_dump(mode="json") for item in label.evidence),
        operations=tuple(item.model_dump(mode="json") for item in label.operations),
        **{
            key: value
            for key, value in base.items()
            if key
            not in {
                "current_turn",
                "dialogue_prefix",
                "source_update_indexes",
                "committed_source_update_indexes_before",
                "committed_source_update_indexes_after",
                "expected_updates",
                "evidence",
                "operations",
            }
        },
    )


def _bundle_manifest_row(
    bundle: UpdateAuditBundle,
    *,
    no_op_event_sample_rate: float,
    no_op_sample_seed: str,
) -> dict[str, Any]:
    records_by_event: dict[str, list[TurnAuditRecord]] = defaultdict(list)
    for record in bundle.records:
        records_by_event[record.event_id].append(record)
    mandatory = {
        event_id
        for event_id, records in records_by_event.items()
        if records[0].expected_updates
        or any(record.decision == "UPDATE" for record in records)
    }
    sampled = _sample_no_op_events(
        bundle,
        records_by_event,
        mandatory=mandatory,
        sample_rate=no_op_event_sample_rate,
        seed=no_op_sample_seed,
    )
    sampled_by_stratum: dict[str, list[str]] = defaultdict(list)
    for event_id in sampled:
        records = records_by_event[event_id]
        stratum = "+".join(sorted({record.reason_code for record in records}))
        sampled_by_stratum[stratum].append(event_id)
    failures = [item for item in bundle.audits if item.status == "FAIL"]
    error_counts = Counter(bundle.artifact_errors)
    for failure in failures:
        error_counts.update(failure.error_codes)
    expected_update_count = sum(
        len(records[0].expected_updates) for records in records_by_event.values()
    )
    cases = select_event_audit_cases(
        bundle,
        no_op_event_sample_rate=no_op_event_sample_rate,
        no_op_sample_seed=no_op_sample_seed,
    )
    judge_events = [case.event_id for case in cases]
    return {
        "method": bundle.paths.method,
        "scenario_index": bundle.paths.scenario_index,
        "status": "READY",
        "root": str(bundle.paths.root),
        "memory_artifact": str(bundle.paths.memory_artifact),
        "stage2_artifact": str(bundle.paths.stage2_artifact),
        "dialogue_root": (
            str(bundle.paths.dialogue_root)
            if bundle.paths.dialogue_root is not None
            else None
        ),
        "source_stage2_sha256": bundle.records[0].source_stage2_sha256,
        "event_count": len(records_by_event),
        "turn_count": len(bundle.records),
        "update_turn_count": sum(
            record.decision == "UPDATE" for record in bundle.records
        ),
        "no_op_turn_count": sum(
            record.decision == "NO_OP" for record in bundle.records
        ),
        "expected_update_count": expected_update_count,
        "mandatory_judge_event_count": len(mandatory),
        "sampled_no_op_event_count": len(sampled),
        "sampled_no_op_events_by_stratum": dict(sampled_by_stratum),
        "judge_event_count": len(judge_events),
        "initial_judge_call_count": len(judge_events) * 2,
        "judge_event_ids": judge_events,
        "deterministic_pass_count": len(bundle.audits) - len(failures),
        "deterministic_failure_count": len(failures)
        + len(bundle.artifact_errors),
        "deterministic_error_counts": dict(sorted(error_counts.items())),
    }


def _sample_no_op_events(
    bundle: UpdateAuditBundle,
    records_by_event: dict[str, list[TurnAuditRecord]],
    *,
    mandatory: set[str],
    sample_rate: float,
    seed: str,
) -> set[str]:
    strata: dict[str, list[str]] = defaultdict(list)
    for event_id, records in records_by_event.items():
        if event_id in mandatory:
            continue
        stratum = "+".join(sorted({record.reason_code for record in records}))
        strata[stratum].append(event_id)
    sampled: set[str] = set()
    for stratum, event_ids in sorted(strata.items()):
        ranked = sorted(
            event_ids,
            key=lambda event_id: _canonical_sha256(
                [
                    seed,
                    bundle.paths.method,
                    bundle.paths.scenario_index,
                    stratum,
                    event_id,
                ]
            ),
        )
        count = (
            min(len(ranked), max(1, math.ceil(len(ranked) * sample_rate)))
            if sample_rate and ranked
            else 0
        )
        sampled.update(ranked[:count])
    return sampled


def _index_stage2_artifacts(evaluation_root: Path) -> dict[str, tuple[Path, ...]]:
    indexed: dict[str, list[Path]] = defaultdict(list)
    hybrid_root = evaluation_root / "vehiclemembench-v2-hybrid"
    if not hybrid_root.is_dir():
        return {}
    for path in sorted(hybrid_root.rglob("stage2-v2-anchored.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            digest = payload.get("artifact_sha256")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(digest, str) and len(digest) == 64:
            indexed[digest].append(path.resolve())
    return {key: tuple(values) for key, values in indexed.items()}


def _choose_stage2(candidates: tuple[Path, ...]) -> Path | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda path: (not (path.parent / "dialogues").is_dir(), str(path)),
    )[0]


def _memory_source_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("source_stage2_sha256")
    return value if isinstance(value, str) else None


def _canonical_sha256(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
