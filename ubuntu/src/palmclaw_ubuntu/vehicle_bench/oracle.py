from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.fact_memory import FactMemoryLinker
from palmclaw_ubuntu.models import (
    FactMemoryCandidate,
    FactMemorySemanticDecision,
    MemoryEvidence,
    ModelUsage,
)
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.vehicle_bench.dataset import (
    VehicleBenchmarkDataset,
    VehicleTask,
)

ORACLE_STAGES = (
    "extraction",
    "structure",
    "gate",
    "route",
    "retrieval",
    "binding",
)


@dataclass(frozen=True)
class OracleModeSpec:
    name: str
    interventions: tuple[str, ...]
    exposes_gold_tool_names: bool = False
    exposes_gold_arguments: bool = False
    requires_fact_annotations: bool = False


ORACLE_MODE_SPECS = {
    spec.name: spec
    for spec in (
        OracleModeSpec("baseline_fact", ()),
        OracleModeSpec(
            "oracle_extraction",
            ("extraction",),
            requires_fact_annotations=True,
        ),
        OracleModeSpec(
            "oracle_structure",
            ("structure",),
            requires_fact_annotations=True,
        ),
        OracleModeSpec(
            "oracle_gate",
            ("gate",),
            requires_fact_annotations=True,
        ),
        OracleModeSpec(
            "oracle_route",
            ("route",),
            exposes_gold_tool_names=True,
        ),
        OracleModeSpec(
            "oracle_retrieval",
            ("retrieval",),
            requires_fact_annotations=True,
        ),
        OracleModeSpec(
            "oracle_binding",
            ("binding",),
            exposes_gold_tool_names=True,
            exposes_gold_arguments=True,
        ),
        OracleModeSpec(
            "oracle_full",
            ORACLE_STAGES,
            exposes_gold_tool_names=True,
            exposes_gold_arguments=True,
            requires_fact_annotations=True,
        ),
    )
}

_QUOTED_EVIDENCE_PATTERNS = (
    re.compile(r"'([^'\n]+)'"),
    re.compile(r"‘([^’\n]+)’"),
    re.compile(r'"([^"\n]+)"'),
    re.compile(r"“([^”\n]+)”"),
)


@dataclass(frozen=True)
class OracleContractAuditResult:
    run_id: str
    status: str
    artifact_dir: Path | None
    audit: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "artifact_dir": (
                str(self.artifact_dir) if self.artifact_dir is not None else None
            ),
            "audit": self.audit,
        }


@dataclass(frozen=True)
class OracleFactRecordKey:
    entity_id: str
    predicate: str
    value: Any
    identity_conditions: dict[str, Any] | None = None
    applicability: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OracleRetrievalLabel:
    task_id: str
    record_status: str
    record_keys: tuple[OracleFactRecordKey, ...] = ()


@dataclass(frozen=True)
class OracleRetrievalAnnotations:
    version: str
    dataset_sha256: str
    tool_schema_sha256: str
    scenario_indices: tuple[int, ...]
    labels: dict[str, OracleRetrievalLabel]
    source_path: Path

    def require(self, task_id: str) -> OracleRetrievalLabel:
        try:
            return self.labels[task_id]
        except KeyError as exc:
            raise ValueError(
                f"Oracle Retrieval annotation is missing task {task_id}"
            ) from exc

    def manifest(self) -> dict[str, Any]:
        present = sum(
            label.record_status == "record_present"
            for label in self.labels.values()
        )
        return {
            "version": self.version,
            "dataset_sha256": self.dataset_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "scenario_indices": list(self.scenario_indices),
            "task_count": len(self.labels),
            "record_present_tasks": present,
            "record_absent_tasks": len(self.labels) - present,
            "sha256": hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
        }


@dataclass(frozen=True)
class OracleGateLabel:
    task_id: str
    scenario_index: int
    candidate_status: str
    candidate_sha256: str | None = None
    original_status: str | None = None
    validation_code: str | None = None


@dataclass(frozen=True)
class OracleGateAnnotations:
    version: str
    dataset_sha256: str
    tool_schema_sha256: str
    scenario_indices: tuple[int, ...]
    labels: dict[str, OracleGateLabel]
    source_path: Path

    def require(self, task_id: str) -> OracleGateLabel:
        try:
            return self.labels[task_id]
        except KeyError as exc:
            raise ValueError(
                f"Oracle Gate annotation is missing task {task_id}"
            ) from exc

    def recoverable_for_scenario(
        self,
        scenario_index: int,
    ) -> tuple[OracleGateLabel, ...]:
        return tuple(
            label
            for label in self.labels.values()
            if label.scenario_index == scenario_index
            and label.candidate_status == "gate_recoverable"
        )

    def manifest(self) -> dict[str, Any]:
        recoverable = sum(
            label.candidate_status == "gate_recoverable"
            for label in self.labels.values()
        )
        return {
            "version": self.version,
            "dataset_sha256": self.dataset_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "scenario_indices": list(self.scenario_indices),
            "task_count": len(self.labels),
            "gate_recoverable_tasks": recoverable,
            "not_gate_recoverable_tasks": len(self.labels) - recoverable,
            "sha256": hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
        }


@dataclass(frozen=True)
class OracleGateOverlayResult:
    candidate_record_ids: dict[str, str]
    run_id: str | None
    applied_candidates: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_record_ids": dict(self.candidate_record_ids),
            "run_id": self.run_id,
            "applied_candidates": self.applied_candidates,
        }


@dataclass(frozen=True)
class OracleStageFact:
    fact_id: str
    stage: str
    scenario_index: int
    task_ids: tuple[str, ...]
    record_key: OracleFactRecordKey
    evidence_quotes: tuple[str, ...]
    memory_type: str
    capability_hints: tuple[str, ...]
    candidate_sha256: str | None = None


@dataclass(frozen=True)
class OracleStageFactAnnotations:
    version: str
    dataset_sha256: str
    tool_schema_sha256: str
    scenario_indices: tuple[int, ...]
    gate_only_tasks: tuple[str, ...]
    facts: tuple[OracleStageFact, ...]
    source_path: Path

    def fact_for_task(
        self,
        task_id: str,
        *,
        stage: str,
    ) -> OracleStageFact | None:
        matches = [
            fact
            for fact in self.facts
            if fact.stage == stage and task_id in fact.task_ids
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Oracle {stage} task maps to multiple facts: {task_id}"
            )
        return matches[0] if matches else None

    def facts_for_scenario(
        self,
        scenario_index: int,
        *,
        stage: str,
    ) -> tuple[OracleStageFact, ...]:
        return tuple(
            fact
            for fact in self.facts
            if fact.scenario_index == scenario_index and fact.stage == stage
        )

    def manifest(self) -> dict[str, Any]:
        structure = tuple(
            fact for fact in self.facts if fact.stage == "structure"
        )
        extraction = tuple(
            fact for fact in self.facts if fact.stage == "extraction"
        )
        return {
            "version": self.version,
            "dataset_sha256": self.dataset_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "scenario_indices": list(self.scenario_indices),
            "gate_only_tasks": len(self.gate_only_tasks),
            "structure_facts": len(structure),
            "structure_tasks": sum(len(fact.task_ids) for fact in structure),
            "extraction_facts": len(extraction),
            "extraction_tasks": sum(len(fact.task_ids) for fact in extraction),
            "sha256": hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
        }


@dataclass(frozen=True)
class OracleStageFactOverlayResult:
    stage: str | None
    fact_record_ids: dict[str, str]
    task_record_ids: dict[str, tuple[str, ...]]
    outcome_counts: dict[str, int]
    run_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "fact_record_ids": dict(self.fact_record_ids),
            "task_record_ids": {
                task_id: list(record_ids)
                for task_id, record_ids in self.task_record_ids.items()
            },
            "outcome_counts": dict(self.outcome_counts),
            "run_ids": list(self.run_ids),
        }


@dataclass(frozen=True)
class OracleFullOverlayResult:
    gate: OracleGateOverlayResult
    structure: OracleStageFactOverlayResult
    extraction: OracleStageFactOverlayResult
    task_record_ids: dict[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate.as_dict(),
            "structure": self.structure.as_dict(),
            "extraction": self.extraction.as_dict(),
            "task_record_ids": {
                task_id: list(record_ids)
                for task_id, record_ids in self.task_record_ids.items()
            },
            "covered_tasks": len(self.task_record_ids),
            "unique_records": len(
                {
                    record_id
                    for record_ids in self.task_record_ids.values()
                    for record_id in record_ids
                }
            ),
        }


def oracle_mode_spec(name: str) -> OracleModeSpec:
    try:
        return ORACLE_MODE_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown VehicleMemBench Oracle mode: {name}") from exc


def cumulative_oracle_stages(through: str) -> tuple[str, ...]:
    if through not in ORACLE_STAGES:
        raise ValueError(f"Unknown VehicleMemBench Oracle stage: {through}")
    return ORACLE_STAGES[: ORACLE_STAGES.index(through) + 1]


def load_oracle_retrieval_annotations(
    path: Path | str,
    *,
    dataset: VehicleBenchmarkDataset,
) -> OracleRetrievalAnnotations:
    source = Path(path).expanduser().resolve()
    payload = _read_json(source)
    version = str(payload.get("version", "")).strip()
    if version != "vehiclemembench-oracle-retrieval-v1":
        raise ValueError(f"Unsupported Oracle Retrieval annotation version: {version}")
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("Oracle Retrieval annotation benchmark must be an object")
    dataset_sha256 = str(benchmark.get("dataset_sha256", ""))
    tool_schema_sha256 = str(benchmark.get("tool_schema_sha256", ""))
    if dataset_sha256 != dataset.manifest.dataset_sha256:
        raise ValueError("Oracle Retrieval annotation dataset hash mismatch")
    if tool_schema_sha256 != dataset.manifest.tool_schema_sha256:
        raise ValueError("Oracle Retrieval annotation Tool schema hash mismatch")
    raw_indices = payload.get("scenario_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("Oracle Retrieval scenario_indices must be a non-empty list")
    scenario_indices = tuple(dict.fromkeys(int(item) for item in raw_indices))
    expected_task_ids = {
        task.id
        for scenario_index in scenario_indices
        for task in dataset.scenario(scenario_index).tasks
    }

    labels: dict[str, OracleRetrievalLabel] = {}
    raw_present = payload.get("record_present")
    if not isinstance(raw_present, list):
        raise ValueError("Oracle Retrieval record_present must be a list")
    for item in raw_present:
        if not isinstance(item, dict):
            raise ValueError("Oracle Retrieval present label must be an object")
        task_id = str(item.get("task_id", ""))
        raw_keys = item.get("record_keys")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise ValueError(
                f"Oracle Retrieval present task requires record_keys: {task_id}"
            )
        keys = tuple(_parse_oracle_fact_key(key, task_id) for key in raw_keys)
        _insert_oracle_label(
            labels,
            OracleRetrievalLabel(
                task_id=task_id,
                record_status="record_present",
                record_keys=keys,
            ),
        )

    raw_absent = payload.get("record_absent")
    if not isinstance(raw_absent, list):
        raise ValueError("Oracle Retrieval record_absent must be a list")
    for raw_task_id in raw_absent:
        task_id = str(raw_task_id)
        _insert_oracle_label(
            labels,
            OracleRetrievalLabel(
                task_id=task_id,
                record_status="record_absent",
            ),
        )

    annotated = set(labels)
    if annotated != expected_task_ids:
        missing = sorted(expected_task_ids - annotated)
        unknown = sorted(annotated - expected_task_ids)
        raise ValueError(
            "Oracle Retrieval task coverage mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    return OracleRetrievalAnnotations(
        version=version,
        dataset_sha256=dataset_sha256,
        tool_schema_sha256=tool_schema_sha256,
        scenario_indices=scenario_indices,
        labels=labels,
        source_path=source,
    )


def load_oracle_gate_annotations(
    path: Path | str,
    *,
    dataset: VehicleBenchmarkDataset,
) -> OracleGateAnnotations:
    source = Path(path).expanduser().resolve()
    payload = _read_json(source)
    version = str(payload.get("version", "")).strip()
    if version != "vehiclemembench-oracle-gate-v1":
        raise ValueError(f"Unsupported Oracle Gate annotation version: {version}")
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("Oracle Gate annotation benchmark must be an object")
    dataset_sha256 = str(benchmark.get("dataset_sha256", ""))
    tool_schema_sha256 = str(benchmark.get("tool_schema_sha256", ""))
    if dataset_sha256 != dataset.manifest.dataset_sha256:
        raise ValueError("Oracle Gate annotation dataset hash mismatch")
    if tool_schema_sha256 != dataset.manifest.tool_schema_sha256:
        raise ValueError("Oracle Gate annotation Tool schema hash mismatch")
    raw_indices = payload.get("scenario_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("Oracle Gate scenario_indices must be a non-empty list")
    scenario_indices = tuple(dict.fromkeys(int(item) for item in raw_indices))
    tasks = {
        task.id: task
        for scenario_index in scenario_indices
        for task in dataset.scenario(scenario_index).tasks
    }

    labels: dict[str, OracleGateLabel] = {}
    raw_recoverable = payload.get("gate_recoverable")
    if not isinstance(raw_recoverable, list):
        raise ValueError("Oracle Gate gate_recoverable must be a list")
    for item in raw_recoverable:
        if not isinstance(item, dict):
            raise ValueError("Oracle Gate recoverable label must be an object")
        task_id = str(item.get("task_id", ""))
        candidate_sha256 = str(item.get("candidate_sha256", ""))
        original_status = str(item.get("original_status", ""))
        validation_code = str(item.get("validation_code", ""))
        if (
            not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256)
            or original_status not in {"review", "rejected"}
            or not validation_code
        ):
            raise ValueError(
                f"Invalid Oracle Gate recoverable label: {task_id}"
            )
        task = tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown Oracle Gate task: {task_id}")
        _insert_oracle_gate_label(
            labels,
            OracleGateLabel(
                task_id=task_id,
                scenario_index=task.scenario_index,
                candidate_status="gate_recoverable",
                candidate_sha256=candidate_sha256,
                original_status=original_status,
                validation_code=validation_code,
            ),
        )

    raw_unrecoverable = payload.get("not_gate_recoverable")
    if not isinstance(raw_unrecoverable, list):
        raise ValueError("Oracle Gate not_gate_recoverable must be a list")
    for raw_task_id in raw_unrecoverable:
        task_id = str(raw_task_id)
        task = tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown Oracle Gate task: {task_id}")
        _insert_oracle_gate_label(
            labels,
            OracleGateLabel(
                task_id=task_id,
                scenario_index=task.scenario_index,
                candidate_status="not_gate_recoverable",
            ),
        )

    annotated = set(labels)
    expected = set(tasks)
    if annotated != expected:
        raise ValueError(
            "Oracle Gate task coverage mismatch: "
            f"missing={sorted(expected - annotated)}, "
            f"unknown={sorted(annotated - expected)}"
        )
    return OracleGateAnnotations(
        version=version,
        dataset_sha256=dataset_sha256,
        tool_schema_sha256=tool_schema_sha256,
        scenario_indices=scenario_indices,
        labels=labels,
        source_path=source,
    )


def load_oracle_stage_fact_annotations(
    path: Path | str,
    *,
    dataset: VehicleBenchmarkDataset,
    retrieval_annotations: OracleRetrievalAnnotations,
    gate_annotations: OracleGateAnnotations | None = None,
) -> OracleStageFactAnnotations:
    source = Path(path).expanduser().resolve()
    payload = _read_json(source)
    version = str(payload.get("version", "")).strip()
    if version != "vehiclemembench-oracle-stage-facts-v1":
        raise ValueError(
            f"Unsupported Oracle Stage Fact annotation version: {version}"
        )
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("Oracle Stage Fact benchmark must be an object")
    dataset_sha256 = str(benchmark.get("dataset_sha256", ""))
    tool_schema_sha256 = str(benchmark.get("tool_schema_sha256", ""))
    if dataset_sha256 != dataset.manifest.dataset_sha256:
        raise ValueError("Oracle Stage Fact dataset hash mismatch")
    if tool_schema_sha256 != dataset.manifest.tool_schema_sha256:
        raise ValueError("Oracle Stage Fact Tool schema hash mismatch")

    raw_indices = payload.get("scenario_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("Oracle Stage Fact scenario_indices must be a list")
    scenario_indices = tuple(dict.fromkeys(int(item) for item in raw_indices))
    if scenario_indices != retrieval_annotations.scenario_indices:
        raise ValueError(
            "Oracle Stage Fact and Retrieval scenario coverage differ"
        )
    tasks = {
        task.id: task
        for scenario_index in scenario_indices
        for task in dataset.scenario(scenario_index).tasks
    }
    histories = {
        scenario_index: dataset.scenario(
            scenario_index
        ).history_path.read_text(encoding="utf-8")
        for scenario_index in scenario_indices
    }

    raw_gate_only = payload.get("gate_only_tasks")
    if not isinstance(raw_gate_only, list):
        raise ValueError("Oracle Stage Fact gate_only_tasks must be a list")
    gate_only_tasks = tuple(dict.fromkeys(str(item) for item in raw_gate_only))
    facts: list[OracleStageFact] = []
    seen_fact_ids: set[str] = set()
    classified_tasks = set(gate_only_tasks)
    for stage, field in (
        ("structure", "structure_facts"),
        ("extraction", "extraction_facts"),
    ):
        raw_facts = payload.get(field)
        if not isinstance(raw_facts, list):
            raise ValueError(f"Oracle Stage Fact {field} must be a list")
        for item in raw_facts:
            if not isinstance(item, dict):
                raise ValueError(f"Oracle {stage} fact must be an object")
            fact_id = str(item.get("fact_id", "")).strip()
            raw_task_ids = item.get("task_ids")
            raw_quotes = item.get("evidence_quotes")
            if (
                not fact_id
                or fact_id in seen_fact_ids
                or not isinstance(raw_task_ids, list)
                or not raw_task_ids
                or not isinstance(raw_quotes, list)
                or not raw_quotes
            ):
                raise ValueError(f"Invalid Oracle {stage} fact: {fact_id}")
            task_ids = tuple(dict.fromkeys(str(value) for value in raw_task_ids))
            unknown = set(task_ids) - set(tasks)
            if unknown:
                raise ValueError(
                    f"Unknown Oracle {stage} tasks: {sorted(unknown)}"
                )
            scenario_set = {tasks[task_id].scenario_index for task_id in task_ids}
            if len(scenario_set) != 1:
                raise ValueError(
                    f"Oracle fact crosses scenarios: {fact_id}"
                )
            scenario_index = next(iter(scenario_set))
            overlap = classified_tasks.intersection(task_ids)
            if overlap:
                raise ValueError(
                    "Oracle Stage Fact task has multiple classifications: "
                    f"{sorted(overlap)}"
                )
            quotes = tuple(str(value).strip() for value in raw_quotes)
            if any(
                not quote or histories[scenario_index].count(quote) != 1
                for quote in quotes
            ):
                raise ValueError(
                    "Oracle Stage Fact evidence must occur exactly once in "
                    f"scenario history: {fact_id}"
                )
            candidate_sha256 = item.get("candidate_sha256")
            if stage == "structure":
                if not isinstance(candidate_sha256, str) or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    candidate_sha256,
                ):
                    raise ValueError(
                        f"Oracle Structure fact requires candidate hash: {fact_id}"
                    )
            elif candidate_sha256 is not None:
                raise ValueError(
                    f"Oracle Extraction fact cannot name a candidate: {fact_id}"
                )
            memory_type = str(item.get("memory_type", "")).strip()
            raw_hints = item.get("capability_hints")
            if not memory_type or not isinstance(raw_hints, list):
                raise ValueError(
                    f"Oracle Stage Fact metadata is incomplete: {fact_id}"
                )
            facts.append(
                OracleStageFact(
                    fact_id=fact_id,
                    stage=stage,
                    scenario_index=scenario_index,
                    task_ids=task_ids,
                    record_key=_parse_oracle_fact_key(item, fact_id),
                    evidence_quotes=quotes,
                    memory_type=memory_type,
                    capability_hints=tuple(
                        dict.fromkeys(str(value) for value in raw_hints)
                    ),
                    candidate_sha256=candidate_sha256,
                )
            )
            classified_tasks.update(task_ids)
            seen_fact_ids.add(fact_id)

    absent_tasks = {
        task_id
        for task_id, label in retrieval_annotations.labels.items()
        if label.record_status == "record_absent"
    }
    if classified_tasks != absent_tasks:
        raise ValueError(
            "Oracle Stage Fact classification must exactly cover Retrieval "
            f"record_absent tasks: missing={sorted(absent_tasks-classified_tasks)}, "
            f"unknown={sorted(classified_tasks-absent_tasks)}"
        )
    if gate_annotations is not None:
        recoverable = {
            task_id
            for task_id, label in gate_annotations.labels.items()
            if label.candidate_status == "gate_recoverable"
        }
        if set(gate_only_tasks) != recoverable:
            raise ValueError(
                "Oracle Stage Fact gate_only_tasks disagree with Gate labels"
            )
    return OracleStageFactAnnotations(
        version=version,
        dataset_sha256=dataset_sha256,
        tool_schema_sha256=tool_schema_sha256,
        scenario_indices=scenario_indices,
        gate_only_tasks=gate_only_tasks,
        facts=tuple(facts),
        source_path=source,
    )


def oracle_fact_candidate_sha256(
    candidate: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
) -> str:
    payload = {
        "candidate": candidate,
        "evidence": list(evidence),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_oracle_gate_overlay(
    repository: SQLiteRepository,
    *,
    session_id: str,
    user_id: str,
    annotations: OracleGateAnnotations,
    scenario_index: int,
) -> OracleGateOverlayResult:
    labels = annotations.recoverable_for_scenario(scenario_index)
    if not labels:
        return OracleGateOverlayResult({}, None, 0)

    events_by_sha: dict[str, list[dict[str, Any]]] = {}
    for event in repository.fact_memory_candidate_events_for_session(session_id):
        candidate = event["candidate"]
        evidence = event["evidence"]
        digest = oracle_fact_candidate_sha256(candidate, evidence)
        events_by_sha.setdefault(digest, []).append(event)

    selected: list[tuple[str, FactMemoryCandidate]] = []
    seen: set[str] = set()
    for label in labels:
        assert label.candidate_sha256 is not None
        if label.candidate_sha256 in seen:
            continue
        matches = events_by_sha.get(label.candidate_sha256, [])
        if len(matches) != 1:
            raise ValueError(
                "Oracle Gate candidate must match exactly one frozen event: "
                f"task={label.task_id}, matches={len(matches)}"
            )
        event = matches[0]
        validation = event["validation"]
        if (
            event["status"] != label.original_status
            or validation.get("code") != label.validation_code
            or validation.get("code")
            in {
                "cross_session_evidence",
                "missing_evidence",
                "missing_evidence_message",
                "pii_candidate",
                "secret_candidate",
            }
        ):
            raise ValueError(
                f"Oracle Gate candidate contract mismatch: {label.task_id}"
            )
        selected.append(
            (
                label.candidate_sha256,
                _stored_fact_candidate(
                    event["candidate"],
                    event["evidence"],
                ),
            )
        )
        seen.add(label.candidate_sha256)

    selected.sort(
        key=lambda item: min(
            (source.message_id for source in item[1].evidence),
            default=2**63 - 1,
        )
    )
    run_id = repository.begin_memory_patch_run(
        session_id=session_id,
        source_turn_id=None,
        backend="oracle",
        model_id="reviewed-gold-gate",
        prompt_version="vehicle-oracle-gate-v1",
        schema_version="fact-memory-candidate-v1",
        status="running",
    )
    linker = FactMemoryLinker(repository, user_id=user_id)
    try:
        candidates = tuple(candidate for _, candidate in selected)
        decisions = {
            index: FactMemorySemanticDecision(
                sequence_index=index,
                decision="ACCEPT",
                confidence=1.0,
                reason="Reviewed Gold Fact matches this candidate",
                evidence_relation="entailment",
            )
            for index in range(len(candidates))
        }
        result = linker.apply_candidates(
            run_id=run_id,
            session_id=session_id,
            candidates=candidates,
            allowed_evidence_message_ids=tuple(
                source.message_id
                for candidate in candidates
                for source in candidate.evidence
            ),
            semantic_decisions=decisions,
        )
        record_ids = {}
        for (digest, _), outcome in zip(selected, result.outcomes, strict=True):
            if outcome.status != "applied" or outcome.result_record_id is None:
                raise RuntimeError(
                    "Oracle Gate candidate did not produce an active record: "
                    f"{digest}"
                )
            record_ids[digest] = outcome.result_record_id
        repository.finish_memory_patch_run(
            run_id,
            status="completed",
            usage=ModelUsage(),
        )
    except Exception as exc:
        repository.finish_memory_patch_run(
            run_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return OracleGateOverlayResult(
        candidate_record_ids=record_ids,
        run_id=run_id,
        applied_candidates=len(record_ids),
    )


def apply_oracle_stage_fact_overlay(
    repository: SQLiteRepository,
    *,
    session_id: str,
    user_id: str,
    annotations: OracleStageFactAnnotations,
    scenario_index: int,
    stage: str,
) -> OracleStageFactOverlayResult:
    if stage not in {"structure", "extraction"}:
        raise ValueError(f"Unsupported Oracle Fact stage: {stage}")
    facts = annotations.facts_for_scenario(scenario_index, stage=stage)
    if not facts:
        return OracleStageFactOverlayResult(stage, {}, {}, {}, ())

    events_by_sha: dict[str, list[dict[str, Any]]] = {}
    if stage == "structure":
        for event in repository.fact_memory_candidate_events_for_session(
            session_id
        ):
            digest = oracle_fact_candidate_sha256(
                event["candidate"],
                event["evidence"],
            )
            events_by_sha.setdefault(digest, []).append(event)
    messages = repository.list_messages(session_id)
    message_matches: dict[str, MemoryEvidence] = {}
    for fact in facts:
        for quote in fact.evidence_quotes:
            matches = [
                message
                for message in messages
                if quote in message.content
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Oracle Stage Fact evidence must match exactly one cached "
                    f"message: fact={fact.fact_id}, quote={quote!r}"
                )
            message = matches[0]
            start = message.content.index(quote)
            message_matches[f"{fact.fact_id}\0{quote}"] = MemoryEvidence(
                message_id=message.id,
                quote=quote,
                start_char=start,
                end_char=start + len(quote),
            )

    linker = FactMemoryLinker(repository, user_id=user_id)
    fact_record_ids: dict[str, str] = {}
    task_record_ids: dict[str, tuple[str, ...]] = {}
    outcome_counts: dict[str, int] = {}
    run_ids: list[str] = []
    for fact in facts:
        semantic_decisions: dict[int, FactMemorySemanticDecision] = {}
        if fact.stage == "structure":
            assert fact.candidate_sha256 is not None
            frozen = events_by_sha.get(fact.candidate_sha256, [])
            if len(frozen) != 1 or frozen[0]["status"] != "applied":
                raise ValueError(
                    "Oracle Structure fact must reference one applied frozen "
                    f"candidate: {fact.fact_id}"
                )
            raw_review = frozen[0]["validation"].get("semantic_review")
            if isinstance(raw_review, dict):
                semantic_decisions[0] = FactMemorySemanticDecision(
                    sequence_index=0,
                    decision=str(raw_review["decision"]),
                    confidence=float(raw_review["confidence"]),
                    reason=str(raw_review["reason"]),
                    evidence_relation=str(raw_review["evidence_relation"]),
                )
        evidence = tuple(
            message_matches[f"{fact.fact_id}\0{quote}"]
            for quote in fact.evidence_quotes
        )
        candidate = FactMemoryCandidate(
            entity_id=fact.record_key.entity_id,
            predicate=fact.record_key.predicate,
            value=fact.record_key.value,
            identity_conditions=dict(
                fact.record_key.identity_conditions or {}
            ),
            applicability=dict(fact.record_key.applicability or {}),
            capability_hints=fact.capability_hints,
            memory_type=fact.memory_type,
            confidence=1.0,
            evidence=evidence,
            directive="UPSERT",
            reason=f"Reviewed Gold {stage} Oracle fact: {fact.fact_id}",
        )
        run_id = repository.begin_memory_patch_run(
            session_id=session_id,
            source_turn_id=None,
            backend="oracle",
            model_id=f"reviewed-gold-{stage}",
            prompt_version=f"vehicle-oracle-{stage}-v1",
            schema_version="fact-memory-candidate-v1",
            status="running",
        )
        run_ids.append(run_id)
        try:
            result = linker.apply_candidates(
                run_id=run_id,
                session_id=session_id,
                candidates=(candidate,),
                allowed_evidence_message_ids=tuple(
                    source.message_id for source in evidence
                ),
                semantic_decisions=semantic_decisions,
            )
            outcome = result.outcomes[0]
            outcome_counts[outcome.status] = (
                outcome_counts.get(outcome.status, 0) + 1
            )
            if outcome.result_record_id is not None:
                fact_record_ids[fact.fact_id] = outcome.result_record_id
                for task_id in fact.task_ids:
                    task_record_ids[task_id] = (outcome.result_record_id,)
            repository.finish_memory_patch_run(
                run_id,
                status="completed",
                usage=ModelUsage(),
            )
        except Exception as exc:
            repository.finish_memory_patch_run(
                run_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
    return OracleStageFactOverlayResult(
        stage=stage,
        fact_record_ids=fact_record_ids,
        task_record_ids=task_record_ids,
        outcome_counts=outcome_counts,
        run_ids=tuple(run_ids),
    )


def apply_oracle_full_overlay(
    repository: SQLiteRepository,
    *,
    session_id: str,
    user_id: str,
    stage_annotations: OracleStageFactAnnotations,
    gate_annotations: OracleGateAnnotations,
    scenario_index: int,
) -> OracleFullOverlayResult:
    structure = apply_oracle_stage_fact_overlay(
        repository,
        session_id=session_id,
        user_id=user_id,
        annotations=stage_annotations,
        scenario_index=scenario_index,
        stage="structure",
    )
    extraction = apply_oracle_stage_fact_overlay(
        repository,
        session_id=session_id,
        user_id=user_id,
        annotations=stage_annotations,
        scenario_index=scenario_index,
        stage="extraction",
    )
    gate = apply_oracle_gate_overlay(
        repository,
        session_id=session_id,
        user_id=user_id,
        annotations=gate_annotations,
        scenario_index=scenario_index,
    )
    task_record_ids: dict[str, tuple[str, ...]] = {}
    for overlay in (structure, extraction):
        for task_id, record_ids in overlay.task_record_ids.items():
            if task_id in task_record_ids:
                raise ValueError(
                    f"Oracle Full task has multiple stage facts: {task_id}"
                )
            task_record_ids[task_id] = record_ids
    for label in gate_annotations.recoverable_for_scenario(scenario_index):
        assert label.candidate_sha256 is not None
        try:
            record_id = gate.candidate_record_ids[label.candidate_sha256]
        except KeyError as exc:
            raise ValueError(
                f"Oracle Full Gate record is missing: {label.task_id}"
            ) from exc
        if label.task_id in task_record_ids:
            raise ValueError(
                f"Oracle Full task has multiple classifications: {label.task_id}"
            )
        task_record_ids[label.task_id] = (record_id,)
    return OracleFullOverlayResult(
        gate=gate,
        structure=structure,
        extraction=extraction,
        task_record_ids=task_record_ids,
    )


def run_oracle_contract_audit(
    dataset: VehicleBenchmarkDataset,
    *,
    scenario_indices: Sequence[int],
    task_limit: int | None = None,
    baseline_run_dir: Path | None = None,
    output_root: Path | None = None,
) -> OracleContractAuditResult:
    indices = tuple(dict.fromkeys(int(index) for index in scenario_indices))
    if not indices:
        raise ValueError("At least one scenario is required")
    if task_limit is not None and task_limit < 1:
        raise ValueError("task_limit must be at least 1")

    tool_names = {
        str(schema["name"])
        for schema in dataset.tool_schemas
        if isinstance(schema, dict) and schema.get("name")
    }
    task_rows: list[dict[str, Any]] = []
    for scenario_index in indices:
        scenario = dataset.scenario(scenario_index)
        history = scenario.history_path.read_text(encoding="utf-8")
        tasks = (
            scenario.tasks[:task_limit]
            if task_limit is not None
            else scenario.tasks
        )
        task_rows.extend(
            _audit_task(task, history=history, tool_names=tool_names)
            for task in tasks
        )

    baseline = _audit_baseline(
        dataset,
        scenario_indices=indices,
        baseline_run_dir=baseline_run_dir,
    )
    summary = _summarize_task_audit(task_rows)
    blocking_issues = (
        summary["missing_gold_memory"]
        + summary["missing_gold_calls"]
        + summary["unknown_gold_tool_calls"]
        + int(not baseline["compatible"])
    )
    audit = {
        "status": "passed" if blocking_issues == 0 else "failed",
        "benchmark": dataset.manifest.as_dict(),
        "scenario_indices": list(indices),
        "task_limit": task_limit,
        "oracle_modes": {
            name: asdict(spec) for name, spec in ORACLE_MODE_SPECS.items()
        },
        "direct_gold_contract": {
            "gold_memory": "task-relevant benchmark evidence",
            "gold_tool_names": "gold_calls[].name",
            "gold_arguments": "gold_calls[].arguments",
        },
        "missing_gold_contract": {
            "fact_annotations": (
                "canonical entity, predicate, value, conditions, and source "
                "evidence are not provided by VehicleMemBench"
            )
        },
        "summary": summary,
        "baseline": baseline,
        "tasks": task_rows,
    }
    run_id = str(uuid.uuid4())
    artifact_dir = (
        Path(output_root).expanduser().resolve() / run_id
        if output_root is not None
        else None
    )
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": run_id,
            "status": audit["status"],
            "created_at": datetime.now(UTC).isoformat(),
            "mode": "oracle_contract_audit",
            "benchmark": dataset.manifest.as_dict(),
            "scenario_indices": list(indices),
            "task_limit": task_limit,
            "baseline_run_dir": (
                str(Path(baseline_run_dir).expanduser().resolve())
                if baseline_run_dir is not None
                else None
            ),
        }
        _write_json(artifact_dir / "manifest.json", manifest)
        _write_json(artifact_dir / "oracle-contract-audit.json", audit)
        (artifact_dir / "results.md").write_text(
            _render_audit_markdown(audit),
            encoding="utf-8",
        )
    return OracleContractAuditResult(
        run_id=run_id,
        status=audit["status"],
        artifact_dir=artifact_dir,
        audit=audit,
    )


def _audit_task(
    task: VehicleTask,
    *,
    history: str,
    tool_names: set[str],
) -> dict[str, Any]:
    quotes = _quoted_evidence(task.gold_memory)
    normalized_history = _normalize_text(history)
    linked_quotes = tuple(
        quote for quote in quotes if _normalize_text(quote) in normalized_history
    )
    unknown_tools = tuple(
        sorted({call.name for call in task.gold_calls if call.name not in tool_names})
    )
    call_signatures = [
        json.dumps(
            call.as_official(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for call in task.gold_calls
    ]
    return {
        "task_id": task.id,
        "scenario_index": task.scenario_index,
        "event_index": task.event_index,
        "reasoning_type": task.reasoning_type,
        "gold_memory_sha256": hashlib.sha256(
            task.gold_memory.encode("utf-8")
        ).hexdigest(),
        "gold_memory_present": bool(task.gold_memory.strip()),
        "gold_memory_line_count": len(task.gold_memory.splitlines()),
        "quoted_evidence_count": len(quotes),
        "history_linked_quote_count": len(linked_quotes),
        "all_quoted_evidence_linked": bool(quotes)
        and len(linked_quotes) == len(quotes),
        "gold_call_count": len(task.gold_calls),
        "gold_tool_names": list(dict.fromkeys(call.name for call in task.gold_calls)),
        "unknown_gold_tool_names": list(unknown_tools),
        "duplicate_gold_call_count": len(call_signatures)
        - len(set(call_signatures)),
        "gold_argument_leaf_count": sum(
            len(_argument_leaves(call.arguments)) for call in task.gold_calls
        ),
    }


def _parse_oracle_fact_key(
    payload: Any,
    task_id: str,
) -> OracleFactRecordKey:
    if not isinstance(payload, dict):
        raise ValueError(f"Oracle Fact key must be an object: {task_id}")
    entity_id = str(payload.get("entity_id", "")).strip()
    predicate = str(payload.get("predicate", "")).strip()
    if not entity_id or not predicate or "value" not in payload:
        raise ValueError(f"Oracle Fact key is incomplete: {task_id}")
    identity = payload.get("identity_conditions")
    applicability = payload.get("applicability")
    if identity is not None and not isinstance(identity, dict):
        raise ValueError(f"Oracle identity_conditions must be an object: {task_id}")
    if applicability is not None and not isinstance(applicability, dict):
        raise ValueError(f"Oracle applicability must be an object: {task_id}")
    return OracleFactRecordKey(
        entity_id=entity_id,
        predicate=predicate,
        value=payload["value"],
        identity_conditions=identity,
        applicability=applicability,
    )


def _insert_oracle_label(
    labels: dict[str, OracleRetrievalLabel],
    label: OracleRetrievalLabel,
) -> None:
    if not label.task_id:
        raise ValueError("Oracle Retrieval task_id cannot be empty")
    if label.task_id in labels:
        raise ValueError(
            f"Duplicate Oracle Retrieval annotation: {label.task_id}"
        )
    labels[label.task_id] = label


def _insert_oracle_gate_label(
    labels: dict[str, OracleGateLabel],
    label: OracleGateLabel,
) -> None:
    if not label.task_id:
        raise ValueError("Oracle Gate task_id cannot be empty")
    if label.task_id in labels:
        raise ValueError(f"Duplicate Oracle Gate annotation: {label.task_id}")
    labels[label.task_id] = label


def _stored_fact_candidate(
    candidate: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
) -> FactMemoryCandidate:
    return FactMemoryCandidate(
        entity_id=str(candidate["entity_id"]),
        predicate=str(candidate["predicate"]),
        value=candidate.get("value"),
        identity_conditions=dict(candidate.get("identity_conditions", {})),
        applicability=dict(candidate.get("applicability", {})),
        capability_hints=tuple(
            str(item) for item in candidate.get("capability_hints", ())
        ),
        memory_type=str(candidate["memory_type"]),
        confidence=float(candidate["confidence"]),
        evidence=tuple(
            MemoryEvidence(
                message_id=int(item["message_id"]),
                quote=str(item["quote"]),
                start_char=item.get("start_char"),
                end_char=item.get("end_char"),
            )
            for item in evidence
        ),
        directive=str(candidate.get("directive", "UPSERT")),
        reason=str(candidate.get("reason", "")),
    )


def _summarize_task_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    quote_tasks = [row for row in rows if row["quoted_evidence_count"] > 0]
    return {
        "tasks": len(rows),
        "missing_gold_memory": sum(
            not row["gold_memory_present"] for row in rows
        ),
        "missing_gold_calls": sum(row["gold_call_count"] == 0 for row in rows),
        "unknown_gold_tool_calls": sum(
            len(row["unknown_gold_tool_names"]) for row in rows
        ),
        "duplicate_gold_calls": sum(
            row["duplicate_gold_call_count"] for row in rows
        ),
        "gold_memory_lines": sum(row["gold_memory_line_count"] for row in rows),
        "gold_tool_calls": sum(row["gold_call_count"] for row in rows),
        "gold_argument_leaves": sum(
            row["gold_argument_leaf_count"] for row in rows
        ),
        "tasks_with_quoted_evidence": len(quote_tasks),
        "tasks_with_all_quotes_linked": sum(
            row["all_quoted_evidence_linked"] for row in quote_tasks
        ),
        "quoted_evidence": sum(row["quoted_evidence_count"] for row in rows),
        "history_linked_quotes": sum(
            row["history_linked_quote_count"] for row in rows
        ),
        "fact_annotation_tasks": 0,
        "fact_annotation_status": "required",
    }


def _audit_baseline(
    dataset: VehicleBenchmarkDataset,
    *,
    scenario_indices: Sequence[int],
    baseline_run_dir: Path | None,
) -> dict[str, Any]:
    if baseline_run_dir is None:
        return {
            "provided": False,
            "compatible": True,
            "run_id": None,
            "fact_profile": None,
        }
    root = Path(baseline_run_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    metrics = _read_json(root / "metrics.json")
    config = manifest.get("config", {})
    benchmark = config.get("benchmark", {})
    configured_indices = tuple(config.get("scenario_indices", ()))
    profiles = tuple(config.get("profiles", ()))
    fact_profile = metrics.get("profiles", {}).get("cloud_fact_patch")
    issues = []
    if benchmark.get("dataset_sha256") != dataset.manifest.dataset_sha256:
        issues.append("dataset_sha256_mismatch")
    if benchmark.get("tool_schema_sha256") != dataset.manifest.tool_schema_sha256:
        issues.append("tool_schema_sha256_mismatch")
    if configured_indices and configured_indices != tuple(scenario_indices):
        issues.append("scenario_indices_mismatch")
    if "cloud_fact_patch" not in profiles or fact_profile is None:
        issues.append("cloud_fact_patch_missing")
    return {
        "provided": True,
        "compatible": not issues,
        "run_id": manifest.get("run_id"),
        "issues": issues,
        "fact_profile": fact_profile,
    }


def _quoted_evidence(value: str) -> tuple[str, ...]:
    found: list[str] = []
    for pattern in _QUOTED_EVIDENCE_PATTERNS:
        found.extend(match.strip() for match in pattern.findall(value) if match.strip())
    return tuple(dict.fromkeys(found))


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = (
        normalized.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return " ".join(normalized.casefold().split())


def _argument_leaves(value: Any) -> tuple[Any, ...]:
    if isinstance(value, dict):
        return tuple(
            leaf for item in value.values() for leaf in _argument_leaves(item)
        )
    if isinstance(value, (list, tuple)):
        return tuple(leaf for item in value for leaf in _argument_leaves(item))
    return (value,)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _render_audit_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    baseline = audit["baseline"]
    baseline_lines = (
        [
            f"- Baseline run: `{baseline['run_id']}`",
            f"- Baseline compatible: `{baseline['compatible']}`",
        ]
        if baseline["provided"]
        else ["- Baseline run: not provided"]
    )
    return "\n".join(
        [
            "# VehicleMemBench Oracle Contract Audit",
            "",
            f"- Status: `{audit['status']}`",
            f"- Scenarios: `{audit['scenario_indices']}`",
            f"- Tasks: `{summary['tasks']}`",
            f"- Missing Gold Memory: `{summary['missing_gold_memory']}`",
            f"- Missing Gold calls: `{summary['missing_gold_calls']}`",
            f"- Unknown Gold Tool calls: `{summary['unknown_gold_tool_calls']}`",
            f"- Duplicate Gold calls: `{summary['duplicate_gold_calls']}`",
            f"- History-linked quoted evidence: "
            f"`{summary['history_linked_quotes']}/{summary['quoted_evidence']}`",
            f"- Fact annotations: `{summary['fact_annotation_status']}`",
            *baseline_lines,
            "",
            "Extraction, Structure, Gate, and record-level Retrieval Oracles "
            "remain blocked until reviewed Fact annotations are available.",
            "",
        ]
    )
