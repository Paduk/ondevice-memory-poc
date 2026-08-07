from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from palmclaw_ubuntu.contracts import EmbeddingModel
from palmclaw_ubuntu.models import MemoryMessage, ModelUsage
from palmclaw_ubuntu.tokens import TokenCounter
from palmclaw_ubuntu.vehicle_bench.dataset import VehicleBenchmarkDataset
from palmclaw_ubuntu.vehicle_bench.memory import (
    VehicleHistoryEntry,
    parse_vehicle_history,
)
from palmclaw_ubuntu.vehicle_bench.vehicle_fact_ontology import (
    VehicleFactOntology,
    VehicleFactOntologyMatch,
    VehicleFactOntologyMatcher,
    load_vehicle_fact_ontology_v1,
)

VEHICLE_ONTOLOGY_MATCHER_LABEL_VERSION = (
    "vehiclemembench-ontology-matcher-labels-v1"
)
VEHICLE_ONTOLOGY_MATCHER_LABEL_RESOURCE = (
    "vehiclemembench_ontology_matcher_s6_v1.json"
)


@dataclass(frozen=True)
class VehicleOntologyMatcherLabel:
    task_id: str
    evidence_quote: str
    expected_tool_name: str


@dataclass(frozen=True)
class VehicleOntologyMatcherLabels:
    version: str
    dataset_sha256: str
    tool_schema_sha256: str
    ontology_sha256: str
    scenario_index: int
    labels: tuple[VehicleOntologyMatcherLabel, ...]
    source_path: Path


@dataclass(frozen=True)
class VehicleOntologyMatcherEvaluationResult:
    run_id: str
    status: str
    artifact_dir: Path | None
    metrics: Mapping[str, Any]
    cases: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "artifact_dir": (
                str(self.artifact_dir) if self.artifact_dir is not None else None
            ),
            "metrics": dict(self.metrics),
        }


def default_vehicle_ontology_matcher_labels() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "evaluation_datasets"
        / VEHICLE_ONTOLOGY_MATCHER_LABEL_RESOURCE
    )


def load_vehicle_ontology_matcher_labels(
    path: Path | str,
    *,
    dataset: VehicleBenchmarkDataset,
    ontology: VehicleFactOntology,
    scenario_index: int,
) -> VehicleOntologyMatcherLabels:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Ontology matcher labels must contain an object")
    version = str(payload.get("version", ""))
    if version != VEHICLE_ONTOLOGY_MATCHER_LABEL_VERSION:
        raise ValueError(f"Unsupported ontology matcher label version: {version}")
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise ValueError("Ontology matcher benchmark metadata must be an object")
    dataset_sha256 = str(benchmark.get("dataset_sha256", ""))
    tool_schema_sha256 = str(benchmark.get("tool_schema_sha256", ""))
    ontology_sha256 = str(benchmark.get("ontology_sha256", ""))
    if dataset_sha256 != dataset.manifest.dataset_sha256:
        raise ValueError("Ontology matcher dataset hash mismatch")
    if tool_schema_sha256 != dataset.manifest.tool_schema_sha256:
        raise ValueError("Ontology matcher Tool schema hash mismatch")
    if ontology_sha256 != ontology.ontology_sha256:
        raise ValueError("Ontology matcher ontology hash mismatch")
    labeled_scenario = int(payload.get("scenario_index", 0))
    if labeled_scenario != scenario_index:
        raise ValueError(
            "Ontology matcher label scenario mismatch: "
            f"{labeled_scenario} != {scenario_index}"
        )
    scenario = dataset.scenario(scenario_index)
    tasks = {task.id: task for task in scenario.tasks}
    entries = parse_vehicle_history(scenario.history_path)
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Ontology matcher cases must be a non-empty list")
    labels = []
    seen_tasks = set()
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise ValueError("Ontology matcher case must be an object")
        label = VehicleOntologyMatcherLabel(
            task_id=str(raw.get("task_id", "")).strip(),
            evidence_quote=str(raw.get("evidence_quote", "")).strip(),
            expected_tool_name=str(raw.get("expected_tool_name", "")).strip(),
        )
        task = tasks.get(label.task_id)
        if task is None or label.task_id in seen_tasks:
            raise ValueError(
                f"Invalid or duplicate ontology matcher task: {label.task_id}"
            )
        if sum(label.evidence_quote in entry.raw for entry in entries) != 1:
            raise ValueError(
                "Ontology matcher evidence must occur in exactly one history "
                f"line: {label.task_id}"
            )
        if label.expected_tool_name not in {
            call.name for call in task.gold_calls
        }:
            raise ValueError(
                "Ontology matcher expected Tool is not a task Gold Tool: "
                f"{label.task_id}:{label.expected_tool_name}"
            )
        if ontology.binding(label.expected_tool_name) is None:
            raise ValueError(
                "Ontology matcher expected Tool is absent from ontology: "
                f"{label.expected_tool_name}"
            )
        labels.append(label)
        seen_tasks.add(label.task_id)
    if seen_tasks != set(tasks):
        raise ValueError(
            "Ontology matcher labels must cover the scenario exactly: "
            f"missing={sorted(set(tasks) - seen_tasks)}"
        )
    return VehicleOntologyMatcherLabels(
        version=version,
        dataset_sha256=dataset_sha256,
        tool_schema_sha256=tool_schema_sha256,
        ontology_sha256=ontology_sha256,
        scenario_index=scenario_index,
        labels=tuple(labels),
        source_path=source,
    )


def run_vehicle_ontology_matcher_evaluation(
    dataset: VehicleBenchmarkDataset,
    *,
    embedding_model: EmbeddingModel,
    scenario_index: int,
    labels_path: Path | str,
    batch_turn_limit: int = 32,
    batch_token_limit: int = 4_096,
    output_root: Path | None = None,
    ontology: VehicleFactOntology | None = None,
) -> VehicleOntologyMatcherEvaluationResult:
    if batch_turn_limit < 1:
        raise ValueError("Matcher batch turn limit must be positive")
    if batch_token_limit < 256:
        raise ValueError("Matcher batch token limit must be at least 256")
    resolved_ontology = ontology or load_vehicle_fact_ontology_v1(
        dataset.tool_schemas
    )
    labels = load_vehicle_ontology_matcher_labels(
        labels_path,
        dataset=dataset,
        ontology=resolved_ontology,
        scenario_index=scenario_index,
    )
    entries = parse_vehicle_history(dataset.scenario(scenario_index).history_path)
    batches = _fact_matcher_batches(
        entries,
        turn_limit=batch_turn_limit,
        token_limit=batch_token_limit,
    )
    entry_by_task: dict[str, VehicleHistoryEntry] = {}
    batch_by_line = {
        entry.line_number: batch_index
        for batch_index, batch in enumerate(batches)
        for entry in batch
    }
    for label in labels.labels:
        entry_by_task[label.task_id] = next(
            entry for entry in entries if label.evidence_quote in entry.raw
        )

    matcher = VehicleFactOntologyMatcher(
        resolved_ontology,
        embedding_model=embedding_model,
        top_k_per_message=2,
        max_candidates=8,
    )
    usage = ModelUsage()
    semantic_errors = []
    single_results = {}
    for label in labels.labels:
        entry = entry_by_task[label.task_id]
        result = matcher.match((_entry_message(entry),))
        single_results[label.task_id] = result.matches
        usage = _sum_usage(usage, result.usage)
        if error := result.metadata.get("semantic_error"):
            semantic_errors.append(str(error))

    relevant_batch_indexes = sorted(
        {batch_by_line[entry.line_number] for entry in entry_by_task.values()}
    )
    batch_results = {}
    for batch_index in relevant_batch_indexes:
        result = matcher.match(
            tuple(_entry_message(entry) for entry in batches[batch_index])
        )
        batch_results[batch_index] = result.matches
        usage = _sum_usage(usage, result.usage)
        if error := result.metadata.get("semantic_error"):
            semantic_errors.append(str(error))

    task_map = {
        task.id: task for task in dataset.scenario(scenario_index).tasks
    }
    cases = []
    for label in labels.labels:
        entry = entry_by_task[label.task_id]
        batch_index = batch_by_line[entry.line_number]
        binding = resolved_ontology.binding(label.expected_tool_name)
        assert binding is not None
        single_matches = single_results[label.task_id]
        batch_matches = batch_results[batch_index]
        single_ids = tuple(item.capability_id for item in single_matches)
        batch_ids = tuple(item.capability_id for item in batch_matches)
        single_rank = (
            single_ids.index(binding.capability_id) + 1
            if binding.capability_id in single_ids
            else None
        )
        batch_rank = (
            batch_ids.index(binding.capability_id) + 1
            if binding.capability_id in batch_ids
            else None
        )
        expected_match = next(
            (
                item
                for item in single_matches
                if item.capability_id == binding.capability_id
            ),
            None,
        )
        cases.append(
            {
                "task_id": label.task_id,
                "scenario_index": scenario_index,
                "reasoning_type": task_map[label.task_id].reasoning_type,
                "history_line_number": entry.line_number,
                "batch_index": batch_index,
                "evidence_quote": label.evidence_quote,
                "expected_tool_name": label.expected_tool_name,
                "expected_capability_id": binding.capability_id,
                "expected_target": binding.target,
                "single_top2_capabilities": list(single_ids),
                "single_rank": single_rank,
                "batch_top8_capabilities": list(batch_ids),
                "batch_rank": batch_rank,
                "top1_hit": single_rank == 1,
                "top2_hit": single_rank is not None,
                "batch_top8_hit": batch_rank is not None,
                "target_binding_in_top2": _target_binding_present(
                    expected_match,
                    binding.target,
                ),
                "single_candidates": [
                    _match_diagnostic(item) for item in single_matches
                ],
            }
        )

    metrics = _matcher_metrics(
        cases,
        embedding_model=embedding_model,
        usage=usage,
        scenario_batch_count=len(batches),
        relevant_batch_count=len(relevant_batch_indexes),
        semantic_errors=semantic_errors,
    )
    run_id = str(uuid.uuid4())
    artifact_dir = None
    status = "completed" if not semantic_errors else "failed"
    if output_root is not None:
        artifact_dir = Path(output_root).expanduser().resolve() / run_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": run_id,
            "status": status,
            "created_at": datetime.now(UTC).isoformat(),
            "mode": "vehicle_ontology_matcher_only",
            "benchmark": dataset.manifest.as_dict(),
            "scenario_index": scenario_index,
            "labels": {
                "version": labels.version,
                "sha256": hashlib.sha256(labels.source_path.read_bytes()).hexdigest(),
            },
            "ontology": {
                "version": resolved_ontology.schema_version,
                "sha256": resolved_ontology.ontology_sha256,
            },
            "matcher": {
                "top_k_per_message": 2,
                "max_candidates_per_batch": 8,
                "semantic_weight": 0.65,
                "batch_turn_limit": batch_turn_limit,
                "batch_token_limit": batch_token_limit,
            },
            "generation_model_calls": 0,
        }
        _write_json(artifact_dir / "manifest.json", manifest)
        _write_json(artifact_dir / "metrics.json", metrics)
        (artifact_dir / "cases.jsonl").write_text(
            "".join(
                json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n"
                for case in cases
            ),
            encoding="utf-8",
        )
        (artifact_dir / "results.md").write_text(
            _render_results(metrics, cases),
            encoding="utf-8",
        )
    return VehicleOntologyMatcherEvaluationResult(
        run_id=run_id,
        status=status,
        artifact_dir=artifact_dir,
        metrics=metrics,
        cases=tuple(cases),
    )


def _fact_matcher_batches(
    entries: Sequence[VehicleHistoryEntry],
    *,
    turn_limit: int,
    token_limit: int,
) -> tuple[tuple[VehicleHistoryEntry, ...], ...]:
    counter = TokenCounter()
    batches = []
    current = []
    current_tokens = 0
    for entry in entries:
        token_count = counter.count(entry.raw) + 8
        if current and (
            len(current) >= turn_limit
            or current_tokens + token_count > token_limit
        ):
            batches.append(tuple(current))
            current = []
            current_tokens = 0
        current.append(entry)
        current_tokens += token_count
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _entry_message(entry: VehicleHistoryEntry) -> MemoryMessage:
    return MemoryMessage(id=entry.line_number, role="user", content=entry.raw)


def _target_binding_present(
    match: VehicleFactOntologyMatch | None,
    target: str,
) -> bool:
    if match is None:
        return False
    bindings = match.prompt_payload.get("bindings", ())
    return any(
        isinstance(item, Mapping) and item.get("target") == target
        for item in bindings
    )


def _match_diagnostic(match: VehicleFactOntologyMatch) -> dict[str, Any]:
    return {
        "capability_id": match.capability_id,
        "score": match.score,
        "lexical_score": match.lexical_score,
        "semantic_score": match.semantic_score,
        "matched_terms": list(match.matched_terms),
    }


def _matcher_metrics(
    cases: Sequence[Mapping[str, Any]],
    *,
    embedding_model: EmbeddingModel,
    usage: ModelUsage,
    scenario_batch_count: int,
    relevant_batch_count: int,
    semantic_errors: Sequence[str],
) -> dict[str, Any]:
    count = len(cases)
    top1 = sum(bool(case["top1_hit"]) for case in cases)
    top2 = sum(bool(case["top2_hit"]) for case in cases)
    batch_top8 = sum(bool(case["batch_top8_hit"]) for case in cases)
    target = sum(bool(case["target_binding_in_top2"]) for case in cases)
    return {
        "case_count": count,
        "top1_hits": top1,
        "top1_recall": top1 / count,
        "top2_hits": top2,
        "top2_recall": top2 / count,
        "mrr_at_2": fmean(
            1 / int(case["single_rank"])
            if case["single_rank"] is not None
            else 0.0
            for case in cases
        ),
        "batch_top8_hits": batch_top8,
        "batch_top8_recall": batch_top8 / count,
        "target_binding_top2_hits": target,
        "target_binding_top2_coverage": target / count,
        "scenario_batch_count": scenario_batch_count,
        "evaluated_batch_count": relevant_batch_count,
        "embedding_call_count": count + relevant_batch_count,
        "embedding_model_id": embedding_model.model_id,
        "embedding_dimensions": embedding_model.dimensions,
        "embedding_usage": usage.as_dict(),
        "semantic_error_count": len(semantic_errors),
        "semantic_errors": list(dict.fromkeys(semantic_errors)),
        "generation_model_calls": 0,
    }


def _sum_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
    return ModelUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        cached_tokens=left.cached_tokens + right.cached_tokens,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _render_results(
    metrics: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> str:
    misses = [case for case in cases if not case["batch_top8_hit"]]
    lines = [
        "# VehicleMemBench Ontology Matcher-only Evaluation",
        "",
        f"- Evidence Top-1: `{metrics['top1_hits']}/{metrics['case_count']}`",
        f"- Evidence Top-2: `{metrics['top2_hits']}/{metrics['case_count']}`",
        f"- Actual-batch Top-8: "
        f"`{metrics['batch_top8_hits']}/{metrics['case_count']}`",
        f"- Target binding in Top-2: "
        f"`{metrics['target_binding_top2_hits']}/{metrics['case_count']}`",
        f"- Generation LLM calls: `{metrics['generation_model_calls']}`",
        "",
        "## Batch Top-8 misses",
        "",
    ]
    if not misses:
        lines.append("None.")
    else:
        lines.extend(
            f"- `{case['task_id']}` expected "
            f"`{case['expected_capability_id']}`; selected "
            f"`{case['batch_top8_capabilities']}`"
            for case in misses
        )
    return "\n".join(lines) + "\n"
