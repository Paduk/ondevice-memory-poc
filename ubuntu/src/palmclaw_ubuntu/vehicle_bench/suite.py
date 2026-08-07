from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.contracts import AgentModel
from palmclaw_ubuntu.privacy import (
    detect_sensitive_spans,
    redact_data_for_cloud,
)
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.vehicle_bench.dataset import VehicleBenchmarkDataset
from palmclaw_ubuntu.vehicle_bench.memory import VehicleMemorySnapshot
from palmclaw_ubuntu.vehicle_bench.oracle import (
    OracleGateAnnotations,
    OracleRetrievalAnnotations,
    OracleStageFactAnnotations,
)
from palmclaw_ubuntu.vehicle_bench.runner import (
    run_agent_evaluation,
    vehicle_agent_metrics,
    write_vehicle_report_artifacts,
)


@dataclass(frozen=True)
class VehicleSuiteResult:
    run_id: str
    status: str
    scenario_indices: tuple[int, ...]
    profiles: tuple[str, ...]
    metrics: dict[str, Any]
    artifact_dir: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "scenario_indices": list(self.scenario_indices),
            "profiles": list(self.profiles),
            "metrics": self.metrics,
            "artifact_dir": str(self.artifact_dir),
        }


def run_vehicle_evaluation_suite(
    dataset: VehicleBenchmarkDataset,
    *,
    repository: SQLiteRepository,
    agent_model: AgentModel,
    profile_names: tuple[str, ...],
    scenario_indices: Sequence[int],
    snapshot_factory: Callable[[int], VehicleMemorySnapshot | None],
    output_root: Path,
    task_limit: int = 10,
    max_tool_rounds: int = 10,
    tool_timeout_seconds: float = 5,
    max_tool_result_chars: int = 20_000,
    resume_run_id: str | None = None,
    agent_input_cost_per_million: float = 0.0,
    agent_output_cost_per_million: float = 0.0,
    memory_input_cost_per_million: float = 0.0,
    memory_output_cost_per_million: float = 0.0,
    embedding_input_cost_per_million: float = 0.0,
    memory_configuration: dict[str, Any] | None = None,
    oracle_retrieval_annotations: OracleRetrievalAnnotations | None = None,
    oracle_gate_annotations: OracleGateAnnotations | None = None,
    oracle_stage_fact_annotations: OracleStageFactAnnotations | None = None,
) -> VehicleSuiteResult:
    scenarios = tuple(dict.fromkeys(int(item) for item in scenario_indices))
    if not scenarios:
        raise ValueError("At least one VehicleMemBench scenario is required")
    for scenario_index in scenarios:
        dataset.scenario(scenario_index)
    profiles = tuple(dict.fromkeys(profile_names))
    output = Path(output_root).expanduser().resolve()
    config = _suite_config(
        dataset,
        agent_model=agent_model,
        profiles=profiles,
        scenarios=scenarios,
        task_limit=task_limit,
        memory_configuration=memory_configuration,
        oracle_configuration=(
            {
                **(
                    {"retrieval": oracle_retrieval_annotations.manifest()}
                    if oracle_retrieval_annotations is not None
                    else {}
                ),
                **(
                    {"gate": oracle_gate_annotations.manifest()}
                    if oracle_gate_annotations is not None
                    else {}
                ),
                **(
                    {"stage_facts": oracle_stage_fact_annotations.manifest()}
                    if oracle_stage_fact_annotations is not None
                    else {}
                ),
            }
            if (
                oracle_retrieval_annotations is not None
                or oracle_gate_annotations is not None
                or oracle_stage_fact_annotations is not None
            )
            else None
        ),
        cost_rates={
            "agent_input": agent_input_cost_per_million,
            "agent_output": agent_output_cost_per_million,
            "memory_input": memory_input_cost_per_million,
            "memory_output": memory_output_cost_per_million,
            "embedding_input": embedding_input_cost_per_million,
        },
    )

    refresh_completed = False
    if resume_run_id is None:
        run_id = repository.begin_evaluation(
            name=(
                "VehicleMemBench Fact-First R5"
                if any(
                    profile
                    in {
                        "cloud_fact_patch",
                        "cloud_schema_informed_fact_patch",
                        "cloud_fact_recursive_hybrid",
                        "cloud_recursive_assisted_fact_patch",
                        "cloud_schema_informed_recursive_assisted_fact_patch",
                        "cloud_joint_planned_fact_patch",
                    }
                    for profile in profiles
                )
                else (
                    "VehicleMemBench Schema Patch P5"
                    if "cloud_schema_patch" in profiles
                    else "VehicleMemBench Phase V4"
                )
            ),
            dataset_name="VehicleMemBench",
            dataset_version=(
                dataset.manifest.upstream_commit or "unversioned-checkout"
            ),
            dataset_sha256=dataset.manifest.dataset_sha256,
            execution_mode="live",
            profiles=profiles,
            seed=0,
            repetitions=1,
        )
        artifact_dir = output / run_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        (artifact_dir / "scenario-runs").mkdir()
        manifest = {
            "run_id": run_id,
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(),
            "config": config,
            "scenarios": {},
        }
        _write_json(artifact_dir / "manifest.json", manifest)
    else:
        run_id = _canonical_uuid(resume_run_id)
        artifact_dir = output / run_id
        manifest = _read_json(artifact_dir / "manifest.json")
        if manifest.get("config") != config:
            raise ValueError("Vehicle suite checkpoint configuration mismatch")
        detail = repository.evaluation_detail(run_id)
        if detail["run"]["status"] in {"completed", "completed_with_errors"}:
            refresh_completed = True

    try:
        for scenario_index in scenarios:
            key = str(scenario_index)
            entry = manifest["scenarios"].get(key)
            if entry is None:
                entry = {
                    "status": "pending",
                    "child_run_id": str(uuid.uuid4()),
                }
                manifest["scenarios"][key] = entry
                _write_json(artifact_dir / "manifest.json", manifest)
            if entry.get("status") == "completed":
                continue

            snapshot = snapshot_factory(scenario_index)
            try:
                calls_before = snapshot.provider_calls() if snapshot else []
                child_id = str(entry["child_run_id"])
                child_dir = artifact_dir / "scenario-runs" / child_id
                entry["status"] = "running"
                _write_json(artifact_dir / "manifest.json", manifest)
                result = run_agent_evaluation(
                    dataset,
                    agent_model=agent_model,
                    profile_names=profiles,
                    scenario_index=scenario_index,
                    task_limit=task_limit,
                    max_tool_rounds=max_tool_rounds,
                    tool_timeout_seconds=tool_timeout_seconds,
                    max_tool_result_chars=max_tool_result_chars,
                    output_root=artifact_dir / "scenario-runs",
                    memory_resolver=(
                        snapshot.resolve if snapshot is not None else None
                    ),
                    memory_manifest=(
                        snapshot.manifest if snapshot is not None else None
                    ),
                    oracle_retrieval_annotations=(
                        oracle_retrieval_annotations
                    ),
                    oracle_gate_annotations=oracle_gate_annotations,
                    oracle_stage_fact_annotations=(
                        oracle_stage_fact_annotations
                    ),
                    resume_run_id=child_id if child_dir.exists() else None,
                    new_run_id=child_id if not child_dir.exists() else None,
                    agent_input_cost_per_million=(agent_input_cost_per_million),
                    agent_output_cost_per_million=(agent_output_cost_per_million),
                )
                calls_after = snapshot.provider_calls() if snapshot else []
                entry.update(
                    {
                        "status": "completed",
                        "child_status": result.status,
                        "provider_usage": summarize_provider_calls(
                            calls_before,
                            calls_after,
                            memory_input_cost_per_million=(
                                memory_input_cost_per_million
                            ),
                            memory_output_cost_per_million=(
                                memory_output_cost_per_million
                            ),
                            embedding_input_cost_per_million=(
                                embedding_input_cost_per_million
                            ),
                        ),
                        "amem_usage": (
                            getattr(snapshot, "amem_usage", lambda: None)()
                            if snapshot is not None
                            else None
                        ),
                    }
                )
                _write_json(artifact_dir / "manifest.json", manifest)
            finally:
                if snapshot is not None:
                    snapshot.close()
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _write_json(artifact_dir / "manifest.json", manifest)
        raise

    records = _suite_records(artifact_dir, manifest, scenarios)
    metrics = vehicle_agent_metrics(
        records,
        profiles,
        agent_input_cost_per_million=agent_input_cost_per_million,
        agent_output_cost_per_million=agent_output_cost_per_million,
    )
    provider_usage = _sum_provider_usage(
        [
            manifest["scenarios"][str(index)].get("provider_usage", {})
            for index in scenarios
        ]
    )
    amem_usage = _sum_amem_usage(
        [
            manifest["scenarios"][str(index)].get("amem_usage")
            for index in scenarios
        ]
    )
    metrics.update(
        {
            "scenario_indices": list(scenarios),
            "scenario_count": len(scenarios),
            "unique_task_count": len({str(record["task_id"]) for record in records}),
            "provider_usage": provider_usage,
            "dataset_privacy_audit": _dataset_privacy_audit(dataset),
            "estimated_total_cost_usd": round(
                provider_usage["estimated_cost_usd"]
                + (
                    float(amem_usage["build_estimated_cost_usd"])
                    if amem_usage is not None
                    else 0.0
                )
                + sum(
                    float(values["estimated_agent_cost_usd"])
                    for values in metrics["profiles"].values()
                ),
                8,
            ),
        }
    )
    if amem_usage is not None:
        metrics["amem_usage"] = amem_usage
    status = "completed" if not metrics["failed_tasks"] else "completed_with_errors"
    write_vehicle_report_artifacts(
        artifact_dir,
        metrics=metrics,
        records=records,
    )
    if refresh_completed:
        repository.refresh_evaluation_report(
            run_id,
            metrics=metrics,
            artifact_dir=str(artifact_dir),
        )
    else:
        for record in records:
            repository.record_evaluation_case(
                evaluation_run_id=run_id,
                profile=str(record["profile"]),
                case_id=str(record["task_id"]),
                repetition=0,
                status=str(record["status"]),
                latency_ms=int(record.get("usage", {}).get("model_latency_ms", 0)),
                expected={
                    "reference_calls": record.get("reference_calls", []),
                    "reference_state": record.get("scorer_input", {}).get(
                        "reference_state"
                    ),
                },
                actual={
                    "predicted_calls": record.get("predicted_calls", []),
                    "predicted_state": record.get("scorer_input", {}).get(
                        "predicted_state"
                    ),
                    "diagnostics": record.get("diagnostics", {}),
                },
                metrics={
                    "score": record.get("score"),
                    "argument_exact_match": record.get(
                        "argument_exact_match"
                    ),
                    "retrieval_quality": record.get("retrieval_quality"),
                    "usage": record.get("usage"),
                    "reasoning_type": record.get("reasoning_type"),
                },
                error=record.get("error"),
            )
        repository.complete_evaluation(
            run_id,
            metrics=metrics,
            artifact_dir=str(artifact_dir),
            with_errors=bool(metrics["failed_tasks"]),
        )
    manifest["status"] = status
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_json(artifact_dir / "manifest.json", manifest)
    return VehicleSuiteResult(
        run_id=run_id,
        status=status,
        scenario_indices=scenarios,
        profiles=profiles,
        metrics=metrics,
        artifact_dir=artifact_dir,
    )


def summarize_provider_calls(
    calls_before: Sequence[dict[str, Any]],
    calls_after: Sequence[dict[str, Any]],
    *,
    memory_input_cost_per_million: float = 0.0,
    memory_output_cost_per_million: float = 0.0,
    embedding_input_cost_per_million: float = 0.0,
) -> dict[str, Any]:
    before_ids = {str(call["id"]) for call in calls_before}
    generation = [
        call
        for call in calls_after
        if call.get("consolidation_run_id") is not None
        or call.get("memory_patch_run_id") is not None
    ]
    retrieval = [
        call
        for call in calls_after
        if str(call["id"]) not in before_ids
        and call.get("consolidation_run_id") is None
        and call.get("memory_patch_run_id") is None
    ]
    result = {
        "generation": _provider_call_totals(
            generation,
            memory_input_cost_per_million=memory_input_cost_per_million,
            memory_output_cost_per_million=memory_output_cost_per_million,
            embedding_input_cost_per_million=(embedding_input_cost_per_million),
        ),
        "retrieval": _provider_call_totals(
            retrieval,
            memory_input_cost_per_million=memory_input_cost_per_million,
            memory_output_cost_per_million=memory_output_cost_per_million,
            embedding_input_cost_per_million=(embedding_input_cost_per_million),
        ),
    }
    result["estimated_cost_usd"] = round(
        float(result["generation"]["estimated_cost_usd"])
        + float(result["retrieval"]["estimated_cost_usd"]),
        8,
    )
    return result


def _provider_call_totals(
    calls: Sequence[dict[str, Any]],
    *,
    memory_input_cost_per_million: float,
    memory_output_cost_per_million: float,
    embedding_input_cost_per_million: float,
) -> dict[str, Any]:
    roles: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    totals = {
        "calls": len(calls),
        "failed_calls": 0,
        "latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "cloud_transmitted_characters": 0,
        "cloud_detected_sensitive_spans": 0,
        "cloud_redacted_sensitive_spans": 0,
        "cloud_sensitive_characters_before": 0,
        "cloud_sensitive_characters_after": 0,
    }
    for call in calls:
        role = str(call.get("role", "unknown"))
        roles[role] += 1
        totals["failed_calls"] += int(bool(call.get("error")))
        totals["latency_ms"] += int(call.get("latency_ms", 0))
        usage = call.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        if role in {
            "embedding",
            "tool_memory_embedding",
            "tool_schema_embedding",
            "fact_memory_embedding",
            "fact_tool_schema_embedding",
            "amem_embedding",
            "amem_retrieval_embedding",
        }:
            totals["estimated_cost_usd"] += (
                input_tokens * embedding_input_cost_per_million / 1_000_000
            )
        else:
            totals["estimated_cost_usd"] += (
                input_tokens * memory_input_cost_per_million
                + output_tokens * memory_output_cost_per_million
            ) / 1_000_000
        privacy = call.get("metadata", {}).get("privacy", {})
        if privacy.get("destination") != "cloud":
            continue
        totals["cloud_transmitted_characters"] += int(privacy.get("output_chars", 0))
        totals["cloud_detected_sensitive_spans"] += int(
            privacy.get("detected_count", 0)
        )
        totals["cloud_redacted_sensitive_spans"] += int(
            privacy.get("redacted_count", 0)
        )
        totals["cloud_sensitive_characters_before"] += int(
            privacy.get("sensitive_chars_before", 0)
        )
        totals["cloud_sensitive_characters_after"] += int(
            privacy.get("sensitive_chars_after", 0)
        )
        categories.update(privacy.get("category_counts", {}))
    totals["estimated_cost_usd"] = round(
        float(totals["estimated_cost_usd"]),
        8,
    )
    totals["roles"] = dict(sorted(roles.items()))
    totals["privacy_categories"] = dict(sorted(categories.items()))
    before = int(totals["cloud_sensitive_characters_before"])
    after = int(totals["cloud_sensitive_characters_after"])
    totals["cloud_exposed_character_rate"] = after / before if before else 0.0
    return totals


def _sum_provider_usage(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sections = ("generation", "retrieval")
    result: dict[str, Any] = {}
    for section in sections:
        rows = [item.get(section, {}) for item in items]
        role_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        for row in rows:
            role_counts.update(row.get("roles", {}))
            category_counts.update(row.get("privacy_categories", {}))
        result[section] = {
            key: sum(float(row.get(key, 0)) for row in rows)
            for key in (
                "calls",
                "failed_calls",
                "latency_ms",
                "input_tokens",
                "output_tokens",
                "estimated_cost_usd",
                "cloud_transmitted_characters",
                "cloud_detected_sensitive_spans",
                "cloud_redacted_sensitive_spans",
                "cloud_sensitive_characters_before",
                "cloud_sensitive_characters_after",
            )
        }
        result[section]["roles"] = dict(sorted(role_counts.items()))
        result[section]["privacy_categories"] = dict(sorted(category_counts.items()))
        before = result[section]["cloud_sensitive_characters_before"]
        after = result[section]["cloud_sensitive_characters_after"]
        result[section]["cloud_exposed_character_rate"] = (
            after / before if before else 0.0
        )
    result["estimated_cost_usd"] = round(
        sum(float(result[section]["estimated_cost_usd"]) for section in sections),
        8,
    )
    return result


def _sum_amem_usage(
    items: Sequence[dict[str, Any] | None],
) -> dict[str, Any] | None:
    present = [item for item in items if item is not None]
    if not present:
        return None
    included = [
        item
        for item in present
        if item.get("formal_aggregate_included", False)
    ]
    generation_usage = {
        key: sum(
            int(
                item.get("generation", {})
                .get("usage", {})
                .get(key, 0)
            )
            for item in included
        )
        for key in (
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "total_tokens",
        )
    }
    provider_roles: Counter[str] = Counter()
    for item in included:
        provider_roles.update(item.get("provider", {}).get("roles", {}))
    return {
        "scenario_count": len(included),
        "excluded_partial_scenario_count": len(present) - len(included),
        "generation_call_count": sum(
            int(item.get("generation", {}).get("call_count", 0))
            for item in included
        ),
        "generation_usage": generation_usage,
        "embedding_count": sum(
            int(item.get("embedding_count", 0)) for item in included
        ),
        "note_count": sum(
            int(item.get("note_count", 0)) for item in included
        ),
        "version_count": sum(
            int(item.get("version_count", 0)) for item in included
        ),
        "link_count": sum(
            int(item.get("link_count", 0)) for item in included
        ),
        "retrieval_run_count": sum(
            int(item.get("retrieval", {}).get("run_count", 0))
            for item in included
        ),
        "retrieval_selected_count": sum(
            int(item.get("retrieval", {}).get("selected_count", 0))
            for item in included
        ),
        "retrieval_latency_ms": sum(
            int(item.get("retrieval", {}).get("latency_ms", 0))
            for item in included
        ),
        "provider": {
            "call_count": sum(
                int(item.get("provider", {}).get("call_count", 0))
                for item in included
            ),
            "failed_call_count": sum(
                int(
                    item.get("provider", {}).get(
                        "failed_call_count",
                        0,
                    )
                )
                for item in included
            ),
            "roles": dict(sorted(provider_roles.items())),
            "latency_ms": sum(
                int(item.get("provider", {}).get("latency_ms", 0))
                for item in included
            ),
            "estimated_cost_usd": round(
                sum(
                    float(
                        item.get("provider", {}).get(
                            "estimated_cost_usd",
                            0.0,
                        )
                    )
                    for item in included
                ),
                8,
            ),
        },
        "build_estimated_cost_usd": round(
            sum(
                float(item.get("build_estimated_cost_usd", 0.0))
                for item in included
            ),
            8,
        ),
    }


def _suite_records(
    artifact_dir: Path,
    manifest: dict[str, Any],
    scenarios: Sequence[int],
) -> list[dict[str, Any]]:
    records = []
    for scenario_index in scenarios:
        child_id = manifest["scenarios"][str(scenario_index)]["child_run_id"]
        cases_path = artifact_dir / "scenario-runs" / child_id / "cases.jsonl"
        records.extend(
            json.loads(line)
            for line in cases_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


def _suite_config(
    dataset: VehicleBenchmarkDataset,
    *,
    agent_model: AgentModel,
    profiles: tuple[str, ...],
    scenarios: tuple[int, ...],
    task_limit: int,
    memory_configuration: dict[str, Any] | None,
    oracle_configuration: dict[str, Any] | None,
    cost_rates: dict[str, float],
) -> dict[str, Any]:
    return {
        "benchmark": dataset.manifest.as_dict(),
        "agent": {
            "backend": agent_model.backend,
            "model_id": agent_model.model_id,
            "prompt_version": agent_model.prompt_version,
        },
        "profiles": list(profiles),
        "scenario_indices": list(scenarios),
        "task_limit": task_limit,
        "memory": memory_configuration,
        "oracle_retrieval": oracle_configuration,
        "cost_rates": cost_rates,
    }


def _dataset_privacy_audit(
    dataset: VehicleBenchmarkDataset,
) -> dict[str, Any]:
    history_categories: Counter[str] = Counter()
    gold_categories: Counter[str] = Counter()
    gold_risk_tasks = []
    for scenario in dataset.scenarios:
        history_categories.update(
            span.category
            for span in detect_sensitive_spans(
                scenario.history_path.read_text(encoding="utf-8")
            )
        )
        for task in scenario.tasks:
            spans = detect_sensitive_spans(task.gold_memory)
            if spans:
                gold_risk_tasks.append(
                    {
                        "task_id": task.id,
                        "detected_count": len(spans),
                        "category_counts": dict(
                            sorted(Counter(span.category for span in spans).items())
                        ),
                    }
                )
                gold_categories.update(span.category for span in spans)
    return {
        "history_category_counts": dict(sorted(history_categories.items())),
        "gold_memory_category_counts": dict(sorted(gold_categories.items())),
        "gold_redaction_risk_task_count": len(gold_risk_tasks),
        "gold_redaction_risk_tasks": gold_risk_tasks,
    }


def _write_json(path: Path, payload: Any) -> None:
    safe, _ = redact_data_for_cloud(
        payload,
        preserve_opaque_keys=(
            "encrypted_content",
            "run_id",
            "child_run_id",
            "consolidation_run_id",
            "consolidation_run_ids",
            "dataset_sha256",
            "cache_key",
        ),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Vehicle suite checkpoint is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_uuid(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"Invalid Vehicle suite run ID: {value}") from exc
    if str(parsed) != value:
        raise ValueError(f"Vehicle suite run ID must be canonical: {value}")
    return value
