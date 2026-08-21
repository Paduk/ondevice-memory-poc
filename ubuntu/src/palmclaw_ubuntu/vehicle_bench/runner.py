from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from palmclaw_ubuntu.contracts import AgentModel
from palmclaw_ubuntu.memory_tool_planning import (
    build_memory_tool_plan,
    choose_memory_tool_hint_result,
    render_joint_tool_candidate_rationale,
)
from palmclaw_ubuntu.models import ToolCall, ToolDefinition
from palmclaw_ubuntu.privacy import (
    detect_sensitive_spans,
    inspect_data_privacy,
    redact_data,
    redact_data_for_cloud,
)
from palmclaw_ubuntu.tool_memory_execution import (
    ToolMemoryExecutionHint,
    ToolMemoryHintResult,
    build_fact_memory_execution_hints,
    build_tool_memory_execution_hints,
    routed_tool_names,
)
from palmclaw_ubuntu.tool_memory_schema import build_tool_memory_ontology
from palmclaw_ubuntu.tools import ToolExecutionContext
from palmclaw_ubuntu.vehicle_bench.agent import (
    VEHICLE_AGENT_PROMPT_VERSION,
    VehicleAgentLoop,
    build_vehicle_system_prompt,
)
from palmclaw_ubuntu.vehicle_bench.dataset import (
    VehicleBenchmarkDataset,
    VehicleTask,
)
from palmclaw_ubuntu.vehicle_bench.memory import (
    VEHICLE_BINDING_ORACLE_PROFILES,
    VEHICLE_FACT_ORACLE_ANNOTATION_PROFILES,
    VEHICLE_FACT_PATCH_PROFILES,
    VEHICLE_FULL_ORACLE_PROFILES,
    VEHICLE_GATE_ORACLE_PROFILES,
    VEHICLE_MEMORY_PROFILES,
    VEHICLE_STAGE_FACT_ORACLE_PROFILES,
    VehicleMemoryContext,
)
from palmclaw_ubuntu.vehicle_bench.oracle import (
    OracleGateAnnotations,
    OracleRetrievalAnnotations,
    OracleStageFactAnnotations,
)
from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime
from palmclaw_ubuntu.vehicle_bench.tools import (
    build_dynamic_vehicle_tool_registry,
    build_oracle_vehicle_tool_registry,
    build_vehicle_tool_registry,
    vehicle_tool_definitions,
)
from palmclaw_ubuntu.vehicle_summary_wiki import (
    register_vehicle_wiki_tools,
)

_OPAQUE_UUID_PATTERN = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\b"
)

VEHICLE_BASELINE_PROFILES = ("no_memory", "gold_memory")
VEHICLE_ORACLE_PROFILES = (
    "oracle_tool_gold_memory",
    "oracle_tool_schema_patch",
    "oracle_tool_fact_patch",
    "oracle_full_pipeline_fact_patch",
)
VEHICLE_AGENT_PROFILES = (
    *VEHICLE_BASELINE_PROFILES,
    *VEHICLE_MEMORY_PROFILES,
    *(
        profile
        for profile in VEHICLE_ORACLE_PROFILES
        if profile not in VEHICLE_MEMORY_PROFILES
    ),
)


@dataclass(frozen=True)
class OfflineTaskResult:
    task_id: str
    reasoning_type: str
    gold_call_count: int
    exact_state_match: bool
    state_f1: float
    value_f1: float
    tool_f1: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reasoning_type": self.reasoning_type,
            "gold_call_count": self.gold_call_count,
            "exact_state_match": self.exact_state_match,
            "state_f1": self.state_f1,
            "value_f1": self.value_f1,
            "tool_f1": self.tool_f1,
        }


@dataclass(frozen=True)
class OfflineSmokeResult:
    scenario_index: int
    task_count: int
    passed: bool
    manifest: dict[str, Any]
    tasks: tuple[OfflineTaskResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "offline_gold_smoke",
            "scenario_index": self.scenario_index,
            "task_count": self.task_count,
            "passed": self.passed,
            "manifest": self.manifest,
            "tasks": [task.as_dict() for task in self.tasks],
        }


@dataclass(frozen=True)
class VehicleAgentEvaluationResult:
    run_id: str
    status: str
    scenario_index: int
    profiles: tuple[str, ...]
    metrics: dict[str, Any]
    tasks: tuple[dict[str, Any], ...]
    artifact_dir: Path | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "scenario_index": self.scenario_index,
            "profiles": list(self.profiles),
            "metrics": self.metrics,
            "artifact_dir": (
                str(self.artifact_dir) if self.artifact_dir is not None else None
            ),
        }


def run_offline_smoke(
    dataset: VehicleBenchmarkDataset,
    *,
    scenario_index: int = 1,
    task_limit: int | None = None,
    tool_timeout_seconds: float = 5,
) -> OfflineSmokeResult:
    scenario = dataset.scenario(scenario_index)
    tasks = scenario.tasks
    if task_limit is not None:
        if task_limit < 1:
            raise ValueError("task_limit must be at least 1")
        tasks = tasks[:task_limit]

    runtime = VehicleWorldRuntime(dataset.root)
    definitions = vehicle_tool_definitions(
        dataset.tool_schemas,
        timeout_seconds=tool_timeout_seconds,
    )
    results: list[OfflineTaskResult] = []
    for task in tasks:
        initial_world = runtime.create_world()
        reference_world = runtime.create_world()
        predicted_world = runtime.create_world()
        initial_state = runtime.state(initial_world)

        for call in task.gold_calls:
            reference_result = runtime.execute(
                reference_world,
                call.name,
                call.arguments,
            )
            if (
                isinstance(reference_result, dict)
                and reference_result.get("success") is False
            ):
                raise RuntimeError(
                    f"Reference tool failed for {task.id}: {reference_result}"
                )

        registry = build_vehicle_tool_registry(
            runtime,
            predicted_world,
            definitions,
        )
        predicted_calls = []
        for call_index, call in enumerate(task.gold_calls):
            result, _ = registry.execute(
                ToolCall(
                    id=f"{task.id}-gold-{call_index}",
                    name=call.name,
                    arguments=call.arguments,
                ),
                ToolExecutionContext(session_id=f"vehicle-{scenario_index}"),
            )
            if result.is_error:
                raise RuntimeError(
                    f"PalmClaw vehicle Tool failed for {task.id}: {result.content}"
                )
            predicted_calls.append(call)

        score = runtime.score(
            initial_state=initial_state,
            reference_state=runtime.state(reference_world),
            predicted_state=runtime.state(predicted_world),
            predicted_calls=predicted_calls,
            reference_calls=task.gold_calls,
        )
        results.append(
            OfflineTaskResult(
                task_id=task.id,
                reasoning_type=task.reasoning_type,
                gold_call_count=len(task.gold_calls),
                exact_state_match=score.exact_state_match,
                state_f1=float(score.state_score.get("f1_positive", 0)),
                value_f1=float(score.state_score.get("f1_change", 0)),
                tool_f1=float(score.tool_score.get("f1", 0)),
            )
        )

    passed = bool(results) and all(
        result.exact_state_match
        and result.state_f1 == 1
        and result.value_f1 == 1
        and result.tool_f1 == 1
        for result in results
    )
    return OfflineSmokeResult(
        scenario_index=scenario_index,
        task_count=len(results),
        passed=passed,
        manifest=dataset.manifest.as_dict(),
        tasks=tuple(results),
    )


def run_agent_evaluation(
    dataset: VehicleBenchmarkDataset,
    *,
    agent_model: AgentModel,
    profile_names: tuple[str, ...] = VEHICLE_BASELINE_PROFILES,
    scenario_index: int = 1,
    task_limit: int | None = None,
    max_tool_rounds: int = 10,
    tool_timeout_seconds: float = 5,
    max_tool_result_chars: int = 20_000,
    output_root: Path | None = None,
    memory_resolver: (Callable[..., VehicleMemoryContext] | None) = None,
    memory_manifest: dict[str, Any] | None = None,
    oracle_retrieval_annotations: OracleRetrievalAnnotations | None = None,
    oracle_gate_annotations: OracleGateAnnotations | None = None,
    oracle_stage_fact_annotations: OracleStageFactAnnotations | None = None,
    resume_run_id: str | None = None,
    new_run_id: str | None = None,
    agent_input_cost_per_million: float = 0.0,
    agent_output_cost_per_million: float = 0.0,
) -> VehicleAgentEvaluationResult:
    profiles = _validate_agent_profiles(profile_names)
    scenario = dataset.scenario(scenario_index)
    selected_tasks = scenario.tasks
    if task_limit is not None:
        if task_limit < 1:
            raise ValueError("task_limit must be at least 1")
        selected_tasks = selected_tasks[:task_limit]

    if resume_run_id is not None and new_run_id is not None:
        raise ValueError("resume_run_id and new_run_id are mutually exclusive")
    if resume_run_id is not None and output_root is None:
        raise ValueError("output_root is required to resume a vehicle evaluation")
    run_id = (
        _validated_run_id(resume_run_id or new_run_id)
        if resume_run_id or new_run_id
        else str(uuid.uuid4())
    )
    runtime = VehicleWorldRuntime(dataset.root)
    definitions = vehicle_tool_definitions(
        dataset.tool_schemas,
        timeout_seconds=tool_timeout_seconds,
    )
    artifact_dir = (
        Path(output_root).expanduser().resolve() / run_id
        if output_root is not None
        else None
    )
    task_ids = tuple(task.id for task in selected_tasks)
    effective_memory_manifest = {
        **dict(memory_manifest or {}),
        **(
            {"oracle_retrieval_annotations": (oracle_retrieval_annotations.manifest())}
            if oracle_retrieval_annotations is not None
            else {}
        ),
        **(
            {"oracle_gate_annotations": oracle_gate_annotations.manifest()}
            if oracle_gate_annotations is not None
            else {}
        ),
        **(
            {
                "oracle_stage_fact_annotations": (
                    oracle_stage_fact_annotations.manifest()
                )
            }
            if oracle_stage_fact_annotations is not None
            else {}
        ),
    }
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if artifact_dir is not None:
        if resume_run_id is not None:
            records_by_key = _load_agent_checkpoint(
                artifact_dir,
                dataset=dataset,
                scenario_index=scenario_index,
                profiles=profiles,
                task_ids=task_ids,
                agent_model=agent_model,
                memory_manifest=effective_memory_manifest,
            )
        else:
            _initialize_agent_artifacts(
                artifact_dir,
                run_id=run_id,
                dataset=dataset,
                scenario_index=scenario_index,
                profiles=profiles,
                task_ids=task_ids,
                agent_model=agent_model,
                memory_manifest=effective_memory_manifest,
            )

    try:
        for profile in profiles:
            for task in selected_tasks:
                key = (profile, task.id)
                previous = records_by_key.get(key)
                if previous is not None and previous.get("status") == "completed":
                    continue
                record = _run_agent_task(
                    runtime=runtime,
                    definitions=definitions,
                    agent_model=agent_model,
                    profile=profile,
                    task=task,
                    max_tool_rounds=max_tool_rounds,
                    max_tool_result_chars=max_tool_result_chars,
                    memory_resolver=memory_resolver,
                    oracle_retrieval_annotations=(oracle_retrieval_annotations),
                    oracle_gate_annotations=oracle_gate_annotations,
                    oracle_stage_fact_annotations=(oracle_stage_fact_annotations),
                )
                records_by_key[key] = record
                if artifact_dir is not None:
                    _append_agent_checkpoint(artifact_dir, record)
                    _write_agent_progress(
                        artifact_dir,
                        status="running",
                        metrics=vehicle_agent_metrics(
                            _ordered_records(
                                records_by_key,
                                profiles=profiles,
                                tasks=selected_tasks,
                            ),
                            profiles,
                            agent_input_cost_per_million=(agent_input_cost_per_million),
                            agent_output_cost_per_million=(
                                agent_output_cost_per_million
                            ),
                        ),
                    )
    except BaseException:
        if artifact_dir is not None:
            _write_agent_progress(
                artifact_dir,
                status="interrupted",
                metrics=vehicle_agent_metrics(
                    _ordered_records(
                        records_by_key,
                        profiles=profiles,
                        tasks=selected_tasks,
                    ),
                    profiles,
                    agent_input_cost_per_million=(agent_input_cost_per_million),
                    agent_output_cost_per_million=(agent_output_cost_per_million),
                ),
            )
        raise

    records = _ordered_records(
        records_by_key,
        profiles=profiles,
        tasks=selected_tasks,
    )
    metrics = vehicle_agent_metrics(
        records,
        profiles,
        agent_input_cost_per_million=agent_input_cost_per_million,
        agent_output_cost_per_million=agent_output_cost_per_million,
    )
    status = "completed" if metrics["failed_tasks"] == 0 else "partial"
    if artifact_dir is not None:
        _finalize_agent_artifacts(
            artifact_dir,
            status=status,
            metrics=metrics,
            records=records,
        )
    return VehicleAgentEvaluationResult(
        run_id=run_id,
        status=status,
        scenario_index=scenario_index,
        profiles=profiles,
        metrics=metrics,
        tasks=tuple(records),
        artifact_dir=artifact_dir,
    )


def _run_agent_task(
    *,
    runtime: VehicleWorldRuntime,
    definitions: Sequence[ToolDefinition],
    agent_model: AgentModel,
    profile: str,
    task: VehicleTask,
    max_tool_rounds: int,
    max_tool_result_chars: int,
    memory_resolver: Callable[..., VehicleMemoryContext] | None,
    oracle_retrieval_annotations: OracleRetrievalAnnotations | None,
    oracle_gate_annotations: OracleGateAnnotations | None,
    oracle_stage_fact_annotations: OracleStageFactAnnotations | None,
) -> dict[str, Any]:
    initial_world = runtime.create_world()
    reference_world = runtime.create_world()
    predicted_world = runtime.create_world()
    initial_state = runtime.state(initial_world)
    reference_outputs = []
    try:
        for call in task.gold_calls:
            output = runtime.execute(reference_world, call.name, call.arguments)
            reference_outputs.append(output)
            if isinstance(output, dict) and output.get("success") is False:
                raise RuntimeError(f"Reference Tool failed: {output}")

        oracle_tool_boundary = profile in VEHICLE_ORACLE_PROFILES
        oracle_tool_names = tuple(call.name for call in task.gold_calls)
        gold_memory = (
            task.gold_memory
            if profile in {"gold_memory", "oracle_tool_gold_memory"}
            else ""
        )
        retrieved_memory = ""
        memory_context = None
        hint_result = ToolMemoryHintResult(hints=(), rejections=())
        selector_tool_names: tuple[str, ...] = ()
        joint_candidate_rationale = ""
        if profile in VEHICLE_MEMORY_PROFILES:
            if memory_resolver is None:
                raise RuntimeError(
                    f"Vehicle memory resolver is required for profile {profile}"
                )
            if profile in VEHICLE_FULL_ORACLE_PROFILES:
                if (
                    oracle_stage_fact_annotations is None
                    or oracle_gate_annotations is None
                    or oracle_retrieval_annotations is None
                ):
                    raise RuntimeError(
                        "Oracle Full requires Stage Fact, Gate, and Retrieval "
                        "annotations"
                    )
                memory_context = memory_resolver(
                    profile,
                    task.query,
                    oracle_tool_names=(
                        oracle_tool_names
                        if profile == "oracle_full_pipeline_fact_patch"
                        else ()
                    ),
                    oracle_retrieval_label=(
                        oracle_retrieval_annotations.require(task.id)
                    ),
                )
            elif profile in VEHICLE_STAGE_FACT_ORACLE_PROFILES:
                if (
                    oracle_stage_fact_annotations is None
                    or oracle_retrieval_annotations is None
                ):
                    raise RuntimeError(
                        "Oracle Stage Fact and Retrieval annotations are required"
                    )
                stage = (
                    "structure"
                    if profile.startswith("oracle_structure_")
                    else "extraction"
                )
                memory_context = memory_resolver(
                    profile,
                    task.query,
                    oracle_retrieval_label=(
                        oracle_retrieval_annotations.require(task.id)
                    ),
                    oracle_stage_fact=(
                        oracle_stage_fact_annotations.fact_for_task(
                            task.id,
                            stage=stage,
                        )
                    ),
                )
            elif profile in VEHICLE_GATE_ORACLE_PROFILES:
                if oracle_gate_annotations is None:
                    raise RuntimeError("Oracle Gate annotations are required")
                kwargs: dict[str, Any] = {
                    "oracle_gate_label": oracle_gate_annotations.require(task.id)
                }
                if profile in VEHICLE_FACT_ORACLE_ANNOTATION_PROFILES:
                    if oracle_retrieval_annotations is None:
                        raise RuntimeError("Oracle Retrieval annotations are required")
                    kwargs["oracle_retrieval_label"] = (
                        oracle_retrieval_annotations.require(task.id)
                    )
                memory_context = memory_resolver(
                    profile,
                    task.query,
                    **kwargs,
                )
            elif profile in VEHICLE_FACT_ORACLE_ANNOTATION_PROFILES:
                if oracle_retrieval_annotations is None:
                    raise RuntimeError("Oracle Retrieval annotations are required")
                memory_context = memory_resolver(
                    profile,
                    task.query,
                    oracle_retrieval_label=(
                        oracle_retrieval_annotations.require(task.id)
                    ),
                )
            elif profile in {
                "oracle_tool_schema_patch",
                "oracle_tool_fact_patch",
            }:
                memory_context = memory_resolver(
                    profile,
                    task.query,
                    oracle_tool_names,
                )
            else:
                memory_context = memory_resolver(profile, task.query)
            retrieved_memory = memory_context.content
            if profile in VEHICLE_FACT_PATCH_PROFILES:
                if memory_context.fact_query_context is None:
                    raise RuntimeError("Fact retrieval query context is missing")
                fallback_hint_result = build_fact_memory_execution_hints(
                    memory_context.fact_records,
                    definitions,
                    memory_context.fact_query_context,
                    query=task.query,
                    routed_tools=routed_tool_names(memory_context.metadata),
                )
                if profile in {
                    "cloud_joint_planned_fact_patch",
                    "cloud_post_normalized_fact_wiki",
                } and not bool(
                    memory_context.metadata.get(
                        "fact_wiki_fallback_used",
                        False,
                    )
                ):
                    plan = None
                    plan_error = None
                    try:
                        plan = build_memory_tool_plan(
                            query=task.query,
                            records=(
                                *memory_context.fact_records,
                                *memory_context.fact_related_records,
                            ),
                            definitions=definitions,
                            query_context=memory_context.fact_query_context,
                            routed_tools=routed_tool_names(memory_context.metadata),
                            current_state=runtime.state(predicted_world),
                        )
                    except Exception as exc:
                        plan_error = redact_data(f"{type(exc).__name__}: {exc}")
                    planned_hint_result = (
                        plan.as_hint_result() if plan is not None else None
                    )
                    hint_result = choose_memory_tool_hint_result(
                        planned_hint_result,
                        fallback_hint_result,
                    )
                    if plan is not None:
                        joint_candidate_rationale = (
                            render_joint_tool_candidate_rationale(plan.candidates)
                        )
                    memory_context = replace(
                        memory_context,
                        metadata={
                            **dict(memory_context.metadata),
                            **(plan.as_metadata() if plan is not None else {}),
                            "memory_tool_plan_fallback": bool(
                                hint_result is fallback_hint_result
                            ),
                            "memory_tool_plan_applied": bool(
                                planned_hint_result is not None
                                and hint_result is planned_hint_result
                            ),
                            "memory_tool_plan_error": plan_error,
                        },
                        trace={
                            **dict(memory_context.trace),
                            "memory_tool_plan": (
                                plan.as_metadata()
                                if plan is not None
                                else {"error": plan_error}
                            ),
                        },
                    )
                else:
                    hint_result = fallback_hint_result
            else:
                hint_result = build_tool_memory_execution_hints(
                    memory_context.records,
                    definitions,
                )
            if profile in VEHICLE_BINDING_ORACLE_PROFILES:
                binding_eligible = bool(
                    memory_context.metadata.get(
                        "oracle_retrieval_all_records_selected",
                        False,
                    )
                )
                if binding_eligible:
                    hint_result = _gold_binding_hints(
                        task,
                        definitions,
                        record_ids=tuple(
                            memory_context.metadata.get(
                                "oracle_retrieval_record_ids",
                                (),
                            )
                        ),
                    )
                    binding_reason = "applied"
                else:
                    binding_reason = str(
                        memory_context.metadata.get(
                            "oracle_retrieval_record_status",
                            "relevant_record_not_selected",
                        )
                    )
                    if binding_reason == "record_present":
                        binding_reason = "relevant_record_not_selected"
                memory_context = replace(
                    memory_context,
                    metadata={
                        **dict(memory_context.metadata),
                        "oracle_binding": True,
                        "oracle_binding_applied": binding_eligible,
                        "oracle_binding_reason": binding_reason,
                        "oracle_binding_gold_call_count": (
                            len(task.gold_calls) if binding_eligible else 0
                        ),
                    },
                    trace={
                        **dict(memory_context.trace),
                        "oracle_binding": {
                            "applied": binding_eligible,
                            "reason": binding_reason,
                            "gold_call_count": (
                                len(task.gold_calls) if binding_eligible else 0
                            ),
                        },
                    },
                )
            if profile in {
                "cloud_schema_patch",
                "cloud_fact_patch",
                "cloud_schema_informed_fact_patch",
                "cloud_fact_recursive_hybrid",
                "cloud_recursive_assisted_fact_patch",
                "cloud_schema_informed_recursive_assisted_fact_patch",
                "cloud_joint_planned_fact_patch",
                "cloud_post_normalized_fact_wiki",
                "oracle_retrieval_fact_patch",
                "oracle_binding_fact_patch",
                "oracle_retrieval_binding_fact_patch",
                "oracle_gate_fact_patch",
                "oracle_gate_retrieval_binding_fact_patch",
                "oracle_structure_fact_patch",
                "oracle_structure_retrieval_binding_fact_patch",
                "oracle_extraction_fact_patch",
                "oracle_extraction_retrieval_binding_fact_patch",
                "oracle_full_memory_fact_patch",
                "oracle_full_pipeline_fact_patch",
            }:
                selector_tool_names = routed_tool_names(memory_context.metadata)
                if profile in {
                    "cloud_joint_planned_fact_patch",
                    "cloud_post_normalized_fact_wiki",
                }:
                    joint_names = tuple(
                        str(item.get("tool_name"))
                        for item in memory_context.metadata.get(
                            "joint_tool_candidates", ()
                        )
                        if item.get("tool_name")
                    )
                    selector_tool_names = joint_names or selector_tool_names
        preloaded_tool_names = tuple(
            dict.fromkeys(
                (
                    *selector_tool_names,
                    *(hint.tool_name for hint in hint_result.hints),
                )
            )
        )
        if oracle_tool_boundary:
            registry, loaded_modules = build_oracle_vehicle_tool_registry(
                runtime,
                predicted_world,
                definitions,
                oracle_tool_names,
                max_result_chars=max_tool_result_chars,
            )
            discovery = None
            preloaded_tool_names = ()
        else:
            registry, discovery = build_dynamic_vehicle_tool_registry(
                runtime,
                predicted_world,
                definitions,
                max_result_chars=max_tool_result_chars,
                preload_tool_names=preloaded_tool_names,
            )
            loaded_modules = ()
        wiki_tool_names: tuple[str, ...] = ()
        if memory_context is not None and memory_context.wiki_traversal is not None:
            wiki_tool_names = register_vehicle_wiki_tools(
                registry,
                memory_context.wiki_traversal,
            )
        system_prompt = build_vehicle_system_prompt(
            runtime.modules,
            gold_memory=gold_memory,
            retrieved_memory=retrieved_memory,
            preselected_tools=oracle_tool_boundary,
            routed_tool_names=selector_tool_names,
            execution_hints=hint_result.hints,
            candidate_rationale=joint_candidate_rationale,
            progressive_memory_retrieval=(
                memory_context.wiki_kind
                if memory_context is not None and wiki_tool_names
                else False
            ),
        )
        loop = VehicleAgentLoop(
            agent_model=agent_model,
            tool_registry=registry,
            max_tool_rounds=max_tool_rounds,
            non_vehicle_tool_names=wiki_tool_names,
        )
        agent_result = loop.run(
            task_id=f"{profile}-{task.id}",
            query=task.query,
            system_prompt=system_prompt,
        )
        if memory_context is not None and memory_context.wiki_traversal is not None:
            wiki_session = memory_context.wiki_traversal
            wiki_kind = memory_context.wiki_kind or "summary_wiki"
            wiki_result = wiki_session.finish(
                evidence_sufficient=bool(
                    wiki_session.trace.selected_page_ids and not wiki_session.terminated
                ),
                fallback_content=memory_context.content,
            )
            wiki_trace = dict(memory_context.trace.get(wiki_kind, {}))
            wiki_trace.update(
                {
                    "traversal": asdict(wiki_result.trace),
                    "fallback_used": wiki_result.fallback_used,
                    "error": wiki_result.error,
                }
            )
            memory_context = replace(
                memory_context,
                metadata={
                    **dict(memory_context.metadata),
                    f"{wiki_kind}_tool_call_count": sum(
                        item.get("call_kind") == "memory_retrieval"
                        for item in agent_result.tool_trace
                    ),
                    f"{wiki_kind}_selected_page_count": len(
                        wiki_result.trace.selected_page_ids
                    ),
                    f"{wiki_kind}_fallback_used": wiki_result.fallback_used,
                    f"{wiki_kind}_termination_reason": (
                        wiki_result.trace.termination_reason
                    ),
                },
                trace={
                    **dict(memory_context.trace),
                    wiki_kind: wiki_trace,
                },
            )
        reference_state = runtime.state(reference_world)
        predicted_state = runtime.state(predicted_world)
        score = runtime.score(
            initial_state=initial_state,
            reference_state=reference_state,
            predicted_state=predicted_state,
            predicted_calls=agent_result.predicted_calls,
            reference_calls=task.gold_calls,
        )
        argument_exact_match = _argument_exact_match(
            agent_result.predicted_calls,
            task.gold_calls,
        )
        retrieval_quality = _retrieval_quality(
            task,
            memory_context,
            hint_result=hint_result,
        )
        record = {
            "profile": profile,
            "task_id": task.id,
            "scenario_index": task.scenario_index,
            "event_index": task.event_index,
            "reasoning_type": task.reasoning_type,
            "query": task.query,
            "status": agent_result.status,
            "terminal_reason": agent_result.terminal_reason,
            "rounds": agent_result.rounds,
            "final_response": agent_result.final_response,
            "context": _task_context(
                profile,
                task,
                memory_context=memory_context,
                selector_tool_names=selector_tool_names,
                hint_result=hint_result,
            ),
            "predicted_calls": [
                call.as_official() for call in agent_result.predicted_calls
            ],
            "reference_calls": [call.as_official() for call in task.gold_calls],
            "discovery_calls": agent_result.discovery_calls,
            "suppressed_duplicate_calls": (agent_result.suppressed_duplicate_calls),
            "loaded_modules": (
                list(loaded_modules)
                if oracle_tool_boundary
                else sorted(discovery.available_modules)
            ),
            "score": score.as_dict(),
            "argument_exact_match": argument_exact_match,
            "retrieval_quality": retrieval_quality,
            "usage": {
                "input_tokens": agent_result.input_tokens,
                "output_tokens": agent_result.output_tokens,
                "model_latency_ms": agent_result.model_latency_ms,
            },
            "model_trace": list(agent_result.model_trace),
            "tool_trace": list(agent_result.tool_trace),
            "memory_trace": (
                {
                    **memory_context.trace,
                    "agent_context": {
                        "selector_tool_names": list(selector_tool_names),
                        **hint_result.as_metadata(),
                    },
                }
                if memory_context is not None
                else None
            ),
            "scorer_input": {
                "initial_state": initial_state,
                "reference_state": reference_state,
                "predicted_state": predicted_state,
                "reference_outputs": reference_outputs,
            },
            "tool_boundary": {
                "initial_tools": (
                    list(dict.fromkeys(oracle_tool_names))
                    if oracle_tool_boundary
                    else [
                        "list_module_tools",
                        *wiki_tool_names,
                        *preloaded_tool_names,
                    ]
                ),
                **(
                    {}
                    if oracle_tool_boundary
                    else {
                        "router_preloaded_tools": list(preloaded_tool_names),
                        "router_preloaded_modules": sorted(discovery.preloaded_modules),
                    }
                ),
                "host_tools_available": False,
                "simulator": "VehicleWorld",
                "oracle_tool_boundary": oracle_tool_boundary,
                "gold_arguments_exposed": bool(
                    memory_context is not None
                    and memory_context.metadata.get(
                        "oracle_binding_applied",
                        False,
                    )
                ),
            },
            "error": None,
        }
        record["diagnostics"] = _task_diagnostics(record)
        return record
    except Exception as exc:
        record = {
            "profile": profile,
            "task_id": task.id,
            "scenario_index": task.scenario_index,
            "event_index": task.event_index,
            "reasoning_type": task.reasoning_type,
            "query": task.query,
            "status": "failed",
            "terminal_reason": "exception",
            "context": _task_context(profile, task, memory_context=None),
            "predicted_calls": [],
            "reference_calls": [call.as_official() for call in task.gold_calls],
            "discovery_calls": 0,
            "score": None,
            "argument_exact_match": False,
            "retrieval_quality": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "model_latency_ms": 0,
            },
            "tool_boundary": {
                "initial_tools": (
                    list(dict.fromkeys(call.name for call in task.gold_calls))
                    if profile in VEHICLE_ORACLE_PROFILES
                    else ["list_module_tools"]
                ),
                "host_tools_available": False,
                "simulator": "VehicleWorld",
                "oracle_tool_boundary": profile in VEHICLE_ORACLE_PROFILES,
                "gold_arguments_exposed": False,
            },
            "error": f"{type(exc).__name__}: {exc}",
        }
        record["diagnostics"] = _task_diagnostics(record)
        return record


def _validate_agent_profiles(profile_names: tuple[str, ...]) -> tuple[str, ...]:
    profiles = tuple(dict.fromkeys(profile_names))
    if not profiles:
        raise ValueError("At least one VehicleMemBench profile is required")
    unknown = sorted(set(profiles) - set(VEHICLE_AGENT_PROFILES))
    if unknown:
        raise ValueError(f"Unknown VehicleMemBench profiles: {unknown}")
    return profiles


def _task_context(
    profile: str,
    task: VehicleTask,
    *,
    memory_context: VehicleMemoryContext | None,
    selector_tool_names: Sequence[str] = (),
    hint_result: ToolMemoryHintResult | None = None,
) -> dict[str, Any]:
    if profile in {"gold_memory", "oracle_tool_gold_memory"}:
        sources = ["gold_memory", "query"]
    elif profile in VEHICLE_MEMORY_PROFILES:
        sources = ["retrieved_memory", "query"]
    else:
        sources = ["query"]
    gold_spans = detect_sensitive_spans(task.gold_memory)
    return {
        "sources": sources,
        "raw_history_included": False,
        "query_updates_memory": False,
        "gold_memory_sha256": (
            hashlib.sha256(task.gold_memory.encode("utf-8")).hexdigest()
            if profile in {"gold_memory", "oracle_tool_gold_memory"}
            else None
        ),
        "retrieved_memory_sha256": (
            hashlib.sha256(memory_context.content.encode("utf-8")).hexdigest()
            if memory_context is not None
            else None
        ),
        "retrieval_metadata": (
            memory_context.metadata if memory_context is not None else {}
        ),
        "selector_tool_names": list(selector_tool_names),
        **(
            hint_result.as_metadata()
            if hint_result is not None
            else ToolMemoryHintResult(hints=(), rejections=()).as_metadata()
        ),
        "prompt_version": VEHICLE_AGENT_PROMPT_VERSION,
        "gold_redaction_audit": {
            "detected_count": len(gold_spans),
            "category_counts": dict(
                sorted(Counter(span.category for span in gold_spans).items())
            ),
            "answer_loss_risk": bool(gold_spans),
        },
    }


def vehicle_agent_metrics(
    records: list[dict[str, Any]],
    profiles: tuple[str, ...],
    *,
    agent_input_cost_per_million: float = 0.0,
    agent_output_cost_per_million: float = 0.0,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "total_tasks": len(records),
        "failed_tasks": sum(record["status"] == "failed" for record in records),
        "profiles": {},
    }
    for profile in profiles:
        selected = [record for record in records if record["profile"] == profile]
        usage_input = sum(
            int(record.get("usage", {}).get("input_tokens", 0)) for record in selected
        )
        usage_output = sum(
            int(record.get("usage", {}).get("output_tokens", 0)) for record in selected
        )
        latencies = [
            int(record.get("usage", {}).get("model_latency_ms", 0))
            for record in selected
        ]
        privacy = _agent_privacy_metrics(selected)
        wiki_traversal = _aggregate_wiki_traversal(selected)
        online_retrieval = _aggregate_online_memory_retrieval(
            selected,
            wiki_traversal=wiki_traversal,
        )
        estimated_agent_cost_usd = round(
            (
                usage_input * agent_input_cost_per_million
                + usage_output * agent_output_cost_per_million
            )
            / 1_000_000,
            8,
        )
        profile_metrics = {
            "tasks": len(selected),
            "completion_rate": _mean(
                record["status"] == "completed" for record in selected
            ),
            "exact_state_match": _mean(
                bool(record.get("score", {}).get("exact_state_match"))
                if record.get("score")
                else False
                for record in selected
            ),
            "state_f1": _mean(
                float(
                    record.get("score", {}).get("state_score", {}).get("f1_positive", 0)
                )
                if record.get("score")
                else 0
                for record in selected
            ),
            "state_recall": _mean(
                float(
                    record.get("score", {})
                    .get("state_score", {})
                    .get("acc_positive", 0)
                )
                if record.get("score")
                else 0
                for record in selected
            ),
            "state_precision": _mean(
                float(
                    record.get("score", {})
                    .get("state_score", {})
                    .get("precision_positive", 0)
                )
                if record.get("score")
                else 0
                for record in selected
            ),
            "value_f1": _mean(
                float(
                    record.get("score", {}).get("state_score", {}).get("f1_change", 0)
                )
                if record.get("score")
                else 0
                for record in selected
            ),
            "tool_f1": _mean(
                float(record.get("score", {}).get("tool_score", {}).get("f1", 0))
                if record.get("score")
                else 0
                for record in selected
            ),
            "argument_exact_match": _mean(
                bool(record.get("argument_exact_match")) for record in selected
            ),
            "retrieval_recall_at_k": _mean(
                float(
                    (record.get("retrieval_quality") or {}).get(
                        "recall_at_k",
                        0,
                    )
                )
                for record in selected
                if record.get("retrieval_quality") is not None
            ),
            "retrieval_pipeline_recall_at_k": _mean(
                float(
                    (record.get("retrieval_quality") or {}).get(
                        "pipeline_recall_at_k",
                        0,
                    )
                )
                for record in selected
                if record.get("retrieval_quality") is not None
            ),
            "retrieval_pipeline_hit_rate": _mean(
                bool(
                    (record.get("retrieval_quality") or {}).get(
                        "all_gold_arguments_available_to_agent",
                        False,
                    )
                )
                for record in selected
                if record.get("retrieval_quality") is not None
            ),
            "retrieval_hit_rate": _mean(
                bool(
                    (record.get("retrieval_quality") or {}).get(
                        "all_gold_arguments_recalled",
                        False,
                    )
                )
                for record in selected
                if record.get("retrieval_quality") is not None
            ),
            "retrieval_selected_count_mean": _mean(
                int(
                    (record.get("retrieval_quality") or {}).get(
                        "selected_count",
                        0,
                    )
                )
                for record in selected
                if record.get("retrieval_quality") is not None
            ),
            "retrieval_context_tokens_mean": _mean(
                int(
                    (record.get("retrieval_quality") or {}).get(
                        "context_tokens",
                        0,
                    )
                )
                for record in selected
                if record.get("retrieval_quality") is not None
            ),
            "retrieval_latency_ms_mean": _mean(
                int(
                    (record.get("retrieval_quality") or {}).get(
                        "latency_ms",
                        0,
                    )
                )
                for record in selected
                if record.get("retrieval_quality") is not None
            ),
            "oracle_retrieval_record_present_tasks": sum(
                record.get("context", {})
                .get("retrieval_metadata", {})
                .get("oracle_retrieval_record_status")
                == "record_present"
                for record in selected
            ),
            "oracle_retrieval_record_absent_tasks": sum(
                record.get("context", {})
                .get("retrieval_metadata", {})
                .get("oracle_retrieval_record_status")
                == "record_absent"
                for record in selected
            ),
            "oracle_retrieval_forced_tasks": sum(
                int(
                    record.get("context", {})
                    .get("retrieval_metadata", {})
                    .get("oracle_retrieval_forced_count", 0)
                )
                > 0
                for record in selected
            ),
            "oracle_retrieval_forced_records": sum(
                int(
                    record.get("context", {})
                    .get("retrieval_metadata", {})
                    .get("oracle_retrieval_forced_count", 0)
                )
                for record in selected
            ),
            "oracle_retrieval_context_changed_tasks": sum(
                bool(
                    record.get("context", {})
                    .get("retrieval_metadata", {})
                    .get("oracle_retrieval_context_changed", False)
                )
                for record in selected
            ),
            "oracle_gate_recoverable_tasks": sum(
                record.get("context", {})
                .get("retrieval_metadata", {})
                .get("oracle_gate_candidate_status")
                == "gate_recoverable"
                for record in selected
            ),
            "oracle_gate_selected_tasks": sum(
                bool(
                    record.get("context", {})
                    .get("retrieval_metadata", {})
                    .get("oracle_gate_all_records_selected", False)
                )
                for record in selected
            ),
            "oracle_gate_forced_records": sum(
                int(
                    record.get("context", {})
                    .get("retrieval_metadata", {})
                    .get("oracle_gate_forced_count", 0)
                )
                for record in selected
            ),
            "oracle_binding_applied_tasks": sum(
                bool(
                    record.get("context", {})
                    .get("retrieval_metadata", {})
                    .get("oracle_binding_applied", False)
                )
                for record in selected
            ),
            "oracle_binding_gold_calls": sum(
                int(
                    record.get("context", {})
                    .get("retrieval_metadata", {})
                    .get("oracle_binding_gold_call_count", 0)
                )
                for record in selected
            ),
            "selector_tool_count_mean": _mean(
                len(
                    record.get("context", {}).get(
                        "selector_tool_names",
                        (),
                    )
                )
                for record in selected
            ),
            "execution_hint_count_mean": _mean(
                int(
                    record.get("context", {}).get(
                        "tool_memory_execution_hint_count",
                        0,
                    )
                )
                for record in selected
            ),
            "execution_hint_rejections": sum(
                int(
                    record.get("context", {}).get(
                        "tool_memory_execution_hint_rejection_count",
                        0,
                    )
                )
                for record in selected
            ),
            "average_vehicle_calls": _mean(
                len(record.get("predicted_calls", ())) for record in selected
            ),
            "average_discovery_calls": _mean(
                int(record.get("discovery_calls", 0)) for record in selected
            ),
            "unnecessary_vehicle_calls": sum(
                int(
                    record.get("diagnostics", {}).get(
                        "unnecessary_vehicle_calls",
                        0,
                    )
                )
                for record in selected
            ),
            "suppressed_duplicate_calls": sum(
                int(record.get("suppressed_duplicate_calls", 0)) for record in selected
            ),
            "extra_discovery_calls": sum(
                int(
                    record.get("diagnostics", {}).get(
                        "extra_discovery_calls",
                        0,
                    )
                )
                for record in selected
            ),
            "failure_taxonomy": dict(
                sorted(
                    Counter(
                        str(
                            record.get("diagnostics", {}).get(
                                "primary_outcome",
                                "unknown",
                            )
                        )
                        for record in selected
                    ).items()
                )
            ),
            "input_tokens": usage_input,
            "output_tokens": usage_output,
            "model_latency_ms": sum(latencies),
            "model_latency_ms_mean": _mean(latencies),
            "model_latency_ms_p95": _percentile(latencies, 0.95),
            "estimated_agent_cost_usd": estimated_agent_cost_usd,
            "cost_rates_configured": bool(
                agent_input_cost_per_million or agent_output_cost_per_million
            ),
            "workflow_stages": {
                "online_memory_retrieval": online_retrieval,
                "quiz_agent": {
                    "input_tokens": usage_input,
                    "output_tokens": usage_output,
                    "total_tokens": usage_input + usage_output,
                    "model_latency_ms": sum(latencies),
                    "estimated_cost_usd": estimated_agent_cost_usd,
                },
            },
            **privacy,
        }
        if wiki_traversal is not None:
            profile_metrics["wiki_traversal"] = wiki_traversal
        if profile in {
            "cloud_recursive_summary",
            "cloud_recursive_summary_patch",
            "cloud_turnwise_recursive_summary",
            "cloud_turnwise_recursive_summary_patch",
            "cloud_turnwise_recursive_summary_patch_compact",
            "cloud_turnwise_recursive_summary_patch_temporal",
            "cloud_turnwise_recursive_summary_patch_temporal_compact",
            "cloud_recursive_summary_gated_wiki",
        }:
            profile_metrics.update(
                {
                    "retrieval_recall_at_k": None,
                    "retrieval_pipeline_recall_at_k": None,
                    "retrieval_pipeline_hit_rate": None,
                    "retrieval_hit_rate": None,
                }
            )
        patch_quality = next(
            (
                record.get("context", {})
                .get("retrieval_metadata", {})
                .get("patch_quality")
                for record in selected
                if record.get("context", {})
                .get("retrieval_metadata", {})
                .get("patch_quality")
            ),
            None,
        )
        if patch_quality is not None:
            profile_metrics["patch_quality"] = patch_quality
        recursive_summary = _aggregate_recursive_summary(selected)
        if recursive_summary is not None:
            profile_metrics["recursive_summary"] = recursive_summary
        fact_quality = _aggregate_fact_quality(selected)
        if fact_quality is not None:
            profile_metrics["fact_quality"] = fact_quality
        reasoning_types = sorted({str(record["reasoning_type"]) for record in selected})
        profile_metrics["reasoning_types"] = {
            reasoning_type: _reasoning_metrics(
                [
                    record
                    for record in selected
                    if record["reasoning_type"] == reasoning_type
                ]
            )
            for reasoning_type in reasoning_types
        }
        metrics["profiles"][profile] = profile_metrics
    return metrics


def _aggregate_wiki_traversal(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    applicable_tasks = 0
    gate_open_tasks = 0
    available_tasks = 0
    traversal_tasks = 0
    tool_invoked_tasks = 0
    fallback_tasks = 0
    search_calls = 0
    read_calls = 0
    read_pages = 0
    selected_pages = 0
    rendered_tokens = 0
    empty_searches = 0
    tool_calls = 0
    successful_steps = 0
    tool_errors = 0
    tool_latency_ms = 0
    hop_counts: list[int] = []
    termination_reasons: Counter[str] = Counter()
    wiki_kinds: Counter[str] = Counter()

    for record in records:
        metadata = record.get("context", {}).get("retrieval_metadata", {}) or {}
        memory_trace = record.get("memory_trace") or {}
        wiki_kind = next(
            (
                kind
                for kind in ("summary_wiki", "fact_wiki")
                if kind in memory_trace
                or any(key.startswith(f"{kind}_") for key in metadata)
            ),
            None,
        )
        if wiki_kind is None:
            continue
        applicable_tasks += 1
        wiki_kinds[wiki_kind] += 1
        if bool(metadata.get("summary_wiki_gate_open", False)):
            gate_open_tasks += 1
        if bool(metadata.get("fact_wiki_available", False)):
            available_tasks += 1

        wiki_trace = memory_trace.get(wiki_kind) or {}
        traversal = wiki_trace.get("traversal")
        if not isinstance(traversal, dict):
            reason = (
                "not_started_gate_closed"
                if wiki_kind == "summary_wiki"
                and not bool(metadata.get("summary_wiki_gate_open", False))
                else "not_started_projection_fallback"
            )
            termination_reasons[reason] += 1
            if bool(
                metadata.get(
                    f"{wiki_kind}_fallback_used",
                    wiki_trace.get("fallback_used", False),
                )
            ):
                fallback_tasks += 1
            continue

        traversal_tasks += 1
        steps = traversal.get("steps") or ()
        successful_steps += len(steps)
        search_calls += int(traversal.get("search_count", 0))
        read_pages += int(traversal.get("read_count", 0))
        read_calls += sum(
            str(step.get("action", "")) == "read"
            for step in steps
            if isinstance(step, dict)
        )
        selected_pages += len(traversal.get("selected_page_ids") or ())
        rendered_tokens += int(traversal.get("rendered_tokens", 0))
        empty_searches += int(traversal.get("empty_search_count", 0))
        hop_counts.append(int(traversal.get("hop_count", 0)))
        fallback_tasks += int(bool(traversal.get("fallback_used", False)))
        reason = str(
            traversal.get("termination_reason")
            or metadata.get(f"{wiki_kind}_termination_reason")
            or "unknown"
        )
        termination_reasons[reason] += 1

        wiki_tool_trace = [
            item
            for item in record.get("tool_trace", ())
            if item.get("call_kind") == "memory_retrieval"
        ]
        tool_calls += len(wiki_tool_trace)
        tool_invoked_tasks += int(bool(wiki_tool_trace))
        tool_latency_ms += sum(
            int(item.get("duration_ms", 0)) for item in wiki_tool_trace
        )
        tool_errors += sum(bool(item.get("is_error")) for item in wiki_tool_trace)

    if not applicable_tasks:
        return None
    sufficient_tasks = int(termination_reasons.get("evidence_sufficient", 0))
    return {
        "applicable_tasks": applicable_tasks,
        "wiki_kinds": dict(sorted(wiki_kinds.items())),
        "gate_open_tasks": gate_open_tasks,
        "available_tasks": available_tasks,
        "traversal_tasks": traversal_tasks,
        "tool_invoked_tasks": tool_invoked_tasks,
        "fallback_tasks": fallback_tasks,
        "tool_calls": tool_calls,
        "successful_steps": successful_steps,
        "search_calls": search_calls,
        "read_calls": read_calls,
        "read_pages": read_pages,
        "selected_pages": selected_pages,
        "rendered_tokens": rendered_tokens,
        "rendered_tokens_mean": (
            rendered_tokens / traversal_tasks if traversal_tasks else 0.0
        ),
        "hop_count_mean": _mean(hop_counts),
        "hop_count_max": max(hop_counts, default=0),
        "empty_searches": empty_searches,
        "evidence_sufficient_tasks": sufficient_tasks,
        "sufficiency_rate": (
            sufficient_tasks / traversal_tasks if traversal_tasks else 0.0
        ),
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "tool_errors": tool_errors,
        "tool_latency_ms": tool_latency_ms,
    }


def _aggregate_online_memory_retrieval(
    records: Sequence[dict[str, Any]],
    *,
    wiki_traversal: dict[str, Any] | None,
) -> dict[str, Any]:
    base_context_tokens = 0
    base_retrieval_latency_ms = 0
    retrieval_tasks = 0
    for record in records:
        quality = record.get("retrieval_quality") or {}
        metadata = record.get("context", {}).get("retrieval_metadata", {}) or {}
        context_tokens = int(
            next(
                (
                    metadata[key]
                    for key in (
                        "fact_memory_retrieval_tokens",
                        "tool_memory_retrieval_tokens",
                        "retrieval_selected_tokens",
                        "retrieval_tokens",
                        "recursive_summary_tokens",
                    )
                    if key in metadata
                ),
                quality.get("context_tokens", 0),
            )
        )
        latency_ms = int(
            next(
                (
                    metadata[key]
                    for key in (
                        "fact_memory_retrieval_latency_ms",
                        "tool_memory_retrieval_latency_ms",
                        "retrieval_latency_ms",
                    )
                    if key in metadata
                ),
                quality.get("latency_ms", 0),
            )
        )
        base_context_tokens += context_tokens
        base_retrieval_latency_ms += latency_ms
        retrieval_tasks += int(bool(context_tokens or latency_ms))

    wiki_tokens = int((wiki_traversal or {}).get("rendered_tokens", 0))
    wiki_latency_ms = int((wiki_traversal or {}).get("tool_latency_ms", 0))
    return {
        "tasks_with_base_retrieval": retrieval_tasks,
        "base_context_tokens": base_context_tokens,
        "wiki_read_context_tokens": wiki_tokens,
        "observed_context_tokens": base_context_tokens + wiki_tokens,
        "base_retrieval_latency_ms": base_retrieval_latency_ms,
        "wiki_tool_latency_ms": wiki_latency_ms,
        "observed_latency_ms": base_retrieval_latency_ms + wiki_latency_ms,
        "wiki_tool_calls": int((wiki_traversal or {}).get("tool_calls", 0)),
        "context_tokens_are_non_additive_to_agent_input": True,
    }


def _aggregate_recursive_summary(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    snapshots: dict[str, dict[str, Any]] = {}
    for record in records:
        metadata = record.get("context", {}).get("retrieval_metadata", {}) or {}
        if metadata.get("memory_strategy") not in {
            "recursive_summary",
            "fact_recursive_hybrid",
            "recursive_assisted_fact_patch",
        }:
            continue
        cache_key = str(
            metadata.get("cache_key")
            or f"scenario-{record.get('scenario_index', 'unknown')}"
        )
        snapshots.setdefault(cache_key, metadata)
    if not snapshots:
        return None
    characters = [
        int(item.get("recursive_summary_characters", 0)) for item in snapshots.values()
    ]
    tokens = [
        int(item.get("recursive_summary_tokens", 0)) for item in snapshots.values()
    ]
    total_steps = sum(
        int(item.get("recursive_summary_total_step_count", 0))
        for item in snapshots.values()
    )
    total_updates = sum(
        int(item.get("recursive_summary_update_count", 0))
        for item in snapshots.values()
    )
    total_noops = sum(
        int(item.get("recursive_summary_noop_count", 0)) for item in snapshots.values()
    )
    status_usage: dict[str, dict[str, int]] = {}
    for item in snapshots.values():
        for status, usage in dict(
            item.get("recursive_summary_status_usage", {})
        ).items():
            bucket = status_usage.setdefault(
                str(status),
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0,
                },
            )
            for key in bucket:
                bucket[key] += int(dict(usage).get(key, 0))
    patch_operations = sum(
        int(item.get("recursive_summary_patch_operation_count", 0))
        for item in snapshots.values()
    )
    patch_adds = sum(
        int(item.get("recursive_summary_patch_add_count", 0))
        for item in snapshots.values()
    )
    patch_replaces = sum(
        int(item.get("recursive_summary_patch_replace_count", 0))
        for item in snapshots.values()
    )
    patch_deletes = sum(
        int(item.get("recursive_summary_patch_delete_count", 0))
        for item in snapshots.values()
    )
    return {
        "snapshot_count": len(snapshots),
        "total_step_count": total_steps,
        "update_count": total_updates,
        "noop_count": total_noops,
        "update_ratio": total_updates / total_steps if total_steps else 0.0,
        "noop_ratio": total_noops / total_steps if total_steps else 0.0,
        "failed_attempt_count": sum(
            int(item.get("recursive_summary_failed_attempt_count", 0))
            for item in snapshots.values()
        ),
        "status_usage": status_usage,
        "redundant_update_count": sum(
            int(item.get("recursive_summary_redundant_update_count", 0))
            for item in snapshots.values()
        ),
        "redundant_update_ratio": (
            sum(
                int(item.get("recursive_summary_redundant_update_count", 0))
                for item in snapshots.values()
            )
            / total_steps
            if total_steps
            else 0.0
        ),
        "update_cadences": sorted(
            {
                str(
                    item.get(
                        "recursive_summary_update_cadence",
                        "calendar_day",
                    )
                )
                for item in snapshots.values()
            }
        ),
        "truncation_count": sum(
            int(item.get("recursive_summary_truncation_count", 0))
            for item in snapshots.values()
        ),
        "update_modes": sorted(
            {
                str(item.get("recursive_summary_update_mode", "full_rewrite"))
                for item in snapshots.values()
            }
        ),
        "patch_operation_count": patch_operations,
        "patch_add_count": patch_adds,
        "patch_replace_count": patch_replaces,
        "patch_delete_count": patch_deletes,
        "patch_add_ratio": patch_adds / patch_operations if patch_operations else 0.0,
        "patch_replace_ratio": (
            patch_replaces / patch_operations if patch_operations else 0.0
        ),
        "patch_delete_ratio": (
            patch_deletes / patch_operations if patch_operations else 0.0
        ),
        "patch_generation_attempt_count": sum(
            int(
                item.get(
                    "recursive_summary_patch_generation_attempt_count",
                    0,
                )
            )
            for item in snapshots.values()
        ),
        "patch_rejection_count": sum(
            int(item.get("recursive_summary_patch_rejection_count", 0))
            for item in snapshots.values()
        ),
        "patch_apply_latency_ms": sum(
            int(item.get("recursive_summary_patch_apply_latency_ms", 0))
            for item in snapshots.values()
        ),
        "compaction_count": sum(
            int(item.get("recursive_summary_compaction_count", 0))
            for item in snapshots.values()
        ),
        "compaction_attempt_count": sum(
            int(item.get("recursive_summary_compaction_attempt_count", 0))
            for item in snapshots.values()
        ),
        "compaction_latency_ms": sum(
            int(item.get("recursive_summary_compaction_latency_ms", 0))
            for item in snapshots.values()
        ),
        "compaction_input_tokens": sum(
            int(item.get("recursive_summary_compaction_input_tokens", 0))
            for item in snapshots.values()
        ),
        "compaction_output_tokens": sum(
            int(item.get("recursive_summary_compaction_output_tokens", 0))
            for item in snapshots.values()
        ),
        "temporal_operation_count": sum(
            int(item.get("recursive_summary_temporal_operation_count", 0))
            for item in snapshots.values()
        ),
        "temporal_non_temporal_count": sum(
            int(item.get("recursive_summary_temporal_non_temporal_count", 0))
            for item in snapshots.values()
        ),
        "temporal_durable_upsert_count": sum(
            int(
                item.get(
                    "recursive_summary_temporal_durable_upsert_count",
                    0,
                )
            )
            for item in snapshots.values()
        ),
        "temporal_current_upsert_count": sum(
            int(item.get("recursive_summary_temporal_current_upsert_count", 0))
            for item in snapshots.values()
        ),
        "temporal_temporary_override_count": sum(
            int(
                item.get(
                    "recursive_summary_temporal_temporary_override_count",
                    0,
                )
            )
            for item in snapshots.values()
        ),
        "temporal_end_temporary_count": sum(
            int(item.get("recursive_summary_temporal_end_temporary_count", 0))
            for item in snapshots.values()
        ),
        "temporal_conditional_upsert_count": sum(
            int(
                item.get(
                    "recursive_summary_temporal_conditional_upsert_count",
                    0,
                )
            )
            for item in snapshots.values()
        ),
        "final_characters_mean": _mean(characters),
        "final_characters_max": max(characters, default=0),
        "final_tokens_mean": _mean(tokens),
        "final_tokens_max": max(tokens, default=0),
    }


def _aggregate_fact_quality(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    snapshots: dict[str, dict[str, Any]] = {}
    for record in records:
        metadata = record.get("context", {}).get("retrieval_metadata", {}) or {}
        quality = metadata.get("fact_quality")
        if not quality:
            continue
        snapshot_key = str(
            metadata.get("cache_key")
            or f"scenario-{record.get('scenario_index', 'unknown')}"
        )
        snapshots.setdefault(snapshot_key, quality)
    if not snapshots:
        return None

    count_keys = (
        "candidate_count",
        "record_count",
        "active_record_count",
        "versioned_record_count",
        "model_call_count",
        "semantic_model_call_count",
    )
    counter_keys = (
        "decision_counts",
        "candidate_status_counts",
        "validation_code_counts",
        "record_status_counts",
    )
    usage_keys = ("model_usage", "semantic_model_usage")
    aggregate: dict[str, Any] = {
        "snapshot_count": len(snapshots),
        **{
            key: sum(int(item.get(key, 0)) for item in snapshots.values())
            for key in count_keys
        },
    }
    for key in counter_keys:
        counts: Counter[str] = Counter()
        for item in snapshots.values():
            counts.update(
                {str(name): int(value) for name, value in (item.get(key) or {}).items()}
            )
        aggregate[key] = dict(sorted(counts.items()))
    for key in usage_keys:
        aggregate[key] = {
            field: sum(
                int((item.get(key) or {}).get(field, 0)) for item in snapshots.values()
            )
            for field in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_tokens",
            )
        }
    candidate_count = int(aggregate["candidate_count"])
    applied_count = int(aggregate["candidate_status_counts"].get("applied", 0))
    resolved_count = applied_count + int(
        aggregate["candidate_status_counts"].get("noop", 0)
    )
    aggregate["candidate_acceptance_rate"] = (
        applied_count / candidate_count if candidate_count else 0.0
    )
    aggregate["candidate_resolution_rate"] = (
        resolved_count / candidate_count if candidate_count else 0.0
    )
    return aggregate


def _task_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    score = record.get("score") or {}
    tool_score = score.get("tool_score") or {}
    predicted_calls = record.get("predicted_calls") or []
    reference_calls = record.get("reference_calls") or []
    reference_modules = {
        str(call.get("name", "")).split("_", 2)[1]
        for call in reference_calls
        if str(call.get("name", "")).count("_") >= 2
    }
    discovery_calls = int(record.get("discovery_calls", 0))
    reference_tool_names = {str(call.get("name", "")) for call in reference_calls}
    predicted_tool_names = {str(call.get("name", "")) for call in predicted_calls}
    codes: list[str] = []
    if record.get("status") == "failed":
        codes.append("execution_error")
    if any(bool(item.get("is_error")) for item in record.get("tool_trace", ())):
        codes.append("tool_execution_error")
    if any(
        item.get("name") == "list_module_tools" and bool(item.get("is_error"))
        for item in record.get("tool_trace", ())
    ):
        codes.append("tool_discovery_error")
    context = record.get("context") or {}
    retrieval = context.get("retrieval_metadata") or {}
    retrieval_selected_count = int(
        retrieval.get(
            "fact_memory_retrieval_selected_count",
            retrieval.get(
                "tool_memory_retrieval_selected_count",
                retrieval.get("retrieval_selected_count", 0),
            ),
        )
    )
    if (
        record.get("profile") in VEHICLE_MEMORY_PROFILES
        and retrieval_selected_count == 0
        and (not score or not bool(score.get("exact_state_match")))
    ):
        codes.append("retrieval_empty")
    elif (
        record.get("profile") in VEHICLE_MEMORY_PROFILES
        and score
        and not bool(score.get("exact_state_match"))
    ):
        codes.append("memory_or_retrieval_mismatch")
    selector_names = {str(name) for name in context.get("selector_tool_names", ())}
    selector_misses = sorted(reference_tool_names - selector_names)
    state_failed = not score or not bool(score.get("exact_state_match"))
    if (
        record.get("profile")
        in {
            "cloud_schema_patch",
            "cloud_fact_patch",
            "cloud_schema_informed_fact_patch",
            "cloud_fact_recursive_hybrid",
            "cloud_recursive_assisted_fact_patch",
            "cloud_schema_informed_recursive_assisted_fact_patch",
            "cloud_joint_planned_fact_patch",
            "cloud_post_normalized_fact_wiki",
        }
        and selector_misses
        and state_failed
    ):
        codes.append("tool_selector_miss")
    hint_rejections = int(context.get("tool_memory_execution_hint_rejection_count", 0))
    if hint_rejections and state_failed:
        codes.append("memory_argument_mapping_rejection")
    if reference_calls and not predicted_calls:
        codes.append("agent_no_action")
    if int(tool_score.get("fp", 0)) > 0:
        codes.append("extra_vehicle_call")
    if int(tool_score.get("fn", 0)) > 0:
        codes.append("missing_vehicle_call")
        if (
            record.get("profile")
            not in {
                "cloud_schema_patch",
                "cloud_fact_patch",
                "cloud_schema_informed_fact_patch",
                "cloud_fact_recursive_hybrid",
                "cloud_recursive_assisted_fact_patch",
                "cloud_schema_informed_recursive_assisted_fact_patch",
                "cloud_joint_planned_fact_patch",
                "cloud_post_normalized_fact_wiki",
            }
            or not selector_misses
        ):
            codes.append("agent_tool_omission")
    if (
        score
        and reference_tool_names <= predicted_tool_names
        and not bool(record.get("argument_exact_match"))
    ):
        codes.append("agent_argument_mismatch")
    if score and not bool(score.get("exact_state_match")):
        codes.append("state_mismatch")
    extra_discovery = max(0, discovery_calls - len(reference_modules))
    if extra_discovery:
        codes.append("extra_discovery_call")
    if not codes:
        codes.append("success")
    priority = (
        "execution_error",
        "tool_discovery_error",
        "tool_execution_error",
        "tool_selector_miss",
        "retrieval_empty",
        "memory_argument_mapping_rejection",
        "agent_tool_omission",
        "agent_argument_mismatch",
        "memory_or_retrieval_mismatch",
        "agent_no_action",
        "extra_vehicle_call",
        "missing_vehicle_call",
        "state_mismatch",
        "extra_discovery_call",
        "success",
    )
    primary = next(item for item in priority if item in codes)
    return {
        "primary_outcome": primary,
        "failure_codes": codes,
        "unnecessary_vehicle_calls": int(tool_score.get("fp", 0)),
        "missing_vehicle_calls": int(tool_score.get("fn", 0)),
        "extra_discovery_calls": extra_discovery,
        "suppressed_duplicate_calls": int(record.get("suppressed_duplicate_calls", 0)),
        "selector_missing_reference_tools": selector_misses,
        "hint_rejection_count": hint_rejections,
        "retrieval_run_id": retrieval.get("retrieval_run_id"),
        "tool_memory_retrieval_run_id": retrieval.get("tool_memory_retrieval_run_id"),
        "fact_memory_retrieval_run_id": retrieval.get("fact_memory_retrieval_run_id"),
        "memory_fingerprint": retrieval.get("memory_fingerprint"),
    }


def _reasoning_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len(records),
        "exact_state_match": _mean(
            bool(record.get("score", {}).get("exact_state_match"))
            if record.get("score")
            else False
            for record in records
        ),
        "state_f1": _mean(
            float(record.get("score", {}).get("state_score", {}).get("f1_positive", 0))
            if record.get("score")
            else 0
            for record in records
        ),
        "value_f1": _mean(
            float(record.get("score", {}).get("state_score", {}).get("f1_change", 0))
            if record.get("score")
            else 0
            for record in records
        ),
        "tool_f1": _mean(
            float(record.get("score", {}).get("tool_score", {}).get("f1", 0))
            if record.get("score")
            else 0
            for record in records
        ),
        "argument_exact_match": _mean(
            bool(record.get("argument_exact_match")) for record in records
        ),
    }


def _argument_exact_match(
    predicted_calls: Sequence[Any],
    reference_calls: Sequence[Any],
) -> bool:
    if len(predicted_calls) != len(reference_calls):
        return False
    return all(
        predicted.name == reference.name
        and json.dumps(
            dict(predicted.arguments),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        == json.dumps(
            dict(reference.arguments),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for predicted, reference in zip(
            predicted_calls,
            reference_calls,
            strict=True,
        )
    )


def _gold_binding_hints(
    task: VehicleTask,
    definitions: Sequence[ToolDefinition],
    *,
    record_ids: Sequence[str],
) -> ToolMemoryHintResult:
    """Build evaluation-only exact Tool hints for a retrieved reviewed Fact."""

    if not record_ids:
        raise ValueError(f"Oracle Binding requires a selected Fact record: {task.id}")
    tools = {
        tool.tool_name: tool for tool in build_tool_memory_ontology(definitions).tools
    }
    hints = []
    for index, call in enumerate(task.gold_calls):
        tool = tools.get(call.name)
        if tool is None:
            raise ValueError(f"Oracle Binding references an unknown Tool: {call.name}")
        hints.append(
            ToolMemoryExecutionHint(
                record_id=record_ids[min(index, len(record_ids) - 1)],
                tool_name=call.name,
                arguments=dict(call.arguments),
                confidence=1.0,
                tool_domain=tool.domain,
                topic=tool.topic,
            )
        )
    return ToolMemoryHintResult(hints=tuple(hints), rejections=())


def _retrieval_quality(
    task: VehicleTask,
    context: VehicleMemoryContext | None,
    *,
    hint_result: ToolMemoryHintResult | None = None,
) -> dict[str, Any] | None:
    if context is None:
        return None
    if context.metadata.get("recall_at_k_applicable") is False:
        return None
    leaves = tuple(
        dict.fromkeys(
            leaf.casefold()
            for call in task.gold_calls
            for value in call.arguments.values()
            for leaf in _argument_leaves(value)
            if leaf
        )
    )
    content = context.content.casefold()
    matched = [leaf for leaf in leaves if leaf in content]
    hint_content = json.dumps(
        [
            {
                "tool_name": hint.tool_name,
                "arguments": hint.arguments,
            }
            for hint in (hint_result.hints if hint_result is not None else ())
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).casefold()
    pipeline_matched = [
        leaf for leaf in leaves if leaf in content or leaf in hint_content
    ]
    metadata = context.metadata
    selected_count = int(
        metadata.get(
            "fact_memory_retrieval_selected_count",
            metadata.get(
                "tool_memory_retrieval_selected_count",
                metadata.get(
                    "retrieval_selected_count",
                    int(bool(context.content)),
                ),
            ),
        )
    )
    return {
        "gold_argument_leaf_count": len(leaves),
        "recalled_argument_leaf_count": len(matched),
        "recall_at_k": len(matched) / len(leaves) if leaves else 1.0,
        "all_gold_arguments_recalled": len(matched) == len(leaves),
        "pipeline_recalled_argument_leaf_count": len(pipeline_matched),
        "pipeline_recall_at_k": (
            len(pipeline_matched) / len(leaves) if leaves else 1.0
        ),
        "all_gold_arguments_available_to_agent": (len(pipeline_matched) == len(leaves)),
        "selected_count": selected_count,
        "context_tokens": int(
            metadata.get(
                "fact_memory_retrieval_tokens",
                metadata.get(
                    "tool_memory_retrieval_tokens",
                    metadata.get("retrieval_tokens", 0),
                ),
            )
        ),
        "latency_ms": int(
            metadata.get(
                "fact_memory_retrieval_latency_ms",
                metadata.get("tool_memory_retrieval_latency_ms", 0),
            )
        ),
    }


def _argument_leaves(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(leaf for item in value.values() for leaf in _argument_leaves(item))
    if isinstance(value, (list, tuple)):
        return tuple(leaf for item in value for leaf in _argument_leaves(item))
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return (str(value).casefold(),)
    return (str(value).strip(),)


def _agent_privacy_metrics(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    reports = [
        item.get("metadata", {}).get("privacy", {})
        for record in records
        for item in record.get("model_trace", ())
        if item.get("metadata", {}).get("privacy", {}).get("destination") == "cloud"
    ]
    before = sum(int(report.get("sensitive_chars_before", 0)) for report in reports)
    after = sum(int(report.get("sensitive_chars_after", 0)) for report in reports)
    return {
        "cloud_calls": len(reports),
        "cloud_transmitted_characters": sum(
            int(report.get("output_chars", 0)) for report in reports
        ),
        "cloud_detected_sensitive_spans": sum(
            int(report.get("detected_count", 0)) for report in reports
        ),
        "cloud_redacted_sensitive_spans": sum(
            int(report.get("redacted_count", 0)) for report in reports
        ),
        "cloud_sensitive_characters_before": before,
        "cloud_sensitive_characters_after": after,
        "cloud_exposed_character_rate": after / before if before else 0.0,
        "gold_redaction_risk_tasks": sum(
            bool(
                record.get("context", {})
                .get("gold_redaction_audit", {})
                .get("answer_loss_risk")
            )
            for record in records
        ),
    }


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _mean(values) -> float:
    items = [float(value) for value in values]
    return statistics.fmean(items) if items else 0


def _initialize_agent_artifacts(
    artifact_dir: Path,
    *,
    run_id: str,
    dataset: VehicleBenchmarkDataset,
    scenario_index: int,
    profiles: tuple[str, ...],
    task_ids: tuple[str, ...],
    agent_model: AgentModel,
    memory_manifest: dict[str, Any] | None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": dataset.manifest.as_dict(),
        "scenario_index": scenario_index,
        "profiles": list(profiles),
        "task_ids": list(task_ids),
        "agent": {
            "backend": agent_model.backend,
            "model_id": agent_model.model_id,
            "prompt_version": VEHICLE_AGENT_PROMPT_VERSION,
        },
        "isolation": {
            "raw_history_included": False,
            "host_tools_available": False,
            "simulator_only": True,
        },
        "memory_snapshot": memory_manifest,
    }
    _write_json_atomic(artifact_dir / "manifest.json", _safe_artifact(manifest))
    _write_json_atomic(
        artifact_dir / "metrics.json",
        _safe_artifact(vehicle_agent_metrics([], profiles)),
    )
    (artifact_dir / "cases.jsonl").write_text("", encoding="utf-8")


def _load_agent_checkpoint(
    artifact_dir: Path,
    *,
    dataset: VehicleBenchmarkDataset,
    scenario_index: int,
    profiles: tuple[str, ...],
    task_ids: tuple[str, ...],
    agent_model: AgentModel,
    memory_manifest: dict[str, Any] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    manifest_path = artifact_dir / "manifest.json"
    cases_path = artifact_dir / "cases.jsonl"
    if not manifest_path.is_file() or not cases_path.is_file():
        raise ValueError(f"Vehicle evaluation checkpoint is incomplete: {artifact_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "benchmark": dataset.manifest.as_dict(),
        "scenario_index": scenario_index,
        "profiles": list(profiles),
        "task_ids": list(task_ids),
        "agent": {
            "backend": agent_model.backend,
            "model_id": agent_model.model_id,
            "prompt_version": VEHICLE_AGENT_PROMPT_VERSION,
        },
        "memory_cache_key": (
            memory_manifest.get("cache_key") if memory_manifest is not None else None
        ),
    }
    actual = {
        "benchmark": manifest.get("benchmark"),
        "scenario_index": manifest.get("scenario_index"),
        "profiles": manifest.get("profiles"),
        "task_ids": manifest.get("task_ids"),
        "agent": manifest.get("agent"),
        "memory_cache_key": (
            manifest.get("memory_snapshot", {}).get("cache_key")
            if manifest.get("memory_snapshot") is not None
            else None
        ),
    }
    if actual != expected:
        raise ValueError(
            "Vehicle evaluation checkpoint configuration mismatch: "
            f"expected={expected}, actual={actual}"
        )

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(
        cases_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = (str(record["profile"]), str(record["task_id"]))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid vehicle checkpoint record at line {line_number}"
            ) from exc
        if key[0] not in profiles or key[1] not in task_ids:
            raise ValueError(f"Unexpected vehicle checkpoint record: {key}")
        records[key] = record
    return records


def _append_agent_checkpoint(
    artifact_dir: Path,
    record: dict[str, Any],
) -> None:
    safe_record = _safe_artifact(record)
    with (artifact_dir / "cases.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def _write_agent_progress(
    artifact_dir: Path,
    *,
    status: str,
    metrics: dict[str, Any],
) -> None:
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(artifact_dir / "metrics.json", _safe_artifact(metrics))


def _finalize_agent_artifacts(
    artifact_dir: Path,
    *,
    status: str,
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    write_vehicle_report_artifacts(
        artifact_dir,
        metrics=metrics,
        records=records,
    )
    _write_agent_progress(artifact_dir, status=status, metrics=metrics)


def write_vehicle_report_artifacts(
    artifact_dir: Path,
    *,
    metrics: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_records = [_safe_artifact(record) for record in records]
    cases_path = artifact_dir / "cases.jsonl"
    temporary = cases_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in safe_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(cases_path)
    with (artifact_dir / "diagnostics.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in safe_records:
            payload = {
                "profile": record["profile"],
                "task_id": record["task_id"],
                "scenario_index": record["scenario_index"],
                "reasoning_type": record["reasoning_type"],
                "status": record["status"],
                "diagnostics": record.get("diagnostics", {}),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _write_vehicle_tsv(artifact_dir / "results.tsv", metrics)
    _write_vehicle_reasoning_tsv(
        artifact_dir / "reasoning_types.tsv",
        metrics,
    )
    _write_vehicle_markdown(artifact_dir / "results.md", metrics)
    for metric, title in (
        ("exact_state_match", "Exact State Match"),
        ("state_f1", "State F1"),
        ("tool_f1", "Tool F1"),
        ("argument_exact_match", "Argument Exact Match"),
        ("retrieval_recall_at_k", "Retrieval Recall@k"),
        ("cloud_exposed_character_rate", "Cloud sensitive-character exposure"),
    ):
        _write_vehicle_svg(
            artifact_dir / f"{metric}.svg",
            title,
            [
                (profile, float(values.get(metric, 0.0)))
                for profile, values in metrics["profiles"].items()
                if values.get(metric) is not None
            ],
        )
    privacy_report = _artifact_privacy_audit(artifact_dir)
    metrics["artifact_privacy_audit"] = privacy_report
    _write_json_atomic(artifact_dir / "privacy_audit.json", privacy_report)
    _write_json_atomic(artifact_dir / "metrics.json", _safe_artifact(metrics))
    _write_vehicle_tsv(artifact_dir / "results.tsv", metrics)
    _write_vehicle_markdown(artifact_dir / "results.md", metrics)


def _ordered_records(
    records: dict[tuple[str, str], dict[str, Any]],
    *,
    profiles: tuple[str, ...],
    tasks: Sequence[VehicleTask],
) -> list[dict[str, Any]]:
    return [
        records[(profile, task.id)]
        for profile in profiles
        for task in tasks
        if (profile, task.id) in records
    ]


def _validated_run_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid vehicle evaluation run ID: {value}") from exc
    if str(parsed) != value:
        raise ValueError(f"Vehicle evaluation run ID must be canonical: {value}")
    return value


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_artifact(payload: Any) -> Any:
    safe, _ = redact_data_for_cloud(
        payload,
        preserve_opaque_keys=(
            "encrypted_content",
            "run_id",
            "child_run_id",
            "consolidation_run_id",
            "consolidation_run_ids",
            "dataset_sha256",
            "tool_schema_sha256",
            "cache_key",
            "task_id",
        ),
    )
    return redact_data(safe)


def _write_vehicle_tsv(path: Path, metrics: dict[str, Any]) -> None:
    columns = (
        "tasks",
        "completion_rate",
        "exact_state_match",
        "state_recall",
        "state_precision",
        "state_f1",
        "value_f1",
        "tool_f1",
        "argument_exact_match",
        "retrieval_recall_at_k",
        "retrieval_hit_rate",
        "retrieval_selected_count_mean",
        "retrieval_context_tokens_mean",
        "retrieval_latency_ms_mean",
        "selector_tool_count_mean",
        "execution_hint_count_mean",
        "execution_hint_rejections",
        "unnecessary_vehicle_calls",
        "suppressed_duplicate_calls",
        "extra_discovery_calls",
        "model_latency_ms_mean",
        "model_latency_ms_p95",
        "input_tokens",
        "output_tokens",
        "estimated_agent_cost_usd",
        "cloud_transmitted_characters",
        "cloud_exposed_character_rate",
        "gold_redaction_risk_tasks",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("profile", *columns))
        for profile, values in metrics["profiles"].items():
            writer.writerow((profile, *(values.get(column, "") for column in columns)))


def _write_vehicle_reasoning_tsv(
    path: Path,
    metrics: dict[str, Any],
) -> None:
    columns = (
        "tasks",
        "exact_state_match",
        "state_f1",
        "value_f1",
        "tool_f1",
        "argument_exact_match",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("profile", "reasoning_type", *columns))
        for profile, values in metrics["profiles"].items():
            for reasoning_type, row in values.get("reasoning_types", {}).items():
                writer.writerow(
                    (
                        profile,
                        reasoning_type,
                        *(row.get(column, "") for column in columns),
                    )
                )


def _write_vehicle_markdown(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# VehicleMemBench Evaluation",
        "",
        "| Profile | Tasks | ESM | State P/R/F1 | Value F1 | Tool F1 | "
        "Arg Exact | Recall@k | Context tokens | Extra calls | Cloud exposure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile, values in metrics["profiles"].items():
        recall = values["retrieval_recall_at_k"]
        recall_text = "N/A" if recall is None else f"{recall:.3f}"
        lines.append(
            f"| {profile} "
            f"| {values['tasks']} "
            f"| {values['exact_state_match']:.3f} "
            f"| {values['state_precision']:.3f}/"
            f"{values['state_recall']:.3f}/{values['state_f1']:.3f} "
            f"| {values['value_f1']:.3f} "
            f"| {values['tool_f1']:.3f} "
            f"| {values['argument_exact_match']:.3f} "
            f"| {recall_text} "
            f"| {values['retrieval_context_tokens_mean']:.1f} "
            f"| {values['unnecessary_vehicle_calls']} "
            f"| {values['cloud_exposed_character_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Workflow stages",
            "",
            "Retrieved context tokens below are diagnostic and are already "
            "reflected in later Agent input; do not add them to billed Agent "
            "tokens.",
            "",
            "| Profile | Base context | Wiki context | Online retrieval "
            "latency (ms) | Agent tokens | Agent latency (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile, values in metrics["profiles"].items():
        stages = values.get("workflow_stages", {})
        online = stages.get("online_memory_retrieval", {})
        agent = stages.get("quiz_agent", {})
        lines.append(
            f"| {profile} "
            f"| {int(online.get('base_context_tokens', 0))} "
            f"| {int(online.get('wiki_read_context_tokens', 0))} "
            f"| {int(online.get('observed_latency_ms', 0))} "
            f"| {int(agent.get('total_tokens', 0))} "
            f"| {int(agent.get('model_latency_ms', 0))} |"
        )
    suite_stages = metrics.get("workflow_stages")
    if suite_stages:
        lines.extend(
            [
                "",
                "### Suite provider stages",
                "",
                "| Stage | Calls | Input tokens | Output tokens | "
                "Provider latency (ms) | Estimated cost (USD) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for stage in (
            "idle_memory_llm",
            "idle_memory_embedding",
            "online_memory_retrieval",
        ):
            values = suite_stages.get(stage, {})
            lines.append(
                f"| {stage} "
                f"| {int(values.get('calls', 0))} "
                f"| {int(values.get('input_tokens', 0))} "
                f"| {int(values.get('output_tokens', 0))} "
                f"| {int(values.get('latency_ms', 0))} "
                f"| {float(values.get('estimated_cost_usd', 0.0)):.8f} |"
            )
    lines.extend(["", "## Failure taxonomy", ""])
    for profile, values in metrics["profiles"].items():
        taxonomy = ", ".join(
            f"{name}={count}"
            for name, count in values.get("failure_taxonomy", {}).items()
        )
        lines.append(f"- `{profile}`: {taxonomy or 'none'}")
    audit = metrics.get("artifact_privacy_audit")
    if audit is not None:
        lines.extend(
            [
                "",
                "## Privacy audit",
                "",
                f"- Residual sensitive spans: {audit['detected_count']}",
                "- Gold redaction-risk tasks are reported per profile.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_vehicle_svg(
    path: Path,
    title: str,
    values: Sequence[tuple[str, float]],
) -> None:
    width = 920
    left = 300
    chart_width = 520
    row_height = 34
    height = 70 + row_height * len(values)
    rows = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#fff"/>',
        (
            f'<text x="20" y="30" font-family="sans-serif" '
            f'font-size="19">{escape(title)}</text>'
        ),
    ]
    for index, (label, raw_value) in enumerate(values):
        value = min(1.0, max(0.0, raw_value))
        y = 48 + index * row_height
        rows.extend(
            [
                (
                    f'<text x="20" y="{y + 17}" font-family="monospace" '
                    f'font-size="12">{escape(label)}</text>'
                ),
                (
                    f'<rect x="{left}" y="{y}" width="{chart_width}" '
                    'height="20" fill="#e5e7eb"/>'
                ),
                (
                    f'<rect x="{left}" y="{y}" '
                    f'width="{chart_width * value:.2f}" height="20" '
                    'fill="#2563eb"/>'
                ),
                (
                    f'<text x="{left + chart_width + 8}" y="{y + 16}" '
                    f'font-family="sans-serif" font-size="12">{value:.3f}</text>'
                ),
            ]
        )
    rows.append("</svg>")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _artifact_privacy_audit(artifact_dir: Path) -> dict[str, Any]:
    reports = []
    for path in artifact_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".jsonl", ".tsv", ".md"}:
            continue
        if path.name == "privacy_audit.json":
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            payload: Any = json.loads(text)
        elif path.suffix == ".jsonl":
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif path.suffix == ".tsv":
            payload = [
                cell
                for row in csv.reader(text.splitlines(), delimiter="\t")
                for cell in row
                if not _is_numeric_cell(cell)
            ]
        else:
            payload = text
        reports.append(inspect_data_privacy(_mask_opaque_identifiers(payload)))
    categories: Counter[str] = Counter()
    for report in reports:
        categories.update(report.category_counts)
    return {
        "files_scanned": len(reports),
        "detected_count": sum(report.detected_count for report in reports),
        "sensitive_characters": sum(
            report.sensitive_chars_before for report in reports
        ),
        "category_counts": dict(sorted(categories.items())),
        "passed": not any(report.detected_count for report in reports),
    }


def _mask_opaque_identifiers(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _mask_opaque_identifiers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_opaque_identifiers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_mask_opaque_identifiers(item) for item in value)
    if isinstance(value, str):
        return _OPAQUE_UUID_PATTERN.sub("[OPAQUE_UUID]", value)
    return value


def _is_numeric_cell(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
