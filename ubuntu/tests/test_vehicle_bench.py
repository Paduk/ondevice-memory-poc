from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from palmclaw_ubuntu.cli import build_parser
from palmclaw_ubuntu.models import (
    AgentResponse,
    FactQueryContext,
    ToolCall,
    ToolMemoryRecord,
)
from palmclaw_ubuntu.providers import FakeEmbeddingModel, ScriptedAgentModel
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.vehicle_bench import (
    OFFICIAL_UPSTREAM_COMMIT,
    OracleGateAnnotations,
    OracleGateLabel,
    OracleRetrievalAnnotations,
    OracleRetrievalLabel,
    VehicleMemoryContext,
    build_vehicle_fact_ontology,
    load_vehicle_benchmark,
    run_agent_evaluation,
    run_offline_smoke,
    run_vehicle_evaluation_suite,
    run_vehicle_ontology_matcher_evaluation,
    summarize_provider_calls,
    vehicle_tool_schema_sha256,
)
from palmclaw_ubuntu.vehicle_bench.dataset import VehicleBenchValidationError
from palmclaw_ubuntu.vehicle_bench.runner import (
    _aggregate_fact_quality,
    _aggregate_online_memory_retrieval,
    _aggregate_wiki_traversal,
    _artifact_privacy_audit,
    _safe_artifact,
    _task_diagnostics,
)
from palmclaw_ubuntu.vehicle_bench.suite import (
    _read_json,
    _sum_amem_usage,
    _sum_provider_usage,
    _write_json,
)


class _InterruptingAgentModel:
    backend = "fake"
    model_id = "scripted-agent"
    prompt_version = "agent-v1"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        del messages, tools
        self.calls += 1
        if self.calls == 1:
            return AgentResponse(content="No action.")
        raise KeyboardInterrupt


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_vehicle_artifacts_preserve_opaque_uuid_identifiers(
    tmp_path: Path,
) -> None:
    identifier = "b42a5250-8818-4182-9312-62bf682b212e"
    payload = {
        "run_id": identifier,
        "scenarios": {"1": {"child_run_id": identifier}},
        "memory_trace": {
            "daily_steps": [
                {"consolidation_run_id": identifier},
            ],
            "consolidation_run_ids": [identifier],
        },
    }
    path = tmp_path / "manifest.json"

    _write_json(path, payload)

    assert _read_json(path) == payload
    assert _safe_artifact(payload) == payload


def test_vehicle_privacy_audit_ignores_uuid_phone_false_positive(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "manifest.json",
        {"run_id": "a25b7684-1333-4318-9838-3cd7dd010973"},
    )

    report = _artifact_privacy_audit(tmp_path)

    assert report["passed"] is True
    assert report["detected_count"] == 0


def _mini_benchmark(root: Path) -> Path:
    schemas = [
        {
            "name": "carcontrol_HUD_switch",
            "description": "Turn HUD on or off.",
            "parameters": {
                "type": "object",
                "properties": {"switch": {"type": "boolean"}},
                "required": ["switch"],
            },
        },
        {
            "name": "carcontrol_HUD_set_brightness_level",
            "description": "Set HUD brightness.",
            "parameters": {
                "type": "object",
                "properties": {"level": {"type": "integer"}},
                "required": ["level"],
            },
        },
    ]
    qa = {
        "related_to_vehicle_preference": [
            {
                "gold_memory": "[2025-01-01] Gary prefers HUD brightness 8.",
                "reasoning_type": "preference_conflict",
                "query": "Apply Gary's HUD preference.",
                "new_answer": [
                    "carcontrol_HUD_switch(switch=true)",
                    "carcontrol_HUD_set_brightness_level(level=8)",
                ],
            }
        ]
    }
    _write(
        root / "evaluation/functions_schema.json",
        json.dumps(schemas),
    )
    _write(
        root / "benchmark/qa_data/qa_1.json",
        json.dumps(qa),
    )
    _write(
        root / "benchmark/history/history_1.txt",
        "[2025-01-01 08:00] Gary: Set HUD brightness to 8.\n",
    )
    _write(root / "environment/__init__.py", "")
    _write(
        root / "environment/utils.py",
        'modules_dict = {"HUD": "Head-up display"}\n',
    )
    _write(
        root / "environment/vehicleworld.py",
        """
from .utils import modules_dict


class HUD:
    def __init__(self):
        self.enabled = False
        self.brightness = 5

    def carcontrol_HUD_switch(self, switch):
        self.enabled = switch
        return {"success": True, "current_state": self.to_dict()}

    def carcontrol_HUD_set_brightness_level(self, level):
        self.brightness = level
        return {"success": True, "current_state": self.to_dict()}

    def to_dict(self):
        return {
            "enabled": {
                "value": self.enabled,
                "description": "HUD power",
                "type": "bool",
            },
            "brightness": {
                "value": self.brightness,
                "description": "HUD brightness",
                "type": "int",
            },
        }


class VehicleWorld:
    def __init__(self):
        self.HUD = HUD()

    def to_dict(self):
        return {
            "HUD": {
                "value": self.HUD.to_dict(),
                "description": modules_dict["HUD"],
                "type": "HUD",
            }
        }
""".lstrip(),
    )
    _write(
        root / "evaluation/eval_utils.py",
        """
import json
from environment.utils import modules_dict


def calculate_turn_result(world1, world2, world3, world4):
    del world1, world3
    exact = world2 == world4 and bool(modules_dict)
    return {
        "differences": [] if exact else ["state mismatch"],
        "TP": 2 if exact else 0,
        "FP": 0 if exact else 1,
        "negative_FP": 0 if exact else 1,
        "f1_positive": 1.0 if exact else 0.0,
        "f1_change": 1.0 if exact else 0.0,
    }


def score_tool_calls(pred_calls, ref_calls):
    def key(call):
        return call["name"], json.dumps(call["args"], sort_keys=True)

    exact = [key(call) for call in pred_calls] == [
        key(call) for call in ref_calls
    ]
    size = len(ref_calls)
    return {
        "tp": size if exact else 0,
        "fp": 0 if exact else len(pred_calls),
        "fn": 0 if exact else size,
        "precision": 1.0 if exact else 0.0,
        "recall": 1.0 if exact else 0.0,
        "f1": 1.0 if exact else 0.0,
    }
""".lstrip(),
    )
    return root


def test_vehicle_loader_validates_schema_pairing_and_gold_calls(tmp_path: Path):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")

    dataset = load_vehicle_benchmark(root, strict=False)

    assert dataset.manifest.scenario_count == 1
    assert dataset.manifest.task_count == 1
    assert dataset.manifest.tool_count == 2
    assert len(dataset.manifest.dataset_sha256) == 64
    assert len(dataset.manifest.tool_schema_sha256) == 64
    calls = dataset.scenario(1).tasks[0].gold_calls
    assert calls[0].arguments == {"switch": True}
    assert calls[1].arguments == {"level": 8}


def test_vehicle_offline_smoke_uses_registry_and_official_scorer(tmp_path: Path):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)

    result = run_offline_smoke(dataset, scenario_index=1)

    assert result.passed is True
    assert result.task_count == 1
    assert result.tasks[0].exact_state_match is True
    assert result.tasks[0].state_f1 == 1
    assert result.tasks[0].value_f1 == 1
    assert result.tasks[0].tool_f1 == 1
    assert str(root) not in sys.path


def test_vehicle_loader_rejects_mismatched_scenarios(tmp_path: Path):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    (root / "benchmark/qa_data/qa_1.json").rename(root / "benchmark/qa_data/qa_2.json")

    with pytest.raises(VehicleBenchValidationError, match="scenario mismatch"):
        load_vehicle_benchmark(root, strict=False)


def test_vehicle_cli_exposes_pinned_offline_smoke_command():
    args = build_parser().parse_args(
        [
            "eval",
            "vehicle",
            "--benchmark-root",
            "/tmp/VehicleMemBench",
        ]
    )

    assert args.scenario == 1
    assert args.scenario_limit == 1
    assert args.task_limit == 10
    assert args.mode == "offline"
    assert args.max_tool_rounds == 10
    assert args.expected_commit is None
    assert args.allow_unpinned is False
    assert args.resume_run is None


def test_vehicle_matcher_only_evaluation_uses_history_and_no_generation_llm(
    tmp_path: Path,
):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    definition = {
        "schema_version": "vehicle-fact-ontology-v1",
        "source_policy": "Tool interface only; excludes QA, History, Gold",
        "source_schema_sha256": vehicle_tool_schema_sha256(
            dataset.tool_schemas
        ),
        "tools": {
            "carcontrol_HUD_switch": {
                "capability": "system.power",
                "target": "hud",
                "arguments": {"switch": "value"},
            },
            "carcontrol_HUD_set_brightness_level": {
                "capability": "display.brightness",
                "target": "hud",
                "arguments": {"level": "value"},
            },
        },
    }
    ontology = build_vehicle_fact_ontology(dataset.tool_schemas, definition)
    labels_path = tmp_path / "matcher-labels.json"
    _write(
        labels_path,
        json.dumps(
            {
                "version": "vehiclemembench-ontology-matcher-labels-v1",
                "benchmark": {
                    "dataset_sha256": dataset.manifest.dataset_sha256,
                    "tool_schema_sha256": dataset.manifest.tool_schema_sha256,
                    "ontology_sha256": ontology.ontology_sha256,
                },
                "scenario_index": 1,
                "cases": [
                    {
                        "task_id": "vehicle-01-00",
                        "evidence_quote": "Set HUD brightness to 8.",
                        "expected_tool_name": (
                            "carcontrol_HUD_set_brightness_level"
                        ),
                    }
                ],
            }
        ),
    )
    embedding = FakeEmbeddingModel()

    result = run_vehicle_ontology_matcher_evaluation(
        dataset,
        embedding_model=embedding,
        scenario_index=1,
        labels_path=labels_path,
        batch_turn_limit=32,
        batch_token_limit=4_096,
        output_root=tmp_path / "artifacts",
        ontology=ontology,
    )

    assert result.status == "completed"
    assert result.metrics["top2_recall"] == 1
    assert result.metrics["batch_top8_recall"] == 1
    assert result.metrics["target_binding_top2_coverage"] == 1
    assert result.metrics["generation_model_calls"] == 0
    assert result.cases[0]["expected_capability_id"] == "display.brightness"
    assert result.artifact_dir is not None
    assert {path.name for path in result.artifact_dir.iterdir()} == {
        "cases.jsonl",
        "manifest.json",
        "metrics.json",
        "results.md",
    }
    assert len(embedding.requests) == 2


def test_vehicle_cli_exposes_matcher_only_command():
    args = build_parser().parse_args(
        [
            "eval",
            "vehicle-matcher",
            "--benchmark-root",
            "/tmp/VehicleMemBench",
        ]
    )

    assert args.scenario == 6
    assert args.labels is None
    assert args.embedding_model is None
    assert args.allow_unpinned is False


def test_vehicle_agent_dynamic_tools_gold_memory_and_artifacts(tmp_path: Path):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    model = ScriptedAgentModel(
        [
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="discover-hud",
                        name="list_module_tools",
                        arguments={"module_name": "HUD"},
                    ),
                ),
            ),
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="hud-switch",
                        name="carcontrol_HUD_switch",
                        arguments={"switch": True},
                    ),
                    ToolCall(
                        id="hud-brightness",
                        name="carcontrol_HUD_set_brightness_level",
                        arguments={"level": 8},
                    ),
                ),
            ),
            AgentResponse(content="HUD preference applied."),
        ]
    )

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("gold_memory",),
        scenario_index=1,
        output_root=tmp_path / "artifacts",
    )

    task = result.tasks[0]
    assert result.status == "completed"
    assert task["status"] == "completed"
    assert task["score"]["exact_state_match"] is True
    assert task["score"]["tool_score"]["f1"] == 1
    assert task["loaded_modules"] == ["HUD"]
    assert task["context"]["raw_history_included"] is False
    assert task["tool_boundary"]["host_tools_available"] is False
    assert result.metrics["profiles"]["gold_memory"]["exact_state_match"] == 1
    assert [tool.name for tool in model.requests[0][1]] == ["list_module_tools"]
    assert {tool.name for tool in model.requests[1][1]} == {
        "list_module_tools",
        "carcontrol_HUD_switch",
        "carcontrol_HUD_set_brightness_level",
    }
    assert all(
        tool.name not in {"file_read", "file_write", "web_fetch"}
        for request in model.requests
        for tool in request[1]
    )
    assert model.requests[0][1][0].strict is True
    assert all(
        tool.strict is False
        for tool in model.requests[1][1]
        if tool.name.startswith("carcontrol_")
    )
    initial_system = model.requests[0][0][0].content
    assert dataset.scenario(1).tasks[0].gold_memory in initial_system
    history = dataset.scenario(1).history_path.read_text(encoding="utf-8").strip()
    assert history not in initial_system

    assert result.artifact_dir is not None
    assert {
        "manifest.json",
        "metrics.json",
        "cases.jsonl",
        "diagnostics.jsonl",
        "results.tsv",
        "reasoning_types.tsv",
        "results.md",
        "exact_state_match.svg",
        "state_f1.svg",
        "tool_f1.svg",
        "argument_exact_match.svg",
        "retrieval_recall_at_k.svg",
        "cloud_exposed_character_rate.svg",
        "privacy_audit.json",
    } == {path.name for path in result.artifact_dir.iterdir()}
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8") for path in result.artifact_dir.iterdir()
    )
    assert history not in artifact_text
    assert '"raw_history_included": false' in artifact_text
    assert task["diagnostics"]["primary_outcome"] == "success"
    assert "state_precision" in result.metrics["profiles"]["gold_memory"]
    assert "state_recall" in result.metrics["profiles"]["gold_memory"]
    assert result.metrics["artifact_privacy_audit"]["passed"] is True


def test_vehicle_schema_patch_wires_selector_and_validated_hints_to_agent(
    tmp_path: Path,
):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    model = ScriptedAgentModel(
        [
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="hud-switch",
                        name="carcontrol_HUD_switch",
                        arguments={"switch": True},
                    ),
                    ToolCall(
                        id="hud-brightness",
                        name="carcontrol_HUD_set_brightness_level",
                        arguments={"level": 8},
                    ),
                ),
            ),
            AgentResponse(content="HUD preference applied."),
        ]
    )
    record = ToolMemoryRecord(
        id="hud-memory",
        session_id="memory-session",
        record_key="hud-key",
        user_id="default_user",
        tool_domain="hud",
        topic="brightness_level",
        scope="global",
        scope_key="default_user",
        conditions={"person": "Gary"},
        value={"level": 8},
        memory_type="preference",
        status="active",
        confidence=0.98,
        version=1,
        supersedes_id=None,
        merged_into_id=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )

    def resolve(profile, query):
        assert profile == "cloud_schema_patch"
        assert query == dataset.scenario(1).tasks[0].query
        return VehicleMemoryContext(
            content="domain=hud topic=brightness_level value={level:8}",
            metadata={
                "tool_memory_retrieval_selected_count": 1,
                "tool_memory_route": {
                    "routes": [
                        {
                            "tool_names": [
                                "carcontrol_HUD_switch",
                                "carcontrol_HUD_set_brightness_level",
                            ]
                        }
                    ]
                },
            },
            trace={"strategy": "schema_patch"},
            records=(record,),
        )

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("cloud_schema_patch",),
        memory_resolver=resolve,
    )

    task = result.tasks[0]
    initial_tools = [definition.name for definition in model.requests[0][1]]
    system_prompt = model.requests[0][0][0].content
    assert initial_tools == [
        "list_module_tools",
        "carcontrol_HUD_switch",
        "carcontrol_HUD_set_brightness_level",
    ]
    assert task["discovery_calls"] == 0
    assert task["loaded_modules"] == ["HUD"]
    assert task["score"]["exact_state_match"] is True
    assert task["tool_boundary"]["router_preloaded_tools"] == [
        "carcontrol_HUD_switch",
        "carcontrol_HUD_set_brightness_level",
    ]
    assert task["context"]["tool_memory_execution_hint_count"] == 1
    assert "Schema-validated Memory-to-Tool Hints" in system_prompt
    assert '"tool_name":"carcontrol_HUD_set_brightness_level"' in system_prompt
    assert '"arguments":{"level":8}' in system_prompt
    assert "all independent requested settings" in system_prompt


def test_vehicle_agent_suppresses_an_exact_completed_call(tmp_path: Path):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    model = ScriptedAgentModel(
        [
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="discover-hud",
                        name="list_module_tools",
                        arguments={"module_name": "HUD"},
                    ),
                ),
            ),
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="hud-switch",
                        name="carcontrol_HUD_switch",
                        arguments={"switch": True},
                    ),
                    ToolCall(
                        id="hud-brightness",
                        name="carcontrol_HUD_set_brightness_level",
                        arguments={"level": 8},
                    ),
                ),
            ),
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="hud-brightness-duplicate",
                        name="carcontrol_HUD_set_brightness_level",
                        arguments={"level": 8},
                    ),
                ),
            ),
            AgentResponse(content="done"),
        ]
    )

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("gold_memory",),
    )

    task = result.tasks[0]
    assert task["score"]["exact_state_match"] is True
    assert task["suppressed_duplicate_calls"] == 1
    assert len(task["predicted_calls"]) == 2
    assert task["tool_trace"][-1]["suppressed_duplicate"] is True
    assert task["diagnostics"]["suppressed_duplicate_calls"] == 1


def test_successful_memory_profile_is_not_classified_as_retrieval_failure():
    diagnostics = _task_diagnostics(
        {
            "profile": "cloud_schema_patch",
            "status": "completed",
            "score": {
                "exact_state_match": True,
                "tool_score": {"fp": 0, "fn": 0},
            },
            "argument_exact_match": True,
            "context": {
                "retrieval_metadata": {
                    "tool_memory_retrieval_selected_count": 0,
                },
                "selector_tool_names": ["carcontrol_navigation_stop"],
                "tool_memory_execution_hint_rejection_count": 0,
            },
            "predicted_calls": [
                {"name": "carcontrol_navigation_stop", "args": {}}
            ],
            "reference_calls": [
                {"name": "carcontrol_navigation_stop", "args": {}}
            ],
            "tool_trace": [],
            "discovery_calls": 0,
        }
    )

    assert diagnostics["primary_outcome"] == "success"
    assert "retrieval_empty" not in diagnostics["failure_codes"]


def test_vehicle_oracle_tool_profile_exposes_schemas_without_gold_arguments(
    tmp_path: Path,
):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    model = ScriptedAgentModel(
        [
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="hud-switch",
                        name="carcontrol_HUD_switch",
                        arguments={"switch": True},
                    ),
                    ToolCall(
                        id="hud-brightness",
                        name="carcontrol_HUD_set_brightness_level",
                        arguments={"level": 8},
                    ),
                ),
            ),
            AgentResponse(content="HUD preference applied."),
        ]
    )

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("oracle_tool_gold_memory",),
        scenario_index=1,
    )

    task = result.tasks[0]
    reference_names = {
        call.name for call in dataset.scenario(1).tasks[0].gold_calls
    }
    assert task["score"]["exact_state_match"] is True
    assert task["discovery_calls"] == 0
    assert task["loaded_modules"] == ["HUD"]
    assert task["tool_boundary"] == {
        "initial_tools": [
            "carcontrol_HUD_switch",
            "carcontrol_HUD_set_brightness_level",
        ],
        "host_tools_available": False,
        "simulator": "VehicleWorld",
        "oracle_tool_boundary": True,
        "gold_arguments_exposed": False,
    }
    assert {tool.name for tool in model.requests[0][1]} == reference_names
    system_prompt = model.requests[0][0][0].content
    assert "preselected for this diagnostic" in system_prompt
    assert "list_module_tools" not in system_prompt
    assert task["context"]["sources"] == ["gold_memory", "query"]


def test_vehicle_oracle_patch_resolver_receives_names_not_arguments(
    tmp_path: Path,
):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    model = ScriptedAgentModel([AgentResponse(content="No action.")])
    requests = []

    def resolve(profile, query, tool_names):
        requests.append((profile, query, tool_names))
        return VehicleMemoryContext(
            content="",
            metadata={},
            trace={"strategy": "schema_patch"},
        )

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("oracle_tool_schema_patch",),
        memory_resolver=resolve,
    )

    task = dataset.scenario(1).tasks[0]
    assert requests == [
        (
            "oracle_tool_schema_patch",
            task.query,
            tuple(call.name for call in task.gold_calls),
        )
    ]
    system_prompt = model.requests[0][0][0].content
    assert task.gold_memory not in system_prompt
    assert "brightness 8" not in system_prompt
    assert result.tasks[0]["tool_boundary"]["gold_arguments_exposed"] is False


def test_vehicle_oracle_retrieval_passes_reviewed_label_without_gold_arguments(
    tmp_path: Path,
):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    model = ScriptedAgentModel([AgentResponse(content="No action.")])
    source = tmp_path / "oracle.json"
    source.write_text("{}", encoding="utf-8")
    task = dataset.scenario(1).tasks[0]
    annotations = OracleRetrievalAnnotations(
        version="vehiclemembench-oracle-retrieval-v1",
        dataset_sha256=dataset.manifest.dataset_sha256,
        tool_schema_sha256=dataset.manifest.tool_schema_sha256,
        scenario_indices=(1,),
        labels={
            task.id: OracleRetrievalLabel(
                task_id=task.id,
                record_status="record_absent",
            )
        },
        source_path=source,
    )
    requests = []

    def resolve(profile, query, *, oracle_retrieval_label):
        requests.append((profile, query, oracle_retrieval_label))
        return VehicleMemoryContext(
            content="",
            metadata={
                "oracle_retrieval": True,
                "oracle_retrieval_record_status": "record_absent",
            },
            trace={"strategy": "fact_patch"},
            fact_query_context=FactQueryContext(),
        )

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("oracle_retrieval_fact_patch",),
        memory_resolver=resolve,
        oracle_retrieval_annotations=annotations,
    )

    assert requests == [
        (
            "oracle_retrieval_fact_patch",
            task.query,
            annotations.require(task.id),
        )
    ]
    assert task.gold_memory not in model.requests[0][0][0].content
    assert result.tasks[0]["tool_boundary"]["gold_arguments_exposed"] is False


def test_vehicle_oracle_binding_exposes_gold_arguments_only_for_selected_fact(
    tmp_path: Path,
):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    task = dataset.scenario(1).tasks[0]
    source = tmp_path / "oracle.json"
    source.write_text("{}", encoding="utf-8")
    annotations = OracleRetrievalAnnotations(
        version="vehiclemembench-oracle-retrieval-v1",
        dataset_sha256=dataset.manifest.dataset_sha256,
        tool_schema_sha256=dataset.manifest.tool_schema_sha256,
        scenario_indices=(1,),
        labels={
            task.id: OracleRetrievalLabel(
                task_id=task.id,
                record_status="record_present",
            )
        },
        source_path=source,
    )
    model = ScriptedAgentModel(
        [
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="hud-switch",
                        name="carcontrol_HUD_switch",
                        arguments={"switch": True},
                    ),
                    ToolCall(
                        id="hud-brightness",
                        name="carcontrol_HUD_set_brightness_level",
                        arguments={"level": 8},
                    ),
                ),
            ),
            AgentResponse(content="Applied."),
        ]
    )

    def resolve(profile, query, *, oracle_retrieval_label):
        assert profile == "oracle_binding_fact_patch"
        assert query == task.query
        assert oracle_retrieval_label == annotations.require(task.id)
        return VehicleMemoryContext(
            content="- Gary prefers HUD brightness 8.",
            metadata={
                "oracle_retrieval": True,
                "oracle_retrieval_record_status": "record_present",
                "oracle_retrieval_record_ids": ["fact-1"],
                "oracle_retrieval_all_records_selected": True,
            },
            trace={"strategy": "fact_patch"},
            fact_query_context=FactQueryContext(),
        )

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("oracle_binding_fact_patch",),
        memory_resolver=resolve,
        oracle_retrieval_annotations=annotations,
    )

    evaluated = result.tasks[0]
    system_prompt = model.requests[0][0][0].content
    assert evaluated["score"]["exact_state_match"] is True
    assert evaluated["tool_boundary"]["gold_arguments_exposed"] is True
    assert evaluated["context"]["retrieval_metadata"][
        "oracle_binding_applied"
    ] is True
    assert '"arguments":{"switch":true}' in system_prompt
    assert '"arguments":{"level":8}' in system_prompt
    assert task.gold_memory not in system_prompt


def test_vehicle_oracle_binding_does_not_expose_gold_for_absent_fact(
    tmp_path: Path,
):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    task = dataset.scenario(1).tasks[0]
    source = tmp_path / "oracle.json"
    source.write_text("{}", encoding="utf-8")
    annotations = OracleRetrievalAnnotations(
        version="vehiclemembench-oracle-retrieval-v1",
        dataset_sha256=dataset.manifest.dataset_sha256,
        tool_schema_sha256=dataset.manifest.tool_schema_sha256,
        scenario_indices=(1,),
        labels={
            task.id: OracleRetrievalLabel(
                task_id=task.id,
                record_status="record_absent",
            )
        },
        source_path=source,
    )
    model = ScriptedAgentModel([AgentResponse(content="No action.")])

    def resolve(profile, query, *, oracle_retrieval_label):
        return VehicleMemoryContext(
            content="",
            metadata={
                "oracle_retrieval": True,
                "oracle_retrieval_record_status": "record_absent",
                "oracle_retrieval_record_ids": [],
                "oracle_retrieval_all_records_selected": False,
            },
            trace={"strategy": "fact_patch"},
            fact_query_context=FactQueryContext(),
        )

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("oracle_binding_fact_patch",),
        memory_resolver=resolve,
        oracle_retrieval_annotations=annotations,
    )

    evaluated = result.tasks[0]
    system_prompt = model.requests[0][0][0].content
    assert evaluated["tool_boundary"]["gold_arguments_exposed"] is False
    assert evaluated["context"]["retrieval_metadata"][
        "oracle_binding_applied"
    ] is False
    assert "[Schema-validated Memory-to-Tool Hints]" not in system_prompt
    assert task.gold_memory not in system_prompt


def test_vehicle_cumulative_gate_passes_both_labels_and_enables_binding(
    tmp_path: Path,
):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    task = dataset.scenario(1).tasks[0]
    source = tmp_path / "oracle.json"
    source.write_text("{}", encoding="utf-8")
    retrieval = OracleRetrievalAnnotations(
        version="vehiclemembench-oracle-retrieval-v1",
        dataset_sha256=dataset.manifest.dataset_sha256,
        tool_schema_sha256=dataset.manifest.tool_schema_sha256,
        scenario_indices=(1,),
        labels={
            task.id: OracleRetrievalLabel(
                task_id=task.id,
                record_status="record_absent",
            )
        },
        source_path=source,
    )
    gate = OracleGateAnnotations(
        version="vehiclemembench-oracle-gate-v1",
        dataset_sha256=dataset.manifest.dataset_sha256,
        tool_schema_sha256=dataset.manifest.tool_schema_sha256,
        scenario_indices=(1,),
        labels={
            task.id: OracleGateLabel(
                task_id=task.id,
                scenario_index=1,
                candidate_status="gate_recoverable",
                candidate_sha256="a" * 64,
                original_status="review",
                validation_code="semantic_review",
            )
        },
        source_path=source,
    )
    model = ScriptedAgentModel(
        [
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="hud-switch",
                        name="carcontrol_HUD_switch",
                        arguments={"switch": True},
                    ),
                    ToolCall(
                        id="hud-brightness",
                        name="carcontrol_HUD_set_brightness_level",
                        arguments={"level": 8},
                    ),
                ),
            ),
            AgentResponse(content="Applied."),
        ]
    )

    def resolve(
        profile,
        query,
        *,
        oracle_gate_label,
        oracle_retrieval_label,
    ):
        assert profile == "oracle_gate_retrieval_binding_fact_patch"
        assert query == task.query
        assert oracle_gate_label == gate.require(task.id)
        assert oracle_retrieval_label == retrieval.require(task.id)
        return VehicleMemoryContext(
            content="- Gary prefers HUD brightness 8.",
            metadata={
                "oracle_gate": True,
                "oracle_gate_candidate_status": "gate_recoverable",
                "oracle_gate_record_ids": ["gate-fact-1"],
                "oracle_retrieval_record_status": "record_present",
                "oracle_retrieval_record_ids": ["gate-fact-1"],
                "oracle_retrieval_all_records_selected": True,
            },
            trace={"strategy": "fact_patch"},
            fact_query_context=FactQueryContext(),
        )

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("oracle_gate_retrieval_binding_fact_patch",),
        memory_resolver=resolve,
        oracle_retrieval_annotations=retrieval,
        oracle_gate_annotations=gate,
    )

    evaluated = result.tasks[0]
    assert evaluated["score"]["exact_state_match"] is True
    assert evaluated["tool_boundary"]["gold_arguments_exposed"] is True
    assert evaluated["context"]["retrieval_metadata"][
        "oracle_binding_applied"
    ] is True
    assert evaluated["context"]["retrieval_metadata"][
        "oracle_gate_candidate_status"
    ] == "gate_recoverable"


def test_vehicle_no_memory_context_excludes_gold_and_history(tmp_path: Path):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    model = ScriptedAgentModel([AgentResponse(content="No action.")])

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("no_memory",),
        scenario_index=1,
    )

    system = model.requests[0][0][0].content
    task = dataset.scenario(1).tasks[0]
    assert task.gold_memory not in system
    assert (
        dataset.scenario(1).history_path.read_text(encoding="utf-8").strip()
        not in system
    )
    assert result.status == "completed"
    assert result.metrics["profiles"]["no_memory"]["exact_state_match"] == 0


def test_vehicle_agent_run_resumes_completed_task_checkpoint(tmp_path: Path):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    output_root = tmp_path / "artifacts"
    profiles = ("no_memory", "gold_memory")

    with pytest.raises(KeyboardInterrupt):
        run_agent_evaluation(
            dataset,
            agent_model=_InterruptingAgentModel(),
            profile_names=profiles,
            scenario_index=1,
            output_root=output_root,
        )

    artifact_dir = next(output_root.iterdir())
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_lines = (
        (artifact_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert manifest["status"] == "interrupted"
    assert len(checkpoint_lines) == 1
    assert json.loads(checkpoint_lines[0])["profile"] == "no_memory"

    resumed_model = ScriptedAgentModel([AgentResponse(content="No action.")])
    result = run_agent_evaluation(
        dataset,
        agent_model=resumed_model,
        profile_names=profiles,
        scenario_index=1,
        output_root=output_root,
        resume_run_id=artifact_dir.name,
    )

    assert result.run_id == artifact_dir.name
    assert result.status == "completed"
    assert len(result.tasks) == 2
    assert len(resumed_model.requests) == 1
    assert [task["profile"] for task in result.tasks] == list(profiles)
    assert (
        len((artifact_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines())
        == 2
    )


def test_vehicle_artifacts_redact_pii_and_report_answer_loss_risk(
    tmp_path: Path,
):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    qa_path = root / "benchmark/qa_data/qa_1.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    task = qa["related_to_vehicle_preference"][0]
    task["gold_memory"] += " Home destination is 123 Main Street."
    qa_path.write_text(json.dumps(qa), encoding="utf-8")
    dataset = load_vehicle_benchmark(root, strict=False)
    model = ScriptedAgentModel([AgentResponse(content="Noted 123 Main Street.")])

    result = run_agent_evaluation(
        dataset,
        agent_model=model,
        profile_names=("gold_memory",),
        output_root=tmp_path / "artifacts",
    )

    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result.artifact_dir.iterdir()
        if path.suffix in {".json", ".jsonl", ".tsv", ".md"}
    )
    profile = result.metrics["profiles"]["gold_memory"]
    assert "123 Main Street" not in artifact_text
    assert "[REDACTED_ADDRESS]" in artifact_text
    assert profile["gold_redaction_risk_tasks"] == 1
    assert result.metrics["artifact_privacy_audit"]["detected_count"] == 0


def test_vehicle_multi_scenario_suite_persists_report_and_resumes(
    tmp_path: Path,
):
    root = _mini_benchmark(tmp_path / "VehicleMemBench")
    _write(
        root / "benchmark/qa_data/qa_2.json",
        (root / "benchmark/qa_data/qa_1.json").read_text(encoding="utf-8"),
    )
    _write(
        root / "benchmark/history/history_2.txt",
        (root / "benchmark/history/history_1.txt").read_text(encoding="utf-8"),
    )
    dataset = load_vehicle_benchmark(root, strict=False)
    output_root = tmp_path / "artifacts"
    model = ScriptedAgentModel([AgentResponse(content="No action.") for _ in range(4)])
    database = tmp_path / "runtime.db"

    with SQLiteRepository(database) as repository:
        result = run_vehicle_evaluation_suite(
            dataset,
            repository=repository,
            agent_model=model,
            profile_names=("no_memory", "gold_memory"),
            scenario_indices=(1, 2),
            snapshot_factory=lambda _: None,
            output_root=output_root,
            task_limit=1,
        )
        stored = repository.evaluation_detail(result.run_id)

    assert result.status == "completed"
    assert result.metrics["scenario_count"] == 2
    assert result.metrics["unique_task_count"] == 2
    assert len(stored["cases"]) == 4
    assert (result.artifact_dir / "results.tsv").is_file()
    assert (result.artifact_dir / "reasoning_types.tsv").is_file()
    assert result.metrics["artifact_privacy_audit"]["passed"] is True

    resumed_model = ScriptedAgentModel([])
    with SQLiteRepository(database) as repository:
        resumed = run_vehicle_evaluation_suite(
            dataset,
            repository=repository,
            agent_model=resumed_model,
            profile_names=("no_memory", "gold_memory"),
            scenario_indices=(1, 2),
            snapshot_factory=lambda _: None,
            output_root=output_root,
            task_limit=1,
            resume_run_id=result.run_id,
        )
    assert resumed.run_id == result.run_id
    assert not resumed_model.requests


def test_vehicle_provider_usage_separates_generation_and_retrieval_cost():
    generation = {
        "id": "generation",
        "role": "memory",
        "consolidation_run_id": "run-1",
        "latency_ms": 10,
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "metadata": {
            "privacy": {
                "destination": "cloud",
                "detected_count": 1,
                "redacted_count": 1,
                "category_counts": {"address": 1},
                "output_chars": 500,
                "sensitive_chars_before": 10,
                "sensitive_chars_after": 0,
            }
        },
        "error": None,
    }
    retrieval = {
        "id": "retrieval",
        "role": "embedding",
        "consolidation_run_id": None,
        "latency_ms": 5,
        "usage": {"input_tokens": 50, "output_tokens": 0},
        "metadata": {},
        "error": None,
    }

    report = summarize_provider_calls(
        (generation,),
        (generation, retrieval),
        memory_input_cost_per_million=2,
        memory_output_cost_per_million=4,
        embedding_input_cost_per_million=1,
    )

    assert report["generation"]["calls"] == 1
    assert report["generation"]["cloud_exposed_character_rate"] == 0
    assert report["generation"]["estimated_cost_usd"] == 0.00028
    assert report["retrieval"]["calls"] == 1
    assert report["retrieval"]["estimated_cost_usd"] == 0.00005
    assert report["estimated_cost_usd"] == 0.00033


def test_vehicle_provider_usage_reports_workflow_stages():
    memory_llm = {
        "id": "memory-llm",
        "role": "memory",
        "consolidation_run_id": "run-1",
        "memory_patch_run_id": None,
        "latency_ms": 10,
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "metadata": {},
        "error": None,
    }
    memory_embedding = {
        "id": "memory-embedding",
        "role": "embedding",
        "consolidation_run_id": "run-1",
        "memory_patch_run_id": None,
        "latency_ms": 3,
        "usage": {"input_tokens": 60, "output_tokens": 0},
        "metadata": {},
        "error": None,
    }
    online_embedding = {
        "id": "online-embedding",
        "role": "fact_memory_embedding",
        "consolidation_run_id": None,
        "memory_patch_run_id": None,
        "latency_ms": 5,
        "usage": {"input_tokens": 50, "output_tokens": 0},
        "metadata": {},
        "error": None,
    }

    report = summarize_provider_calls(
        (memory_llm, memory_embedding),
        (memory_llm, memory_embedding, online_embedding),
        memory_input_cost_per_million=2,
        memory_output_cost_per_million=4,
        embedding_input_cost_per_million=1,
    )
    stages = report["workflow_stages"]

    assert report["generation"]["calls"] == 2
    assert stages["idle_memory_llm"]["roles"] == {"memory": 1}
    assert stages["idle_memory_llm"]["estimated_cost_usd"] == 0.00028
    assert stages["idle_memory_embedding"]["roles"] == {"embedding": 1}
    assert stages["idle_memory_embedding"]["estimated_cost_usd"] == 0.00006
    assert stages["online_memory_retrieval"]["roles"] == {
        "fact_memory_embedding": 1
    }
    assert stages["online_memory_retrieval"]["estimated_cost_usd"] == 0.00005

    aggregate = _sum_provider_usage((report, report))
    assert aggregate["workflow_stages"]["idle_memory_llm"]["calls"] == 2
    assert aggregate["workflow_stages"]["idle_memory_embedding"][
        "input_tokens"
    ] == 120
    assert aggregate["workflow_stages"]["online_memory_retrieval"][
        "latency_ms"
    ] == 10


def test_vehicle_wiki_workflow_metrics_aggregate_traversal_and_stage_costs():
    records = [
        {
            "context": {
                "retrieval_metadata": {
                    "summary_wiki_gate_open": False,
                    "recursive_summary_tokens": 10,
                }
            },
            "memory_trace": {
                "summary_wiki": {"traversal": None, "fallback_used": False}
            },
            "tool_trace": [],
        },
        {
            "context": {
                "retrieval_metadata": {
                    "summary_wiki_gate_open": True,
                    "recursive_summary_tokens": 12,
                    "summary_wiki_termination_reason": "evidence_sufficient",
                }
            },
            "memory_trace": {
                "summary_wiki": {
                    "traversal": {
                        "steps": [
                            {"action": "search"},
                            {"action": "read"},
                        ],
                        "termination_reason": "evidence_sufficient",
                        "search_count": 1,
                        "read_count": 2,
                        "hop_count": 1,
                        "empty_search_count": 0,
                        "selected_page_ids": ["page-1", "page-2"],
                        "rendered_tokens": 20,
                        "fallback_used": False,
                    }
                }
            },
            "tool_trace": [
                {
                    "call_kind": "memory_retrieval",
                    "duration_ms": 3,
                    "is_error": False,
                },
                {
                    "call_kind": "memory_retrieval",
                    "duration_ms": 4,
                    "is_error": False,
                },
            ],
        },
    ]

    traversal = _aggregate_wiki_traversal(records)
    assert traversal is not None
    assert traversal["applicable_tasks"] == 2
    assert traversal["gate_open_tasks"] == 1
    assert traversal["traversal_tasks"] == 1
    assert traversal["search_calls"] == 1
    assert traversal["read_calls"] == 1
    assert traversal["read_pages"] == 2
    assert traversal["selected_pages"] == 2
    assert traversal["rendered_tokens"] == 20
    assert traversal["hop_count_max"] == 1
    assert traversal["sufficiency_rate"] == 1
    assert traversal["termination_reasons"] == {
        "evidence_sufficient": 1,
        "not_started_gate_closed": 1,
    }
    assert traversal["tool_latency_ms"] == 7

    online = _aggregate_online_memory_retrieval(
        records,
        wiki_traversal=traversal,
    )
    assert online["base_context_tokens"] == 22
    assert online["wiki_read_context_tokens"] == 20
    assert online["observed_context_tokens"] == 42
    assert online["wiki_tool_calls"] == 2
    assert online["observed_latency_ms"] == 7
    assert online["context_tokens_are_non_additive_to_agent_input"] is True


def test_vehicle_amem_usage_excludes_partial_history_from_formal_aggregate():
    complete = {
        "formal_aggregate_included": True,
        "note_count": 2,
        "version_count": 3,
        "embedding_count": 3,
        "link_count": 1,
        "generation": {
            "call_count": 3,
            "usage": {"input_tokens": 8, "output_tokens": 3},
        },
        "retrieval": {
            "run_count": 10,
            "selected_count": 12,
            "latency_ms": 20,
        },
        "provider": {
            "call_count": 6,
            "failed_call_count": 0,
            "roles": {
                "amem_construction": 2,
                "amem_evolution": 1,
                "amem_embedding": 2,
                "amem_retrieval_embedding": 1,
            },
            "latency_ms": 30,
            "estimated_cost_usd": 0.001,
        },
        "build_estimated_cost_usd": 0.0008,
    }
    partial = {
        **complete,
        "formal_aggregate_included": False,
        "partial_history": True,
        "note_count": 99,
    }

    result = _sum_amem_usage((complete, partial))

    assert result is not None
    assert result["scenario_count"] == 1
    assert result["excluded_partial_scenario_count"] == 1
    assert result["note_count"] == 2
    assert result["generation_call_count"] == 3
    assert result["retrieval_run_count"] == 10
    assert result["provider"]["roles"]["amem_construction"] == 2
    assert result["provider"]["estimated_cost_usd"] == 0.001
    assert result["build_estimated_cost_usd"] == 0.0008


def test_vehicle_provider_usage_separates_patch_and_tool_memory_embedding():
    patch = {
        "id": "patch",
        "role": "memory_patch",
        "consolidation_run_id": None,
        "memory_patch_run_id": "patch-run-1",
        "latency_ms": 20,
        "usage": {"input_tokens": 200, "output_tokens": 40},
        "metadata": {},
        "error": None,
    }
    retrieval = {
        "id": "tool-retrieval",
        "role": "tool_memory_embedding",
        "consolidation_run_id": None,
        "memory_patch_run_id": None,
        "latency_ms": 4,
        "usage": {"input_tokens": 30, "output_tokens": 0},
        "metadata": {},
        "error": None,
    }

    report = summarize_provider_calls(
        (patch,),
        (patch, retrieval),
        memory_input_cost_per_million=2,
        memory_output_cost_per_million=4,
        embedding_input_cost_per_million=1,
    )

    assert report["generation"]["roles"] == {"memory_patch": 1}
    assert report["generation"]["estimated_cost_usd"] == 0.00056
    assert report["retrieval"]["roles"] == {"tool_memory_embedding": 1}
    assert report["retrieval"]["estimated_cost_usd"] == 0.00003
    assert report["estimated_cost_usd"] == 0.00059


def test_vehicle_fact_quality_aggregates_each_snapshot_once():
    first = {
        "candidate_count": 3,
        "candidate_status_counts": {"applied": 2, "review": 1},
        "decision_counts": {"ADD": 2, "REVIEW": 1},
        "validation_code_counts": {"accepted": 2, "semantic_review": 1},
        "record_status_counts": {"active": 2},
        "record_count": 2,
        "active_record_count": 2,
        "versioned_record_count": 0,
        "model_call_count": 1,
        "semantic_model_call_count": 1,
        "model_usage": {"input_tokens": 100, "output_tokens": 10},
        "semantic_model_usage": {"input_tokens": 20, "output_tokens": 5},
    }
    second = {
        "candidate_count": 2,
        "candidate_status_counts": {"applied": 1, "noop": 1},
        "decision_counts": {"UPDATE": 1, "NOOP": 1},
        "validation_code_counts": {"accepted": 2},
        "record_status_counts": {"active": 1, "superseded": 1},
        "record_count": 2,
        "active_record_count": 1,
        "versioned_record_count": 1,
        "model_call_count": 1,
        "semantic_model_call_count": 0,
        "model_usage": {"input_tokens": 80, "output_tokens": 8},
        "semantic_model_usage": {},
    }
    records = [
        {
            "scenario_index": 1,
            "context": {
                "retrieval_metadata": {
                    "cache_key": "snapshot-1",
                    "fact_quality": first,
                }
            },
        },
        {
            "scenario_index": 1,
            "context": {
                "retrieval_metadata": {
                    "cache_key": "snapshot-1",
                    "fact_quality": first,
                }
            },
        },
        {
            "scenario_index": 2,
            "context": {
                "retrieval_metadata": {
                    "cache_key": "snapshot-2",
                    "fact_quality": second,
                }
            },
        },
    ]

    result = _aggregate_fact_quality(records)

    assert result is not None
    assert result["snapshot_count"] == 2
    assert result["candidate_count"] == 5
    assert result["candidate_status_counts"] == {
        "applied": 3,
        "noop": 1,
        "review": 1,
    }
    assert result["model_usage"]["input_tokens"] == 180
    assert result["candidate_acceptance_rate"] == 0.6
    assert result["candidate_resolution_rate"] == 0.8


def test_official_vehicle_benchmark_when_configured():
    configured = os.getenv("VEHICLEMEMBENCH_ROOT")
    if not configured:
        pytest.skip("Set VEHICLEMEMBENCH_ROOT for the official integration smoke")
    dataset = load_vehicle_benchmark(
        configured,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
    )

    result = run_offline_smoke(dataset, scenario_index=1)

    assert result.passed is True
    assert result.task_count == 10
    assert dataset.manifest.scenario_count == 50
    assert dataset.manifest.task_count == 500
    assert dataset.manifest.tool_count == 111
