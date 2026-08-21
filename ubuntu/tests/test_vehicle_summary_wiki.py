from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from palmclaw_ubuntu.models import (
    AgentResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from palmclaw_ubuntu.providers import ScriptedAgentModel
from palmclaw_ubuntu.tools import ToolExecutionContext, ToolRegistry
from palmclaw_ubuntu.vehicle_bench.agent import (
    VehicleAgentLoop,
    build_vehicle_system_prompt,
)
from palmclaw_ubuntu.vehicle_summary_wiki import (
    MEMORY_WIKI_READ,
    MEMORY_WIKI_SEARCH,
    build_summary_wiki_runtime,
    build_summary_wiki_versions,
    register_summary_wiki_tools,
)


def _memory_rows() -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": "summary-v1",
            "version": 1,
            "created_at": "2026-01-01T00:00:00Z",
            "content": (
                "**Gary**\n- hud_brightness: 8\n- night_volume: 20"
            ),
        },
        {
            "id": "summary-v2",
            "version": 2,
            "created_at": "2026-01-02T00:00:00Z",
            "content": "**Gary**\n- hud_brightness: 8",
        },
    )


def _daily_steps() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "index": index,
            "date": f"2026-01-0{index + 1}",
            "status": "updated",
            "summary_sha256": hashlib.sha256(
                row["content"].encode("utf-8")
            ).hexdigest(),
            "consolidation_run_id": f"run-{index + 1}",
        }
        for index, row in enumerate(_memory_rows())
    )


def _runtime(query: str):
    return build_summary_wiki_runtime(
        namespace="test-summary",
        query=query,
        final_summary=str(_memory_rows()[-1]["content"]),
        memories=_memory_rows(),
        daily_steps=_daily_steps(),
        base_cache_key="base-cache",
    )


def test_summary_wiki_versions_use_summary_provenance_not_raw_history() -> None:
    versions = build_summary_wiki_versions(_memory_rows(), _daily_steps())

    assert [version.version_id for version in versions] == [
        "summary-v1",
        "summary-v2",
    ]
    assert versions[0].source_ids == ("consolidation:run-1",)
    assert versions[0].created_at.startswith("2026-01-01")


def test_summary_wiki_gate_preserves_covered_query_and_opens_for_gap() -> None:
    closed = _runtime("Set Gary's HUD brightness preference.")
    missing = _runtime("Set Gary's night volume preference.")
    conditional = _runtime("Set Gary's HUD brightness when it rains.")

    assert closed.gate.opened is False
    assert missing.gate.opened is True
    assert "night" in missing.gate.missing_terms
    assert "volume" in missing.gate.missing_terms
    assert conditional.gate.opened is True
    assert "conditional" in conditional.gate.reasons
    assert closed.cache_signature == _runtime(
        "Set Gary's HUD brightness preference."
    ).cache_signature


def test_summary_wiki_tools_search_and_read_bounded_pages() -> None:
    runtime = _runtime("Set Gary's night volume preference.")
    registry = ToolRegistry()
    register_summary_wiki_tools(registry, runtime.session)
    read_definition = next(
        definition
        for definition in registry.definitions()
        if definition.name == MEMORY_WIKI_READ
    )
    assert "uniqueItems" not in read_definition.parameters["properties"][
        "page_ids"
    ]

    search, _ = registry.execute(
        ToolCall(
            id="search",
            name=MEMORY_WIKI_SEARCH,
            arguments={"query": "Gary night volume"},
        ),
        ToolExecutionContext(session_id="task"),
    )
    page_id = runtime.session.trace.steps[0].result_page_ids[0]
    read, _ = registry.execute(
        ToolCall(
            id="read",
            name=MEMORY_WIKI_READ,
            arguments={"page_ids": [page_id]},
        ),
        ToolExecutionContext(session_id="task"),
    )

    assert search.is_error is False
    assert read.is_error is False
    assert runtime.session.trace.search_count == 1
    assert runtime.session.trace.read_count == 1
    assert runtime.session.trace.rendered_tokens <= 1_200


class _VehicleTool:
    def __init__(self):
        self.definition = ToolDefinition(
            name="vehicle_set_value",
            description="Set a test vehicle value.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            timeout_seconds=2,
            side_effect="simulator_state",
            retry_safety="unsafe",
        )

    def run(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        return ToolResult(tool_call_id="", content=str(arguments["value"]))


def test_agent_loop_excludes_wiki_calls_from_predicted_vehicle_calls() -> None:
    runtime = _runtime("Set Gary's night volume preference.")
    page_id = next(
        page.page_id
        for page in runtime.pages
        if "night_volume" in page.content
    )
    registry = ToolRegistry()
    register_summary_wiki_tools(registry, runtime.session)
    registry.register(_VehicleTool())
    model = ScriptedAgentModel(
        (
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="search",
                        name=MEMORY_WIKI_SEARCH,
                        arguments={"query": "night volume"},
                    ),
                ),
            ),
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="read",
                        name=MEMORY_WIKI_READ,
                        arguments={"page_ids": [page_id]},
                    ),
                ),
            ),
            AgentResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="vehicle",
                        name="vehicle_set_value",
                        arguments={"value": 20},
                    ),
                ),
            ),
            AgentResponse(content="Done."),
        )
    )
    result = VehicleAgentLoop(
        agent_model=model,
        tool_registry=registry,
        max_tool_rounds=5,
        non_vehicle_tool_names=(MEMORY_WIKI_SEARCH, MEMORY_WIKI_READ),
    ).run(
        task_id="wiki-task",
        query="Set the volume.",
        system_prompt=build_vehicle_system_prompt(
            {"Test": "Test module"},
            retrieved_memory=str(_memory_rows()[-1]["content"]),
            progressive_memory_retrieval=True,
        ),
    )

    assert [call.name for call in result.predicted_calls] == [
        "vehicle_set_value"
    ]
    assert [step.get("call_kind") for step in result.tool_trace] == [
        "memory_retrieval",
        "memory_retrieval",
        None,
    ]
