from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from palmclaw_ubuntu.models import ToolDefinition, ToolResult
from palmclaw_ubuntu.tools import (
    ToolExecutionContext,
    ToolRegistry,
)
from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime

LIST_MODULE_TOOLS = "list_module_tools"


def vehicle_tool_definitions(
    schemas: Sequence[Mapping[str, Any]],
    *,
    timeout_seconds: float = 5,
) -> tuple[ToolDefinition, ...]:
    definitions = []
    for schema in schemas:
        parameters = dict(schema["parameters"])
        parameters.setdefault("additionalProperties", False)
        definitions.append(
            ToolDefinition(
                name=str(schema["name"]),
                description=str(schema["description"]),
                parameters=parameters,
                timeout_seconds=timeout_seconds,
                side_effect="simulator_state",
                retry_safety="unsafe",
                strict=False,
            )
        )
    return tuple(definitions)


class VehicleSimulatorTool:
    def __init__(
        self,
        runtime: VehicleWorldRuntime,
        world: Any,
        definition: ToolDefinition,
        module_name: str,
    ):
        self.runtime = runtime
        self.world = world
        self.definition = definition
        self.module_name = module_name

    def run(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        result = self.runtime.execute(
            self.world,
            self.definition.name,
            arguments,
        )
        is_error = isinstance(result, Mapping) and result.get("success") is False
        return ToolResult(
            tool_call_id="",
            content=json.dumps(result, ensure_ascii=False),
            is_error=is_error,
            metadata={
                "simulator": "VehicleWorld",
                "module": self.module_name,
            },
        )


class VehicleModuleDiscoveryTool:
    def __init__(
        self,
        runtime: VehicleWorldRuntime,
        world: Any,
        definitions: Sequence[ToolDefinition],
        registry: ToolRegistry,
    ):
        self.runtime = runtime
        self.world = world
        self.registry = registry
        self.loaded_modules: set[str] = set()
        self.preloaded_tools: set[str] = set()
        self._modules_by_tool = runtime.module_for_tools(world)
        self._definitions = {definition.name: definition for definition in definitions}
        module_names = list(runtime.modules)
        self.definition = ToolDefinition(
            name=LIST_MODULE_TOOLS,
            description=(
                "List and enable the available tools for one vehicle module. "
                "Call this before using a vehicle function from that module."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "module_name": {
                        "type": "string",
                        "enum": module_names,
                        "description": "Vehicle module to inspect",
                    }
                },
                "required": ["module_name"],
                "additionalProperties": False,
            },
            timeout_seconds=2,
            side_effect="none",
            retry_safety="safe",
        )

    def run(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        module_name = str(arguments["module_name"])
        if module_name in self.loaded_modules:
            names = self._module_tool_names(module_name)
            return self._result(module_name, names, already_loaded=True)

        names = self._module_tool_names(module_name)
        if not names:
            return ToolResult(
                tool_call_id="",
                content=json.dumps(
                    {
                        "success": False,
                        "error": f"Vehicle module not found: {module_name}",
                    },
                    ensure_ascii=False,
                ),
                is_error=True,
                metadata={
                    "error_code": "unknown_vehicle_module",
                    "module": module_name,
                },
            )
        for name in names:
            self._register_tool(name)
        self.loaded_modules.add(module_name)
        return self._result(module_name, names, already_loaded=False)

    def preload(self, tool_names: Sequence[str]) -> tuple[str, ...]:
        """Expose router-selected schemas while retaining discovery fallback."""

        selected = tuple(dict.fromkeys(str(name) for name in tool_names))
        unknown = sorted(set(selected) - set(self._definitions))
        if unknown:
            raise ValueError(f"Unknown routed vehicle Tools: {unknown}")
        for name in selected:
            self._register_tool(name)
            self.preloaded_tools.add(name)
        return selected

    @property
    def available_modules(self) -> set[str]:
        return {
            *self.loaded_modules,
            *(self._modules_by_tool[name] for name in self.preloaded_tools),
        }

    @property
    def preloaded_modules(self) -> set[str]:
        return {
            self._modules_by_tool[name]
            for name in self.preloaded_tools
        }

    def _module_tool_names(self, module_name: str) -> list[str]:
        return sorted(
            name
            for name, owner in self._modules_by_tool.items()
            if owner == module_name and name in self._definitions
        )

    def _register_tool(self, name: str) -> None:
        if self.registry.definition(name) is not None:
            return
        definition = self._definitions[name]
        self.registry.register(
            VehicleSimulatorTool(
                self.runtime,
                self.world,
                definition,
                self._modules_by_tool[name],
            )
        )

    @staticmethod
    def _result(
        module_name: str,
        names: Sequence[str],
        *,
        already_loaded: bool,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id="",
            content=json.dumps(
                {
                    "success": True,
                    "module": module_name,
                    "already_loaded": already_loaded,
                    "tool_count": len(names),
                    "tools": list(names),
                },
                ensure_ascii=False,
            ),
            metadata={
                "module": module_name,
                "loaded_tool_count": len(names),
                "already_loaded": already_loaded,
            },
        )


def build_dynamic_vehicle_tool_registry(
    runtime: VehicleWorldRuntime,
    world: Any,
    definitions: Sequence[ToolDefinition],
    *,
    max_result_chars: int = 20_000,
    preload_tool_names: Sequence[str] = (),
) -> tuple[ToolRegistry, VehicleModuleDiscoveryTool]:
    _validate_definition_runtime_match(runtime, world, definitions)
    registry = ToolRegistry(max_result_chars=max_result_chars)
    discovery = VehicleModuleDiscoveryTool(
        runtime,
        world,
        definitions,
        registry,
    )
    registry.register(discovery)
    discovery.preload(preload_tool_names)
    return registry, discovery


def build_oracle_vehicle_tool_registry(
    runtime: VehicleWorldRuntime,
    world: Any,
    definitions: Sequence[ToolDefinition],
    allowed_tool_names: Sequence[str],
    *,
    max_result_chars: int = 20_000,
) -> tuple[ToolRegistry, tuple[str, ...]]:
    """Build a diagnostic registry restricted to reference Tool schemas.

    Only Tool names are used to select schemas. Reference arguments and call
    ordering never enter the registry or Agent prompt.
    """
    _validate_definition_runtime_match(runtime, world, definitions)
    selected_names = tuple(dict.fromkeys(str(name) for name in allowed_tool_names))
    if not selected_names:
        raise ValueError("Oracle Tool boundary requires at least one Tool name")
    by_name = {definition.name: definition for definition in definitions}
    unknown = sorted(set(selected_names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown Oracle vehicle Tools: {unknown}")
    modules = runtime.module_for_tools(world)
    tools = [
        VehicleSimulatorTool(
            runtime,
            world,
            by_name[name],
            modules[name],
        )
        for name in selected_names
    ]
    loaded_modules = tuple(sorted({modules[name] for name in selected_names}))
    return (
        ToolRegistry(tools, max_result_chars=max_result_chars),
        loaded_modules,
    )


def build_vehicle_tool_registry(
    runtime: VehicleWorldRuntime,
    world: Any,
    definitions: Sequence[ToolDefinition],
    *,
    max_result_chars: int = 20_000,
) -> ToolRegistry:
    _validate_definition_runtime_match(runtime, world, definitions)
    modules = runtime.module_for_tools(world)
    tools = [
        VehicleSimulatorTool(
            runtime,
            world,
            definition,
            modules[definition.name],
        )
        for definition in definitions
    ]
    return ToolRegistry(tools, max_result_chars=max_result_chars)


def _validate_definition_runtime_match(
    runtime: VehicleWorldRuntime,
    world: Any,
    definitions: Sequence[ToolDefinition],
) -> None:
    available = runtime.tool_map(world)
    definition_names = {definition.name for definition in definitions}
    runtime_names = set(available)
    if definition_names != runtime_names:
        missing_methods = sorted(definition_names - runtime_names)
        missing_schemas = sorted(runtime_names - definition_names)
        raise RuntimeError(
            "Vehicle tool schema/runtime mismatch: "
            f"missing_methods={missing_methods}, missing_schemas={missing_schemas}"
        )
