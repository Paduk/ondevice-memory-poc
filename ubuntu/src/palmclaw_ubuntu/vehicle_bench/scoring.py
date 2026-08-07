from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from palmclaw_ubuntu.vehicle_bench.dataset import GoldToolCall

_IMPORT_LOCK = threading.Lock()


@dataclass(frozen=True)
class VehicleTaskScore:
    exact_state_match: bool
    state_score: dict[str, Any]
    tool_score: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "exact_state_match": self.exact_state_match,
            "state_score": self.state_score,
            "tool_score": self.tool_score,
        }


class VehicleWorldRuntime:
    """Loads the trusted upstream simulator and scorer under an isolated name."""

    def __init__(self, benchmark_root: Path):
        self.root = benchmark_root.resolve(strict=True)
        namespace = f"_palmclaw_vmb_{_namespace_token(self.root)}"
        with _IMPORT_LOCK:
            self._environment = _load_environment_package(self.root, namespace)
            self._eval_utils = _load_eval_utils(
                self.root,
                namespace,
                self._environment,
            )
        vehicleworld = importlib.import_module(f"{namespace}.environment.vehicleworld")
        self._world_type = vehicleworld.VehicleWorld
        utils = importlib.import_module(f"{namespace}.environment.utils")
        self.modules: dict[str, str] = dict(utils.modules_dict)
        self._calculate_turn_result: Callable[..., dict[str, Any]] = (
            self._eval_utils.calculate_turn_result
        )
        self._score_tool_calls: Callable[..., dict[str, Any]] = (
            self._eval_utils.score_tool_calls
        )

    def create_world(self) -> Any:
        return self._world_type()

    def state(self, world: Any) -> dict[str, Any]:
        return world.to_dict()

    def tool_map(self, world: Any) -> dict[str, Callable[..., Any]]:
        result: dict[str, Callable[..., Any]] = {}
        for module_name in self.modules:
            module = getattr(world, module_name, None)
            if module is None:
                continue
            for attribute in dir(module):
                if not attribute.startswith("carcontrol_"):
                    continue
                function = getattr(module, attribute, None)
                if callable(function):
                    if attribute in result:
                        raise RuntimeError(
                            f"Duplicate VehicleWorld tool method: {attribute}"
                        )
                    result[attribute] = function
        return result

    def module_for_tools(self, world: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        for module_name in self.modules:
            module = getattr(world, module_name, None)
            if module is None:
                continue
            for attribute in dir(module):
                if attribute.startswith("carcontrol_") and callable(
                    getattr(module, attribute, None)
                ):
                    result[attribute] = module_name
        return result

    def execute(
        self,
        world: Any,
        name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        function = self.tool_map(world).get(name)
        if function is None:
            raise KeyError(f"VehicleWorld tool not found: {name}")
        return function(**dict(arguments))

    def score(
        self,
        *,
        initial_state: dict[str, Any],
        reference_state: dict[str, Any],
        predicted_state: dict[str, Any],
        predicted_calls: Sequence[GoldToolCall],
        reference_calls: Sequence[GoldToolCall],
    ) -> VehicleTaskScore:
        state_score = self._calculate_turn_result(
            initial_state,
            reference_state,
            initial_state,
            predicted_state,
        )
        tool_score = self._score_tool_calls(
            [call.as_official() for call in predicted_calls],
            [call.as_official() for call in reference_calls],
        )
        exact_match = (
            len(state_score.get("differences", [])) == 0
            and state_score.get("FP", 0) == 0
            and state_score.get("negative_FP", 0) == 0
        )
        return VehicleTaskScore(
            exact_state_match=exact_match,
            state_score=state_score,
            tool_score=tool_score,
        )


def _namespace_token(root: Path) -> str:
    import hashlib

    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]


def _load_environment_package(root: Path, namespace: str) -> ModuleType:
    parent = sys.modules.get(namespace)
    if parent is None:
        parent = ModuleType(namespace)
        parent.__path__ = [str(root)]  # type: ignore[attr-defined]
        sys.modules[namespace] = parent

    name = f"{namespace}.environment"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    package_root = root / "environment"
    spec = importlib.util.spec_from_file_location(
        name,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load VehicleMemBench environment package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_eval_utils(
    root: Path,
    namespace: str,
    environment: ModuleType,
) -> ModuleType:
    name = f"{namespace}.eval_utils"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    environment_utils = importlib.import_module(f"{namespace}.environment.utils")
    spec = importlib.util.spec_from_file_location(
        name,
        root / "evaluation" / "eval_utils.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load VehicleMemBench official scorer")
    module = importlib.util.module_from_spec(spec)

    previous_environment = sys.modules.get("environment")
    previous_utils = sys.modules.get("environment.utils")
    root_text = str(root)
    root_was_on_path = root_text in sys.path
    sys.modules["environment"] = environment
    sys.modules["environment.utils"] = environment_utils
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        if previous_environment is None:
            sys.modules.pop("environment", None)
        else:
            sys.modules["environment"] = previous_environment
        if previous_utils is None:
            sys.modules.pop("environment.utils", None)
        else:
            sys.modules["environment.utils"] = previous_utils
        if not root_was_on_path:
            while root_text in sys.path:
                sys.path.remove(root_text)
    return module
