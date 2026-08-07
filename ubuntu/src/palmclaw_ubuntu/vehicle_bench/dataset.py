from __future__ import annotations

import ast
import hashlib
import io
import json
import subprocess
import token
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

OFFICIAL_UPSTREAM_COMMIT = "5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b"
REASONING_TYPES = frozenset(
    {
        "conditional_constraint",
        "coreference_resolution",
        "error_correction",
        "preference_conflict",
        "state_shift",
    }
)
_QA_KEY = "related_to_vehicle_preference"


class VehicleBenchValidationError(ValueError):
    """Raised when a VehicleMemBench checkout does not match its contract."""


@dataclass(frozen=True)
class GoldToolCall:
    name: str
    arguments: dict[str, Any]
    source: str

    def as_official(self) -> dict[str, Any]:
        return {"name": self.name, "args": dict(self.arguments)}


@dataclass(frozen=True)
class VehicleTask:
    id: str
    scenario_index: int
    event_index: int
    query: str
    gold_memory: str
    reasoning_type: str
    gold_calls: tuple[GoldToolCall, ...]


@dataclass(frozen=True)
class VehicleScenario:
    index: int
    history_path: Path
    qa_path: Path
    tasks: tuple[VehicleTask, ...]


@dataclass(frozen=True)
class BenchmarkManifest:
    upstream_commit: str | None
    dataset_sha256: str
    tool_schema_sha256: str
    scenario_count: int
    task_count: int
    tool_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "upstream_commit": self.upstream_commit,
            "dataset_sha256": self.dataset_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "scenario_count": self.scenario_count,
            "task_count": self.task_count,
            "tool_count": self.tool_count,
        }


@dataclass(frozen=True)
class VehicleBenchmarkDataset:
    root: Path
    scenarios: tuple[VehicleScenario, ...]
    tool_schemas: tuple[dict[str, Any], ...]
    manifest: BenchmarkManifest

    def scenario(self, index: int) -> VehicleScenario:
        for scenario in self.scenarios:
            if scenario.index == index:
                return scenario
        raise KeyError(f"VehicleMemBench scenario not found: {index}")


def load_vehicle_benchmark(
    root: Path | str,
    *,
    expected_commit: str | None = None,
    strict: bool = True,
) -> VehicleBenchmarkDataset:
    benchmark_root = _resolve_root(root)
    history_root = _contained_directory(benchmark_root, "benchmark/history")
    qa_root = _contained_directory(benchmark_root, "benchmark/qa_data")
    schema_path = _contained_file(
        benchmark_root,
        "evaluation/functions_schema.json",
    )
    _contained_file(benchmark_root, "environment/vehicleworld.py")
    _contained_file(benchmark_root, "evaluation/eval_utils.py")

    schemas = _load_tool_schemas(schema_path)
    validators = {}
    for item in schemas:
        parameters = dict(item["parameters"])
        parameters.setdefault("additionalProperties", False)
        validators[item["name"]] = Draft202012Validator(parameters)
    history_paths = _numbered_files(history_root, "history", ".txt")
    qa_paths = _numbered_files(qa_root, "qa", ".json")
    if set(history_paths) != set(qa_paths):
        missing_history = sorted(set(qa_paths) - set(history_paths))
        missing_qa = sorted(set(history_paths) - set(qa_paths))
        raise VehicleBenchValidationError(
            "History/QA scenario mismatch: "
            f"missing_history={missing_history}, missing_qa={missing_qa}"
        )

    scenarios = tuple(
        _load_scenario(
            index,
            history_paths[index],
            qa_paths[index],
            validators,
            strict=strict,
        )
        for index in sorted(history_paths)
    )
    task_count = sum(len(scenario.tasks) for scenario in scenarios)
    if strict:
        if len(scenarios) != 50:
            raise VehicleBenchValidationError(
                f"Expected 50 scenarios, found {len(scenarios)}"
            )
        if task_count != 500:
            raise VehicleBenchValidationError(f"Expected 500 tasks, found {task_count}")
        if len(schemas) != 111:
            raise VehicleBenchValidationError(
                f"Expected 111 vehicle tools, found {len(schemas)}"
            )

    commit = _git_commit(benchmark_root)
    if expected_commit is not None and commit != expected_commit:
        found = commit or "not a Git checkout"
        raise VehicleBenchValidationError(
            "VehicleMemBench commit mismatch: "
            f"expected {expected_commit}, found {found}"
        )

    dataset_files = [
        *(history_paths[index] for index in sorted(history_paths)),
        *(qa_paths[index] for index in sorted(qa_paths)),
    ]
    manifest = BenchmarkManifest(
        upstream_commit=commit,
        dataset_sha256=_hash_files(benchmark_root, dataset_files),
        tool_schema_sha256=_sha256_file(schema_path),
        scenario_count=len(scenarios),
        task_count=task_count,
        tool_count=len(schemas),
    )
    return VehicleBenchmarkDataset(
        root=benchmark_root,
        scenarios=scenarios,
        tool_schemas=schemas,
        manifest=manifest,
    )


def _resolve_root(root: Path | str) -> Path:
    path = Path(root).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise VehicleBenchValidationError(
            f"VehicleMemBench root does not exist: {path}"
        ) from exc
    if not resolved.is_dir():
        raise VehicleBenchValidationError(
            f"VehicleMemBench root is not a directory: {resolved}"
        )
    return resolved


def _contained_directory(root: Path, relative: str) -> Path:
    path = _contained_path(root, relative)
    if not path.is_dir():
        raise VehicleBenchValidationError(f"Required directory is missing: {relative}")
    return path


def _contained_file(root: Path, relative: str) -> Path:
    path = _contained_path(root, relative)
    if not path.is_file():
        raise VehicleBenchValidationError(f"Required file is missing: {relative}")
    return path


def _contained_path(root: Path, relative: str) -> Path:
    unresolved = root / relative
    try:
        path = unresolved.resolve(strict=True)
    except OSError as exc:
        raise VehicleBenchValidationError(
            f"Required VehicleMemBench path is missing: {relative}"
        ) from exc
    if not path.is_relative_to(root):
        raise VehicleBenchValidationError(
            f"VehicleMemBench path escapes checkout root: {relative}"
        )
    return path


def _numbered_files(
    directory: Path,
    prefix: str,
    suffix: str,
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.glob(f"{prefix}_*{suffix}"):
        stem_prefix = f"{prefix}_"
        number = path.name[len(stem_prefix) : -len(suffix)]
        if not number.isdigit():
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(directory):
            raise VehicleBenchValidationError(
                f"Benchmark file escapes its directory: {path.name}"
            )
        index = int(number)
        if index in result:
            raise VehicleBenchValidationError(
                f"Duplicate benchmark scenario index: {index}"
            )
        result[index] = resolved
    if not result:
        raise VehicleBenchValidationError(
            f"No {prefix}_N{suffix} files found in {directory}"
        )
    return result


def _load_tool_schemas(path: Path) -> tuple[dict[str, Any], ...]:
    raw = _read_json(path)
    if not isinstance(raw, list) or not raw:
        raise VehicleBenchValidationError(
            "functions_schema.json must contain a non-empty list"
        )
    schemas: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise VehicleBenchValidationError(
                f"Tool schema at index {index} is not an object"
            )
        name = item.get("name")
        description = item.get("description")
        parameters = item.get("parameters")
        if not isinstance(name, str) or not name.startswith("carcontrol_"):
            raise VehicleBenchValidationError(
                f"Invalid vehicle tool name at schema index {index}: {name!r}"
            )
        if name in names:
            raise VehicleBenchValidationError(f"Duplicate vehicle tool schema: {name}")
        if not isinstance(description, str) or not description.strip():
            raise VehicleBenchValidationError(
                f"Vehicle tool description is missing: {name}"
            )
        if not isinstance(parameters, dict):
            raise VehicleBenchValidationError(
                f"Vehicle tool parameters must be an object: {name}"
            )
        try:
            Draft202012Validator.check_schema(parameters)
        except Exception as exc:
            raise VehicleBenchValidationError(
                f"Invalid JSON Schema for vehicle tool {name}: {exc}"
            ) from exc
        names.add(name)
        schemas.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
    return tuple(schemas)


def _load_scenario(
    index: int,
    history_path: Path,
    qa_path: Path,
    validators: dict[str, Draft202012Validator],
    *,
    strict: bool,
) -> VehicleScenario:
    if not history_path.read_text(encoding="utf-8").strip():
        raise VehicleBenchValidationError(f"Empty history file: {history_path.name}")
    raw = _read_json(qa_path)
    if not isinstance(raw, dict):
        raise VehicleBenchValidationError(f"QA root must be an object: {qa_path.name}")
    events = raw.get(_QA_KEY)
    if not isinstance(events, list) or not events:
        raise VehicleBenchValidationError(
            f"QA file must contain non-empty {_QA_KEY!r}: {qa_path.name}"
        )
    if strict and len(events) != 10:
        raise VehicleBenchValidationError(
            f"Expected 10 tasks in {qa_path.name}, found {len(events)}"
        )

    tasks: list[VehicleTask] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise VehicleBenchValidationError(
                f"QA event {event_index} is not an object: {qa_path.name}"
            )
        query = event.get("query")
        gold_memory = event.get("gold_memory")
        reasoning_type = event.get("reasoning_type")
        answers = event.get("new_answer")
        if not isinstance(query, str) or not query.strip():
            raise VehicleBenchValidationError(
                f"Missing query at {qa_path.name}:{event_index}"
            )
        if not isinstance(gold_memory, str):
            raise VehicleBenchValidationError(
                f"Missing gold_memory at {qa_path.name}:{event_index}"
            )
        if reasoning_type not in REASONING_TYPES:
            raise VehicleBenchValidationError(
                f"Unknown reasoning_type at {qa_path.name}:{event_index}: "
                f"{reasoning_type!r}"
            )
        if not isinstance(answers, list) or not answers:
            raise VehicleBenchValidationError(
                f"Missing new_answer at {qa_path.name}:{event_index}"
            )

        calls: list[GoldToolCall] = []
        for answer in answers:
            if not isinstance(answer, str):
                raise VehicleBenchValidationError(
                    f"Gold tool call is not text at {qa_path.name}:{event_index}"
                )
            call = _parse_gold_call(answer)
            validator = validators.get(call.name)
            if validator is None:
                raise VehicleBenchValidationError(
                    f"Gold call references unknown tool {call.name}: {qa_path.name}"
                )
            try:
                validator.validate(call.arguments)
            except ValidationError as exc:
                raise VehicleBenchValidationError(
                    f"Invalid gold arguments for {call.name} at "
                    f"{qa_path.name}:{event_index}: {exc.message}"
                ) from exc
            calls.append(call)

        tasks.append(
            VehicleTask(
                id=f"vehicle-{index:02d}-{event_index:02d}",
                scenario_index=index,
                event_index=event_index,
                query=query,
                gold_memory=gold_memory,
                reasoning_type=reasoning_type,
                gold_calls=tuple(calls),
            )
        )
    return VehicleScenario(
        index=index,
        history_path=history_path,
        qa_path=qa_path,
        tasks=tuple(tasks),
    )


def _parse_gold_call(source: str) -> GoldToolCall:
    normalized = _normalize_json_literals(source)
    try:
        expression = ast.parse(normalized, mode="eval").body
    except SyntaxError as exc:
        raise VehicleBenchValidationError(
            f"Invalid gold tool call syntax: {source!r}"
        ) from exc
    if not isinstance(expression, ast.Call) or not isinstance(
        expression.func, ast.Name
    ):
        raise VehicleBenchValidationError(
            f"Gold answer must be a direct function call: {source!r}"
        )
    if expression.args:
        raise VehicleBenchValidationError(
            f"Gold tool call must use keyword arguments only: {source!r}"
        )
    arguments: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None or keyword.arg in arguments:
            raise VehicleBenchValidationError(
                f"Invalid gold keyword argument: {source!r}"
            )
        try:
            value = ast.literal_eval(keyword.value)
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise VehicleBenchValidationError(
                f"Gold argument must be a JSON literal: {source!r}"
            ) from exc
        arguments[keyword.arg] = value
    return GoldToolCall(
        name=expression.func.id,
        arguments=arguments,
        source=source,
    )


def _normalize_json_literals(source: str) -> str:
    replacements = {"true": "True", "false": "False", "null": "None"}
    try:
        tokens = []
        for item in tokenize.generate_tokens(io.StringIO(source).readline):
            if item.type == token.NAME and item.string in replacements:
                item = tokenize.TokenInfo(
                    item.type,
                    replacements[item.string],
                    item.start,
                    item.end,
                    item.line,
                )
            tokens.append(item)
        return tokenize.untokenize(tokens)
    except (IndentationError, tokenize.TokenError) as exc:
        raise VehicleBenchValidationError(
            f"Invalid gold tool call syntax: {source!r}"
        ) from exc


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VehicleBenchValidationError(
            f"Cannot read JSON file {path}: {exc}"
        ) from exc


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip().lower()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        return None
    return commit


def _hash_files(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
