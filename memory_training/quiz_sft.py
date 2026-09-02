"""Build compact, validated VehicleMemBench V2 Tool-Calling SFT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from .config import DEFAULT_DATA_ROOT, DEFAULT_WORKSPACE_ROOT, split_for_scenario
from .dataset import IndexedMemoryDataset, default_catalog_path, ensure_catalog
from .ubuntu_bridge import enable_ubuntu_runtime

SCHEMA_VERSION = "vehiclemembench-v2-quiz-tool-sft-v1"
MANIFEST_SCHEMA_VERSION = "vehiclemembench-v2-quiz-tool-sft-manifest-v1"
TOOL_SCHEMA_VERSION = "vehiclemembench-v1-official-tools-v1"
QUIZ_FILES = ("turn_quiz.jsonl", "final_quiz.jsonl")
REASONING_TYPES = {
    "conditional_constraint",
    "coreference_resolution",
    "error_correction",
    "preference_conflict",
    "state_shift",
}

SYSTEM_PROMPT = (
    "You are an in-vehicle AI agent. Use the supplied memory only when the current "
    "request needs personalization, and let the current request override memory. "
    "Call the minimum provided vehicle functions needed to fulfill every requested "
    "setting. Do not emit a natural-language answer when tool calls are required."
)


class IndexedQuizSFTDataset(Sequence[dict[str, Any]]):
    """Worker-safe lazy access to one SFT partition of ``quiz_sft.jsonl``."""

    def __init__(self, path: Path, *, split: str) -> None:
        if split not in {"excluded", "train", "validation", "test"}:
            raise ValueError(f"Unknown Quiz SFT split: {split}")
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.split = split
        self._entries: list[tuple[int, int, int]] = []
        with self.path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                line_number += 1
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.path.name}:{line_number}"
                    ) from exc
                if row.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError(
                        f"Unsupported Quiz SFT schema in line {line_number}"
                    )
                if row.get("sft_split") == split:
                    scenario = row.get("scenario_index")
                    if not isinstance(scenario, int):
                        raise TypeError(f"Invalid scenario in line {line_number}")
                    self._entries.append((offset, len(raw), scenario))
        self._handle: BinaryIO | None = None
        self._handle_pid: int | None = None

    def __len__(self) -> int:
        return len(self._entries)

    def _file(self) -> BinaryIO:
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            self.close()
            self._handle = self.path.open("rb")
            self._handle_pid = pid
        return self._handle

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        offset, length, scenario = self._entries[index]
        handle = self._file()
        handle.seek(offset)
        row = json.loads(handle.read(length))
        if row.get("scenario_index") != scenario or row.get("sft_split") != self.split:
            raise ValueError(f"Quiz SFT index/source mismatch at position {index}")
        return row

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._handle_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()


class VehicleToolSchemaStore:
    """Resolve compact per-row Tool references and verify their shared artifact."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != TOOL_SCHEMA_VERSION:
            raise ValueError(f"Unsupported shared Tool schema: {self.path}")
        tools = payload.get("tools")
        if not isinstance(tools, list) or not tools:
            raise ValueError(f"Shared Tool schema is empty: {self.path}")
        self.sha256 = _sha256(self.path)
        self._by_name = {
            str(tool["function"]["name"]): tool
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }
        if len(self._by_name) != len(tools):
            raise ValueError(f"Duplicate or malformed shared Tool schema: {self.path}")

    def tools_for(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        reference = row.get("tool_schema_ref")
        if not isinstance(reference, dict):
            raise TypeError("Quiz SFT row has no tool_schema_ref")
        if reference.get("sha256") != self.sha256:
            raise ValueError("Quiz SFT Tool schema hash does not match shared artifact")
        names = reference.get("tool_names")
        if not isinstance(names, list) or not names:
            raise ValueError("Quiz SFT row has no candidate Tool names")
        try:
            return [self._by_name[str(name)] for name in names]
        except KeyError as exc:
            raise ValueError(
                f"Quiz SFT references unknown Tool: {exc.args[0]}"
            ) from exc


def sft_split_for_scenario(scenario_index: int) -> str:
    """Apply the experiment split, including the explicitly excluded pilot set."""
    if 101 <= scenario_index <= 110:
        return "train"
    if scenario_index == 111:
        return "validation"
    if 112 <= scenario_index <= 120:
        return "test"
    # Training-only original V1 augmentation (V1 S1-S50 -> 201-250).
    if 201 <= scenario_index <= 250:
        return "train"
    # Training-only V1-style structural clones.
    if 301 <= scenario_index <= 320:
        return "train"
    if not 1 <= scenario_index <= 100:
        raise ValueError(
            "scenario_index must be S1-S100, encoded T1-T20, "
            "training-only encoded V1 S1-S50, or V1-style S301-S320: "
            f"{scenario_index}"
        )
    if scenario_index <= 14:
        return "excluded"
    if scenario_index <= 80:
        return "train"
    if scenario_index <= 85:
        return "validation"
    return "test"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_tool_schemas(dataset_root: Path) -> tuple[list[dict[str, Any]], str]:
    source = dataset_root / "evaluation" / "functions_schema.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty Tool schema list: {source}")
    tools = []
    names: set[str] = set()
    for index, schema in enumerate(payload):
        if not isinstance(schema, dict):
            raise TypeError(f"Tool schema {index} is not an object")
        name = schema.get("name")
        description = schema.get("description")
        parameters = schema.get("parameters")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Invalid or duplicate Tool name at index {index}: {name}")
        if not isinstance(description, str) or not isinstance(parameters, dict):
            raise TypeError(f"Incomplete Tool schema for {name}")
        names.add(name)
        normalized_parameters = dict(parameters)
        normalized_parameters.setdefault("additionalProperties", False)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": normalized_parameters,
                },
            }
        )
    return tools, _sha256(source)


def _tool_validators(tools: list[dict[str, Any]]) -> dict[str, Draft202012Validator]:
    return {
        str(tool["function"]["name"]): Draft202012Validator(
            tool["function"]["parameters"]
        )
        for tool in tools
    }


def _load_tool_modules(
    dataset_root: Path, tools: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    enable_ubuntu_runtime()
    from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime

    runtime = VehicleWorldRuntime(dataset_root)
    by_tool = runtime.module_for_tools(runtime.create_world())
    schema_names = {str(tool["function"]["name"]) for tool in tools}
    if set(by_tool) != schema_names:
        raise ValueError(
            "Official Tool schemas and VehicleWorld functions differ: "
            f"missing_modules={sorted(schema_names - set(by_tool))[:5]}, "
            f"missing_schemas={sorted(set(by_tool) - schema_names)[:5]}"
        )
    by_module: dict[str, list[str]] = {}
    for name, module in sorted(by_tool.items()):
        by_module.setdefault(module, []).append(name)
    return by_tool, by_module


def _validate_gold_calls(
    raw_calls: Any,
    *,
    validators: dict[str, Draft202012Validator],
    location: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_calls, list) or not raw_calls:
        raise ValueError(f"Missing Gold Tool calls in {location}")
    calls = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            raise TypeError(f"Gold call {index} is not an object in {location}")
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or name not in validators:
            raise ValueError(f"Unknown Gold Tool {name!r} in {location}")
        if not isinstance(arguments, dict):
            raise TypeError(
                f"Gold arguments must be an object for {name} in {location}"
            )
        try:
            validators[name].validate(arguments)
        except JsonSchemaValidationError as exc:
            raise ValueError(
                f"Invalid Gold arguments for {name} in {location}: {exc.message}"
            ) from exc
        calls.append({"name": name, "arguments": arguments})
    return calls


def _user_message(memory: str, query: str) -> str:
    return f"[Memory]\n{memory or '(empty)'}\n\n[Current request]\n{query}"


def _assistant_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": call["arguments"],
            },
        }
        for call in calls
    ]


def _deterministic_choice(candidates: list[str], *, key: str) -> str | None:
    if not candidates:
        return None
    ordered = sorted(candidates)
    index = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(ordered)
    return ordered[index]


def _candidate_tools(
    *,
    sample_id: str,
    gold_names: set[str],
    tool_modules: dict[str, str],
    module_tools: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Return Gold + one same-module + one global negative candidate set."""
    gold_modules = sorted({tool_modules[name] for name in gold_names})
    same_module_pool = [
        name
        for module in gold_modules
        for name in module_tools[module]
        if name not in gold_names
    ]
    same_module = _deterministic_choice(
        same_module_pool, key=f"{sample_id}:same-module"
    )
    selected = set(gold_names)
    if same_module is not None:
        selected.add(same_module)

    global_pool = [
        name
        for name, module in tool_modules.items()
        if name not in selected and module not in gold_modules
    ]
    # Singleton modules have no same-module negative. Preserve the candidate budget
    # with a second global negative rather than duplicating a schema.
    required_global = 2 if same_module is None else 1
    for offset in range(required_global):
        candidate = _deterministic_choice(
            [name for name in global_pool if name not in selected],
            key=f"{sample_id}:global:{offset}",
        )
        if candidate is not None:
            selected.add(candidate)

    # Sorting avoids encoding the Gold Tool through candidate position.
    return sorted(selected), gold_modules


def _convert_row(
    source: dict[str, Any],
    *,
    source_name: str,
    line_number: int,
    summaries: IndexedMemoryDataset,
    validators: dict[str, Draft202012Validator],
    tool_modules: dict[str, str],
    module_tools: dict[str, list[str]],
    tools_sha256: str,
) -> dict[str, Any]:
    location = f"{source_name}:{line_number}"
    scenario = source.get("scenario_index")
    if not isinstance(scenario, int):
        raise TypeError(f"Invalid scenario_index in {location}")
    source_split = source.get("split")
    if source_split != split_for_scenario(scenario):
        raise ValueError(f"Invalid source split in {location}: {source_split}")
    quiz_type = source.get("quiz_type")
    expected_type = "TURN" if source_name == "turn_quiz.jsonl" else "FINAL"
    if quiz_type != expected_type:
        raise ValueError(
            f"Expected {expected_type} Quiz in {location}, got {quiz_type}"
        )

    memory_ref = source.get("memory_ref")
    if not isinstance(memory_ref, dict):
        raise TypeError(f"Missing memory_ref in {location}")
    global_turn = memory_ref.get("global_turn_index")
    if not isinstance(global_turn, int):
        raise TypeError(f"Invalid memory global_turn_index in {location}")
    row_id = summaries.catalog.row_id_for_turn(scenario, global_turn)
    summary = summaries[summaries.position_for_row_id(row_id)]
    expected_summary_id = memory_ref.get("summary_sample_id")
    if summary.get("sample_id") != expected_summary_id:
        raise ValueError(
            f"Summary reference mismatch in {location}: "
            f"{expected_summary_id!r} != {summary.get('sample_id')!r}"
        )
    memory = summary.get("target", {}).get("next_memory")
    if not isinstance(memory, str):
        raise TypeError(f"Referenced Summary has no next_memory in {location}")
    memory_hash = _text_sha256(memory)
    if memory_hash != memory_ref.get("memory_snapshot_sha256"):
        raise ValueError(f"Memory snapshot hash mismatch in {location}")

    input_payload = source.get("input")
    if not isinstance(input_payload, dict):
        raise TypeError(f"Missing input object in {location}")
    query = input_payload.get("query")
    reasoning_type = input_payload.get("reasoning_type")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Empty Quiz query in {location}")
    if reasoning_type not in REASONING_TYPES:
        raise ValueError(f"Invalid reasoning_type {reasoning_type!r} in {location}")
    calls = _validate_gold_calls(
        source.get("target", {}).get("gold_calls"),
        validators=validators,
        location=location,
    )
    sample_id = str(source["sample_id"])
    selected_tool_names, selected_modules = _candidate_tools(
        sample_id=sample_id,
        gold_names={call["name"] for call in calls},
        tool_modules=tool_modules,
        module_tools=module_tools,
    )
    assistant_calls = _assistant_tool_calls(calls)
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "scenario_index": scenario,
        "source_split": source_split,
        "sft_split": sft_split_for_scenario(scenario),
        "quiz_type": quiz_type,
        "quiz_id": str(source["quiz_id"]),
        "reasoning_type": reasoning_type,
        "memory_ref": {
            key: memory_ref[key]
            for key in (
                "checkpoint_id",
                "position",
                "global_turn_index",
                "turn_id",
                "memory_snapshot_sha256",
                "summary_sample_id",
                "patch_sample_id",
                "delta_sample_id",
            )
            if key in memory_ref
        },
        "tool_schema_ref": {
            "path": "vehicle_tools.json",
            "sha256": tools_sha256,
            "schema_version": TOOL_SCHEMA_VERSION,
            "selection_policy": "gold_plus_same_module_plus_global_negative",
            "module_names": selected_modules,
            "tool_names": selected_tool_names,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(memory, query.strip())},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_calls,
            },
        ],
        "target": {"tool_calls": calls},
    }


def build_quiz_sft(
    data_root: Path,
    dataset_root: Path,
    *,
    output_path: Path | None = None,
    tools_path: Path | None = None,
    manifest_path: Path | None = None,
    catalog_path: Path | None = None,
    expected_scenarios: Sequence[int] = tuple(range(1, 101)),
) -> dict[str, Any]:
    """Build all rows atomically and return the generated manifest."""
    data_root = data_root.resolve()
    output_path = (output_path or data_root / "quiz_sft.jsonl").resolve()
    tools_path = (tools_path or data_root / "vehicle_tools.json").resolve()
    manifest_path = (manifest_path or data_root / "quiz_sft_manifest.json").resolve()
    catalog = ensure_catalog(
        data_root,
        catalog_path or default_catalog_path(DEFAULT_WORKSPACE_ROOT),
    )
    summaries = IndexedMemoryDataset(catalog, "summary")
    tools, source_tool_sha256 = _load_tool_schemas(dataset_root.resolve())
    validators = _tool_validators(tools)
    tool_modules, module_tools = _load_tool_modules(dataset_root.resolve(), tools)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_tools = tools_path.with_suffix(tools_path.suffix + ".tmp")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    for temporary in (temporary_output, temporary_tools, temporary_manifest):
        temporary.unlink(missing_ok=True)

    tool_payload = {
        "schema_version": TOOL_SCHEMA_VERSION,
        "source": str(dataset_root.resolve() / "evaluation" / "functions_schema.json"),
        "source_sha256": source_tool_sha256,
        "tools": tools,
        "modules": module_tools,
    }
    temporary_tools.write_text(
        json.dumps(tool_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tools_sha256 = _sha256(temporary_tools)

    count_by_split: Counter[str] = Counter()
    count_by_type: Counter[str] = Counter()
    count_by_reason: Counter[str] = Counter()
    count_by_candidates: Counter[int] = Counter()
    count_by_scenario_type: Counter[tuple[int, str]] = Counter()
    total_calls = 0
    seen_ids: set[str] = set()
    rows = 0
    try:
        with temporary_output.open("w", encoding="utf-8") as output:
            for source_name in QUIZ_FILES:
                with (data_root / source_name).open(encoding="utf-8") as source_handle:
                    for line_number, line in enumerate(source_handle, start=1):
                        try:
                            source = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSON in {source_name}:{line_number}"
                            ) from exc
                        row = _convert_row(
                            source,
                            source_name=source_name,
                            line_number=line_number,
                            summaries=summaries,
                            validators=validators,
                            tool_modules=tool_modules,
                            module_tools=module_tools,
                            tools_sha256=tools_sha256,
                        )
                        sample_id = row["sample_id"]
                        if sample_id in seen_ids:
                            raise ValueError(f"Duplicate Quiz sample_id: {sample_id}")
                        seen_ids.add(sample_id)
                        output.write(_canonical_json(row) + "\n")
                        count_by_split[row["sft_split"]] += 1
                        count_by_type[row["quiz_type"]] += 1
                        count_by_reason[row["reasoning_type"]] += 1
                        count_by_scenario_type[
                            (int(row["scenario_index"]), str(row["quiz_type"]))
                        ] += 1
                        count_by_candidates[
                            len(row["tool_schema_ref"]["tool_names"])
                        ] += 1
                        total_calls += len(row["target"]["tool_calls"])
                        rows += 1

        expected_scenarios = tuple(dict.fromkeys(expected_scenarios))
        if not expected_scenarios:
            raise ValueError("Expected Quiz scenarios cannot be empty")
        expected_by_split = Counter(
            sft_split_for_scenario(scenario) for scenario in expected_scenarios
        )
        expected_by_split = {
            split: count * 40 for split, count in expected_by_split.items()
        }
        if dict(count_by_split) != expected_by_split:
            raise ValueError(
                f"Unexpected SFT split counts: {dict(count_by_split)} "
                f"!= {expected_by_split}"
            )
        expected_by_type = {
            "TURN": len(expected_scenarios) * 30,
            "FINAL": len(expected_scenarios) * 10,
        }
        if dict(count_by_type) != expected_by_type:
            raise ValueError(f"Unexpected Quiz type counts: {dict(count_by_type)}")
        expected_scenario_types = {
            (scenario, quiz_type): count
            for scenario in expected_scenarios
            for quiz_type, count in (("TURN", 30), ("FINAL", 10))
        }
        if dict(count_by_scenario_type) != expected_scenario_types:
            raise ValueError(
                "Unexpected per-scenario Quiz counts: "
                f"{dict(count_by_scenario_type)}"
            )

        output_sha256 = _sha256(temporary_output)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root),
            "source_files": {
                name: {"sha256": _sha256(data_root / name)} for name in QUIZ_FILES
            },
            "split_policy": {
                "excluded": "S1-S14",
                "train": "S15-S80 + T1-T10",
                "validation": "S81-S85 + T11",
                "test": "S86-S100 + T12-T20",
            },
            "counts": {
                "rows": rows,
                "tool_calls": total_calls,
                "by_sft_split": dict(count_by_split),
                "by_quiz_type": dict(count_by_type),
                "by_reasoning_type": dict(sorted(count_by_reason.items())),
                "by_candidate_tool_count": {
                    str(count): rows
                    for count, rows in sorted(count_by_candidates.items())
                },
            },
            "files": {
                "quiz_sft": {
                    "path": str(output_path),
                    "bytes": temporary_output.stat().st_size,
                    "sha256": output_sha256,
                },
                "vehicle_tools": {
                    "path": str(tools_path),
                    "bytes": temporary_tools.stat().st_size,
                    "sha256": tools_sha256,
                    "source_sha256": source_tool_sha256,
                    "tool_count": len(tools),
                },
            },
        }
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_output, output_path)
        os.replace(temporary_tools, tools_path)
        os.replace(temporary_manifest, manifest_path)
        return manifest
    except BaseException:
        for temporary in (temporary_output, temporary_tools, temporary_manifest):
            temporary.unlink(missing_ok=True)
        raise
    finally:
        summaries.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/hj153lee/VehicleMemBench")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tools-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    manifest = build_quiz_sft(
        args.data_root,
        args.dataset_root,
        output_path=args.output,
        tools_path=args.tools_output,
        manifest_path=args.manifest_output,
        catalog_path=args.catalog,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
