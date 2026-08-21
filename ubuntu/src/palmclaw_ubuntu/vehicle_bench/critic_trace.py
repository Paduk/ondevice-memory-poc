from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

CriticMethod = Literal["summary", "combined"]

SOURCE_INVENTORY_VERSION = "vehiclemembench-v2-critic-source-r2-v1"
SOURCE_REPETITION = 2
SOURCE_DATE_TAG = "20260815"

SUMMARY_PROMPT_VERSION = "vehicle-turnwise-recursive-summary-v1"
SUMMARY_SCHEMA_VERSION = "recursive-summary-v1"
COMBINED_PROMPT_VERSION = (
    "vehicle-turnwise-recursive-summary-temporal-compact-v1-soft30"
)
COMBINED_SCHEMA_VERSION = (
    "recursive-summary-temporal-patch-periodic-compaction-v1-soft-target"
)

_TURN_MARKER = re.compile(r"^\[VehicleMemBench history turn=(\d+)\](?:\n|$)")


@dataclass(frozen=True)
class CriticTraceSource:
    method: CriticMethod
    scenario_index: int
    experiment_root: Path
    database_path: Path
    cache_manifest_path: Path
    cache_key: str
    dataset_sha256: str
    history_sha256: str
    history_line_count: int
    model_id: str
    prompt_version: str
    schema_version: str
    max_memory_chars: int


@dataclass(frozen=True)
class CriticSourceTurn:
    method: CriticMethod
    scenario_index: int
    turn_index: int
    message_id: int
    role: str
    content: str
    date: str
    consolidation_run_id: str
    model_call_id: str
    original_decision: Literal["NO_OP", "UPDATE"]
    original_before_memory: str
    original_after_memory: str
    original_before_sha256: str
    original_after_sha256: str
    recorded_provider_after_sha256: str | None
    storage_transformed: bool
    model_id: str
    prompt_version: str
    schema_version: str
    usage: Mapping[str, int]
    latency_ms: int
    metadata: Mapping[str, Any]

    def as_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "method": self.method,
            "scenario_index": self.scenario_index,
            "turn_index": self.turn_index,
            "message_id": self.message_id,
            "role": self.role,
            "date": self.date,
            "consolidation_run_id": self.consolidation_run_id,
            "model_call_id": self.model_call_id,
            "original_decision": self.original_decision,
            "original_before_sha256": self.original_before_sha256,
            "original_after_sha256": self.original_after_sha256,
            "recorded_provider_after_sha256": self.recorded_provider_after_sha256,
            "storage_transformed": self.storage_transformed,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "usage": dict(self.usage),
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }
        if include_text:
            result.update(
                {
                    "content": self.content,
                    "original_before_memory": self.original_before_memory,
                    "original_after_memory": self.original_after_memory,
                }
            )
        return result


def resolve_r2_experiment_root(
    artifact_root: Path | str,
    *,
    method: CriticMethod,
    scenario_index: int,
) -> Path:
    root = Path(artifact_root).expanduser().resolve()
    _validate_scenario_index(scenario_index)
    if method == "summary":
        name = (
            f"turnwise-summary-fresh-r{SOURCE_REPETITION}-s{scenario_index}-"
            f"{SOURCE_DATE_TAG}"
        )
        return root / name
    if method == "combined":
        if scenario_index <= 10:
            name = (
                "turnwise-temporal-vs-combined-fresh-"
                f"r{SOURCE_REPETITION}-s{scenario_index}-{SOURCE_DATE_TAG}"
            )
            return root / name / "combined"
        name = (
            f"turnwise-combined-fresh-r{SOURCE_REPETITION}-s{scenario_index}-"
            f"{SOURCE_DATE_TAG}"
        )
        return root / name
    raise ValueError(f"Unsupported critic method: {method}")


def discover_r2_trace_source(
    artifact_root: Path | str,
    *,
    method: CriticMethod,
    scenario_index: int,
) -> CriticTraceSource:
    experiment_root = resolve_r2_experiment_root(
        artifact_root,
        method=method,
        scenario_index=scenario_index,
    )
    cache_root = experiment_root / "cache"
    database_paths = sorted(cache_root.rglob("memory.db"))
    if len(database_paths) != 1:
        raise ValueError(
            f"Expected exactly one R2 {method} memory.db for scenario "
            f"{scenario_index}, found {len(database_paths)} under {cache_root}"
        )
    database_path = database_paths[0].resolve()
    cache_manifest_path = database_path.with_name("manifest.json")
    if not cache_manifest_path.is_file():
        raise ValueError(f"Missing cache manifest: {cache_manifest_path}")
    raw_manifest = _read_json_object(cache_manifest_path)
    config = _require_mapping(raw_manifest, "config")
    recursive = _require_mapping(config, "recursive_summary")
    history = _require_mapping(raw_manifest, "history")
    actual_scenario = _require_int(config, "scenario_index")
    if actual_scenario != scenario_index:
        raise ValueError(
            f"Cache scenario mismatch: expected {scenario_index}, got "
            f"{actual_scenario}"
        )
    expected_prompt, expected_schema = _expected_versions(method)
    prompt_version = _require_str(recursive, "prompt_version")
    schema_version = _require_str(recursive, "schema_version")
    if prompt_version != expected_prompt or schema_version != expected_schema:
        raise ValueError(
            f"Unexpected R2 {method} prompt/schema for scenario {scenario_index}: "
            f"{prompt_version} / {schema_version}"
        )
    if recursive.get("update_cadence") != "history_entry":
        raise ValueError(
            f"R2 {method} scenario {scenario_index} is not turn-wise"
        )
    return CriticTraceSource(
        method=method,
        scenario_index=scenario_index,
        experiment_root=experiment_root.resolve(),
        database_path=database_path,
        cache_manifest_path=cache_manifest_path.resolve(),
        cache_key=_require_str(raw_manifest, "cache_key"),
        dataset_sha256=_require_str(config, "dataset_sha256"),
        history_sha256=_require_str(config, "history_sha256"),
        history_line_count=_require_int(history, "line_count"),
        model_id=_require_str(recursive, "model_id"),
        prompt_version=prompt_version,
        schema_version=schema_version,
        max_memory_chars=_require_int(recursive, "max_memory_chars"),
    )


def extract_source_turns(source: CriticTraceSource) -> tuple[CriticSourceTurn, ...]:
    connection = _open_readonly(source.database_path)
    try:
        connection.row_factory = sqlite3.Row
        messages = {
            int(row["id"]): row
            for row in connection.execute(
                """
                SELECT id, session_id, role, content
                FROM messages
                ORDER BY id
                """
            ).fetchall()
        }
        run_rows = connection.execute(
            """
            SELECT *
            FROM consolidation_runs
            ORDER BY end_message_id, started_at, id
            """
        ).fetchall()
        call_rows = connection.execute(
            """
            SELECT *
            FROM model_calls
            WHERE role = 'memory'
            ORDER BY created_at, id
            """
        ).fetchall()
    finally:
        connection.close()
    calls_by_run: dict[str, list[sqlite3.Row]] = {}
    for call in call_rows:
        run_id = str(call["consolidation_run_id"] or "")
        calls_by_run.setdefault(run_id, []).append(call)
    if len(run_rows) != source.history_line_count:
        raise ValueError(
            f"R2 {source.method} scenario {source.scenario_index} expected "
            f"{source.history_line_count} runs, found {len(run_rows)}"
        )
    if len(messages) != source.history_line_count:
        raise ValueError(
            f"R2 {source.method} scenario {source.scenario_index} expected "
            f"{source.history_line_count} messages, found {len(messages)}"
        )
    current_memory = ""
    turns: list[CriticSourceTurn] = []
    session_id: str | None = None
    for expected_turn_index, run in enumerate(run_rows):
        run_id = str(run["id"])
        if run["status"] != "completed":
            raise ValueError(f"Source consolidation run is not completed: {run_id}")
        start_message_id = int(run["start_message_id"])
        end_message_id = int(run["end_message_id"])
        if start_message_id != end_message_id:
            raise ValueError(f"Source run is not turn-wise: {run_id}")
        message = messages.get(end_message_id)
        if message is None:
            raise ValueError(f"Source run references a missing message: {run_id}")
        if session_id is None:
            session_id = str(run["session_id"])
        if str(run["session_id"]) != session_id:
            raise ValueError("Source database contains multiple sessions")
        if str(message["session_id"]) != session_id:
            raise ValueError(f"Message session mismatch for source run: {run_id}")
        content = str(message["content"])
        marker = _TURN_MARKER.match(content)
        if marker is None or int(marker.group(1)) != expected_turn_index:
            raise ValueError(
                f"Unexpected history turn marker at source turn "
                f"{expected_turn_index}"
            )
        successful_calls: list[tuple[sqlite3.Row, dict[str, Any]]] = []
        for call in calls_by_run.get(run_id, ()):
            metadata = _decode_json_object(call["metadata_json"], "metadata_json")
            if metadata.get("update_status") in {"updated", "noop"}:
                successful_calls.append((call, metadata))
        if len(successful_calls) != 1:
            raise ValueError(
                f"Expected one successful memory call for source run {run_id}, "
                f"found {len(successful_calls)}"
            )
        call, metadata = successful_calls[0]
        _validate_call_versions(source, call)
        update_status = str(metadata["update_status"])
        run_output = str(run["output"] or "")
        before_memory = current_memory
        if update_status == "updated":
            if not run_output.strip():
                raise ValueError(f"Updated source run has empty output: {run_id}")
            after_memory = run_output
            original_decision: Literal["NO_OP", "UPDATE"] = "UPDATE"
        else:
            if run_output:
                raise ValueError(f"NO_OP source run has stored output: {run_id}")
            after_memory = before_memory
            original_decision = "NO_OP"
        before_sha256 = _text_sha256(before_memory)
        after_sha256 = _text_sha256(after_memory)
        storage_transformed = _validate_recorded_hashes(
            metadata,
            before_sha256=before_sha256,
            after_sha256=after_sha256,
            run_id=run_id,
        )
        recorded_after_sha256 = metadata.get("memory_sha256_after")
        usage = _decode_json_object(call["usage_json"], "usage_json")
        turns.append(
            CriticSourceTurn(
                method=source.method,
                scenario_index=source.scenario_index,
                turn_index=expected_turn_index,
                message_id=end_message_id,
                role=str(message["role"]),
                content=content,
                date=str(metadata.get("date", "")),
                consolidation_run_id=run_id,
                model_call_id=str(call["id"]),
                original_decision=original_decision,
                original_before_memory=before_memory,
                original_after_memory=after_memory,
                original_before_sha256=before_sha256,
                original_after_sha256=after_sha256,
                recorded_provider_after_sha256=(
                    str(recorded_after_sha256)
                    if recorded_after_sha256 is not None
                    else None
                ),
                storage_transformed=storage_transformed,
                model_id=str(call["model_id"]),
                prompt_version=str(call["prompt_version"]),
                schema_version=str(call["schema_version"]),
                usage={
                    key: int(usage.get(key, 0) or 0)
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "cached_tokens",
                    )
                },
                latency_ms=int(call["latency_ms"]),
                metadata=metadata,
            )
        )
        current_memory = after_memory
    return tuple(turns)


def build_r2_source_inventory(
    artifact_root: Path | str,
    *,
    scenario_indices: Iterable[int] = range(1, 51),
) -> dict[str, Any]:
    root = Path(artifact_root).expanduser().resolve()
    scenarios = tuple(scenario_indices)
    if not scenarios:
        raise ValueError("At least one scenario is required")
    entries: list[dict[str, Any]] = []
    totals: dict[str, dict[str, int]] = {
        method: {"scenarios": 0, "turns": 0, "updates": 0, "noops": 0}
        for method in ("summary", "combined")
    }
    dataset_hashes: set[str] = set()
    for method in ("summary", "combined"):
        for scenario_index in scenarios:
            source = discover_r2_trace_source(
                root,
                method=method,
                scenario_index=scenario_index,
            )
            turns = extract_source_turns(source)
            update_count = sum(turn.original_decision == "UPDATE" for turn in turns)
            noop_count = len(turns) - update_count
            final_memory = turns[-1].original_after_memory if turns else ""
            entries.append(
                {
                    "method": method,
                    "scenario_index": scenario_index,
                    "experiment_root": str(source.experiment_root),
                    "database_path": str(source.database_path),
                    "database_sha256": _file_sha256(source.database_path),
                    "cache_manifest_path": str(source.cache_manifest_path),
                    "cache_key": source.cache_key,
                    "dataset_sha256": source.dataset_sha256,
                    "history_sha256": source.history_sha256,
                    "model_id": source.model_id,
                    "prompt_version": source.prompt_version,
                    "schema_version": source.schema_version,
                    "turn_count": len(turns),
                    "update_count": update_count,
                    "noop_count": noop_count,
                    "storage_transform_count": sum(
                        turn.storage_transformed for turn in turns
                    ),
                    "final_memory_sha256": _text_sha256(final_memory),
                    "final_memory_chars": len(final_memory),
                }
            )
            method_totals = totals[method]
            method_totals["scenarios"] += 1
            method_totals["turns"] += len(turns)
            method_totals["updates"] += update_count
            method_totals["noops"] += noop_count
            dataset_hashes.add(source.dataset_sha256)
    if len(dataset_hashes) != 1:
        raise ValueError(f"Source traces use multiple datasets: {dataset_hashes}")
    return {
        "version": SOURCE_INVENTORY_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_repetition": SOURCE_REPETITION,
        "source_date_tag": SOURCE_DATE_TAG,
        "artifact_root": str(root),
        "dataset_sha256": next(iter(dataset_hashes)),
        "scenario_indices": list(scenarios),
        "totals": totals,
        "sources": entries,
    }


def write_source_inventory(path: Path | str, inventory: Mapping[str, Any]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _expected_versions(method: CriticMethod) -> tuple[str, str]:
    if method == "summary":
        return SUMMARY_PROMPT_VERSION, SUMMARY_SCHEMA_VERSION
    if method == "combined":
        return COMBINED_PROMPT_VERSION, COMBINED_SCHEMA_VERSION
    raise ValueError(f"Unsupported critic method: {method}")


def _validate_scenario_index(scenario_index: int) -> None:
    if scenario_index < 1 or scenario_index > 50:
        raise ValueError("VehicleMemBench scenario must be between 1 and 50")


def _validate_call_versions(
    source: CriticTraceSource,
    call: sqlite3.Row,
) -> None:
    actual = (
        str(call["model_id"]),
        str(call["prompt_version"]),
        str(call["schema_version"]),
    )
    expected = (source.model_id, source.prompt_version, source.schema_version)
    if actual != expected:
        raise ValueError(
            f"Source model call version mismatch for scenario "
            f"{source.scenario_index}: expected {expected}, got {actual}"
        )


def _validate_recorded_hashes(
    metadata: Mapping[str, Any],
    *,
    before_sha256: str,
    after_sha256: str,
    run_id: str,
) -> bool:
    recorded_before = metadata.get("memory_sha256_before")
    if recorded_before is not None and recorded_before != before_sha256:
        raise ValueError(f"Recorded before-memory hash mismatch: {run_id}")
    recorded_after = metadata.get("memory_sha256_after")
    # The provider records its after-hash before local storage redaction and
    # line-safe truncation. The database output is the authoritative input to
    # the next turn, so an after-hash mismatch is retained as a storage
    # transformation instead of treated as source corruption.
    return recorded_after is not None and recorded_after != after_sha256


def _read_json_object(path: Path) -> dict[str, Any]:
    return _decode_json_object(path.read_text(encoding="utf-8"), str(path))


def _decode_json_object(value: Any, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON object in {label}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected JSON object in {label}")
    return decoded


def _require_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"Expected object field: {key}")
    return item


def _require_str(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"Expected non-empty string field: {key}")
    return item


def _require_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int):
        raise ValueError(f"Expected integer field: {key}")
    return item


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
