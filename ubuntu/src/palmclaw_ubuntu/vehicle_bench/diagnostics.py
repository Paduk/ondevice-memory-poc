from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.normalization import (
    contains_update_intent as fact_contains_update_intent,
)
from palmclaw_ubuntu.normalization import (
    normalized_value_support as fact_normalized_value_support,
)
from palmclaw_ubuntu.privacy import redact_data_for_cloud
from palmclaw_ubuntu.tool_memory_schema import normalize_identifier

_RUNTIME_BOUND_FIELDS = frozenset(
    {
        "driver",
        "light",
        "occupant",
        "passenger",
        "person",
        "seat",
        "side",
        "window",
        "zone",
    }
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True)
class VehicleR0Result:
    output_dir: Path
    summary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "summary": dict(self.summary),
        }


def run_vehicle_r0_diagnostics(
    run_dir: Path,
    *,
    memory_cache_dir: Path,
    output_dir: Path | None = None,
) -> VehicleR0Result:
    """Audit frozen schema-patch artifacts without invoking a model."""
    resolved_run = run_dir.resolve()
    resolved_cache = memory_cache_dir.resolve()
    manifest = _read_json(resolved_run / "manifest.json")
    cases = _read_jsonl(resolved_run / "cases.jsonl")
    _validate_run(manifest, cases)

    target = (output_dir or (resolved_run / "r0-offline")).resolve()
    target.mkdir(parents=True, exist_ok=True)
    database_paths = _resolve_memory_databases(
        manifest,
        cases,
        memory_cache_dir=resolved_cache,
    )

    proposal_rows: list[dict[str, Any]] = []
    active_records: dict[int, list[dict[str, Any]]] = {}
    for scenario_index, database_path in sorted(database_paths.items()):
        with closing(_read_only_database(database_path)) as connection:
            proposal_rows.extend(
                _proposal_replay_rows(connection, scenario_index=scenario_index)
            )
            active_records[scenario_index] = _active_records(connection)

    proposal_summary = _proposal_summary(proposal_rows)
    routing_rows = [
        _routing_replay_row(
            case,
            active_records=active_records.get(int(case["scenario_index"]), ()),
        )
        for case in cases
    ]
    routing_summary = _routing_summary(routing_rows)
    summary = {
        "mode": "offline_trace_replay",
        "source_run_dir": str(resolved_run),
        "scenario_indices": sorted(database_paths),
        "task_count": len(cases),
        "model_calls": 0,
        "source_databases_opened_read_only": len(database_paths),
        "proposal_replay": proposal_summary,
        "routing_replay": routing_summary,
    }

    safe_summary, _ = redact_data_for_cloud(summary)
    safe_proposals, _ = redact_data_for_cloud(proposal_rows)
    safe_routing, _ = redact_data_for_cloud(routing_rows)
    _write_json(target / "r0-summary.json", safe_summary)
    _write_jsonl(target / "proposal-replay.jsonl", safe_proposals)
    _write_jsonl(target / "routing-replay.jsonl", safe_routing)
    (target / "results.md").write_text(
        _render_markdown(safe_summary),
        encoding="utf-8",
    )
    return VehicleR0Result(output_dir=target, summary=safe_summary)


def _validate_run(
    manifest: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> None:
    profiles = tuple(manifest.get("config", {}).get("profiles", ()))
    if "cloud_schema_patch" not in profiles:
        raise ValueError("R0 diagnostics require a cloud_schema_patch run")
    if not cases:
        raise ValueError("R0 diagnostics require at least one case")
    task_ids = [str(case.get("task_id", "")) for case in cases]
    if any(not task_id for task_id in task_ids):
        raise ValueError("R0 case is missing task_id")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("R0 cases contain duplicate task IDs")


def _resolve_memory_databases(
    manifest: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    memory_cache_dir: Path,
) -> dict[int, Path]:
    dataset_hash = str(
        manifest.get("config", {})
        .get("benchmark", {})
        .get("dataset_sha256", "")
    )
    if len(dataset_hash) < 16:
        raise ValueError("R0 manifest is missing the benchmark dataset hash")
    cache_keys: dict[int, set[str]] = {}
    for case in cases:
        scenario_index = int(case["scenario_index"])
        metadata = case.get("context", {}).get("retrieval_metadata", {})
        cache_key = str(metadata.get("cache_key", ""))
        if not cache_key:
            raise ValueError(
                f"R0 case {case['task_id']} is missing its memory cache key"
            )
        cache_keys.setdefault(scenario_index, set()).add(cache_key)

    resolved: dict[int, Path] = {}
    for scenario_index, keys in cache_keys.items():
        if len(keys) != 1:
            raise ValueError(
                f"Scenario {scenario_index} references multiple memory caches"
            )
        cache_key = next(iter(keys))
        database_path = (
            memory_cache_dir
            / dataset_hash[:16]
            / f"scenario-{scenario_index:02d}"
            / cache_key[:16]
            / "memory.db"
        )
        if not database_path.is_file():
            raise ValueError(f"R0 memory database not found: {database_path}")
        resolved[scenario_index] = database_path
    return resolved


def _read_only_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _proposal_replay_rows(
    connection: sqlite3.Connection,
    *,
    scenario_index: int,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            proposals.id,
            proposals.operation,
            proposals.target_record_id,
            proposals.patch_json,
            proposals.evidence_json,
            proposals.status,
            proposals.rejection_reason,
            proposals.validation_json,
            proposals.sequence_index,
            runs.id AS run_id
        FROM memory_patch_proposals AS proposals
        JOIN memory_patch_runs AS runs ON runs.id = proposals.run_id
        ORDER BY proposals.created_at, proposals.id
        """
    ).fetchall()
    replay_rows = []
    for row in rows:
        patch = json.loads(row["patch_json"])
        evidence = json.loads(row["evidence_json"])
        validation = json.loads(row["validation_json"] or "{}")
        code = _rejection_code(row["rejection_reason"], validation)
        preview, reasons = _relaxed_preview(
            patch,
            evidence,
            current_status=str(row["status"]),
            rejection_code=code,
        )
        replay_rows.append(
            {
                "scenario_index": scenario_index,
                "proposal_id": str(row["id"]),
                "run_id": str(row["run_id"]),
                "sequence_index": row["sequence_index"],
                "operation": str(row["operation"]),
                "current_status": str(row["status"]),
                "rejection_code": code,
                "dedup_key": _proposal_dedup_key(patch, evidence),
                "relaxed_preview": preview,
                "preview_reasons": reasons,
            }
        )
    return replay_rows


def _active_records(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT tool_domain, topic, entity_id, conditions_json, value_json
        FROM tool_memory_records
        WHERE status = 'active'
        ORDER BY tool_domain, topic, id
        """
    ).fetchall()
    return [
        {
            "tool_domain": str(row["tool_domain"]),
            "topic": str(row["topic"]),
            "entity_id": row["entity_id"],
            "conditions": json.loads(row["conditions_json"]),
            "value": json.loads(row["value_json"]),
        }
        for row in rows
    ]


def _relaxed_preview(
    patch: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    *,
    current_status: str,
    rejection_code: str | None,
) -> tuple[str, list[str]]:
    if current_status == "applied":
        return "accepted_existing", []
    evidence_text = " ".join(str(item.get("quote", "")) for item in evidence)
    value = patch.get("value")
    value_supported, unresolved = _normalized_value_support(value, evidence_text)
    if rejection_code == "unsupported_value":
        if value_supported:
            return "would_accept", ["normalized_value_entailment"]
        if unresolved:
            return "would_review", [f"unresolved_value:{item}" for item in unresolved]
        return "still_reject", ["value_not_supported"]
    if rejection_code == "missing_explicit_update":
        if not value_supported:
            return "would_review", [
                *(f"unresolved_value:{item}" for item in unresolved),
                "update_value_requires_review",
            ]
        if _contains_update_intent(evidence_text):
            return "would_accept", ["normalized_update_intent"]
        return "would_review", ["update_intent_requires_semantic_review"]
    return "still_reject", [rejection_code or "unknown_rejection"]


def _normalized_value_support(
    value: Any,
    evidence_text: str,
) -> tuple[bool, list[str]]:
    support = fact_normalized_value_support(value, evidence_text)
    return support.supported, list(support.unresolved)


def _value_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else str(key)
            yield from _value_leaves(nested, nested_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            yield from _value_leaves(nested, f"{prefix}_{index}")
        return
    if value is not None:
        yield prefix, value


def _contains_update_intent(text: str) -> bool:
    return fact_contains_update_intent(text)


def _proposal_dedup_key(
    patch: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> str:
    identity = patch.get("identity")
    if not isinstance(identity, Mapping):
        identity = {}
    conditions = identity.get("conditions")
    if not isinstance(conditions, Mapping):
        conditions = {}
    evidence_ids = sorted(
        {
            int(item["message_id"])
            for item in evidence
            if item.get("message_id") is not None
        }
    )
    payload = {
        "domain": normalize_identifier(str(identity.get("tool_domain", ""))),
        "topic": normalize_identifier(str(identity.get("topic", ""))),
        "entity": normalize_identifier(
            str(
                conditions.get("person")
                or conditions.get("subject")
                or conditions.get("driver")
                or ""
            )
        ).split("_")[0],
        "value": patch.get("value"),
        "target_record_id": patch.get("target_record_id"),
        "evidence_message_ids": evidence_ids,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _rejection_code(
    reason: str | None,
    validation: Mapping[str, Any],
) -> str | None:
    code = validation.get("code")
    if code:
        return str(code)
    if reason:
        return str(reason).split(":", 1)[0]
    return None


def _proposal_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["dedup_key"]), row)
    return {
        "raw_proposal_count": len(rows),
        "unique_proposal_count": len(unique),
        "raw_status_counts": _count(rows, "current_status"),
        "unique_status_counts": _count(unique.values(), "current_status"),
        "raw_rejection_code_counts": _count(rows, "rejection_code", skip_none=True),
        "unique_rejection_code_counts": _count(
            unique.values(),
            "rejection_code",
            skip_none=True,
        ),
        "raw_relaxed_preview_counts": _count(rows, "relaxed_preview"),
        "unique_relaxed_preview_counts": _count(
            unique.values(),
            "relaxed_preview",
        ),
        "operation_counts": _count(rows, "operation"),
    }


def _routing_replay_row(
    case: Mapping[str, Any],
    *,
    active_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = case.get("context", {}).get("retrieval_metadata", {})
    route = metadata.get("tool_memory_route", {})
    selector_names = tuple(
        str(name) for name in case.get("context", {}).get("selector_tool_names", ())
    )
    reference_calls = tuple(case.get("reference_calls", ()))
    reference_names = tuple(str(call.get("name", "")) for call in reference_calls)
    selected_count = int(metadata.get("tool_memory_retrieval_selected_count", 0))
    candidates = (
        case.get("memory_trace", {}).get("retrieval", {}).get("candidates", ())
    )
    exclusion_counts = Counter(
        str(candidate.get("exclusion_reason"))
        for candidate in candidates
        if candidate.get("exclusion_reason")
    )
    missing_tools = sorted(set(reference_names) - set(selector_names))
    matching_records = sum(
        1
        for record in active_records
        if any(_record_matches_call(record, call) for call in reference_calls)
    )
    if missing_tools:
        outcome = (
            "tool_misrouting_with_answer_memory"
            if matching_records
            else "tool_misrouting_memory_missing"
        )
    elif selected_count == 0 and exclusion_counts.get("condition_mismatch", 0):
        outcome = "condition_mismatch"
    elif selected_count == 0:
        outcome = (
            "answer_memory_outside_selected_candidates"
            if matching_records
            else "memory_missing"
        )
    else:
        outcome = "memory_selected"
    retrieval_quality = case.get("retrieval_quality", {})
    return {
        "task_id": str(case["task_id"]),
        "scenario_index": int(case["scenario_index"]),
        "reasoning_type": str(case.get("reasoning_type", "")),
        "current_outcome": outcome,
        "exact_state_match": bool(
            case.get("score", {}).get("exact_state_match", False)
        ),
        "reference_tool_names": list(reference_names),
        "selector_tool_names": list(selector_names),
        "missing_reference_tools": missing_tools,
        "route_ambiguous": bool(route.get("ambiguous", False)),
        "classifier_used": bool(route.get("classifier_used", False)),
        "candidate_count": int(
            metadata.get("tool_memory_retrieval_candidate_count", 0)
        ),
        "selected_count": selected_count,
        "matching_answer_record_count": matching_records,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "recall_at_k": float(retrieval_quality.get("recall_at_k", 0.0)),
    }


def _record_matches_tool(record: Mapping[str, Any], tool_name: str) -> bool:
    normalized_name = tool_name.removeprefix("carcontrol_")
    if "_set_" not in normalized_name:
        return False
    raw_domain, raw_topic = normalized_name.split("_set_", 1)
    domain = normalize_identifier(_CAMEL_BOUNDARY.sub("_", raw_domain))
    topic = normalize_identifier(_CAMEL_BOUNDARY.sub("_", raw_topic))
    record_domain = normalize_identifier(str(record.get("tool_domain", "")))
    record_topic = normalize_identifier(str(record.get("topic", "")))
    return record_domain == domain and (
        record_topic == topic
        or set(record_topic.split("_")) == set(topic.split("_"))
    )


def _record_matches_call(
    record: Mapping[str, Any],
    call: Mapping[str, Any],
) -> bool:
    if not _record_matches_tool(record, str(call.get("name", ""))):
        return False
    expected = {
        _normalized_scalar(value)
        for key, value in call.get("args", {}).items()
        if normalize_identifier(str(key)) not in _RUNTIME_BOUND_FIELDS
    }
    available = {
        _normalized_scalar(value)
        for key, value in _value_leaves(record.get("value"))
        if normalize_identifier(key) not in _RUNTIME_BOUND_FIELDS
    }
    return bool(expected) and expected <= available


def _normalized_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(float(value))
    return normalize_identifier(str(value))


def _routing_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_count = len(rows)
    return {
        "task_count": task_count,
        "current_outcome_counts": _count(rows, "current_outcome"),
        "zero_selected_tasks": sum(int(row["selected_count"]) == 0 for row in rows),
        "ambiguous_route_tasks": sum(bool(row["route_ambiguous"]) for row in rows),
        "classifier_used_tasks": sum(bool(row["classifier_used"]) for row in rows),
        "tool_misrouting_tasks": sum(
            bool(row["missing_reference_tools"]) for row in rows
        ),
        "answer_memory_route_misses": sum(
            row["current_outcome"] == "tool_misrouting_with_answer_memory"
            for row in rows
        ),
        "mean_selected_memory": (
            sum(int(row["selected_count"]) for row in rows) / task_count
            if task_count
            else 0.0
        ),
        "mean_recall_at_k": (
            sum(float(row["recall_at_k"]) for row in rows) / task_count
            if task_count
            else 0.0
        ),
    }


def _count(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    *,
    skip_none: bool = False,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if skip_none and value is None:
            continue
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _render_markdown(summary: Mapping[str, Any]) -> str:
    proposals = summary["proposal_replay"]
    routing = summary["routing_replay"]
    return "\n".join(
        (
            "# VehicleMemBench R0 Offline Diagnostics",
            "",
            f"- Scenarios: {', '.join(map(str, summary['scenario_indices']))}",
            f"- Tasks: {summary['task_count']}",
            f"- Model calls: {summary['model_calls']}",
            "",
            "## Proposal replay",
            "",
            f"- Raw proposals: {proposals['raw_proposal_count']}",
            f"- Unique proposals: {proposals['unique_proposal_count']}",
            (
                "- Raw current status: "
                f"{_inline_counts(proposals['raw_status_counts'])}"
            ),
            (
                "- Unique relaxed preview: "
                f"{_inline_counts(proposals['unique_relaxed_preview_counts'])}"
            ),
            "",
            "The relaxed result is a deterministic preview, not an applied patch.",
            "",
            "## Routing trace replay",
            "",
            f"- Zero selected Memory tasks: {routing['zero_selected_tasks']}",
            f"- Tool mis-routing tasks: {routing['tool_misrouting_tasks']}",
            (
                "- Route misses with an answer-matching active record: "
                f"{routing['answer_memory_route_misses']}"
            ),
            f"- Ambiguous routes: {routing['ambiguous_route_tasks']}",
            f"- Classifier used: {routing['classifier_used_tasks']}",
            f"- Mean selected Memory: {routing['mean_selected_memory']:.3f}",
            f"- Mean Recall@k: {routing['mean_recall_at_k']:.3f}",
            "",
        )
    )


def _inline_counts(counts: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items()) or "(none)"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"R0 artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"R0 artifact must contain a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"R0 artifact not found: {path}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid R0 JSONL object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
