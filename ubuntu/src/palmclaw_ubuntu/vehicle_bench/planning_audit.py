from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from palmclaw_ubuntu.memory_tool_planning import (
    build_memory_tool_plan,
    choose_memory_tool_hint_result,
)
from palmclaw_ubuntu.models import FactMemoryRecord, FactQueryContext
from palmclaw_ubuntu.tool_memory_execution import ToolMemoryHintResult
from palmclaw_ubuntu.vehicle_bench.dataset import load_vehicle_benchmark
from palmclaw_ubuntu.vehicle_bench.tools import vehicle_tool_definitions

DEFAULT_RUN_IDS = (
    "a9215a40-ff50-4722-9173-f33d1ad02f4c",
    "15f5a789-8dd5-4d6c-9190-887290bd5bff",
)


def audit_joint_memory_tool_planning(
    *,
    benchmark_root: Path,
    evaluation_root: Path,
    cache_root: Path,
    run_ids: Sequence[str] = DEFAULT_RUN_IDS,
    max_tools: int = 12,
    schema_token_budget: int = 1_800,
) -> dict[str, Any]:
    """Replay fixed selected Facts without Memory, embedding, or Agent calls."""

    dataset = load_vehicle_benchmark(benchmark_root)
    definitions = vehicle_tool_definitions(dataset.tool_schemas)
    cases = _load_cases(evaluation_root, run_ids)
    rows = []
    for case in cases:
        context = _mapping(case.get("context"))
        metadata = _mapping(context.get("retrieval_metadata"))
        selected_ids = tuple(
            str(item)
            for item in metadata.get("fact_memory_retrieval_selected_ids", ())
        )
        cache_key = str(metadata.get("fact_component_cache_key") or "")
        scenario_index = int(case["scenario_index"])
        records = _load_selected_records(
            cache_root=cache_root,
            dataset_prefix=dataset.manifest.dataset_sha256[:16],
            scenario_index=scenario_index,
            cache_key=cache_key,
            selected_ids=selected_ids,
        )
        routed = tuple(str(item) for item in context.get("selector_tool_names", ()))
        plan = build_memory_tool_plan(
            query=str(case["query"]),
            records=records,
            definitions=definitions,
            query_context=_query_context(metadata),
            routed_tools=routed,
            current_state=_mapping(
                _mapping(context.get("simulator")).get("initial_state")
            )
            or None,
            max_tools=max_tools,
            schema_token_budget=schema_token_budget,
        )
        baseline_hints = tuple(
            _mapping(item)
            for item in context.get("tool_memory_execution_hints", ())
        )
        planned_hint_result = plan.as_hint_result()
        fallback_hint_result = ToolMemoryHintResult(
            hints=tuple(_artifact_hint(item) for item in baseline_hints),
            rejections=(),
        )
        effective_hint_result = choose_memory_tool_hint_result(
            planned_hint_result,
            fallback_hint_result,
        )
        effective_hints = effective_hint_result.hints
        joint_selector = (
            tuple(item.tool_name for item in plan.candidates) or routed
        )
        baseline_boundary = tuple(
            dict.fromkeys(
                (
                    *routed,
                    *(str(item.get("tool_name")) for item in baseline_hints),
                )
            )
        )
        joint_boundary = tuple(
            dict.fromkeys(
                (
                    *joint_selector,
                    *(item.tool_name for item in effective_hints),
                )
            )
        )
        reference_calls = tuple(
            {
                "tool_name": str(item["name"]),
                "arguments": dict(_mapping(item.get("args"))),
            }
            for item in case.get("reference_calls", ())
        )
        plan_calls = tuple(
            {"tool_name": item.tool_name, "arguments": dict(item.arguments)}
            for item in plan.calls
        )
        baseline_calls = tuple(
            {
                "tool_name": str(item.get("tool_name")),
                "arguments": dict(_mapping(item.get("arguments"))),
            }
            for item in baseline_hints
        )
        effective_calls = tuple(
            {
                "tool_name": item.tool_name,
                "arguments": dict(item.arguments),
            }
            for item in effective_hints
        )
        reference_tools = {item["tool_name"] for item in reference_calls}
        bundle_sizes = [len(bundle.records) for bundle in plan.bundles]
        rows.append(
            {
                "task_id": str(case["task_id"]),
                "scenario_index": scenario_index,
                "reasoning_type": str(case.get("reasoning_type") or "unknown"),
                "selected_fact_count": len(selected_ids),
                "related_fact_count": max(0, len(records) - len(selected_ids)),
                "planning_fact_count": len(records),
                "bundle_count": len(plan.bundles),
                "max_bundle_size": max(bundle_sizes, default=0),
                "multi_fact_bundle_count": sum(size > 1 for size in bundle_sizes),
                "plan_status": plan.status,
                "baseline_selector_count": len(routed),
                "joint_selector_count": len(joint_selector),
                "baseline_selector_complete": reference_tools <= set(routed),
                "joint_selector_complete": reference_tools <= set(joint_selector),
                "baseline_boundary_complete": reference_tools
                <= set(baseline_boundary),
                "joint_boundary_complete": reference_tools <= set(joint_boundary),
                "baseline_hint_count": len(baseline_hints),
                "planned_call_count": len(plan.calls),
                "effective_hint_count": len(effective_hints),
                "baseline_rejection_count": int(
                    context.get("tool_memory_execution_hint_rejection_count", 0)
                ),
                "plan_rejection_count": len(plan.rejections),
                "multi_fact_call_count": sum(
                    len(item.record_ids) > 1 for item in plan.calls
                ),
                "baseline_exact_reference_calls": _matching_call_count(
                    baseline_calls,
                    reference_calls,
                ),
                "planned_exact_reference_calls": _matching_call_count(
                    plan_calls,
                    reference_calls,
                ),
                "effective_exact_reference_calls": _matching_call_count(
                    effective_calls,
                    reference_calls,
                ),
                "reference_call_count": len(reference_calls),
                "planned_nonreference_call_count": sum(
                    item["tool_name"] not in reference_tools for item in plan_calls
                ),
                "schema_tokens_selected": int(
                    plan.metadata.get("schema_tokens_selected", 0)
                ),
                "fallback": effective_hint_result is fallback_hint_result,
            }
        )
    return {
        "mode": "fixed_cache_offline_joint_planning_audit_v1",
        "benchmark": dataset.manifest.as_dict(),
        "source_run_ids": list(run_ids),
        "settings": {
            "max_tools": max_tools,
            "schema_token_budget": schema_token_budget,
            "memory_model_calls": 0,
            "embedding_calls": 0,
            "agent_calls": 0,
            "gold_used_for_generation": False,
            "gold_used_for_scoring_only": True,
        },
        "summary": _summarize(rows),
        "tasks": rows,
    }


def write_planning_audit(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["tasks"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "results.md").write_text(
        _render_markdown(_mapping(result["summary"])),
        encoding="utf-8",
    )


def _load_cases(evaluation_root: Path, run_ids: Sequence[str]) -> list[dict[str, Any]]:
    cases = []
    for run_id in run_ids:
        path = evaluation_root / run_id / "cases.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation cases: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
    return sorted(cases, key=lambda item: (item["scenario_index"], item["task_id"]))


def _load_selected_records(
    *,
    cache_root: Path,
    dataset_prefix: str,
    scenario_index: int,
    cache_key: str,
    selected_ids: Sequence[str],
) -> tuple[FactMemoryRecord, ...]:
    if not cache_key or not selected_ids:
        return ()
    preferred_database = (
        cache_root
        / dataset_prefix
        / f"scenario-{scenario_index:02d}"
        / cache_key[:16]
        / "memory.db"
    )
    scenario_root = preferred_database.parent.parent
    databases = [preferred_database]
    databases.extend(
        database
        for database in sorted(scenario_root.glob("*/memory.db"))
        if database != preferred_database
    )
    if not preferred_database.is_file():
        raise FileNotFoundError(f"Missing fixed Fact cache: {preferred_database}")
    for database in databases:
        records = _read_selected_records(database, selected_ids)
        if records is not None:
            return records
    raise RuntimeError(
        "Selected Fact records do not coexist in any fixed scenario cache: "
        f"scenario={scenario_index}, preferred={cache_key[:16]}"
    )


def _read_selected_records(
    database: Path,
    selected_ids: Sequence[str],
) -> tuple[FactMemoryRecord, ...] | None:
    uri = f"file:{database.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        resolved_ids = _resolve_redacted_ids(connection, selected_ids)
        if resolved_ids is None:
            return None
        placeholders = ",".join("?" for _ in resolved_ids)
        rows = connection.execute(
            f"SELECT * FROM fact_memory_records WHERE id IN ({placeholders})",
            resolved_ids,
        ).fetchall()
        related_ids = _related_record_ids(connection, rows, resolved_ids)
        all_ids = (*resolved_ids, *related_ids)
        all_placeholders = ",".join("?" for _ in all_ids)
        if related_ids:
            rows = connection.execute(
                f"SELECT * FROM fact_memory_records WHERE id IN ({all_placeholders})",
                all_ids,
            ).fetchall()
        source_rows = connection.execute(
            f"""
            SELECT record_id, MIN(message_id) AS first_message_id
            FROM fact_memory_record_sources
            WHERE record_id IN ({all_placeholders})
            GROUP BY record_id
            """,
            all_ids,
        ).fetchall()
    finally:
        connection.close()
    source_ids = {
        str(row["record_id"]): int(row["first_message_id"])
        for row in source_rows
    }
    records_by_id = {
        str(row["id"]): _decode_record(row, source_ids) for row in rows
    }
    missing = [
        record_id for record_id in resolved_ids if record_id not in records_by_id
    ]
    if missing:
        return None
    return tuple(records_by_id[record_id] for record_id in all_ids)


def _related_record_ids(
    connection: sqlite3.Connection,
    selected_rows: Sequence[sqlite3.Row],
    selected_ids: Sequence[str],
) -> tuple[str, ...]:
    selected_set = set(selected_ids)
    keys = set(selected_rows[0].keys()) if selected_rows else set()
    if "bundle_id" in keys:
        bundle_ids = tuple(
            dict.fromkeys(
                str(row["bundle_id"])
                for row in selected_rows
                if row["bundle_id"] is not None
            )
        )
        if not bundle_ids:
            return ()
        placeholders = ",".join("?" for _ in bundle_ids)
        rows = connection.execute(
            f"""
            SELECT id
            FROM fact_memory_records
            WHERE status = 'active' AND bundle_id IN ({placeholders})
            ORDER BY id
            """,
            bundle_ids,
        ).fetchall()
    else:
        placeholders = ",".join("?" for _ in selected_ids)
        rows = connection.execute(
            f"""
            WITH source_events AS (
                SELECT record_id, MIN(message_id) AS first_message_id
                FROM fact_memory_record_sources
                GROUP BY record_id
            )
            SELECT record.id
            FROM fact_memory_records AS record
            JOIN source_events AS source ON source.record_id = record.id
            WHERE record.status = 'active'
              AND source.first_message_id IN (
                  SELECT first_message_id
                  FROM source_events
                  WHERE record_id IN ({placeholders})
              )
            ORDER BY record.id
            """,
            tuple(selected_ids),
        ).fetchall()
    return tuple(
        str(row["id"]) for row in rows if str(row["id"]) not in selected_set
    )[:12]


def _resolve_redacted_ids(
    connection: sqlite3.Connection,
    selected_ids: Sequence[str],
) -> tuple[str, ...] | None:
    resolved = []
    for record_id in selected_ids:
        if "[REDACTED_" not in record_id:
            resolved.append(record_id)
            continue
        prefix = record_id.split("[REDACTED_", 1)[0]
        candidates = connection.execute(
            "SELECT id FROM fact_memory_records WHERE id LIKE ?",
            (f"{prefix}%",),
        ).fetchall()
        if len(candidates) != 1:
            return None
        resolved.append(str(candidates[0]["id"]))
    return tuple(resolved)


def _decode_record(
    row: sqlite3.Row,
    source_ids: Mapping[str, int],
) -> FactMemoryRecord:
    record_id = str(row["id"])
    keys = set(row.keys())
    stored_bundle = row["bundle_id"] if "bundle_id" in keys else None
    bundle_id = (
        str(stored_bundle)
        if stored_bundle
        else (
            f"message:{source_ids[record_id]}"
            if record_id in source_ids
            else f"fact:{record_id}"
        )
    )
    return FactMemoryRecord(
        id=record_id,
        session_id=str(row["session_id"]),
        record_key=str(row["record_key"]),
        user_id=str(row["user_id"]),
        entity_id=str(row["entity_id"]),
        predicate=str(row["predicate"]),
        identity_conditions=json.loads(row["identity_conditions_json"]),
        applicability=json.loads(row["applicability_json"]),
        capability_hints=tuple(json.loads(row["capability_hints_json"])),
        value=json.loads(row["value_json"]),
        memory_type=str(row["memory_type"]),
        status=str(row["status"]),
        confidence=float(row["confidence"]),
        version=int(row["version"]),
        supersedes_id=row["supersedes_id"],
        merged_into_id=row["merged_into_id"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        bundle_id=bundle_id,
    )


def _query_context(metadata: Mapping[str, Any]) -> FactQueryContext:
    value = _mapping(metadata.get("fact_memory_query_context"))
    return FactQueryContext(
        entities=tuple(str(item) for item in value.get("entities", ())),
        entity_roles={
            str(key): str(item)
            for key, item in _mapping(value.get("entity_roles")).items()
        },
        requester_entity=(
            str(value["requester_entity"])
            if value.get("requester_entity") is not None
            else None
        ),
        time_of_day=(
            str(value["time_of_day"])
            if value.get("time_of_day") is not None
            else None
        ),
        weather=str(value["weather"]) if value.get("weather") is not None else None,
        situation=(
            str(value["situation"])
            if value.get("situation") is not None
            else None
        ),
    )


def _artifact_hint(value: Mapping[str, Any]):
    from palmclaw_ubuntu.tool_memory_execution import ToolMemoryExecutionHint

    return ToolMemoryExecutionHint(
        record_id=str(value.get("record_id") or "artifact"),
        tool_name=str(value.get("tool_name") or ""),
        arguments=dict(_mapping(value.get("arguments"))),
        confidence=float(value.get("confidence", 0.0)),
        tool_domain=str(value.get("tool_domain") or "artifact"),
        topic=str(value.get("topic") or "artifact"),
    )


def _matching_call_count(
    candidates: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
) -> int:
    remaining = list(candidates)
    matched = 0
    for reference in references:
        for index, candidate in enumerate(remaining):
            if candidate == reference:
                matched += 1
                remaining.pop(index)
                break
    return matched


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_count = len(rows)
    reference_calls = sum(int(row["reference_call_count"]) for row in rows)
    status_counts = Counter(str(row["plan_status"]) for row in rows)
    return {
        "task_count": task_count,
        "reference_call_count": reference_calls,
        "baseline_selector_complete_tasks": sum(
            bool(row["baseline_selector_complete"]) for row in rows
        ),
        "joint_selector_complete_tasks": sum(
            bool(row["joint_selector_complete"]) for row in rows
        ),
        "baseline_boundary_complete_tasks": sum(
            bool(row["baseline_boundary_complete"]) for row in rows
        ),
        "joint_boundary_complete_tasks": sum(
            bool(row["joint_boundary_complete"]) for row in rows
        ),
        "baseline_exact_reference_calls": sum(
            int(row["baseline_exact_reference_calls"]) for row in rows
        ),
        "planned_exact_reference_calls": sum(
            int(row["planned_exact_reference_calls"]) for row in rows
        ),
        "effective_exact_reference_calls": sum(
            int(row["effective_exact_reference_calls"]) for row in rows
        ),
        "baseline_rejection_count": sum(
            int(row["baseline_rejection_count"]) for row in rows
        ),
        "plan_rejection_count": sum(
            int(row["plan_rejection_count"]) for row in rows
        ),
        "plan_status_counts": dict(sorted(status_counts.items())),
        "fallback_tasks": sum(bool(row["fallback"]) for row in rows),
        "tasks_with_multi_fact_bundle": sum(
            int(row["multi_fact_bundle_count"]) > 0 for row in rows
        ),
        "multi_fact_call_count": sum(
            int(row["multi_fact_call_count"]) for row in rows
        ),
        "planned_nonreference_call_count": sum(
            int(row["planned_nonreference_call_count"]) for row in rows
        ),
        "baseline_selector_count_mean": _mean(
            rows, "baseline_selector_count"
        ),
        "joint_selector_count_mean": _mean(rows, "joint_selector_count"),
        "baseline_hint_count_mean": _mean(rows, "baseline_hint_count"),
        "planned_call_count_mean": _mean(rows, "planned_call_count"),
        "effective_hint_count_mean": _mean(rows, "effective_hint_count"),
        "related_fact_count_mean": _mean(rows, "related_fact_count"),
        "schema_tokens_selected_mean": _mean(rows, "schema_tokens_selected"),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return round(fmean(float(row[key]) for row in rows), 3) if rows else 0.0


def _render_markdown(summary: Mapping[str, Any]) -> str:
    task_count = int(summary["task_count"])
    reference_calls = int(summary["reference_call_count"])
    return "\n".join(
        (
            "# Joint Memory–Tool Planning fixed-cache audit",
            "",
            "| Metric | Baseline | Joint planner |",
            "| --- | ---: | ---: |",
            (
                "| Selector complete tasks | "
                f"{summary['baseline_selector_complete_tasks']}/{task_count} | "
                f"{summary['joint_selector_complete_tasks']}/{task_count} |"
            ),
            (
                "| Initial boundary complete tasks | "
                f"{summary['baseline_boundary_complete_tasks']}/{task_count} | "
                f"{summary['joint_boundary_complete_tasks']}/{task_count} |"
            ),
            (
                "| Exact reference calls in hints/plan | "
                f"{summary['baseline_exact_reference_calls']}/{reference_calls} | "
                f"{summary['effective_exact_reference_calls']}/{reference_calls} |"
            ),
            (
                "| Mapping rejections | "
                f"{summary['baseline_rejection_count']} | "
                f"{summary['plan_rejection_count']} |"
            ),
            (
                "| Mean selected Tools | "
                f"{summary['baseline_selector_count_mean']} | "
                f"{summary['joint_selector_count_mean']} |"
            ),
            (
                "| Mean executable calls | "
                f"{summary['baseline_hint_count_mean']} | "
                f"{summary['effective_hint_count_mean']} |"
            ),
            "",
            f"- Plan status: `{summary['plan_status_counts']}`",
            f"- Fallback tasks: `{summary['fallback_tasks']}/{task_count}`",
            (
                "- Tasks with a multi-Fact bundle: "
                f"`{summary['tasks_with_multi_fact_bundle']}/{task_count}`"
            ),
            f"- Multi-Fact planned calls: `{summary['multi_fact_call_count']}`",
            (
                "- Planned calls whose Tool is not in the reference: "
                f"`{summary['planned_nonreference_call_count']}`"
            ),
            (
                "- Mean selected schema tokens: "
                f"`{summary['schema_tokens_selected_mean']}`"
            ),
            (
                "- Mean related Facts added for planning: "
                f"`{summary['related_fact_count_mean']}`"
            ),
            "",
            (
                "Gold reference calls are used only for scoring after "
                "deterministic replay."
            ),
            "No Memory, embedding, or Agent model is invoked.",
            "",
        )
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--max-tools", type=int, default=12)
    parser.add_argument("--schema-token-budget", type=int, default=1_800)
    args = parser.parse_args(argv)
    result = audit_joint_memory_tool_planning(
        benchmark_root=args.benchmark_root,
        evaluation_root=args.evaluation_root,
        cache_root=args.cache_root,
        run_ids=tuple(args.run_ids or DEFAULT_RUN_IDS),
        max_tools=args.max_tools,
        schema_token_budget=args.schema_token_budget,
    )
    write_planning_audit(result, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
