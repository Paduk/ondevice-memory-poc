"""Deterministically convert flat V2 fact logs into speaker-grouped memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import build_catalog
from .methods.delta_v2 import compact_operations
from .methods.operations import apply_operations, normalize_memory
from .quiz_sft import build_quiz_sft

DEFAULT_SOURCE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "hybrid-s1-s100-plus-temporal-t1-t20-patch-t1t10-v2"
)
DEFAULT_OUTPUT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-s1-s100-plus-temporal-t1-t20-v2"
)
FACT_PATTERN = re.compile(
    r"^- \[(?P<timestamp>[^]]+)] (?P<subject>[^:]+): (?P<payload>.+)$"
)


@dataclass(frozen=True)
class Fact:
    timestamp: str
    subject: str
    payload: str

    @property
    def flat_line(self) -> str:
        return f"- [{self.timestamp}] {self.subject}: {self.payload}"

    @property
    def grouped_line(self) -> str:
        return f"- [{self.timestamp}] {self.payload}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenarios", nargs="+", type=int, default=list(range(1, 121)))
    parser.add_argument(
        "--vehiclemembench-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def parse_flat_memory(memory: str) -> tuple[Fact, ...]:
    facts = []
    for line in normalize_memory(memory).splitlines():
        if not line:
            continue
        match = FACT_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"Malformed flat-memory fact: {line!r}")
        facts.append(Fact(**match.groupdict()))
    return tuple(facts)


def group_memory(
    memory: str,
    *,
    subject_ids: Mapping[str, str],
    subject_order: Sequence[str],
) -> str:
    by_subject: dict[str, list[Fact]] = defaultdict(list)
    for fact in parse_flat_memory(memory):
        if fact.subject != "shared-vehicle" and fact.subject not in subject_ids:
            raise ValueError(f"No speaker_id for memory subject: {fact.subject}")
        by_subject[fact.subject].append(fact)
    blocks = []
    for subject in subject_order:
        facts = by_subject.pop(subject, None)
        if not facts:
            continue
        header = group_header(subject, subject_ids)
        blocks.append("\n".join([header, *(fact.grouped_line for fact in facts)]))
    if by_subject:
        raise ValueError(f"Subjects missing from deterministic order: {sorted(by_subject)}")
    return normalize_memory("\n\n".join(blocks))


def group_header(subject: str, subject_ids: Mapping[str, str]) -> str:
    # speaker_id remains in row metadata for identity tracking. Memory text uses
    # the V1-compatible human-readable header because names are unique within
    # every scenario in this export.
    del subject_ids
    if subject == "shared-vehicle":
        return "### shared-vehicle"
    return f"### {subject}"


def transform_operations(
    previous_flat: str,
    operations: Sequence[Mapping[str, Any]],
    *,
    subject_ids: Mapping[str, str],
    subject_order: Sequence[str],
) -> tuple[dict[str, str], ...]:
    """Translate flat-log exact patches into equivalent grouped-memory patches."""
    flat = normalize_memory(previous_flat)
    grouped = group_memory(flat, subject_ids=subject_ids, subject_order=subject_order)
    transformed = []
    operation_index = 0
    while operation_index < len(operations):
        raw = operations[operation_index]
        op = str(raw["op"])
        target = str(raw["target"])
        content = str(raw["content"])
        before_facts = parse_flat_memory(flat)
        # The source occasionally expresses a timestamp-changing replacement as
        # DELETE(last fact for a speaker) + ADD(same speaker). Applying those
        # literally would temporarily remove the whole speaker block and append
        # it at the end. Collapse the adjacent pair into one block replacement
        # so the stable speaker-group order is preserved.
        if op == "delete" and operation_index + 1 < len(operations):
            old = _single_fact(target)
            following = operations[operation_index + 1]
            same_subject = [value for value in before_facts if value.subject == old.subject]
            if following["op"] == "add" and len(same_subject) == 1:
                new = _single_fact(str(following["content"]))
                if new.subject == old.subject:
                    header = group_header(old.subject, subject_ids)
                    grouped_op = {
                        "op": "replace",
                        "target": f"{header}\n{old.grouped_line}",
                        "content": f"{header}\n{new.grouped_line}",
                    }
                    flat, _ = apply_operations(flat, [raw, following])
                    grouped, _ = apply_operations(grouped, [grouped_op])
                    expected = group_memory(
                        flat, subject_ids=subject_ids, subject_order=subject_order
                    )
                    if grouped != expected:
                        raise ValueError(
                            "Grouped delete/add collapse does not reproduce transformed memory"
                        )
                    transformed.append(grouped_op)
                    operation_index += 2
                    continue
        if op == "add":
            fact = _single_fact(content)
            existing = [value for value in before_facts if value.subject == fact.subject]
            if existing:
                grouped_op = {
                    "op": "add",
                    "target": existing[-1].grouped_line,
                    "content": fact.grouped_line,
                }
            else:
                grouped_op = {
                    "op": "add",
                    "target": "",
                    "content": f"{group_header(fact.subject, subject_ids)}\n{fact.grouped_line}",
                }
        elif op == "replace":
            old = _single_fact(target)
            new = _single_fact(content)
            if old.subject != new.subject:
                raise ValueError("Grouped conversion does not allow replace to move subjects")
            grouped_op = {
                "op": "replace",
                "target": old.grouped_line,
                "content": new.grouped_line,
            }
        elif op == "delete":
            old = _single_fact(target)
            same_subject = [value for value in before_facts if value.subject == old.subject]
            grouped_op = (
                {
                    "op": "delete",
                    "target": (
                        f"{group_header(old.subject, subject_ids)}\n{old.grouped_line}"
                    ),
                    "content": "",
                }
                if len(same_subject) == 1
                else {"op": "delete", "target": old.grouped_line, "content": ""}
            )
        else:
            raise ValueError(f"Unknown flat operation: {op}")
        flat, _ = apply_operations(flat, [raw])
        grouped, _ = apply_operations(grouped, [grouped_op])
        expected = group_memory(flat, subject_ids=subject_ids, subject_order=subject_order)
        if grouped != expected:
            raise ValueError("Grouped operation does not reproduce transformed memory")
        transformed.append(grouped_op)
        operation_index += 1
    return tuple(transformed)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    benchmark_root = args.vehiclemembench_root.expanduser().resolve(strict=True)
    scenarios = tuple(dict.fromkeys(args.scenarios))
    if not scenarios or any(value < 1 or value > 120 for value in scenarios):
        raise ValueError("Scenarios must be unique encoded IDs in 1..120")
    if output.exists():
        if not args.force:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    subject_ids, subject_order = _scan_subjects(source, set(scenarios))
    paths = {name: output / f"{name}.jsonl" for name in ("summary", "patch", "delta")}
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    compaction_path = output / "compaction.jsonl"
    compaction_handle = compaction_path.open("w", encoding="utf-8")
    counts: Counter[str] = Counter()
    memory_by_turn: dict[tuple[int, int], tuple[str, str]] = {}
    try:
        with (source / "summary.jsonl").open(encoding="utf-8") as summaries, (
            source / "patch.jsonl"
        ).open(encoding="utf-8") as patches:
            current_scenario = None
            base_summary = ""
            pending_updates: list[list[list[str]]] = []
            last_row = None
            for line_number, (summary_line, patch_line) in enumerate(
                zip(summaries, patches, strict=True), 1
            ):
                summary = json.loads(summary_line)
                patch = json.loads(patch_line)
                if _identity(summary) != _identity(patch):
                    raise ValueError(f"Summary/Patch mismatch at source line {line_number}")
                scenario = int(summary["scenario_index"])
                if scenario not in scenarios:
                    continue
                if scenario != current_scenario:
                    if current_scenario is not None and pending_updates:
                        _write_compaction(
                            compaction_handle,
                            last_row,
                            base_summary,
                            pending_updates,
                            trigger="FINAL_FLUSH",
                        )
                        counts["final_flushes"] += 1
                    current_scenario = scenario
                    base_summary = ""
                    pending_updates = []
                ids = subject_ids[scenario]
                order = subject_order[scenario]
                previous_flat = patch["input"]["previous_memory"]
                next_flat = summary["target"]["next_memory"]
                previous_grouped = group_memory(
                    previous_flat, subject_ids=ids, subject_order=order
                )
                next_grouped = group_memory(next_flat, subject_ids=ids, subject_order=order)
                source_ops = patch["target"].get("operations", [])
                grouped_ops = transform_operations(
                    previous_flat, source_ops, subject_ids=ids, subject_order=order
                )
                replayed = previous_grouped
                if grouped_ops:
                    replayed, _ = apply_operations(replayed, grouped_ops)
                if replayed != next_grouped:
                    raise ValueError(f"Grouped replay mismatch at source line {line_number}")
                _assert_fact_equivalence(previous_flat, previous_grouped, ids)
                _assert_fact_equivalence(next_flat, next_grouped, ids)

                before_hash = _text_sha256(previous_grouped)
                after_hash = _text_sha256(next_grouped)
                provenance = {
                    **dict(summary.get("provenance") or {}),
                    "source_before_memory_sha256": summary["provenance"]["before_memory_sha256"],
                    "source_after_memory_sha256": summary["provenance"]["after_memory_sha256"],
                    "before_memory_sha256": before_hash,
                    "after_memory_sha256": after_hash,
                    "grouping": "speaker_name_first_appearance",
                }
                common = {
                    key: summary[key]
                    for key in (
                        "scenario_index",
                        "split",
                        "run_id",
                        "global_turn_index",
                        "event_turn_index",
                        "turn_id",
                        "event_id",
                        "timestamp",
                        "current_turn",
                    )
                    if key in summary
                }
                common["run_id"] = f"{common['run_id']}-grouped-v2"
                common["train_eligible"] = bool(summary.get("train_eligible", True))
                target_common = {
                    key: summary["target"][key]
                    for key in ("decision", "reason_code", "reason")
                    if key in summary["target"]
                }
                index = int(summary["global_turn_index"])
                summary_row = {
                    "schema_version": "vehiclemembench-v2-grouped-summary-sft-v2",
                    "sample_id": summary["sample_id"],
                    **common,
                    "input": {"previous_memory": previous_grouped},
                    "target": {**target_common, "next_memory": next_grouped},
                    "provenance": provenance,
                }
                patch_row = {
                    "schema_version": "vehiclemembench-v2-grouped-patch-sft-v2",
                    "sample_id": patch["sample_id"],
                    **common,
                    "input": {"previous_memory": previous_grouped},
                    "target": {**target_common, "operations": list(grouped_ops)},
                    "provenance": provenance,
                }
                delta_row = {
                    "schema_version": "vehiclemembench-v2-grouped-delta-v2-sft-v2",
                    "sample_id": patch["sample_id"].replace(":patch:", ":delta:"),
                    **common,
                    "input": {
                        "base_summary": base_summary,
                        "pending_updates": pending_updates,
                    },
                    "target": {
                        **target_common,
                        "operations": compact_operations(grouped_ops) if grouped_ops else [],
                    },
                    "provenance": provenance,
                }
                for name, row in (
                    ("summary", summary_row),
                    ("patch", patch_row),
                    ("delta", delta_row),
                ):
                    handles[name].write(_json_line(row))
                memory_by_turn[(scenario, index)] = (after_hash, next_grouped)
                counts["rows"] += 1
                counts["updates" if grouped_ops else "no_ops"] += 1
                counts["operations"] += len(grouped_ops)
                counts.update(operation["op"] for operation in grouped_ops)
                if grouped_ops:
                    pending_updates.append(compact_operations(grouped_ops))
                    if len(pending_updates) == 5:
                        _write_compaction(
                            compaction_handle,
                            summary_row,
                            base_summary,
                            pending_updates,
                            trigger="UPDATE_INTERVAL",
                            next_summary=next_grouped,
                        )
                        base_summary = next_grouped
                        pending_updates = []
                        counts["interval_compactions"] += 1
                materialized = _apply_compact_batches(base_summary, pending_updates)
                if materialized != next_grouped:
                    raise ValueError(f"Delta-v2 replay mismatch at source line {line_number}")
                last_row = summary_row
            if current_scenario is not None and pending_updates:
                _write_compaction(
                    compaction_handle,
                    last_row,
                    base_summary,
                    pending_updates,
                    trigger="FINAL_FLUSH",
                )
                counts["final_flushes"] += 1
    finally:
        for handle in handles.values():
            handle.close()
        compaction_handle.close()

    _write_grouped_quiz_sources(source, output, scenarios, memory_by_turn)
    manifest = {
        "schema_version": "vehiclemembench-v2-grouped-memory-export-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "scenarios": list(scenarios),
        "representation": {
            "group_key": "speaker_name",
            "identity_metadata": "speaker_id retained outside rendered memory",
            "group_order": "first fact appearance",
            "fact_order": "source order within speaker",
            "shared_group": "shared-vehicle",
            "delta": "compact tuple Delta-v2 with deterministic 5-UPDATE compaction",
        },
        "counts": dict(counts),
        "files": {
            **{name: _file_record(path) for name, path in paths.items()},
            "compaction": _file_record(compaction_path),
        },
        "subject_maps": {
            str(scenario): subject_ids[scenario] for scenario in scenarios
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    quiz_manifest = {
        "schema_version": "vehiclemembench-v2-grouped-quiz-source-v2",
        "scenarios": list(scenarios),
        "counts": {"turn": len(scenarios) * 30, "final": len(scenarios) * 10},
        "files": {
            "turn_quiz": _file_record(output / "turn_quiz.jsonl"),
            "final_quiz": _file_record(output / "final_quiz.jsonl"),
        },
    }
    (output / "quiz_manifest.json").write_text(
        json.dumps(quiz_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    catalog = output / "catalog.sqlite"
    catalog_result = build_catalog(output, catalog)
    quiz_sft_manifest = build_quiz_sft(
        output,
        benchmark_root,
        catalog_path=catalog,
        expected_scenarios=scenarios,
    )
    result = {
        "output": str(output),
        "scenarios": list(scenarios),
        "counts": dict(counts),
        "catalog": catalog_result,
        "quiz_sft_counts": quiz_sft_manifest["counts"],
    }
    (output / "preparation-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _scan_subjects(
    source: Path, scenarios: set[int]
) -> tuple[dict[int, dict[str, str]], dict[int, list[str]]]:
    subject_ids: dict[int, dict[str, str]] = defaultdict(dict)
    orders: dict[int, list[str]] = defaultdict(list)
    seen: dict[int, set[str]] = defaultdict(set)
    with (source / "summary.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scenario = int(row["scenario_index"])
            if scenario not in scenarios:
                continue
            turn = row["current_turn"]
            name, speaker_id = str(turn["speaker_name"]), str(turn["speaker_id"])
            existing = subject_ids[scenario].get(name)
            if existing is not None and existing != speaker_id:
                raise ValueError(f"Ambiguous speaker name in S{scenario}: {name}")
            subject_ids[scenario][name] = speaker_id
            for fact in parse_flat_memory(row["target"]["next_memory"]):
                if fact.subject not in seen[scenario]:
                    seen[scenario].add(fact.subject)
                    orders[scenario].append(fact.subject)
    for scenario in scenarios:
        if not orders[scenario]:
            raise ValueError(f"No memory subjects found for S{scenario}")
        for subject in orders[scenario]:
            if subject != "shared-vehicle" and subject not in subject_ids[scenario]:
                raise ValueError(f"No speaker mapping for S{scenario}: {subject}")
    return dict(subject_ids), dict(orders)


def _write_grouped_quiz_sources(
    source: Path,
    output: Path,
    scenarios: Sequence[int],
    memory_by_turn: Mapping[tuple[int, int], tuple[str, str]],
) -> None:
    selected = set(scenarios)
    for name in ("turn_quiz.jsonl", "final_quiz.jsonl"):
        with (source / name).open(encoding="utf-8") as src, (output / name).open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                row = json.loads(line)
                scenario = int(row["scenario_index"])
                if scenario not in selected:
                    continue
                ref = row["memory_ref"]
                key = (scenario, int(ref["global_turn_index"]))
                ref["memory_snapshot_sha256"] = memory_by_turn[key][0]
                row.setdefault("provenance", {})["grouped_memory_sha256"] = memory_by_turn[key][0]
                dst.write(_json_line(row))


def _write_compaction(
    handle,
    row: Mapping[str, Any],
    base_summary: str,
    pending_updates: Sequence[Sequence[Sequence[str]]],
    *,
    trigger: str,
    next_summary: str | None = None,
) -> None:
    materialized = _apply_compact_batches(base_summary, pending_updates)
    if next_summary is not None and materialized != next_summary:
        raise ValueError("Compaction target differs from pending-update replay")
    handle.write(
        _json_line(
            {
                "schema_version": "vehiclemembench-v2-grouped-delta-v2-compaction-v2",
                "scenario_index": row["scenario_index"],
                "split": row["split"],
                "global_turn_index": row["global_turn_index"],
                "turn_id": row["turn_id"],
                "trigger": trigger,
                "input": {
                    "base_summary": base_summary,
                    "pending_updates": list(pending_updates),
                },
                "target": {
                    "next_summary": materialized,
                    "next_summary_sha256": _text_sha256(materialized),
                },
            }
        )
    )


def _apply_compact_batches(
    base_summary: str, pending_updates: Sequence[Sequence[Sequence[str]]]
) -> str:
    from .methods.delta_v2 import expand_compact_operations

    memory = normalize_memory(base_summary)
    for batch in pending_updates:
        memory, _ = apply_operations(memory, expand_compact_operations(batch))
    return memory


def _assert_fact_equivalence(
    flat_memory: str, grouped_memory: str, subject_ids: Mapping[str, str]
) -> None:
    expected = Counter(fact.flat_line for fact in parse_flat_memory(flat_memory))
    actual: Counter[str] = Counter()
    subject = None
    reverse_header = {group_header(name, subject_ids): name for name in subject_ids}
    reverse_header["### shared-vehicle"] = "shared-vehicle"
    for line in normalize_memory(grouped_memory).splitlines():
        if not line:
            continue
        if line.startswith("### "):
            subject = reverse_header.get(line)
            if subject is None:
                raise ValueError(f"Unknown grouped header: {line}")
            continue
        if subject is None or not line.startswith("- ["):
            raise ValueError(f"Malformed grouped fact: {line}")
        match = re.fullmatch(r"- \[([^]]+)] (.+)", line)
        if match is None:
            raise ValueError(f"Malformed grouped fact: {line}")
        actual[f"- [{match.group(1)}] {subject}: {match.group(2)}"] += 1
    if actual != expected:
        raise ValueError("Grouped memory does not preserve the flat fact multiset")


def _single_fact(value: str) -> Fact:
    facts = parse_flat_memory(value)
    if len(facts) != 1:
        raise ValueError("V2 grouped conversion requires one fact per operation field")
    return facts[0]


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("scenario_index"),
        row.get("global_turn_index"),
        row.get("turn_id"),
        row.get("target", {}).get("decision"),
        row.get("train_eligible", True),
    )


def _text_sha256(value: str) -> str:
    return hashlib.sha256(normalize_memory(value).encode("utf-8")).hexdigest()


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            lines += 1
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "line_count": lines,
        "bytes": path.stat().st_size,
    }


def main() -> None:
    result = prepare(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
