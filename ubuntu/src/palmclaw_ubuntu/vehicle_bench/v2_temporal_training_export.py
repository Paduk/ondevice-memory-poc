"""Build a lossless deterministic Temporal-Patch view from Hybrid V2 Patch."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memory_training.methods.operations import apply_operations, normalize_memory
from memory_training.methods.temporal_patch import validate_temporal_operations

from palmclaw_ubuntu.tokens import TokenCounter
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import text_sha256

V2_TEMPORAL_PATCH_VIEW_VERSION = "vehiclemembench-v2-temporal-patch-sft-v1"
V2_TEMPORAL_EXPORT_VERSION = "vehiclemembench-v2-temporal-patch-export-v1"
V2_TEMPORAL_CLASSIFIER_VERSION = "structured-gold-conservative-v1"

_TEMPORARY_CONDITION = re.compile(
    r"\b(?:temporar(?:y|ily)|for now|until|for the next|during recovery|"
    r"while recover(?:ing|y))\b",
    re.IGNORECASE,
)
_TEMPORARY_CUES = (
    re.compile(r"\btemporarily\b", re.IGNORECASE),
    re.compile(r"\btemporary\b", re.IGNORECASE),
    re.compile(r"\bfor now\b", re.IGNORECASE),
    re.compile(r"\buntil\s+[^,.;!?]+", re.IGNORECASE),
    re.compile(
        r"\bfor the next\s+(?:\w+\s+){0,3}"
        r"(?:hours?|days?|weeks?|months?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bduring recovery\b", re.IGNORECASE),
    re.compile(r"\bwhile recover(?:ing|y)\b", re.IGNORECASE),
)
_END_CUES = (
    re.compile(r"\bno longer\b", re.IGNORECASE),
    re.compile(r"\b(?:has|is) ended\b", re.IGNORECASE),
    re.compile(r"\bis over\b", re.IGNORECASE),
    re.compile(r"\bback to (?:normal|the baseline)\b", re.IGNORECASE),
    re.compile(r"\b(?:fully )?recovered\b", re.IGNORECASE),
)
_SLUG_COMPONENT = re.compile(r"[^a-z0-9]+")


def temporalize_patch_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Add conservative temporal metadata without changing Patch semantics."""
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise TypeError("Patch row target must be an object")
    decision = target.get("decision")
    operations = target.get("operations")
    if not isinstance(operations, (list, tuple)):
        raise TypeError("Patch row operations must be an array")
    converted = []
    if decision == "NO_OP":
        if operations:
            raise ValueError("NO_OP Patch row cannot contain operations")
    elif decision == "UPDATE":
        if not operations:
            raise ValueError("UPDATE Patch row requires operations")
        turn = row.get("current_turn")
        if not isinstance(turn, Mapping) or not isinstance(turn.get("text"), str):
            raise TypeError("Patch row requires current_turn.text")
        source_text = str(turn["text"])
        for operation in operations:
            if not isinstance(operation, Mapping):
                raise TypeError("Patch operation must be an object")
            converted.append(
                _temporalize_operation(operation, source_text=source_text)
            )
        prepared = validate_temporal_operations(converted)
        for operation in prepared:
            if operation["temporal_cue"] not in source_text:
                raise ValueError("Temporal cue is not an exact current-turn substring")
        converted = list(prepared)
    else:
        raise ValueError(f"Unsupported Patch decision: {decision}")

    result = dict(row)
    result["schema_version"] = V2_TEMPORAL_PATCH_VIEW_VERSION
    result["sample_id"] = str(row.get("sample_id", "")).replace(
        ":patch:", ":temporal_patch:"
    )
    result["target"] = {**dict(target), "operations": converted}
    result["provenance"] = {
        **dict(row.get("provenance", {})),
        "temporal_classifier_version": V2_TEMPORAL_CLASSIFIER_VERSION,
        "semantic_source": "structured_gold_patch",
    }
    return result


def _temporalize_operation(
    operation: Mapping[str, Any], *, source_text: str
) -> dict[str, str]:
    if set(operation) != {"op", "target", "content"}:
        raise ValueError("Source Patch operation has invalid fields")
    if any(not isinstance(operation[key], str) for key in operation):
        raise TypeError("Source Patch operation fields must be strings")
    op = str(operation["op"])
    line = str(operation["target"] if op == "delete" else operation["content"])
    condition = _memory_condition(line)
    identity_key = temporal_identity_key(line)
    cue = ""
    if op == "delete":
        if condition and _TEMPORARY_CONDITION.search(condition):
            cue = _first_cue(source_text, _END_CUES)
        action = "end_temporary" if cue else "non_temporal"
    else:
        if op == "add" and condition and _TEMPORARY_CONDITION.search(condition):
            cue = _first_cue(source_text, _TEMPORARY_CUES)
        if cue:
            action = "temporary_override"
        elif condition:
            action = "conditional_upsert"
        else:
            action = "durable_upsert"
    return {
        "op": op,
        "target": str(operation["target"]),
        "content": str(operation["content"]),
        "identity_key": identity_key,
        "temporal_action": action,
        "temporal_cue": cue,
    }


def temporal_identity_key(memory_line: str) -> str:
    """Return a learnable stable subject/setting/context identity."""
    normalized = normalize_memory(memory_line)
    if not normalized.startswith("- [") or "] " not in normalized:
        raise ValueError("Hybrid memory line has no timestamp prefix")
    body = normalized.split("] ", 1)[1]
    if ": " not in body or ";" not in body:
        raise ValueError("Hybrid memory line has no subject or attribute")
    subject, payload = body.split(": ", 1)
    attribute = payload.split(";", 1)[0].strip()
    context_match = re.search(r"(?:^|; )context=\((.*?)\)(?:;|$)", payload)
    components = [_slug(subject), _slug(attribute)]
    if context_match:
        components.append(_slug(context_match.group(1)))
    identity = ".".join(component for component in components if component)
    if not identity:
        raise ValueError("Hybrid memory line produced an empty identity")
    if len(identity) > 160:
        suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
        identity = f"{identity[:143].rstrip('._')}.{suffix}"
    return identity


def _memory_condition(memory_line: str) -> str:
    marker = "; condition="
    if marker not in memory_line:
        return ""
    return memory_line.split(marker, 1)[1].strip()


def _first_cue(source: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    matches = [match for pattern in patterns if (match := pattern.search(source))]
    if not matches:
        return ""
    selected = min(matches, key=lambda match: (match.start(), len(match.group(0))))
    return selected.group(0).strip()


def _slug(value: str) -> str:
    return _SLUG_COMPONENT.sub("_", value.lower()).strip("_")


def export_temporal_patch_dataset(
    *, source_root: Path, output_root: Path
) -> dict[str, Any]:
    """Stream, replay, and atomically export an alternate Temporal dataset."""
    source_root = source_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(
        (source_root / "manifest.json").read_text(encoding="utf-8")
    )
    source_patch = source_root / "patch.jsonl"
    destination = output_root / "patch.jsonl"
    temporary = output_root / ".patch.jsonl.tmp"
    try:
        token_counter = TokenCounter("o200k_base")
        token_counter_encoding = "o200k_base"
    except ValueError:
        # The training environment intentionally retains tiktoken 0.3.x for its
        # legacy genai dependency. Token counts affect export diagnostics only;
        # Temporal operations and replayed memory are unchanged by this fallback.
        token_counter = TokenCounter("cl100k_base")
        token_counter_encoding = "cl100k_base"
    counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    scenario_rows: list[dict[str, Any]] = []
    current_scenario: int | None = None
    current_memory = ""
    scenario_adds = 0
    scenario_max_tokens = 0
    scenario_max_chars = 0
    scenario_turns = 0
    scenario_updates = 0
    scenario_operations = 0
    scenario_compaction_triggers = 0
    adds_since_compaction = 0
    previous_tokens = 0
    scenario_actions: Counter[str] = Counter()

    def finish_scenario() -> None:
        if current_scenario is None:
            return
        scenario_rows.append(
            {
                "scenario_index": current_scenario,
                "turn_count": scenario_turns,
                "update_count": scenario_updates,
                "operation_count": scenario_operations,
                "add_count": scenario_adds,
                "max_memory_tokens": scenario_max_tokens,
                "max_memory_characters": scenario_max_chars,
                "compaction_trigger_count": scenario_compaction_triggers,
                "temporal_action_counts": dict(sorted(scenario_actions.items())),
                "final_memory_sha256": text_sha256(current_memory),
            }
        )

    try:
        with (
            source_patch.open(encoding="utf-8") as source_handle,
            temporary.open("w", encoding="utf-8") as output_handle,
        ):
            for line_number, line in enumerate(source_handle, start=1):
                row = json.loads(line)
                scenario = int(row["scenario_index"])
                if current_scenario != scenario:
                    finish_scenario()
                    current_scenario = scenario
                    current_memory = ""
                    scenario_adds = 0
                    scenario_max_tokens = 0
                    scenario_max_chars = 0
                    scenario_turns = 0
                    scenario_updates = 0
                    scenario_operations = 0
                    scenario_compaction_triggers = 0
                    adds_since_compaction = 0
                    previous_tokens = 0
                    scenario_actions = Counter()
                previous = normalize_memory(row["input"]["previous_memory"])
                if previous != current_memory:
                    raise ValueError(
                        f"Temporal replay before-memory mismatch at line {line_number}"
                    )
                converted = temporalize_patch_row(row)
                operations = converted["target"]["operations"]
                if converted["target"]["decision"] == "UPDATE":
                    stripped = [
                        {key: operation[key] for key in ("op", "target", "content")}
                        for operation in operations
                    ]
                    current_memory, stats = apply_operations(current_memory, stripped)
                    scenario_updates += 1
                    scenario_operations += stats.operation_count
                    scenario_adds += stats.add_count
                    adds_since_compaction += stats.add_count
                    for operation in operations:
                        action = operation["temporal_action"]
                        action_counts[action] += 1
                        scenario_actions[action] += 1
                expected_hash = str(converted["provenance"]["after_memory_sha256"])
                if text_sha256(current_memory) != expected_hash:
                    raise ValueError(
                        f"Temporal replay after-memory mismatch at line {line_number}"
                    )
                if converted["target"]["decision"] == "UPDATE":
                    current_tokens = token_counter.count(current_memory)
                    scenario_max_tokens = max(scenario_max_tokens, current_tokens)
                    scenario_max_chars = max(scenario_max_chars, len(current_memory))
                    if (
                        adds_since_compaction >= 64
                        or previous_tokens < 1_000 <= current_tokens
                        or len(current_memory) > 8_192
                    ):
                        scenario_compaction_triggers += 1
                        adds_since_compaction = 0
                    previous_tokens = current_tokens
                output_handle.write(
                    json.dumps(converted, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                scenario_turns += 1
                counts["turns"] += 1
                counts[converted["target"]["decision"].lower()] += 1
        finish_scenario()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    _link_supporting_files(source_root=source_root, output_root=output_root)
    source_files = dict(source_manifest["files"])
    source_files["patch"] = _file_record(destination)
    compaction_trigger_count = sum(
        int(row["compaction_trigger_count"]) for row in scenario_rows
    )
    manifest = {
        **source_manifest,
        "schema_version": V2_TEMPORAL_EXPORT_VERSION,
        "source_dataset_root": str(source_root),
        "source_patch_sha256": _sha256(source_patch),
        "temporal_classifier_version": V2_TEMPORAL_CLASSIFIER_VERSION,
        "counts": {
            **dict(counts),
            "operations": sum(action_counts.values()),
            "temporal_actions": dict(sorted(action_counts.items())),
        },
        "compaction": {
            "enabled": False,
            "reason": (
                "Temporal-only export; original V1 soft-30 compaction is not "
                "materialized"
            ),
            "add_threshold": 64,
            "token_threshold": 1000,
            "target_ratio": 0.70,
            "trigger_count": compaction_trigger_count,
            "token_count_encoding": token_counter_encoding,
        },
        "files": source_files,
        "scenarios": scenario_rows,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _link_supporting_files(*, source_root: Path, output_root: Path) -> None:
    for name in (
        "summary.jsonl",
        "delta.jsonl",
        "compaction.jsonl",
        "quiz_manifest.json",
        "turn_quiz.jsonl",
        "final_quiz.jsonl",
        "quiz_sft.jsonl",
        "quiz_sft_manifest.json",
        "vehicle_tools.json",
    ):
        source = source_root / name
        if not source.exists():
            continue
        destination = output_root / name
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(
                    f"Temporal dataset has conflicting file: {destination}"
                )
            continue
        destination.symlink_to(source)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    lines = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            lines += 1
    return {
        "path": str(path),
        "line_count": lines,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
