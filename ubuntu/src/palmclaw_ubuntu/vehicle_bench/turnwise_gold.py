from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from palmclaw_ubuntu.providers import apply_recursive_summary_patch
from palmclaw_ubuntu.vehicle_bench.critic_trace import CriticSourceTurn
from palmclaw_ubuntu.vehicle_bench.dataset import VehicleScenario, VehicleTask
from palmclaw_ubuntu.vehicle_bench.memory import VehicleHistoryEntry

TURNWISE_GOLD_VERSION = "vehiclemembench-v2-turnwise-memory-gold-v1"

AlignmentStatus = Literal["MATCHED_EXACT", "REVIEW_REQUIRED", "UNMATCHED"]

_GOLD_DATE = re.compile(r"^\[([A-Za-z]+ \d{1,2}, \d{4})\]\s*(.*)$")
_GOLD_TIME = re.compile(r"\bAt (\d{1,2}:\d{2}\s*[AP]M)\b", re.IGNORECASE)
_LATER_EVENT = re.compile(r"(?<=[.!?])\s+(?=At \d{1,2}:\d{2}\s*[AP]M\b)")
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "for",
        "he",
        "her",
        "him",
        "his",
        "i",
        "in",
        "it",
        "of",
        "on",
        "she",
        "the",
        "their",
        "them",
        "they",
        "to",
        "was",
        "with",
    }
)


@dataclass(frozen=True)
class VehicleOracleEvent:
    event_id: str
    scenario_index: int
    date: str
    time: str | None
    person: str | None
    fact_summary: str
    evidence_hints: tuple[str, ...]
    task_ids: tuple[str, ...]
    reasoning_types: tuple[str, ...]
    gold_tool_names: tuple[str, ...]
    gold_tool_arguments: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_tool_arguments"] = [
            dict(arguments) for arguments in self.gold_tool_arguments
        ]
        return payload


@dataclass(frozen=True)
class VehicleOracleAlignment:
    event_id: str
    status: AlignmentStatus
    turn_index: int | None
    line_number: int | None
    message_id: int | None
    evidence_quote: str | None
    score: float
    match_basis: str
    candidate_decisions: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_decisions"] = dict(self.candidate_decisions)
        return payload


def build_vehicle_oracle_ledger(
    scenario: VehicleScenario,
    history_entries: Sequence[VehicleHistoryEntry],
) -> tuple[VehicleOracleEvent, ...]:
    """Convert official QA gold memories into deduplicated atomic events."""

    speakers = tuple(dict.fromkeys(entry.speaker for entry in history_entries))
    events: dict[str, VehicleOracleEvent] = {}
    for task in scenario.tasks:
        for line in task.gold_memory.splitlines():
            if not line.strip():
                continue
            for event in _events_from_gold_line(
                scenario.index,
                task,
                line.strip(),
                speakers=speakers,
            ):
                key = _event_dedup_key(event)
                previous = events.get(key)
                if previous is None:
                    events[key] = event
                    continue
                events[key] = replace(
                    previous,
                    task_ids=_merge_unique(previous.task_ids, event.task_ids),
                    reasoning_types=_merge_unique(
                        previous.reasoning_types,
                        event.reasoning_types,
                    ),
                    gold_tool_names=_merge_unique(
                        previous.gold_tool_names,
                        event.gold_tool_names,
                    ),
                    gold_tool_arguments=tuple(
                        [*previous.gold_tool_arguments, *event.gold_tool_arguments]
                    ),
                    evidence_hints=_merge_unique(
                        previous.evidence_hints,
                        event.evidence_hints,
                    ),
                )
    return tuple(
        sorted(
            events.values(),
            key=lambda event: (
                event.date,
                event.time or "",
                event.event_id,
            ),
        )
    )


def align_vehicle_oracle_events(
    events: Sequence[VehicleOracleEvent],
    history_entries: Sequence[VehicleHistoryEntry],
    *,
    trace_turns: Mapping[str, Sequence[CriticSourceTurn]] | None = None,
) -> tuple[VehicleOracleAlignment, ...]:
    """Align gold events without silently accepting semantic-only matches."""

    ordered = tuple(
        sorted(history_entries, key=lambda entry: (entry.timestamp, entry.line_number))
    )
    trace_map = trace_turns or {}
    for method, turns in trace_map.items():
        if len(turns) != len(ordered):
            raise ValueError(
                f"Trace/history turn mismatch for {method}: "
                f"{len(turns)} != {len(ordered)}"
            )
    alignments: list[VehicleOracleAlignment] = []
    for event in events:
        ranked = sorted(
            (
                (
                    _alignment_score(
                        event,
                        entry,
                        candidate_update_votes=sum(
                            turns[turn_index].original_decision == "UPDATE"
                            for turns in trace_map.values()
                        ),
                    ),
                    turn_index,
                    entry,
                )
                for turn_index, entry in enumerate(ordered)
                if entry.date == event.date
            ),
            key=lambda item: (-item[0][0], item[1]),
        )
        if not ranked or ranked[0][0][0] < 25.0:
            alignments.append(
                VehicleOracleAlignment(
                    event_id=event.event_id,
                    status="UNMATCHED",
                    turn_index=None,
                    line_number=None,
                    message_id=None,
                    evidence_quote=None,
                    score=ranked[0][0][0] if ranked else 0.0,
                    match_basis=ranked[0][0][1] if ranked else "no_same_date_turn",
                    candidate_decisions={},
                )
            )
            continue
        (score, basis, evidence), turn_index, entry = ranked[0]
        runner_up = ranked[1][0][0] if len(ranked) > 1 else 0.0
        exact = basis.startswith("exact_quote")
        status: AlignmentStatus = (
            "MATCHED_EXACT"
            if exact and score - runner_up >= 5.0
            else "REVIEW_REQUIRED"
        )
        decisions = {
            method: turns[turn_index].original_decision
            for method, turns in trace_map.items()
        }
        message_ids = {
            turns[turn_index].message_id for turns in trace_map.values()
        }
        message_id = next(iter(message_ids)) if len(message_ids) == 1 else None
        alignments.append(
            VehicleOracleAlignment(
                event_id=event.event_id,
                status=status,
                turn_index=turn_index,
                line_number=entry.line_number,
                message_id=message_id,
                evidence_quote=evidence or entry.content,
                score=round(score, 3),
                match_basis=basis,
                candidate_decisions=decisions,
            )
        )
    return tuple(alignments)


def build_turnwise_gold_smoke_audit(
    *,
    scenario: VehicleScenario,
    history_entries: Sequence[VehicleHistoryEntry],
    events: Sequence[VehicleOracleEvent],
    alignments: Sequence[VehicleOracleAlignment],
) -> dict[str, Any]:
    if {item.event_id for item in events} != {
        item.event_id for item in alignments
    }:
        raise ValueError("Oracle event/alignment coverage mismatch")
    status_counts = {
        status: sum(item.status == status for item in alignments)
        for status in ("MATCHED_EXACT", "REVIEW_REQUIRED", "UNMATCHED")
    }
    aligned_turns = {
        item.turn_index for item in alignments if item.turn_index is not None
    }
    duplicate_turns = len(alignments) - len(aligned_turns) - status_counts["UNMATCHED"]
    return {
        "version": TURNWISE_GOLD_VERSION,
        "scenario_index": scenario.index,
        "history_turns": len(history_entries),
        "tasks": len(scenario.tasks),
        "oracle_events": len(events),
        "alignment_status": status_counts,
        "aligned_unique_turns": len(aligned_turns),
        "duplicate_event_turn_alignments": duplicate_turns,
        "ready_for_canonical_replay": (
            status_counts["REVIEW_REQUIRED"] == 0
            and status_counts["UNMATCHED"] == 0
            and duplicate_turns == 0
        ),
    }


def apply_reviewed_alignment_overrides(
    alignments: Sequence[VehicleOracleAlignment],
    overrides: Mapping[str, int],
    history_entries: Sequence[VehicleHistoryEntry],
    *,
    trace_turns: Mapping[str, Sequence[CriticSourceTurn]] | None = None,
) -> tuple[VehicleOracleAlignment, ...]:
    """Apply explicit reviewed event-to-turn decisions; never infer overrides."""

    ordered = tuple(
        sorted(history_entries, key=lambda entry: (entry.timestamp, entry.line_number))
    )
    known = {item.event_id for item in alignments}
    if unknown := set(overrides) - known:
        raise ValueError(
            f"Alignment overrides reference unknown events: {sorted(unknown)}"
        )
    trace_map = trace_turns or {}
    resolved = []
    for alignment in alignments:
        turn_index = overrides.get(alignment.event_id)
        if turn_index is None:
            resolved.append(alignment)
            continue
        if not 0 <= turn_index < len(ordered):
            raise ValueError(
                f"Alignment override turn is out of range: {alignment.event_id}"
            )
        entry = ordered[turn_index]
        message_ids = {
            turns[turn_index].message_id for turns in trace_map.values()
        }
        message_id = next(iter(message_ids)) if len(message_ids) == 1 else None
        resolved.append(
            VehicleOracleAlignment(
                event_id=alignment.event_id,
                status="MATCHED_EXACT",
                turn_index=turn_index,
                line_number=entry.line_number,
                message_id=message_id,
                evidence_quote=entry.content,
                score=alignment.score,
                match_basis=f"reviewed_override:{alignment.match_basis}",
                candidate_decisions={
                    method: turns[turn_index].original_decision
                    for method, turns in trace_map.items()
                },
            )
        )
    return tuple(resolved)


def build_turnwise_canonical_labels(
    *,
    method: Literal["summary", "combined"],
    history_entries: Sequence[VehicleHistoryEntry],
    trace_turns: Sequence[CriticSourceTurn],
    events: Sequence[VehicleOracleEvent],
    alignments: Sequence[VehicleOracleAlignment],
) -> tuple[dict[str, Any], ...]:
    """Replay approved oracle events into deterministic Summary/Patch labels."""

    ordered = tuple(
        sorted(history_entries, key=lambda entry: (entry.timestamp, entry.line_number))
    )
    if len(ordered) != len(trace_turns):
        raise ValueError("Canonical label trace/history length mismatch")
    event_by_id = {event.event_id: event for event in events}
    events_by_turn: dict[int, list[VehicleOracleEvent]] = defaultdict(list)
    for alignment in alignments:
        if alignment.status != "MATCHED_EXACT" or alignment.turn_index is None:
            raise ValueError(
                "Canonical replay requires every oracle event to be reviewed "
                "and matched"
            )
        events_by_turn[alignment.turn_index].append(event_by_id[alignment.event_id])

    memory = ""
    labels = []
    for turn_index, (entry, trace_turn) in enumerate(
        zip(ordered, trace_turns, strict=True)
    ):
        turn_events = sorted(
            events_by_turn.get(turn_index, ()), key=lambda x: x.event_id
        )
        before_memory = memory
        operations = []
        evidence = []
        if turn_events:
            for event in turn_events:
                content = _canonical_event_line(event)
                operation = {"op": "add", "target": "", "content": content}
                memory, _ = apply_recursive_summary_patch(memory, [operation])
                operations.append(operation)
                evidence.append(
                    {
                        "message_id": trace_turn.message_id,
                        "quote": entry.content,
                    }
                )
            decision = "UPDATE"
            scope = "VEHICLE"
            reason_code = "NEW_OR_UPDATED_VEHICLE_MEMORY"
            reason = "This turn is aligned to a verified vehicle-memory event."
        else:
            decision = "NO_OP"
            scope = "NON_VEHICLE_OR_NO_NEW_FACT"
            reason_code = "NO_NEW_VEHICLE_FACT"
            reason = (
                "This turn introduces no new vehicle fact verified by the "
                "scenario oracle."
            )
        candidate_assessment = (
            "ACCEPT" if trace_turn.original_decision == decision else "CORRECT"
        )
        label = {
            "version": TURNWISE_GOLD_VERSION,
            "scenario": trace_turn.scenario_index,
            "method": method,
            "turn_index": turn_index,
            "message_id": trace_turn.message_id,
            "timestamp": entry.timestamp.isoformat(),
            "speaker": entry.speaker,
            "scope": scope,
            "decision": decision,
            "reason_code": reason_code,
            "reason": reason,
            "evidence": evidence,
            "oracle_event_ids": [event.event_id for event in turn_events],
            "candidate_decision": trace_turn.original_decision,
            "candidate_assessment": candidate_assessment,
            "correction_type": (
                "NONE" if candidate_assessment == "ACCEPT" else "DIRECT_SEMANTIC"
            ),
            "before_memory_sha256": _text_sha256(before_memory),
            "after_memory_sha256": _text_sha256(memory),
            "after_memory": memory,
        }
        if method == "summary":
            label["next_memory"] = memory if decision == "UPDATE" else None
        else:
            label["operations"] = operations
        labels.append(label)
    return tuple(labels)


def write_turnwise_gold_smoke_artifacts(
    output_dir: Path | str,
    *,
    scenario: VehicleScenario,
    history_entries: Sequence[VehicleHistoryEntry],
    events: Sequence[VehicleOracleEvent],
    alignments: Sequence[VehicleOracleAlignment],
    audit: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    summary_labels: Sequence[Mapping[str, Any]] | None = None,
    combined_labels: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": TURNWISE_GOLD_VERSION,
        "scenario_index": scenario.index,
        "history_path": str(scenario.history_path),
        "qa_path": str(scenario.qa_path),
        "history_sha256": _file_sha256(scenario.history_path),
        "qa_sha256": _file_sha256(scenario.qa_path),
        "source": dict(source_metadata),
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(
        root / "oracle_ledger.json",
        {
            "version": TURNWISE_GOLD_VERSION,
            "scenario_index": scenario.index,
            "events": [event.as_dict() for event in events],
        },
    )
    _write_jsonl(
        root / "turn_alignment.jsonl",
        (
            {
                **alignment.as_dict(),
                "oracle_event": event.as_dict(),
            }
            for event, alignment in zip(events, alignments, strict=True)
        ),
    )
    _write_jsonl(
        root / "review_queue.jsonl",
        (
            {
                **alignment.as_dict(),
                "oracle_event": event.as_dict(),
                "causal_context": _causal_context(
                    history_entries,
                    alignment.turn_index,
                ),
            }
            for event, alignment in zip(events, alignments, strict=True)
            if alignment.status != "MATCHED_EXACT"
        ),
    )
    _write_json(root / "audit.json", dict(audit))
    if summary_labels is not None:
        _write_jsonl(root / "summary_labels.jsonl", summary_labels)
    if combined_labels is not None:
        _write_jsonl(root / "combined_labels.jsonl", combined_labels)
    return root


def _events_from_gold_line(
    scenario_index: int,
    task: VehicleTask,
    line: str,
    *,
    speakers: Sequence[str],
) -> tuple[VehicleOracleEvent, ...]:
    match = _GOLD_DATE.match(line)
    if match is None:
        raise ValueError(f"Gold memory line lacks a date: {line}")
    date = datetime.strptime(match.group(1), "%B %d, %Y").date().isoformat()
    body = match.group(2).strip()
    segments = tuple(part.strip() for part in _LATER_EVENT.split(body) if part.strip())
    result = []
    for atomic_index, segment in enumerate(segments):
        time_match = _GOLD_TIME.search(segment)
        time = (
            datetime.strptime(time_match.group(1).upper(), "%I:%M %p").time().strftime(
                "%H:%M"
            )
            if time_match is not None
            else None
        )
        evidence = _extract_balanced_quotes(segment)
        person = _first_named_speaker(segment, speakers)
        digest = hashlib.sha256(
            f"{scenario_index}|{date}|{time}|{_normalize_text(segment)}".encode()
        ).hexdigest()[:12]
        result.append(
            VehicleOracleEvent(
                event_id=(
                    f"s{scenario_index:02d}-{task.event_index:02d}-"
                    f"{atomic_index:02d}-{digest}"
                ),
                scenario_index=scenario_index,
                date=date,
                time=time,
                person=person,
                fact_summary=segment,
                evidence_hints=evidence,
                task_ids=(task.id,),
                reasoning_types=(task.reasoning_type,),
                gold_tool_names=tuple(call.name for call in task.gold_calls),
                gold_tool_arguments=tuple(call.arguments for call in task.gold_calls),
            )
        )
    return tuple(result)


def _alignment_score(
    event: VehicleOracleEvent,
    entry: VehicleHistoryEntry,
    *,
    candidate_update_votes: int,
) -> tuple[float, str, str | None]:
    normalized_content = _normalize_text(entry.content)
    exact_evidence = next(
        (
            hint
            for hint in event.evidence_hints
            if _normalize_text(hint) in normalized_content
        ),
        None,
    )
    score = 0.0
    basis = []
    if exact_evidence is not None:
        score += 100.0
        basis.append("exact_quote")
    if event.time is not None:
        event_minutes = _clock_minutes(event.time)
        entry_minutes = entry.timestamp.hour * 60 + entry.timestamp.minute
        distance = abs(event_minutes - entry_minutes)
        if distance == 0:
            score += 14.0
            basis.append("same_time")
        elif distance <= 5:
            score += 10.0
            basis.append(f"near_time={distance}m")
    if event.person is not None and entry.speaker == event.person:
        score += 12.0
        basis.append("same_speaker")
    comparable = " ".join(event.evidence_hints) or event.fact_summary
    token_score = _token_similarity(comparable, entry.content)
    sequence_score = SequenceMatcher(
        None,
        _normalize_text(comparable),
        normalized_content,
    ).ratio()
    similarity = max(token_score, sequence_score)
    score += 45.0 * similarity
    basis.append(f"similarity={similarity:.3f}")
    tool_overlap = _tool_overlap(event.gold_tool_names, entry.content)
    if tool_overlap:
        score += 15.0 * tool_overlap
        basis.append(f"tool_overlap={tool_overlap:.3f}")
    if candidate_update_votes:
        score += 6.0 * candidate_update_votes
        basis.append(f"candidate_updates={candidate_update_votes}")
    return score, "+".join(basis), exact_evidence


def _extract_balanced_quotes(value: str) -> tuple[str, ...]:
    """Extract single/double quoted speech without splitting contractions."""

    found: list[str] = []
    quote: str | None = None
    start = 0
    for index, char in enumerate(value):
        if char not in {"'", '"', "‘", "’", "“", "”"}:
            continue
        canonical = "'" if char in {"'", "‘", "’"} else '"'
        previous = value[index - 1] if index else " "
        following = value[index + 1] if index + 1 < len(value) else " "
        if quote is None:
            if previous.isalnum() and following.isalnum():
                continue
            if following.isspace():
                continue
            quote = canonical
            start = index + 1
            continue
        if canonical != quote:
            continue
        if previous.isalnum() and following.isalnum():
            continue
        text = value[start:index].strip()
        if text:
            found.append(text)
        quote = None
    return tuple(dict.fromkeys(found))


def _first_named_speaker(value: str, speakers: Sequence[str]) -> str | None:
    matches = []
    for speaker in speakers:
        aliases = (speaker, speaker.split()[0], speaker.split()[-1])
        positions = [value.find(alias) for alias in aliases if value.find(alias) >= 0]
        if positions:
            matches.append((min(positions), speaker))
    return min(matches)[1] if matches else None


def _event_dedup_key(event: VehicleOracleEvent) -> str:
    return "|".join(
        (
            event.date,
            event.time or "",
            event.person or "",
            _normalize_text(event.fact_summary),
        )
    )


def _merge_unique(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*left, *right)))


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tool_overlap(tool_names: Sequence[str], content: str) -> float:
    tool_tokens = {
        token
        for name in tool_names
        for token in _tokens(_split_identifier(name.removeprefix("carcontrol_")))
    }
    content_tokens = set(_tokens(content))
    if not tool_tokens or not content_tokens:
        return 0.0
    return len(tool_tokens & content_tokens) / min(len(tool_tokens), 3)


def _split_identifier(value: str) -> str:
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    return value.replace("_", " ")


def _clock_minutes(value: str) -> int:
    hour, minute = value.split(":", 1)
    return int(hour) * 60 + int(minute)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _WORD.findall(_normalize_text(value))
        if token not in _STOPWORDS
    )


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = (
        normalized.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return " ".join(normalized.casefold().split())


def _causal_context(
    entries: Sequence[VehicleHistoryEntry],
    turn_index: int | None,
    *,
    window: int = 3,
) -> list[dict[str, Any]]:
    if turn_index is None:
        return []
    ordered = tuple(
        sorted(entries, key=lambda item: (item.timestamp, item.line_number))
    )
    start = max(0, turn_index - window)
    return [
        {
            "turn_index": index,
            "line_number": ordered[index].line_number,
            "timestamp": ordered[index].timestamp.isoformat(),
            "speaker": ordered[index].speaker,
            "content": ordered[index].content,
        }
        for index in range(start, turn_index + 1)
    ]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_event_line(event: VehicleOracleEvent) -> str:
    timestamp = f"{event.date} {event.time}" if event.time else event.date
    return f"- [{timestamp}] {event.fact_summary}"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]] | Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
