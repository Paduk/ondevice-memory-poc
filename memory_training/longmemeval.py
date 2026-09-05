"""Deterministic LongMemEval evidence-to-Patch conversion utilities.

The conversion never calls an LLM.  It uses the official question metadata for
compact facts and the official oracle ``has_answer`` markers for source turns.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .methods.operations import apply_operations, normalize_memory

SCHEMA_VERSION = "palmclaw-longmemeval-patch-sft-v2"
TASK_TYPES = frozenset({"knowledge-update", "temporal-reasoning"})
METADATA_TYPES = {
    "knowledge-update": frozenset({"knowledge_update"}),
    "temporal-reasoning": frozenset(
        {"temp_reasoning_explicit", "temp_reasoning_implicit"}
    ),
}
DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"
TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class EvidenceEvent:
    date: str
    session_id: str
    text: str


def load_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"Expected an array of objects: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_memalpha_split(
    oracle: Sequence[Mapping[str, Any]], train_ids: Iterable[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the public Mem-alpha 200/300 split restricted to KU/TR."""
    train_set = set(train_ids)
    official_ids = {str(row["question_id"]) for row in oracle}
    unknown = sorted(train_set - official_ids)
    if unknown:
        raise ValueError(f"Split contains {len(unknown)} unknown question IDs")
    selected = [dict(row) for row in oracle if row.get("question_type") in TASK_TYPES]
    development = [row for row in selected if str(row["question_id"]) in train_set]
    test = [row for row in selected if str(row["question_id"]) not in train_set]
    if len(development) != 91 or len(test) != 120:
        raise ValueError(
            f"Unexpected KU/TR split sizes: development={len(development)}, test={len(test)}"
        )
    return development, test


def split_development(
    rows: Sequence[dict[str, Any]], *, validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratify non-abstention development questions; retain abstentions as audit-only."""
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    usable = [row for row in rows if not str(row["question_id"]).endswith("_abs")]
    abstention = [row for row in rows if str(row["question_id"]).endswith("_abs")]
    generator = random.Random(seed)
    training: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for task_type in sorted(TASK_TYPES):
        group = [row for row in usable if row["question_type"] == task_type]
        group.sort(key=lambda row: str(row["question_id"]))
        generator.shuffle(group)
        validation_count = max(1, round(len(group) * validation_fraction))
        validation.extend(group[:validation_count])
        training.extend(group[validation_count:])
    training.sort(key=lambda row: str(row["question_id"]))
    validation.sort(key=lambda row: str(row["question_id"]))
    abstention.sort(key=lambda row: str(row["question_id"]))
    return training, validation, abstention


def evidence_events(row: Mapping[str, Any]) -> list[EvidenceEvent]:
    sessions = row.get("haystack_sessions")
    dates = row.get("haystack_dates")
    session_ids = row.get("haystack_session_ids")
    if not isinstance(sessions, list) or not isinstance(dates, list):
        raise TypeError("LongMemEval oracle row has invalid sessions/dates")
    if not isinstance(session_ids, list) or not (
        len(sessions) == len(dates) == len(session_ids)
    ):
        raise ValueError("LongMemEval oracle session arrays are not aligned")
    ordered = sorted(
        zip(dates, session_ids, sessions, strict=True),
        key=lambda value: (_parse_date(str(value[0])), str(value[1])),
    )
    result = []
    for date, session_id, messages in ordered:
        if not isinstance(messages, list):
            raise TypeError("LongMemEval session must be a list")
        for message in messages:
            if (
                isinstance(message, Mapping)
                and message.get("role") == "user"
                and message.get("has_answer") is True
                and isinstance(message.get("content"), str)
            ):
                result.append(
                    EvidenceEvent(str(date), str(session_id), _one_line(message["content"]))
                )
    return result


def build_question_rows(
    oracle_row: Mapping[str, Any],
    metadata_row: Mapping[str, Any],
    *,
    split: str,
    scenario_index: int,
    question_ordinal: int,
    include_noop: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build replay-exact Patch SFT rows for one official question."""
    question_id = str(oracle_row["question_id"])
    if metadata_row.get("question_id") != question_id:
        raise ValueError(f"Metadata mismatch for {question_id}")
    task_type = str(oracle_row["question_type"])
    allowed_types = METADATA_TYPES.get(task_type, frozenset())
    if metadata_row.get("question_type") not in allowed_types:
        raise ValueError(f"Metadata task mismatch for {question_id}")
    events = evidence_events(oracle_row)
    if not events and not include_noop:
        raise ValueError(f"No positive user evidence for {question_id}")
    facts, fact_sources = (
        _facts_for_events(task_type, events, metadata_row) if events else ([], [])
    )
    if len(facts) != len(events):
        raise AssertionError("Every evidence event must have one target fact block")

    rows = []
    memory = ""
    previous_block = ""
    operation_counts: Counter[str] = Counter()
    positive_targets = iter(zip(events, facts, fact_sources, strict=True))
    current_positive = next(positive_targets, None)
    user_turns = _ordered_user_turns(oracle_row) if include_noop else [
        (event.date, event.session_id, event.text, True) for event in events
    ]
    for event_index, (date, session_id, text, has_answer) in enumerate(user_turns):
        operations: list[dict[str, str]] = []
        fact_source = "official_has_answer_false"
        if has_answer:
            if current_positive is None:
                raise AssertionError("More positive turns than target facts")
            event, fact, fact_source = current_positive
            if (date, session_id, text) != (event.date, event.session_id, event.text):
                raise AssertionError("Positive turn ordering does not match evidence events")
            block = f"- [observed {event.date}] {fact}"
            if task_type == "knowledge-update" and previous_block:
                operations = [{"op": "replace", "target": previous_block, "content": block}]
            else:
                operations = [{"op": "add", "target": "", "content": block}]
            next_memory, _ = apply_operations(memory, operations)
            operation_counts.update(operation["op"] for operation in operations)
            previous_block = block
            current_positive = next(positive_targets, None)
            decision = "UPDATE"
        else:
            next_memory = memory
            operation_counts["no_op"] += 1
            decision = "NO_OP"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"lme:{question_id}:patch:{event_index:02d}",
                "scenario_index": scenario_index,
                "split": split,
                "global_turn_index": question_ordinal * 100 + event_index,
                "event_turn_index": event_index,
                "turn_id": f"lme:{question_id}:{session_id}:{event_index}",
                "timestamp": date,
                "current_turn": {
                    "speaker_id": "user",
                    "speaker_name": "User",
                    "text": text,
                },
                "input": {"previous_memory": memory},
                "target": {
                    "decision": decision,
                    "operations": operations,
                    "next_memory": next_memory,
                },
                "train_eligible": True,
                "provenance": {
                    "dataset": "LongMemEval",
                    "question_id": question_id,
                    "question_type": task_type,
                    "session_id": session_id,
                    "has_answer": has_answer,
                    "label_source": (
                        "official_metadata_and_has_answer"
                        if has_answer
                        else "official_has_answer_false_as_no_op"
                    ),
                    "cloud_labeling": False,
                    "target_fact_source": fact_source,
                },
            }
        )
        memory = next_memory
    if current_positive is not None:
        raise AssertionError("Fewer positive turns than target facts")
    source_counts = Counter(f"fact_source:{value}" for value in fact_sources)
    return rows, {
        "fact_fallbacks": source_counts["fact_source:evidence_utterance_fallback"],
        **operation_counts,
        **source_counts,
    }


def _ordered_user_turns(
    row: Mapping[str, Any],
) -> list[tuple[str, str, str, bool]]:
    sessions = row.get("haystack_sessions")
    dates = row.get("haystack_dates")
    session_ids = row.get("haystack_session_ids")
    if not isinstance(sessions, list) or not isinstance(dates, list):
        raise TypeError("LongMemEval oracle row has invalid sessions/dates")
    if not isinstance(session_ids, list) or not (
        len(sessions) == len(dates) == len(session_ids)
    ):
        raise ValueError("LongMemEval oracle session arrays are not aligned")
    ordered = sorted(
        zip(dates, session_ids, sessions, strict=True),
        key=lambda value: (_parse_date(str(value[0])), str(value[1])),
    )
    result = []
    for date, session_id, messages in ordered:
        for message in messages:
            if isinstance(message, Mapping) and message.get("role") == "user":
                content = message.get("content")
                if not isinstance(content, str):
                    raise TypeError("LongMemEval user turn has non-string content")
                result.append(
                    (
                        str(date),
                        str(session_id),
                        _one_line(content),
                        message.get("has_answer") is True,
                    )
                )
    return result


def aligned_views(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Create catalog-aligned views; only ``patch``/``general_patch`` are intended."""
    patch = dict(row)
    summary = dict(row)
    summary["target"] = {
        "decision": row["target"]["decision"],
        "next_memory": row["target"]["next_memory"],
    }
    delta = dict(row)
    delta["input"] = {
        "base_summary": row["input"]["previous_memory"],
        "pending_deltas": [],
        "updates_since_compaction": 0,
    }
    delta["target"] = {
        "decision": row["target"]["decision"],
        "operations": row["target"]["operations"],
    }
    return {"summary": summary, "patch": patch, "delta": delta}


def dataset_statistics(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    question_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    audit_abstentions: Sequence[Mapping[str, Any]],
    conversion_counts: Mapping[str, int],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "questions": {},
        "rows": {},
        "operations": dict(conversion_counts),
        "audit_only_abstention_questions": len(audit_abstentions),
    }
    split_sets = {}
    for split, questions in question_records.items():
        ids = {str(row["question_id"]) for row in questions}
        split_sets[split] = ids
        result["questions"][split] = {
            "total": len(ids),
            "by_task": dict(sorted(Counter(row["question_type"] for row in questions).items())),
        }
    overlaps = {
        f"{left}_{right}": len(split_sets[left] & split_sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    result["question_overlap"] = overlaps
    for split, rows in rows_by_split.items():
        source_lengths = [len(str(row["current_turn"]["text"])) for row in rows]
        memory_lengths = [len(str(row["target"]["next_memory"])) for row in rows]
        result["rows"][split] = {
            "total": len(rows),
            "by_task": dict(
                sorted(Counter(row["provenance"]["question_type"] for row in rows).items())
            ),
            "source_chars_mean": round(sum(source_lengths) / len(source_lengths), 1) if rows else 0,
            "source_chars_max": max(source_lengths, default=0),
            "next_memory_chars_mean": round(sum(memory_lengths) / len(memory_lengths), 1) if rows else 0,
            "next_memory_chars_max": max(memory_lengths, default=0),
        }
    return result


def _facts_for_events(
    task_type: str,
    events: Sequence[EvidenceEvent],
    metadata_row: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    content = metadata_row.get("question_content")
    if not isinstance(content, Mapping):
        raise TypeError("LongMemEval metadata has invalid question_content")
    if task_type == "knowledge-update":
        updated = content.get("updated_fact")
        updated_text = _one_line(updated) if isinstance(updated, str) else ""
        facts = []
        sources = []
        for index, event in enumerate(events):
            if index == len(events) - 1:
                if updated_text:
                    facts.append(updated_text)
                    sources.append("official_updated_fact")
                else:
                    # Six official KU records encode updated_fact as a compact
                    # object rather than prose.  The marked evidence utterance
                    # is the deterministic, non-synthetic fallback.
                    facts.append(event.text)
                    sources.append("evidence_utterance_fallback")
                continue
            old_answer = str(content.get("old_answer", ""))
            extracted = _sentence_containing(event.text, old_answer)
            if extracted is None:
                extracted = event.text
                sources.append("evidence_utterance_fallback")
            else:
                sources.append("old_answer_sentence")
            facts.append(extracted)
        return facts, sources

    raw_facts = content.get("facts")
    if isinstance(raw_facts, Mapping):
        raw_facts = list(raw_facts.values())
    if not isinstance(raw_facts, list) or not raw_facts:
        raise ValueError("temporal-reasoning metadata has no facts")
    facts = [_temporal_fact_text(value) for value in raw_facts]
    assignments: dict[int, list[str]] = defaultdict(list)
    for fact in facts:
        best = max(
            range(len(events)),
            key=lambda index: (_overlap(fact, events[index].text), -index),
        )
        assignments[best].append(fact)
    result = []
    sources = []
    for index, event in enumerate(events):
        assigned = assignments.get(index)
        if assigned:
            result.append(" ".join(assigned))
            sources.append("official_temporal_fact")
        else:
            result.append(event.text)
            sources.append("evidence_utterance_fallback")
    return result, sources


def _sentence_containing(text: str, answer: str) -> str | None:
    variants = [answer]
    variants.extend(re.findall(r"(?:or|/|;)\s*([^;)]+)", answer, flags=re.IGNORECASE))
    variants.extend(re.findall(r"\((?:or\s+)?([^)]+)\)", answer, flags=re.IGNORECASE))
    needles = [set(TOKEN_RE.findall(value.lower())) for value in variants]
    needles = [tokens for tokens in needles if tokens]
    sentences = SENTENCE_RE.split(text)
    scored = [
        (max((_coverage(tokens, sentence) for tokens in needles), default=0.0), sentence)
        for sentence in sentences
    ]
    score, sentence = max(scored, default=(0.0, ""), key=lambda value: value[0])
    return _one_line(sentence) if score >= 0.6 else None


def _temporal_fact_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return _one_line(str(value))
    event = _one_line(str(value.get("event", "")))
    date = _one_line(str(value.get("date", "")))
    if date and event:
        return f"On {date}, {event}"
    return event


def _coverage(tokens: set[str], text: str) -> float:
    present = set(TOKEN_RE.findall(text.lower()))
    return len(tokens & present) / len(tokens)


def _overlap(left: str, right: str) -> float:
    a = set(TOKEN_RE.findall(left.lower()))
    b = set(TOKEN_RE.findall(right.lower()))
    return len(a & b) / math.sqrt(len(a) * len(b)) if a and b else 0.0


def _one_line(value: str) -> str:
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(f"Unsupported LongMemEval date: {value}") from exc


def validate_replay(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if row["target"]["decision"] == "NO_OP":
            if row["target"]["operations"]:
                raise AssertionError(f"NO_OP has operations: {row['sample_id']}")
            if normalize_memory(str(row["input"]["previous_memory"])) != normalize_memory(
                str(row["target"]["next_memory"])
            ):
                raise AssertionError(f"NO_OP changed memory: {row['sample_id']}")
            continue
        actual, _ = apply_operations(
            str(row["input"]["previous_memory"]), row["target"]["operations"]
        )
        expected = normalize_memory(str(row["target"]["next_memory"]))
        if actual != expected:
            raise AssertionError(f"Patch replay mismatch: {row['sample_id']}")
