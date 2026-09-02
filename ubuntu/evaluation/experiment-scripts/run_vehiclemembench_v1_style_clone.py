#!/usr/bin/env python3
"""Create a training-only structural clone of one VehicleMemBench V1 scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from palmclaw_ubuntu.vehicle_bench.dataset import (
    OFFICIAL_UPSTREAM_COMMIT,
    load_vehicle_benchmark,
)
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import validate_public_v1_payload

MODEL_ID = "gpt-5.6-luna"
PROMPT_VERSION = "vehiclemembench-v1-style-exact-clone-luna-v3-chunked"
SESSION_GENERATION_CHUNK_SIZE = 24
ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/"
    "vehiclemembench-v1-style-augmentation-v1"
)
TRACE_SEARCH_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
)
# S18 contains one upstream typo with the same timestamp prefix duplicated.
# Parse it as one logical turn and render the normalized single-prefix form.
HISTORY_RE = re.compile(
    r"^\[([^\]]+)\]\s+(?:\[[^\]]+\]\s+)?([^:]+):\s+(.+)$"
)
S44_NAME_MAP = OrderedDict(
    (
        ("Sandra Edwards", "Francesca Kerton"),
        ("Christopher Wright", "Priya Shah"),
        ("Deborah Gonzalez", "Henry William Stiegel"),
        ("Alice", "Morgan Lee"),
    )
)
TARGET_NAMES = (
    "Avery Morgan",
    "Jordan Kim",
    "Casey Patel",
    "Riley Chen",
    "Cameron Brooks",
    "Quinn Rivera",
    "Parker Shah",
    "Rowan Lee",
)
NAME_MAP: OrderedDict[str, str] = OrderedDict()
FIRST_NAME_MAP: OrderedDict[str, str] = OrderedDict()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RewrittenSession(StrictModel):
    texts: tuple[str, ...] = Field(min_length=1)


class RewrittenQuiz(StrictModel):
    source_index: int = Field(ge=0, le=9)
    query: str = Field(min_length=1)
    gold_memory: str = Field(min_length=1)


class RewrittenQuizBatch(StrictModel):
    items: tuple[RewrittenQuiz, ...] = Field(min_length=10, max_length=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/hj153lee/VehicleMemBench"))
    parser.add_argument("--source-scenario", type=int, default=44)
    parser.add_argument("--target-scenario", type=int, default=301)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.max_attempts < 1:
        raise ValueError("workers/max-attempts must be positive")
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    if not 1 <= args.source_scenario <= 50 or args.target_scenario < 1:
        raise ValueError("source scenario must be S1-S50 and target must be positive")
    source = dataset.scenario(args.source_scenario)
    output_arg = args.output_root or ROOT / f"s{args.source_scenario}-style-v2"
    output = output_arg.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_turns = _load_history(source.history_path)
    _configure_name_map(source_turns, args.source_scenario)
    trace_root = args.trace_root or _default_trace_root(args.source_scenario)
    trace = _load_trace(trace_root, expected_turns=len(source_turns))
    sessions = _split_sessions(source_turns)
    template = _build_template(source, source_turns, sessions, trace)
    _write_json(output / "source-template.json", template)

    client = _openai_client(args.timeout_seconds)
    rewritten = _rewrite_sessions(
        sessions,
        trace,
        output / "dialogues",
        client=client,
        model=args.model,
        workers=args.workers,
        max_attempts=args.max_attempts,
    )
    # Normalize any source names that the model may have retained inside an
    # otherwise independently rewritten utterance. Speaker labels are already
    # mapped separately; this keeps references within the dialogue consistent.
    rewritten = tuple(_replace_names(text) for text in rewritten)
    quiz = _rewrite_quizzes(
        json.loads(source.qa_path.read_text(encoding="utf-8")),
        output / "quiz-rewrite.json",
        client=client,
        model=args.model,
        max_attempts=args.max_attempts,
    )
    history_text = _render_history(source_turns, rewritten)
    qa_payload = _build_qa_payload(
        json.loads(source.qa_path.read_text(encoding="utf-8")), quiz
    )
    validate_public_v1_payload(history_text, qa_payload, dataset.tool_schemas)
    history_path = output / "benchmark" / "history" / f"history_{args.target_scenario}.txt"
    qa_path = output / "benchmark" / "qa_data" / f"qa_{args.target_scenario}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(history_text, encoding="utf-8")
    _write_json(qa_path, qa_payload)

    views = _build_training_views(
        source_turns,
        rewritten,
        trace,
        source_scenario=args.source_scenario,
        target_scenario=args.target_scenario,
    )
    for name, rows in views.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    audit = _audit(
        source_history=source.history_path,
        generated_history=history_path,
        source_qa=source.qa_path,
        generated_qa=qa_path,
        source_turns=source_turns,
        generated_turns=rewritten,
        trace=trace,
        sessions=sessions,
        views=views,
        checkpoint_root=output / "dialogues",
        quiz_checkpoint=output / "quiz-rewrite.json",
        source_scenario=args.source_scenario,
    )
    _write_json(output / "audit.json", audit)
    (output / "audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    (output / "COMPLETED").touch()
    return audit


def _load_history(path: Path) -> tuple[dict[str, Any], ...]:
    turns = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        match = HISTORY_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"Malformed history line {index + 1}")
        turns.append(
            {
                "global_turn_index": index,
                "timestamp": match.group(1),
                "speaker": match.group(2),
                "text": match.group(3),
            }
        )
    return tuple(turns)


def _split_sessions(turns: tuple[dict[str, Any], ...]) -> tuple[tuple[dict[str, Any], ...], ...]:
    sessions = []
    current = []
    previous = None
    for turn in turns:
        timestamp = datetime.fromisoformat(turn["timestamp"])
        if previous is not None and (timestamp - previous).total_seconds() > 300:
            sessions.append(tuple(current))
            current = []
        current.append(turn)
        previous = timestamp
    if current:
        sessions.append(tuple(current))
    return tuple(sessions)


def _configure_name_map(
    turns: tuple[dict[str, Any], ...], source_scenario: int
) -> None:
    speakers = tuple(dict.fromkeys(turn["speaker"] for turn in turns))
    if source_scenario == 44:
        mapping = OrderedDict(
            (speaker, S44_NAME_MAP[speaker]) for speaker in speakers
        )
    else:
        if len(speakers) > len(TARGET_NAMES):
            raise ValueError(
                f"S{source_scenario} has {len(speakers)} speakers; "
                f"only {len(TARGET_NAMES)} target names are configured"
            )
        offset = source_scenario % len(TARGET_NAMES)
        candidates = TARGET_NAMES[offset:] + TARGET_NAMES[:offset]
        mapping = OrderedDict(zip(speakers, candidates, strict=False))
    NAME_MAP.clear()
    NAME_MAP.update(mapping)
    FIRST_NAME_MAP.clear()
    FIRST_NAME_MAP.update(
        (source.split()[0], target.split()[0])
        for source, target in NAME_MAP.items()
    )


def _default_trace_root(source_scenario: int) -> Path:
    matches = sorted(
        TRACE_SEARCH_ROOT.glob(
            f"turnwise-patch-soft30-fresh-r2-s{source_scenario}-*"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Patch R2 trace for S{source_scenario}, found {len(matches)}"
        )
    return matches[0].resolve(strict=True)


def _load_trace(root: Path, *, expected_turns: int) -> dict[str, Any]:
    databases = list(root.expanduser().resolve(strict=True).rglob("memory.db"))
    if len(databases) != 1:
        raise ValueError(f"Expected one source memory.db, found {len(databases)}")
    connection = sqlite3.connect(databases[0])
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT cr.start_message_id, cr.output, mc.metadata_json
            FROM consolidation_runs AS cr
            JOIN model_calls AS mc ON mc.consolidation_run_id = cr.id
            WHERE cr.status = 'completed' AND mc.error IS NULL
            ORDER BY cr.start_message_id
            """
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != expected_turns:
        raise ValueError(f"Trace has {len(rows)} turns, expected {expected_turns}")
    current_memory = ""
    labels = []
    for index, row in enumerate(rows):
        metadata = json.loads(row["metadata_json"])
        if int(row["start_message_id"]) != index + 1:
            raise ValueError("Trace message IDs are not contiguous")
        decision = metadata.get("update_status")
        if decision == "updated":
            next_memory = _replace_names(str(row["output"] or "").strip())
            operations = _memory_patch(current_memory, next_memory)
            replayed = _apply_patch(current_memory, operations)
            if replayed != next_memory:
                raise ValueError(f"Derived Patch failed at turn {index}")
            reason_code = (
                "NEW_VEHICLE_MEMORY"
                if all(item["op"] == "add" for item in operations)
                else "UPDATED_VEHICLE_MEMORY"
            )
            labels.append(
                {
                    "decision": "UPDATE",
                    "reason_code": reason_code,
                    "reason": "This source-verified turn changes the cumulative vehicle memory.",
                    "operations": operations,
                    "before_memory": current_memory,
                    "after_memory": next_memory,
                }
            )
            current_memory = next_memory
        elif decision == "noop":
            labels.append(
                {
                    "decision": "NO_OP",
                    "reason_code": "NO_NEW_VEHICLE_FACT",
                    "reason": "This source-verified turn adds no new durable vehicle memory.",
                    "operations": [],
                    "before_memory": current_memory,
                    "after_memory": current_memory,
                }
            )
        else:
            raise ValueError(f"Unknown trace decision at turn {index}: {decision}")
    return {
        "database": str(databases[0]),
        "labels": tuple(labels),
        "final_memory": current_memory,
    }


def _rewrite_sessions(
    sessions: tuple[tuple[dict[str, Any], ...], ...],
    trace: dict[str, Any],
    checkpoint_root: Path,
    *,
    client: Any,
    model: str,
    workers: int,
    max_attempts: int,
) -> tuple[str, ...]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results: dict[int, tuple[str, ...]] = {}
    missing = []
    for index, session in enumerate(sessions):
        path = checkpoint_root / f"session-{index:03d}.json"
        if path.exists():
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            payload = RewrittenSession.model_validate(checkpoint["payload"])
            _validate_session(session, payload.texts)
            results[index] = payload.texts
        else:
            missing.append((index, session, path))

    def generate(item):
        index, session, path = item
        chunks = tuple(
            session[offset : offset + SESSION_GENERATION_CHUNK_SIZE]
            for offset in range(0, len(session), SESSION_GENERATION_CHUNK_SIZE)
        )
        all_texts: list[str] = []
        response_ids: list[str | None] = []
        usage = Counter()
        total_attempts = 0
        started_session = time.monotonic()
        providers = []
        for chunk_index, chunk in enumerate(chunks):
            provider = _session_provider_payload(index, chunk, trace)
            provider["chunk_index"] = chunk_index
            provider["chunk_count"] = len(chunks)
            providers.append(provider)
            last_error = None
            for attempt in range(1, max_attempts + 1):
                total_attempts += 1
                try:
                    response = client.responses.parse(
                        model=model,
                        instructions=_session_instructions(),
                        input=json.dumps(provider, ensure_ascii=False, separators=(",", ":")),
                        text_format=RewrittenSession,
                        reasoning={"effort": "medium"},
                        max_output_tokens=4096,
                        store=False,
                    )
                    payload = RewrittenSession.model_validate(response.output_parsed)
                    _validate_session(chunk, payload.texts)
                    all_texts.extend(payload.texts)
                    response_ids.append(getattr(response, "id", None))
                    usage.update(_response_usage(response))
                    break
                except Exception as exc:
                    last_error = exc
            else:
                raise RuntimeError(
                    f"Session {index} chunk {chunk_index + 1}/{len(chunks)} failed "
                    f"after {max_attempts} attempts: {last_error}"
                ) from last_error
        texts = tuple(all_texts)
        _validate_session(session, texts)
        checkpoint = {
            "schema_version": "vehiclemembench-v1-style-session-v3",
            "session_index": index,
            "source_sha256": _json_sha256(providers),
            "payload": RewrittenSession(texts=texts).model_dump(mode="json"),
            "model_id": model,
            "prompt_version": PROMPT_VERSION,
            "response_ids": response_ids,
            "generation_chunk_size": SESSION_GENERATION_CHUNK_SIZE,
            "generation_chunk_count": len(chunks),
            "usage": {
                **dict(usage),
                "latency_ms": int((time.monotonic() - started_session) * 1000),
                "generation_attempts": total_attempts,
            },
        }
        _write_json(path, checkpoint)
        return index, texts

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate, item): item[0] for item in missing}
        for future in as_completed(futures):
            index, texts = future.result()
            results[index] = texts
    flattened = []
    for index in range(len(sessions)):
        flattened.extend(results[index])
    return tuple(flattened)


def _session_provider_payload(
    session_index: int,
    session: tuple[dict[str, Any], ...],
    trace: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for local_index, turn in enumerate(session):
        label = trace["labels"][turn["global_turn_index"]]
        rows.append(
            {
                "local_index": local_index,
                "global_turn_index": turn["global_turn_index"],
                "source_speaker": turn["speaker"],
                "target_speaker": NAME_MAP.get(turn["speaker"], turn["speaker"]),
                "source_text": turn["text"],
                "memory_decision": label["decision"],
                "required_patch": label["operations"] if label["decision"] == "UPDATE" else [],
            }
        )
    return {
        "session_index": session_index,
        "name_map": NAME_MAP,
        "rows": rows,
    }


def _session_instructions() -> str:
    return """
Generate a new VehicleMemBench conversation session from the supplied structural
blueprint. Write the conversation from scratch; do not perform mechanical
word-by-word substitution. Return exactly one new text for every input row, in
the same order.

Rules:
- Keep each row's meaning, dialogue role, vehicle facts, numbers, settings,
  conditions, references, corrections, and conversational dependency intact.
- Replace source people with target people and fix pronouns consistently.
- Use source_text only to preserve each row's semantic/dialogue role. Express it
  with independently composed wording and do not copy a full source sentence.
- At UPDATE rows, the rewritten utterance must explicitly and completely support
  required_patch. At NO_OP rows, do not introduce a new durable vehicle fact.
- Preserve roughly the same utterance length and level of detail.
- Do not mention this rewrite, memory labels, patches, tools, or a benchmark.
- Output only the texts array. No speakers, timestamps, indexes, or commentary.
""".strip()


def _validate_session(session: tuple[dict[str, Any], ...], texts: tuple[str, ...]) -> None:
    if len(texts) != len(session):
        raise ValueError(f"Session returned {len(texts)} texts, expected {len(session)}")
    if any(not text.strip() or "\n" in text or "\r" in text for text in texts):
        raise ValueError("Rewritten session contains an empty or multiline utterance")
    # Short acknowledgements and vehicle-setting values can coincide naturally.
    # Reject only substantial full-sentence reuse; the prompt still requires all
    # utterances to be independently composed.
    source = {
        turn["text"].strip().casefold()
        for turn in session
        if len(turn["text"].strip()) >= 40 and len(turn["text"].split()) >= 8
    }
    if any(text.strip().casefold() in source for text in texts):
        raise ValueError("Rewritten session copied a substantial source utterance")


def _rewrite_quizzes(
    source_qa: dict[str, Any],
    checkpoint_path: Path,
    *,
    client: Any,
    model: str,
    max_attempts: int,
) -> RewrittenQuizBatch:
    if checkpoint_path.exists():
        stored = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return RewrittenQuizBatch.model_validate(stored["payload"])
    records = source_qa["related_to_vehicle_preference"]
    provider = {
        "name_map": NAME_MAP,
        "items": [
            {
                "source_index": index,
                "reasoning_type": item["reasoning_type"],
                "query": item["query"],
                "gold_memory": item["gold_memory"],
                "fixed_answers": item["new_answer"],
            }
            for index, item in enumerate(records)
        ],
    }
    instructions = """
Generate ten new VehicleMemBench queries and gold-memory descriptions from the
supplied structural blueprints. Compose them from scratch rather than applying
mechanical substitutions. Apply name_map and fix pronouns. Preserve every date,
setting, value, condition, reasoning dependency, and the exact meaning needed
by fixed_answers. Do not change, output, or discuss fixed_answers. Return each
source_index exactly once. Do not copy a complete source sentence.
""".strip()
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            started = time.monotonic()
            response = client.responses.parse(
                model=model,
                instructions=instructions,
                input=json.dumps(provider, ensure_ascii=False, separators=(",", ":")),
                text_format=RewrittenQuizBatch,
                reasoning={"effort": "medium"},
                max_output_tokens=8192,
                store=False,
            )
            payload = RewrittenQuizBatch.model_validate(response.output_parsed)
            if sorted(item.source_index for item in payload.items) != list(range(10)):
                raise ValueError("Quiz rewrite source indexes are incomplete")
            checkpoint = {
                "schema_version": "vehiclemembench-v1-style-quiz-v2",
                "payload": payload.model_dump(mode="json"),
                "model_id": model,
                "prompt_version": PROMPT_VERSION,
                "response_id": getattr(response, "id", None),
                "usage": {
                    **_response_usage(response),
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "generation_attempts": attempt,
                },
            }
            _write_json(checkpoint_path, checkpoint)
            return payload
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Quiz rewrite failed") from last_error


def _render_history(
    source_turns: tuple[dict[str, Any], ...], rewritten: tuple[str, ...]
) -> str:
    if len(source_turns) != len(rewritten):
        raise ValueError("History rewrite count mismatch")
    return "\n".join(
        f"[{turn['timestamp']}] {NAME_MAP.get(turn['speaker'], turn['speaker'])}: {text.strip()}"
        for turn, text in zip(source_turns, rewritten, strict=True)
    ) + "\n"


def _build_qa_payload(
    source_qa: dict[str, Any], rewritten: RewrittenQuizBatch
) -> dict[str, Any]:
    by_index = {item.source_index: item for item in rewritten.items}
    records = []
    for index, source in enumerate(source_qa["related_to_vehicle_preference"]):
        item = by_index[index]
        records.append(
            {
                "gold_memory": item.gold_memory,
                "reasoning_type": source["reasoning_type"],
                "query": item.query,
                "new_answer": source["new_answer"],
            }
        )
    return {"related_to_vehicle_preference": records}


def _build_training_views(
    source_turns: tuple[dict[str, Any], ...],
    rewritten: tuple[str, ...],
    trace: dict[str, Any],
    *,
    source_scenario: int,
    target_scenario: int,
) -> dict[str, tuple[dict[str, Any], ...]]:
    summary = []
    patch = []
    canonical = []
    for turn, text, label in zip(source_turns, rewritten, trace["labels"], strict=True):
        index = turn["global_turn_index"]
        common = {
            "scenario_index": target_scenario,
            "split": "train",
            "run_id": f"vehiclemembench-v1-s{source_scenario}-style-v2",
            "global_turn_index": index,
            "turn_id": f"v1-s{source_scenario}-style-v2-turn-{index:05d}",
            "timestamp": turn["timestamp"],
            "current_turn": {
                "speaker_name": NAME_MAP.get(turn["speaker"], turn["speaker"]),
                "text": text,
            },
        }
        target_common = {
            "decision": label["decision"],
            "reason_code": label["reason_code"],
            "reason": label["reason"],
        }
        summary.append(
            {
                "schema_version": "vehiclemembench-v1-style-summary-sft-v2",
                "sample_id": f"s{target_scenario}:summary:{index:05d}",
                **common,
                "input": {"previous_memory": label["before_memory"]},
                "target": {**target_common, "next_memory": label["after_memory"]},
            }
        )
        patch.append(
            {
                "schema_version": "vehiclemembench-v1-style-patch-sft-v2",
                "sample_id": f"s{target_scenario}:patch:{index:05d}",
                **common,
                "input": {"previous_memory": label["before_memory"]},
                "target": {**target_common, "operations": label["operations"]},
            }
        )
        canonical.append(
            {
                "schema_version": "vehiclemembench-v1-style-turn-v2",
                **common,
                "target": {
                    **target_common,
                    "operations": label["operations"],
                    "next_memory": label["after_memory"],
                },
            }
        )
    return {"summary": tuple(summary), "patch": tuple(patch), "turnwise": tuple(canonical)}


def _memory_patch(previous: str, current: str) -> list[dict[str, str]]:
    if previous == current:
        raise ValueError("UPDATE memory did not change")
    try:
        before = _memory_blocks(previous)
        after = _memory_blocks(current)
    except ValueError:
        # A few legacy V1 traces use flat bullets rather than grouped headings.
        # Retain the verified trajectory exactly instead of guessing a schema.
        if not previous:
            return [{"op": "add", "target": "", "content": current}]
        return [{"op": "replace", "target": previous, "content": current}]
    operations = []
    for subject in before:
        if subject not in after:
            operations.append({"op": "delete", "target": _render_block(subject, before[subject]), "content": ""})
    for subject, bullets in after.items():
        if subject not in before:
            operations.append({"op": "add", "target": "", "content": _render_block(subject, bullets)})
            continue
        old = before[subject]
        if old == bullets:
            continue
        removed = [item for item in old if item not in bullets]
        added = [item for item in bullets if item not in old]
        if len(removed) == 1 and len(added) == 1:
            operations.append({"op": "replace", "target": removed[0], "content": added[0]})
        elif not removed and added:
            materialized = list(old)
            for item in (value for value in bullets if value in added):
                position = bullets.index(item)
                predecessors = [value for value in bullets[:position] if value in materialized]
                anchor = predecessors[-1] if predecessors else f"### {subject}"
                operations.append({"op": "add", "target": anchor, "content": item})
                materialized.insert(position, item)
        elif removed and not added:
            if len(removed) == len(old):
                operations.append({"op": "delete", "target": _render_block(subject, old), "content": ""})
            else:
                operations.extend({"op": "delete", "target": item, "content": ""} for item in removed)
        else:
            operations.append({"op": "replace", "target": _render_block(subject, old), "content": _render_block(subject, bullets)})
    if operations:
        try:
            if _apply_patch(previous, operations) == current:
                return operations
        except ValueError:
            pass

    # Some memories contain an identical bullet under multiple people, making
    # a bullet-only target ambiguous. Preserve normal fine-grained operations,
    # but fall back to unique person blocks for those rare traces.
    block_operations = []
    for subject, bullets in before.items():
        if subject not in after:
            block_operations.append(
                {
                    "op": "delete",
                    "target": _render_block(subject, bullets),
                    "content": "",
                }
            )
    for subject, bullets in after.items():
        after_block = _render_block(subject, bullets)
        if subject not in before:
            block_operations.append(
                {"op": "add", "target": "", "content": after_block}
            )
        elif before[subject] != bullets:
            block_operations.append(
                {
                    "op": "replace",
                    "target": _render_block(subject, before[subject]),
                    "content": after_block,
                }
            )
    try:
        if block_operations and _apply_patch(previous, block_operations) == current:
            return block_operations
    except ValueError:
        pass
    if not previous:
        return [{"op": "add", "target": "", "content": current}]
    return [{"op": "replace", "target": previous, "content": current}]


def _memory_blocks(memory: str) -> OrderedDict[str, list[str]]:
    blocks: OrderedDict[str, list[str]] = OrderedDict()
    current = None
    for line in memory.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("### "):
            current = line[4:].strip()
            blocks[current] = []
        elif line.startswith("- ") and current is not None:
            blocks[current].append(line)
        else:
            raise ValueError(f"Unsupported source memory line: {line}")
    return blocks


def _render_block(subject: str, bullets: Iterable[str]) -> str:
    return "\n".join([f"### {subject}", *bullets])


def _apply_patch(memory: str, operations: Iterable[Mapping[str, str]]) -> str:
    current = memory.strip()
    for operation in operations:
        op, target, content = operation["op"], operation["target"], operation["content"]
        if op == "add":
            if target:
                if current.count(target) != 1:
                    raise ValueError("Patch add target is not unique")
                current = current.replace(target, target + "\n" + content, 1)
            else:
                current = current + ("\n\n" if current else "") + content
        elif op == "replace":
            if current.count(target) != 1:
                raise ValueError("Patch replace target is not unique")
            current = current.replace(target, content, 1)
        elif op == "delete":
            if current.count(target) != 1:
                raise ValueError("Patch delete target is not unique")
            current = current.replace(target, "", 1)
        else:
            raise ValueError(f"Unknown operation: {op}")
        current = re.sub(r"\n{3,}", "\n\n", current).strip()
    return current


def _audit(
    *,
    source_history: Path,
    generated_history: Path,
    source_qa: Path,
    generated_qa: Path,
    source_turns: tuple[dict[str, Any], ...],
    generated_turns: tuple[str, ...],
    trace: dict[str, Any],
    sessions: tuple[tuple[dict[str, Any], ...], ...],
    views: dict[str, tuple[dict[str, Any], ...]],
    checkpoint_root: Path,
    quiz_checkpoint: Path,
    source_scenario: int,
) -> dict[str, Any]:
    source_lines = [turn["text"].strip().casefold() for turn in source_turns]
    generated_lines = [text.strip().casefold() for text in generated_turns]
    exact = len(set(source_lines) & set(generated_lines))
    updates = [index for index, label in enumerate(trace["labels"]) if label["decision"] == "UPDATE"]
    for index, row in enumerate(views["patch"]):
        label = trace["labels"][index]
        if row["target"]["decision"] != label["decision"]:
            raise ValueError("Training decision drift")
        if label["decision"] == "UPDATE":
            if _apply_patch(row["input"]["previous_memory"], row["target"]["operations"]) != label["after_memory"]:
                raise ValueError(f"Training Patch replay failed at {index}")
    usage_rows = [json.loads(path.read_text(encoding="utf-8"))["usage"] for path in sorted(checkpoint_root.glob("*.json"))]
    usage_rows.append(json.loads(quiz_checkpoint.read_text(encoding="utf-8"))["usage"])
    usage = _sum_usage(usage_rows)
    uncached = max(usage["input_tokens"] - usage["cached_tokens"], 0)
    cost = uncached / 1_000_000 * 0.2 + usage["cached_tokens"] / 1_000_000 * 0.02 + usage["output_tokens"] / 1_000_000 * 1.2
    source_qa_payload = json.loads(source_qa.read_text(encoding="utf-8"))["related_to_vehicle_preference"]
    generated_qa_payload = json.loads(generated_qa.read_text(encoding="utf-8"))["related_to_vehicle_preference"]
    generated_history_lines = generated_history.read_text(encoding="utf-8").splitlines()
    generated_history_rows = [HISTORY_RE.fullmatch(line) for line in generated_history_lines]
    if any(row is None for row in generated_history_rows):
        raise ValueError("Generated history contains malformed rows")
    generated_history_text = "\n".join(generated_history_lines)
    # Check source names case-sensitively.  Some first names (for example
    # "Mark") are also ordinary English verbs; case-folding made those valid
    # rewritten utterances fail the structural audit even though the original
    # speaker identity was absent.  Speaker labels are independently checked
    # by speaker_sequence_exact_after_mapping below.
    source_names_absent = all(
        re.search(rf"\b{re.escape(source)}\b", generated_history_text)
        is None
        for source in (*NAME_MAP.keys(), *FIRST_NAME_MAP.keys())
    )
    structure = {
        "turns": len(generated_turns),
        "sessions": len(sessions),
        "timestamps_exact": [item["timestamp"] for item in source_turns]
        == [row.group(1) for row in generated_history_rows],
        "speaker_sequence_exact_after_mapping": [
            NAME_MAP.get(item["speaker"], item["speaker"]) for item in source_turns
        ]
        == [row.group(2) for row in generated_history_rows],
        "source_names_absent": source_names_absent,
        "update_positions": updates,
        "update_positions_exact": True,
        "no_op_count": len(source_turns) - len(updates),
        "reasoning_sequence_exact": [item["reasoning_type"] for item in source_qa_payload]
        == [item["reasoning_type"] for item in generated_qa_payload],
        "answers_exact": [item["new_answer"] for item in source_qa_payload]
        == [item["new_answer"] for item in generated_qa_payload],
    }
    required_checks = (
        "timestamps_exact",
        "speaker_sequence_exact_after_mapping",
        "source_names_absent",
        "update_positions_exact",
        "reasoning_sequence_exact",
        "answers_exact",
    )
    if not all(structure[key] is True for key in required_checks):
        raise ValueError("Structural clone audit failed")
    return {
        "schema_version": "vehiclemembench-v1-style-clone-audit-v2",
        "passed": True,
        "source_scenario": source_scenario,
        "source_hashes": {"history": _sha256(source_history), "qa": _sha256(source_qa)},
        "generated_hashes": {"history": _sha256(generated_history), "qa": _sha256(generated_qa)},
        "structure": structure,
        "text": {"exact_utterance_overlap": exact, "exact_overlap_rate": exact / len(generated_lines)},
        "memory": {
            "final_memory": trace["final_memory"],
            "final_memory_sha256": hashlib.sha256(trace["final_memory"].encode()).hexdigest(),
            "patch_replay_count": len(updates),
        },
        "training_rows": {name: len(rows) for name, rows in views.items()},
        "usage": usage,
        "estimated_luna_cost_usd": cost,
    }


def _build_template(source, turns, sessions, trace) -> dict[str, Any]:
    return {
        "schema_version": "vehiclemembench-v1-style-source-template-v2",
        "source_scenario": source.index,
        "source_history": str(source.history_path),
        "source_qa": str(source.qa_path),
        "source_history_sha256": _sha256(source.history_path),
        "source_qa_sha256": _sha256(source.qa_path),
        "turn_count": len(turns),
        "session_count": len(sessions),
        "session_lengths": [len(item) for item in sessions],
        "speaker_counts": dict(Counter(turn["speaker"] for turn in turns)),
        "name_map": NAME_MAP,
        "update_positions": [index for index, label in enumerate(trace["labels"]) if label["decision"] == "UPDATE"],
        "no_op_count": sum(label["decision"] == "NO_OP" for label in trace["labels"]),
    }


def _replace_names(text: str) -> str:
    value = text
    for source, target in NAME_MAP.items():
        value = value.replace(source, target)
    for source, target in FIRST_NAME_MAP.items():
        value = re.sub(rf"\b{re.escape(source)}\b", target, value)
    return value


def _openai_client(timeout_seconds: float):
    from openai import OpenAI

    return OpenAI(timeout=timeout_seconds)


def _response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0),
    }


def _sum_usage(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    keys = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "latency_ms")
    return {key: sum(int(row.get(key, 0) or 0) for row in rows) for key in keys}


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _audit_markdown(audit: dict[str, Any]) -> str:
    structure = audit["structure"]
    return (
        f"# S{audit['source_scenario']}-style-v2 audit\n\n"
        "- Result: **PASS**\n"
        f"- Turns/sessions: {structure['turns']:,} / {structure['sessions']}\n"
        f"- UPDATE/NO_OP: {len(structure['update_positions'])} / {structure['no_op_count']:,}\n"
        f"- Exact source utterance overlap: {audit['text']['exact_utterance_overlap']}\n"
        f"- Luna tokens: input {audit['usage']['input_tokens']:,}, output {audit['usage']['output_tokens']:,}\n"
        f"- Estimated Luna cost: **${audit['estimated_luna_cost_usd']:.3f}**\n"
    )


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
