"""Causal replay inputs for the V2 Hybrid gold-to-cloud memory gap study."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from palmclaw_ubuntu.contracts import RecursiveSummaryMemoryModel
from palmclaw_ubuntu.vehicle_bench.memory import (
    VehicleHistoryEntry,
    build_daily_history_batches,
    build_turn_history_batches,
)
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import canonical_json_sha256
from palmclaw_ubuntu.vehicle_bench.v1_stage3 import V1Stage3Artifact
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import V2HybridArtifact
from palmclaw_ubuntu.vehicle_bench.v2_quality_evaluation import (
    QualityArtifactPaths,
)
from palmclaw_ubuntu.vehicle_bench.v2_turn_quiz import V2TurnQuizArtifact

ReplayCadence = Literal["calendar_day", "history_entry"]
ReplayQuizKind = Literal["immediate", "delayed", "composite", "final"]
ReplayArm = Literal["turnwise_summary", "turnwise_combined"]
V2_CLOUD_MEMORY_REPLAY_SCHEMA_VERSION = "vehiclemembench-v2-cloud-memory-replay-v1"


@dataclass(frozen=True)
class V2ReplayBatch:
    index: int
    cadence: ReplayCadence
    start_turn_index: int
    end_turn_index: int
    date: str
    content: str
    content_sha256: str


@dataclass(frozen=True)
class V2ReplayQuizCheckpoint:
    quiz_id: str
    quiz_kind: ReplayQuizKind
    cutoff_turn_index: int
    query: str
    reasoning_type: str
    source_quiz_sha256: str


@dataclass(frozen=True)
class V2HybridCloudReplayPlan:
    scenario_index: int
    source_stage2_sha256: str
    dialogue_turn_count: int
    turn_ids: tuple[str, ...]
    daily_batches: tuple[V2ReplayBatch, ...]
    turn_batches: tuple[V2ReplayBatch, ...]
    quiz_checkpoints: tuple[V2ReplayQuizCheckpoint, ...]
    final_turn_index: int

    def batches_through(
        self,
        cadence: ReplayCadence,
        cutoff_turn_index: int,
    ) -> tuple[V2ReplayBatch, ...]:
        """Return only batches fully available at the causal cutoff."""
        if not 0 <= cutoff_turn_index <= self.final_turn_index:
            raise ValueError("Replay cutoff is outside the dialogue")
        source = (
            self.daily_batches if cadence == "calendar_day" else self.turn_batches
        )
        return tuple(
            batch for batch in source if batch.end_turn_index <= cutoff_turn_index
        )


def execute_cloud_memory_replay(
    plan: V2HybridCloudReplayPlan,
    *,
    arm: ReplayArm,
    model: RecursiveSummaryMemoryModel,
    output_dir: Path,
    max_attempts: int = 2,
) -> dict[str, object]:
    """Run or resume one memory trajectory and materialize quiz snapshots."""
    if max_attempts < 1:
        raise ValueError("Memory replay attempts must be positive")
    cadence: ReplayCadence = "history_entry"
    if getattr(model, "update_cadence", None) != cadence:
        raise ValueError(f"{arm} requires model cadence {cadence}")
    batches = plan.daily_batches if cadence == "calendar_day" else plan.turn_batches
    root = output_dir.expanduser().resolve()
    steps_dir = root / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": V2_CLOUD_MEMORY_REPLAY_SCHEMA_VERSION,
        "arm": arm,
        "scenario_index": plan.scenario_index,
        "source_stage2_sha256": plan.source_stage2_sha256,
        "model_id": model.model_id,
        "prompt_version": model.prompt_version,
        "schema_version_model": model.schema_version,
        "instructions_sha256": _text_sha256(
            str(getattr(model, "instructions", ""))
        ),
        "compaction_instructions_sha256": _text_sha256(
            str(getattr(model, "compaction_instructions", ""))
        ),
        "redact_pii": bool(getattr(model, "redact_pii", False)),
        "update_cadence": cadence,
        "batch_hashes": [batch.content_sha256 for batch in batches],
    }
    signature = _canonical_sha256(config)
    config_path = root / "config.json"
    if config_path.is_file():
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        if stored.get("replay_signature") != signature:
            raise ValueError(f"Stale memory replay output: {root}")
    else:
        _write_json(config_path, {**config, "replay_signature": signature})

    current_memory = ""
    add_count_since_compaction = 0
    completed: list[dict[str, object]] = []
    memory_after_cutoff: dict[int, str] = {}
    for batch in batches:
        checkpoint = steps_dir / f"{batch.index:05d}.json"
        before_sha256 = _text_sha256(current_memory)
        step_signature = _canonical_sha256(
            {
                "replay_signature": signature,
                "batch_sha256": batch.content_sha256,
                "before_memory_sha256": before_sha256,
            }
        )
        if checkpoint.is_file():
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            if record.get("step_signature") != step_signature:
                raise ValueError(f"Stale memory replay step: {checkpoint}")
        else:
            setter = getattr(model, "set_compaction_state", None)
            if callable(setter):
                setter(patch_add_count=add_count_since_compaction)
            final_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                started = time.monotonic()
                try:
                    response = model.update(
                        previous_memory=current_memory,
                        date=batch.date,
                        daily_history=batch.content,
                    )
                    latency_ms = round((time.monotonic() - started) * 1_000)
                    status = str(response.metadata.get("update_status", ""))
                    if status not in {"updated", "noop"}:
                        raise RuntimeError("Memory response has invalid update_status")
                    if status == "updated" and not response.content.strip():
                        raise RuntimeError("Memory update returned empty content")
                    if status == "noop" and response.content.strip():
                        raise RuntimeError("Memory no-op returned content")
                    after_memory = (
                        response.content.strip()
                        if status == "updated"
                        else current_memory
                    )
                    record = {
                        "step_signature": step_signature,
                        "batch_index": batch.index,
                        "start_turn_index": batch.start_turn_index,
                        "end_turn_index": batch.end_turn_index,
                        "date": batch.date,
                        "batch_sha256": batch.content_sha256,
                        "before_memory_sha256": before_sha256,
                        "after_memory_sha256": _text_sha256(after_memory),
                        "after_memory": after_memory,
                        "update_status": status,
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                        "usage": response.usage.as_dict(),
                        "response_id": response.response_id,
                        "metadata": dict(response.metadata),
                    }
                    _write_json(checkpoint, record)
                    break
                except Exception as exc:  # provider and schema retry boundary
                    final_error = exc
            else:
                assert final_error is not None
                raise RuntimeError(
                    f"Memory replay failed at batch {batch.index}: {final_error}"
                ) from final_error
        current_memory = str(record["after_memory"])
        add_count_since_compaction = int(
            dict(record.get("metadata", {})).get(
                "patch_add_count_since_compaction",
                add_count_since_compaction,
            )
        )
        completed.append(record)
        memory_after_cutoff[batch.end_turn_index] = current_memory

    snapshots = []
    for quiz in plan.quiz_checkpoints:
        available = plan.batches_through(cadence, quiz.cutoff_turn_index)
        memory = (
            memory_after_cutoff[available[-1].end_turn_index] if available else ""
        )
        snapshots.append(
            {
                "quiz_id": quiz.quiz_id,
                "quiz_kind": quiz.quiz_kind,
                "cutoff_turn_index": quiz.cutoff_turn_index,
                "completed_batch_count": len(available),
                "memory": memory,
                "memory_sha256": _text_sha256(memory),
                "source_quiz_sha256": quiz.source_quiz_sha256,
            }
        )
    artifact: dict[str, object] = {
        **config,
        "replay_signature": signature,
        "status": "COMPLETED",
        "dialogue_turn_count": plan.dialogue_turn_count,
        "batch_count": len(batches),
        "update_count": sum(
            record["update_status"] == "updated" for record in completed
        ),
        "noop_count": sum(record["update_status"] == "noop" for record in completed),
        "input_tokens": sum(
            int(dict(record["usage"]).get("input_tokens", 0))
            for record in completed
        ),
        "output_tokens": sum(
            int(dict(record["usage"]).get("output_tokens", 0))
            for record in completed
        ),
        "cached_tokens": sum(
            int(dict(record["usage"]).get("cached_tokens", 0))
            for record in completed
        ),
        "latency_ms": sum(int(record["latency_ms"]) for record in completed),
        "patch_operation_count": sum(
            int(dict(record.get("metadata", {})).get("patch_operation_count", 0))
            for record in completed
        ),
        "compaction_count": sum(
            bool(dict(record.get("metadata", {})).get("compaction_triggered", False))
            for record in completed
        ),
        "quiz_snapshots": snapshots,
        "final_memory": current_memory,
        "final_memory_sha256": _text_sha256(current_memory),
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    _write_json(root / "memory-replay.json", artifact)
    return artifact


def load_hybrid_cloud_replay_plan(
    paths: QualityArtifactPaths,
    *,
    max_batch_tokens: int = 8_000,
) -> V2HybridCloudReplayPlan:
    """Load and cross-check one frozen Hybrid S1-S5 replay source."""
    if paths.method != "hybrid":
        raise ValueError("Cloud memory gap replay requires a Hybrid artifact")
    stage3 = V1Stage3Artifact.model_validate_json(
        paths.final_quiz_artifact.read_text(encoding="utf-8")
    )
    hybrid = V2HybridArtifact.model_validate_json(
        paths.memory_artifact.read_text(encoding="utf-8")
    )
    turn_quiz = V2TurnQuizArtifact.model_validate_json(
        paths.turn_quiz_artifact.read_text(encoding="utf-8")
    )
    return build_hybrid_cloud_replay_plan(
        scenario_index=paths.scenario_index,
        stage3=stage3,
        hybrid=hybrid,
        turn_quiz=turn_quiz,
        max_batch_tokens=max_batch_tokens,
    )


def build_hybrid_cloud_replay_plan(
    *,
    scenario_index: int,
    stage3: V1Stage3Artifact,
    hybrid: V2HybridArtifact,
    turn_quiz: V2TurnQuizArtifact,
    max_batch_tokens: int = 8_000,
) -> V2HybridCloudReplayPlan:
    if scenario_index < 1:
        raise ValueError("Scenario index must be positive")
    source_hashes = {
        stage3.source_stage2_sha256,
        hybrid.source_stage2_sha256,
        turn_quiz.source_stage2_sha256,
    }
    if len(source_hashes) != 1:
        raise ValueError("Hybrid replay artifacts do not share frozen Stage 2")

    turns = tuple(stage3.scenario.dialogue_turns)
    if not turns:
        raise ValueError("Hybrid replay dialogue is empty")
    labels = tuple(
        label
        for checkpoint in sorted(
            hybrid.event_checkpoints,
            key=lambda item: item.timeline_index,
        )
        for label in checkpoint.turn_labels
    )
    if len(labels) != len(turns):
        raise ValueError("Hybrid labels and dialogue turns have different lengths")
    for index, (turn, label) in enumerate(zip(turns, labels, strict=True)):
        if label.global_turn_index != index:
            raise ValueError("Hybrid global turn indexes are not contiguous")
        if label.turn_id != turn.turn_id:
            raise ValueError(f"Hybrid turn alignment mismatch at index {index}")
        if label.source_event_id != turn.source_event_id:
            raise ValueError(f"Hybrid event alignment mismatch at index {index}")

    entries = _history_entries(turns)
    daily = _replay_batches(
        entries,
        cadence="calendar_day",
        max_batch_tokens=max_batch_tokens,
    )
    turnwise = _replay_batches(
        entries,
        cadence="history_entry",
        max_batch_tokens=max_batch_tokens,
    )
    final_turn_index = len(turns) - 1
    checkpoints: list[V2ReplayQuizCheckpoint] = []
    for quiz in turn_quiz.quizzes:
        if not 0 <= quiz.global_turn_index <= final_turn_index:
            raise ValueError(f"Turn Quiz cutoff is out of range: {quiz.quiz_id}")
        cutoff_label = labels[quiz.global_turn_index]
        expected_cutoff_event = quiz.source_event_id
        delayed = re.search(r"-at-([A-Za-z0-9._-]+)$", quiz.quiz_id)
        if delayed is not None:
            expected_cutoff_event = delayed.group(1)
        if (
            cutoff_label.source_event_id != expected_cutoff_event
            or cutoff_label.event_turn_index != quiz.event_turn_index
        ):
            raise ValueError(f"Turn Quiz causal cutoff mismatch: {quiz.quiz_id}")
        checkpoints.append(
            V2ReplayQuizCheckpoint(
                quiz_id=quiz.quiz_id,
                quiz_kind=_turn_quiz_kind(quiz.quiz_id),
                cutoff_turn_index=quiz.global_turn_index,
                query=quiz.query,
                reasoning_type=quiz.reasoning_type,
                source_quiz_sha256=quiz.quiz_sha256,
            )
        )
    for quiz in stage3.scenario.final_quizzes:
        checkpoints.append(
            V2ReplayQuizCheckpoint(
                quiz_id=f"final-{quiz.quiz_id}",
                quiz_kind="final",
                cutoff_turn_index=final_turn_index,
                query=quiz.query,
                reasoning_type=quiz.reasoning_type,
                source_quiz_sha256=canonical_json_sha256(
                    quiz.model_dump(mode="json")
                ),
            )
        )
    if len(checkpoints) != 40 or len({item.quiz_id for item in checkpoints}) != 40:
        raise ValueError("Hybrid replay requires 30 unique Turn and 10 Final quizzes")
    return V2HybridCloudReplayPlan(
        scenario_index=scenario_index,
        source_stage2_sha256=next(iter(source_hashes)),
        dialogue_turn_count=len(turns),
        turn_ids=tuple(turn.turn_id for turn in turns),
        daily_batches=daily,
        turn_batches=turnwise,
        quiz_checkpoints=tuple(checkpoints),
        final_turn_index=final_turn_index,
    )


def _history_entries(turns: Sequence[object]) -> tuple[VehicleHistoryEntry, ...]:
    entries: list[VehicleHistoryEntry] = []
    previous: datetime | None = None
    for index, turn in enumerate(turns):
        timestamp = datetime.strptime(str(turn.timestamp), "%Y-%m-%dT%H:%M")
        if previous is not None and timestamp < previous:
            raise ValueError("Hybrid dialogue turns are not chronological")
        previous = timestamp
        text = str(turn.text).strip()
        raw = f"[{turn.timestamp.replace('T', ' ')}] {turn.speaker_name}: {text}"
        entries.append(
            VehicleHistoryEntry(
                line_number=index + 1,
                timestamp=timestamp,
                speaker=str(turn.speaker_name),
                content=text,
                raw=raw,
            )
        )
    return tuple(entries)


def _turn_quiz_kind(quiz_id: str) -> ReplayQuizKind:
    if quiz_id.startswith("turn-quiz-delayed-"):
        return "delayed"
    if quiz_id.startswith("turn-quiz-composite-"):
        return "composite"
    return "immediate"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _replay_batches(
    entries: Sequence[VehicleHistoryEntry],
    *,
    cadence: ReplayCadence,
    max_batch_tokens: int,
) -> tuple[V2ReplayBatch, ...]:
    built = (
        build_daily_history_batches(entries, max_tokens=max_batch_tokens)
        if cadence == "calendar_day"
        else build_turn_history_batches(entries, max_tokens=max_batch_tokens)
    )
    offset = 0
    batches: list[V2ReplayBatch] = []
    for batch in built:
        start = offset
        end = start + batch.line_count - 1
        batches.append(
            V2ReplayBatch(
                index=batch.index,
                cadence=cadence,
                start_turn_index=start,
                end_turn_index=end,
                date=batch.start_date,
                content=batch.content,
                content_sha256=hashlib.sha256(batch.content.encode()).hexdigest(),
            )
        )
        offset = end + 1
    if offset != len(entries):
        raise ValueError("Replay batches do not cover every dialogue turn")
    return tuple(batches)
