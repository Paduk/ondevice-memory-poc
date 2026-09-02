from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from palmclaw_ubuntu.vehicle_bench.v1_stage3 import V1Stage3Artifact
from palmclaw_ubuntu.vehicle_bench.v2_hybrid import V2HybridArtifact
from palmclaw_ubuntu.vehicle_bench.v2_native_turnwise import (
    V2NativeScenarioArtifact,
)
from palmclaw_ubuntu.vehicle_bench.v2_turn_quiz import V2TurnQuizArtifact

QualityMethod = Literal["post_hoc", "hybrid", "native_turnwise"]
QuizKind = Literal["immediate", "delayed", "composite", "final"]

QUALITY_EVALUATION_SCHEMA_VERSION = "vehiclemembench-v2-quality-input-v1"
READINESS_MANIFEST_VERSION = "vehiclemembench-v2-three-way-readiness-v1"


@dataclass(frozen=True)
class QualityArtifactPaths:
    method: QualityMethod
    scenario_index: int
    root: Path
    memory_artifact: Path
    turn_quiz_artifact: Path
    final_quiz_artifact: Path


@dataclass(frozen=True)
class QualityGoldCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class QualityQuiz:
    schema_version: str
    method: QualityMethod
    scenario_index: int
    quiz_id: str
    quiz_kind: QuizKind
    reasoning_type: str
    query: str
    gold_calls: tuple[QualityGoldCall, ...]
    memory: str
    memory_sha256: str
    source_stage2_sha256: str
    source_quiz_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_quality_artifact_paths(
    evaluation_root: Path,
    scenario_indices: Iterable[int] | None = None,
) -> tuple[QualityArtifactPaths, ...]:
    evaluation_root = evaluation_root.expanduser().resolve()
    hybrid_root = evaluation_root / "vehiclemembench-v2-hybrid"
    posthoc_root = evaluation_root / "vehiclemembench-v2-posthoc"
    native_root = evaluation_root / "vehiclemembench-v2-native"
    selected = tuple(range(1, 6) if scenario_indices is None else scenario_indices)
    if not selected or any(index < 1 or index > 100 for index in selected):
        raise ValueError("quality scenario indexes must be in [1, 100]")
    if len(set(selected)) != len(selected):
        raise ValueError("quality scenario indexes must be unique")

    paths: list[QualityArtifactPaths] = []
    for scenario_index in selected:
        tag = f"{scenario_index:02d}"
        posthoc = posthoc_root / f"posthoc-terra-anchored-s{tag}-r1"
        paths.append(
            QualityArtifactPaths(
                method="post_hoc",
                scenario_index=scenario_index,
                root=posthoc,
                memory_artifact=posthoc / "hybrid.json",
                turn_quiz_artifact=posthoc / "turn-quiz-30/turn-quizzes.json",
                final_quiz_artifact=posthoc / "final-v1/stage3.json",
            )
        )

        if scenario_index == 1:
            hybrid = hybrid_root / "hybrid-full-terra-s1-r2"
        else:
            r2 = hybrid_root / f"hybrid-pilot-terra-s{tag}-r2"
            r1 = hybrid_root / f"hybrid-pilot-terra-s{tag}-r1"
            hybrid = r2 if r2.is_dir() else r1
        paths.append(
            QualityArtifactPaths(
                method="hybrid",
                scenario_index=scenario_index,
                root=hybrid,
                memory_artifact=hybrid / "hybrid.json",
                turn_quiz_artifact=(
                    hybrid / "turn-quiz-30-v1/turn-quizzes.json"
                ),
                final_quiz_artifact=hybrid / "final-v1/stage3.json",
            )
        )

        native = native_root / (
            f"native-turnwise-terra-anchored-s{scenario_index}-r1"
        )
        native_quiz = native / "quiz-30-final-10-v1"
        paths.append(
            QualityArtifactPaths(
                method="native_turnwise",
                scenario_index=scenario_index,
                root=native,
                memory_artifact=native / "native.json",
                turn_quiz_artifact=(
                    native_quiz / "turn-quiz-30/turn-quizzes.json"
                ),
                final_quiz_artifact=native_quiz / "final-v1/stage3.json",
            )
        )
    return tuple(paths)


def load_quality_quizzes(paths: QualityArtifactPaths) -> tuple[QualityQuiz, ...]:
    memory_payload = _validated_memory_payload(paths)
    source_stage2_sha256 = str(memory_payload["source_stage2_sha256"])
    snapshots = _memory_snapshots(memory_payload)
    final_memory = str(memory_payload["final_memory"])
    final_memory_sha256 = _text_sha256(final_memory)
    if final_memory_sha256 != memory_payload["final_memory_sha256"]:
        raise ValueError(f"Invalid final memory hash: {paths.memory_artifact}")
    snapshots[final_memory_sha256] = final_memory

    turn_artifact = V2TurnQuizArtifact.model_validate_json(
        paths.turn_quiz_artifact.read_text(encoding="utf-8")
    )
    if turn_artifact.source_stage2_sha256 != source_stage2_sha256:
        raise ValueError(f"Turn Quiz source mismatch: {paths.root}")
    if turn_artifact.audit.quiz_count != 30 or not turn_artifact.audit.passed:
        raise ValueError(f"Turn Quiz artifact is incomplete: {paths.root}")

    quizzes: list[QualityQuiz] = []
    for quiz in turn_artifact.quizzes:
        memory = snapshots.get(quiz.memory_snapshot_sha256)
        if memory is None:
            raise ValueError(
                f"Missing memory snapshot {quiz.memory_snapshot_sha256}: "
                f"{paths.root}"
            )
        quizzes.append(
            QualityQuiz(
                schema_version=QUALITY_EVALUATION_SCHEMA_VERSION,
                method=paths.method,
                scenario_index=paths.scenario_index,
                quiz_id=quiz.quiz_id,
                quiz_kind=_turn_quiz_kind(quiz.quiz_id),
                reasoning_type=quiz.reasoning_type,
                query=quiz.query,
                gold_calls=tuple(
                    QualityGoldCall(name=call.name, arguments=dict(call.arguments))
                    for call in quiz.gold_calls
                ),
                memory=memory,
                memory_sha256=quiz.memory_snapshot_sha256,
                source_stage2_sha256=source_stage2_sha256,
                source_quiz_sha256=quiz.quiz_sha256,
            )
        )

    final_artifact = V1Stage3Artifact.model_validate_json(
        paths.final_quiz_artifact.read_text(encoding="utf-8")
    )
    if final_artifact.source_stage2_sha256 != source_stage2_sha256:
        raise ValueError(f"Final Quiz source mismatch: {paths.root}")
    if final_artifact.audit.quiz_count != 10 or not final_artifact.audit.passed:
        raise ValueError(f"Final Quiz artifact is incomplete: {paths.root}")
    for quiz in final_artifact.scenario.final_quizzes:
        quizzes.append(
            QualityQuiz(
                schema_version=QUALITY_EVALUATION_SCHEMA_VERSION,
                method=paths.method,
                scenario_index=paths.scenario_index,
                quiz_id=f"final-{quiz.quiz_id}",
                quiz_kind="final",
                reasoning_type=quiz.reasoning_type,
                query=quiz.query,
                gold_calls=tuple(
                    QualityGoldCall(name=call.name, arguments=dict(call.arguments))
                    for call in quiz.gold_calls
                ),
                memory=final_memory,
                memory_sha256=final_memory_sha256,
                source_stage2_sha256=source_stage2_sha256,
                source_quiz_sha256=_canonical_sha256(
                    quiz.model_dump(mode="json")
                ),
            )
        )
    if len(quizzes) != 40:
        raise ValueError(f"Expected 40 quality quizzes: {paths.root}")
    return tuple(quizzes)


def build_readiness_manifest(
    artifacts: tuple[QualityArtifactPaths, ...],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    hashes_by_scenario: dict[int, set[str]] = {}
    for item in artifacts:
        missing = [
            str(path)
            for path in (
                item.memory_artifact,
                item.turn_quiz_artifact,
                item.final_quiz_artifact,
            )
            if not path.is_file()
        ]
        if missing:
            records.append(
                {
                    "method": item.method,
                    "scenario_index": item.scenario_index,
                    "status": "PENDING",
                    "root": str(item.root),
                    "missing_paths": missing,
                }
            )
            continue
        try:
            quizzes = load_quality_quizzes(item)
        except Exception as exc:
            records.append(
                {
                    "method": item.method,
                    "scenario_index": item.scenario_index,
                    "status": "INVALID",
                    "root": str(item.root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        source_hash = quizzes[0].source_stage2_sha256
        hashes_by_scenario.setdefault(item.scenario_index, set()).add(source_hash)
        records.append(
            {
                "method": item.method,
                "scenario_index": item.scenario_index,
                "status": "READY",
                "root": str(item.root),
                "source_stage2_sha256": source_hash,
                "turn_quiz_count": sum(q.quiz_kind != "final" for q in quizzes),
                "final_quiz_count": sum(q.quiz_kind == "final" for q in quizzes),
            }
        )
    mismatches = {
        str(index): sorted(values)
        for index, values in hashes_by_scenario.items()
        if len(values) > 1
    }
    ready = sum(record["status"] == "READY" for record in records)
    pending = sum(record["status"] == "PENDING" for record in records)
    invalid = sum(record["status"] == "INVALID" for record in records)
    return {
        "version": READINESS_MANIFEST_VERSION,
        "status": "INVALID" if mismatches or invalid else "READY_OR_PENDING",
        "artifact_count": len(records),
        "ready_count": ready,
        "pending_count": pending,
        "invalid_count": invalid,
        "source_hash_mismatches": mismatches,
        "artifacts": records,
    }


def _validated_memory_payload(paths: QualityArtifactPaths) -> dict[str, Any]:
    raw = paths.memory_artifact.read_text(encoding="utf-8")
    if paths.method == "native_turnwise":
        artifact = V2NativeScenarioArtifact.model_validate_json(raw)
    else:
        artifact = V2HybridArtifact.model_validate_json(raw)
    if not artifact.audit.completed:
        raise ValueError(f"Memory artifact is incomplete: {paths.memory_artifact}")
    return artifact.model_dump(mode="json")


def _memory_snapshots(payload: dict[str, Any]) -> dict[str, str]:
    snapshots: dict[str, str] = {_text_sha256(""): ""}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            content = value.get("after_memory")
            digest = value.get("after_memory_sha256")
            if isinstance(content, str) and isinstance(digest, str):
                if _text_sha256(content) != digest:
                    raise ValueError(f"Memory snapshot hash mismatch: {digest}")
                previous = snapshots.setdefault(digest, content)
                if previous != content:
                    raise ValueError(f"Memory hash collision: {digest}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return snapshots


def _turn_quiz_kind(quiz_id: str) -> QuizKind:
    if quiz_id.startswith("turn-quiz-delayed-"):
        return "delayed"
    if quiz_id.startswith("turn-quiz-composite-"):
        return "composite"
    return "immediate"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
