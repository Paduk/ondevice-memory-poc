#!/usr/bin/env python3
"""Run/resume field-grounded answerability audits for ready V2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.v2_quality_evaluation import (
    QualityArtifactPaths,
    QualityQuiz,
    default_quality_artifact_paths,
    load_quality_quizzes,
)
from palmclaw_ubuntu.vehicle_bench.v2_quality_judge import (
    ANSWERABILITY_PROMPT_VERSION,
    ANSWERABILITY_SCHEMA_VERSION,
    OpenAIV2AnswerabilityJudge,
)

DEFAULT_EVALUATION_ROOT = Path("/mnt/data/hj153lee/PalmClaw/evaluation")
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_EVALUATION_ROOT / "vehiclemembench-v2-three-way-evaluation"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("post_hoc", "hybrid", "native_turnwise"),
    )
    parser.add_argument("--scenarios", nargs="+", type=int)
    parser.add_argument(
        "--quiz-limit-per-artifact",
        type=int,
        help="Limit quizzes per method/scenario for a smoke run.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.workers, args.batch_size, args.max_attempts) < 1:
        raise ValueError("workers, batch size, and attempts must be positive")
    output_root = args.output_root.expanduser().resolve() / "answerability"
    output_root.mkdir(parents=True, exist_ok=True)
    judge = OpenAIV2AnswerabilityJudge(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    ready: list[tuple[QualityArtifactPaths, tuple[QualityQuiz, ...]]] = []
    for paths in default_quality_artifact_paths(args.evaluation_root):
        if args.methods and paths.method not in args.methods:
            continue
        if args.scenarios and paths.scenario_index not in args.scenarios:
            continue
        required = (
            paths.memory_artifact,
            paths.turn_quiz_artifact,
            paths.final_quiz_artifact,
        )
        if all(path.is_file() for path in required):
            quizzes = load_quality_quizzes(paths)
            if args.quiz_limit_per_artifact is not None:
                if args.quiz_limit_per_artifact < 1:
                    raise ValueError("quiz limit must be positive")
                quizzes = quizzes[: args.quiz_limit_per_artifact]
            ready.append((paths, quizzes))
    if not ready:
        raise ValueError("No complete V2 artifacts are ready for answerability")

    work = []
    reports: list[dict[str, Any]] = []
    for paths, quizzes in ready:
        by_memory: dict[str, list[QualityQuiz]] = defaultdict(list)
        for quiz in quizzes:
            by_memory[quiz.memory_sha256].append(quiz)
        for memory_sha256, grouped in sorted(by_memory.items()):
            for offset in range(0, len(grouped), args.batch_size):
                batch = tuple(grouped[offset : offset + args.batch_size])
                chunk = offset // args.batch_size + 1
                checkpoint = (
                    output_root
                    / paths.method
                    / f"s{paths.scenario_index:02d}"
                    / f"{memory_sha256[:16]}-c{chunk:02d}.json"
                )
                signature = _signature(batch, args.model)
                if checkpoint.is_file():
                    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if stored.get("evaluation_signature") != signature:
                        raise ValueError(
                            f"Stale answerability checkpoint: {checkpoint}"
                        )
                    reports.append(stored)
                else:
                    work.append((paths, batch, checkpoint, signature))

    def audit(item):
        paths, batch, checkpoint, signature = item
        last_error = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                report = judge.audit(batch)
                report.update(
                    {
                        "evaluation_signature": signature,
                        "method": paths.method,
                        "scenario_index": paths.scenario_index,
                        "attempt": attempt,
                        "quiz_metadata": {
                            quiz.quiz_id: {
                                "quiz_kind": quiz.quiz_kind,
                                "reasoning_type": quiz.reasoning_type,
                                "source_quiz_sha256": quiz.source_quiz_sha256,
                            }
                            for quiz in batch
                        },
                    }
                )
                _write_json(checkpoint, report)
                return report
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    failures = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(work) or 1)) as pool:
        futures = {pool.submit(audit, item): item for item in work}
        for future in as_completed(futures):
            paths, batch, _, _ = futures[future]
            try:
                reports.append(future.result())
            except Exception as exc:
                failures.append(
                    f"{paths.method}/S{paths.scenario_index}/"
                    f"{batch[0].memory_sha256[:12]}: {type(exc).__name__}: {exc}"
                )
    summary = _aggregate(reports, failures=failures, model=args.model)
    _write_json(output_root / "answerability-summary.json", summary)
    if failures:
        raise RuntimeError(f"Answerability failures: {'; '.join(failures)}")
    return summary


def _aggregate(
    reports: list[dict[str, Any]],
    *,
    failures: list[str],
    model: str,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        metadata = report["quiz_metadata"]
        for result in report["results"]:
            enriched = {**result, **metadata[result["quiz_id"]]}
            groups[(report["method"], "all")].append(enriched)
            groups[(report["method"], enriched["quiz_kind"])].append(enriched)
    metrics = {}
    for (method, kind), selected in sorted(groups.items()):
        counts = Counter(item["verdict"] for item in selected)
        metrics[f"{method}/{kind}"] = {
            "quizzes": len(selected),
            "verdicts": dict(sorted(counts.items())),
            "full_support_rate": counts["FULL_SUPPORT"] / len(selected),
            "mean_support_coverage": sum(
                float(item["support_coverage"]) for item in selected
            )
            / len(selected),
        }
    usage = {
        key: sum(int(report.get("usage", {}).get(key, 0)) for report in reports)
        for key in ("input_tokens", "cached_tokens", "output_tokens", "total_tokens")
    }
    return {
        "schema_version": ANSWERABILITY_SCHEMA_VERSION,
        "prompt_version": ANSWERABILITY_PROMPT_VERSION,
        "status": "COMPLETED" if not failures else "PARTIAL",
        "model": model,
        "batch_count": len(reports),
        "failure_count": len(failures),
        "failures": failures,
        "usage": usage,
        "metrics": metrics,
    }


def _signature(quizzes: tuple[QualityQuiz, ...], model: str) -> str:
    body = {
        "schema_version": ANSWERABILITY_SCHEMA_VERSION,
        "prompt_version": ANSWERABILITY_PROMPT_VERSION,
        "model": model,
        "memory_sha256": quizzes[0].memory_sha256,
        "quiz_sha256s": [quiz.source_quiz_sha256 for quiz in quizzes],
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
