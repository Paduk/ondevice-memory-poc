"""Filesystem-backed aggregation for concurrent memory-training runs."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

STALE_AFTER_SECONDS = 300
ARCHIVABLE_STATES = {"COMPLETED", "FAILED"}


class RunNotArchivableError(RuntimeError):
    """Raised when an active or not-yet-initialized run is archived."""


class DashboardRepository:
    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root.resolve()
        self.trash_root = self.runs_root.parent / "trash" / "runs"
        self._evaluation_progress_samples: dict[str, list[tuple[float, float]]] = {}

    def overview(self) -> dict[str, Any]:
        active = _active_training_runs()
        runs = [
            self._summary(path, active=active.get(path.name))
            for path in self._run_directories()
        ]
        runs.sort(key=lambda run: run["created_at"] or "", reverse=True)
        counts = Counter(run["state"] for run in runs)
        matrix: dict[str, dict[str, Any]] = {}
        for run in runs:
            model = run["model_key"]
            method = run["method"]
            if model == "unknown" or method == "unknown":
                continue
            matrix.setdefault(model, {}).setdefault(method, run)
        return {
            "generated_at": _utc_now(),
            "runs_root": str(self.runs_root),
            "counts": {
                state: counts.get(state, 0)
                for state in (
                    "RUNNING",
                    "VALIDATING",
                    "PAUSED",
                    "COMPLETED",
                    "FAILED",
                    "QUEUED",
                )
            },
            "stale_count": sum(bool(run["stale"]) for run in runs),
            "matrix": matrix,
            "runs": runs,
        }

    def detail(self, run_id: str) -> dict[str, Any]:
        path = self._run_path(run_id)
        summary = self._summary(path, active=_active_training_runs().get(path.name))
        metrics = _dashboard_metrics(
            _read_jsonl(path / "metrics.jsonl"),
            current_step=summary["global_step"],
        )
        validations = []
        validation_paths = [
            *path.glob("validation-epoch-*.json"),
            *path.glob("validation-v2-epoch-*.json"),
        ]
        for validation_path in sorted(validation_paths):
            payload = _read_json(validation_path, {})
            payload["artifact"] = validation_path.name
            validations.append(payload)
        checkpoints = []
        checkpoint_root = path / "checkpoints"
        if checkpoint_root.is_dir():
            for checkpoint in sorted(checkpoint_root.iterdir()):
                progress = _read_json(checkpoint / "progress.json", {})
                checkpoints.append(
                    {
                        "name": checkpoint.name,
                        "path": str(checkpoint),
                        "progress": progress,
                        "size_bytes": _directory_size(checkpoint),
                    }
                )
        return {
            "summary": summary,
            "config": _read_json(path / "config.json", {}),
            "status": _read_json(path / "status.json", {}),
            "best_checkpoint": _read_json(path / "best-checkpoint.json", None),
            "ollama_test": _read_json(path / "ollama-test-summary.json", None),
            "metrics": metrics,
            "validations": validations,
            "checkpoints": checkpoints,
        }

    def comparison(self, run_ids: list[str]) -> dict[str, Any]:
        details = [self.detail(run_id) for run_id in run_ids]
        return {
            "generated_at": _utc_now(),
            "runs": [
                {
                    "summary": detail["summary"],
                    "metrics": detail["metrics"],
                    "validations": detail["validations"],
                }
                for detail in details
            ],
        }

    def evaluations(self) -> dict[str, Any]:
        active = _active_hf_evaluations()
        active_training = _active_training_runs()
        jobs = []
        for run_path in self._run_directories():
            run_summary = self._summary(
                run_path, active=active_training.get(run_path.name)
            )
            if run_summary["state"] == "VALIDATING":
                jobs.append(self._validation_job(run_path, run_summary))
            for path in sorted(run_path.iterdir()):
                manifest = (
                    _read_json(path / "manifest.json", {}) if path.is_dir() else {}
                )
                if manifest.get("schema_version") == "palmclaw-hf-closed-loop-test-v1":
                    jobs.append(
                        self._hf_test_job(run_path, path, active.get(path.resolve()))
                    )
        jobs.sort(key=lambda job: job.get("updated_at") or "", reverse=True)
        return {"generated_at": _utc_now(), "jobs": jobs}

    def evaluation_detail(self, run_id: str, evaluation_name: str) -> dict[str, Any]:
        run_path = self._run_path(run_id)
        if evaluation_name == "validation":
            status = _read_json(run_path / "status.json", {})
            artifacts = []
            validation_paths = [
                *run_path.glob("validation-epoch-*.json"),
                *run_path.glob("validation-v2-epoch-*.json"),
            ]
            for path in sorted(validation_paths):
                payload = _read_json(path, {})
                artifacts.append(
                    {
                        "artifact": path.name,
                        "closed_loop": payload.get("closed_loop", {}),
                        "closed_loop_quiz": _compact_quiz(
                            payload.get("closed_loop_quiz")
                        ),
                        "gold_quiz": _compact_quiz(payload.get("quiz")),
                    }
                )
            return {"type": "VALIDATION", "status": status, "artifacts": artifacts}
        if not evaluation_name or Path(evaluation_name).name != evaluation_name:
            raise KeyError(evaluation_name)
        path = (run_path / evaluation_name).resolve()
        if path.parent != run_path or not path.is_dir():
            raise KeyError(evaluation_name)
        manifest = _read_json(path / "manifest.json", {})
        if manifest.get("schema_version") != "palmclaw-hf-closed-loop-test-v1":
            raise KeyError(evaluation_name)
        scenarios = []
        for artifact in sorted((path / "scenarios").glob("s*.json")):
            payload = _read_json(artifact, {})
            memory = payload.get("memory", {})
            scenarios.append(
                {
                    "scenario_index": payload.get("scenario_index"),
                    "memory": {
                        key: memory.get(key)
                        for key in (
                            "turns",
                            "update_f1",
                            "state_f1",
                            "final_state_f1",
                            "latency_seconds",
                            "total_tokens",
                        )
                    },
                    "closed_loop_quiz": _compact_quiz(payload.get("closed_loop_quiz")),
                }
            )
        return {
            "type": "TEST",
            "manifest": manifest,
            "progress": _read_json(path / "progress.json", {}),
            "summary": _read_json(path / "summary.json", {}),
            "scenarios": scenarios,
        }

    def export_summary(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.overview(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
        return output_path

    def archive(self, run_id: str) -> dict[str, Any]:
        """Atomically move a finished run out of the dashboard into recoverable trash."""
        path = self._run_path(run_id)
        summary = self._summary(path)
        if summary["state"] not in ARCHIVABLE_STATES:
            raise RunNotArchivableError(
                f"Only COMPLETED or FAILED runs can be archived: {summary['state']}"
            )
        self.trash_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = self.trash_root / f"{run_id}--{stamp}"
        path.replace(destination)
        return {
            "run_id": run_id,
            "state": summary["state"],
            "archived_at": _utc_now(),
            "archive_path": str(destination),
            "recoverable": True,
        }

    def _run_directories(self) -> list[Path]:
        if not self.runs_root.is_dir():
            return []
        return sorted(
            path
            for path in self.runs_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

    def _run_path(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise KeyError(run_id)
        path = (self.runs_root / run_id).resolve()
        if path.parent != self.runs_root or not path.is_dir():
            raise KeyError(run_id)
        return path

    def _summary(
        self, path: Path, *, active: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        config = _read_json(path / "config.json", {})
        status = _read_json(path / "status.json", {})
        best = _read_json(path / "best-checkpoint.json", {})
        raw_metrics = _read_jsonl(path / "metrics.jsonl")
        ollama_test = _read_json(path / "ollama-test-summary.json", {})
        arguments = config.get("arguments", {})
        model = config.get("model", {})
        state = _normalize_state(status.get("state"))
        created_at = config.get("created_at") or _file_time(path)
        updated_at = (
            status.get("updated_at") or _latest_metric_time(raw_metrics) or created_at
        )
        elapsed = _elapsed_seconds(
            created_at, updated_at if state != "RUNNING" else None
        )
        total_steps = int(config.get("total_optimizer_steps") or 0)
        status_step = status.get("global_step")
        metrics = _dashboard_metrics(raw_metrics, current_step=status_step)
        latest_train = _latest_event(metrics, "train")
        global_step = int(status_step or latest_train.get("global_step") or 0)
        latest_validation = _latest_validation(metrics)
        current_evaluation = status.get("evaluation") or {}
        estimated_eta = (
            _estimated_eta_seconds(
                raw_metrics,
                current_step=global_step,
                total_steps=total_steps,
            )
            if state == "RUNNING"
            else None
        )
        progress = min(global_step / total_steps, 1.0) if total_steps else 0.0
        stale = state == "RUNNING" and _age_seconds(updated_at) > STALE_AFTER_SECONDS
        return {
            "run_id": path.name,
            "model_key": model.get("key", "unknown"),
            "model_family": model.get("family", "unknown"),
            "parameters_b": model.get("parameters_b"),
            "method": config.get("method", "unknown"),
            "gpu": (active or {}).get("gpu"),
            "seed": arguments.get("seed"),
            "state": state,
            "stale": stale,
            "created_at": created_at,
            "updated_at": updated_at,
            "elapsed_seconds": elapsed,
            "eta_seconds": estimated_eta,
            "global_step": global_step,
            "total_steps": total_steps,
            "progress": progress,
            "epoch": status.get("epoch", latest_train.get("epoch")),
            "epochs": arguments.get("epochs"),
            "batch": status.get("batch", latest_train.get("batch")),
            "latest_train": latest_train,
            "latest_validation": latest_validation,
            "validation_type": current_evaluation.get("type"),
            "best_checkpoint": best,
            "ollama_test": _compact_ollama_test(ollama_test),
            "failure": status.get("traceback") or status.get("message")
            if state == "FAILED"
            else None,
        }

    def _validation_job(
        self, run_path: Path, run_summary: dict[str, Any]
    ) -> dict[str, Any]:
        status = _read_json(run_path / "status.json", {})
        evaluation = status.get("evaluation") or {}
        item = evaluation.get("item")
        items = evaluation.get("items")
        phase_progress = (
            float(item) / float(items)
            if isinstance(item, (int, float))
            and isinstance(items, (int, float))
            and items
            else None
        )
        return {
            "job_id": f"{run_path.name}:validation",
            "run_id": run_path.name,
            "evaluation_name": "validation",
            "type": evaluation.get("type", "VALIDATION"),
            "state": "RUNNING",
            "phase": evaluation.get("phase", "epoch_validation"),
            "epoch": status.get("epoch"),
            "current_scenario": evaluation.get("scenario_index"),
            "completed_scenarios": [],
            "scenario_count": evaluation.get("scenario_total"),
            "item": item,
            "items": items,
            "phase_progress": phase_progress,
            "progress": phase_progress,
            "esm": run_summary["latest_validation"].get("closed_loop_quiz_esm"),
            "update_f1": run_summary["latest_validation"].get("closed_loop_update_f1"),
            "gpu": run_summary.get("gpu"),
            "eta_seconds": None,
            "elapsed_seconds": None,
            "updated_at": status.get("updated_at"),
            "checkpoint": status.get("checkpoint"),
            "stale": _age_seconds(status.get("updated_at")) > STALE_AFTER_SECONDS,
        }

    def _hf_test_job(
        self,
        run_path: Path,
        path: Path,
        active: dict[str, Any] | None,
    ) -> dict[str, Any]:
        manifest = _read_json(path / "manifest.json", {})
        progress = _read_json(path / "progress.json", {})
        summary = _read_json(path / "summary.json", {})
        expected = [int(value) for value in manifest.get("scenarios", [])]
        completed = [
            int(value)
            for value in (
                progress.get("completed_scenarios")
                or summary.get("completed_scenarios")
                or [
                    int(artifact.stem.removeprefix("s"))
                    for artifact in (path / "scenarios").glob("s*.json")
                ]
            )
        ]
        complete = bool(summary.get("complete"))
        if complete:
            state = "COMPLETED"
        elif active is not None:
            state = "RUNNING"
        elif str(progress.get("status", "")).upper() == "FAILED":
            state = "FAILED"
        else:
            state = "PAUSED"
        current = progress.get("current_scenario")
        if current is None and active is not None:
            current = next(
                (value for value in expected if value not in completed), None
            )
        overall = progress.get("progress")
        if not isinstance(overall, (int, float)):
            overall = len(completed) / len(expected) if expected else 0.0
        started_at = (
            progress.get("started_at")
            or manifest.get("created_at")
            or _file_time(path / "manifest.json")
        )
        updated_at = (
            progress.get("updated_at")
            or _latest_file_time(path / "scenarios")
            or _file_time(path / "manifest.json")
        )
        elapsed = progress.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)):
            elapsed = _elapsed_seconds(
                started_at, None if state == "RUNNING" else updated_at
            )
        eta = progress.get("eta_seconds")
        if not isinstance(eta, (int, float)) and completed and state == "RUNNING":
            eta = (
                float(elapsed or 0.0)
                / len(completed)
                * (len(expected) - len(completed))
            )
        eta = self._live_evaluation_eta(
            path,
            progress=float(overall),
            updated_at=progress.get("updated_at"),
            active=active is not None and not complete,
            fallback=float(eta) if isinstance(eta, (int, float)) else None,
        )
        quiz = summary.get("closed_loop_quiz", {})
        memory = summary.get("memory", {})
        return {
            "job_id": f"{run_path.name}:{path.name}",
            "run_id": run_path.name,
            "evaluation_name": path.name,
            "type": "TEST",
            "state": state,
            "phase": progress.get("phase") or "closed_loop_memory",
            "epoch": None,
            "current_scenario": current,
            "current_scenarios": progress.get("current_scenarios")
            or ([current] if current else []),
            "completed_scenarios": sorted(completed),
            "scenario_count": len(expected),
            "item": progress.get("item"),
            "items": progress.get("items"),
            "phase_progress": (
                float(progress["item"]) / float(progress["items"])
                if progress.get("items")
                else None
            ),
            "progress": float(overall),
            "esm": progress.get("esm", quiz.get("esm")),
            "update_f1": progress.get("update_f1", memory.get("update_f1")),
            "gpu": progress.get("gpu") or (active or {}).get("gpu"),
            "eta_seconds": eta,
            "elapsed_seconds": elapsed,
            "updated_at": _utc_now() if active is not None else updated_at,
            "checkpoint": manifest.get("checkpoint"),
            "stale": state == "RUNNING" and active is None,
        }

    def _live_evaluation_eta(
        self,
        path: Path,
        *,
        progress: float,
        updated_at: Any,
        active: bool,
        fallback: float | None,
    ) -> float | None:
        """Estimate active Test ETA from recent progress instead of old restarts."""
        key = str(path.resolve())
        if not active or progress >= 1.0:
            self._evaluation_progress_samples.pop(key, None)
            return fallback
        timestamp = _timestamp_seconds(updated_at)
        if timestamp is None:
            return fallback
        samples = self._evaluation_progress_samples.setdefault(key, [])
        if samples and progress < samples[-1][1]:
            samples.clear()
        if not samples or (timestamp, progress) != samples[-1]:
            samples.append((timestamp, progress))
        cutoff = timestamp - 300.0
        samples[:] = [sample for sample in samples[-20:] if sample[0] >= cutoff]
        if len(samples) < 2:
            return fallback
        elapsed = samples[-1][0] - samples[0][0]
        advanced = samples[-1][1] - samples[0][1]
        if elapsed <= 0 or advanced <= 0:
            return fallback
        return elapsed / advanced * max(0.0, 1.0 - progress)


def _compact_ollama_test(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    quiz = value.get("quiz") or {}
    return {
        "created_at": value.get("created_at"),
        "scenarios": value.get("scenarios", []),
        "memory": value.get("memory") or {},
        "memory_teacher_forced": value.get("memory_teacher_forced") or {},
        "memory_mode_comparison": value.get("memory_mode_comparison") or {},
        "quiz_all": quiz.get("all") or {},
        "quiz_turn": quiz.get("TURN") or {},
        "quiz_final": quiz.get("FINAL") or {},
        "environment": value.get("environment") or {},
    }


def _compact_quiz(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in (
            "tasks",
            "esm",
            "tool_f1",
            "arg_exact",
            "parse_success_rate",
            "execution_success_rate",
            "latency_seconds",
            "total_tokens",
        )
        if key in value
    }


def _active_hf_evaluations() -> dict[Path, dict[str, Any]]:
    active: dict[Path, dict[str, Any]] = {}
    proc = Path("/proc")
    try:
        processes = [path for path in proc.iterdir() if path.name.isdigit()]
    except OSError:
        return active
    for process in processes:
        try:
            arguments = [
                value.decode("utf-8", errors="replace")
                for value in (process / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except OSError:
            continue
        if "memory_training.evaluate_hf_closed_loop" not in arguments:
            continue
        try:
            output_index = arguments.index("--output-dir") + 1
            output_dir = Path(arguments[output_index]).resolve()
        except (ValueError, IndexError, OSError):
            continue
        gpu = None
        try:
            environment = (process / "environ").read_bytes().split(b"\0")
            prefix = b"CUDA_VISIBLE_DEVICES="
            gpu_value = next(
                (
                    value[len(prefix) :]
                    for value in environment
                    if value.startswith(prefix)
                ),
                None,
            )
            gpu = gpu_value.decode("utf-8", errors="replace") if gpu_value else None
        except OSError:
            pass
        active[output_dir] = {"pid": int(process.name), "gpu": gpu}
    return active


def _active_training_runs() -> dict[str, dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    proc = Path("/proc")
    try:
        processes = [path for path in proc.iterdir() if path.name.isdigit()]
    except OSError:
        return discovered
    for process in processes:
        try:
            arguments = [
                value.decode("utf-8", errors="replace")
                for value in (process / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except OSError:
            continue
        if not {
            "memory_training.train",
            "memory_training.validate_hf_checkpoint_v2",
        }.intersection(arguments):
            continue
        try:
            run_id = arguments[arguments.index("--run-id") + 1]
        except (ValueError, IndexError):
            continue
        gpu = _process_cuda_visible_devices(process)
        entry = discovered.setdefault(run_id, {"pids": [], "gpus": set()})
        entry["pids"].append(int(process.name))
        if gpu:
            entry["gpus"].update(
                part.strip() for part in gpu.split(",") if part.strip()
            )
    return {
        run_id: {
            "pids": sorted(entry["pids"]),
            "gpu": ",".join(sorted(entry["gpus"], key=_gpu_sort_key)) or None,
        }
        for run_id, entry in discovered.items()
    }


def _process_cuda_visible_devices(process: Path) -> str | None:
    try:
        environment = (process / "environ").read_bytes().split(b"\0")
    except OSError:
        return None
    prefix = b"CUDA_VISIBLE_DEVICES="
    value = next(
        (item[len(prefix) :] for item in environment if item.startswith(prefix)), None
    )
    return value.decode("utf-8", errors="replace") if value else None


def _gpu_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except OSError:
        pass
    return records


def _normalize_state(value: Any) -> str:
    value = str(value or "QUEUED").upper()
    if value in {"COMPLETED", "CANARY_COMPLETE", "FINISHED"}:
        return "COMPLETED"
    if value in {"RUNNING", "VALIDATING", "PAUSED", "FAILED"}:
        return value
    return "QUEUED"


def _latest_event(records: list[dict[str, Any]], event: str) -> dict[str, Any]:
    return next(
        (record for record in reversed(records) if record.get("event") == event), {}
    )


def _latest_validation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine the latest cheap step Validation with the latest epoch metrics."""
    teacher_forced = _latest_event(records, "validation_teacher_forced")
    epoch = _latest_event(records, "validation_epoch")
    result: dict[str, Any] = {}
    if teacher_forced:
        result.update(
            {
                "teacher_forced_loss": teacher_forced.get("loss"),
                "teacher_forced_perplexity": teacher_forced.get("perplexity"),
                "teacher_forced_global_step": teacher_forced.get("global_step"),
            }
        )
    result.update(epoch)
    return result


def _dashboard_metrics(
    records: list[dict[str, Any]], *, current_step: Any
) -> list[dict[str, Any]]:
    """Hide abandoned post-checkpoint history and retain the newest row per step.

    A resumed run appends to its existing JSONL. If it resumes from an older
    checkpoint, the file therefore contains two records for the same step and
    records beyond the live progress. The raw log remains untouched for audit;
    only the dashboard view is normalized.
    """
    try:
        live_step = int(current_step) if current_step is not None else None
    except (TypeError, ValueError):
        live_step = None
    deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
    unstepped: list[dict[str, Any]] = []
    for record in records:
        try:
            step = int(record["global_step"])
        except (KeyError, TypeError, ValueError):
            unstepped.append(record)
            continue
        if live_step is not None and step > live_step:
            continue
        deduplicated[(str(record.get("event", "")), step)] = record
    stepped = sorted(
        deduplicated.values(),
        key=lambda record: (
            int(record.get("global_step", 0)),
            str(record.get("created_at", "")),
            str(record.get("event", "")),
        ),
    )
    return [*unstepped, *stepped]


def _estimated_eta_seconds(
    records: list[dict[str, Any]], *, current_step: int, total_steps: int
) -> float | None:
    """Estimate remaining train time from the latest contiguous logging segment."""
    if current_step >= total_steps or total_steps <= 0:
        return 0.0 if total_steps else None
    train = [
        record
        for record in records
        if record.get("event") == "train"
        and int(record.get("global_step", 0)) <= current_step
    ]
    if not train:
        return None
    segment = [train[-1]]
    next_step = int(train[-1].get("global_step", 0))
    for record in reversed(train[:-1]):
        try:
            step = int(record["global_step"])
        except (KeyError, TypeError, ValueError):
            break
        if step >= next_step:
            break
        segment.append(record)
        next_step = step
        if len(segment) >= 51:
            break
    segment.reverse()
    intervals = []
    for previous, current in pairwise(segment):
        start = _parse_time(previous.get("created_at"))
        end = _parse_time(current.get("created_at"))
        step_delta = int(current["global_step"]) - int(previous["global_step"])
        if start is not None and end is not None and end > start and step_delta > 0:
            intervals.append((end - start).total_seconds() / step_delta)
    if not intervals:
        value = train[-1].get("eta_seconds")
        return float(value) if isinstance(value, (int, float)) else None
    seconds_per_step = statistics.median(intervals[-20:])
    return seconds_per_step * max(total_steps - current_step, 0)


def _latest_metric_time(records: list[dict[str, Any]]) -> str | None:
    return str(records[-1].get("created_at")) if records else None


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _timestamp_seconds(value: Any) -> float | None:
    parsed = _parse_time(value)
    return parsed.timestamp() if parsed is not None else None


def _elapsed_seconds(start: Any, end: Any = None) -> float | None:
    start_time = _parse_time(start)
    end_time = _parse_time(end) if end else datetime.now(timezone.utc)
    if not start_time or not end_time:
        return None
    return max(0.0, (end_time - start_time).total_seconds())


def _age_seconds(value: Any) -> float:
    parsed = _parse_time(value)
    if not parsed:
        return float("inf")
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _file_time(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def _latest_file_time(path: Path) -> str | None:
    try:
        timestamps = [item.stat().st_mtime for item in path.iterdir() if item.is_file()]
    except OSError:
        return None
    if not timestamps:
        return None
    return datetime.fromtimestamp(max(timestamps), timezone.utc).isoformat()


def _directory_size(path: Path) -> int:
    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
