"""Filesystem-backed aggregation for concurrent memory-training runs."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
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

    def overview(self) -> dict[str, Any]:
        runs = [self._summary(path) for path in self._run_directories()]
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
                for state in ("RUNNING", "COMPLETED", "FAILED", "QUEUED")
            },
            "stale_count": sum(bool(run["stale"]) for run in runs),
            "matrix": matrix,
            "runs": runs,
        }

    def detail(self, run_id: str) -> dict[str, Any]:
        path = self._run_path(run_id)
        summary = self._summary(path)
        metrics = _read_jsonl(path / "metrics.jsonl")
        validations = []
        for validation_path in sorted(path.glob("validation-epoch-*.json")):
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

    def _summary(self, path: Path) -> dict[str, Any]:
        config = _read_json(path / "config.json", {})
        status = _read_json(path / "status.json", {})
        best = _read_json(path / "best-checkpoint.json", {})
        metrics = _read_jsonl(path / "metrics.jsonl")
        ollama_test = _read_json(path / "ollama-test-summary.json", {})
        arguments = config.get("arguments", {})
        model = config.get("model", {})
        state = _normalize_state(status.get("state"))
        created_at = config.get("created_at") or _file_time(path)
        updated_at = (
            status.get("updated_at") or _latest_metric_time(metrics) or created_at
        )
        elapsed = _elapsed_seconds(
            created_at, updated_at if state != "RUNNING" else None
        )
        latest_train = _latest_event(metrics, "train")
        latest_validation = _latest_event(metrics, "validation_epoch")
        total_steps = int(config.get("total_optimizer_steps") or 0)
        global_step = int(
            status.get("global_step") or latest_train.get("global_step") or 0
        )
        progress = min(global_step / total_steps, 1.0) if total_steps else 0.0
        stale = state == "RUNNING" and _age_seconds(updated_at) > STALE_AFTER_SECONDS
        return {
            "run_id": path.name,
            "model_key": model.get("key", "unknown"),
            "model_family": model.get("family", "unknown"),
            "parameters_b": model.get("parameters_b"),
            "method": config.get("method", "unknown"),
            "seed": arguments.get("seed"),
            "state": state,
            "stale": stale,
            "created_at": created_at,
            "updated_at": updated_at,
            "elapsed_seconds": elapsed,
            "eta_seconds": latest_train.get("eta_seconds"),
            "global_step": global_step,
            "total_steps": total_steps,
            "progress": progress,
            "epoch": status.get("epoch", latest_train.get("epoch")),
            "epochs": arguments.get("epochs"),
            "batch": status.get("batch", latest_train.get("batch")),
            "latest_train": latest_train,
            "latest_validation": latest_validation,
            "best_checkpoint": best,
            "ollama_test": _compact_ollama_test(ollama_test),
            "failure": status.get("traceback") or status.get("message")
            if state == "FAILED"
            else None,
        }


def _compact_ollama_test(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    quiz = value.get("quiz") or {}
    return {
        "created_at": value.get("created_at"),
        "scenarios": value.get("scenarios", []),
        "memory": value.get("memory") or {},
        "quiz_all": quiz.get("all") or {},
        "quiz_turn": quiz.get("TURN") or {},
        "quiz_final": quiz.get("FINAL") or {},
        "environment": value.get("environment") or {},
    }


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
    if value in {"RUNNING", "FAILED"}:
        return value
    return "QUEUED"


def _latest_event(records: list[dict[str, Any]], event: str) -> dict[str, Any]:
    return next(
        (record for record in reversed(records) if record.get("event") == event), {}
    )


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


def _directory_size(path: Path) -> int:
    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
