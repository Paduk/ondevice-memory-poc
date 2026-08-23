"""Small filesystem tracker consumed by the later training dashboard."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunTracker:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "metrics.jsonl"
        self.status_path = run_dir / "status.json"

    def initialize(self, config: dict[str, Any]) -> None:
        self.write_json("config.json", config)
        self.status("RUNNING", global_step=0, message="run initialized")

    def metric(self, event: str, **values: Any) -> None:
        payload = {"created_at": utc_now(), "event": event, **values}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def status(self, state: str, **values: Any) -> None:
        self.write_json(
            "status.json", {"updated_at": utc_now(), "state": state, **values}
        )

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path


class MLflowMirror:
    """Optional MLflow mirror; local JSON remains the recovery source."""

    def __init__(
        self,
        tracking_uri: str | None,
        *,
        experiment_name: str,
        run_name: str,
    ) -> None:
        self.active = False
        self.error: str | None = None
        if not tracking_uri:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name)
            self.mlflow = mlflow
            self.active = True
        except Exception as exc:  # noqa: BLE001 - MLflow cannot stop local training.
            self.error = str(exc)

    def parameters(self, values: dict[str, Any]) -> None:
        if self.active:
            self._call(
                self.mlflow.log_params,
                {key: str(value)[:500] for key, value in values.items()},
            )

    def metrics(self, values: dict[str, Any], *, step: int) -> None:
        if not self.active:
            return
        numeric = {
            key: float(value)
            for key, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        self._call(self.mlflow.log_metrics, numeric, step=step)

    def artifact(self, path: Path) -> None:
        if self.active and path.exists():
            self._call(self.mlflow.log_artifact, str(path))

    def close(self, status: str = "FINISHED") -> None:
        if self.active:
            self._call(self.mlflow.end_run, status=status)
            self.active = False

    def _call(self, function: Any, *args: Any, **kwargs: Any) -> None:
        try:
            function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - MLflow is a non-blocking mirror.
            self.error = str(exc)
            self.active = False
