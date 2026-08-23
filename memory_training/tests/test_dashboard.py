from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from memory_training.dashboard.repository import DashboardRepository
from memory_training.dashboard.server import create_app


def _write_run(
    root, run_id: str = "granite-patch-r1", state: str = "RUNNING"
) -> None:
    run = root / run_id
    run.mkdir(parents=True)
    config = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "method": "patch",
        "model": {"key": "granite4.1-3b", "family": "granite4.1", "parameters_b": 3},
        "arguments": {"seed": 42, "epochs": 3},
        "total_optimizer_steps": 100,
    }
    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "global_step": 25,
        "epoch": 0,
    }
    metrics = [
        {
            "created_at": "2026-01-01T00:01:00+00:00",
            "event": "train",
            "global_step": 25,
            "loss": 0.7,
            "target_tokens_per_second": 123.0,
            "eta_seconds": 60,
        },
        {
            "created_at": "2026-01-01T00:02:00+00:00",
            "event": "validation_epoch",
            "global_step": 25,
            "closed_loop_final_state_f1": 0.8,
        },
    ]
    (run / "config.json").write_text(json.dumps(config))
    (run / "status.json").write_text(json.dumps(status))
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(metric) for metric in metrics) + "\n{partial"
    )
    (run / "ollama-test-summary.json").write_text(
        json.dumps(
            {
                "created_at": "2026-01-02T00:00:00+00:00",
                "scenarios": list(range(91, 101)),
                "memory": {"final_state_f1": 0.75, "latency_seconds": 12.0},
                "quiz": {
                    "all": {
                        "exact_state_match": 0.6,
                        "argument_exact_match": 0.5,
                    }
                },
                "environment": {"ollama_version": "0.6.5"},
            }
        )
    )


def test_repository_builds_matrix_and_comparison(tmp_path) -> None:
    _write_run(tmp_path)
    repository = DashboardRepository(tmp_path)
    overview = repository.overview()
    run = overview["runs"][0]
    assert run["progress"] == 0.25
    assert run["latest_train"]["loss"] == 0.7
    assert run["ollama_test"]["quiz_all"]["exact_state_match"] == 0.6
    assert overview["matrix"]["granite4.1-3b"]["patch"]["run_id"] == "granite-patch-r1"
    comparison = repository.comparison(["granite-patch-r1"])
    assert len(comparison["runs"][0]["metrics"]) == 2
    assert repository.detail("granite-patch-r1")["ollama_test"]["memory"][
        "final_state_f1"
    ] == 0.75


def test_dashboard_api_archives_only_finished_runs_and_rejects_traversal(
    tmp_path,
) -> None:
    _write_run(tmp_path)
    _write_run(tmp_path, "failed-run", state="FAILED")
    client = TestClient(create_app(tmp_path))
    health = client.get("/api/health").json()
    assert health["read_only"] is False
    assert health["deletion_mode"] == "recoverable_archive"
    assert client.get("/api/runs").json()["counts"]["RUNNING"] == 1
    assert client.get("/api/runs/granite-patch-r1").status_code == 200
    assert client.get("/api/runs/%2E%2E").status_code == 404
    assert (
        client.delete(
            "/api/runs/granite-patch-r1", params={"confirm": "granite-patch-r1"}
        ).status_code
        == 409
    )
    assert (
        client.delete("/api/runs/failed-run", params={"confirm": "wrong"}).status_code
        == 400
    )
    archived = client.delete(
        "/api/runs/failed-run", params={"confirm": "failed-run"}
    )
    assert archived.status_code == 200
    archive_path = Path(archived.json()["archive_path"])
    assert archived.json()["recoverable"] is True
    assert archive_path.is_dir()
    assert client.get("/api/runs/failed-run").status_code == 404
    assert client.get("/api/runs").json()["counts"]["FAILED"] == 0
    assert client.get("/").status_code == 200
    script = client.get("/static/app.js").text
    assert "if (state.selected.size) await loadComparison();" in script
    assert 'const logarithmic=labelText==="train.loss";' in script
