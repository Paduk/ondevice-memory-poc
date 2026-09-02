from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from memory_training.dashboard.repository import DashboardRepository
from memory_training.dashboard.server import create_app


def _write_run(root, run_id: str = "granite-patch-r1", state: str = "RUNNING") -> None:
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
            "quiz_esm": 0.6,
            "quiz_tool_f1": 0.7,
            "quiz_arg_exact": 0.5,
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
    assert run["latest_validation"]["quiz_esm"] == 0.6
    assert run["ollama_test"]["quiz_all"]["exact_state_match"] == 0.6
    assert overview["matrix"]["granite4.1-3b"]["patch"]["run_id"] == "granite-patch-r1"
    comparison = repository.comparison(["granite-patch-r1"])
    assert len(comparison["runs"][0]["metrics"]) == 2
    assert (
        repository.detail("granite-patch-r1")["ollama_test"]["memory"]["final_state_f1"]
        == 0.75
    )


def test_run_summary_exposes_active_training_gpu(tmp_path) -> None:
    _write_run(tmp_path)
    repository = DashboardRepository(tmp_path)
    run_path = tmp_path / "granite-patch-r1"

    summary = repository._summary(run_path, active={"gpu": "2"})

    assert summary["gpu"] == "2"


def test_run_summary_exposes_v2_validation_type(tmp_path) -> None:
    _write_run(tmp_path, state="VALIDATING")
    run_path = tmp_path / "granite-patch-r1"
    status = json.loads((run_path / "status.json").read_text())
    status["evaluation"] = {"type": "VALIDATION_V2", "phase": "one_step"}
    (run_path / "status.json").write_text(json.dumps(status))

    summary = DashboardRepository(tmp_path)._summary(run_path)

    assert summary["validation_type"] == "VALIDATION_V2"


def test_live_evaluation_eta_uses_recent_progress_rate(tmp_path) -> None:
    repository = DashboardRepository(tmp_path)
    evaluation = tmp_path / "run" / "test"

    first = repository._live_evaluation_eta(
        evaluation,
        progress=0.1,
        updated_at="2026-01-01T00:00:00+00:00",
        active=True,
        fallback=9999.0,
    )
    second = repository._live_evaluation_eta(
        evaluation,
        progress=0.2,
        updated_at="2026-01-01T00:01:00+00:00",
        active=True,
        fallback=9999.0,
    )

    assert first == 9999.0
    assert second == 480.0


def test_repository_hides_abandoned_resume_metrics_and_recomputes_eta(
    tmp_path,
) -> None:
    _write_run(tmp_path)
    run = tmp_path / "granite-patch-r1"
    status = json.loads((run / "status.json").read_text())
    status["global_step"] = 27
    (run / "status.json").write_text(json.dumps(status))
    metrics = [
        {
            "created_at": "2026-01-01T00:01:00+00:00",
            "event": "train",
            "global_step": 25,
            "loss": 0.9,
        },
        {
            "created_at": "2026-01-01T00:02:00+00:00",
            "event": "train",
            "global_step": 26,
            "loss": 0.8,
        },
        {
            "created_at": "2026-01-01T00:03:00+00:00",
            "event": "train",
            "global_step": 27,
            "loss": 0.7,
        },
        {
            "created_at": "2026-01-01T00:04:00+00:00",
            "event": "train",
            "global_step": 28,
            "loss": 0.6,
        },
        {
            "created_at": "2026-01-01T01:01:00+00:00",
            "event": "train",
            "global_step": 26,
            "loss": 0.2,
        },
        {
            "created_at": "2026-01-01T01:02:00+00:00",
            "event": "train",
            "global_step": 27,
            "loss": 0.1,
        },
        {
            "created_at": "2026-01-01T01:02:30+00:00",
            "event": "validation_teacher_forced",
            "global_step": 27,
            "loss": 0.3,
            "perplexity": 1.35,
        },
    ]
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(metric) for metric in metrics) + "\n"
    )

    detail = DashboardRepository(tmp_path).detail("granite-patch-r1")
    train = [m for m in detail["metrics"] if m["event"] == "train"]
    assert [(m["global_step"], m["loss"]) for m in train] == [
        (25, 0.9),
        (26, 0.2),
        (27, 0.1),
    ]
    assert detail["summary"]["latest_validation"]["teacher_forced_loss"] == 0.3
    assert detail["summary"]["eta_seconds"] == 60 * (100 - 27)


def test_repository_eta_supports_sparse_logs_and_hides_during_validation(
    tmp_path,
) -> None:
    _write_run(tmp_path)
    run = tmp_path / "granite-patch-r1"
    metrics = [
        {
            "created_at": "2026-01-01T00:01:00+00:00",
            "event": "train",
            "global_step": 20,
            "eta_seconds": 999,
        },
        {
            "created_at": "2026-01-01T00:02:00+00:00",
            "event": "train",
            "global_step": 25,
            "eta_seconds": 999,
        },
    ]
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(metric) for metric in metrics) + "\n"
    )

    repository = DashboardRepository(tmp_path)
    assert repository.detail("granite-patch-r1")["summary"]["eta_seconds"] == 900

    status = json.loads((run / "status.json").read_text())
    status["state"] = "VALIDATING"
    (run / "status.json").write_text(json.dumps(status))
    assert repository.detail("granite-patch-r1")["summary"]["eta_seconds"] is None


def test_dashboard_discovers_validation_and_resumable_hf_test_jobs(tmp_path) -> None:
    _write_run(tmp_path, state="VALIDATING")
    run = tmp_path / "granite-patch-r1"
    status = json.loads((run / "status.json").read_text())
    status["evaluation"] = {
        "phase": "closed_loop_memory",
        "scenario_index": 83,
        "item": 25,
        "items": 100,
    }
    (run / "status.json").write_text(json.dumps(status))
    evaluation = run / "test-hf-closed-loop-s91"
    (evaluation / "scenarios").mkdir(parents=True)
    (evaluation / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "palmclaw-hf-closed-loop-test-v1",
                "scenarios": [91],
                "checkpoint": "/checkpoint/epoch-03",
            }
        )
    )
    (evaluation / "summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "completed_scenarios": [91],
                "memory": {"update_f1": 0.5},
                "closed_loop_quiz": {"esm": 0.6},
            }
        )
    )
    (evaluation / "scenarios" / "s091.json").write_text(
        json.dumps(
            {
                "scenario_index": 91,
                "memory": {"turns": 10, "update_f1": 0.5},
                "closed_loop_quiz": {"tasks": 2, "esm": 0.6},
            }
        )
    )

    repository = DashboardRepository(tmp_path)
    jobs = repository.evaluations()["jobs"]
    assert {(job["type"], job["state"]) for job in jobs} == {
        ("VALIDATION", "RUNNING"),
        ("TEST", "COMPLETED"),
    }
    detail = repository.evaluation_detail("granite-patch-r1", "test-hf-closed-loop-s91")
    assert detail["scenarios"][0]["closed_loop_quiz"]["esm"] == 0.6

    client = TestClient(create_app(tmp_path))
    assert len(client.get("/api/evaluations").json()["jobs"]) == 2


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
    archived = client.delete("/api/runs/failed-run", params={"confirm": "failed-run"})
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
    assert '"validation_teacher_forced","validation_epoch"' in script
