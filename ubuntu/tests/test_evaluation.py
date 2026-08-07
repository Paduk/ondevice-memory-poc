from __future__ import annotations

import json
from pathlib import Path

from palmclaw_ubuntu.cli import main
from palmclaw_ubuntu.evaluation import (
    EvaluationRunner,
    evaluation_profiles,
    load_evaluation_dataset,
)
from palmclaw_ubuntu.storage import SQLiteRepository


def test_builtin_dataset_is_versioned_and_contains_research_cases():
    dataset, digest = load_evaluation_dataset()

    assert dataset.name == "palmclaw-synthetic-memory"
    assert dataset.version == "1.0.0"
    assert len(dataset.cases) == 8
    assert len(digest) == 64
    assert {
        "unsupported_hallucination",
        "unresolved_conflict",
        "synthetic_contact_pii",
        "retrieval_distractors",
    }.issubset({case.id for case in dataset.cases})


def test_profiles_cover_baselines_and_targeted_ablations():
    profiles = evaluation_profiles()

    assert {
        "no_memory",
        "cloud_summary",
        "cloud_structured_bm25",
        "cloud_structured_embedding",
        "local_structured_bm25",
        "local_structured_embedding",
        "local_full",
        "ablation_no_retrieval",
        "ablation_no_gate",
        "ablation_no_redaction",
    } == set(profiles)


def test_offline_evaluation_generates_metrics_traces_and_safe_artifacts(
    settings,
    tmp_path: Path,
):
    with SQLiteRepository(settings.database_path) as repository:
        runner = EvaluationRunner(repository=repository, settings=settings)
        result = runner.run(
            profile_names=(
                "no_memory",
                "local_full",
                "ablation_no_gate",
                "ablation_no_redaction",
            ),
            output_root=tmp_path / "artifacts",
            seed=23,
        )
        stored = repository.evaluation_detail(result.run_id)

    profiles = result.metrics["profiles"]
    assert result.status == "completed"
    assert profiles["no_memory"]["task_success_rate"] == 0
    assert profiles["local_full"]["task_success_rate"] == 1
    assert profiles["local_full"]["memory_f1"] == 1
    assert profiles["local_full"]["retrieval_recall_at_k"] == 1
    assert profiles["local_full"]["conflict_f1"] == 1
    assert profiles["local_full"]["pii_f1"] == 1
    assert profiles["ablation_no_gate"]["memory_f1"] < 1
    assert profiles["ablation_no_gate"]["unsupported_memory_rate"] > 0
    assert profiles["ablation_no_gate"]["wrong_auto_overwrite_rate"] > 0
    assert profiles["local_full"]["cloud_exposed_span_rate"] == 0
    assert profiles["ablation_no_redaction"]["cloud_exposed_span_rate"] == 1
    assert len(stored["cases"]) == 32

    expected_files = {
        "metrics.json",
        "cases.jsonl",
        "results.tsv",
        "results.md",
        "task_success_rate.svg",
        "memory_f1.svg",
        "retrieval_recall_at_k.svg",
        "cloud_exposed_span_rate.svg",
    }
    assert expected_files.issubset(
        {path.name for path in result.artifact_dir.iterdir()}
    )
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result.artifact_dir.iterdir()
        if path.suffix in {".json", ".jsonl", ".tsv", ".md"}
    )
    assert "alice@example.test" not in artifact_text
    assert "[REDACTED_EMAIL]" in artifact_text
    assert "alice@example.test" not in json.dumps(stored)


def test_evaluation_cli_profiles_run_list_and_show(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    monkeypatch.setenv("PALMCLAW_BACKEND", "fake")
    monkeypatch.setenv("PALMCLAW_MEMORY_BACKEND", "fake")
    common = ["--data-dir", str(tmp_path / "runtime")]

    assert main([*common, "eval", "profiles"]) == 0
    profiles = json.loads(capsys.readouterr().out)
    assert "local_full" in profiles

    assert (
        main(
            [
                *common,
                "eval",
                "run",
                "--profiles",
                "no_memory",
                "--case-limit",
                "1",
                "--output-dir",
                str(tmp_path / "artifacts"),
            ]
        )
        == 0
    )
    run = json.loads(capsys.readouterr().out)
    assert run["status"] == "completed"
    assert run["profiles"]["no_memory"]["cases"] == 1

    assert main([*common, "eval", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["id"] == run["run_id"]

    assert main([*common, "eval", "show", run["run_id"]]) == 0
    detail = json.loads(capsys.readouterr().out)
    assert detail["run"]["dataset_version"] == "1.0.0"
    assert len(detail["cases"]) == 1
