from __future__ import annotations

import json

from palmclaw_ubuntu.cli import _amem_dry_run_report, build_parser, main
from palmclaw_ubuntu.models import ToolMemoryIdentity
from palmclaw_ubuntu.storage import SQLiteRepository


def test_vehicle_cli_accepts_amem_profile_and_debug_limits():
    args = build_parser().parse_args(
        [
            "eval",
            "vehicle",
            "--benchmark-root",
            "/tmp/VehicleMemBench",
            "--mode",
            "live",
            "--profiles",
            "cloud_amem",
            "--amem-link-candidates",
            "4",
            "--amem-retrieval-top-k",
            "7",
            "--amem-note-limit",
            "11",
        ]
    )

    assert args.profiles == "cloud_amem"
    assert args.amem_link_candidates == 4
    assert args.amem_retrieval_top_k == 7
    assert args.amem_note_limit == 11


def test_vehicle_cli_accepts_amem_style_threshold():
    args = build_parser().parse_args(
        [
            "eval",
            "vehicle",
            "--benchmark-root",
            "/tmp/VehicleMemBench",
            "--mode",
            "live",
            "--profiles",
            "cloud_amem_style",
            "--amem-style-evolution-threshold",
            "0.82",
        ]
    )

    assert args.profiles == "cloud_amem_style"
    assert args.amem_style_evolution_threshold == 0.82


def test_amem_dry_run_report_fixes_calls_token_and_cost_contract():
    report = _amem_dry_run_report(
        ((6, 2_698), (7, 100)),
        note_limit=100,
        max_output_tokens_per_call=2_048,
        output_cost_per_million=4.0,
    )

    assert report["expected_generation_calls"] == 398
    assert report["maximum_generation_output_tokens"] == 815_104
    assert report["maximum_generation_output_cost_usd"] == 3.260416
    assert report["estimated_input_tokens"] is None
    assert report["estimated_latency_ms"] is None
    assert report["scenarios"][0] == {
        "scenario_index": 6,
        "history_note_count": 2_698,
        "selected_note_count": 100,
        "partial_history": True,
        "expected_generation_calls": 199,
        "maximum_generation_output_tokens": 407_552,
        "estimated_input_tokens": None,
        "estimated_latency_ms": None,
    }


def test_amem_style_dry_run_reports_similarity_dependent_call_range():
    report = _amem_dry_run_report(
        ((6, 2_698),),
        note_limit=None,
        max_output_tokens_per_call=2_048,
        output_cost_per_million=6.0,
        profile="cloud_amem_style",
        evolution_similarity_threshold=0.75,
    )

    assert report["profile"] == "cloud_amem_style"
    assert report["minimum_generation_calls"] == 2_698
    assert report["expected_generation_calls"] == 5_395
    assert report["evolution_similarity_threshold"] == 0.75


def test_fake_cli_session_ask_and_history(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PALMCLAW_MEMORY_TRIGGER_MESSAGES", "2")
    monkeypatch.setenv("PALMCLAW_MEMORY_STRATEGY", "summary")
    common = [
        "--data-dir",
        str(tmp_path / "runtime"),
        "--backend",
        "fake",
    ]
    assert main([*common, "session", "new", "--title", "CLI"]) == 0
    session_id = capsys.readouterr().out.strip()
    assert session_id

    assert main([*common, "ask", "hello"]) == 0
    ask_output = capsys.readouterr()
    assert "Echo: hello" in ask_output.out
    assert "status=completed" in ask_output.err

    assert main([*common, "history"]) == 0
    history = capsys.readouterr().out
    assert "\tuser\thello" in history
    assert "\tassistant\tEcho: hello" in history

    assert main([*common, "memory", "show"]) == 0
    memory = capsys.readouterr().out
    assert "User said: hello" in memory


def test_fake_cli_memory_patch_pending_run_and_trace(
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.setenv("PALMCLAW_PATCH_MEMORY_ENABLED", "1")
    monkeypatch.setenv("PALMCLAW_PATCH_MEMORY_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setenv("PALMCLAW_MEMORY_TRIGGER_MESSAGES", "100")
    common = [
        "--data-dir",
        str(tmp_path / "runtime"),
        "--backend",
        "fake",
    ]
    assert main([*common, "session", "new", "--title", "Patch CLI"]) == 0
    session_id = capsys.readouterr().out.strip()
    assert main([*common, "ask", "--session", session_id, "hello"]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                *common,
                "memory",
                "patch",
                "pending",
                "--session",
                session_id,
            ]
        )
        == 0
    )
    pending = json.loads(capsys.readouterr().out)
    assert pending["ready_count"] == 1
    assert pending["jobs"][0]["source_turn_id"]

    assert (
        main(
            [
                *common,
                "memory",
                "patch",
                "run",
                "--session",
                session_id,
            ]
        )
        == 0
    )
    run = json.loads(capsys.readouterr().out)
    assert run["completed_count"] == 1
    assert run["queue"]["ready_count"] == 0

    assert (
        main(
            [
                *common,
                "memory",
                "patch",
                "trace",
                run["run_ids"][0],
            ]
        )
        == 0
    )
    trace = json.loads(capsys.readouterr().out)
    assert trace["run"]["status"] == "completed"
    assert trace["job"]["status"] == "completed"
    assert trace["model_calls"][0]["role"] == "memory_patch"


def test_structured_memory_cli_inspect_search_and_trace(
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.setenv("PALMCLAW_MEMORY_TRIGGER_MESSAGES", "2")
    monkeypatch.setenv("PALMCLAW_MEMORY_STRATEGY", "structured")
    monkeypatch.setenv("PALMCLAW_RETRIEVAL_MODE", "bm25")
    common = [
        "--data-dir",
        str(tmp_path / "runtime"),
        "--backend",
        "fake",
    ]
    assert main([*common, "session", "new", "--title", "Phase 3"]) == 0
    session_id = capsys.readouterr().out.strip()
    assert (
        main(
            [
                *common,
                "ask",
                "--session",
                session_id,
                "I like Python coding",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main([*common, "memory", "show", "--session", session_id]) == 0
    memories = json.loads(capsys.readouterr().out)
    memory = memories[0]
    assert memory["status"] == "verified"
    assert memory["value"] == "I like Python coding"

    assert main([*common, "memory", "inspect", memory["id"]]) == 0
    detail = json.loads(capsys.readouterr().out)
    assert detail["sources"][0]["message_id"]
    assert detail["embeddings"][0]["model_id"]

    assert main([*common, "memory", "versions", memory["fact_key"]]) == 0
    versions = json.loads(capsys.readouterr().out)
    assert len(versions) == 1

    assert (
        main(
            [
                *common,
                "memory",
                "search",
                "--session",
                session_id,
                "--mode",
                "bm25",
                "Python",
            ]
        )
        == 0
    )
    search = json.loads(capsys.readouterr().out)
    assert search["metadata"]["retrieval_selected_count"] == 1

    assert (
        main(
            [
                *common,
                "memory",
                "retrieval-trace",
                search["run_id"],
            ]
        )
        == 0
    )
    trace = json.loads(capsys.readouterr().out)
    assert trace["candidates"][0]["selected"] == 1

    assert (
        main(
            [
                *common,
                "memory",
                "compare",
                "--session",
                session_id,
                "Python",
            ]
        )
        == 0
    )
    comparison = json.loads(capsys.readouterr().out)["comparison"]
    assert set(comparison) == {"full", "bm25", "embedding", "hybrid"}
    assert comparison["bm25"]["selected_ids"] == [memory["id"]]

    assert (
        main(
            [
                *common,
                "memory",
                "retrievals",
                "--session",
                session_id,
            ]
        )
        == 0
    )
    retrievals = json.loads(capsys.readouterr().out)
    assert search["run_id"] in {run["id"] for run in retrievals}


def test_tool_memory_cli_search_list_and_trace(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PALMCLAW_TOOL_MEMORY_RETRIEVAL_ENABLED", "1")
    monkeypatch.setenv("PALMCLAW_TOOL_MEMORY_RETRIEVAL_MODE", "bm25")
    data_dir = tmp_path / "runtime"
    common = [
        "--data-dir",
        str(data_dir),
        "--backend",
        "fake",
    ]
    assert main([*common, "session", "new", "--title", "Tool retrieval"]) == 0
    session_id = capsys.readouterr().out.strip()
    with SQLiteRepository(data_dir / "palmclaw.db") as repository:
        record = repository.insert_tool_memory_record(
            session_id=session_id,
            identity=ToolMemoryIdentity(
                user_id="default_user",
                tool_domain="file",
                topic="write",
                scope="global",
                scope_key="default_user",
            ),
            value="reports/result.md",
            memory_type="preference",
            confidence=0.95,
            version=1,
        )

    assert (
        main(
            [
                *common,
                "memory",
                "tool",
                "search",
                "--session",
                session_id,
                "save",
                "report",
                "file",
            ]
        )
        == 0
    )
    search = json.loads(capsys.readouterr().out)
    assert search["records"][0]["id"] == record.id
    assert search["metadata"]["tool_memory_retrieval_selected_count"] == 1

    assert (
        main(
            [
                *common,
                "memory",
                "tool",
                "retrievals",
                "--session",
                session_id,
            ]
        )
        == 0
    )
    retrievals = json.loads(capsys.readouterr().out)
    assert retrievals[0]["id"] == search["run_id"]

    assert (
        main(
            [
                *common,
                "memory",
                "tool",
                "trace",
                search["run_id"],
            ]
        )
        == 0
    )
    trace = json.loads(capsys.readouterr().out)
    assert trace["run"]["status"] == "completed"
    assert trace["candidates"][0]["record_id"] == record.id
    assert trace["candidates"][0]["selected"] == 1
