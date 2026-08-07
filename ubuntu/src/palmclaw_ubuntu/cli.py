from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.application import create_runtime
from palmclaw_ubuntu.config import Settings
from palmclaw_ubuntu.skills import SkillsLoader
from palmclaw_ubuntu.storage import SQLiteRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="palmclaw",
        description="PalmClaw Ubuntu CLI agent runtime",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Runtime data directory (or PALMCLAW_DATA_DIR)",
    )
    parser.add_argument(
        "--backend",
        choices=("openai", "fake"),
        help="Model backend (or PALMCLAW_BACKEND)",
    )
    parser.add_argument(
        "--memory-backend",
        choices=("openai", "local", "fake"),
        help="MemoryModel backend (or PALMCLAW_MEMORY_BACKEND)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local configuration")
    doctor.set_defaults(handler=_doctor)

    session = subparsers.add_parser("session", help="Manage sessions")
    session_subparsers = session.add_subparsers(
        dest="session_command",
        required=True,
    )
    session_new = session_subparsers.add_parser("new")
    session_new.add_argument("--title", default="Chat")
    session_new.set_defaults(handler=_session_new)
    session_list = session_subparsers.add_parser("list")
    session_list.set_defaults(handler=_session_list)
    session_use = session_subparsers.add_parser("use")
    session_use.add_argument("session_id")
    session_use.set_defaults(handler=_session_use)
    session_show = session_subparsers.add_parser("show")
    session_show.add_argument("session_id", nargs="?")
    session_show.set_defaults(handler=_session_show)

    ask = subparsers.add_parser("ask", help="Run one agent turn")
    ask.add_argument("text", nargs="+")
    ask.add_argument("--session")
    ask.set_defaults(handler=_ask)

    chat = subparsers.add_parser("chat", help="Start an interactive chat")
    chat.add_argument("--session")
    chat.set_defaults(handler=_chat)

    history = subparsers.add_parser("history", help="Show session history")
    history.add_argument("--session")
    history.add_argument("--limit", type=int, default=100)
    history.set_defaults(handler=_history)

    trace = subparsers.add_parser("trace", help="Show one turn trace")
    trace.add_argument("turn_id")
    trace.set_defaults(handler=_trace)

    memory = subparsers.add_parser("memory", help="Inspect session memory")
    memory_subparsers = memory.add_subparsers(
        dest="memory_command",
        required=True,
    )
    memory_show = memory_subparsers.add_parser("show")
    memory_show.add_argument("--session")
    memory_show.add_argument("--all-versions", action="store_true")
    memory_show.set_defaults(handler=_memory_show)
    memory_inspect = memory_subparsers.add_parser(
        "inspect",
        help="Show one memory with evidence and embedding metadata",
    )
    memory_inspect.add_argument("memory_id")
    memory_inspect.set_defaults(handler=_memory_inspect)
    memory_versions = memory_subparsers.add_parser(
        "versions",
        help="Show the immutable version chain for a fact key",
    )
    memory_versions.add_argument("fact_key")
    memory_versions.set_defaults(handler=_memory_versions)
    memory_search = memory_subparsers.add_parser(
        "search",
        help="Run and trace memory retrieval",
    )
    memory_search.add_argument("query", nargs="+")
    memory_search.add_argument("--session")
    memory_search.add_argument(
        "--mode",
        choices=("none", "full", "bm25", "embedding", "hybrid"),
    )
    memory_search.add_argument("--top-k", type=int)
    memory_search.set_defaults(handler=_memory_search)
    memory_compare = memory_subparsers.add_parser(
        "compare",
        help="Compare full, BM25, embedding, and hybrid retrieval",
    )
    memory_compare.add_argument("query", nargs="+")
    memory_compare.add_argument("--session")
    memory_compare.add_argument("--top-k", type=int)
    memory_compare.set_defaults(handler=_memory_compare)
    memory_retrievals = memory_subparsers.add_parser(
        "retrievals",
        help="List retrieval runs",
    )
    memory_retrievals.add_argument("--session")
    memory_retrievals.add_argument("--limit", type=int, default=20)
    memory_retrievals.set_defaults(handler=_memory_retrievals)
    memory_trace = memory_subparsers.add_parser(
        "retrieval-trace",
        help="Show candidate scores and selection decisions",
    )
    memory_trace.add_argument("run_id")
    memory_trace.set_defaults(handler=_memory_retrieval_trace)
    memory_review = memory_subparsers.add_parser(
        "review",
        help="List candidates awaiting manual review",
    )
    memory_review.add_argument("--session")
    memory_review.set_defaults(handler=_memory_review)
    memory_resolve = memory_subparsers.add_parser(
        "resolve",
        help="Accept or reject a candidate in the review queue",
    )
    memory_resolve.add_argument("memory_id")
    memory_resolve.add_argument(
        "--decision",
        required=True,
        choices=("accept", "reject"),
    )
    memory_resolve.add_argument("--reason", default="cli_review")
    memory_resolve.set_defaults(handler=_memory_resolve)
    memory_backend_report = memory_subparsers.add_parser(
        "backend-report",
        help="Compare recorded Cloud and Local MemoryModel runs",
    )
    memory_backend_report.add_argument("--session")
    memory_backend_report.set_defaults(handler=_memory_backend_report)
    memory_patch = memory_subparsers.add_parser(
        "patch",
        help="Inspect and run the durable Tool-memory patch queue",
    )
    memory_patch_subparsers = memory_patch.add_subparsers(
        dest="memory_patch_command",
        required=True,
    )
    memory_patch_pending = memory_patch_subparsers.add_parser(
        "pending",
        help="Show queued, running, and retryable patch jobs",
    )
    memory_patch_pending.add_argument("--session")
    memory_patch_pending.add_argument("--limit", type=int, default=100)
    memory_patch_pending.set_defaults(handler=_memory_patch_pending)
    memory_patch_run = memory_patch_subparsers.add_parser(
        "run",
        help="Process one bounded batch of queued patch jobs",
    )
    memory_patch_run.add_argument("--session")
    memory_patch_run.add_argument("--limit", type=int)
    memory_patch_run.set_defaults(handler=_memory_patch_run)
    memory_patch_trace = memory_patch_subparsers.add_parser(
        "trace",
        help="Show one patch run, model call, and proposal trace",
    )
    memory_patch_trace.add_argument("run_id")
    memory_patch_trace.set_defaults(handler=_memory_patch_trace)
    memory_fact = memory_subparsers.add_parser(
        "fact",
        help="Inspect Fact-first records and extraction metrics",
    )
    memory_fact_subparsers = memory_fact.add_subparsers(
        dest="memory_fact_command",
        required=True,
    )
    memory_fact_show = memory_fact_subparsers.add_parser(
        "show",
        help="Show active or all Fact-first record versions",
    )
    memory_fact_show.add_argument("--session")
    memory_fact_show.add_argument("--all-versions", action="store_true")
    memory_fact_show.set_defaults(handler=_memory_fact_show)
    memory_fact_metrics = memory_fact_subparsers.add_parser(
        "metrics",
        help="Show Fact extraction, linking, and token counts",
    )
    memory_fact_metrics.add_argument("--session")
    memory_fact_metrics.set_defaults(handler=_memory_fact_metrics)
    memory_tool = memory_subparsers.add_parser(
        "tool",
        help="Inspect routed Tool-memory retrieval",
    )
    memory_tool_subparsers = memory_tool.add_subparsers(
        dest="memory_tool_command",
        required=True,
    )
    memory_tool_search = memory_tool_subparsers.add_parser(
        "search",
        help="Route a query and retrieve compact Tool memory",
    )
    memory_tool_search.add_argument("query", nargs="+")
    memory_tool_search.add_argument("--session")
    memory_tool_search.add_argument(
        "--mode",
        choices=("none", "full", "bm25", "embedding", "hybrid"),
    )
    memory_tool_search.add_argument("--top-k", type=int)
    memory_tool_search.set_defaults(handler=_memory_tool_search)
    memory_tool_retrievals = memory_tool_subparsers.add_parser(
        "retrievals",
        help="List Tool-memory retrieval runs",
    )
    memory_tool_retrievals.add_argument("--session")
    memory_tool_retrievals.add_argument("--limit", type=int, default=20)
    memory_tool_retrievals.set_defaults(handler=_memory_tool_retrievals)
    memory_tool_trace = memory_tool_subparsers.add_parser(
        "trace",
        help="Show Tool-memory routing, ranking, and exclusions",
    )
    memory_tool_trace.add_argument("run_id")
    memory_tool_trace.set_defaults(handler=_memory_tool_trace)

    evaluation = subparsers.add_parser(
        "eval",
        help="Run and inspect Phase 5 evaluations",
    )
    evaluation_subparsers = evaluation.add_subparsers(
        dest="evaluation_command",
        required=True,
    )
    evaluation_profiles_parser = evaluation_subparsers.add_parser(
        "profiles",
        help="List built-in baseline and ablation profiles",
    )
    evaluation_profiles_parser.set_defaults(handler=_evaluation_profiles)
    evaluation_run = evaluation_subparsers.add_parser(
        "run",
        help="Run a versioned evaluation dataset",
    )
    evaluation_run.add_argument("--dataset", type=Path)
    evaluation_run.add_argument(
        "--profiles",
        help="Comma-separated profile names; defaults to all",
    )
    evaluation_run.add_argument(
        "--mode",
        choices=("offline", "live"),
        default="offline",
    )
    evaluation_run.add_argument("--repetitions", type=int, default=1)
    evaluation_run.add_argument("--seed", type=int, default=17)
    evaluation_run.add_argument("--output-dir", type=Path)
    evaluation_run.add_argument("--name", default="Phase 5 Evaluation")
    evaluation_run.add_argument("--case-limit", type=int)
    evaluation_run.set_defaults(handler=_evaluation_run)
    evaluation_vehicle = evaluation_subparsers.add_parser(
        "vehicle",
        help="Run VehicleMemBench offline smoke or live Agent evaluation",
    )
    evaluation_vehicle.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="Trusted VehicleMemBench Git checkout",
    )
    evaluation_vehicle.add_argument(
        "--mode",
        choices=("offline", "live"),
        default="offline",
    )
    evaluation_vehicle.add_argument(
        "--profiles",
        help=(
            "Comma-separated live profiles: no_memory, gold_memory, "
            "cloud_amem, cloud_amem_style, cloud_summary, "
            "cloud_recursive_summary, "
            "cloud_fact_recursive_hybrid, "
            "cloud_recursive_assisted_fact_patch, "
            "cloud_schema_informed_recursive_assisted_fact_patch, "
            "cloud_joint_planned_fact_patch, "
            "cloud_structured_bm25, "
            "cloud_structured_embedding, cloud_structured_hybrid, "
            "cloud_schema_patch, cloud_fact_patch, "
            "cloud_schema_informed_fact_patch, "
            "oracle_retrieval_fact_patch, "
            "oracle_binding_fact_patch, "
            "oracle_retrieval_binding_fact_patch, "
            "oracle_gate_fact_patch, "
            "oracle_gate_retrieval_binding_fact_patch, "
            "oracle_structure_fact_patch, "
            "oracle_structure_retrieval_binding_fact_patch, "
            "oracle_extraction_fact_patch, "
            "oracle_extraction_retrieval_binding_fact_patch, "
            "oracle_full_memory_fact_patch, "
            "oracle_full_pipeline_fact_patch, "
            "oracle_tool_gold_memory, oracle_tool_schema_patch, "
            "oracle_tool_fact_patch"
        ),
    )
    evaluation_vehicle.add_argument("--scenario", type=int, default=1)
    evaluation_vehicle.add_argument(
        "--scenario-limit",
        type=int,
        default=1,
        help="Evaluate consecutive scenarios and create a suite report",
    )
    evaluation_vehicle.add_argument("--task-limit", type=int, default=10)
    evaluation_vehicle.add_argument("--max-tool-rounds", type=int, default=10)
    evaluation_vehicle.add_argument("--model")
    evaluation_vehicle.add_argument("--memory-model")
    evaluation_vehicle.add_argument("--embedding-model")
    evaluation_vehicle.add_argument("--memory-cache-dir", type=Path)
    evaluation_vehicle.add_argument(
        "--oracle-annotations",
        type=Path,
        help="Reviewed Oracle Retrieval annotation JSON",
    )
    evaluation_vehicle.add_argument(
        "--oracle-gate-annotations",
        type=Path,
        help="Reviewed Oracle Gate candidate annotation JSON",
    )
    evaluation_vehicle.add_argument(
        "--oracle-stage-fact-annotations",
        type=Path,
        help="Reviewed Oracle Structure and Extraction Fact annotation JSON",
    )
    evaluation_vehicle.add_argument(
        "--memory-batch-tokens",
        type=int,
        default=8_000,
    )
    evaluation_vehicle.add_argument("--patch-batch-turns", type=int)
    evaluation_vehicle.add_argument("--patch-batch-tokens", type=int)
    evaluation_vehicle.add_argument("--memory-top-k", type=int)
    evaluation_vehicle.add_argument("--memory-token-budget", type=int)
    evaluation_vehicle.add_argument(
        "--amem-link-candidates",
        type=int,
        default=5,
        help="A-MEM link/evolution candidate count (1-5)",
    )
    evaluation_vehicle.add_argument(
        "--amem-style-evolution-threshold",
        type=float,
        default=0.75,
        help=(
            "cloud_amem_style top-candidate cosine threshold for evolution "
            "(-1 to 1)"
        ),
    )
    evaluation_vehicle.add_argument(
        "--amem-retrieval-top-k",
        type=int,
        default=10,
        help="A-MEM embedding seed count (1-10)",
    )
    evaluation_vehicle.add_argument(
        "--amem-note-limit",
        type=int,
        help="Debug-only history note limit; marks the run partial",
    )
    evaluation_vehicle.add_argument(
        "--amem-dry-run",
        action="store_true",
        help="Report A-MEM history size and expected generation calls only",
    )
    evaluation_vehicle.add_argument(
        "--resume-run",
        help="Resume a compatible VehicleMemBench run ID",
    )
    evaluation_vehicle.add_argument("--output-dir", type=Path)
    evaluation_vehicle.add_argument(
        "--expected-commit",
        help="Required upstream commit; defaults to the validated Phase V1 commit",
    )
    evaluation_vehicle.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Allow a non-Git checkout or a different upstream commit",
    )
    evaluation_vehicle.set_defaults(handler=_evaluation_vehicle)
    evaluation_vehicle_matcher = evaluation_subparsers.add_parser(
        "vehicle-matcher",
        help="Evaluate Vehicle ontology matching with embeddings and no LLM",
    )
    evaluation_vehicle_matcher.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="Trusted VehicleMemBench Git checkout",
    )
    evaluation_vehicle_matcher.add_argument("--scenario", type=int, default=6)
    evaluation_vehicle_matcher.add_argument("--labels", type=Path)
    evaluation_vehicle_matcher.add_argument("--embedding-model")
    evaluation_vehicle_matcher.add_argument("--batch-turns", type=int)
    evaluation_vehicle_matcher.add_argument("--batch-tokens", type=int)
    evaluation_vehicle_matcher.add_argument("--output-dir", type=Path)
    evaluation_vehicle_matcher.add_argument("--expected-commit")
    evaluation_vehicle_matcher.add_argument(
        "--allow-unpinned",
        action="store_true",
    )
    evaluation_vehicle_matcher.set_defaults(
        handler=_evaluation_vehicle_matcher
    )
    evaluation_vehicle_r0 = evaluation_subparsers.add_parser(
        "vehicle-r0",
        help="Replay frozen VehicleMemBench patch and routing diagnostics offline",
    )
    evaluation_vehicle_r0.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed cloud_schema_patch artifact directory",
    )
    evaluation_vehicle_r0.add_argument(
        "--memory-cache-dir",
        type=Path,
        required=True,
        help="VehicleMemBench memory cache root",
    )
    evaluation_vehicle_r0.add_argument("--output-dir", type=Path)
    evaluation_vehicle_r0.set_defaults(handler=_evaluation_vehicle_r0)
    evaluation_vehicle_oracle = evaluation_subparsers.add_parser(
        "vehicle-oracle-audit",
        help="Audit VehicleMemBench Gold contracts for staged Oracle evaluation",
    )
    evaluation_vehicle_oracle.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="Trusted VehicleMemBench Git checkout",
    )
    evaluation_vehicle_oracle.add_argument("--scenario", type=int, default=6)
    evaluation_vehicle_oracle.add_argument(
        "--scenario-limit",
        type=int,
        default=5,
    )
    evaluation_vehicle_oracle.add_argument("--task-limit", type=int, default=10)
    evaluation_vehicle_oracle.add_argument("--baseline-run-dir", type=Path)
    evaluation_vehicle_oracle.add_argument("--output-dir", type=Path)
    evaluation_vehicle_oracle.add_argument(
        "--expected-commit",
        help="Required upstream commit; defaults to the validated commit",
    )
    evaluation_vehicle_oracle.add_argument(
        "--allow-unpinned",
        action="store_true",
    )
    evaluation_vehicle_oracle.set_defaults(handler=_evaluation_vehicle_oracle_audit)
    evaluation_list = evaluation_subparsers.add_parser(
        "list",
        help="List evaluation runs",
    )
    evaluation_list.add_argument("--limit", type=int, default=20)
    evaluation_list.set_defaults(handler=_evaluation_list)
    evaluation_show = evaluation_subparsers.add_parser(
        "show",
        help="Show one evaluation run and its case traces",
    )
    evaluation_show.add_argument("run_id")
    evaluation_show.set_defaults(handler=_evaluation_show)

    skill = subparsers.add_parser("skill", help="Inspect skills")
    skill_subparsers = skill.add_subparsers(
        dest="skill_command",
        required=True,
    )
    skill_list = skill_subparsers.add_parser("list")
    skill_list.set_defaults(handler=_skill_list)

    tool = subparsers.add_parser("tool", help="Inspect tools")
    tool_subparsers = tool.add_subparsers(
        dest="tool_command",
        required=True,
    )
    tool_list = tool_subparsers.add_parser("list")
    tool_list.set_defaults(handler=_tool_list)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.backend:
        os.environ["PALMCLAW_BACKEND"] = args.backend
    if args.memory_backend:
        os.environ["PALMCLAW_MEMORY_BACKEND"] = args.memory_backend
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except (KeyError, RuntimeError, TimeoutError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) else str(exc)
        print(f"Error: {message}", file=sys.stderr)
        return 2


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env(data_dir=args.data_dir)


def _repository(args: argparse.Namespace) -> SQLiteRepository:
    settings = _settings(args)
    settings.ensure_directories()
    return SQLiteRepository(settings.database_path)


def _doctor(args: argparse.Namespace) -> int:
    settings = _settings(args)
    settings.ensure_directories()
    with SQLiteRepository(settings.database_path):
        pass
    checks = {
        "backend": settings.backend,
        "data_dir": str(settings.data_dir),
        "database": str(settings.database_path),
        "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "agent_model": settings.agent_model,
        "memory_model": settings.memory_model,
        "memory_backend": settings.memory_backend,
        "local_memory_model": settings.local_memory_model,
        "local_memory_base_url": settings.local_memory_base_url,
        "memory_strategy": settings.memory_strategy,
        "memory_gate_enabled": settings.memory_gate_enabled,
        "cloud_pii_redaction": settings.cloud_pii_redaction,
        "local_pii_storage": settings.local_pii_storage,
        "retrieval_mode": settings.retrieval_mode,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "patch_memory_enabled": settings.patch_memory_enabled,
        "patch_memory_backend": settings.patch_memory_backend,
        "patch_memory_model": settings.patch_memory_model,
        "patch_memory_user_id": settings.patch_memory_user_id,
        "patch_memory_batch_size": settings.patch_memory_batch_size,
        "patch_memory_batch_tokens": settings.patch_memory_batch_tokens,
        "tool_memory_retrieval_enabled": (
            settings.tool_memory_retrieval_enabled
        ),
        "tool_memory_retrieval_mode": settings.tool_memory_retrieval_mode,
        "tool_memory_top_k": settings.tool_memory_top_k,
        "tool_memory_context_tokens": settings.tool_memory_context_tokens,
        "tool_memory_semantic_routing_enabled": (
            settings.tool_memory_semantic_routing_enabled
        ),
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if settings.backend == "openai":
        return int(
            not (
                checks["openai_api_key"]
                and checks["agent_model"]
                and checks["memory_model"]
            )
        )
    return 0


def _session_new(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session = repository.create_session(args.title)
        repository.set_active_session(session.id)
        print(session.id)
    return 0


def _session_list(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        active = repository.get_active_session_id()
        for session in repository.list_sessions():
            marker = "*" if session.id == active else " "
            print(f"{marker} {session.id}\t{session.title}\t{session.updated_at}")
    return 0


def _session_use(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        repository.set_active_session(args.session_id)
    print(args.session_id)
    return 0


def _session_show(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session_id = args.session_id or repository.get_active_session_id()
        if not session_id:
            raise RuntimeError("No active session")
        session = repository.require_session(session_id)
        turns = repository.list_turns(session_id)
    print(
        json.dumps(
            {
                "id": session.id,
                "title": session.title,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "turns": turns,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _resolve_session(
    repository: SQLiteRepository,
    requested_session_id: str | None,
) -> str:
    if requested_session_id:
        repository.require_session(requested_session_id)
        return requested_session_id
    active = repository.get_active_session_id()
    if active:
        return active
    session = repository.create_session("Chat")
    repository.set_active_session(session.id)
    return session.id


def _ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with create_runtime(settings) as runtime:
        session_id = _resolve_session(
            runtime.repository,
            args.session,
        )
        runtime.repository.set_active_session(session_id)
        result = runtime.turn_coordinator.run(session_id, " ".join(args.text))
        print(result.content)
        print(
            f"[turn={result.turn_id} rounds={result.rounds} status={result.status}]",
            file=sys.stderr,
        )
    return int(result.status != "completed")


def _chat(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with create_runtime(settings) as runtime:
        session_id = _resolve_session(
            runtime.repository,
            args.session,
        )
        runtime.repository.set_active_session(session_id)
        print(f"Session: {session_id}. Type /exit to quit.")
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text in {"/exit", "/quit"}:
                break
            result = runtime.turn_coordinator.run(session_id, text)
            print(f"agent> {result.content}")
    return 0


def _history(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session_id = _resolve_session(repository, args.session)
        messages = repository.list_messages(
            session_id,
            limit=max(1, args.limit),
        )
    for message in messages:
        suffix = f" tool_call_id={message.tool_call_id}" if message.tool_call_id else ""
        print(f"{message.id}\t{message.role}{suffix}\t{message.content}")
    return 0


def _trace(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        trace = repository.get_trace(args.turn_id)
    print(json.dumps(trace, ensure_ascii=False, indent=2))
    return 0


def _memory_show(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session_id = _resolve_session(repository, args.session)
        memories = repository.list_memories(
            session_id,
            include_superseded=args.all_versions,
        )
    print(json.dumps(memories, ensure_ascii=False, indent=2))
    return 0


def _memory_inspect(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        memory = repository.memory_detail(args.memory_id)
    print(json.dumps(memory, ensure_ascii=False, indent=2))
    return 0


def _memory_versions(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        memories = repository.memory_versions(args.fact_key)
    print(json.dumps(memories, ensure_ascii=False, indent=2))
    return 0


def _memory_search(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with create_runtime(settings) as runtime:
        session_id = _resolve_session(runtime.repository, args.session)
        result = runtime.memory_engine.retrieve(
            session_id,
            turn_id=None,
            query=" ".join(args.query),
            mode=args.mode,
            top_k=args.top_k,
        )
        payload = {
            "run_id": result.run_id,
            "content": result.content,
            "memories": [asdict(memory) for memory in result.memories],
            "metadata": dict(result.metadata),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _memory_compare(args: argparse.Namespace) -> int:
    settings = _settings(args)
    if settings.memory_strategy != "structured":
        raise ValueError("memory compare requires PALMCLAW_MEMORY_STRATEGY=structured")
    with create_runtime(settings) as runtime:
        session_id = _resolve_session(runtime.repository, args.session)
        query = " ".join(args.query)
        comparison = {}
        for mode in ("full", "bm25", "embedding", "hybrid"):
            result = runtime.memory_engine.retrieve(
                session_id,
                turn_id=None,
                query=query,
                mode=mode,
                top_k=args.top_k,
            )
            comparison[mode] = {
                "run_id": result.run_id,
                "selected_ids": [memory.id for memory in result.memories],
                "selected_values": [memory.value for memory in result.memories],
                "metadata": dict(result.metadata),
            }
    print(
        json.dumps(
            {"query": query, "comparison": comparison},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _memory_retrievals(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session_id = _resolve_session(repository, args.session)
        runs = repository.list_retrieval_runs(
            session_id,
            limit=max(1, args.limit),
        )
    print(json.dumps(runs, ensure_ascii=False, indent=2))
    return 0


def _memory_retrieval_trace(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        trace = repository.retrieval_trace(args.run_id)
    print(json.dumps(trace, ensure_ascii=False, indent=2))
    return 0


def _memory_review(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session_id = _resolve_session(repository, args.session)
        reviews = repository.list_memory_reviews(session_id)
    print(json.dumps(reviews, ensure_ascii=False, indent=2))
    return 0


def _memory_resolve(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        memory = repository.resolve_memory_review(
            args.memory_id,
            decision=args.decision,
            reason=args.reason,
        )
    print(json.dumps(memory, ensure_ascii=False, indent=2))
    return 0


def _memory_backend_report(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session_id = _resolve_session(repository, args.session)
        report = repository.memory_backend_report(session_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _memory_patch_pending(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session_id = (
            _resolve_session(repository, args.session)
            if args.session is not None
            else None
        )
        payload = repository.memory_patch_queue_status(session_id=session_id)
        payload["jobs"] = repository.memory_patch_jobs(
            session_id=session_id,
            statuses=("queued", "running", "retryable"),
            limit=max(1, args.limit),
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _memory_patch_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with create_runtime(settings) as runtime:
        worker = runtime.patch_worker or runtime.fact_worker
        if worker is None:
            raise RuntimeError(
                "Background memory is disabled; enable "
                "PALMCLAW_PATCH_MEMORY_ENABLED or PALMCLAW_FACT_MEMORY_ENABLED"
            )
        session_id = (
            _resolve_session(runtime.repository, args.session)
            if args.session is not None
            else None
        )
        result = worker.run_pending(
            limit=args.limit or settings.patch_memory_batch_size,
            session_id=session_id,
        )
        payload = {
            **result.as_dict(),
            "queue": runtime.repository.memory_patch_queue_status(
                session_id=session_id
            ),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return int(result.failed_count > 0)


def _memory_patch_trace(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        trace = repository.memory_patch_trace(args.run_id)
    print(json.dumps(trace, ensure_ascii=False, indent=2))
    return 0


def _memory_fact_show(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session_id = _resolve_session(repository, args.session)
        records = repository.list_fact_memory_records(
            session_id,
            active_only=not args.all_versions,
        )
        payload = []
        for record in records:
            detail = repository.fact_memory_record_detail(record.id)
            payload.append(
                {
                    "record": asdict(detail["record"]),
                    "sources": [
                        asdict(source) for source in detail["sources"]
                    ],
                    "status_events": detail["status_events"],
                }
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _memory_fact_metrics(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session_id = _resolve_session(repository, args.session)
        payload = repository.fact_memory_metrics(session_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _memory_tool_search(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with create_runtime(settings) as runtime:
        retriever = (
            runtime.fact_memory_retriever
            if runtime.fact_memory_retriever is not None
            else runtime.tool_memory_retriever
        )
        if retriever is None:
            raise RuntimeError(
                "Memory retrieval is disabled; set "
                "PALMCLAW_TOOL_MEMORY_RETRIEVAL_ENABLED=1"
            )
        session_id = _resolve_session(runtime.repository, args.session)
        result = retriever.retrieve(
            session_id,
            turn_id=None,
            query=" ".join(args.query),
            mode=args.mode,
            top_k=args.top_k,
        )
        payload = {
            "run_id": result.run_id,
            "content": result.content,
            "records": [asdict(record) for record in result.records],
            "metadata": dict(result.metadata),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _memory_tool_retrievals(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        session_id = _resolve_session(repository, args.session)
        runs = repository.list_tool_memory_retrievals(
            session_id,
            limit=max(1, args.limit),
        )
    print(json.dumps(runs, ensure_ascii=False, indent=2))
    return 0


def _memory_tool_trace(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        trace = repository.tool_memory_retrieval_trace(args.run_id)
    print(json.dumps(trace, ensure_ascii=False, indent=2))
    return 0


def _evaluation_profiles(args: argparse.Namespace) -> int:
    del args
    from palmclaw_ubuntu.evaluation import evaluation_profiles

    payload = {name: asdict(profile) for name, profile in evaluation_profiles().items()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _evaluation_run(args: argparse.Namespace) -> int:
    from palmclaw_ubuntu.evaluation import EvaluationRunner

    settings = _settings(args)
    settings.ensure_directories()
    selected_profiles = tuple(
        item.strip() for item in (args.profiles or "").split(",") if item.strip()
    )
    with SQLiteRepository(settings.database_path) as repository:
        runner = EvaluationRunner(
            repository=repository,
            settings=settings,
        )
        result = runner.run(
            dataset_path=args.dataset,
            profile_names=selected_profiles,
            execution_mode=args.mode,
            repetitions=args.repetitions,
            seed=args.seed,
            output_root=args.output_dir,
            name=args.name,
            case_limit=args.case_limit,
        )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.status,
                "artifact_dir": str(result.artifact_dir),
                "profiles": result.metrics["profiles"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return int(result.status == "failed")


def _amem_dry_run_report(
    scenario_note_counts: Sequence[tuple[int, int]],
    *,
    note_limit: int | None,
    max_output_tokens_per_call: int,
    output_cost_per_million: float,
    profile: str = "cloud_amem",
    evolution_similarity_threshold: float = 0.75,
) -> dict[str, Any]:
    from palmclaw_ubuntu.vehicle_bench import estimate_amem_generation_calls

    if note_limit is not None and note_limit < 1:
        raise ValueError("--amem-note-limit must be positive")
    if profile not in {"cloud_amem", "cloud_amem_style"}:
        raise ValueError(f"Unsupported A-MEM profile: {profile}")
    scenarios = []
    for scenario_index, entry_count in scenario_note_counts:
        selected_count = (
            min(entry_count, note_limit)
            if note_limit is not None
            else entry_count
        )
        expected_calls = estimate_amem_generation_calls(selected_count)
        scenario = {
            "scenario_index": scenario_index,
            "history_note_count": entry_count,
            "selected_note_count": selected_count,
            "partial_history": note_limit is not None,
            "expected_generation_calls": expected_calls,
            "maximum_generation_output_tokens": (
                expected_calls * max_output_tokens_per_call
            ),
            "estimated_input_tokens": None,
            "estimated_latency_ms": None,
        }
        if profile == "cloud_amem_style":
            scenario["minimum_generation_calls"] = selected_count
        scenarios.append(scenario)
    expected_calls = sum(
        item["expected_generation_calls"] for item in scenarios
    )
    maximum_output_tokens = sum(
        item["maximum_generation_output_tokens"] for item in scenarios
    )
    return {
        "profile": profile,
        "dry_run": True,
        "scenarios": scenarios,
        "expected_generation_calls": expected_calls,
        "maximum_generation_output_tokens": maximum_output_tokens,
        "maximum_generation_output_cost_usd": (
            round(
                maximum_output_tokens
                * output_cost_per_million
                / 1_000_000,
                8,
            )
            if output_cost_per_million
            else None
        ),
        "memory_output_cost_per_million": (
            output_cost_per_million if output_cost_per_million else None
        ),
        "estimated_input_tokens": None,
        "estimated_latency_ms": None,
        "estimate_basis": (
            "N to 2N-1 generation calls; the exact evolution count depends "
            "on the similarity gate. Reported expected calls and output "
            "cost are conservative maxima."
            if profile == "cloud_amem_style"
            else "2N-1 generation calls; output token limit only. Input "
            "token, latency, and full cost estimates require a bounded "
            "Cloud smoke."
        ),
        **(
            {
                "evolution_similarity_threshold": (
                    evolution_similarity_threshold
                ),
                "minimum_generation_calls": sum(
                    item["minimum_generation_calls"] for item in scenarios
                ),
            }
            if profile == "cloud_amem_style"
            else {}
        ),
    }


def _evaluation_vehicle(args: argparse.Namespace) -> int:
    from palmclaw_ubuntu.vehicle_bench import (
        OFFICIAL_UPSTREAM_COMMIT,
        VEHICLE_BASELINE_PROFILES,
        VEHICLE_FACT_INSTRUCTIONS,
        VEHICLE_FACT_ORACLE_ANNOTATION_PROFILES,
        VEHICLE_FACT_PATCH_PROFILES,
        VEHICLE_FACT_PROMPT_VERSION,
        VEHICLE_FULL_ORACLE_PROFILES,
        VEHICLE_GATE_ORACLE_PROFILES,
        VEHICLE_PATCH_INSTRUCTIONS,
        VEHICLE_PATCH_PROMPT_VERSION,
        VEHICLE_RECURSIVE_ASSISTED_FACT_INSTRUCTIONS,
        VEHICLE_RECURSIVE_ASSISTED_FACT_PROMPT_VERSION,
        VEHICLE_RECURSIVE_SUMMARY_INSTRUCTIONS,
        VEHICLE_RECURSIVE_SUMMARY_PROMPT_VERSION,
        VEHICLE_SCHEMA_INFORMED_FACT_INSTRUCTIONS,
        VEHICLE_SCHEMA_INFORMED_FACT_PROMPT_VERSION,
        VEHICLE_STAGE_FACT_ORACLE_PROFILES,
        VEHICLE_STRUCTURED_INSTRUCTIONS,
        VEHICLE_STRUCTURED_PROMPT_VERSION,
        VEHICLE_SUMMARY_INSTRUCTIONS,
        VEHICLE_SUMMARY_PROMPT_VERSION,
        VehicleMemoryBuilder,
        load_oracle_gate_annotations,
        load_oracle_retrieval_annotations,
        load_oracle_stage_fact_annotations,
        load_vehicle_benchmark,
        load_vehicle_fact_ontology_v1,
        parse_vehicle_history,
        required_memory_strategies,
        run_agent_evaluation,
        run_offline_smoke,
    )
    from palmclaw_ubuntu.vehicle_bench.runner import (
        write_vehicle_report_artifacts,
    )
    from palmclaw_ubuntu.vehicle_bench.suite import (
        run_vehicle_evaluation_suite,
        summarize_provider_calls,
    )

    if args.allow_unpinned and args.expected_commit:
        raise ValueError("--allow-unpinned and --expected-commit cannot be combined")
    expected_commit = None
    if not args.allow_unpinned:
        expected_commit = args.expected_commit or OFFICIAL_UPSTREAM_COMMIT
    dataset = load_vehicle_benchmark(
        args.benchmark_root,
        expected_commit=expected_commit,
    )
    if args.mode == "offline":
        result = run_offline_smoke(
            dataset,
            scenario_index=args.scenario,
            task_limit=args.task_limit,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return int(not result.passed)

    if args.amem_dry_run:
        dry_settings = _settings(args)
        profiles = tuple(
            item.strip()
            for item in (args.profiles or "").split(",")
            if item.strip()
        )
        amem_profiles = tuple(
            profile
            for profile in profiles
            if profile in {"cloud_amem", "cloud_amem_style"}
        )
        if len(amem_profiles) != 1:
            raise RuntimeError(
                "--amem-dry-run requires exactly one A-MEM profile"
            )
        if args.scenario_limit < 1:
            raise ValueError("--scenario-limit must be at least 1")
        if args.amem_note_limit is not None and args.amem_note_limit < 1:
            raise ValueError("--amem-note-limit must be positive")
        print(
            json.dumps(
                _amem_dry_run_report(
                    tuple(
                        (
                            scenario_index,
                            len(
                                parse_vehicle_history(
                                    dataset.scenario(
                                        scenario_index
                                    ).history_path
                                )
                            ),
                        )
                        for scenario_index in range(
                            args.scenario,
                            args.scenario + args.scenario_limit,
                        )
                    ),
                    note_limit=args.amem_note_limit,
                    max_output_tokens_per_call=(
                        dry_settings.memory_max_output_tokens
                    ),
                    output_cost_per_million=(
                        dry_settings.memory_output_cost_per_million
                    ),
                    profile=amem_profiles[0],
                    evolution_similarity_threshold=(
                        args.amem_style_evolution_threshold
                    ),
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    from palmclaw_ubuntu.providers import (
        OpenAIAMemModel,
        OpenAIEmbeddingModel,
        OpenAIFactMemoryModel,
        OpenAIMemoryModel,
        OpenAIPatchMemoryModel,
        OpenAIPostNormalizedFactMemoryModel,
        OpenAIRecursiveSummaryMemoryModel,
        OpenAIResponsesAgentModel,
        OpenAISchemaInformedFactMemoryModel,
        OpenAIStructuredMemoryModel,
    )

    settings = _settings(args)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    model_id = (args.model or settings.agent_model or "").strip()
    if not model_id:
        raise RuntimeError("Vehicle live evaluation requires an Agent model")
    selected_profiles = tuple(
        item.strip() for item in (args.profiles or "").split(",") if item.strip()
    )
    profiles = selected_profiles or VEHICLE_BASELINE_PROFILES
    schema_informed_profiles = {
        "cloud_schema_informed_fact_patch",
        "cloud_schema_informed_recursive_assisted_fact_patch",
        "cloud_joint_planned_fact_patch",
    }
    selected_schema_informed_profiles = (
        set(profiles) & schema_informed_profiles
    )
    if selected_schema_informed_profiles:
        other_fact_profiles = (
            set(profiles) & set(VEHICLE_FACT_PATCH_PROFILES)
        ) - selected_schema_informed_profiles
        if len(selected_schema_informed_profiles) > 1:
            other_fact_profiles.update(selected_schema_informed_profiles)
        if other_fact_profiles:
            raise RuntimeError(
                "Schema-informed Fact and other Fact profiles require "
                "separate runs"
            )
    oracle_retrieval_annotations = None
    if any(
        profile in VEHICLE_FACT_ORACLE_ANNOTATION_PROFILES
        for profile in profiles
    ):
        if args.oracle_annotations is None:
            raise RuntimeError(
                "Fact Oracle profiles require --oracle-annotations"
            )
        oracle_retrieval_annotations = load_oracle_retrieval_annotations(
            args.oracle_annotations,
            dataset=dataset,
        )
    oracle_gate_annotations = None
    if any(
        profile in (*VEHICLE_GATE_ORACLE_PROFILES, *VEHICLE_FULL_ORACLE_PROFILES)
        for profile in profiles
    ):
        if args.oracle_gate_annotations is None:
            raise RuntimeError(
                "Oracle Gate profiles require --oracle-gate-annotations"
            )
        oracle_gate_annotations = load_oracle_gate_annotations(
            args.oracle_gate_annotations,
            dataset=dataset,
        )
    requested_oracle_stages = {
        (
            "structure"
            if profile.startswith("oracle_structure_")
            else "extraction"
        )
        for profile in profiles
        if profile in VEHICLE_STAGE_FACT_ORACLE_PROFILES
    }
    if any(profile in VEHICLE_FULL_ORACLE_PROFILES for profile in profiles):
        requested_oracle_stages.add("full")
    if len(requested_oracle_stages) > 1:
        raise RuntimeError(
            "Structure, Extraction, and Full Oracle profiles require "
            "separate runs"
        )
    oracle_stage = (
        next(iter(requested_oracle_stages))
        if requested_oracle_stages
        else None
    )
    oracle_stage_fact_annotations = None
    if oracle_stage is not None:
        if (
            args.oracle_stage_fact_annotations is None
            or oracle_retrieval_annotations is None
        ):
            raise RuntimeError(
                "Oracle Stage and Full profiles require "
                "--oracle-stage-fact-annotations and --oracle-annotations"
            )
        oracle_stage_fact_annotations = load_oracle_stage_fact_annotations(
            args.oracle_stage_fact_annotations,
            dataset=dataset,
            retrieval_annotations=oracle_retrieval_annotations,
            gate_annotations=oracle_gate_annotations,
        )
    agent_model = OpenAIResponsesAgentModel(
        model_id,
        timeout_seconds=settings.model_timeout_seconds,
        max_output_tokens=settings.agent_max_output_tokens,
        reasoning_effort=settings.agent_reasoning_effort,
        redact_pii=settings.cloud_pii_redaction,
        pii_allowlist=settings.pii_allowlist,
    )
    output_root = args.output_dir or (
        Path(__file__).resolve().parents[2] / "evaluation" / "vehiclemembench"
    )
    strategies = required_memory_strategies(profiles)
    summary_model = None
    recursive_summary_model = None
    structured_model = None
    embedding_model = None
    patch_model = None
    fact_model = None
    amem_model = None
    if strategies:
        memory_model_id = (args.memory_model or settings.memory_model or "").strip()
        embedding_model_id = (
            args.embedding_model or settings.embedding_model or ""
        ).strip()
        if not memory_model_id or not embedding_model_id:
            raise RuntimeError(
                "Vehicle memory profiles require memory and embedding models"
            )
        summary_model = OpenAIMemoryModel(
            memory_model_id,
            timeout_seconds=settings.model_timeout_seconds,
            max_output_tokens=settings.memory_max_output_tokens,
            reasoning_effort=settings.memory_reasoning_effort,
            redact_pii=settings.cloud_pii_redaction,
            pii_allowlist=settings.pii_allowlist,
            instructions=VEHICLE_SUMMARY_INSTRUCTIONS,
            prompt_version=VEHICLE_SUMMARY_PROMPT_VERSION,
        )
        if "recursive_summary" in strategies:
            recursive_summary_model = OpenAIRecursiveSummaryMemoryModel(
                memory_model_id,
                timeout_seconds=settings.model_timeout_seconds,
                max_output_tokens=settings.memory_max_output_tokens,
                reasoning_effort=settings.memory_reasoning_effort,
                redact_pii=settings.cloud_pii_redaction,
                pii_allowlist=settings.pii_allowlist,
                instructions=VEHICLE_RECURSIVE_SUMMARY_INSTRUCTIONS,
                prompt_version=(
                    VEHICLE_RECURSIVE_SUMMARY_PROMPT_VERSION
                ),
            )
        structured_model = OpenAIStructuredMemoryModel(
            memory_model_id,
            timeout_seconds=settings.model_timeout_seconds,
            max_output_tokens=settings.memory_max_output_tokens,
            reasoning_effort=settings.memory_reasoning_effort,
            redact_pii=settings.cloud_pii_redaction,
            pii_allowlist=settings.pii_allowlist,
            instructions=VEHICLE_STRUCTURED_INSTRUCTIONS,
            prompt_version=VEHICLE_STRUCTURED_PROMPT_VERSION,
        )
        embedding_model = OpenAIEmbeddingModel(
            embedding_model_id,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.model_timeout_seconds,
            redact_pii=settings.cloud_pii_redaction,
            pii_allowlist=settings.pii_allowlist,
        )
        if {"amem", "amem_style"} & set(strategies):
            amem_model = OpenAIAMemModel(
                memory_model_id,
                timeout_seconds=settings.model_timeout_seconds,
                max_output_tokens=settings.memory_max_output_tokens,
                reasoning_effort=settings.memory_reasoning_effort,
                redact_pii=settings.cloud_pii_redaction,
                pii_allowlist=settings.pii_allowlist,
            )
        if "schema_patch" in strategies:
            patch_model = OpenAIPatchMemoryModel(
                memory_model_id,
                timeout_seconds=settings.model_timeout_seconds,
                max_output_tokens=settings.memory_max_output_tokens,
                reasoning_effort=settings.memory_reasoning_effort,
                redact_pii=settings.cloud_pii_redaction,
                pii_allowlist=settings.pii_allowlist,
                instructions=VEHICLE_PATCH_INSTRUCTIONS,
                prompt_version=VEHICLE_PATCH_PROMPT_VERSION,
            )
        if "fact_patch" in strategies:
            if set(profiles) & {
                "cloud_schema_informed_recursive_assisted_fact_patch",
                "cloud_joint_planned_fact_patch",
            }:
                fact_model = OpenAIPostNormalizedFactMemoryModel(
                    memory_model_id,
                    ontology=load_vehicle_fact_ontology_v1(
                        dataset.tool_schemas
                    ),
                    embedding_model=embedding_model,
                    timeout_seconds=settings.model_timeout_seconds,
                    max_output_tokens=max(
                        settings.memory_max_output_tokens,
                        2_048,
                    ),
                    reasoning_effort=settings.memory_reasoning_effort,
                    redact_pii=settings.cloud_pii_redaction,
                    pii_allowlist=settings.pii_allowlist,
                    instructions=VEHICLE_FACT_INSTRUCTIONS,
                    prompt_version=VEHICLE_FACT_PROMPT_VERSION,
                )
            elif selected_schema_informed_profiles:
                fact_model = OpenAISchemaInformedFactMemoryModel(
                    memory_model_id,
                    ontology=load_vehicle_fact_ontology_v1(
                        dataset.tool_schemas
                    ),
                    embedding_model=embedding_model,
                    timeout_seconds=settings.model_timeout_seconds,
                    max_output_tokens=max(
                        settings.memory_max_output_tokens,
                        2_048,
                    ),
                    reasoning_effort=settings.memory_reasoning_effort,
                    redact_pii=settings.cloud_pii_redaction,
                    pii_allowlist=settings.pii_allowlist,
                    instructions=VEHICLE_SCHEMA_INFORMED_FACT_INSTRUCTIONS,
                    prompt_version=(
                        VEHICLE_SCHEMA_INFORMED_FACT_PROMPT_VERSION
                    ),
                )
            else:
                fact_model = OpenAIFactMemoryModel(
                    memory_model_id,
                    timeout_seconds=settings.model_timeout_seconds,
                    max_output_tokens=max(
                        settings.memory_max_output_tokens,
                        2_048,
                    ),
                    reasoning_effort=settings.memory_reasoning_effort,
                    redact_pii=settings.cloud_pii_redaction,
                    pii_allowlist=settings.pii_allowlist,
                    instructions=VEHICLE_FACT_INSTRUCTIONS,
                    prompt_version=VEHICLE_FACT_PROMPT_VERSION,
                )
    cache_root = args.memory_cache_dir or (settings.data_dir / "vehiclemembench-memory")

    def snapshot_factory(scenario_index: int):
        if not strategies:
            return None
        builder = VehicleMemoryBuilder(
            dataset=dataset,
            scenario_index=scenario_index,
            cache_root=cache_root,
            summary_model=summary_model,
            recursive_summary_model=recursive_summary_model,
            structured_model=structured_model,
            embedding_model=embedding_model,
            patch_model=patch_model,
            fact_model=fact_model,
            amem_model=amem_model,
            batch_token_limit=args.memory_batch_tokens,
            retrieval_top_k=(
                args.memory_top_k
                if args.memory_top_k is not None
                else settings.memory_top_k
            ),
            retrieval_token_budget=(
                args.memory_token_budget
                if args.memory_token_budget is not None
                else settings.memory_context_tokens
            ),
            model_timeout_seconds=settings.model_timeout_seconds,
            patch_user_id=f"vehicle_scenario_{scenario_index}",
            patch_batch_size=(
                args.patch_batch_turns
                if args.patch_batch_turns is not None
                else settings.patch_memory_batch_size
            ),
            patch_batch_token_limit=(
                args.patch_batch_tokens
                if args.patch_batch_tokens is not None
                else settings.patch_memory_batch_tokens
            ),
            patch_max_attempts=settings.patch_memory_max_attempts,
            patch_lease_seconds=max(
                settings.patch_memory_lease_seconds,
                settings.model_timeout_seconds + 1,
            ),
            pii_allowlist=settings.pii_allowlist,
            memory_input_cost_per_million=(settings.memory_input_cost_per_million),
            memory_output_cost_per_million=(settings.memory_output_cost_per_million),
            embedding_input_cost_per_million=(
                settings.embedding_input_cost_per_million
            ),
            semantic_routing_enabled=(
                settings.tool_memory_semantic_routing_enabled
            ),
            oracle_gate_annotations=oracle_gate_annotations,
            oracle_stage_fact_annotations=oracle_stage_fact_annotations,
            oracle_stage=oracle_stage,
            amem_link_candidates=args.amem_link_candidates,
            amem_style_evolution_threshold=(
                args.amem_style_evolution_threshold
            ),
            amem_retrieval_top_k=args.amem_retrieval_top_k,
            amem_note_limit=args.amem_note_limit,
        )
        recursive_assisted_profiles = set(profiles) & {
            "cloud_recursive_assisted_fact_patch",
            "cloud_schema_informed_recursive_assisted_fact_patch",
            "cloud_joint_planned_fact_patch",
        }
        if (
            "cloud_fact_recursive_hybrid" in profiles
            and recursive_assisted_profiles
        ):
            raise RuntimeError(
                "Fact + Recursive Hybrid and Recursive-assisted Fact "
                "require separate runs"
            )
        if "cloud_fact_recursive_hybrid" in profiles:
            unsupported = set(strategies) - {
                "fact_patch",
                "recursive_summary",
            }
            if unsupported:
                raise RuntimeError(
                    "cloud_fact_recursive_hybrid can only be combined with "
                    "Fact-patch and Recursive Summary profiles"
                )
            return builder.build_fact_recursive_hybrid()
        if recursive_assisted_profiles:
            recursive_assisted_profile = next(
                iter(recursive_assisted_profiles)
            )
            unsupported = set(strategies) - {
                "fact_patch",
                "recursive_summary",
            }
            if unsupported:
                raise RuntimeError(
                    f"{recursive_assisted_profile} can only be combined with "
                    "Fact-patch and Recursive Summary profiles"
                )
            assert fact_model is not None

            def assisted_fact_model_factory(
                recursive_summary: str,
            ):
                if recursive_assisted_profile in {
                    "cloud_schema_informed_recursive_assisted_fact_patch",
                    "cloud_joint_planned_fact_patch",
                }:
                    return OpenAIPostNormalizedFactMemoryModel(
                        memory_model_id,
                        ontology=load_vehicle_fact_ontology_v1(
                            dataset.tool_schemas
                        ),
                        embedding_model=embedding_model,
                        timeout_seconds=settings.model_timeout_seconds,
                        max_output_tokens=max(
                            settings.memory_max_output_tokens,
                            2_048,
                        ),
                        reasoning_effort=settings.memory_reasoning_effort,
                        redact_pii=settings.cloud_pii_redaction,
                        pii_allowlist=settings.pii_allowlist,
                        instructions=(
                            VEHICLE_RECURSIVE_ASSISTED_FACT_INSTRUCTIONS
                        ),
                        prompt_version=(
                            VEHICLE_RECURSIVE_ASSISTED_FACT_PROMPT_VERSION
                        ),
                        auxiliary_context={
                            "recursive_summary": recursive_summary,
                            "evidence_policy": (
                                "recall_only_not_citable; every candidate "
                                "must be grounded in source_batch"
                            ),
                        },
                    )
                return OpenAIFactMemoryModel(
                    memory_model_id,
                    timeout_seconds=settings.model_timeout_seconds,
                    max_output_tokens=max(
                        settings.memory_max_output_tokens,
                        2_048,
                    ),
                    reasoning_effort=settings.memory_reasoning_effort,
                    redact_pii=settings.cloud_pii_redaction,
                    pii_allowlist=settings.pii_allowlist,
                    instructions=(
                        VEHICLE_RECURSIVE_ASSISTED_FACT_INSTRUCTIONS
                    ),
                    prompt_version=(
                        VEHICLE_RECURSIVE_ASSISTED_FACT_PROMPT_VERSION
                    ),
                    auxiliary_context={
                        "recursive_summary": recursive_summary,
                        "evidence_policy": (
                            "recall_only_not_citable; every candidate must "
                            "be grounded in source_batch"
                        ),
                    },
                )

            return builder.build_recursive_assisted_fact(
                fact_model_factory=assisted_fact_model_factory,
                profile_name=recursive_assisted_profile,
            )
        return builder.build(strategies=strategies)

    if args.scenario_limit < 1:
        raise ValueError("--scenario-limit must be at least 1")
    if args.scenario_limit > 1:
        settings.ensure_directories()
        with SQLiteRepository(settings.database_path) as repository:
            suite_result = run_vehicle_evaluation_suite(
                dataset,
                repository=repository,
                agent_model=agent_model,
                profile_names=profiles,
                scenario_indices=range(
                    args.scenario,
                    args.scenario + args.scenario_limit,
                ),
                snapshot_factory=snapshot_factory,
                output_root=output_root,
                task_limit=args.task_limit,
                max_tool_rounds=args.max_tool_rounds,
                tool_timeout_seconds=settings.tool_timeout_seconds,
                max_tool_result_chars=settings.max_tool_result_chars,
                resume_run_id=args.resume_run,
                agent_input_cost_per_million=(settings.agent_input_cost_per_million),
                agent_output_cost_per_million=(settings.agent_output_cost_per_million),
                memory_input_cost_per_million=(settings.memory_input_cost_per_million),
                memory_output_cost_per_million=(
                    settings.memory_output_cost_per_million
                ),
                embedding_input_cost_per_million=(
                    settings.embedding_input_cost_per_million
                ),
                memory_configuration=(
                    {
                        "model_id": memory_model_id,
                        "embedding_model_id": embedding_model_id,
                        "embedding_dimensions": settings.embedding_dimensions,
                        "summary_prompt_version": (VEHICLE_SUMMARY_PROMPT_VERSION),
                        "recursive_summary_prompt_version": (
                            VEHICLE_RECURSIVE_SUMMARY_PROMPT_VERSION
                            if recursive_summary_model is not None
                            else None
                        ),
                        "recursive_summary_schema_version": (
                            recursive_summary_model.schema_version
                            if recursive_summary_model is not None
                            else None
                        ),
                        "recursive_summary_max_memory_chars": (
                            recursive_summary_model.max_memory_chars
                            if recursive_summary_model is not None
                            else None
                        ),
                        "structured_prompt_version": (
                            VEHICLE_STRUCTURED_PROMPT_VERSION
                        ),
                        "patch_prompt_version": (
                            VEHICLE_PATCH_PROMPT_VERSION
                            if patch_model is not None
                            else None
                        ),
                        "fact_prompt_version": (
                            fact_model.prompt_version
                            if fact_model is not None
                            else None
                        ),
                        "recursive_assisted_fact_prompt_version": (
                            VEHICLE_RECURSIVE_ASSISTED_FACT_PROMPT_VERSION
                            if set(profiles)
                            & {
                                "cloud_schema_informed_recursive_assisted_fact_patch",
                                "cloud_joint_planned_fact_patch",
                            }
                            else (
                                VEHICLE_RECURSIVE_ASSISTED_FACT_PROMPT_VERSION
                                if "cloud_recursive_assisted_fact_patch"
                                in profiles
                                else None
                            )
                        ),
                        "fact_schema_version": (
                            fact_model.schema_version
                            if fact_model is not None
                            else None
                        ),
                        "batch_tokens": args.memory_batch_tokens,
                        "patch_batch_turns": (
                            args.patch_batch_turns
                            if args.patch_batch_turns is not None
                            else settings.patch_memory_batch_size
                        ),
                        "patch_batch_tokens": (
                            args.patch_batch_tokens
                            if args.patch_batch_tokens is not None
                            else settings.patch_memory_batch_tokens
                        ),
                        "pii_redaction": settings.cloud_pii_redaction,
                    }
                    if strategies
                    else None
                ),
                oracle_retrieval_annotations=(
                    oracle_retrieval_annotations
                ),
                oracle_gate_annotations=oracle_gate_annotations,
                oracle_stage_fact_annotations=(
                    oracle_stage_fact_annotations
                ),
            )
        print(json.dumps(suite_result.as_dict(), ensure_ascii=False, indent=2))
        return int(suite_result.status != "completed")

    snapshot = snapshot_factory(args.scenario)
    calls_before = snapshot.provider_calls() if snapshot is not None else []
    try:
        result = run_agent_evaluation(
            dataset,
            agent_model=agent_model,
            profile_names=profiles,
            scenario_index=args.scenario,
            task_limit=args.task_limit,
            max_tool_rounds=args.max_tool_rounds,
            tool_timeout_seconds=settings.tool_timeout_seconds,
            max_tool_result_chars=settings.max_tool_result_chars,
            output_root=output_root,
            memory_resolver=(snapshot.resolve if snapshot is not None else None),
            memory_manifest=(snapshot.manifest if snapshot is not None else None),
            oracle_retrieval_annotations=oracle_retrieval_annotations,
            oracle_gate_annotations=oracle_gate_annotations,
            oracle_stage_fact_annotations=oracle_stage_fact_annotations,
            resume_run_id=args.resume_run,
            agent_input_cost_per_million=(settings.agent_input_cost_per_million),
            agent_output_cost_per_million=(settings.agent_output_cost_per_million),
        )
        if snapshot is not None:
            provider_usage = summarize_provider_calls(
                calls_before,
                snapshot.provider_calls(),
                memory_input_cost_per_million=(
                    settings.memory_input_cost_per_million
                ),
                memory_output_cost_per_million=(
                    settings.memory_output_cost_per_million
                ),
                embedding_input_cost_per_million=(
                    settings.embedding_input_cost_per_million
                ),
            )
            result.metrics["provider_usage"] = provider_usage
            amem_usage = getattr(snapshot, "amem_usage", lambda: None)()
            if amem_usage is not None:
                result.metrics["amem_usage"] = amem_usage
            result.metrics["estimated_total_cost_usd"] = round(
                provider_usage["estimated_cost_usd"]
                + (
                    float(amem_usage["build_estimated_cost_usd"])
                    if amem_usage is not None
                    else 0.0
                )
                + sum(
                    float(values["estimated_agent_cost_usd"])
                    for values in result.metrics["profiles"].values()
                ),
                8,
            )
            if result.artifact_dir is not None:
                write_vehicle_report_artifacts(
                    result.artifact_dir,
                    metrics=result.metrics,
                    records=result.tasks,
                )
    finally:
        if snapshot is not None:
            snapshot.close()
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return int(result.status != "completed")


def _evaluation_vehicle_matcher(args: argparse.Namespace) -> int:
    from palmclaw_ubuntu.providers import OpenAIEmbeddingModel
    from palmclaw_ubuntu.vehicle_bench import (
        OFFICIAL_UPSTREAM_COMMIT,
        default_vehicle_ontology_matcher_labels,
        load_vehicle_benchmark,
        run_vehicle_ontology_matcher_evaluation,
    )

    if args.allow_unpinned and args.expected_commit:
        raise ValueError("--allow-unpinned and --expected-commit cannot be combined")
    expected_commit = None
    if not args.allow_unpinned:
        expected_commit = args.expected_commit or OFFICIAL_UPSTREAM_COMMIT
    dataset = load_vehicle_benchmark(
        args.benchmark_root,
        expected_commit=expected_commit,
    )
    settings = _settings(args)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    embedding_model_id = (
        args.embedding_model or settings.embedding_model or ""
    ).strip()
    if not embedding_model_id:
        raise RuntimeError("Vehicle matcher evaluation requires an embedding model")
    embedding_model = OpenAIEmbeddingModel(
        embedding_model_id,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.model_timeout_seconds,
        redact_pii=settings.cloud_pii_redaction,
        pii_allowlist=settings.pii_allowlist,
    )
    output_root = args.output_dir or (
        Path(__file__).resolve().parents[2]
        / "evaluation"
        / "vehiclemembench-ontology-matcher"
    )
    result = run_vehicle_ontology_matcher_evaluation(
        dataset,
        embedding_model=embedding_model,
        scenario_index=args.scenario,
        labels_path=args.labels or default_vehicle_ontology_matcher_labels(),
        batch_turn_limit=(
            args.batch_turns
            if args.batch_turns is not None
            else settings.patch_memory_batch_size
        ),
        batch_token_limit=(
            args.batch_tokens
            if args.batch_tokens is not None
            else settings.patch_memory_batch_tokens
        ),
        output_root=output_root,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return int(result.status != "completed")


def _evaluation_vehicle_oracle_audit(args: argparse.Namespace) -> int:
    from palmclaw_ubuntu.vehicle_bench import (
        OFFICIAL_UPSTREAM_COMMIT,
        load_vehicle_benchmark,
        run_oracle_contract_audit,
    )

    if args.allow_unpinned and args.expected_commit:
        raise ValueError("--allow-unpinned and --expected-commit cannot be combined")
    if args.scenario_limit < 1:
        raise ValueError("--scenario-limit must be at least 1")
    expected_commit = None
    if not args.allow_unpinned:
        expected_commit = args.expected_commit or OFFICIAL_UPSTREAM_COMMIT
    dataset = load_vehicle_benchmark(
        args.benchmark_root,
        expected_commit=expected_commit,
    )
    output_root = args.output_dir or (
        Path(__file__).resolve().parents[2]
        / "evaluation"
        / "vehiclemembench-oracle"
    )
    result = run_oracle_contract_audit(
        dataset,
        scenario_indices=range(
            args.scenario,
            args.scenario + args.scenario_limit,
        ),
        task_limit=args.task_limit,
        baseline_run_dir=args.baseline_run_dir,
        output_root=output_root,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return int(result.status != "passed")


def _evaluation_list(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        runs = repository.list_evaluation_runs(limit=max(1, args.limit))
    print(json.dumps(runs, ensure_ascii=False, indent=2))
    return 0


def _evaluation_vehicle_r0(args: argparse.Namespace) -> int:
    from palmclaw_ubuntu.vehicle_bench.diagnostics import (
        run_vehicle_r0_diagnostics,
    )

    result = run_vehicle_r0_diagnostics(
        args.run_dir,
        memory_cache_dir=args.memory_cache_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


def _evaluation_show(args: argparse.Namespace) -> int:
    with _repository(args) as repository:
        detail = repository.evaluation_detail(args.run_id)
    print(json.dumps(detail, ensure_ascii=False, indent=2))
    return 0


def _skill_list(args: argparse.Namespace) -> int:
    settings = _settings(args)
    loader = SkillsLoader(
        settings.builtin_skills_root,
        settings.workspace_skills_root,
    )
    for skill in loader.list_skills():
        print(
            f"{skill.name}\t{skill.source}\t"
            f"always={str(skill.always).lower()}\t{skill.description}"
        )
    return 0


def _tool_list(args: argparse.Namespace) -> int:
    settings = _settings(args)
    original_backend = settings.backend
    original_memory_backend = os.getenv("PALMCLAW_MEMORY_BACKEND")
    os.environ["PALMCLAW_BACKEND"] = "fake"
    os.environ["PALMCLAW_MEMORY_BACKEND"] = "fake"
    try:
        fake_settings = Settings.from_env(data_dir=settings.data_dir)
        with create_runtime(fake_settings) as runtime:
            for definition in runtime.tool_registry.definitions():
                print(f"{definition.name}\t{definition.description}")
    finally:
        os.environ["PALMCLAW_BACKEND"] = original_backend
        if original_memory_backend is None:
            os.environ.pop("PALMCLAW_MEMORY_BACKEND", None)
        else:
            os.environ["PALMCLAW_MEMORY_BACKEND"] = original_memory_backend
    return 0
