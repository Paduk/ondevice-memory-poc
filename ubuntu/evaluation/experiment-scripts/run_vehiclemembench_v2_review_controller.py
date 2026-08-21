#!/usr/bin/env python3
"""Register paused V2 runs and resume them after Human Review."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.consensus_gold_pipeline import GoldCheckpointStore
from palmclaw_ubuntu.vehicle_bench.dataset import load_vehicle_benchmark
from palmclaw_ubuntu.vehicle_bench.human_review import HumanReviewQueue
from palmclaw_ubuntu.vehicle_bench.memory import (
    VehicleHistoryEntry,
    parse_vehicle_history,
)

DEFAULT_PROJECT_ROOT = Path("/home/hj153lee/PalmClaw/ubuntu")
DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2/consensus-gold-v2"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--session-prefix", default="vehiclemem_v2_s")
    parser.add_argument("--luna-model", default="gpt-5.6-luna")
    parser.add_argument("--terra-model", default="gpt-5.6-terra")
    parser.add_argument("--sol-model", default="gpt-5.6-sol")
    parser.add_argument("--qa-coverage-model", default="gpt-5.6-sol")
    parser.add_argument(
        "--memory-policy",
        choices=("temporal", "patch-r2"),
        default="temporal",
    )
    parser.add_argument("--compaction-add-threshold", type=int, default=30)
    parser.add_argument("--compaction-target-ratio", type=float, default=0.70)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    project_root = args.project_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    queue_path = (
        args.queue.expanduser().resolve()
        if args.queue is not None
        else output_root / "review_queue.sqlite"
    )
    queue = HumanReviewQueue(queue_path)
    dataset = load_vehicle_benchmark(dataset_root, strict=True)
    entry_cache: dict[int, tuple[VehicleHistoryEntry, ...]] = {}
    known_review_ids = {record.review_id for record in queue.list()}

    print(f"Human Review controller: {queue_path}", flush=True)
    while True:
        _register_paused_runs(
            queue=queue,
            output_root=output_root,
            dataset=dataset,
            entry_cache=entry_cache,
            known_review_ids=known_review_ids,
            session_prefix=args.session_prefix,
        )
        _launch_submitted_runs(
            queue=queue,
            project_root=project_root,
            output_root=output_root,
            session_prefix=args.session_prefix,
            luna_model=args.luna_model,
            terra_model=args.terra_model,
            sol_model=args.sol_model,
            qa_coverage_model=args.qa_coverage_model,
            memory_policy=args.memory_policy,
            compaction_add_threshold=args.compaction_add_threshold,
            compaction_target_ratio=args.compaction_target_ratio,
        )
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


def _register_paused_runs(
    *,
    queue: HumanReviewQueue,
    output_root: Path,
    dataset,
    entry_cache: dict[int, tuple[VehicleHistoryEntry, ...]],
    known_review_ids: set[str],
    session_prefix: str,
) -> None:
    for scenario_dir in sorted(output_root.glob("scenario-*")):
        checkpoint_path = scenario_dir / "checkpoint.sqlite"
        if not checkpoint_path.is_file():
            continue
        try:
            scenario_index = int(scenario_dir.name.removeprefix("scenario-"))
            if _tmux_pane_state(f"{session_prefix}{scenario_index}") == "RUNNING":
                continue
            entries = entry_cache.get(scenario_index)
            if entries is None:
                scenario = dataset.scenario(scenario_index)
                entries = tuple(
                    sorted(
                        parse_vehicle_history(scenario.history_path),
                        key=lambda entry: (entry.timestamp, entry.line_number),
                    )
                )
                entry_cache[scenario_index] = entries
            store = GoldCheckpointStore(
                checkpoint_path,
                scenario_index=scenario_index,
                entries=entries,
            )
            state = store.state()
            if state.status == "PAUSED_REVIEW":
                record = queue.register_paused(
                    checkpoint_path=checkpoint_path,
                    state=state,
                    entry=entries[state.next_turn_index],
                )
                if record.review_id not in known_review_ids:
                    print(f"Pending: {record.review_id}", flush=True)
                    known_review_ids.add(record.review_id)
        except (KeyError, RuntimeError, ValueError) as error:
            print(f"Could not register {checkpoint_path}: {error}", flush=True)


def _launch_submitted_runs(
    *,
    queue: HumanReviewQueue,
    project_root: Path,
    output_root: Path,
    session_prefix: str,
    luna_model: str,
    terra_model: str,
    sol_model: str,
    qa_coverage_model: str,
    memory_policy: str,
    compaction_add_threshold: int,
    compaction_target_ratio: float,
) -> None:
    launched_scenarios: set[int] = set()
    for record in queue.list(status="SUBMITTED"):
        scenario_index = record.scenario_index
        if scenario_index in launched_scenarios:
            continue
        session = f"{session_prefix}{scenario_index}"
        pane_state = _tmux_pane_state(session)
        if pane_state == "RUNNING":
            continue
        command = _runner_command(
            project_root=project_root,
            output_root=output_root,
            queue_path=queue.path,
            scenario_index=scenario_index,
            luna_model=luna_model,
            terra_model=terra_model,
            sol_model=sol_model,
            qa_coverage_model=qa_coverage_model,
            memory_policy=memory_policy,
            compaction_add_threshold=compaction_add_threshold,
            compaction_target_ratio=compaction_target_ratio,
        )
        if pane_state == "DEAD":
            subprocess.run(
                ["tmux", "respawn-pane", "-k", "-t", session, command],
                check=True,
            )
        else:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session, command],
                check=True,
            )
            subprocess.run(
                ["tmux", "set-option", "-t", session, "remain-on-exit", "on"],
                check=True,
            )
        launched_scenarios.add(scenario_index)
        print(f"Resumed S{scenario_index}: {record.review_id}", flush=True)


def _tmux_pane_state(session: str) -> str:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", session, "-F", "#{pane_dead}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "MISSING"
    return "DEAD" if result.stdout.strip() == "1" else "RUNNING"


def _runner_command(
    *,
    project_root: Path,
    output_root: Path,
    queue_path: Path,
    scenario_index: int,
    luna_model: str,
    terra_model: str,
    sol_model: str,
    qa_coverage_model: str,
    memory_policy: str,
    compaction_add_threshold: int,
    compaction_target_ratio: float,
) -> str:
    runner = (
        project_root
        / "evaluation/experiment-scripts/run_vehiclemembench_v2_consensus_gold_s1.py"
    )
    log_path = output_root / f"scenario-{scenario_index:02d}" / "runner.log"
    arguments = [
        sys.executable,
        str(runner),
        "--scenario",
        str(scenario_index),
        "--output-root",
        str(output_root),
        "--review-queue",
        str(queue_path),
        "--luna-model",
        luna_model,
        "--terra-model",
        terra_model,
        "--sol-model",
        sol_model,
        "--qa-coverage-model",
        qa_coverage_model,
        "--memory-policy",
        memory_policy,
        "--compaction-add-threshold",
        str(compaction_add_threshold),
        "--compaction-target-ratio",
        str(compaction_target_ratio),
        "--execute",
        "--run-all",
    ]
    return (
        f"cd {shlex.quote(str(project_root))} && "
        f"{shlex.join(arguments)} >> {shlex.quote(str(log_path))} 2>&1"
    )


if __name__ == "__main__":
    raise SystemExit(main())
