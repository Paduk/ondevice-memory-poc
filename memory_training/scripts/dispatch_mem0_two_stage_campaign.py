"""Dispatch queued Mem0 two-stage pipelines onto stably idle GPUs."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path("/mnt/data/hj153lee/PalmClaw/on-device-memory-training")
RUNNER = Path(
    "/home/hj153lee/PalmClaw/memory_training/scripts/run_mem0_two_stage_train_eval.sh"
)
LLAMA_RECOVERY_RUNNER = Path(
    "/home/hj153lee/PalmClaw/memory_training/scripts/"
    "run_llama32_mem0_one_pass_recovery_sequence.sh"
)
LLAMA_RECOVERY_SESSION = "llama32-mem0-onepass-recovery"
STATE_PATH = WORKSPACE / "logs" / "mem0-two-stage-dispatcher-state.json"
LOG_PATH = WORKSPACE / "logs" / "mem0-two-stage-dispatcher.log"
POLL_SECONDS = 60
FREE_MEMORY_LIMIT_MIB = 512
FREE_UTILIZATION_LIMIT = 5
FREE_CONFIRMATIONS = 2
CLEANUP_TRIGGER_BYTES = 15 * 1024**3
DISPATCH_PRIORITY = {
    "qwen2b": 0,
    "llama3b": 1,
    "qwen08b": 2,
    "llama1b": 3,
    "granite350m": 4,
}

JOBS = (
    {
        "key": "qwen2b",
        "model": "qwen3.5-2b",
        "one_pass": "qwen35-2b-mem0-one-pass-multitask-noop5-e4-b4-trainseed45-r1",
        "two_stage": "qwen35-2b-mem0-two-stage-multitask-extract1to5-managerall-e4-b4-trainseed45-r1",
    },
    {
        "key": "qwen08b",
        "model": "qwen3.5-0.8b",
        "one_pass": "qwen35-0.8b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1",
        "two_stage": "qwen35-0.8b-mem0-two-stage-multitask-extract1to5-managerall-e4-b8-trainseed45-r1",
    },
    {
        "key": "llama3b",
        "model": "llama3.2-3b",
        "one_pass": "llama3.2-3b-mem0-one-pass-multitask-noop5-e4-b4-trainseed45-r1",
        "two_stage": "llama3.2-3b-mem0-two-stage-multitask-extract1to5-managerall-e4-b4-trainseed45-r1",
    },
    {
        "key": "granite350m",
        "model": "granite4-350m",
        "one_pass": "granite4-350m-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1",
        "two_stage": "granite4-350m-mem0-two-stage-multitask-extract1to5-managerall-e4-b8-trainseed45-r1",
    },
    {
        "key": "llama1b",
        "model": "llama3.2-1b",
        "one_pass": "llama3.2-1b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1",
        "two_stage": "llama3.2-1b-mem0-two-stage-multitask-extract1to5-managerall-e4-b8-trainseed45-r1",
    },
)

ONE_PASS_RESERVATIONS = {
    2: "granite350m-mem0-onepass-gpu2",
    3: "qwen2b-mem0-onepass-gpu3",
    4: "llama32-mem0-onepass-sequence-gpu4",
    5: "qwen08b-mem0-onepass-gpu5",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def write_state(value: dict[str, Any]) -> None:
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, STATE_PATH)


def run_is_active(run_id: str) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if run_id in command and any(
            token in command
            for token in (
                "memory_training.train",
                "memory_training.validate_hf_checkpoint_v2",
                "memory_training.evaluate_hf_closed_loop",
            )
        ):
            return True
    return False


def cleanup_old_k_checkpoints() -> dict[str, Any] | None:
    """Remove only authorized inactive K=2/5/10 epoch-01/02 checkpoints."""
    usage_before = shutil.disk_usage(WORKSPACE)
    if usage_before.free >= CLEANUP_TRIGGER_BYTES:
        return None
    runs_root = (WORKSPACE / "runs").resolve()
    removed: list[str] = []
    skipped: list[dict[str, str]] = []
    pattern = re.compile(r".*delta_v3_compact_k(?:2|5|10)-.*")
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or not pattern.fullmatch(run_dir.name):
            continue
        active = run_is_active(run_dir.name)
        best = read_json(run_dir / "best-checkpoint.json").get("checkpoint")
        best_path = Path(best).resolve() if isinstance(best, str) else None
        for epoch in ("epoch-01", "epoch-02"):
            target = (run_dir / "checkpoints" / epoch).resolve()
            if not target.is_dir():
                continue
            if runs_root not in target.parents or target.name not in {"epoch-01", "epoch-02"}:
                raise RuntimeError(f"Unsafe cleanup target rejected: {target}")
            if active:
                skipped.append({"path": str(target), "reason": "active_run"})
                continue
            if best_path == target:
                skipped.append({"path": str(target), "reason": "best_checkpoint"})
                continue
            shutil.rmtree(target)
            removed.append(str(target))
            log(f"Disk guard removed authorized old checkpoint: {target}")
    usage_after = shutil.disk_usage(WORKSPACE)
    return {
        "timestamp": utc_now(),
        "trigger_free_bytes": usage_before.free,
        "free_bytes_after": usage_after.free,
        "reclaimed_bytes": usage_after.free - usage_before.free,
        "removed": removed,
        "skipped": skipped,
    }


def tmux_session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def evaluation_complete(run_id: str) -> bool:
    run_dir = WORKSPACE / "runs" / run_id
    best = read_json(run_dir / "best-checkpoint.json")
    checkpoint = best.get("checkpoint")
    if not isinstance(checkpoint, str):
        return False
    epoch = Path(checkpoint).name
    validation = run_dir / f"eval-fixed-v2-validation-best-{epoch}.json"
    summary = read_json(run_dir / f"eval-fixed-v2-test-best-{epoch}" / "summary.json")
    expected = summary.get("expected_scenarios")
    completed = summary.get("completed_scenarios")
    return (
        validation.stat().st_size > 0
        if validation.exists()
        else False
    ) and (
        summary.get("complete") is True
        and isinstance(expected, list)
        and isinstance(completed, list)
        and len(expected) == 24
        and len(completed) == 24
    )


def gpu_snapshot() -> dict[int, dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        log(f"nvidia-smi failed: {result.stderr.strip()}")
        return {}
    values: dict[int, dict[str, int]] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            index, memory, utilization = map(int, fields)
        except ValueError:
            continue
        values[index] = {"memory_used_mib": memory, "utilization": utilization}
    return values


def session_name(job: dict[str, str]) -> str:
    return f"mem0-ts-{job['key']}"


def main() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE_PATH)
    assignments = state.get("assignments")
    if not isinstance(assignments, dict):
        assignments = {}
    cleanup_history = state.get("cleanup_history")
    if not isinstance(cleanup_history, list):
        cleanup_history = []
    llama_recovery_assignment = state.get("llama_recovery_assignment")
    if not isinstance(llama_recovery_assignment, dict):
        llama_recovery_assignment = {}
    free_counts = {index: 0 for index in range(8)}
    log("Dynamic Two-stage dispatcher started.")

    while True:
        cleanup = cleanup_old_k_checkpoints()
        if cleanup is not None:
            cleanup_history.append(cleanup)
        gpu_values = gpu_snapshot()
        reserved = {
            index
            for index, name in ONE_PASS_RESERVATIONS.items()
            if tmux_session_exists(name)
        }
        active_gpu_by_job: dict[str, int] = {}
        job_states: dict[str, dict[str, Any]] = {}

        for job in JOBS:
            key = job["key"]
            assignment = assignments.get(key, {})
            session = session_name(job)
            if evaluation_complete(job["two_stage"]):
                stage = "COMPLETED"
            elif tmux_session_exists(session):
                stage = "RUNNING"
                gpu = assignment.get("gpu")
                if isinstance(gpu, int):
                    active_gpu_by_job[key] = gpu
            elif assignment:
                run_status = read_json(
                    WORKSPACE / "runs" / job["two_stage"] / "status.json"
                )
                if run_status.get("state") == "COMPLETED":
                    log(
                        f"Requeueing post-training evaluation for {job['model']} "
                        "after its session exited incomplete."
                    )
                    assignments.pop(key, None)
                    assignment = {}
                    stage = "READY"
                elif run_status.get("state") == "FAILED":
                    log(
                        f"Requeueing failed pipeline for {job['model']} on a "
                        "newly confirmed idle GPU."
                    )
                    assignments.pop(key, None)
                    assignment = {}
                    stage = "READY"
                else:
                    stage = "FAILED_OR_EXITED"
            else:
                # Two-stage starts from the same base model and is independent of
                # the One-pass checkpoint, so an idle unreserved GPU can run it
                # immediately.  The One-pass completion bit remains diagnostic.
                stage = "READY"
            job_states[key] = {
                "model": job["model"],
                "stage": stage,
                "one_pass_complete": evaluation_complete(job["one_pass"]),
                "two_stage_complete": evaluation_complete(job["two_stage"]),
                "assignment": assignment or None,
                "session": session,
            }

        llama3_job = next(job for job in JOBS if job["key"] == "llama3b")
        llama1_job = next(job for job in JOBS if job["key"] == "llama1b")
        llama_recovery_complete = evaluation_complete(
            llama3_job["one_pass"]
        ) and evaluation_complete(llama1_job["one_pass"])
        original_llama_sequence = tmux_session_exists(ONE_PASS_RESERVATIONS[4])
        recovery_session_active = tmux_session_exists(LLAMA_RECOVERY_SESSION)
        if llama_recovery_complete:
            llama_recovery_stage = "COMPLETED"
        elif recovery_session_active:
            llama_recovery_stage = "RUNNING"
        elif original_llama_sequence:
            llama_recovery_stage = "ORIGINAL_SEQUENCE_RUNNING"
        elif llama_recovery_assignment:
            llama_recovery_stage = "FAILED_OR_EXITED"
        else:
            llama_recovery_stage = "READY"

        occupied_by_campaign = set(active_gpu_by_job.values())
        recovery_gpu = llama_recovery_assignment.get("gpu")
        if recovery_session_active and isinstance(recovery_gpu, int):
            occupied_by_campaign.add(recovery_gpu)
        for index in range(8):
            gpu = gpu_values.get(index)
            idle = bool(
                gpu
                and gpu["memory_used_mib"] <= FREE_MEMORY_LIMIT_MIB
                and gpu["utilization"] <= FREE_UTILIZATION_LIMIT
                and index not in reserved
                and index not in occupied_by_campaign
            )
            free_counts[index] = free_counts[index] + 1 if idle else 0

        free_gpus = [
            index
            for index in range(8)
            if free_counts[index] >= FREE_CONFIRMATIONS
        ]
        if llama_recovery_stage == "READY" and free_gpus:
            gpu = free_gpus.pop(0)
            command = f"bash {LLAMA_RECOVERY_RUNNER} {gpu}"
            result = subprocess.run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    LLAMA_RECOVERY_SESSION,
                    command,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                log(
                    f"Failed to dispatch Llama One-pass recovery on GPU {gpu}: "
                    f"{result.stderr.strip()}"
                )
            else:
                llama_recovery_assignment = {
                    "gpu": gpu,
                    "assigned_at": utc_now(),
                    "session": LLAMA_RECOVERY_SESSION,
                }
                llama_recovery_stage = "RUNNING"
                free_counts[gpu] = 0
                log(f"Dispatched Llama One-pass recovery on GPU {gpu}.")
        ready_jobs = sorted(
            (
                job
                for job in JOBS
                if job_states[job["key"]]["stage"] == "READY"
            ),
            key=lambda job: DISPATCH_PRIORITY[job["key"]],
        )
        for job, gpu in zip(ready_jobs, free_gpus, strict=False):
            session = session_name(job)
            command = f"bash {RUNNER} {gpu} {job['model']}"
            result = subprocess.run(
                ["tmux", "new-session", "-d", "-s", session, command],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                log(
                    f"Failed to dispatch {job['model']} on GPU {gpu}: "
                    f"{result.stderr.strip()}"
                )
                continue
            assignments[job["key"]] = {
                "gpu": gpu,
                "assigned_at": utc_now(),
                "session": session,
            }
            free_counts[gpu] = 0
            log(f"Dispatched {job['model']} Two-stage on GPU {gpu} ({session}).")

        payload = {
            "updated_at": utc_now(),
            "dispatcher_pid": os.getpid(),
            "reserved_one_pass_gpus": sorted(reserved),
            "stable_free_gpus": free_gpus,
            "gpu_snapshot": gpu_values,
            "assignments": assignments,
            "llama_recovery_assignment": llama_recovery_assignment,
            "llama_recovery": {
                "stage": llama_recovery_stage,
                "session": LLAMA_RECOVERY_SESSION,
                "original_sequence_active": original_llama_sequence,
                "llama3_one_pass_complete": evaluation_complete(
                    llama3_job["one_pass"]
                ),
                "llama1_one_pass_complete": evaluation_complete(
                    llama1_job["one_pass"]
                ),
            },
            "cleanup_history": cleanup_history,
            "jobs": job_states,
        }
        write_state(payload)
        if llama_recovery_complete and all(
            evaluation_complete(job["two_stage"]) for job in JOBS
        ):
            log("All One-pass recovery and Two-stage pipelines completed.")
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
