"""Snapshot the full One-pass/Two-stage campaign every fixed interval."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_training.scripts.dispatch_mem0_two_stage_campaign import JOBS, STATE_PATH


WORKSPACE = Path("/mnt/data/hj153lee/PalmClaw/on-device-memory-training")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def matching_pids(run_id: str) -> list[int]:
    matches: list[int] = []
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
            matches.append(int(entry.name))
    return sorted(matches)


def evaluation(run_id: str) -> dict[str, Any]:
    run_dir = WORKSPACE / "runs" / run_id
    best = read_json(run_dir / "best-checkpoint.json")
    checkpoint = best.get("checkpoint")
    epoch = Path(checkpoint).name if isinstance(checkpoint, str) else None
    validation = (
        run_dir / f"eval-fixed-v2-validation-best-{epoch}.json" if epoch else None
    )
    summary_path = (
        run_dir / f"eval-fixed-v2-test-best-{epoch}" / "summary.json"
        if epoch
        else None
    )
    summary = read_json(summary_path) if summary_path else {}
    expected = summary.get("expected_scenarios")
    completed = summary.get("completed_scenarios")
    return {
        "best_epoch": epoch,
        "validation_complete": bool(validation and validation.exists() and validation.stat().st_size),
        "test_complete": summary.get("complete") is True,
        "test_completed_scenarios": len(completed) if isinstance(completed, list) else 0,
        "test_expected_scenarios": len(expected) if isinstance(expected, list) else 0,
    }


def gpu_snapshot() -> dict[str, Any]:
    fields = (
        "index,memory.used,memory.free,utilization.gpu,utilization.memory,"
        "power.draw,temperature.gpu"
    )
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return {"error": result.stderr.strip()}
    names = (
        "index",
        "memory_used_mib",
        "memory_free_mib",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "power_watts",
        "temperature_c",
    )
    output: dict[str, Any] = {}
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(names):
            row = dict(zip(names, values, strict=True))
            output[row["index"]] = row
    return output


def snapshot(previous: dict[str, int | None]) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    alerts: list[str] = []
    for job in JOBS:
        for method, run_id in (("one_pass", job["one_pass"]), ("two_stage", job["two_stage"])):
            label = f"{job['key']}:{method}"
            run_dir = WORKSPACE / "runs" / run_id
            status = read_json(run_dir / "status.json")
            progress = evaluation(run_id)
            pids = matching_pids(run_id)
            step = status.get("global_step")
            state = status.get("state", "NOT_STARTED")
            if progress["test_complete"]:
                stage = "COMPLETED"
            elif pids and state == "RUNNING":
                stage = "TRAINING"
            elif pids:
                stage = "EVALUATING"
            elif state == "COMPLETED":
                stage = "POST_TRAINING"
            else:
                stage = state
            runs[label] = {
                "run_id": run_id,
                "state": state,
                "stage": stage,
                "global_step": step,
                "epoch": status.get("epoch"),
                "status_updated_at": status.get("updated_at"),
                "pids": pids,
                **progress,
            }
            if state == "FAILED":
                alerts.append(f"{label}:FAILED")
            if state == "RUNNING" and not pids:
                alerts.append(f"{label}:RUNNING_WITHOUT_PROCESS")
            if (
                state == "RUNNING"
                and isinstance(step, int)
                and previous.get(label) == step
            ):
                alerts.append(f"{label}:NO_STEP_PROGRESS_30M")
            if state == "COMPLETED" and not progress["test_complete"] and not pids:
                alerts.append(f"{label}:PIPELINE_EXITED_BEFORE_TEST_COMPLETE")
            previous[label] = step if isinstance(step, int) else None

    dispatcher = read_json(STATE_PATH)
    dispatcher_jobs = dispatcher.get("jobs")
    if isinstance(dispatcher_jobs, dict):
        for key, job_state in dispatcher_jobs.items():
            if (
                isinstance(job_state, dict)
                and job_state.get("stage") == "FAILED_OR_EXITED"
            ):
                alerts.append(f"{key}:TWO_STAGE_SESSION_EXITED_INCOMPLETE")
    llama_recovery = dispatcher.get("llama_recovery")
    if (
        isinstance(llama_recovery, dict)
        and llama_recovery.get("stage") == "FAILED_OR_EXITED"
    ):
        alerts.append("LLAMA_ONE_PASS_RECOVERY_EXITED_INCOMPLETE")
    dispatcher_pid = dispatcher.get("dispatcher_pid")
    dispatcher_alive = bool(
        isinstance(dispatcher_pid, int) and Path(f"/proc/{dispatcher_pid}").exists()
    )
    if not dispatcher_alive and not all(
        runs[f"{job['key']}:two_stage"]["test_complete"] for job in JOBS
    ):
        alerts.append("TWO_STAGE_DISPATCHER_NOT_RUNNING")
    disk = shutil.disk_usage(WORKSPACE)
    if disk.free < 10 * 1024**3:
        alerts.append("WORKSPACE_DISK_FREE_BELOW_10_GIB")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "gpus": gpu_snapshot(),
        "dispatcher": {**dispatcher, "alive": dispatcher_alive},
        "workspace_disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "alerts": alerts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--duration-seconds", type=int, default=43200)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = math.floor(args.duration_seconds / args.interval_seconds) + 1
    previous: dict[str, int | None] = {}
    started = time.monotonic()
    for index in range(samples):
        payload = {
            "sample_index": index,
            "sample_count": samples,
            **snapshot(previous),
        }
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if index + 1 < samples:
            deadline = started + (index + 1) * args.interval_seconds
            time.sleep(max(0.0, deadline - time.monotonic()))


if __name__ == "__main__":
    main()
