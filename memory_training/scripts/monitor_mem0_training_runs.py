"""Periodically snapshot Mem0 training progress and GPU health."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path("/mnt/data/hj153lee/PalmClaw/on-device-memory-training")
RUNS = {
    "one_pass": (
        "granite4-1b-mem0-one-pass-multitask-noop5-e4-b8-trainseed45-r1",
        5,
    ),
    "two_stage": (
        "granite4-1b-mem0-two-stage-multitask-extract1to5-managerall-e4-b8-trainseed45-r1",
        4,
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _last_jsonl(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            buffer = b""
            while position > 0 and b"\n" not in buffer.rstrip(b"\n"):
                size = min(8192, position)
                position -= size
                handle.seek(position)
                buffer = handle.read(size) + buffer
        lines = buffer.splitlines()
        return json.loads(lines[-1]) if lines else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _matching_pids(run_id: str, commands: tuple[str, ...]) -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if run_id in command and any(value in command for value in commands):
            matches.append(int(entry.name))
    return sorted(matches)


def _training_pids(run_id: str) -> list[int]:
    return _matching_pids(run_id, ("memory_training.train",))


def _pipeline_pids(run_id: str) -> list[int]:
    return _matching_pids(
        run_id,
        (
            "memory_training.train",
            "memory_training.validate_hf_checkpoint_v2",
            "memory_training.evaluate_hf_closed_loop",
        ),
    )


def _evaluation_progress(root: Path) -> dict[str, Any]:
    validation_files = sorted(root.glob("eval-fixed-v2-validation-best-epoch-*.json"))
    test_files = sorted(root.glob("eval-fixed-v2-test-best-epoch-*/summary.json"))
    validation = _read_json(validation_files[-1]) if validation_files else {}
    test = _read_json(test_files[-1]) if test_files else {}
    completed = test.get("completed_scenarios", [])
    expected = test.get("expected_scenarios", [])
    return {
        "validation_artifact": str(validation_files[-1]) if validation_files else None,
        "validation_complete": bool(validation),
        "test_artifact": str(test_files[-1]) if test_files else None,
        "test_complete": test.get("complete") is True,
        "test_completed_scenarios": len(completed) if isinstance(completed, list) else 0,
        "test_expected_scenarios": len(expected) if isinstance(expected, list) else 0,
    }


def _gpu_snapshot(index: int) -> dict[str, Any]:
    fields = (
        "index,memory.used,memory.free,utilization.gpu,utilization.memory,"
        "power.draw,temperature.gpu"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return {"index": index, "error": result.stderr.strip()}
    names = [
        "index",
        "memory_used_mib",
        "memory_free_mib",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "power_watts",
        "temperature_c",
    ]
    values = [value.strip() for value in result.stdout.strip().split(",")]
    return dict(zip(names, values, strict=True))


def _snapshot(
    previous_steps: dict[str, int | None],
    runs_config: dict[str, tuple[str, int]],
) -> dict[str, Any]:
    runs = {}
    alerts = []
    for label, (run_id, gpu) in runs_config.items():
        root = WORKSPACE / "runs" / run_id
        status = _read_json(root / "status.json")
        metric = _last_jsonl(root / "metrics.jsonl")
        step = status.get("global_step", metric.get("global_step"))
        pids = _training_pids(run_id)
        pipeline_pids = _pipeline_pids(run_id)
        evaluation = _evaluation_progress(root)
        state = status.get("state", "UNKNOWN")
        if evaluation["test_complete"]:
            stage = "COMPLETED"
        elif pids:
            stage = "TRAINING"
        elif state == "FAILED":
            stage = "FAILED"
        elif evaluation["test_artifact"]:
            stage = "TESTING"
        elif evaluation["validation_artifact"]:
            stage = "POST_VALIDATION"
        elif state == "COMPLETED":
            stage = "POST_TRAINING"
        else:
            stage = state
        runs[label] = {
            "run_id": run_id,
            "gpu": gpu,
            "state": state,
            "global_step": step,
            "epoch": status.get("epoch", metric.get("epoch")),
            "batch": status.get("batch", metric.get("batch")),
            "loss": metric.get("loss"),
            "memory_loss": metric.get("memory_loss"),
            "quiz_loss": metric.get("quiz_loss"),
            "eta_seconds": metric.get("eta_seconds"),
            "target_tokens_per_second": metric.get("target_tokens_per_second"),
            "peak_cuda_memory_mib": metric.get("peak_cuda_memory_mib"),
            "status_updated_at": status.get("updated_at"),
            "pids": pids,
            "pipeline_pids": pipeline_pids,
            "pipeline_stage": stage,
            **evaluation,
        }
        if state == "FAILED":
            alerts.append(f"{label}:FAILED")
        if state == "RUNNING" and not pids:
            alerts.append(f"{label}:RUNNING_WITHOUT_PROCESS")
        if (
            state == "COMPLETED"
            and not evaluation["test_complete"]
            and not pipeline_pids
        ):
            alerts.append(f"{label}:PIPELINE_EXITED_BEFORE_TEST_COMPLETE")
        if (
            state == "RUNNING"
            and previous_steps.get(label) is not None
            and step == previous_steps[label]
        ):
            alerts.append(f"{label}:NO_STEP_PROGRESS")
        previous_steps[label] = step if isinstance(step, int) else None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "gpus": {
            str(index): _gpu_snapshot(index)
            for index in sorted({gpu for _, gpu in runs_config.values()})
        },
        "alerts": alerts,
    }


def _parse_run(value: str) -> tuple[str, tuple[str, int]]:
    try:
        label, run_id, gpu_text = value.split(":", 2)
        gpu = int(gpu_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "--run must be LABEL:RUN_ID:GPU"
        ) from exc
    if not label or not run_id or gpu < 0:
        raise argparse.ArgumentTypeError("--run must be LABEL:RUN_ID:GPU")
    return label, (run_id, gpu)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=1200)
    parser.add_argument("--duration-seconds", type=int, default=18000)
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        help="Run mapping as LABEL:RUN_ID:GPU; repeat for multiple runs",
    )
    parser.add_argument("--stop-when-complete", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 1 or args.duration_seconds < 0:
        raise ValueError("interval must be positive and duration non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs_config = dict(args.run) if args.run else RUNS
    if len(runs_config) != len(args.run or runs_config):
        raise ValueError("Run labels must be unique")
    samples = math.floor(args.duration_seconds / args.interval_seconds) + 1
    previous_steps: dict[str, int | None] = {}
    started = time.monotonic()
    for index in range(samples):
        payload = {
            "sample_index": index,
            "sample_count": samples,
            **_snapshot(previous_steps, runs_config),
        }
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if args.stop_when_complete and all(
            run["test_complete"] for run in payload["runs"].values()
        ):
            break
        if index + 1 < samples:
            deadline = started + (index + 1) * args.interval_seconds
            time.sleep(max(0.0, deadline - time.monotonic()))


if __name__ == "__main__":
    main()
