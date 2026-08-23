"""Validate storage, datasets, GPU, Hugging Face access, and Ollama connectivity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_WORKSPACE_ROOT,
    TARGET_MODELS,
    split_for_scenario,
)
from .workspace import initialize_workspace

MEMORY_FILES = ("summary.jsonl", "patch.jsonl", "delta.jsonl")
QUIZ_FILES = ("turn_quiz.jsonl", "final_quiz.jsonl")
EXPECTED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "trl",
    "datasets",
    "mlflow",
    "fastapi",
    "uvicorn",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _manifest_files(data_root: Path) -> dict[str, dict[str, Any]]:
    training = _read_json(data_root / "manifest.json")
    quiz = _read_json(data_root / "quiz_manifest.json")
    files = {}
    for manifest in (training, quiz):
        for metadata in manifest.get("files", {}).values():
            files[Path(metadata["path"]).name] = metadata
    return files


def validate_file_integrity(data_root: Path, *, hashes: bool) -> dict[str, Any]:
    metadata_by_name = _manifest_files(data_root)
    results: dict[str, Any] = {}
    for name in (*MEMORY_FILES, "compaction.jsonl", *QUIZ_FILES):
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata = metadata_by_name.get(name)
        if metadata is None:
            raise ValueError(f"No manifest entry for {name}")
        actual_bytes = path.stat().st_size
        if actual_bytes != metadata["bytes"]:
            raise ValueError(
                f"Size mismatch for {name}: {actual_bytes} != {metadata['bytes']}"
            )
        result = {
            "bytes": actual_bytes,
            "expected_lines": metadata["line_count"],
            "sha256_checked": hashes,
        }
        if hashes:
            actual_hash = _sha256(path)
            if actual_hash != metadata["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {name}")
            result["sha256"] = actual_hash
        results[name] = result
    return results


def validate_memory_views(
    data_root: Path, *, max_rows: int | None = None
) -> dict[str, Any]:
    paths = [data_root / name for name in MEMORY_FILES]
    split_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    rows = 0
    with (
        paths[0].open(encoding="utf-8") as summary_handle,
        paths[1].open(encoding="utf-8") as patch_handle,
        paths[2].open(encoding="utf-8") as delta_handle,
    ):
        streams = (summary_handle, patch_handle, delta_handle)
        for line_number, lines in enumerate(
            zip_longest(*streams, fillvalue=None), start=1
        ):
            if max_rows is not None and rows >= max_rows:
                break
            if any(line is None for line in lines):
                raise ValueError("Summary/Patch/Delta line counts differ")
            parsed = []
            for name, line in zip(MEMORY_FILES, lines, strict=True):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {name}:{line_number}") from exc
                parsed.append(row)
            summary, patch, delta = parsed
            identity = (
                summary.get("scenario_index"),
                summary.get("global_turn_index"),
                summary.get("turn_id"),
            )
            for name, row in zip(MEMORY_FILES[1:], (patch, delta), strict=True):
                candidate = (
                    row.get("scenario_index"),
                    row.get("global_turn_index"),
                    row.get("turn_id"),
                )
                if candidate != identity:
                    raise ValueError(
                        f"Cross-view identity mismatch in {name}:{line_number}"
                    )
            scenario = summary.get("scenario_index")
            split = summary.get("split")
            if not isinstance(scenario, int) or split != split_for_scenario(scenario):
                raise ValueError(f"Invalid split in summary.jsonl:{line_number}")
            if any(row.get("split") != split for row in (patch, delta)):
                raise ValueError(f"Cross-view split mismatch at line {line_number}")
            decisions = [row.get("target", {}).get("decision") for row in parsed]
            if len(set(decisions)) != 1 or decisions[0] not in {"NO_OP", "UPDATE"}:
                raise ValueError(f"Invalid decision alignment at line {line_number}")
            if decisions[0] == "NO_OP" and (
                patch["target"].get("operations") or delta["target"].get("operations")
            ):
                raise ValueError(f"NO_OP has operations at line {line_number}")
            split_counts[split] += 1
            decision_counts[decisions[0]] += 1
            rows += 1
    return {
        "rows_checked": rows,
        "complete": max_rows is None,
        "split_counts": dict(split_counts),
        "decision_counts": dict(decision_counts),
    }


def validate_quizzes(
    data_root: Path, *, max_rows_per_file: int | None = None
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in QUIZ_FILES:
        rows = 0
        splits: Counter[str] = Counter()
        with (data_root / name).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if max_rows_per_file is not None and rows >= max_rows_per_file:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {name}:{line_number}") from exc
                scenario = row.get("scenario_index")
                split = row.get("split")
                if not isinstance(scenario, int) or split != split_for_scenario(
                    scenario
                ):
                    raise ValueError(f"Invalid split in {name}:{line_number}")
                memory_ref = row.get("memory_ref", {})
                required_refs = {
                    "summary_sample_id",
                    "patch_sample_id",
                    "delta_sample_id",
                    "memory_snapshot_sha256",
                }
                if not required_refs.issubset(memory_ref):
                    raise ValueError(f"Incomplete memory_ref in {name}:{line_number}")
                if not row.get("target", {}).get("gold_calls"):
                    raise ValueError(f"Missing gold_calls in {name}:{line_number}")
                splits[split] += 1
                rows += 1
        output[name] = {"rows_checked": rows, "split_counts": dict(splits)}
    return output


def inspect_gpus() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "error": "nvidia-smi not found"}
    command = [
        executable,
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    devices = []
    for line in completed.stdout.splitlines():
        index, name, total, free, driver = (part.strip() for part in line.split(","))
        devices.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_free_mib": int(free),
                "driver": driver,
            }
        )
    return {"available": bool(devices), "devices": devices}


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "PalmClaw-preflight/1"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object from {url}")
    return value


def inspect_huggingface(timeout: float) -> dict[str, Any]:
    results = []
    for model in TARGET_MODELS:
        url = f"https://huggingface.co/api/models/{model.hf_id}"
        try:
            metadata = _get_json(url, timeout)
            results.append(
                {
                    "key": model.key,
                    "hf_id": model.hf_id,
                    "reachable": True,
                    "private": bool(metadata.get("private")),
                    "gated": metadata.get("gated", False),
                    "sha": metadata.get("sha"),
                }
            )
        except (OSError, ValueError, urllib.error.URLError) as exc:
            results.append(
                {
                    "key": model.key,
                    "hf_id": model.hf_id,
                    "reachable": False,
                    "error": str(exc),
                }
            )
    return {"models": results}


def inspect_ollama(base_url: str, timeout: float) -> dict[str, Any]:
    try:
        version = _get_json(f"{base_url.rstrip('/')}/api/version", timeout)
        tags = _get_json(f"{base_url.rstrip('/')}/api/tags", timeout)
        installed = {item.get("name") for item in tags.get("models", [])}
        return {
            "reachable": True,
            "version": version.get("version"),
            "installed_models": sorted(name for name in installed if name),
            "target_tags_present": {
                model.ollama_tag: model.ollama_tag in installed
                for model in TARGET_MODELS
            },
        }
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"reachable": False, "error": str(exc)}


def inspect_python_packages() -> dict[str, str | None]:
    versions = {}
    for package in EXPECTED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    data_root = args.data_root.resolve()
    initialize_workspace(workspace, data_root)
    disk = shutil.disk_usage(workspace)
    if disk.free < args.minimum_free_gib * 1024**3:
        raise RuntimeError(
            f"Only {disk.free / 1024**3:.1f} GiB free; "
            f"requires {args.minimum_free_gib:.1f} GiB"
        )

    full = args.mode == "full"
    report = {
        "schema_version": "palmclaw-memory-training-preflight-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "workspace": str(workspace),
        "data_root": str(data_root),
        "disk": {
            "total_gib": round(disk.total / 1024**3, 2),
            "used_gib": round(disk.used / 1024**3, 2),
            "free_gib": round(disk.free / 1024**3, 2),
        },
        "file_integrity": validate_file_integrity(data_root, hashes=args.hashes),
        "memory_views": validate_memory_views(
            data_root, max_rows=None if full else args.sample_rows
        ),
        "quizzes": validate_quizzes(
            data_root, max_rows_per_file=None if full else args.sample_rows
        ),
        "gpus": inspect_gpus(),
        "huggingface": inspect_huggingface(args.timeout),
        "ollama": inspect_ollama(args.ollama_url, args.timeout),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "packages": inspect_python_packages(),
        },
    }
    report_dir = workspace / "reports" / "preflight"
    report_dir.mkdir(parents=True, exist_ok=True)
    latest = report_dir / "latest.json"
    temporary = latest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(latest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample-rows", type=int, default=1000)
    parser.add_argument("--hashes", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=40.0)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    try:
        report = run_preflight(args)
    except Exception as exc:  # CLI boundary records concise failure.
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1) from exc
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
