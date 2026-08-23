"""Manage the isolated Ollama server used by Qwen3.5 Test exports."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from .config import DEFAULT_WORKSPACE_ROOT

DEFAULT_VERSION = "0.32.15"
DEFAULT_ROOT = DEFAULT_WORKSPACE_ROOT / "tools" / f"ollama-v{DEFAULT_VERSION}"
DEFAULT_MODELS = DEFAULT_WORKSPACE_ROOT / "ollama-qwen35-models"
DEFAULT_LOG = DEFAULT_WORKSPACE_ROOT / "logs" / "ollama-qwen35.log"
DEFAULT_HOST = "127.0.0.1:11435"
SESSION = "palmclaw-ollama-qwen35"


def status(args: argparse.Namespace) -> dict[str, Any]:
    binary = args.ollama_root / "bin" / "ollama"
    value: dict[str, Any] = {
        "binary": str(binary),
        "binary_exists": binary.is_file(),
        "host": args.host,
        "models": str(args.models),
        "tmux_session": SESSION,
        "tmux_running": _tmux_running(),
    }
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            response = client.get(f"http://{args.host}/api/version")
            response.raise_for_status()
            tags = client.get(f"http://{args.host}/api/tags")
            tags.raise_for_status()
        value["server_version"] = response.json().get("version")
        value["installed_models"] = [
            {
                "name": model.get("name") or model.get("model"),
                "size": model.get("size"),
                "digest": model.get("digest"),
                "details": model.get("details"),
            }
            for model in tags.json().get("models", [])
        ]
        value["reachable"] = True
    except Exception as exc:  # noqa: BLE001 - status must remain available offline.
        value["reachable"] = False
        value["error"] = f"{type(exc).__name__}: {exc}"
    return value


def start(args: argparse.Namespace) -> dict[str, Any]:
    binary = (args.ollama_root / "bin" / "ollama").resolve()
    library = (args.ollama_root / "lib" / "ollama").resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    args.models.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if not _tmux_running():
        environment = {
            "OLLAMA_HOST": args.host,
            "OLLAMA_MODELS": str(args.models.resolve()),
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_VULKAN": "0",
            "LD_LIBRARY_PATH": str(library),
        }
        assignments = " ".join(
            shlex.quote(f"{key}={value}") for key, value in environment.items()
        )
        command = (
            f"env {assignments} {shlex.quote(str(binary))} serve "
            f">>{shlex.quote(str(args.log.resolve()))} 2>&1"
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", SESSION, command], check=True
        )
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        value = status(args)
        if value["reachable"]:
            return value
        time.sleep(0.5)
    raise RuntimeError(f"Isolated Ollama did not start; inspect {args.log}")


def stop(args: argparse.Namespace) -> dict[str, Any]:
    if _tmux_running():
        subprocess.run(["tmux", "kill-session", "-t", SESSION], check=True)
    return status(args)


def _tmux_running() -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", SESSION],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "start", "stop"))
    parser.add_argument("--ollama-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--cuda-visible-devices", default=os.getenv("PALMCLAW_OLLAMA_GPU", "6")
    )
    parser.add_argument("--wait-seconds", type=float, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    operation = {"status": status, "start": start, "stop": stop}[args.action]
    print(json.dumps(operation(args), indent=2))


if __name__ == "__main__":
    main()
