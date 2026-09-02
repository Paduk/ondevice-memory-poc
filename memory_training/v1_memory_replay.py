"""Replay original VehicleMemBench V1 histories through a trained memory model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .methods import METHODS
from .ollama_client import OllamaClient
from .validation import state_from_input, state_to_input

TURN_PATTERN = re.compile(r"^\[(?P<timestamp>[^]]+)] (?P<speaker>[^:]+): (?P<text>.*)$")
DUPLICATED_TIMESTAMP_PATTERN = re.compile(
    r"^\[(?P<timestamp>[^]]+)] \[(?P<duplicate>[^]]+)] "
    r"(?P<speaker>[^:]+): (?P<text>.*)$"
)


def parse_history(path: Path) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        match = TURN_PATTERN.fullmatch(raw)
        if match is None:
            duplicated = DUPLICATED_TIMESTAMP_PATTERN.fullmatch(raw)
            if duplicated is None or duplicated["timestamp"] != duplicated["duplicate"]:
                raise ValueError(f"Invalid V1 history line {path}:{line_number}")
            turns.append(
                {
                    "timestamp": duplicated["timestamp"],
                    "speaker": duplicated["speaker"],
                    "text": duplicated["text"],
                }
            )
        else:
            turns.append(match.groupdict())
    if not turns:
        raise ValueError(f"Empty V1 history: {path}")
    return turns


def replay_scenario(
    client: OllamaClient,
    *,
    method: Any,
    model: str,
    history_path: Path,
    scenario: int,
    output_root: Path,
    seed: int,
    context_length: int,
    max_new_tokens: int,
    checkpoint_interval: int,
    turn_limit: int | None,
) -> dict[str, Any]:
    turns = parse_history(history_path)
    if turn_limit is not None:
        if turn_limit < 1:
            raise ValueError("turn_limit must be positive")
        turns = turns[:turn_limit]
    scenario_dir = output_root / f"s{scenario:03d}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = scenario_dir / "checkpoint.json"
    turns_path = scenario_dir / "turns.jsonl"
    summary_path = scenario_dir / "summary.json"
    signature = _signature(
        {
            "schema": "palmclaw-v1-memory-replay-v1",
            "history_sha256": _file_sha256(history_path),
            "method": method.name,
            "model": model,
            "seed": seed,
            "context_length": context_length,
            "max_new_tokens": max_new_tokens,
            "turn_limit": turn_limit,
        }
    )
    if summary_path.is_file():
        stored = _read_json(summary_path)
        if stored.get("signature") != signature:
            raise ValueError(f"Stale V1 replay summary: {summary_path}")
        return stored

    state = method.initial_state()
    start_position = 0
    counters = {"updates": 0, "no_ops": 0, "invalid": 0}
    usage = {
        "prefill_tokens": 0,
        "decode_tokens": 0,
        "latency_seconds": 0.0,
    }
    if checkpoint_path.is_file():
        checkpoint = _read_json(checkpoint_path)
        if checkpoint.get("signature") != signature:
            raise ValueError(f"Stale V1 replay checkpoint: {checkpoint_path}")
        start_position = int(checkpoint["next_position"])
        state = (
            state_from_input(method, checkpoint["state"])
            if "state" in checkpoint
            else str(checkpoint["memory"])
        )
        counters.update(checkpoint.get("counters") or {})
        usage.update(checkpoint.get("usage") or {})
        _trim_jsonl(turns_path, start_position)

    for position in range(start_position, len(turns)):
        turn = turns[position]
        turn_id = f"v1-s{scenario:03d}-turn-{position + 1:05d}"
        row = {
            "turn_id": turn_id,
            "timestamp": turn["timestamp"],
            "current_turn": {
                "speaker_id": turn["speaker"],
                "speaker_name": turn["speaker"],
                "text": turn["text"],
            },
            "input": state_to_input(method, state),
        }
        result = client.chat(
            model=model,
            messages=(
                {"role": "system", "content": method.system_prompt},
                {"role": "user", "content": method.format_input(row)},
            ),
            json_mode=True,
            think=False,
            temperature=0,
            seed=seed,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
        )
        decision = "INVALID"
        error = None
        try:
            parsed = method.parse_output(result.content)
            state = method.apply_output(state, parsed, turn_id=turn_id)
            decision = parsed.decision
            counters["updates" if decision == "UPDATE" else "no_ops"] += 1
        except Exception as exc:  # noqa: BLE001 - preserve invalid generations.
            counters["invalid"] += 1
            error = f"{type(exc).__name__}: {exc}"
        usage["prefill_tokens"] += result.prompt_tokens
        usage["decode_tokens"] += result.output_tokens
        usage["latency_seconds"] += result.latency_seconds
        _append_jsonl(
            turns_path,
            {
                "position": position,
                "turn_id": turn_id,
                "decision": decision,
                "error": error,
                "output": result.content,
                "memory_sha256": _text_sha256(method.materialize_memory(state)),
                "prefill_tokens": result.prompt_tokens,
                "decode_tokens": result.output_tokens,
                "latency_seconds": result.latency_seconds,
            },
        )
        if (position + 1) % checkpoint_interval == 0 or position + 1 == len(turns):
            _write_json(
                checkpoint_path,
                {
                    "signature": signature,
                    "next_position": position + 1,
                    "state": state_to_input(method, state),
                    "memory": method.materialize_memory(state),
                    "counters": counters,
                    "usage": usage,
                },
            )

    report = {
        "schema_version": "palmclaw-v1-memory-replay-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "scenario_index": scenario,
        "method": method.name,
        "model": model,
        "turns": len(turns),
        "counters": counters,
        "usage": usage,
        "final_memory": method.materialize_memory(state),
        "final_memory_sha256": _text_sha256(method.materialize_memory(state)),
    }
    _write_json(summary_path, report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_json(args.run_dir / "config.json")
    method_name = args.method or str(config.get("method") or "")
    if method_name not in {"summary", "patch", "delta_v2", "delta_v3"}:
        raise ValueError("V1 replay supports Summary, Patch, and Delta-v2")
    method = METHODS[method_name]()
    client = OllamaClient(args.ollama_url, timeout_seconds=args.timeout_seconds)
    tags = {str(item.get("name") or item.get("model")) for item in client.tags()}
    if args.memory_model not in tags:
        raise ValueError(f"Ollama model is unavailable: {args.memory_model}")
    reports = []
    for scenario in args.scenarios:
        history_path = (
            args.dataset_root / "benchmark" / "history" / f"history_{scenario}.txt"
        )
        reports.append(
            replay_scenario(
                client,
                method=method,
                model=args.memory_model,
                history_path=history_path,
                scenario=scenario,
                output_root=args.output_dir,
                seed=args.seed,
                context_length=args.context_length,
                max_new_tokens=args.max_new_tokens,
                checkpoint_interval=args.checkpoint_interval,
                turn_limit=args.turn_limit,
            )
        )
    client.close()
    result = {
        "schema_version": "palmclaw-v1-memory-replay-suite-v1",
        "method": method_name,
        "model": args.memory_model,
        "scenarios": args.scenarios,
        "reports": reports,
    }
    _write_json(args.output_dir / "summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--memory-model", required=True)
    parser.add_argument(
        "--method", choices=("summary", "patch", "delta_v2", "delta_v3")
    )
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/hj153lee/VehicleMemBench")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--turn-limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    return parser


def _signature(value: Any) -> str:
    return _text_sha256(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _trim_jsonl(path: Path, rows: int) -> None:
    if not path.is_file():
        return
    kept = path.read_text(encoding="utf-8").splitlines()[:rows]
    path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
