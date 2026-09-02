"""Run one final V1 memory update from an existing turn-wise R2 trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .methods import METHODS
from .ollama_client import OllamaClient
from .v1_memory_replay import TURN_PATTERN
from .validation import state_from_input, state_to_input


def _load_final_input(source_run: Path) -> tuple[str, dict[str, str], Path]:
    databases = list((source_run / "cache").glob("**/memory.db"))
    if len(databases) != 1:
        raise ValueError(f"Expected one memory.db under {source_run}, got {databases}")
    database = databases[0]
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        message_row = connection.execute(
            "SELECT content, created_at FROM messages ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if message_row is None:
            raise ValueError(f"Missing final V1 turn in {database}")
        memory_row = connection.execute(
            "SELECT content FROM memories "
            "WHERE created_at < ? AND status IN ('verified', 'superseded') "
            "ORDER BY created_at DESC LIMIT 1",
            (message_row[1],),
        ).fetchone()
    finally:
        connection.close()
    previous_memory = str(memory_row[0]) if memory_row is not None else ""
    raw_turn = str(message_row[0]).splitlines()[-1]
    match = TURN_PATTERN.fullmatch(raw_turn)
    if match is None:
        raise ValueError(f"Invalid final V1 turn in {database}: {raw_turn}")
    return previous_memory, match.groupdict(), database


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_json(args.run_dir / "config.json")
    method_name = args.method or str(config.get("method") or "")
    if method_name not in {"summary", "patch", "delta_v2", "delta_v3"}:
        raise ValueError("Final-step evaluation supports Summary, Patch, and Delta-v2")
    method = METHODS[method_name]()
    client = OllamaClient(args.ollama_url, timeout_seconds=args.timeout_seconds)
    reports = []
    for scenario in args.scenarios:
        source_run = Path(args.source_run_pattern.format(scenario=scenario))
        previous_memory, turn, database = _load_final_input(source_run)
        turn_id = f"v1-s{scenario:03d}-final-step"
        state = state_from_input(
            method,
            (
                {"base_summary": previous_memory, "pending_updates": []}
                if method_name in {"delta_v2", "delta_v3"}
                else {"previous_memory": previous_memory}
            ),
        )
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
            model=args.memory_model,
            messages=(
                {"role": "system", "content": method.system_prompt},
                {"role": "user", "content": method.format_input(row)},
            ),
            json_mode=True,
            think=False,
            temperature=0,
            seed=args.seed,
            context_length=args.context_length,
            max_new_tokens=args.max_new_tokens,
        )
        parsed = method.parse_output(result.content)
        final_state = method.apply_output(
            state,
            parsed,
            turn_id=turn_id,
        )
        final_memory = method.materialize_memory(final_state)
        report = {
            "schema_version": "palmclaw-v1-final-step-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scenario_index": scenario,
            "method": method_name,
            "model": args.memory_model,
            "source_database": str(database),
            "current_turn": row["current_turn"],
            "previous_memory": previous_memory,
            "decision": parsed.decision,
            "raw_output": result.content,
            "final_memory": final_memory,
            "final_memory_sha256": _text_sha256(final_memory),
            "usage": {
                "prefill_tokens": result.prompt_tokens,
                "decode_tokens": result.output_tokens,
                "latency_seconds": result.latency_seconds,
            },
        }
        scenario_dir = args.output_dir / f"s{scenario:03d}"
        _write_json(scenario_dir / "summary.json", report)
        reports.append(report)
    suite = {
        "schema_version": "palmclaw-v1-final-step-suite-v1",
        "method": method_name,
        "model": args.memory_model,
        "scenarios": args.scenarios,
        "reports": reports,
    }
    _write_json(args.output_dir / "summary.json", suite)
    client.close()
    return suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--memory-model", required=True)
    parser.add_argument(
        "--method", choices=("summary", "patch", "delta_v2", "delta_v3")
    )
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--source-run-pattern", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    return parser


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
