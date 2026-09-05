"""Replay method serializers and executors against all canonical training rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import DEFAULT_DATA_ROOT
from .methods import (
    METHODS,
    DeltaMethod,
    DeltaState,
    DeltaV2Method,
    DeltaV2State,
    MemoryMethod,
)
from .methods.delta_v2 import (
    delta_v2_input_from_state,
    expand_compact_operations,
)
from .methods.operations import apply_operations, normalize_memory


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_input_state(
    method: MemoryMethod[Any], state: Any, row: Mapping[str, Any]
) -> None:
    memory_input = row["input"]
    if isinstance(method, DeltaMethod):
        if not isinstance(state, DeltaState):
            raise TypeError("Delta validator received non-Delta state")
        pending = [
            {"turn_id": batch.turn_id, "operations": list(batch.operations)}
            for batch in state.pending_deltas
        ]
        if normalize_memory(memory_input["base_summary"]) != state.base_summary:
            raise ValueError("Delta base_summary differs from replay state")
        if memory_input["pending_deltas"] != pending:
            raise ValueError("Delta pending_deltas differ from replay state")
        if memory_input["updates_since_compaction"] != state.updates_since_compaction:
            raise ValueError("Delta compaction counter differs from replay state")
    elif isinstance(method, DeltaV2Method):
        if not isinstance(state, DeltaV2State):
            raise TypeError("Delta-v2 validator received non-Delta-v2 state")
        expected = delta_v2_input_from_state(
            state, compaction_interval=method.compaction_interval
        )
        if memory_input != expected:
            raise ValueError("Delta-v2 input differs from replay state")
    elif normalize_memory(memory_input["previous_memory"]) != method.materialize_memory(
        state
    ):
        raise ValueError("Previous memory differs from replay state")


def validate_method(
    method_name: str, data_root: Path, *, max_rows: int | None = None
) -> dict[str, Any]:
    method = METHODS[method_name]()
    path = data_root / f"{method.source_view}.jsonl"
    state = method.initial_state()
    scenario = None
    rows = 0
    decisions: Counter[str] = Counter()
    target_characters = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_rows is not None and rows >= max_rows:
                break
            row = json.loads(line)
            if row["scenario_index"] != scenario:
                scenario = row["scenario_index"]
                state = method.initial_state()
            try:
                _validate_input_state(method, state, row)
                prompt = method.format_input(row)
                target = method.format_target(row)
                if not prompt or not target:
                    raise ValueError("Empty serialized prompt or target")
                output = method.parse_output(target)
                state = method.apply_output(state, output, turn_id=row["turn_id"])
                memory = method.materialize_memory(state)
                expected_hash = row["provenance"]["after_memory_sha256"]
                if _sha256(memory) != expected_hash:
                    raise ValueError("Materialized memory hash mismatch")
            except Exception as exc:
                raise RuntimeError(
                    f"{method_name} validation failed at {path.name}:{line_number}"
                ) from exc
            decisions[output.decision] += 1
            target_characters += len(target)
            rows += 1
    return {
        "method": method_name,
        "source_view": method.source_view,
        "rows_checked": rows,
        "complete": max_rows is None,
        "decisions": dict(decisions),
        "target_characters": target_characters,
        "mean_target_characters": target_characters / rows if rows else 0.0,
    }


def validate_compactions(
    data_root: Path, *, method_name: str = "delta", max_rows: int | None = None
) -> dict[str, Any]:
    method = METHODS[method_name]()
    path = data_root / "compaction.jsonl"
    rows = 0
    triggers: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_rows is not None and rows >= max_rows:
                break
            row = json.loads(line)
            memory_input = row["input"]
            memory = normalize_memory(memory_input["base_summary"])
            try:
                if isinstance(method, DeltaV2Method):
                    for batch in memory_input["pending_updates"]:
                        memory, _ = apply_operations(
                            memory, expand_compact_operations(batch)
                        )
                else:
                    for batch in memory_input["pending_deltas"]:
                        memory, _ = apply_operations(memory, batch["operations"])
                if memory != normalize_memory(row["target"]["next_summary"]):
                    raise ValueError("Compaction next_summary mismatch")
                if _sha256(memory) != row["target"]["next_summary_sha256"]:
                    raise ValueError("Compaction next_summary hash mismatch")
            except Exception as exc:
                raise RuntimeError(
                    f"Compaction validation failed at {path.name}:{line_number}"
                ) from exc
            triggers[row["trigger"]] += 1
            rows += 1
    return {
        "method": f"{method_name}_compaction",
        "source_view": "compaction",
        "rows_checked": rows,
        "complete": max_rows is None,
        "triggers": dict(triggers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS)
    )
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    results = [
        validate_method(name, args.data_root, max_rows=args.max_rows)
        for name in args.methods
    ]
    for method_name in args.methods:
        if isinstance(METHODS[method_name](), (DeltaMethod, DeltaV2Method)):
            results.append(
                validate_compactions(
                    args.data_root,
                    method_name=method_name,
                    max_rows=args.max_rows,
                )
            )
    print(json.dumps({"status": "PASS", "results": results}, indent=2))


if __name__ == "__main__":
    main()
