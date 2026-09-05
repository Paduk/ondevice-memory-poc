"""Select mini-validation candidates and the final full-validation winner."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_EPOCH_RE = re.compile(r"epoch-(\d+)\.json$")


def select_mini_candidates(
    run_dir: Path,
    *,
    false_update_threshold: float,
) -> dict[str, Any]:
    eligible = []
    for path in sorted(run_dir.glob("validation-epoch-*.json")):
        match = _EPOCH_RE.search(path.name)
        if match is None:
            continue
        payload = _read_json(path)
        false_update = float(
            payload.get("one_step", {}).get("false_update_rate", 0.0)
        )
        if false_update > false_update_threshold:
            continue
        epoch = int(match.group(1))
        eligible.append(
            {
                "epoch": epoch,
                "checkpoint": str(run_dir / "checkpoints" / f"epoch-{epoch:02d}"),
                "mini_esm": float(
                    payload.get("closed_loop_quiz", {}).get("esm", -1.0)
                ),
                "mini_final_state_f1": float(
                    payload.get("closed_loop", {}).get("final_state_f1", -1.0)
                ),
                "false_update_rate": false_update,
                "selected_by": [],
            }
        )
    if not eligible:
        raise RuntimeError("No mini-validation checkpoint passed the eligibility gate")
    best_esm = max(
        eligible,
        key=lambda row: (
            row["mini_esm"],
            row["mini_final_state_f1"],
            row["epoch"],
        ),
    )
    best_state = max(
        eligible,
        key=lambda row: (
            row["mini_final_state_f1"],
            row["mini_esm"],
            row["epoch"],
        ),
    )
    selected: dict[int, dict[str, Any]] = {}
    for role, candidate in (
        ("best_mini_esm", best_esm),
        ("best_mini_final_state_f1", best_state),
    ):
        row = selected.setdefault(candidate["epoch"], dict(candidate))
        row["selected_by"] = [*row["selected_by"], role]
    return {
        "schema_version": "palmclaw-mini-validation-candidates-v1",
        "false_update_threshold": false_update_threshold,
        "candidate_count": len(selected),
        "candidates": list(selected.values()),
    }


def select_full_winner(run_dir: Path) -> dict[str, Any]:
    candidates = []
    for path in sorted(run_dir.glob("validation-v2-candidate-epoch-*.json")):
        payload = _read_json(path)
        candidates.append(
            {
                "epoch": int(payload["epoch"]),
                "checkpoint": str(payload["checkpoint"]),
                "full_esm": float(
                    payload.get("closed_loop_quiz", {}).get("esm", -1.0)
                ),
                "full_final_state_f1": float(
                    payload.get("closed_loop", {}).get("final_state_f1", -1.0)
                ),
                "full_update_f1": float(
                    payload.get("closed_loop", {}).get("update_f1", -1.0)
                ),
                "result": str(path),
            }
        )
    if not candidates:
        raise RuntimeError("No full-validation candidate result was found")
    for row in candidates:
        row["composite_score"] = (
            0.60 * row["full_esm"]
            + 0.25 * row["full_final_state_f1"]
            + 0.15 * row["full_update_f1"]
        )
    winner = max(candidates, key=lambda row: (
        row["composite_score"], row["full_esm"],
        row["full_final_state_f1"], row["full_update_f1"], -row["epoch"],
    ))
    return {
        "schema_version": "palmclaw-full-validation-winner-v1",
        "selection_metric": "composite.full_esm_60.full_final_state_f1_25.full_update_f1_15",
        "candidate_count": len(candidates),
        "winner": winner,
        "candidates": candidates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mini", "full"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--false-update-threshold", type=float, default=0.20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "mini":
        result = select_mini_candidates(
            args.run_dir,
            false_update_threshold=args.false_update_threshold,
        )
    else:
        result = select_full_winner(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
