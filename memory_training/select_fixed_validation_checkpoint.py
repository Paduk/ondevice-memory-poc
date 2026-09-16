"""Select a checkpoint from fixed closed-loop Validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def select_checkpoint(run_dir: Path, epochs: list[int]) -> dict[str, Any]:
    rows = []
    for epoch in epochs:
        path = run_dir / f"eval-fixed-validation-epoch-{epoch:02d}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "epoch": epoch,
            "checkpoint": result["checkpoint"],
            "closed_loop_quiz_esm": result["closed_loop_quiz"]["esm"],
            "closed_loop_final_state_f1": result["closed_loop"]["final_state_f1"],
            "closed_loop_update_f1": result["closed_loop"]["update_f1"],
            "output": str(path),
        }
        row["composite_score"] = (
            0.60 * float(row["closed_loop_quiz_esm"])
            + 0.25 * float(row["closed_loop_final_state_f1"])
            + 0.15 * float(row["closed_loop_update_f1"])
        )
        rows.append(row)
    winner = max(
        rows,
        key=lambda row: (
            row["composite_score"],
            row["closed_loop_quiz_esm"],
            row["closed_loop_final_state_f1"],
            row["closed_loop_update_f1"],
            -row["epoch"],
        ),
    )
    return {
        "schema_version": "palmclaw-fixed-validation-selection-v1",
        "selection_metric": "composite_60_25_15",
        "evaluated_epochs": epochs,
        "results": rows,
        "winner": winner,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_checkpoint(args.run_dir, args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
