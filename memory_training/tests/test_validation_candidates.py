from __future__ import annotations

import json
from pathlib import Path

from memory_training.validation_candidates import (
    select_full_winner,
    select_mini_candidates,
)


def _write_mini(path: Path, *, esm: float, state: float, false_update: float = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "closed_loop_quiz": {"esm": esm},
                "closed_loop": {"final_state_f1": state},
                "one_step": {"false_update_rate": false_update},
            }
        )
    )


def test_selects_at_most_two_distinct_mini_candidates(tmp_path: Path) -> None:
    _write_mini(tmp_path / "validation-epoch-01.json", esm=0.8, state=0.4)
    _write_mini(tmp_path / "validation-epoch-02.json", esm=0.7, state=0.6)
    _write_mini(
        tmp_path / "validation-epoch-03.json",
        esm=0.9,
        state=0.9,
        false_update=0.3,
    )
    result = select_mini_candidates(tmp_path, false_update_threshold=0.2)
    assert result["candidate_count"] == 2
    assert {row["epoch"] for row in result["candidates"]} == {1, 2}


def test_deduplicates_candidate_that_wins_both_metrics(tmp_path: Path) -> None:
    _write_mini(tmp_path / "validation-epoch-01.json", esm=0.8, state=0.7)
    _write_mini(tmp_path / "validation-epoch-02.json", esm=0.7, state=0.6)
    result = select_mini_candidates(tmp_path, false_update_threshold=0.2)
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["selected_by"] == [
        "best_mini_esm",
        "best_mini_final_state_f1",
    ]


def test_full_winner_uses_esm_then_state_f1(tmp_path: Path) -> None:
    for epoch, esm, state in ((1, 0.7, 0.5), (2, 0.7, 0.6)):
        (tmp_path / f"validation-v2-candidate-epoch-{epoch:02d}.json").write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "checkpoint": str(tmp_path / f"epoch-{epoch:02d}"),
                    "closed_loop_quiz": {"esm": esm},
                    "closed_loop": {"final_state_f1": state},
                }
            )
        )
    result = select_full_winner(tmp_path)
    assert result["winner"]["epoch"] == 2
