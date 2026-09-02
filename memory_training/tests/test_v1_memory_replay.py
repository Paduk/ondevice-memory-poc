from pathlib import Path

import pytest

from memory_training.v1_memory_replay import parse_history


def test_parse_history_repairs_identical_duplicate_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "history.txt"
    path.write_text(
        "[2025-04-12 14:05] [2025-04-12 14:05] Person: corrected setting\n",
        encoding="utf-8",
    )

    assert parse_history(path) == [
        {
            "timestamp": "2025-04-12 14:05",
            "speaker": "Person",
            "text": "corrected setting",
        }
    ]


def test_parse_history_rejects_different_duplicate_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "history.txt"
    path.write_text(
        "[2025-04-12 14:05] [2025-04-12 14:06] Person: bad row\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid V1 history line"):
        parse_history(path)
