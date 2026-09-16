from __future__ import annotations

import json

from memory_training.evaluate_quiz_baseline import (
    _build_kv_lww,
    _cloud_patch_overrides,
    _recent_window,
    _retrieve_kv,
)


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    def decode(self, tokens: list[str], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(tokens)


def _turn(index: int, text: str, *, event: str = "e1") -> dict:
    return {
        "global_turn_index": index,
        "timestamp": f"2026-01-01 00:0{index}",
        "event_id": event,
        "current_turn": {"speaker_name": "Ada", "text": text},
    }


def test_recent_window_never_reads_after_quiz_cutoff() -> None:
    turns = [_turn(0, "old words"), _turn(1, "current words"), _turn(2, "future leak")]
    memory = _recent_window(turns, 1, WordTokenizer(), 20)
    assert "old words" in memory
    assert "current words" in memory
    assert "future leak" not in memory


def test_kv_lww_overwrites_same_speaker_topic_and_retrieves_by_query() -> None:
    turns = [
        _turn(0, "Keep the music volume at 20."),
        _turn(1, "Change the music volume to 15.", event="e2"),
        _turn(2, "Unrelated work meeting.", event="e3"),
    ]
    store = _build_kv_lww(turns, 2)
    assert len(store) == 1
    assert "15" in store["ada::music_volume"]["evidence"]
    assert "volume at 20" not in store["ada::music_volume"]["evidence"]
    memory = _retrieve_kv(
        store, "Apply my remembered music setting", tokenizer=WordTokenizer(), budget=50
    )
    assert "### Ada" in memory
    assert "carcontrol_music_set_volume.volume; value=15" in memory
    assert "key=" not in memory


def test_cloud_patch_override_reads_state_at_quiz_cutoff(tmp_path) -> None:
    step = tmp_path / "s901" / "memory_steps" / "00003.json"
    step.parent.mkdir(parents=True)
    step.write_text(
        json.dumps({"state_after": {"previous_memory": "### Ada\n- volume=15"}}),
        encoding="utf-8",
    )
    source = [
        {
            "sample_id": "s901:q1",
            "scenario_index": 901,
            "memory_ref": {"global_turn_index": 3},
        }
    ]
    assert _cloud_patch_overrides(source, [0], tmp_path) == {
        "s901:q1": "### Ada\n- volume=15"
    }
