from __future__ import annotations

import json

from memory_training.checkpoints import TrainingProgress
from memory_training.methods import PatchMethod, SummaryMethod
from memory_training.tracking import RunTracker
from memory_training.training_data import (
    ChatExampleEncoder,
    EncodedExample,
    SFTCollator,
)
from memory_training.validation import memory_scores


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        del tokenize, enable_thinking
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def _summary_row(decision: str = "NO_OP") -> dict:
    target = {"decision": decision, "next_memory": ""}
    if decision == "UPDATE":
        target["next_memory"] = "- durable fact"
    return {
        "scenario_index": 81,
        "turn_id": "turn-1",
        "timestamp": "2026-01-01T00:00",
        "current_turn": {"speaker_name": "Alex", "text": "Hello"},
        "input": {"previous_memory": ""},
        "target": target,
    }


def test_chat_encoder_masks_prompt_and_keeps_exact_target() -> None:
    tokenizer = CharacterTokenizer()
    method = SummaryMethod()
    encoded = ChatExampleEncoder(tokenizer, max_length=4096).encode(
        _summary_row(), method, row_id=7
    )
    target = method.format_target(_summary_row())
    decoded_target = "".join(
        chr(token)
        for token, label in zip(encoded.input_ids, encoded.labels, strict=True)
        if label != -100
    )
    assert decoded_target.endswith(target)
    assert encoded.labels.count(-100) > 0


def test_collator_masks_padding() -> None:
    collated = SFTCollator(0)(
        [
            EncodedExample((1, 2), (-100, 2), 1, 81, "NO_OP"),
            EncodedExample((3,), (3,), 2, 81, "UPDATE"),
        ]
    )
    assert collated["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert collated["labels"].tolist() == [[-100, 2], [3, -100]]
    assert collated["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_memory_scores_are_line_based_and_exact_is_strict() -> None:
    scores = memory_scores("- a\n- b", "- a\n- c")
    assert scores["exact"] == 0.0
    assert scores["f1"] == 0.5


def test_tracker_and_progress_are_dashboard_readable(tmp_path) -> None:
    tracker = RunTracker(tmp_path)
    tracker.initialize({"method": PatchMethod.name})
    tracker.metric("train", global_step=1, loss=0.5)
    progress = TrainingProgress(global_step=1)
    tracker.status("RUNNING", **progress.__dict__)
    status = json.loads((tmp_path / "status.json").read_text())
    metric = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert status["global_step"] == 1
    assert metric["loss"] == 0.5
