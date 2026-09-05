from __future__ import annotations

import json
from pathlib import Path

from memory_training.checkpoints import TrainingProgress
from memory_training.methods import DeltaV3AppendMethod, PatchMethod, SummaryMethod
from memory_training.quiz_sft import IndexedQuizSFTDataset
from memory_training.tracking import RunTracker
from memory_training.training_data import (
    BalancedMultitaskBatchSampler,
    ChatExampleEncoder,
    DeltaAppendChatExampleEncoder,
    DeltaAppendSFTDataset,
    EncodedExample,
    QuizChatExampleEncoder,
    SFTCollator,
    quiz_epoch_indices,
)
from memory_training.validation import (
    _quiz_row_with_memory,
    closed_loop_quiz_snapshot_requests,
    memory_scores,
    parse_tool_calls,
    quiz_indices_for_scenarios,
    random_quiz_validation_indices,
    stratified_quiz_validation_indices,
)


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


class ToolCharacterTokenizer(CharacterTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
        tools=None,
    ):
        del tokenize, enable_thinking
        rendered = "<tools>" + ",".join(
            tool["function"]["name"] for tool in (tools or [])
        )
        rendered += "</tools>"
        for message in messages:
            rendered += f"<{message['role']}>{message.get('content') or ''}"
            for call in message.get("tool_calls", []):
                function = call["function"]
                rendered += (
                    f"<call:{function['name']}>{json.dumps(function['arguments'])}"
                )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class FixedBatchSampler:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


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


class MemoryRows:
    def __init__(self, rows):
        self.rows = list(rows)
        self.positions = {
            row_id: position for position, (row_id, _row) in enumerate(self.rows)
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index][1]

    def row_id_at_position(self, position):
        return self.rows[position][0]

    def position_for_row_id(self, row_id):
        return self.positions[row_id]


def _delta_append_row(turn, *, pending, decision, operations=()):
    return {
        "sample_id": f"delta-append-{turn}",
        "scenario_index": 15,
        "global_turn_index": turn,
        "turn_id": f"turn-{turn}",
        "timestamp": f"2026-01-01T00:{turn:02d}",
        "current_turn": {
            "speaker_id": "p1",
            "speaker_name": "Alex",
            "text": f"Turn {turn}",
        },
        "input": {"base_summary": "", "pending_updates": pending},
        "target": {"decision": decision, "operations": list(operations)},
    }


def test_delta_append_dataset_keeps_prior_outputs_but_masks_their_loss() -> None:
    rows = MemoryRows(
        [
            (
                10,
                _delta_append_row(
                    0,
                    pending=[],
                    decision="UPDATE",
                    operations=[["add", "- durable fact"]],
                ),
            ),
            (
                11,
                _delta_append_row(
                    1,
                    pending=[[["add", "- durable fact"]]],
                    decision="NO_OP",
                ),
            ),
        ]
    )
    method = DeltaV3AppendMethod()
    encoder = DeltaAppendChatExampleEncoder(CharacterTokenizer(), max_length=4096)
    dataset = DeltaAppendSFTDataset(rows, rows, method, encoder)

    encoded = dataset[1]
    rendered = "".join(chr(token) for token in encoded.input_ids)
    trained = "".join(
        chr(token)
        for token, label in zip(encoded.input_ids, encoded.labels, strict=True)
        if label != -100
    )

    assert method.format_target(rows[0]) in rendered
    assert rendered.count('"base_summary"') == 1
    assert '"pending_updates"' not in rendered
    assert trained.endswith('{"decision":"NO_OP"}')
    assert "durable fact" not in trained


def test_delta_append_dataset_resets_at_turn_budget() -> None:
    rows = MemoryRows(
        [
            (20, _delta_append_row(0, pending=[], decision="NO_OP")),
            (21, _delta_append_row(1, pending=[], decision="NO_OP")),
            (22, _delta_append_row(2, pending=[], decision="NO_OP")),
        ]
    )
    method = DeltaV3AppendMethod()
    encoder = DeltaAppendChatExampleEncoder(CharacterTokenizer(), max_length=4096)
    dataset = DeltaAppendSFTDataset(rows, rows, method, encoder, max_history_turns=2)

    rendered = "".join(chr(token) for token in dataset[2].input_ids)
    assert "Turn 0" not in rendered
    assert "Turn 1" not in rendered
    assert "Turn 2" in rendered
    assert rendered.count('"base_summary"') == 1


def test_delta_append_dataset_resets_after_five_updates() -> None:
    rows = MemoryRows(
        [
            (
                30 + turn,
                _delta_append_row(
                    turn,
                    pending=[],
                    decision="UPDATE" if turn < 5 else "NO_OP",
                    operations=[["add", f"- fact {turn}"]] if turn < 5 else [],
                ),
            )
            for turn in range(6)
        ]
    )
    method = DeltaV3AppendMethod()
    encoder = DeltaAppendChatExampleEncoder(CharacterTokenizer(), max_length=4096)
    dataset = DeltaAppendSFTDataset(rows, rows, method, encoder)

    history_lengths = [
        len(dataset.contexts[rows.row_id_at_position(position)].history_row_ids)
        for position in range(len(rows))
    ]
    assert history_lengths == [0, 1, 2, 3, 4, 0]


def test_quiz_encoder_masks_tool_schemas_and_keeps_tool_call_target() -> None:
    tokenizer = ToolCharacterTokenizer()
    row = {
        "sample_id": "quiz-1",
        "scenario_index": 81,
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Set brightness."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "set_brightness",
                            "arguments": {"level": 3},
                        },
                    }
                ],
            },
        ],
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_brightness",
                "description": "Set brightness",
                "parameters": {"type": "object"},
            },
        }
    ]
    encoded = QuizChatExampleEncoder(tokenizer, max_length=4096).encode(
        row, tools, row_id=3
    )
    decoded_target = "".join(
        chr(token)
        for token, label in zip(encoded.input_ids, encoded.labels, strict=True)
        if label != -100
    )
    assert "<call:set_brightness>" in decoded_target
    assert "<tools>" not in decoded_target
    assert encoded.task_type == "quiz"
    assert encoded.decision == "TOOL_CALL"


def test_multitask_sampler_alternates_memory_and_quiz_deterministically() -> None:
    memory = FixedBatchSampler([(0, 1), (2, 3), (4,)])
    first = BalancedMultitaskBatchSampler(
        memory,
        memory_size=5,
        quiz_size=4,
        quiz_indices=(3, 0, 2, 1),
        quiz_batch_size=2,
    )
    repeated = BalancedMultitaskBatchSampler(
        memory,
        memory_size=5,
        quiz_size=4,
        quiz_indices=(3, 0, 2, 1),
        quiz_batch_size=2,
    )
    batches = list(first)
    assert batches == list(repeated)
    assert len(batches) == 5
    assert sum(all(index < 5 for index in batch) for batch in batches) == 3
    assert sum(all(index >= 5 for index in batch) for batch in batches) == 2


def test_quiz_schedule_exposes_each_example_exactly_twice_across_epochs() -> None:
    schedules = [
        quiz_epoch_indices(11, epoch=epoch, epochs=3, total_passes=2, seed=42)
        for epoch in range(3)
    ]
    combined = [index for schedule in schedules for index in schedule]
    assert [len(schedule) for schedule in schedules] == [7, 7, 8]
    assert len(combined) == 22
    assert all(combined.count(index) == 2 for index in range(11))


def test_quiz_validation_subset_is_fixed_and_scenario_balanced(tmp_path: Path) -> None:
    path = tmp_path / "quiz_sft.jsonl"
    rows = []
    reasons = [
        "conditional_constraint",
        "coreference_resolution",
        "error_correction",
        "preference_conflict",
        "state_shift",
    ]
    for scenario in range(81, 86):
        for index in range(40):
            rows.append(
                {
                    "schema_version": "vehiclemembench-v2-quiz-tool-sft-v1",
                    "sample_id": f"s{scenario}-q{index}",
                    "scenario_index": scenario,
                    "sft_split": "validation",
                    "quiz_type": "TURN" if index < 30 else "FINAL",
                    "reasoning_type": reasons[index % len(reasons)],
                    "memory_ref": {"global_turn_index": index},
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = IndexedQuizSFTDataset(path, split="validation")
    first = stratified_quiz_validation_indices(source, seed=7)
    assert first == stratified_quiz_validation_indices(source, seed=7)
    assert len(first) == 50
    selected = [source[index] for index in first]
    for scenario in range(81, 86):
        scenario_rows = [row for row in selected if row["scenario_index"] == scenario]
        assert len(scenario_rows) == 10
        assert sum(row["quiz_type"] == "TURN" for row in scenario_rows) == 8
        assert sum(row["quiz_type"] == "FINAL" for row in scenario_rows) == 2

    closed_indices = quiz_indices_for_scenarios(source, (83, 84))
    gold_indices = random_quiz_validation_indices(
        source,
        total_rows=50,
        excluded_scenarios=(83, 84),
        seed=7,
    )
    assert len(closed_indices) == 80
    assert len(gold_indices) == 50
    assert {source[index]["scenario_index"] for index in gold_indices} <= {
        81,
        82,
        85,
    }
    assert set(closed_indices).isdisjoint(gold_indices)
    requests = closed_loop_quiz_snapshot_requests(source, closed_indices)
    assert set(requests) == {83, 84}
    assert sum(
        len(sample_ids)
        for by_turn in requests.values()
        for sample_ids in by_turn.values()
    ) == 80


def test_quiz_memory_override_preserves_current_request() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "Use tools."},
            {
                "role": "user",
                "content": "[Memory]\n- gold\n\n[Current request]\nSet volume.",
            },
        ]
    }
    updated = _quiz_row_with_memory(row, "- predicted")
    assert updated["messages"][1]["content"] == (
        "[Memory]\n- predicted\n\n[Current request]\nSet volume."
    )
    assert row["messages"][1]["content"].startswith("[Memory]\n- gold")


def test_tool_call_parser_supports_qwen_xml_and_granite_json() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_brightness",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "integer"},
                        "zone": {"type": "string"},
                    },
                },
            },
        }
    ]
    qwen = (
        "<tool_call><function=set_brightness>"
        "<parameter=level>3</parameter><parameter=zone>rear_left</parameter>"
        "</function></tool_call>"
    )
    granite = (
        '<tool_call>{"name":"set_brightness","arguments":'
        '{"level":3,"zone":"rear_left"}}</tool_call>'
    )
    expected = [
        {
            "name": "set_brightness",
            "arguments": {"level": 3, "zone": "rear_left"},
        }
    ]
    assert parse_tool_calls(qwen, tools) == expected
    assert parse_tool_calls(granite, tools) == expected


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


def test_tracker_resume_refreshes_effective_config_and_keeps_history(tmp_path) -> None:
    tracker = RunTracker(tmp_path)
    tracker.initialize(
        {
            "created_at": "2026-01-01T00:00:00+00:00",
            "arguments": {"closed_loop_full_scenarios": None},
        }
    )
    checkpoint = Path("/workspace/checkpoints/step-0000500")
    tracker.resume(
        {
            "created_at": "ignored",
            "arguments": {"closed_loop_full_scenarios": [89]},
        },
        checkpoint=checkpoint,
        global_step=500,
    )

    config = json.loads((tmp_path / "config.json").read_text())
    status = json.loads((tmp_path / "status.json").read_text())
    assert config["created_at"] == "2026-01-01T00:00:00+00:00"
    assert config["arguments"]["closed_loop_full_scenarios"] == [89]
    assert config["resume_history"][-1]["checkpoint"] == str(checkpoint)
    assert status["global_step"] == 500
