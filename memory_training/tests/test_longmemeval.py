from memory_training.longmemeval import (
    build_question_rows,
    evidence_events,
    validate_replay,
)
from memory_training.evaluate_longmemeval_patch import full_user_turns


def _oracle(question_id: str, question_type: str, sessions):
    return {
        "question_id": question_id,
        "question_type": question_type,
        "haystack_dates": ["2023/01/02 (Mon) 10:00", "2023/01/01 (Sun) 10:00"],
        "haystack_session_ids": ["new", "old"],
        "haystack_sessions": sessions,
    }


def test_knowledge_update_is_sorted_add_then_replace():
    oracle = _oracle(
        "ku1",
        "knowledge-update",
        [
            [{"role": "user", "content": "My new favorite is blue.", "has_answer": True}],
            [{"role": "user", "content": "My favorite was red.", "has_answer": True}],
        ],
    )
    metadata = {
        "question_id": "ku1",
        "question_type": "knowledge_update",
        "question_content": {
            "old_answer": "red",
            "updated_fact": "My favorite color is now blue.",
        },
    }
    rows, counts = build_question_rows(
        oracle,
        metadata,
        split="train",
        scenario_index=1,
        question_ordinal=1,
    )
    assert [row["target"]["operations"][0]["op"] for row in rows] == [
        "add",
        "replace",
    ]
    assert rows[0]["timestamp"].startswith("2023/01/01")
    assert "now blue" in rows[1]["target"]["next_memory"]
    assert "was red" not in rows[1]["target"]["next_memory"]
    assert counts["add"] == counts["replace"] == 1
    validate_replay(rows)


def test_temporal_facts_are_added_and_false_evidence_is_ignored():
    oracle = _oracle(
        "tr1",
        "temporal-reasoning",
        [
            [
                {"role": "user", "content": "I visited Paris on Tuesday.", "has_answer": True},
                {"role": "user", "content": "I also like tea.", "has_answer": False},
            ],
            [{"role": "user", "content": "I visited Rome on Monday.", "has_answer": True}],
        ],
    )
    metadata = {
        "question_id": "tr1",
        "question_type": "temp_reasoning_explicit",
        "question_content": {
            "facts": ["I visited Rome on Monday.", "I visited Paris on Tuesday."]
        },
    }
    events = evidence_events(oracle)
    assert len(events) == 2
    rows, counts = build_question_rows(
        oracle,
        metadata,
        split="validation",
        scenario_index=81,
        question_ordinal=1,
    )
    assert all(row["target"]["operations"][0]["op"] == "add" for row in rows)
    assert "tea" not in rows[-1]["target"]["next_memory"]
    assert counts["add"] == 2
    validate_replay(rows)


def test_false_evidence_can_be_emitted_as_sequential_noop():
    oracle = _oracle(
        "tr2",
        "temporal-reasoning",
        [
            [
                {"role": "user", "content": "I visited Paris.", "has_answer": True},
                {"role": "user", "content": "Can you suggest a café?", "has_answer": False},
            ],
            [{"role": "user", "content": "I visited Rome.", "has_answer": True}],
        ],
    )
    metadata = {
        "question_id": "tr2",
        "question_type": "temp_reasoning_explicit",
        "question_content": {"facts": ["I visited Rome.", "I visited Paris."]},
    }
    rows, counts = build_question_rows(
        oracle,
        metadata,
        split="train",
        scenario_index=1,
        question_ordinal=1,
        include_noop=True,
    )
    assert [row["target"]["decision"] for row in rows] == [
        "UPDATE",
        "UPDATE",
        "NO_OP",
    ]
    assert rows[-1]["target"]["operations"] == []
    assert rows[-1]["input"]["previous_memory"] == rows[-1]["target"]["next_memory"]
    assert counts["no_op"] == 1
    validate_replay(rows)


def test_full_evaluation_turns_are_chronological_and_user_only():
    row = {
        "question_id": "q1",
        "haystack_dates": ["2023/01/02 (Mon) 10:00", "2023/01/01 (Sun) 10:00"],
        "haystack_session_ids": ["new", "old"],
        "haystack_sessions": [
            [
                {"role": "assistant", "content": "Do not process me."},
                {"role": "user", "content": "  newer   fact  "},
            ],
            [{"role": "user", "content": "older fact"}],
        ],
    }
    assert full_user_turns(row) == [
        ("2023/01/01 (Sun) 10:00", "old", "older fact"),
        ("2023/01/02 (Mon) 10:00", "new", "newer fact"),
    ]
