from memory_training.methods.operations import apply_operations
from memory_training.prepare_grouped_memory_data import (
    group_memory,
    transform_operations,
)

IDS = {"A": "person-a", "B": "person-b"}
ORDER = ("A", "B")


def test_group_memory_and_positioned_add() -> None:
    before = (
        "- [2026-01-01] A: setting=1\n\n"
        "- [2026-01-02] B: setting=2"
    )
    operation = {
        "op": "add",
        "target": "",
        "content": "- [2026-01-03] A: volume=3",
    }
    grouped = group_memory(before, subject_ids=IDS, subject_order=ORDER)
    converted = transform_operations(
        before, [operation], subject_ids=IDS, subject_order=ORDER
    )
    assert converted == (
        {
            "op": "add",
            "target": "- [2026-01-01] setting=1",
            "content": "- [2026-01-03] volume=3",
        },
    )
    after, _ = apply_operations(grouped, converted)
    assert after == (
        "### A\n"
        "- [2026-01-01] setting=1\n"
        "- [2026-01-03] volume=3\n\n"
        "### B\n"
        "- [2026-01-02] setting=2"
    )


def test_grouped_delete_removes_empty_subject_block() -> None:
    before = (
        "- [2026-01-01] A: setting=1\n\n"
        "- [2026-01-02] B: setting=2"
    )
    operation = {
        "op": "delete",
        "target": "- [2026-01-01] A: setting=1",
        "content": "",
    }
    grouped = group_memory(before, subject_ids=IDS, subject_order=ORDER)
    converted = transform_operations(
        before, [operation], subject_ids=IDS, subject_order=ORDER
    )
    after, _ = apply_operations(grouped, converted)
    assert after == "### B\n- [2026-01-02] setting=2"


def test_grouped_delete_add_same_subject_becomes_block_replace() -> None:
    previous = "- [2025-01-01] A: setting=1"
    operations = [
        {
            "op": "delete",
            "target": "- [2025-01-01] A: setting=1",
            "content": "",
        },
        {
            "op": "add",
            "target": "",
            "content": "- [2025-02-01] A: setting=2",
        },
    ]
    transformed = transform_operations(
        previous,
        operations,
        subject_ids=IDS,
        subject_order=ORDER,
    )
    assert transformed == (
        {
            "op": "replace",
            "target": "### A\n- [2025-01-01] setting=1",
            "content": "### A\n- [2025-02-01] setting=2",
        },
    )
