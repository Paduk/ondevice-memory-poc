"""Exact-block deterministic executor used by Patch and Delta."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

VALID_OPERATIONS = frozenset({"add", "replace", "delete"})


@dataclass(frozen=True)
class PatchStats:
    operation_count: int
    add_count: int
    replace_count: int
    delete_count: int
    inserted_characters: int
    deleted_characters: int


def normalize_memory(content: str) -> str:
    return "\n".join(
        line.rstrip() for line in content.replace("\r\n", "\n").splitlines()
    ).strip()


def normalize_fragment(content: str) -> str:
    lines = [
        line.rstrip()
        for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def validate_operations(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("operations must be an array")
    if not 1 <= len(value) <= 32:
        raise ValueError("UPDATE must contain between 1 and 32 operations")
    operations = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"operation {index} must be an object")
        if set(raw) != {"op", "target", "content"}:
            raise ValueError(f"operation {index} has invalid fields")
        op, target, content = raw["op"], raw["target"], raw["content"]
        if op not in VALID_OPERATIONS:
            raise ValueError(f"operation {index} has invalid op: {op}")
        if not isinstance(target, str) or not isinstance(content, str):
            raise TypeError(f"operation {index} target/content must be strings")
        target = normalize_fragment(target)
        content = normalize_fragment(content)
        if op == "add" and not content:
            raise ValueError(f"add operation {index} has empty content")
        if op == "add" and target and _starts_user_block(content):
            raise ValueError(
                f"add operation {index} must append a complete new user block"
            )
        if op == "replace" and (not target or not content):
            raise ValueError(f"replace operation {index} is incomplete")
        if op == "delete" and (not target or content):
            raise ValueError(f"delete operation {index} is invalid")
        operations.append({"op": str(op), "target": target, "content": content})
    return tuple(operations)


def _starts_user_block(content: str) -> bool:
    lines = content.splitlines()
    if not lines:
        return False
    first = lines[0]
    if first.startswith("### ") or (
        first.startswith("**") and first.rstrip(":").endswith("**")
    ):
        return True
    return bool(
        first.startswith("- ")
        and first.endswith(":")
        and any(line.startswith(("  ", "\t")) for line in lines[1:])
    )


def apply_operations(
    previous_memory: str, operations: Sequence[Mapping[str, Any]]
) -> tuple[str, PatchStats]:
    prepared = validate_operations(operations)
    current = normalize_memory(previous_memory)
    inserted = 0
    deleted = 0
    counts = {"add": 0, "replace": 0, "delete": 0}
    for index, operation in enumerate(prepared):
        op = operation["op"]
        target = operation["target"]
        content = operation["content"]
        if op == "add":
            if target:
                _, end = _unique_complete_block(current, target, index=index)
                current = current[:end] + "\n" + content + current[end:]
            else:
                current = f"{current}\n\n{content}" if current else content
            inserted += len(content)
        elif op == "replace":
            start, end = _unique_complete_block(current, target, index=index)
            current = current[:start] + content + current[end:]
            inserted += len(content)
            deleted += len(target)
        else:
            start, end = _unique_complete_block(current, target, index=index)
            prefix = current[:start]
            suffix = current[end:]
            if prefix and suffix:
                left_newlines = len(prefix) - len(prefix.rstrip("\n"))
                right_newlines = len(suffix) - len(suffix.lstrip("\n"))
                separator = "\n" * max(1, left_newlines, right_newlines)
                current = prefix.rstrip("\n") + separator + suffix.lstrip("\n")
            else:
                current = (prefix or suffix).strip("\n")
            deleted += len(target)
        counts[op] += 1
    memory = normalize_memory(current)
    if memory == normalize_memory(previous_memory):
        raise ValueError("Patch made no state change")
    return memory, PatchStats(
        operation_count=len(prepared),
        add_count=counts["add"],
        replace_count=counts["replace"],
        delete_count=counts["delete"],
        inserted_characters=inserted,
        deleted_characters=deleted,
    )


def _unique_complete_block(memory: str, target: str, *, index: int) -> tuple[int, int]:
    start = memory.find(target)
    if start < 0 or memory.find(target, start + 1) >= 0:
        raise ValueError(f"operation {index} target must occur exactly once")
    end = start + len(target)
    if (start > 0 and memory[start - 1] != "\n") or (
        end < len(memory) and memory[end] != "\n"
    ):
        raise ValueError(f"operation {index} target must cover complete lines")
    return start, end
