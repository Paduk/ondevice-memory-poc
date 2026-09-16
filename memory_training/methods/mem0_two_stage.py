"""Shared-model two-stage fact extraction and memory-management baseline."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .base import MemoryMethod, ParsedMemoryOutput
from .common import compact_json, parse_json_object
from .mem0_one_pass import Mem0OnePassMethod, _events, _facts


MEM0_TWO_STAGE_SYSTEM_PROMPT = """Perform the memory task selected by the task field.
For EXTRACT, extract durable vehicle-memory facts from source_batch and return exactly {"facts":[...]}.
Use an empty facts list when no durable fact exists. An explicit end of a temporary condition may set ephemeral_invalidation:true.
For MANAGE, compare new_facts with retrieved_facts and return exactly {"memory":[...]}.
Events are ADD(subject,text), UPDATE(id,subject,text), DELETE(id), or NONE(id).
Output JSON only."""


class Mem0TwoStageMethod(MemoryMethod[dict[str, dict[str, str]]]):
    """One shared model invoked once for EXTRACT and again for MANAGE."""

    name = "mem0_two_stage"
    source_view = "patch"
    system_prompt = MEM0_TWO_STAGE_SYSTEM_PROMPT

    def __init__(self, *, retrieval_k: int = 20) -> None:
        self.retriever = Mem0OnePassMethod(retrieval_k=retrieval_k)

    def initial_state(self) -> dict[str, dict[str, str]]:
        return {}

    def format_input(self, row: Mapping[str, Any]) -> str:
        task = row.get("task_type")
        value = row.get("input")
        if not isinstance(value, Mapping):
            raise TypeError("input must be an object")
        if task == "EXTRACT":
            if set(value) != {"source_batch"}:
                raise ValueError("EXTRACT requires source_batch")
        elif task == "MANAGE":
            if set(value) != {"new_facts", "retrieved_facts"}:
                raise ValueError("MANAGE requires new_facts and retrieved_facts")
        else:
            raise ValueError(f"Unknown two-stage task: {task!r}")
        return compact_json({"task": task, **value})

    def format_target(self, row: Mapping[str, Any]) -> str:
        task = row.get("task_type")
        target = row.get("target")
        if not isinstance(target, Mapping):
            raise TypeError("target must be an object")
        if task == "EXTRACT":
            return compact_json({"facts": _facts(target.get("facts"))})
        if task == "MANAGE":
            return compact_json({"memory": _events(target.get("memory"))})
        raise ValueError(f"Unknown two-stage task: {task!r}")

    def runtime_row(
        self,
        canonical: Mapping[str, Any],
        state: Mapping[str, Mapping[str, str]],
    ) -> dict[str, Any]:
        """Build the first, EXTRACT, call for predicted-state evaluation."""
        row = copy.deepcopy(dict(canonical))
        turn = canonical.get("current_turn")
        if not isinstance(turn, Mapping):
            canonical_input = canonical.get("input")
            turns = (
                canonical_input.get("turns")
                if isinstance(canonical_input, Mapping)
                else None
            )
            if not isinstance(turns, Sequence) or not turns:
                raise ValueError("Two-stage runtime row has no current turn")
            turn = turns[-1]
        row["task_type"] = "EXTRACT"
        row["input"] = {"source_batch": [copy.deepcopy(dict(turn))]}
        row["_runtime_state"] = {
            key: dict(value) for key, value in state.items()
        }
        return row

    def manager_row(
        self,
        extraction_row: Mapping[str, Any],
        facts: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build MANAGE using predicted facts and predicted active records."""
        state = extraction_row.get("_runtime_state")
        if not isinstance(state, Mapping):
            raise TypeError("Two-stage extraction row is missing runtime state")
        normalized_facts = list(_facts(facts))
        query = "\n".join(
            f"{fact['subject']}: {fact['text']}" for fact in normalized_facts
        )
        row = copy.deepcopy(dict(extraction_row))
        row["task_type"] = "MANAGE"
        row["input"] = {
            "new_facts": normalized_facts,
            "retrieved_facts": self.retriever.retrieve_facts(query, state),
        }
        row.pop("_runtime_state", None)
        return row

    def parse_output(self, text: str) -> ParsedMemoryOutput:
        value = parse_json_object(text)
        if set(value) == {"facts"}:
            facts = _facts(value["facts"])
            decision = "UPDATE" if facts else "NO_OP"
            return ParsedMemoryOutput(decision, {"facts": facts, "task": "EXTRACT"})
        if set(value) == {"memory"}:
            events = _events(value["memory"])
            decision = "UPDATE" if any(e["event"] != "NONE" for e in events) else "NO_OP"
            return ParsedMemoryOutput(decision, {"memory": events, "task": "MANAGE"})
        raise ValueError("Two-stage output must contain exactly facts or memory")

    def apply_output(
        self,
        state: dict[str, dict[str, str]],
        output: ParsedMemoryOutput,
        *,
        turn_id: str | None = None,
    ) -> dict[str, dict[str, str]]:
        if output.payload.get("task") == "EXTRACT":
            return {key: dict(value) for key, value in state.items()}
        result = {key: dict(value) for key, value in state.items()}
        add_index = 0
        for event in _events(output.payload.get("memory", ())):
            kind = event["event"]
            if kind == "ADD":
                add_index += 1
                base = turn_id or "turn"
                record_id = f"{base}:fact:{add_index:02d}"
                while record_id in result:
                    add_index += 1
                    record_id = f"{base}:fact:{add_index:02d}"
                result[record_id] = {
                    "subject": event["subject"],
                    "text": event["text"],
                }
            elif kind == "UPDATE":
                if event["id"] not in result:
                    raise ValueError(f"Unknown UPDATE record id: {event['id']}")
                result[event["id"]] = {
                    "subject": event["subject"],
                    "text": event["text"],
                }
            elif kind == "DELETE":
                if event["id"] not in result:
                    raise ValueError(f"Unknown DELETE record id: {event['id']}")
                del result[event["id"]]
        return result

    def materialize_memory(self, state: dict[str, dict[str, str]]) -> str:
        return self.retriever.materialize_memory(state)
