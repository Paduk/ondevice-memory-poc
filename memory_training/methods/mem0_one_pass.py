"""One-pass fact extraction and CRUD memory-update baseline."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .base import MemoryMethod, ParsedMemoryOutput
from .common import compact_json, parse_json_object, require_exact_fields


MEM0_ONE_PASS_SYSTEM_PROMPT = """Maintain durable vehicle memory as fact records.
Given the current source batch and retrieved facts, extract new facts and apply memory events.
Return exactly {"facts":[...],"memory":[...]}.
Events are ADD(subject,text), UPDATE(id,subject,text), DELETE(id), or NONE(id).
For an explicit end of a temporary condition, a fact may set ephemeral_invalidation:true.
Output JSON only."""


def _facts(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("facts must be an array")
    facts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) not in (
            {"subject", "text"},
            {"subject", "text", "ephemeral_invalidation"},
        ):
            raise ValueError(
                "Each fact must contain subject/text and may contain "
                "ephemeral_invalidation"
            )
        subject, text = item["subject"], item["text"]
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("Fact subject must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Fact text must be a non-empty string")
        fact: dict[str, Any] = {"subject": subject.strip(), "text": text.strip()}
        if "ephemeral_invalidation" in item:
            if item["ephemeral_invalidation"] is not True:
                raise ValueError("ephemeral_invalidation must be true when present")
            fact["ephemeral_invalidation"] = True
        facts.append(fact)
    return tuple(facts)


def _events(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("memory must be an array")
    events: list[dict[str, str]] = []
    fields = {
        "ADD": ("event", "subject", "text"),
        "UPDATE": ("event", "id", "subject", "text"),
        "DELETE": ("event", "id"),
        "NONE": ("event", "id"),
    }
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("Each memory event must be an object")
        event = item.get("event")
        if event not in fields or set(item) != set(fields[event]):
            raise ValueError(f"Invalid {event!r} memory event fields")
        normalized = {str(key): str(item[key]).strip() for key in fields[event]}
        if any(not value for key, value in normalized.items() if key != "event"):
            raise ValueError("Memory event fields must be non-empty strings")
        events.append(normalized)
    return tuple(events)


class Mem0OnePassMethod(MemoryMethod[dict[str, dict[str, str]]]):
    """Joint turn -> extracted facts + CRUD events in one generation."""

    name = "mem0_one_pass"
    # The compatible data root hard-links the joint view under all legacy names.
    source_view = "patch"
    system_prompt = MEM0_ONE_PASS_SYSTEM_PROMPT

    def __init__(self, *, retrieval_k: int = 20) -> None:
        self.retrieval_k = retrieval_k
        self._embedder: Any | None = None
        self._embedding_cache: dict[str, Any] = {}

    def initial_state(self) -> dict[str, dict[str, str]]:
        return {}

    def format_input(self, row: Mapping[str, Any]) -> str:
        value = row.get("input")
        if not isinstance(value, Mapping) or set(value) != {
            "source_batch",
            "retrieved_facts",
        }:
            raise ValueError(
                "Mem0 one-pass input requires source_batch and retrieved_facts"
            )
        return compact_json(value)

    def format_target(self, row: Mapping[str, Any]) -> str:
        target = row.get("target")
        if not isinstance(target, Mapping):
            raise TypeError("target must be an object")
        facts = _facts(target.get("facts"))
        events = _events(target.get("memory"))
        return compact_json({"facts": facts, "memory": events})

    def runtime_input(
        self,
        canonical: Mapping[str, Any],
        state: Mapping[str, Mapping[str, str]],
    ) -> dict[str, Any]:
        """Build test-time top-k retrieval only from predicted active records."""
        canonical_input = canonical.get("input")
        if not isinstance(canonical_input, Mapping):
            raise TypeError("Canonical one-pass input must be an object")
        source_batch = canonical_input.get("source_batch")
        if not isinstance(source_batch, Sequence) or isinstance(
            source_batch, (str, bytes)
        ):
            raise TypeError("source_batch must be an array")
        query = "\n".join(
            f"{turn.get('speaker_name', '')}: {turn.get('text', '')}".strip()
            for turn in source_batch
            if isinstance(turn, Mapping)
        )
        retrieved = self.retrieve_facts(query, state)
        return {
            "source_batch": list(source_batch),
            "retrieved_facts": retrieved,
        }

    def retrieve_facts(
        self, query: str, state: Mapping[str, Mapping[str, str]]
    ) -> list[dict[str, str]]:
        if not state:
            return []
        import numpy as np

        if self._embedder is None:
            from ..prepare_mem0_style_data import LocalSentenceEmbedder

            self._embedder = LocalSentenceEmbedder(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=os.environ.get("PALMCLAW_MEM0_RETRIEVAL_DEVICE", "cpu"),
                batch_size=64,
            )
        query_embedding = self._embedder.encode([query])[0]
        missing: list[tuple[str, str]] = []
        for record_id, fact in state.items():
            text = f"{fact['subject']}: {fact['text']}"
            if text not in self._embedding_cache:
                missing.append((record_id, text))
        if missing:
            matrix = self._embedder.encode([text for _, text in missing])
            for (_, text), embedding in zip(missing, matrix, strict=True):
                self._embedding_cache[text] = embedding
        scored = []
        for record_id, fact in state.items():
            text = f"{fact['subject']}: {fact['text']}"
            scored.append(
                (float(np.dot(query_embedding, self._embedding_cache[text])), record_id)
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": record_id,
                "fact_version_id": f"{record_id}:v01",
                "subject": state[record_id]["subject"],
                "text": state[record_id]["text"],
            }
            for _, record_id in scored[: self.retrieval_k]
        ]

    def parse_output(self, text: str) -> ParsedMemoryOutput:
        value = parse_json_object(text)
        require_exact_fields(value, {"facts", "memory"})
        facts = _facts(value["facts"])
        events = _events(value["memory"])
        decision = "UPDATE" if any(e["event"] != "NONE" for e in events) else "NO_OP"
        return ParsedMemoryOutput(decision, {"facts": facts, "memory": events})

    def apply_output(
        self,
        state: dict[str, dict[str, str]],
        output: ParsedMemoryOutput,
        *,
        turn_id: str | None = None,
    ) -> dict[str, dict[str, str]]:
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
                record_id = event["id"]
                if record_id not in result:
                    raise ValueError(f"Unknown UPDATE record id: {record_id}")
                result[record_id] = {
                    "subject": event["subject"],
                    "text": event["text"],
                }
            elif kind == "DELETE":
                record_id = event["id"]
                if record_id not in result:
                    raise ValueError(f"Unknown DELETE record id: {record_id}")
                del result[record_id]
        return result

    def materialize_memory(self, state: dict[str, dict[str, str]]) -> str:
        grouped: dict[str, list[str]] = defaultdict(list)
        for fact in state.values():
            grouped[fact["subject"]].append(fact["text"])
        return "\n\n".join(
            "\n".join([f"### {subject}", *(f"- {text}" for text in texts)])
            for subject, texts in grouped.items()
        )
