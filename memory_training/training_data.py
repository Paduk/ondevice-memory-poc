"""Chat-template encoding and epoch-plan-aware batching for memory SFT."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from .dataset import IndexedMemoryDataset
from .methods import DeltaV3AppendMethod, MemoryMethod
from .methods.delta_v3_compact import DeltaV3CompactMethod
from .methods.delta_v2 import delta_v2_state_from_input
from .quiz_sft import IndexedQuizSFTDataset, VehicleToolSchemaStore
from .sampling import EpochPlan


@dataclass(frozen=True)
class EncodedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    row_id: int
    scenario_index: int
    decision: str
    task_type: str = "memory"
    sample_id: str = ""


class ChatExampleEncoder:
    """Encode only assistant output into the causal-LM loss."""

    def __init__(self, tokenizer: Any, *, max_length: int = 4096) -> None:
        if max_length < 64:
            raise ValueError("max_length must be at least 64")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(
        self, row: dict[str, Any], method: MemoryMethod[Any], *, row_id: int
    ) -> EncodedExample:
        messages = [
            {"role": "system", "content": method.system_prompt},
            {"role": "user", "content": method.format_input(row)},
        ]
        prompt = self._render(messages, add_generation_prompt=True)
        target = method.format_target(row)
        full = self._render(
            [*messages, {"role": "assistant", "content": target}],
            add_generation_prompt=False,
        )
        prompt_ids = self._tokenize(prompt)
        full_ids = self._tokenize(full)
        boundary = _common_prefix_length(prompt_ids, full_ids)
        if boundary == len(full_ids):
            raise ValueError("Chat template produced no assistant target tokens")
        input_ids, boundary = _left_truncate(full_ids, boundary, self.max_length)
        labels = [-100] * boundary + input_ids[boundary:]
        if all(label == -100 for label in labels):
            raise ValueError("Truncation removed every assistant target token")
        return EncodedExample(
            input_ids=tuple(input_ids),
            labels=tuple(labels),
            row_id=row_id,
            scenario_index=int(row["scenario_index"]),
            decision=str(row["target"]["decision"]),
            task_type="memory",
            sample_id=str(row.get("sample_id", "")),
        )

    def generation_inputs(
        self, row: dict[str, Any], method: MemoryMethod[Any]
    ) -> dict[str, torch.Tensor]:
        prompt = self.generation_prompt(row, method)
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        return {key: value for key, value in encoded.items() if key != "token_type_ids"}

    def generation_prompt(self, row: dict[str, Any], method: MemoryMethod[Any]) -> str:
        messages = [
            {"role": "system", "content": method.system_prompt},
            {"role": "user", "content": method.format_input(row)},
        ]
        return self._render(messages, add_generation_prompt=True)

    def stable_generation_prefix(
        self, row: dict[str, Any], method: MemoryMethod[Any]
    ) -> str:
        """Render the memory-only prefix available before the next turn arrives."""
        content = method.format_input(row)
        prompt = self._render(
            [
                {"role": "system", "content": method.system_prompt},
                {"role": "user", "content": content},
            ],
            add_generation_prompt=True,
        )
        return _stable_current_turn_prefix(prompt, content)

    def cache_anchor_prefixes(
        self, row: dict[str, Any], method: MemoryMethod[Any]
    ) -> tuple[str, ...]:
        """Return increasing immutable prompt prefixes safe to cache.

        Hybrid linear-attention models cannot roll recurrent state backwards.
        They therefore retain exact snapshots at semantic boundaries instead
        of cropping a cache produced for a longer prompt.  Every method can
        retain the chat/system prefix and the complete memory prefix.  Compact
        Delta-v3 additionally exposes the base-memory boundary so the same B
        snapshot remains reusable while pending P batches accumulate.
        """
        content = method.format_input(row)
        prompt = self._render(
            [
                {"role": "system", "content": method.system_prompt},
                {"role": "user", "content": content},
            ],
            add_generation_prompt=True,
        )
        content_start = prompt.find(content)
        if content_start < 0:
            raise ValueError("Rendered memory prompt does not contain its input")
        prefixes = [prompt[:content_start]]
        if isinstance(method, DeltaV3CompactMethod):
            pending_marker = "\nP:\n"
            marker_start = content.find(pending_marker)
            if marker_start < 0:
                raise ValueError("Compact Delta-v3 prompt has no pending boundary")
            prefixes.append(
                prompt[:content_start]
                + content[: marker_start + len(pending_marker)]
            )
        prefixes.append(_stable_current_turn_prefix(prompt, content))
        return tuple(prefixes)

    def _render(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools is not None:
            kwargs["tools"] = tools
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except (TypeError, ValueError):
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])


@dataclass(frozen=True)
class DeltaAppendContext:
    """One target turn's immutable cache-epoch base and preceding row ids."""

    base_summary: str
    history_row_ids: tuple[int, ...]


class DeltaAppendChatExampleEncoder(ChatExampleEncoder):
    """Mask prior turns and train only the latest append-only assistant output."""

    def generation_prompt_append(
        self,
        row: dict[str, Any],
        method: DeltaV3AppendMethod,
        *,
        base_summary: str,
        history_rows: Sequence[dict[str, Any]],
        history_outputs: Sequence[str],
    ) -> str:
        """Render the exact multi-turn prompt shared by training and inference."""
        messages = self._append_messages(
            row,
            method,
            base_summary=base_summary,
            history_rows=history_rows,
            history_outputs=history_outputs,
        )
        return self._render(messages, add_generation_prompt=True)

    def prompt_token_count(self, prompt: str) -> int:
        """Return the untruncated token count used for cache-budget decisions."""
        return len(self._tokenize(prompt))

    def stable_epoch_prefix(
        self,
        row: dict[str, Any],
        method: DeltaV3AppendMethod,
        *,
        base_summary: str,
    ) -> str:
        """Render the fresh-epoch prefix known immediately after compaction."""
        content = method.format_epoch_turn(row, base_summary=base_summary)
        prompt = self._render(
            [
                {"role": "system", "content": method.system_prompt},
                {"role": "user", "content": content},
            ],
            add_generation_prompt=True,
        )
        return _stable_current_turn_prefix(prompt, content)

    def encode_append(
        self,
        row: dict[str, Any],
        method: DeltaV3AppendMethod,
        *,
        context: DeltaAppendContext,
        history_rows: Sequence[dict[str, Any]],
        row_id: int,
    ) -> EncodedExample:
        if len(history_rows) != len(context.history_row_ids):
            raise ValueError("Append-only history rows do not match their context")
        history_outputs = [
            method.format_target(history_row) for history_row in history_rows
        ]
        messages = self._append_messages(
            row,
            method,
            base_summary=context.base_summary,
            history_rows=history_rows,
            history_outputs=history_outputs,
        )
        prompt = self._render(messages, add_generation_prompt=True)
        full = self._render(
            [
                *messages,
                {"role": "assistant", "content": method.format_target(row)},
            ],
            add_generation_prompt=False,
        )
        prompt_ids = self._tokenize(prompt)
        full_ids = self._tokenize(full)
        boundary = _common_prefix_length(prompt_ids, full_ids)
        if boundary == len(full_ids):
            raise ValueError("Chat template produced no assistant target tokens")
        input_ids, boundary = _left_truncate(full_ids, boundary, self.max_length)
        labels = [-100] * boundary + input_ids[boundary:]
        if all(label == -100 for label in labels):
            raise ValueError("Truncation removed every assistant target token")
        return EncodedExample(
            input_ids=tuple(input_ids),
            labels=tuple(labels),
            row_id=row_id,
            scenario_index=int(row["scenario_index"]),
            decision=str(row["target"]["decision"]),
            task_type="memory",
            sample_id=str(row.get("sample_id", "")),
        )

    @staticmethod
    def _append_messages(
        row: dict[str, Any],
        method: DeltaV3AppendMethod,
        *,
        base_summary: str,
        history_rows: Sequence[dict[str, Any]],
        history_outputs: Sequence[str],
    ) -> list[dict[str, Any]]:
        if len(history_rows) != len(history_outputs):
            raise ValueError("Append-only rows and assistant outputs must align")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": method.system_prompt}
        ]
        for index, (history_row, history_output) in enumerate(
            zip(history_rows, history_outputs, strict=True)
        ):
            content = (
                method.format_epoch_turn(history_row, base_summary=base_summary)
                if index == 0
                else method.format_followup_turn(history_row)
            )
            messages.extend(
                (
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": history_output},
                )
            )
        current = (
            method.format_followup_turn(row)
            if history_rows
            else method.format_epoch_turn(row, base_summary=base_summary)
        )
        messages.append({"role": "user", "content": current})
        return messages


class DeltaAppendSFTDataset(Dataset[EncodedExample]):
    """Build bounded append-only histories from the existing aligned Delta view."""

    def __init__(
        self,
        source: IndexedMemoryDataset,
        context_source: IndexedMemoryDataset,
        method: DeltaV3AppendMethod,
        encoder: DeltaAppendChatExampleEncoder,
        *,
        max_history_turns: int = 32,
    ) -> None:
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be positive")
        self.source = source
        self.context_source = context_source
        self.method = method
        self.encoder = encoder
        self.contexts = build_delta_append_contexts(
            context_source, method, max_history_turns=max_history_turns
        )
        missing = [
            source.row_id_at_position(position)
            for position in range(len(source))
            if source.row_id_at_position(position) not in self.contexts
        ]
        if missing:
            raise ValueError(
                f"Append-only contexts are missing target rows: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> EncodedExample:
        row = self.source[index]
        row_id = self.source.row_id_at_position(index)
        context = self.contexts[row_id]
        history_rows = [
            self.context_source[self.context_source.position_for_row_id(history_row_id)]
            for history_row_id in context.history_row_ids
        ]
        return self.encoder.encode_append(
            row,
            self.method,
            context=context,
            history_rows=history_rows,
            row_id=row_id,
        )


def build_delta_append_contexts(
    source: IndexedMemoryDataset,
    method: DeltaV3AppendMethod,
    *,
    max_history_turns: int = 32,
) -> dict[int, DeltaAppendContext]:
    """Create cache epochs ending at five UPDATEs or a bounded turn history."""
    if max_history_turns < 1:
        raise ValueError("max_history_turns must be positive")
    grouped: dict[int, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for position in range(len(source)):
        row = source[position]
        row_id = source.row_id_at_position(position)
        grouped[int(row["scenario_index"])].append(
            (int(row["global_turn_index"]), row_id, row)
        )

    contexts: dict[int, DeltaAppendContext] = {}
    for rows in grouped.values():
        rows.sort(key=lambda item: item[0])
        history: list[int] = []
        updates = 0
        base_summary = ""
        previous_turn: int | None = None
        for global_turn, row_id, row in rows:
            discontinuity = (
                previous_turn is not None and global_turn != previous_turn + 1
            )
            if not history or discontinuity:
                state = delta_v2_state_from_input(
                    row.get("input"), compaction_interval=method.compaction_interval
                )
                base_summary = method.materialize_memory(state)
                history = []
                updates = 0
            contexts[row_id] = DeltaAppendContext(
                base_summary=base_summary,
                history_row_ids=tuple(history),
            )
            history.append(row_id)
            decision = row.get("target", {}).get("decision")
            if decision == "UPDATE":
                updates += 1
            elif decision != "NO_OP":
                raise ValueError(f"Invalid append-only target decision at row {row_id}")
            if (
                updates >= method.compaction_interval
                or len(history) >= max_history_turns
            ):
                history = []
                updates = 0
            previous_turn = global_turn
    return contexts


def _stable_current_turn_prefix(prompt: str, content: str) -> str:
    markers = ('"current_turn":', "\nT:\n")
    marker = next((value for value in markers if value in content), None)
    if marker is None:
        raise ValueError("Rendered memory prompt has no stable current_turn boundary")
    dynamic_start = content.find(marker)
    content_start = prompt.find(content)
    if content_start < 0:
        raise ValueError("Rendered memory prompt has no stable current_turn boundary")
    return prompt[:content_start] + content[: dynamic_start + len(marker)]


class QuizChatExampleEncoder(ChatExampleEncoder):
    """Encode assistant Tool Calls or natural-language QA targets."""

    def encode(
        self,
        row: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        *,
        row_id: int,
    ) -> EncodedExample:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("Quiz SFT row has no chat messages")
        assistant = messages[-1]
        if assistant.get("role") != "assistant":
            raise ValueError("Quiz SFT row has no assistant target")
        has_tool_calls = bool(assistant.get("tool_calls"))
        has_answer = isinstance(assistant.get("content"), str) and bool(
            assistant["content"].strip()
        )
        if not has_tool_calls and not has_answer:
            raise ValueError("Quiz SFT row has neither Tool Call nor text target")
        if has_tool_calls and has_answer:
            raise ValueError("Quiz SFT row mixes Tool Call and text targets")
        prompt = self._render(messages[:-1], add_generation_prompt=True, tools=tools)
        full = self._render(messages, add_generation_prompt=False, tools=tools)
        prompt_ids = self._tokenize(prompt)
        full_ids = self._tokenize(full)
        boundary = _common_prefix_length(prompt_ids, full_ids)
        if boundary == len(full_ids):
            raise ValueError("Chat template produced no Quiz target tokens")
        input_ids, boundary = _left_truncate(full_ids, boundary, self.max_length)
        labels = [-100] * boundary + input_ids[boundary:]
        if all(label == -100 for label in labels):
            raise ValueError("Truncation removed every Tool Call target token")
        return EncodedExample(
            input_ids=tuple(input_ids),
            labels=tuple(labels),
            row_id=row_id,
            scenario_index=int(row["scenario_index"]),
            decision="TOOL_CALL" if has_tool_calls else "ANSWER",
            task_type="quiz",
            sample_id=str(row.get("sample_id", "")),
        )


class MethodSFTDataset(Dataset[EncodedExample]):
    def __init__(
        self,
        source: IndexedMemoryDataset,
        method: MemoryMethod[Any],
        encoder: ChatExampleEncoder,
    ) -> None:
        self.source = source
        self.method = method
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> EncodedExample:
        row = self.source[index]
        if not isinstance(row, dict):
            raise TypeError("Expected one source row")
        # Canonical sample_id is a string, while plans use integer catalog row ids.
        row_id = self.source.row_id_at_position(index)
        return self.encoder.encode(row, self.method, row_id=row_id)


class QuizSFTDataset(Dataset[EncodedExample]):
    def __init__(
        self,
        source: IndexedQuizSFTDataset,
        tools: VehicleToolSchemaStore | None,
        encoder: QuizChatExampleEncoder,
    ) -> None:
        self.source = source
        self.tools = tools
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> EncodedExample:
        row = self.source[index]
        if not isinstance(row, dict):
            raise TypeError("Expected one Quiz SFT row")
        selected_tools = self.tools.tools_for(row) if self.tools is not None else None
        return self.encoder.encode(row, selected_tools, row_id=index)


class MultitaskSFTDataset(Dataset[EncodedExample]):
    """Expose Memory and Quiz datasets through one non-overlapping index space."""

    def __init__(
        self, memory: Dataset[EncodedExample], quiz: Dataset[EncodedExample]
    ) -> None:
        if not len(memory) or not len(quiz):
            raise ValueError(
                "Multitask training requires non-empty Memory and Quiz data"
            )
        self.memory = memory
        self.quiz = quiz

    def __len__(self) -> int:
        return len(self.memory) + len(self.quiz)

    def __getitem__(self, index: int) -> EncodedExample:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.memory):
            return self.memory[index]
        return self.quiz[index - len(self.memory)]


class SFTCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: Sequence[EncodedExample]) -> dict[str, Any]:
        width = max(len(example.input_ids) for example in examples)
        input_ids = []
        labels = []
        attention_mask = []
        for example in examples:
            padding = width - len(example.input_ids)
            input_ids.append((*example.input_ids, *([self.pad_token_id] * padding)))
            labels.append((*example.labels, *([-100] * padding)))
            attention_mask.append((*([1] * len(example.input_ids)), *([0] * padding)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "metadata": [
                {
                    "row_id": example.row_id,
                    "scenario_index": example.scenario_index,
                    "decision": example.decision,
                    "task_type": example.task_type,
                    "sample_id": example.sample_id,
                }
                for example in examples
            ],
        }


class PlannedBatchSampler(Sampler[list[int]]):
    """Keep trajectory windows ordered while batching independent samples."""

    def __init__(
        self,
        source: IndexedMemoryDataset,
        plan: EpochPlan,
        *,
        batch_size: int,
        start_batch: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batches = self._build(source, plan, batch_size)
        if not 0 <= start_batch <= len(self.batches):
            raise ValueError("start_batch is outside the epoch plan")
        self.start_batch = start_batch

    @staticmethod
    def _build(
        source: IndexedMemoryDataset, plan: EpochPlan, batch_size: int
    ) -> tuple[tuple[int, ...], ...]:
        batches: list[tuple[int, ...]] = []
        independent: list[int] = []

        def flush() -> None:
            nonlocal independent
            while independent:
                batches.append(tuple(independent[:batch_size]))
                independent = independent[batch_size:]

        for unit in plan.units:
            positions = [source.position_for_row_id(row_id) for row_id in unit.row_ids]
            if unit.kind == "independent":
                independent.extend(positions)
                if len(independent) >= batch_size:
                    batches.append(tuple(independent[:batch_size]))
                    independent = independent[batch_size:]
                continue
            flush()
            batches.extend(
                tuple(positions[start : start + batch_size])
                for start in range(0, len(positions), batch_size)
            )
        flush()
        return tuple(batches)

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.batches[self.start_batch :]:
            yield list(batch)

    def __len__(self) -> int:
        return len(self.batches) - self.start_batch


class BalancedMultitaskBatchSampler(Sampler[list[int]]):
    """Distribute a capped Quiz schedule evenly among planned Memory batches."""

    def __init__(
        self,
        memory_batches: PlannedBatchSampler,
        *,
        memory_size: int,
        quiz_size: int,
        quiz_indices: Sequence[int],
        quiz_batch_size: int,
        start_batch: int = 0,
    ) -> None:
        if memory_size < 1 or quiz_size < 1:
            raise ValueError("Memory and Quiz sizes must be positive")
        if quiz_batch_size < 1:
            raise ValueError("Quiz batch size must be positive")
        if any(not 0 <= index < quiz_size for index in quiz_indices):
            raise IndexError("Quiz schedule contains an out-of-range index")
        memory = [tuple(batch) for batch in memory_batches]
        quiz_batches = [
            tuple(memory_size + index for index in quiz_indices[start:stop])
            for start in range(0, len(quiz_indices), quiz_batch_size)
            for stop in (min(start + quiz_batch_size, len(quiz_indices)),)
        ]

        batches: list[tuple[int, ...]] = []
        quiz_cursor = 0
        for memory_index, memory_batch in enumerate(memory, start=1):
            batches.append(memory_batch)
            target = memory_index * len(quiz_batches) // len(memory)
            while quiz_cursor < target:
                batches.append(quiz_batches[quiz_cursor])
                quiz_cursor += 1
        batches.extend(quiz_batches[quiz_cursor:])
        if not 0 <= start_batch <= len(batches):
            raise ValueError("start_batch is outside the multitask epoch plan")
        self.batches = tuple(batches)
        self.start_batch = start_batch

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.batches[self.start_batch :]:
            yield list(batch)

    def __len__(self) -> int:
        return len(self.batches) - self.start_batch


def quiz_epoch_indices(
    quiz_size: int,
    *,
    epoch: int,
    epochs: int,
    total_passes: int,
    seed: int,
) -> tuple[int, ...]:
    """Partition exact shuffled full-dataset passes across all training epochs."""
    if quiz_size < 1 or epochs < 1 or total_passes < 1:
        raise ValueError("Quiz size, epochs, and total passes must be positive")
    if not 0 <= epoch < epochs:
        raise ValueError("epoch is outside the Quiz schedule")
    schedule: list[int] = []
    for pass_index in range(total_passes):
        order = list(range(quiz_size))
        random.Random(seed + pass_index * 1_000_003).shuffle(order)
        schedule.extend(order)
    start = len(schedule) * epoch // epochs
    stop = len(schedule) * (epoch + 1) // epochs
    return tuple(schedule[start:stop])


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    boundary = 0
    for left_id, right_id in zip(left, right, strict=False):
        if left_id != right_id:
            break
        boundary += 1
    return boundary


def _left_truncate(
    full_ids: list[int], boundary: int, max_length: int
) -> tuple[list[int], int]:
    if len(full_ids) <= max_length:
        return full_ids, boundary
    target_length = len(full_ids) - boundary
    if target_length >= max_length:
        # Preserve the start of structured JSON rather than a suffix with no schema.
        return full_ids[boundary : boundary + max_length], 0
    keep_prompt = max_length - target_length
    start = boundary - keep_prompt
    return full_ids[start:], keep_prompt
