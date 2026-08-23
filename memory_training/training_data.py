"""Chat-template encoding and epoch-plan-aware batching for memory SFT."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from .dataset import IndexedMemoryDataset
from .methods import MemoryMethod
from .sampling import EpochPlan


@dataclass(frozen=True)
class EncodedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    row_id: int
    scenario_index: int
    decision: str


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
        )

    def generation_inputs(
        self, row: dict[str, Any], method: MemoryMethod[Any]
    ) -> dict[str, torch.Tensor]:
        messages = [
            {"role": "system", "content": method.system_prompt},
            {"role": "user", "content": method.format_input(row)},
        ]
        prompt = self._render(messages, add_generation_prompt=True)
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        return {key: value for key, value in encoded.items() if key != "token_type_ids"}

    def _render(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool
    ) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except (TypeError, ValueError):
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])


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
