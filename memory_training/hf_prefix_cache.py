"""Batch-1 greedy HF generation with reusable cross-turn prefix caches."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrefixReusePlan:
    logical_prompt_tokens: int
    candidate_prefix_tokens: int
    reused_prefix_tokens: int
    evaluated_prefill_tokens: int
    cropped_tokens: int


@dataclass(frozen=True)
class PrefixGenerationResult:
    text: str
    logical_prompt_tokens: int
    candidate_prefix_tokens: int
    reused_prefix_tokens: int
    evaluated_prefill_tokens: int
    decode_tokens: int
    tokenization_seconds: float
    cache_management_seconds: float
    prefill_seconds: float
    decode_seconds: float
    model_seconds: float
    ttft_seconds: float
    end_to_end_seconds: float
    kv_cache_bytes: int
    cache_length: int
    peak_cuda_allocated_bytes: int


@dataclass(frozen=True)
class PrefixPrefillResult:
    """One cache-only prefill, normally executed between user turns."""

    logical_prefix_tokens: int
    evaluated_prefill_tokens: int
    tokenization_seconds: float
    cache_management_seconds: float
    prefill_seconds: float
    end_to_end_seconds: float
    kv_cache_bytes: int
    cache_length: int
    peak_cuda_allocated_bytes: int


class PrefixCacheState:
    """Own one scenario's exact cached-token identity and mutable HF cache."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.previous_prompt_ids: tuple[int, ...] = ()
        self.cache: Any | None = None

    def reset(self) -> None:
        self.previous_prompt_ids = ()
        self.cache = None

    def prepare(self, prompt_ids: tuple[int, ...]) -> PrefixReusePlan:
        if not prompt_ids:
            raise ValueError("Generation prompt must contain at least one token")
        candidate = _common_prefix_length(self.previous_prompt_ids, prompt_ids)
        reused = 0
        cropped = 0
        if self.enabled and self.cache is not None:
            # A forward pass over at least one prompt token is required to obtain
            # logits for the first generated token.
            reused = min(candidate, len(prompt_ids) - 1)
            cache_length = _cache_length(self.cache)
            if cache_length < reused:
                raise ValueError(
                    f"Cache is shorter than reusable prefix: {cache_length} < {reused}"
                )
            cropped = cache_length - reused
            _crop_cache(self.cache, cropped)
        return PrefixReusePlan(
            logical_prompt_tokens=len(prompt_ids),
            candidate_prefix_tokens=candidate,
            reused_prefix_tokens=reused,
            evaluated_prefill_tokens=len(prompt_ids) - reused,
            cropped_tokens=cropped,
        )

    def commit(self, cached_token_ids: tuple[int, ...], cache: Any) -> None:
        if self.enabled:
            cache_length = _cache_length(cache)
            if cache_length != len(cached_token_ids):
                raise ValueError(
                    "Cached token identity does not match HF cache length: "
                    f"{len(cached_token_ids)} != {cache_length}"
                )
            self.previous_prompt_ids = cached_token_ids
            self.cache = cache
        else:
            self.reset()


class HFPrefixCacheGenerator:
    """Generate one dependent trajectory while retaining exact token prefixes."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_length: int,
        max_new_tokens: int,
        cache_enabled: bool,
        device: Any | None = None,
    ) -> None:
        if max_length < 1 or max_new_tokens < 1:
            raise ValueError("Generation lengths must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = device or next(model.parameters()).device
        self.state = PrefixCacheState(enabled=cache_enabled)
        self.eos_token_ids = _token_ids(tokenizer.eos_token_id)
        self.snapshot_cache_enabled = cache_enabled and _uses_linear_attention(model)
        self._snapshot_caches: OrderedDict[tuple[int, ...], Any] = OrderedDict()
        self._max_snapshot_caches = 8

    @property
    def cache_enabled(self) -> bool:
        return self.state.enabled

    @property
    def cache_strategy(self) -> str:
        if not self.cache_enabled:
            return "off"
        return "immutable_prefix_snapshots" if self.snapshot_cache_enabled else "crop"

    def reset(self) -> None:
        self.state.reset()
        self._snapshot_caches.clear()

    def prefill(
        self,
        prompt_prefix: str,
        *,
        cache_anchor_prefixes: Sequence[str] = (),
        reference_prompt: str | None = None,
    ) -> PrefixPrefillResult:
        """Build a reusable prefix cache without decoding an assistant output."""
        if not self.cache_enabled:
            raise RuntimeError("Cache-only prefill requires cache mode ON")
        if self.snapshot_cache_enabled:
            return self._prefill_with_snapshots(
                prompt_prefix,
                cache_anchor_prefixes=cache_anchor_prefixes,
                reference_prompt=reference_prompt,
            )
        end_to_end_started = time.perf_counter()
        tokenization_started = time.perf_counter()
        prompt_tensor, prompt_ids = self._tokenize_prompt(prompt_prefix)
        tokenization_seconds = time.perf_counter() - tokenization_started

        cache_started = time.perf_counter()
        plan = self.state.prepare(prompt_ids)
        cache = self.state.cache
        cache_management_seconds = time.perf_counter() - cache_started
        suffix = prompt_tensor[:, plan.reused_prefix_tokens :]
        if suffix.shape[-1] == 0:
            raise ValueError("Cache-only prefill requires at least one evaluated token")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        _synchronize(self.device)
        prefill_started = time.perf_counter()
        with torch.inference_mode():
            output = self._forward(suffix, cache)
            cache = output.past_key_values
        _synchronize(self.device)
        prefill_seconds = time.perf_counter() - prefill_started
        self.state.commit(prompt_ids, cache)
        return PrefixPrefillResult(
            logical_prefix_tokens=len(prompt_ids),
            evaluated_prefill_tokens=plan.evaluated_prefill_tokens,
            tokenization_seconds=tokenization_seconds,
            cache_management_seconds=cache_management_seconds,
            prefill_seconds=prefill_seconds,
            end_to_end_seconds=time.perf_counter() - end_to_end_started,
            kv_cache_bytes=_tensor_bytes(cache),
            cache_length=_cache_length(cache),
            peak_cuda_allocated_bytes=(
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            ),
        )

    def generate(
        self,
        prompt: str,
        *,
        cache_anchor_prefixes: Sequence[str] = (),
    ) -> PrefixGenerationResult:
        if self.snapshot_cache_enabled:
            return self._generate_with_snapshots(
                prompt,
                cache_anchor_prefixes=cache_anchor_prefixes,
            )
        end_to_end_started = time.perf_counter()
        tokenization_started = time.perf_counter()
        prompt_tensor, prompt_ids = self._tokenize_prompt(prompt)
        tokenization_seconds = time.perf_counter() - tokenization_started

        cache_started = time.perf_counter()
        plan = self.state.prepare(prompt_ids)
        cache = self.state.cache if self.cache_enabled else None
        cache_management_seconds = time.perf_counter() - cache_started
        suffix = prompt_tensor[:, plan.reused_prefix_tokens :]

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        _synchronize(self.device)
        prefill_started = time.perf_counter()
        with torch.inference_mode():
            output = self._forward(suffix, cache)
            cache = output.past_key_values
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
        _synchronize(self.device)
        prefill_seconds = time.perf_counter() - prefill_started
        generated_ids = [int(next_token.item())]

        decode_started = time.perf_counter()
        with torch.inference_mode():
            while (
                len(generated_ids) < self.max_new_tokens
                and generated_ids[-1] not in self.eos_token_ids
            ):
                output = self._forward(next_token, cache)
                cache = output.past_key_values
                next_token = torch.argmax(
                    output.logits[:, -1, :], dim=-1, keepdim=True
                )
                generated_ids.append(int(next_token.item()))
        _synchronize(self.device)
        decode_seconds = time.perf_counter() - decode_started
        # The final sampled token has not been forwarded through the model yet.
        # Everything before it is present in past_key_values and can be reused
        # when an append-only chat template retains the assistant output.
        cached_token_ids = (*prompt_ids, *generated_ids[:-1])
        self.state.commit(cached_token_ids, cache)

        text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        end_to_end_seconds = time.perf_counter() - end_to_end_started
        peak_cuda = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        cache_length = _cache_length(cache)
        return PrefixGenerationResult(
            text=text,
            logical_prompt_tokens=plan.logical_prompt_tokens,
            candidate_prefix_tokens=plan.candidate_prefix_tokens,
            reused_prefix_tokens=plan.reused_prefix_tokens,
            evaluated_prefill_tokens=plan.evaluated_prefill_tokens,
            decode_tokens=len(generated_ids),
            tokenization_seconds=tokenization_seconds,
            cache_management_seconds=cache_management_seconds,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            model_seconds=prefill_seconds + decode_seconds,
            ttft_seconds=(
                tokenization_seconds + cache_management_seconds + prefill_seconds
            ),
            end_to_end_seconds=end_to_end_seconds,
            kv_cache_bytes=_tensor_bytes(cache),
            cache_length=cache_length,
            peak_cuda_allocated_bytes=peak_cuda,
        )

    def _prefill_with_snapshots(
        self,
        prompt_prefix: str,
        *,
        cache_anchor_prefixes: Sequence[str],
        reference_prompt: str | None,
    ) -> PrefixPrefillResult:
        end_to_end_started = time.perf_counter()
        tokenization_started = time.perf_counter()
        prompt_tensor, prompt_ids = self._tokenize_prompt(
            reference_prompt or prompt_prefix
        )
        anchors = self._snapshot_anchor_ids(
            prompt_ids,
            (*cache_anchor_prefixes, prompt_prefix),
            keep_prompt_token=reference_prompt is None,
        )
        tokenization_seconds = time.perf_counter() - tokenization_started
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        cache, reused, prefill_seconds, cache_management_seconds = (
            self._build_snapshot_anchors(prompt_tensor, prompt_ids, anchors)
        )
        if cache is None:
            raise RuntimeError("Snapshot prefill did not produce an HF cache")
        logical_prefix_tokens = len(anchors[-1])
        return PrefixPrefillResult(
            logical_prefix_tokens=logical_prefix_tokens,
            evaluated_prefill_tokens=logical_prefix_tokens - reused,
            tokenization_seconds=tokenization_seconds,
            cache_management_seconds=cache_management_seconds,
            prefill_seconds=prefill_seconds,
            end_to_end_seconds=time.perf_counter() - end_to_end_started,
            kv_cache_bytes=_tensor_bytes(tuple(self._snapshot_caches.values())),
            cache_length=_cache_length(cache),
            peak_cuda_allocated_bytes=(
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            ),
        )

    def _generate_with_snapshots(
        self,
        prompt: str,
        *,
        cache_anchor_prefixes: Sequence[str],
    ) -> PrefixGenerationResult:
        end_to_end_started = time.perf_counter()
        tokenization_started = time.perf_counter()
        prompt_tensor, prompt_ids = self._tokenize_prompt(prompt)
        anchors = self._snapshot_anchor_ids(
            prompt_ids,
            cache_anchor_prefixes,
            keep_prompt_token=False,
        )
        tokenization_seconds = time.perf_counter() - tokenization_started
        if not anchors:
            raise ValueError(
                "Linear-attention prefix caching requires a non-empty cache anchor"
            )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        cache, reused, anchor_prefill_seconds, cache_management_seconds = (
            self._build_snapshot_anchors(prompt_tensor, prompt_ids, anchors)
        )
        anchor_length = len(anchors[-1])
        suffix = prompt_tensor[:, anchor_length:]
        if suffix.shape[-1] == 0:
            raise ValueError("Generation requires at least one evaluated prompt token")

        _synchronize(self.device)
        prefill_started = time.perf_counter()
        with torch.inference_mode():
            output = self._forward(suffix, cache)
            cache = output.past_key_values
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
        _synchronize(self.device)
        suffix_prefill_seconds = time.perf_counter() - prefill_started
        generated_ids = [int(next_token.item())]

        decode_started = time.perf_counter()
        with torch.inference_mode():
            while (
                len(generated_ids) < self.max_new_tokens
                and generated_ids[-1] not in self.eos_token_ids
            ):
                output = self._forward(next_token, cache)
                cache = output.past_key_values
                next_token = torch.argmax(
                    output.logits[:, -1, :], dim=-1, keepdim=True
                )
                generated_ids.append(int(next_token.item()))
        _synchronize(self.device)
        decode_seconds = time.perf_counter() - decode_started
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        prefill_seconds = anchor_prefill_seconds + suffix_prefill_seconds
        end_to_end_seconds = time.perf_counter() - end_to_end_started
        peak_cuda = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return PrefixGenerationResult(
            text=text,
            logical_prompt_tokens=len(prompt_ids),
            candidate_prefix_tokens=anchor_length,
            reused_prefix_tokens=reused,
            evaluated_prefill_tokens=len(prompt_ids) - reused,
            decode_tokens=len(generated_ids),
            tokenization_seconds=tokenization_seconds,
            cache_management_seconds=cache_management_seconds,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            model_seconds=prefill_seconds + decode_seconds,
            ttft_seconds=(
                tokenization_seconds + cache_management_seconds + prefill_seconds
            ),
            end_to_end_seconds=end_to_end_seconds,
            kv_cache_bytes=_tensor_bytes(
                (cache, tuple(self._snapshot_caches.values()))
            ),
            cache_length=_cache_length(cache),
            peak_cuda_allocated_bytes=peak_cuda,
        )

    def _snapshot_anchor_ids(
        self,
        prompt_ids: tuple[int, ...],
        prefixes: Sequence[str],
        *,
        keep_prompt_token: bool,
    ) -> tuple[tuple[int, ...], ...]:
        maximum = len(prompt_ids) if keep_prompt_token else len(prompt_ids) - 1
        anchors: set[tuple[int, ...]] = set()
        for prefix in prefixes:
            encoded = self.tokenizer(
                prefix,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )
            raw_ids = encoded["input_ids"]
            if torch.is_tensor(raw_ids) and raw_ids.ndim == 2:
                raw_ids = raw_ids[0]
            prefix_ids = tuple(int(value) for value in raw_ids)
            length = min(
                _common_prefix_length(prefix_ids, prompt_ids),
                maximum,
            )
            if length > 0:
                anchors.add(prompt_ids[:length])
        return tuple(sorted(anchors, key=len))

    def _build_snapshot_anchors(
        self,
        prompt_tensor: torch.Tensor,
        prompt_ids: tuple[int, ...],
        anchors: Sequence[tuple[int, ...]],
    ) -> tuple[Any, int, float, float]:
        if not anchors:
            raise ValueError("At least one snapshot anchor is required")
        target_length = len(anchors[-1])
        management_seconds = 0.0
        management_started = time.perf_counter()
        matching = [
            key
            for key in self._snapshot_caches
            if len(key) <= target_length and prompt_ids[: len(key)] == key
        ]
        seed = max(matching, key=len) if matching else ()
        cache = deepcopy(self._snapshot_caches[seed]) if seed else None
        if seed:
            self._snapshot_caches.move_to_end(seed)
        management_seconds += time.perf_counter() - management_started
        reused = len(seed)
        cursor = reused
        prefill_seconds = 0.0
        for anchor in anchors:
            if len(anchor) <= cursor:
                continue
            suffix = prompt_tensor[:, cursor : len(anchor)]
            _synchronize(self.device)
            prefill_started = time.perf_counter()
            with torch.inference_mode():
                output = self._forward(suffix, cache)
                cache = output.past_key_values
            _synchronize(self.device)
            prefill_seconds += time.perf_counter() - prefill_started
            cursor = len(anchor)
            management_started = time.perf_counter()
            self._store_snapshot(anchor, cache)
            management_seconds += time.perf_counter() - management_started
        if cursor != target_length:
            raise RuntimeError("Snapshot cache did not reach its target anchor")
        return cache, reused, prefill_seconds, management_seconds

    def _store_snapshot(self, token_ids: tuple[int, ...], cache: Any) -> None:
        self._snapshot_caches[token_ids] = deepcopy(cache)
        self._snapshot_caches.move_to_end(token_ids)
        while len(self._snapshot_caches) > self._max_snapshot_caches:
            self._snapshot_caches.popitem(last=False)

    def _tokenize_prompt(self, prompt: str) -> tuple[torch.Tensor, tuple[int, ...]]:
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        prompt_tensor = encoded["input_ids"].to(self.device)
        if prompt_tensor.shape[0] != 1:
            raise ValueError("HF prefix-cache generation requires batch size 1")
        prompt_ids = tuple(int(value) for value in prompt_tensor[0].tolist())
        if not prompt_ids:
            raise ValueError("HF prefix-cache prompt must contain at least one token")
        return prompt_tensor, prompt_ids

    def _forward(self, input_ids: torch.Tensor, cache: Any | None) -> Any:
        past_length = _cache_length(cache) if cache is not None else 0
        token_count = int(input_ids.shape[-1])
        attention_mask = torch.ones(
            (1, past_length + token_count),
            dtype=torch.long,
            device=self.device,
        )
        cache_position = torch.arange(
            past_length,
            past_length + token_count,
            dtype=torch.long,
            device=self.device,
        )
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )


def _common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    length = 0
    for left_id, right_id in zip(left, right, strict=False):
        if left_id != right_id:
            break
        length += 1
    return length


def _uses_linear_attention(model: Any) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    getter = getattr(config, "get_text_config", None)
    text_config = getter(decoder=True) if callable(getter) else config
    layer_types = getattr(text_config, "layer_types", ()) or ()
    return any("linear_attention" in str(layer_type) for layer_type in layer_types)


def _cache_length(cache: Any | None) -> int:
    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        raise TypeError("HF cache does not expose get_seq_length()")
    return int(getter())


def _crop_cache(cache: Any, tokens_to_remove: int) -> None:
    if tokens_to_remove < 0:
        raise ValueError("Cannot extend an HF cache through crop")
    crop = getattr(cache, "crop", None)
    if not callable(crop):
        raise TypeError("HF cache does not support crop()")
    # Transformers 5.x uses a negative value to mean "remove this many".
    # Positive values retain the legacy "crop to absolute length" meaning.
    crop(-tokens_to_remove)


def _token_ids(value: int | list[int] | tuple[int, ...] | None) -> frozenset[int]:
    if value is None:
        return frozenset()
    if isinstance(value, int):
        return frozenset({value})
    return frozenset(int(token_id) for token_id in value)


def _synchronize(device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_bytes(value: Any) -> int:
    seen_objects: set[int] = set()
    seen_tensors: set[int] = set()

    def visit(item: Any) -> int:
        if item is None or isinstance(item, (str, bytes, int, float, bool)):
            return 0
        object_id = id(item)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if torch.is_tensor(item):
            storage_id = item.untyped_storage().data_ptr()
            if storage_id in seen_tensors:
                return 0
            seen_tensors.add(storage_id)
            return int(item.untyped_storage().nbytes())
        if isinstance(item, dict):
            return sum(visit(key) + visit(content) for key, content in item.items())
        if isinstance(item, (list, tuple, set)):
            return sum(visit(content) for content in item)
        attributes = getattr(item, "__dict__", None)
        return visit(attributes) if isinstance(attributes, dict) else 0

    return visit(value)
