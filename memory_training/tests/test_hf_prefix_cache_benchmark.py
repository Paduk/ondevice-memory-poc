from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from memory_training.benchmark_hf_prefix_cache import (
    _compare_reference,
    aggregate_records,
)
from memory_training.cache_benchmark_manifest import (
    build_manifest,
    validate_manifest,
)
from memory_training.compare_hf_prefix_cache import (
    build_comparison,
    render_markdown,
)
from memory_training.dataset import DatasetCatalog, build_catalog
from memory_training.delta_append_runtime import DeltaAppendPromptSession
from memory_training.hf_prefix_cache import HFPrefixCacheGenerator, PrefixCacheState
from memory_training.methods import DeltaV3AppendMethod, DeltaV3CompactK5Method
from memory_training.training_data import ChatExampleEncoder, DeltaAppendChatExampleEncoder


class FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length
        self.crop_calls: list[int] = []

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, tokens_to_remove: int) -> None:
        self.crop_calls.append(tokens_to_remove)
        self.length -= abs(tokens_to_remove)


class CharacterTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_: object,
    ) -> str:
        assert tokenize is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}


class TensorCharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(
                [[ord(character) for character in text]], dtype=torch.long
            )
        }

    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(chr(token_id) for token_id in token_ids if token_id)


class FakeModel:
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: FakeCache | None,
        **_: object,
    ) -> SimpleNamespace:
        cache = past_key_values or FakeCache(0)
        cache.length += int(input_ids.shape[-1])
        logits = torch.zeros((1, input_ids.shape[-1], 256))
        logits[..., 0] = 1
        return SimpleNamespace(past_key_values=cache, logits=logits)


class FakeLinearAttentionConfig:
    def get_text_config(self, *, decoder: bool) -> SimpleNamespace:
        assert decoder is True
        return SimpleNamespace(layer_types=["linear_attention", "full_attention"])


class FakeLinearAttentionModel(FakeModel):
    config = FakeLinearAttentionConfig()


def _write_views(root: Path) -> None:
    handles = {
        view: (root / f"{view}.jsonl").open("w", encoding="utf-8")
        for view in ("summary", "patch", "delta")
    }
    try:
        for turn in range(12):
            decision = "UPDATE" if turn in {2, 7} else "NO_OP"
            for view, handle in handles.items():
                handle.write(
                    json.dumps(
                        {
                            "sample_id": f"s081:{view}:{turn:05d}",
                            "scenario_index": 81,
                            "split": "validation",
                            "global_turn_index": turn,
                            "turn_id": f"s81-turn-{turn}",
                            "current_turn": {"text": f"turn {turn}"},
                            "input": {},
                            "target": {
                                "decision": decision,
                                **(
                                    {"next_memory": "memory"}
                                    if view == "summary"
                                    else {"operations": []}
                                ),
                            },
                        }
                    )
                    + "\n"
                )
    finally:
        for handle in handles.values():
            handle.close()
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "quiz_manifest.json").write_text("{}", encoding="utf-8")


def test_prefix_cache_state_crops_generation_and_changed_prompt_suffix() -> None:
    state = PrefixCacheState(enabled=True)
    state.previous_prompt_ids = (1, 2, 3, 4, 5)
    cache = FakeCache(8)
    state.cache = cache

    plan = state.prepare((1, 2, 3, 9, 10))

    assert plan.candidate_prefix_tokens == 3
    assert plan.reused_prefix_tokens == 3
    assert plan.evaluated_prefill_tokens == 2
    assert plan.cropped_tokens == 5
    assert cache.crop_calls == [-5]
    assert cache.length == 3


def test_prefix_cache_state_keeps_one_token_for_first_token_logits() -> None:
    state = PrefixCacheState(enabled=True)
    state.previous_prompt_ids = (1, 2, 3)
    state.cache = FakeCache(5)

    plan = state.prepare((1, 2, 3))

    assert plan.candidate_prefix_tokens == 3
    assert plan.reused_prefix_tokens == 2
    assert plan.evaluated_prefill_tokens == 1


def test_disabled_prefix_cache_never_reuses_tokens() -> None:
    state = PrefixCacheState(enabled=False)
    state.previous_prompt_ids = (1, 2, 3)
    state.cache = FakeCache(5)

    plan = state.prepare((1, 2, 4))

    assert plan.candidate_prefix_tokens == 2
    assert plan.reused_prefix_tokens == 0
    assert plan.evaluated_prefill_tokens == 3


def test_prefix_cache_identity_includes_forwarded_generated_tokens() -> None:
    state = PrefixCacheState(enabled=True)
    state.commit((1, 2, 3, 4, 5), FakeCache(5))

    plan = state.prepare((1, 2, 3, 4, 5, 6, 7))

    assert plan.candidate_prefix_tokens == 5
    assert plan.reused_prefix_tokens == 5
    assert plan.evaluated_prefill_tokens == 2


def test_cache_only_prefill_is_reused_by_next_generation() -> None:
    generator = HFPrefixCacheGenerator(
        FakeModel(),
        TensorCharacterTokenizer(),
        max_length=64,
        max_new_tokens=1,
        cache_enabled=True,
        device=torch.device("cpu"),
    )

    background = generator.prefill("memory-prefix")
    generated = generator.generate("memory-prefix-current-turn")

    assert background.evaluated_prefill_tokens == len("memory-prefix")
    assert generated.reused_prefix_tokens == len("memory-prefix")
    assert generated.evaluated_prefill_tokens == len("-current-turn")


def test_linear_attention_uses_immutable_anchor_snapshots() -> None:
    generator = HFPrefixCacheGenerator(
        FakeLinearAttentionModel(),
        TensorCharacterTokenizer(),
        max_length=64,
        max_new_tokens=1,
        cache_enabled=True,
        device=torch.device("cpu"),
    )

    first = generator.generate("stable-first", cache_anchor_prefixes=("stable-",))
    second = generator.generate("stable-second", cache_anchor_prefixes=("stable-",))

    assert generator.cache_strategy == "immutable_prefix_snapshots"
    assert first.reused_prefix_tokens == 0
    assert second.reused_prefix_tokens == len("stable-")
    assert second.evaluated_prefill_tokens == len("second")


def test_compact_delta_exposes_base_and_full_state_cache_anchors() -> None:
    encoder = ChatExampleEncoder(CharacterTokenizer(), max_length=4096)
    method = DeltaV3CompactK5Method()
    row = {
        "turn_id": "turn-1",
        "timestamp": "2026-01-01T00:00:00Z",
        "current_turn": {
            "speaker_id": "p1",
            "speaker_name": "Alex",
            "text": "Remember this.",
        },
        "input": {"base_summary": "base fact", "pending_updates": []},
    }

    static, base, full_state = encoder.cache_anchor_prefixes(row, method)

    assert base.startswith(static)
    assert full_state.startswith(base)
    assert base.endswith("B:\nbase fact\nP:\n")
    assert full_state.endswith("\nT:\n")


def _append_row(turn: int, *, decision: str = "NO_OP") -> dict[str, object]:
    return {
        "scenario_index": 81,
        "global_turn_index": turn,
        "turn_id": f"turn-{turn}",
        "timestamp": f"2026-01-01T00:00:{turn:02d}Z",
        "current_turn": {
            "speaker_id": "p1",
            "speaker_name": "Alex",
            "text": f"Turn {turn}",
        },
        "target": {"decision": decision},
    }


def test_delta_append_session_retains_outputs_and_resets_after_five_updates() -> None:
    method = DeltaV3AppendMethod()
    encoder = DeltaAppendChatExampleEncoder(CharacterTokenizer(), max_length=4096)
    session = DeltaAppendPromptSession(method, encoder)
    stable_prefix = encoder.stable_epoch_prefix(
        _append_row(0), method, base_summary="memory-0"
    )
    assert '"base_summary":"memory-0"' in stable_prefix
    assert stable_prefix.endswith('"current_turn":')
    assert "Turn 0" not in stable_prefix

    for turn in range(5):
        row = _append_row(turn, decision="UPDATE")
        prepared = session.prepare(row, materialized_memory=f"memory-{turn}")
        assert prepared.reset_before is (turn == 0)
        if turn:
            assert '"base_summary":"memory-0"' in prepared.prompt
            assert f'{{"decision":"UPDATE","turn":{turn - 1}}}' in prepared.prompt
        committed = session.commit(
            assistant_output=f'{{"decision":"UPDATE","turn":{turn}}}',
            decision_for_epoch="UPDATE",
        )

    assert committed.epoch_end_reason == "update_compaction"
    next_prepared = session.prepare(
        _append_row(5), materialized_memory="compacted-memory"
    )
    assert next_prepared.reset_reason == "update_compaction"
    assert next_prepared.epoch_index == 1
    assert '"base_summary":"compacted-memory"' in next_prepared.prompt
    assert '"turn":4' not in next_prepared.prompt


def test_delta_append_session_resets_when_manifest_skips_turns() -> None:
    method = DeltaV3AppendMethod()
    encoder = DeltaAppendChatExampleEncoder(CharacterTokenizer(), max_length=4096)
    session = DeltaAppendPromptSession(method, encoder)
    session.prepare(_append_row(2), materialized_memory="first")
    session.commit(
        assistant_output='{"decision":"NO_OP"}',
        decision_for_epoch="NO_OP",
    )

    prepared = session.prepare(_append_row(7), materialized_memory="after-gap")

    assert prepared.reset_reason == "turn_discontinuity"
    assert '"base_summary":"after-gap"' in prepared.prompt
    assert "Turn 2" not in prepared.prompt


def test_build_and_validate_fixed_ratio_manifest(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_views(data_root)
    catalog_path = tmp_path / "catalog.sqlite"
    build_catalog(data_root, catalog_path)

    manifest = build_manifest(
        data_root=data_root,
        catalog_path=catalog_path,
        scenarios=[81],
        split="validation",
        noop_per_update=2,
        sampling_seed=9,
        include_quiz_anchors=False,
    )
    catalog = DatasetCatalog(catalog_path, data_root)
    selected = validate_manifest(manifest, catalog)

    assert manifest["totals"] == {
        "original_turns": 12,
        "selected_turns": 6,
        "gold_updates": 2,
        "gold_noops": 4,
    }
    assert list(selected) == [81]
    assert len(selected[81]) == 6

    changed = dict(manifest)
    changed["dataset_source_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="different dataset"):
        validate_manifest(changed, catalog)


def test_aggregate_records_reports_required_views() -> None:
    base = {
        "logical_prompt_tokens": 100,
        "candidate_prefix_tokens": 80,
        "reused_prefix_tokens": 80,
        "evaluated_prefill_tokens": 20,
        "decode_tokens": 7,
        "tokenization_seconds": 0.01,
        "cache_management_seconds": 0.001,
        "prefill_seconds": 0.02,
        "decode_seconds": 0.03,
        "model_seconds": 0.05,
        "ttft_seconds": 0.031,
        "end_to_end_seconds": 0.061,
        "kv_cache_bytes": 1024,
        "peak_cuda_allocated_bytes": 2048,
        "error": None,
        "compacted": False,
    }
    records = [
        {**base, "gold_decision": "NO_OP", "predicted_decision": "NO_OP"},
        {
            **base,
            "gold_decision": "UPDATE",
            "predicted_decision": "UPDATE",
            "compacted": True,
        },
    ]

    result = aggregate_records(records)

    assert result["all_turns"]["turns"] == 2
    assert result["all_turns"]["cache_reuse_ratio"] == pytest.approx(0.8)
    assert result["by_gold_decision"]["UPDATE"]["turns"] == 1
    assert result["by_predicted_decision"]["NO_OP"]["turns"] == 1
    assert result["delta_compaction"]["turns"] == 1


def test_comparison_reports_three_views_and_cache_savings() -> None:
    base_record = {
        "logical_prompt_tokens": 100,
        "candidate_prefix_tokens": 80,
        "reused_prefix_tokens": 0,
        "evaluated_prefill_tokens": 100,
        "decode_tokens": 7,
        "tokenization_seconds": 0.01,
        "cache_management_seconds": 0.001,
        "prefill_seconds": 0.1,
        "decode_seconds": 0.03,
        "model_seconds": 0.13,
        "ttft_seconds": 0.111,
        "end_to_end_seconds": 0.141,
        "kv_cache_bytes": 1024,
        "peak_cuda_allocated_bytes": 2048,
        "error": None,
        "compacted": False,
        "cache_epoch_reset_before": False,
        "cache_epoch_reset_reason": None,
        "gold_decision": "UPDATE",
        "predicted_decision": "UPDATE",
    }

    def summary(method: str, cache_mode: str) -> dict[str, object]:
        record = dict(base_record)
        if cache_mode == "on":
            record.update(
                reused_prefix_tokens=80,
                evaluated_prefill_tokens=20,
                prefill_seconds=0.02,
                model_seconds=0.05,
                ttft_seconds=0.031,
                end_to_end_seconds=0.061,
            )
        return {
            "model": "granite4-1b",
            "method": method,
            "cache_mode": cache_mode,
            "turn_manifest_signature": "manifest",
            "replay_mode": "controlled",
            "max_length": 4096,
            "max_new_tokens": 768,
            "repetitions": 1,
            "metrics": aggregate_records([record]),
        }

    comparison = build_comparison(
        [summary("summary", "off"), summary("summary", "on")],
        projected_prefill_tokens_per_second=30,
    )

    assert set(comparison["views"]) == {
        "all_turns",
        "gold_update",
        "predicted_update",
    }
    cache_on = comparison["views"]["all_turns"][1]
    assert cache_on["prefill_token_reduction_vs_off"] == pytest.approx(0.8)
    assert cache_on["model_cost_token_reduction_vs_off"] == pytest.approx(
        1 - 27 / 107
    )
    assert cache_on["projected_prefill_seconds_mean"] == pytest.approx(20 / 30)
    assert "Predicted" not in render_markdown(comparison)
    assert "predicted_update" in render_markdown(comparison)


def test_reference_comparison_checks_output_and_state(tmp_path: Path) -> None:
    reference = {
        "repetition": 1,
        "scenario_index": 81,
        "row_id": 7,
        "output": '{"decision":"NO_OP"}',
        "predicted_decision": "NO_OP",
        "applied_update": False,
        "compacted": False,
        "state_sha256": "same",
        "error": None,
    }
    path = tmp_path / "turns.jsonl"
    path.write_text(json.dumps(reference) + "\n", encoding="utf-8")

    assert _compare_reference([reference], path)["checked"] is True
    changed = {**reference, "state_sha256": "different"}
    with pytest.raises(ValueError, match="differs from its reference"):
        _compare_reference([changed], path)
