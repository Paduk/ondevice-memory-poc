from __future__ import annotations

import json
from pathlib import Path

import pytest

import palmclaw_ubuntu.vehicle_bench.memory as vehicle_memory_module
from palmclaw_ubuntu.models import (
    AgentResponse,
    AMemConstructionResponse,
    AMemEvolutionDecision,
    CompactAMemNoteDraft,
    CompactAMemResponse,
    FactMemoryCandidate,
    FactMemoryExtractionResponse,
    MemoryCandidate,
    MemoryEvidence,
    MemoryResponse,
    ModelUsage,
    PatchMemoryResponse,
    StructuredMemoryResponse,
    ToolCall,
    ToolMemoryIdentity,
    ToolMemoryPatch,
)
from palmclaw_ubuntu.providers import (
    FakeAMemModel,
    FakeCompactAMemModel,
    FakeEmbeddingModel,
    ScriptedAgentModel,
)
from palmclaw_ubuntu.vehicle_bench import (
    OracleFactRecordKey,
    OracleRetrievalLabel,
    VehicleMemoryBuilder,
    build_daily_history_batches,
    build_history_batches,
    build_turn_history_batches,
    estimate_amem_generation_calls,
    load_vehicle_benchmark,
    parse_vehicle_history,
    required_memory_strategies,
    run_agent_evaluation,
)
from palmclaw_ubuntu.vehicle_summary_wiki import (
    MEMORY_WIKI_READ,
    MEMORY_WIKI_SEARCH,
)


class _SummaryModel:
    backend = "fake"
    model_id = "vehicle-summary-fake"
    prompt_version = "vehicle-memory-summary-test-v1"
    schema_version = "summary-v1"

    def __init__(self):
        self.requests = []

    def consolidate(self, messages, previous_memory):
        self.requests.append((tuple(messages), previous_memory))
        return MemoryResponse(content="- Gary HUD brightness: 8")


class _EmptySummaryModel(_SummaryModel):
    def consolidate(self, messages, previous_memory):
        self.requests.append((tuple(messages), previous_memory))
        return MemoryResponse(content="")


class _RecursiveSummaryModel:
    backend = "fake"
    model_id = "vehicle-recursive-summary-fake"
    prompt_version = "vehicle-recursive-summary-test-v1"
    schema_version = "recursive-summary-v1"
    max_memory_chars = 8_192
    max_output_tokens = 2_048
    reasoning_effort = "low"
    redact_pii = True

    def __init__(
        self,
        *,
        fail_once: bool = False,
        update_cadence: str = "calendar_day",
    ):
        self.requests = []
        self.fail_once = fail_once
        self.update_cadence = update_cadence

    def update(self, *, previous_memory, date, daily_history):
        self.requests.append((previous_memory, date, daily_history))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("malformed recursive summary arguments")
        if "HUD brightness to 8" in daily_history:
            return MemoryResponse(
                content="**Gary**\n- hud_brightness: 8",
                usage=ModelUsage(
                    input_tokens=20,
                    output_tokens=8,
                    total_tokens=28,
                ),
                metadata={
                    "date": date,
                    "update_status": "updated",
                    "truncated": False,
                    "memory_chars": 33,
                },
            )
        return MemoryResponse(
            content="",
            usage=ModelUsage(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
            ),
            metadata={
                "date": date,
                "update_status": "noop",
                "truncated": False,
                "memory_chars": len(previous_memory),
            },
        )


class _FailingRecursiveSummaryModel(_RecursiveSummaryModel):
    def update(self, *, previous_memory, date, daily_history):
        self.requests.append((previous_memory, date, daily_history))
        raise RuntimeError("permanent recursive summary failure")


class _RecursiveSummaryPatchModel(_RecursiveSummaryModel):
    prompt_version = "vehicle-recursive-summary-patch-test-v3"
    schema_version = "recursive-summary-patch-repair-v1"

    def update(self, *, previous_memory, date, daily_history):
        response = super().update(
            previous_memory=previous_memory,
            date=date,
            daily_history=daily_history,
        )
        updated = response.metadata["update_status"] == "updated"
        return MemoryResponse(
            content=response.content,
            usage=response.usage,
            metadata={
                **response.metadata,
                "update_mode": "deterministic_patch",
                "patch_operation_count": int(updated),
                "patch_add_count": int(updated),
                "patch_replace_count": 0,
                "patch_delete_count": 0,
                "patch_apply_latency_ms": 0,
            },
        )


class _CompactingRecursiveSummaryPatchModel(_RecursiveSummaryPatchModel):
    prompt_version = "vehicle-turnwise-recursive-summary-patch-compact-test-v3"
    schema_version = "recursive-summary-patch-periodic-compaction-v3-soft-target"
    compaction_add_threshold = 64
    compaction_token_threshold = 1_000
    compaction_target_ratio = 0.70
    max_compaction_attempts = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compaction_states = []

    def set_compaction_state(self, *, patch_add_count):
        self.compaction_states.append(patch_add_count)

    def update(self, *, previous_memory, date, daily_history):
        response = super().update(
            previous_memory=previous_memory,
            date=date,
            daily_history=daily_history,
        )
        return MemoryResponse(
            content=response.content,
            usage=response.usage,
            metadata={
                **response.metadata,
                "update_mode": (
                    "deterministic_patch_with_periodic_compaction"
                ),
                "compaction_triggered": False,
                "compaction_attempts": 0,
                "compaction_latency_ms": 0,
            },
        )


class _TemporalRecursiveSummaryPatchModel(_RecursiveSummaryPatchModel):
    prompt_version = "vehicle-turnwise-recursive-summary-temporal-patch-test-v1"
    schema_version = "recursive-summary-temporal-patch-v1"

    def update(self, *, previous_memory, date, daily_history):
        response = super().update(
            previous_memory=previous_memory,
            date=date,
            daily_history=daily_history,
        )
        updated = response.metadata["update_status"] == "updated"
        return MemoryResponse(
            content=response.content,
            usage=response.usage,
            metadata={
                **response.metadata,
                "update_mode": "deterministic_temporal_patch",
                "temporal_operation_count": int(updated),
                "temporal_durable_upsert_count": int(updated),
            },
        )


class _TemporalCompactingRecursiveSummaryPatchModel(
    _CompactingRecursiveSummaryPatchModel
):
    prompt_version = "vehicle-turnwise-temporal-compact-test-v1"
    schema_version = "recursive-summary-temporal-compact-test-v1"

    def update(self, *, previous_memory, date, daily_history):
        response = super().update(
            previous_memory=previous_memory,
            date=date,
            daily_history=daily_history,
        )
        updated = response.metadata["update_status"] == "updated"
        return MemoryResponse(
            content=response.content,
            usage=response.usage,
            metadata={
                **response.metadata,
                "update_mode": (
                    "deterministic_temporal_patch_with_periodic_compaction"
                ),
                "temporal_operation_count": int(updated),
                "temporal_durable_upsert_count": int(updated),
            },
        )


class _StructuredModel:
    backend = "fake"
    model_id = "vehicle-structured-fake"
    prompt_version = "vehicle-memory-structured-test-v1"
    schema_version = "structured-v1"

    def __init__(self, *, fail_once: bool = False):
        self.requests = []
        self.fail_once = fail_once

    def extract(self, messages, existing_memory):
        self.requests.append((tuple(messages), existing_memory))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("transient structured failure")
        message = messages[-1]
        quote = "[2025-01-01 08:00] Gary: Set HUD brightness to 8."
        return StructuredMemoryResponse(
            candidates=(
                MemoryCandidate(
                    subject="Gary",
                    predicate="hud_brightness",
                    value="8",
                    scope="session",
                    memory_type="preference",
                    confidence=0.99,
                    sensitivity="low",
                    evidence=(
                        MemoryEvidence(
                            message_id=message.id,
                            quote=quote,
                        ),
                    ),
                ),
            )
        )


class _PatchModel:
    backend = "fake"
    model_id = "vehicle-patch-fake"
    prompt_version = "vehicle-tool-memory-patch-test-v1"
    schema_version = "tool-memory-patch-v1"

    def __init__(self):
        self.requests = []

    def propose(
        self,
        messages,
        ontology,
        active_records,
        *,
        user_id,
        session_id,
    ):
        self.requests.append(
            (tuple(messages), ontology, tuple(active_records), user_id, session_id)
        )
        source = messages[0]
        if "HUD brightness to 8" not in source.content:
            return PatchMemoryResponse(
                patches=(),
                usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            )
        return PatchMemoryResponse(
            patches=(
                ToolMemoryPatch(
                    operation="ADD",
                    identity=ToolMemoryIdentity(
                        user_id=user_id,
                        tool_domain="hud",
                        topic="brightness_level",
                        scope="global",
                        scope_key=user_id,
                        conditions={"person": "Gary"},
                    ),
                    value=8,
                    memory_type="preference",
                    confidence=0.99,
                    evidence=(
                        MemoryEvidence(
                            message_id=source.id,
                            quote=source.content,
                        ),
                    ),
                    reason="explicit HUD preference",
                ),
            ),
            usage=ModelUsage(input_tokens=30, output_tokens=12, total_tokens=42),
            response_id="vehicle-patch-response",
        )


class _FactModel:
    backend = "fake"
    model_id = "vehicle-fact-fake"
    prompt_version = "vehicle-fact-memory-test-v1"
    schema_version = "fact-memory-candidate-v1"
    semantic_prompt_version = "vehicle-fact-semantic-test-v1"
    semantic_schema_version = "fact-memory-semantic-decision-v1"
    max_output_tokens = 2_048
    reasoning_effort = "low"
    redact_pii = True

    def __init__(self):
        self.requests = []

    def extract(
        self,
        messages,
        active_records,
        *,
        user_id,
        session_id,
    ):
        self.requests.append(
            (tuple(messages), tuple(active_records), user_id, session_id)
        )
        source = messages[0]
        return FactMemoryExtractionResponse(
            candidates=(
                FactMemoryCandidate(
                    entity_id="Gary",
                    predicate="HUD brightness",
                    value=8,
                    identity_conditions={},
                    applicability={},
                    capability_hints=("HUD",),
                    memory_type="preference",
                    confidence=0.99,
                    evidence=(
                        MemoryEvidence(
                            message_id=source.id,
                            quote=source.content,
                        ),
                    ),
                ),
            ),
            usage=ModelUsage(input_tokens=20, output_tokens=8, total_tokens=28),
            metadata={
                "recursive_assisted": bool(getattr(self, "auxiliary_context", None)),
                "assisted_candidate_count": (
                    1 if getattr(self, "auxiliary_context", None) else 0
                ),
            },
        )

    def review_uncertain(
        self,
        cases,
        messages,
        active_records,
        *,
        user_id,
        session_id,
    ):
        raise AssertionError("Fixture candidates should not require review")


class _EmptyFactModel(_FactModel):
    def extract(
        self,
        messages,
        active_records,
        *,
        user_id,
        session_id,
    ):
        self.requests.append(
            (tuple(messages), tuple(active_records), user_id, session_id)
        )
        return FactMemoryExtractionResponse(
            candidates=(),
            usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
        )


class _SchemaFactModel(_FactModel):
    schema_informed = True
    canonical_predicates = True
    entity_resolution_version = "vehicle-history-speaker-v1"
    linking_policy_version = "canonical-predicate-exact-v1"


class _SchemaEmptyFactModel(_EmptyFactModel):
    schema_informed = True
    canonical_predicates = True
    entity_resolution_version = "vehicle-history-speaker-v1"
    linking_policy_version = "canonical-predicate-exact-v1"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _memory_benchmark(root: Path) -> Path:
    schema = [
        {
            "name": "carcontrol_HUD_set_brightness_level",
            "description": "Set HUD brightness.",
            "parameters": {
                "type": "object",
                "properties": {"level": {"type": "integer"}},
                "required": ["level"],
            },
        }
    ]
    qa = {
        "related_to_vehicle_preference": [
            {
                "gold_memory": "Gary prefers HUD brightness 8.",
                "reasoning_type": "preference_conflict",
                "query": "Set Gary's usual HUD brightness.",
                "new_answer": [
                    "carcontrol_HUD_set_brightness_level(level=8)",
                ],
            }
        ]
    }
    _write(
        root / "evaluation/functions_schema.json",
        json.dumps(schema),
    )
    _write(
        root / "benchmark/qa_data/qa_1.json",
        json.dumps(qa),
    )
    _write(
        root / "benchmark/history/history_1.txt",
        (
            "[2025-01-01 08:00] Gary: Set HUD brightness to 8.\n"
            "[2025-01-02 09:00] Justin: Nice weather today.\n"
        ),
    )
    _write(root / "environment/__init__.py", "")
    _write(
        root / "environment/utils.py",
        'modules_dict = {"HUD": "Head-up display"}\n',
    )
    _write(
        root / "environment/vehicleworld.py",
        """
from .utils import modules_dict


class HUD:
    def __init__(self):
        self.brightness = 5

    def carcontrol_HUD_set_brightness_level(self, level):
        self.brightness = level
        return {"success": True, "current_state": self.to_dict()}

    def to_dict(self):
        return {
            "brightness": {
                "value": self.brightness,
                "description": "HUD brightness",
                "type": "int",
            }
        }


class VehicleWorld:
    def __init__(self):
        self.HUD = HUD()

    def to_dict(self):
        return {
            "HUD": {
                "value": self.HUD.to_dict(),
                "description": modules_dict["HUD"],
                "type": "HUD",
            }
        }
""".lstrip(),
    )
    _write(
        root / "evaluation/eval_utils.py",
        """
import json
from environment.utils import modules_dict


def calculate_turn_result(world1, world2, world3, world4):
    del world1, world3
    exact = world2 == world4 and bool(modules_dict)
    return {
        "differences": [] if exact else ["state mismatch"],
        "TP": 1 if exact else 0,
        "FP": 0 if exact else 1,
        "negative_FP": 0 if exact else 1,
        "f1_positive": 1.0 if exact else 0.0,
        "f1_change": 1.0 if exact else 0.0,
    }


def score_tool_calls(pred_calls, ref_calls):
    def key(call):
        return call["name"], json.dumps(call["args"], sort_keys=True)

    exact = [key(call) for call in pred_calls] == [
        key(call) for call in ref_calls
    ]
    return {
        "tp": len(ref_calls) if exact else 0,
        "fp": 0 if exact else len(pred_calls),
        "fn": 0 if exact else len(ref_calls),
        "precision": 1.0 if exact else 0.0,
        "recall": 1.0 if exact else 0.0,
        "f1": 1.0 if exact else 0.0,
    }
""".lstrip(),
    )
    return root


def _builder(
    dataset,
    cache_root: Path,
    summary_model,
    structured_model,
    patch_model=None,
    fact_model=None,
    recursive_summary_model=None,
    amem_model=None,
    compact_amem_model=None,
    amem_note_limit=None,
    **builder_kwargs,
) -> VehicleMemoryBuilder:
    return VehicleMemoryBuilder(
        dataset=dataset,
        scenario_index=1,
        cache_root=cache_root,
        summary_model=summary_model,
        recursive_summary_model=recursive_summary_model,
        structured_model=structured_model,
        embedding_model=FakeEmbeddingModel(8),
        patch_model=patch_model,
        fact_model=fact_model,
        amem_model=amem_model,
        compact_amem_model=compact_amem_model,
        batch_token_limit=256,
        retrieval_top_k=3,
        retrieval_token_budget=1_000,
        model_timeout_seconds=2,
        patch_batch_size=2,
        patch_lease_seconds=3,
        amem_note_limit=amem_note_limit,
        **builder_kwargs,
    )


def _amem_model(note_count: int) -> FakeAMemModel:
    return FakeAMemModel(
        construction_responses=tuple(
            AMemConstructionResponse(
                context=f"vehicle history note {index}",
                keywords=(f"vehicle-{index}",),
                tags=("history",),
                usage=ModelUsage(
                    input_tokens=2,
                    output_tokens=1,
                    total_tokens=3,
                ),
            )
            for index in range(note_count)
        ),
        evolution_responses=tuple(
            AMemEvolutionDecision(
                should_evolve=False,
                usage=ModelUsage(
                    input_tokens=3,
                    output_tokens=1,
                    total_tokens=4,
                ),
            )
            for _ in range(max(0, note_count - 1))
        ),
    )


def test_history_parser_preserves_speaker_timestamp_and_duplicate_prefix(
    tmp_path: Path,
):
    path = tmp_path / "history.txt"
    _write(
        path,
        (
            "[2025-04-12 14:04] Kathleen Ramirez: Set screen brightness to 80.\n"
            "[2025-04-12 14:05] [2025-04-12 14:05] Kathleen Ramirez: "
            "I meant 64.\n"
        ),
    )

    entries = parse_vehicle_history(path)
    batches = build_history_batches(entries, max_tokens=256)

    assert len(entries) == 2
    assert entries[1].speaker == "Kathleen Ramirez"
    assert entries[1].timestamp.isoformat(timespec="minutes") == "2025-04-12T14:05"
    assert entries[1].content == "I meant 64."
    assert len(batches) == 1
    assert entries[1].raw in batches[0].content


def test_daily_history_batches_sort_dates_and_preserve_timestamp_order(
    tmp_path: Path,
):
    path = tmp_path / "history.txt"
    _write(
        path,
        (
            "[2025-01-02 09:00] Gary: Day two.\n"
            "[2025-01-01 10:00] Gary: Later on day one.\n"
            "[2025-01-01 08:00] Gary: Earlier on day one.\n"
        ),
    )

    batches = build_daily_history_batches(
        parse_vehicle_history(path),
        max_tokens=256,
    )

    assert [batch.start_date for batch in batches] == [
        "2025-01-01",
        "2025-01-02",
    ]
    assert batches[0].content.index("08:00") < batches[0].content.index("10:00")
    assert all(batch.start_date == batch.end_date for batch in batches)


def test_turn_history_batches_ignore_dates_and_keep_one_entry_per_turn(
    tmp_path: Path,
):
    path = tmp_path / "history.txt"
    _write(
        path,
        (
            "[2025-01-01 08:00] Gary: First turn.\n"
            "[2025-01-01 08:01] Justin: Second turn.\n"
        ),
    )

    batches = build_turn_history_batches(
        parse_vehicle_history(path),
        max_tokens=256,
    )

    assert len(batches) == 2
    assert [batch.line_count for batch in batches] == [1, 1]
    assert "First turn" in batches[0].content
    assert "Second turn" not in batches[0].content
    assert "Second turn" in batches[1].content


def test_amem_vehicle_snapshot_builds_once_and_retrieval_is_read_only(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    cache_root = tmp_path / "amem-cache"
    model = _amem_model(2)
    builder = _builder(
        dataset,
        cache_root,
        _SummaryModel(),
        _StructuredModel(),
        amem_model=model,
    )

    assert required_memory_strategies(("cloud_amem",)) == ("amem",)
    with builder.build(strategies=("amem",)) as snapshot:
        profile = snapshot.manifest["profiles"]["amem"]
        fingerprint = snapshot.memory_fingerprint()
        contexts = [
            snapshot.resolve(
                "cloud_amem",
                f"Vehicle preference query {index}",
            )
            for index in range(10)
        ]

        assert profile["status"] == "ready"
        assert profile["ingested_notes"] == 2
        assert profile["generation_call_estimate"] == 3
        assert len(model.construction_requests) == 2
        assert len(model.evolution_requests) == 1
        assert all(context.records == () for context in contexts)
        assert all(context.fact_records == () for context in contexts)
        assert all(
            context.metadata["context_sources"] == ["retrieved_memory", "query"]
            for context in contexts
        )
        assert all(context.trace["strategy"] == "amem" for context in contexts)
        assert snapshot.memory_fingerprint() == fingerprint
        scripted = ScriptedAgentModel(
            [
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="discover-hud",
                            name="list_module_tools",
                            arguments={"module_name": "HUD"},
                        ),
                    ),
                ),
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="set-hud",
                            name="carcontrol_HUD_set_brightness_level",
                            arguments={"level": 8},
                        ),
                    ),
                ),
                AgentResponse(content="Applied."),
            ]
        )
        evaluation = run_agent_evaluation(
            dataset,
            agent_model=scripted,
            profile_names=("cloud_amem",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "amem-artifacts",
        )
        task = evaluation.tasks[0]
        system_prompt = scripted.requests[0][0][0].content

        assert task["score"]["exact_state_match"] is True
        assert task["context"]["sources"] == [
            "retrieved_memory",
            "query",
        ]
        assert task["context"]["raw_history_included"] is False
        assert task["context"]["selector_tool_names"] == []
        assert task["context"]["tool_memory_execution_hint_count"] == 0
        assert task["tool_boundary"]["initial_tools"] == ["list_module_tools"]
        assert "Gary prefers HUD brightness 8." not in system_prompt
        assert "carcontrol_HUD_set_brightness_level(level=8)" not in system_prompt
        assert (
            dataset.scenario(1).history_path.read_text(encoding="utf-8").strip()
            not in system_prompt
        )
        usage = snapshot.amem_usage()
        assert usage["retrieval"]["run_count"] == 11
        assert usage["provider"]["roles"] == {
            "amem_construction": 2,
            "amem_embedding": 2,
            "amem_evolution": 1,
            "amem_retrieval_embedding": 11,
        }
        assert usage["provider"]["call_count"] == 16
        assert usage["provider"]["failed_call_count"] == 0
        assert usage["provider"]["usage"]["total_tokens"] > 0
        assert usage["build_provider_calls_incurred"] is True
        assert snapshot.memory_fingerprint() == fingerprint

    cached_model = FakeAMemModel()
    cached_builder = _builder(
        dataset,
        cache_root,
        _SummaryModel(),
        _StructuredModel(),
        amem_model=cached_model,
    )
    with cached_builder.build(strategies=("amem",)) as cached:
        assert cached.memory_fingerprint() == fingerprint
        assert cached_model.construction_requests == []
        assert cached_model.evolution_requests == []
        assert cached.amem_usage()["build_provider_calls_incurred"] is False


def test_amem_style_profile_gates_evolution_and_uses_independent_session(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    model = FakeAMemModel(
        construction_responses=tuple(
            AMemConstructionResponse(
                context=f"style note {index}",
                keywords=(f"style-{index}",),
                tags=("history",),
            )
            for index in range(2)
        )
    )
    builder = _builder(
        dataset,
        tmp_path / "amem-style-cache",
        _SummaryModel(),
        _StructuredModel(),
        amem_model=model,
        amem_style_evolution_threshold=1.0,
    )

    assert required_memory_strategies(("cloud_amem_style",)) == ("amem_style",)
    with builder.build(strategies=("amem_style",)) as snapshot:
        profile = snapshot.manifest["profiles"]["amem_style"]
        context = snapshot.resolve(
            "cloud_amem_style",
            "Vehicle preference query",
        )
        usage = snapshot.amem_usage()

        assert snapshot.amem_session_id is None
        assert snapshot.amem_style_session_id == profile["session_id"]
        assert profile["evolution_similarity_threshold"] == 1.0
        assert profile["evolution_call_count"] == 0
        assert profile["evolution_skipped_count"] == 1
        assert len(model.construction_requests) == 2
        assert model.evolution_requests == []
        assert context.trace["strategy"] == "amem_style"
        assert usage["strategy"] == "amem_style"
        assert usage["generation"]["call_count"] == 2


def test_compact_amem_profile_e2e_resumes_without_gold_leak(tmp_path: Path):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    compact_model = FakeCompactAMemModel(
        responses=(
            CompactAMemResponse(
                notes=(
                    CompactAMemNoteDraft(
                        content="Gary usually sets HUD brightness to 8.",
                        context="Stable vehicle display preference for Gary.",
                        keywords=("Gary", "HUD", "brightness", "8"),
                        tags=("vehicle", "preference"),
                        memory_kind="preference",
                        source_message_ids=(1,),
                    ),
                ),
                usage=ModelUsage(
                    input_tokens=12,
                    output_tokens=6,
                    total_tokens=18,
                ),
            ),
            CompactAMemResponse(notes=()),
        )
    )
    builder = _builder(
        dataset,
        tmp_path / "compact-amem-cache",
        _SummaryModel(),
        _StructuredModel(),
        amem_model=FakeAMemModel(),
        compact_amem_model=compact_model,
    )

    assert required_memory_strategies(("cloud_compact_amem_style",)) == (
        "compact_amem",
    )
    with builder.build(strategies=("compact_amem",)) as snapshot:
        profile = snapshot.manifest["profiles"]["compact_amem"]
        context = snapshot.resolve(
            "cloud_compact_amem_style",
            "Set Gary's usual HUD brightness.",
        )
        fingerprint = snapshot.memory_fingerprint()
        scripted = ScriptedAgentModel(
            [
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="discover-hud",
                            name="list_module_tools",
                            arguments={"module_name": "HUD"},
                        ),
                    ),
                ),
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="set-hud",
                            name="carcontrol_HUD_set_brightness_level",
                            arguments={"level": 8},
                        ),
                    ),
                ),
                AgentResponse(content="Applied."),
            ]
        )
        evaluation = run_agent_evaluation(
            dataset,
            agent_model=scripted,
            profile_names=("cloud_compact_amem_style",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "compact-amem-artifacts",
        )
        task = evaluation.tasks[0]
        system_prompt = scripted.requests[0][0][0].content

        assert snapshot.amem_session_id is None
        assert snapshot.amem_style_session_id is None
        assert snapshot.compact_amem_session_id == profile["session_id"]
        assert profile["status"] == "ready"
        assert profile["raw_history_line_count"] == 2
        assert profile["episode_count"] == 2
        assert profile["ingested_notes"] == 1
        assert profile["compression_ratio"] == 0.5
        assert profile["evolution_call_count"] == 0
        assert profile["processed_episode_indices"] == [0, 1]
        assert profile["usage"]["generation"]["compaction"] == {"completed": 2}
        assert profile["usage"]["provider"]["roles"]["amem_compaction"] == 2
        assert context.trace["strategy"] == "compact_amem"
        assert context.metadata["context_sources"] == [
            "retrieved_memory",
            "query",
        ]
        assert task["score"]["exact_state_match"] is True
        assert task["context"]["raw_history_included"] is False
        assert task["context"]["selector_tool_names"] == []
        assert evaluation.metrics["artifact_privacy_audit"]["passed"] is True
        assert "Gary prefers HUD brightness 8." not in system_prompt
        assert "carcontrol_HUD_set_brightness_level(level=8)" not in system_prompt
        assert (
            dataset.scenario(1).history_path.read_text(encoding="utf-8").strip()
            not in system_prompt
        )
        assert snapshot.amem_usage()["strategy"] == "compact_amem"
        assert snapshot.memory_fingerprint() == fingerprint

    resumed_model = FakeCompactAMemModel()
    resumed_builder = _builder(
        dataset,
        tmp_path / "compact-amem-cache",
        _SummaryModel(),
        _StructuredModel(),
        amem_model=FakeAMemModel(),
        compact_amem_model=resumed_model,
    )
    with resumed_builder.build(strategies=("compact_amem",)) as snapshot:
        profile = snapshot.manifest["profiles"]["compact_amem"]
        assert resumed_model.requests == []
        assert profile["processed_episode_indices"] == []
        assert profile["skipped_episode_indices"] == [0, 1]
        assert snapshot.memory_fingerprint() == fingerprint


def test_amem_vehicle_artifacts_remove_source_and_metadata_pii(tmp_path: Path):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    history_path = root / "benchmark/history/history_1.txt"
    history_path.write_text(
        history_path.read_text(encoding="utf-8").replace(
            "Set HUD brightness to 8.",
            "Set HUD brightness to 8 and email alice@example.com.",
        ),
        encoding="utf-8",
    )
    dataset = load_vehicle_benchmark(root, strict=False)
    model = FakeAMemModel(
        construction_responses=(
            AMemConstructionResponse(
                context="Contact alice@example.com about HUD brightness",
                keywords=("alice@example.com", "HUD"),
                tags=("preference",),
            ),
            AMemConstructionResponse(
                context="Unrelated weather note",
                keywords=("weather",),
                tags=("history",),
            ),
        ),
        evolution_responses=(AMemEvolutionDecision(should_evolve=False),),
    )
    builder = _builder(
        dataset,
        tmp_path / "amem-pii-cache",
        _SummaryModel(),
        _StructuredModel(),
        amem_model=model,
    )

    with builder.build(strategies=("amem",)) as snapshot:
        result = run_agent_evaluation(
            dataset,
            agent_model=ScriptedAgentModel([AgentResponse(content="No action.")]),
            profile_names=("cloud_amem",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "amem-pii-artifacts",
        )

    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result.artifact_dir.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".tsv", ".md"}
    )
    assert "alice@example.com" not in artifact_text
    assert "[REDACTED_EMAIL]" in artifact_text
    assert result.metrics["artifact_privacy_audit"]["passed"] is True


def test_amem_note_limit_marks_partial_history_and_changes_cache_key(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    assert estimate_amem_generation_calls(0) == 0
    assert estimate_amem_generation_calls(1) == 1
    assert estimate_amem_generation_calls(2) == 3
    full_builder = _builder(
        dataset,
        tmp_path / "amem-cache",
        _SummaryModel(),
        _StructuredModel(),
        amem_model=_amem_model(2),
    )
    partial_builder = _builder(
        dataset,
        tmp_path / "amem-cache",
        _SummaryModel(),
        _StructuredModel(),
        amem_model=_amem_model(1),
        amem_note_limit=1,
    )

    with full_builder.build(strategies=("amem",)) as full:
        full_key = full.manifest["cache_key"]
        assert full.amem_usage()["formal_aggregate_included"] is True
    with partial_builder.build(strategies=("amem",)) as partial:
        profile = partial.manifest["profiles"]["amem"]
        assert partial.manifest["cache_key"] != full_key
        assert profile["partial_history"] is True
        assert profile["formal_aggregate_included"] is False
        assert profile["generation_call_estimate"] == 1
        assert partial.amem_usage()["formal_aggregate_included"] is False


def test_amem_cache_config_covers_result_affecting_parameter_matrix(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)

    def make_builder(**kwargs):
        return _builder(
            dataset,
            tmp_path / "config-cache",
            _SummaryModel(),
            _StructuredModel(),
            amem_model=_amem_model(2),
            **kwargs,
        )

    base = make_builder()
    link = make_builder(amem_link_candidates=4)
    retrieval = make_builder(amem_retrieval_top_k=9)
    budget = make_builder(amem_retrieval_token_budget=999)
    partial = make_builder(amem_note_limit=1)
    prompt = make_builder()
    prompt.amem_model.construction_prompt_version = "amem-construction-v2"
    embedding = make_builder()
    embedding.embedding_model.model_id = "different-embedding"
    configs = [
        builder._cache_config()
        for builder in (
            base,
            link,
            retrieval,
            budget,
            partial,
            prompt,
            embedding,
        )
    ]

    assert len(
        {
            json.dumps(config, sort_keys=True, separators=(",", ":"))
            for config in configs
        }
    ) == len(configs)
    assert configs[0]["amem"]["link_candidate_limit"] == 5
    assert configs[0]["amem"]["retrieval_top_k"] == 10
    assert configs[0]["amem"]["retrieval_token_budget"] == 1_000
    assert configs[0]["embedding"]["model_id"] == "fake-hash-embedding-v1"


@pytest.mark.parametrize("failure_stage", ("before_ingest", "after_graph"))
def test_amem_manifest_interruption_resumes_without_duplicate_calls(
    tmp_path: Path,
    monkeypatch,
    failure_stage: str,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    model = _amem_model(2)
    builder = _builder(
        dataset,
        tmp_path / failure_stage,
        _SummaryModel(),
        _StructuredModel(),
        amem_model=model,
    )
    original_write = vehicle_memory_module._write_json_atomic
    interrupted = False

    def interrupt_manifest(path, payload):
        nonlocal interrupted
        profile = payload.get("profiles", {}).get("amem", {})
        should_interrupt = (
            failure_stage == "before_ingest"
            and profile.get("status") == "building"
            and profile.get("ingested_notes") == 0
        ) or (failure_stage == "after_graph" and profile.get("status") == "ready")
        if should_interrupt and not interrupted:
            interrupted = True
            raise RuntimeError(f"planned {failure_stage} manifest failure")
        original_write(path, payload)

    monkeypatch.setattr(
        vehicle_memory_module,
        "_write_json_atomic",
        interrupt_manifest,
    )
    with pytest.raises(RuntimeError, match=f"planned {failure_stage}"):
        builder.build(strategies=("amem",))
    calls_after_failure = (
        len(model.construction_requests),
        len(model.evolution_requests),
    )
    monkeypatch.setattr(
        vehicle_memory_module,
        "_write_json_atomic",
        original_write,
    )

    with builder.build(strategies=("amem",)) as resumed:
        fingerprint = resumed.memory_fingerprint()
        assert resumed.manifest["status"] == "ready"
        assert resumed.manifest["profiles"]["amem"]["ingested_notes"] == 2
        assert len(model.construction_requests) == 2
        assert len(model.evolution_requests) == 1
        if failure_stage == "after_graph":
            assert calls_after_failure == (2, 1)
    with builder.build(strategies=("amem",)) as cached:
        assert cached.memory_fingerprint() == fingerprint
        assert len(model.construction_requests) == 2
        assert len(model.evolution_requests) == 1


def test_recursive_summary_builds_daily_updates_noops_and_immutable_context(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    recursive_model = _RecursiveSummaryModel()
    builder = _builder(
        dataset,
        tmp_path / "recursive-cache",
        _SummaryModel(),
        _StructuredModel(),
        recursive_summary_model=recursive_model,
    )

    assert required_memory_strategies(("cloud_recursive_summary",)) == (
        "recursive_summary",
    )
    with builder.build(strategies=("recursive_summary",)) as snapshot:
        assert snapshot.summary_session_id is None
        assert snapshot.structured_session_id is None
        fingerprint = snapshot.memory_fingerprint()
        first = snapshot.resolve(
            "cloud_recursive_summary",
            "Set Gary's usual HUD brightness.",
        )
        second = snapshot.resolve(
            "cloud_recursive_summary",
            "A different query",
        )
        progress = snapshot.repository.consolidation_progress(
            snapshot.recursive_summary_session_id
        )
        memories = snapshot.repository.list_memories(
            snapshot.recursive_summary_session_id,
            include_superseded=True,
        )

        assert first.content == "**Gary**\n- hud_brightness: 8"
        assert second.content == first.content
        assert first.metadata["memory_strategy"] == "recursive_summary"
        assert first.metadata["retrieval_mode"] == "full_recursive_summary"
        assert first.metadata["query_dependent"] is False
        assert first.metadata["recall_at_k_applicable"] is False
        assert first.trace["update_count"] == 1
        assert first.trace["noop_count"] == 1
        assert [step["status"] for step in first.trace["daily_steps"]] == [
            "updated",
            "noop",
        ]
        assert len(memories) == 1
        assert progress["completed_message_count"] == 2
        assert progress["run_status_counts"] == {"completed": 2}
        assert snapshot.memory_fingerprint() == fingerprint
        assert all(call["role"] == "memory" for call in snapshot.provider_calls())
        scripted = ScriptedAgentModel(
            [
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="discover-hud",
                            name="list_module_tools",
                            arguments={"module_name": "HUD"},
                        ),
                    ),
                ),
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="set-hud",
                            name="carcontrol_HUD_set_brightness_level",
                            arguments={"level": 8},
                        ),
                    ),
                ),
                AgentResponse(content="Applied."),
            ]
        )
        result = run_agent_evaluation(
            dataset,
            agent_model=scripted,
            profile_names=("cloud_recursive_summary",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "recursive-artifacts",
        )
        task = result.tasks[0]
        profile_metrics = result.metrics["profiles"]["cloud_recursive_summary"]

        assert task["score"]["exact_state_match"] is True
        assert task["retrieval_quality"] is None
        assert profile_metrics["retrieval_recall_at_k"] is None
        recursive_metrics = profile_metrics["recursive_summary"]
        assert recursive_metrics["snapshot_count"] == 1
        assert recursive_metrics["update_count"] == 1
        assert recursive_metrics["noop_count"] == 1
        assert recursive_metrics["truncation_count"] == 0
        assert recursive_metrics["final_characters_max"] == len(first.content)
        assert recursive_metrics["final_tokens_max"] > 0
        assert task["context"]["selector_tool_names"] == []
        assert task["context"]["tool_memory_execution_hint_count"] == 0
        assert task["tool_boundary"]["initial_tools"] == ["list_module_tools"]
        system_prompt = scripted.requests[0][0][0].content
        assert first.content in system_prompt
        assert (
            dataset.scenario(1).history_path.read_text(encoding="utf-8").strip()
            not in system_prompt
        )
        assert snapshot.memory_fingerprint() == fingerprint

    assert [request[1] for request in recursive_model.requests] == [
        "2025-01-01",
        "2025-01-02",
    ]
    assert recursive_model.requests[0][0] == ""
    assert recursive_model.requests[1][0] == ("**Gary**\n- hud_brightness: 8")
    with builder.build(strategies=("recursive_summary",)) as cached:
        assert cached.manifest["profiles"]["recursive_summary"]["status"] == "ready"
    assert len(recursive_model.requests) == 2


def test_recursive_summary_patch_profile_isolated_from_rewrite_cache(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    cache_root = tmp_path / "recursive-cache"
    baseline_builder = _builder(
        dataset,
        cache_root,
        _SummaryModel(),
        _StructuredModel(),
        recursive_summary_model=_RecursiveSummaryModel(),
    )
    patch_builder = _builder(
        dataset,
        cache_root,
        _SummaryModel(),
        _StructuredModel(),
        recursive_summary_model=_RecursiveSummaryPatchModel(),
    )

    assert required_memory_strategies(("cloud_recursive_summary_patch",)) == (
        "recursive_summary",
    )
    with baseline_builder.build(strategies=("recursive_summary",)) as baseline_snapshot:
        baseline_cache_dir = baseline_snapshot.cache_dir
        baseline_content = baseline_snapshot.resolve(
            "cloud_recursive_summary",
            "Set Gary's usual HUD brightness.",
        ).content

    with patch_builder.build(strategies=("recursive_summary",)) as patch_snapshot:
        patch_context = patch_snapshot.resolve(
            "cloud_recursive_summary_patch",
            "Set Gary's usual HUD brightness.",
        )
        patch_profile = patch_snapshot.manifest["profiles"]["recursive_summary"]

        assert patch_snapshot.cache_dir != baseline_cache_dir
        assert patch_context.content == baseline_content
        assert (
            patch_context.metadata["recursive_summary_update_mode"]
            == "deterministic_patch"
        )
        assert patch_context.metadata["recursive_summary_patch_operation_count"] == 1
        assert patch_profile["update_mode"] == "deterministic_patch"
        assert patch_profile["patch_operation_count"] == 1
        assert [
            step["patch_operation_count"] for step in patch_context.trace["daily_steps"]
        ] == [1, 0]
        evaluation = run_agent_evaluation(
            dataset,
            agent_model=ScriptedAgentModel(
                [
                    AgentResponse(
                        content="",
                        tool_calls=(
                            ToolCall(
                                id="discover-hud",
                                name="list_module_tools",
                                arguments={"module_name": "HUD"},
                            ),
                        ),
                    ),
                    AgentResponse(
                        content="",
                        tool_calls=(
                            ToolCall(
                                id="set-hud",
                                name=("carcontrol_HUD_set_brightness_level"),
                                arguments={"level": 8},
                            ),
                        ),
                    ),
                    AgentResponse(content="Applied."),
                ]
            ),
            profile_names=("cloud_recursive_summary_patch",),
            scenario_index=1,
            memory_resolver=patch_snapshot.resolve,
            memory_manifest=patch_snapshot.manifest,
            output_root=tmp_path / "recursive-patch-artifacts",
        )

        metrics = evaluation.metrics["profiles"]["cloud_recursive_summary_patch"]
        assert evaluation.tasks[0]["score"]["exact_state_match"] is True
        assert metrics["retrieval_recall_at_k"] is None
        assert metrics["recursive_summary"]["update_modes"] == ["deterministic_patch"]
        assert metrics["recursive_summary"]["patch_operation_count"] == 1


@pytest.mark.parametrize(
    ("profile_name", "model_class", "update_mode"),
    (
        (
            "cloud_turnwise_recursive_summary",
            _RecursiveSummaryModel,
            "full_rewrite",
        ),
        (
            "cloud_turnwise_recursive_summary_patch",
            _RecursiveSummaryPatchModel,
            "deterministic_patch",
        ),
        (
            "cloud_turnwise_recursive_summary_patch_compact",
            _CompactingRecursiveSummaryPatchModel,
            "deterministic_patch_with_periodic_compaction",
        ),
        (
            "cloud_turnwise_recursive_summary_patch_temporal",
            _TemporalRecursiveSummaryPatchModel,
            "deterministic_temporal_patch",
        ),
        (
            "cloud_turnwise_recursive_summary_patch_temporal_compact",
            _TemporalCompactingRecursiveSummaryPatchModel,
            "deterministic_temporal_patch_with_periodic_compaction",
        ),
    ),
)
def test_turnwise_recursive_profiles_update_each_history_entry_and_track_status(
    tmp_path: Path,
    profile_name: str,
    model_class,
    update_mode: str,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    _write(
        root / "benchmark/history/history_1.txt",
        (
            "[2025-01-01 08:00] Gary: Set HUD brightness to 8.\n"
            "[2025-01-01 08:01] Justin: Nice weather today.\n"
        ),
    )
    dataset = load_vehicle_benchmark(root, strict=False)
    model = model_class(update_cadence="history_entry")
    builder = _builder(
        dataset,
        tmp_path / profile_name,
        _SummaryModel(),
        _StructuredModel(),
        recursive_summary_model=model,
    )

    assert required_memory_strategies((profile_name,)) == ("recursive_summary",)
    with builder.build(strategies=("recursive_summary",)) as snapshot:
        context = snapshot.resolve(profile_name, "Set Gary's HUD brightness")
        profile = snapshot.manifest["profiles"]["recursive_summary"]

        assert len(model.requests) == 2
        assert all(request[1] == "2025-01-01" for request in model.requests)
        assert all(
            request[2].count("VehicleMemBench history turn=") == 1
            for request in model.requests
        )
        assert profile["update_cadence"] == "history_entry"
        assert profile["total_step_count"] == 2
        assert profile["update_count"] == 1
        assert profile["noop_count"] == 1
        assert profile["redundant_update_count"] == 0
        assert profile["update_ratio"] == pytest.approx(0.5)
        assert profile["noop_ratio"] == pytest.approx(0.5)
        assert profile["status_usage"]["updated"]["input_tokens"] == 20
        assert profile["status_usage"]["noop"]["input_tokens"] == 10
        assert len(context.trace["turn_steps"]) == 2
        assert context.trace["update_cadence"] == "history_entry"
        assert len(context.trace["final_summary_sha256"]) == 64
        assert context.metadata["recursive_summary_update_mode"] == update_mode
        if isinstance(model, _CompactingRecursiveSummaryPatchModel):
            assert model.compaction_states == [0, 1]
            assert profile["patch_add_count_since_compaction"] == 1
            assert profile["compaction_count"] == 0
            assert profile["compaction_latency_ms"] == 0
        if isinstance(model, _TemporalRecursiveSummaryPatchModel):
            assert profile["temporal_operation_count"] == 1
            assert profile["temporal_durable_upsert_count"] == 1


def test_turnwise_history_entry_limit_is_cache_isolated(tmp_path: Path):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    _write(
        root / "benchmark/history/history_1.txt",
        (
            "[2025-01-01 08:00] Gary: Set HUD brightness to 8.\n"
            "[2025-01-01 08:01] Justin: Nice weather today.\n"
        ),
    )
    dataset = load_vehicle_benchmark(root, strict=False)
    model = _RecursiveSummaryModel(update_cadence="history_entry")
    builder = _builder(
        dataset,
        tmp_path / "turnwise-limit",
        _SummaryModel(),
        _StructuredModel(),
        recursive_summary_model=model,
        history_entry_limit=1,
    )

    with builder.build(strategies=("recursive_summary",)) as snapshot:
        assert len(model.requests) == 1
        assert snapshot.manifest["config"]["history_entry_limit"] == 1
        assert snapshot.manifest["history"]["line_count"] == 1


def test_recursive_summary_retries_model_failure_within_daily_run(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    recursive_model = _RecursiveSummaryModel(fail_once=True)
    builder = _builder(
        dataset,
        tmp_path / "recursive-retry-cache",
        _SummaryModel(),
        _StructuredModel(),
        recursive_summary_model=recursive_model,
    )

    with builder.build(strategies=("recursive_summary",)) as snapshot:
        calls = snapshot.provider_calls()
        assert len(calls) == 3
        assert sum(bool(call["error"]) for call in calls) == 1
        assert snapshot.resolve(
            "cloud_recursive_summary",
            "Gary HUD brightness",
        ).content.endswith("hud_brightness: 8")

    assert len(recursive_model.requests) == 3


def test_fact_recursive_hybrid_composes_cached_contexts_and_fact_hints(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    recursive_model = _RecursiveSummaryModel()
    fact_model = _FactModel()
    cache_root = tmp_path / "hybrid-cache"

    fact_summary_model = _SummaryModel()
    fact_summary_model.max_output_tokens = 1_024
    fact_structured_model = _StructuredModel()
    fact_structured_model.max_output_tokens = 1_024
    fact_builder = _builder(
        dataset,
        cache_root,
        fact_summary_model,
        fact_structured_model,
        fact_model=fact_model,
    )
    with fact_builder.build(strategies=("fact_patch",)):
        pass

    multi_summary_model = _SummaryModel()
    multi_summary_model.max_output_tokens = 3_072
    multi_structured_model = _StructuredModel()
    multi_structured_model.max_output_tokens = 3_072
    multi_builder = _builder(
        dataset,
        cache_root,
        multi_summary_model,
        multi_structured_model,
        fact_model=fact_model,
    )
    with multi_builder.build(strategies=("structured", "fact_patch")) as multi_snapshot:
        multi_fact_cache_key = multi_snapshot.manifest["cache_key"]

    recursive_summary_model = _SummaryModel()
    recursive_summary_model.max_output_tokens = 2_048
    recursive_structured_model = _StructuredModel()
    recursive_structured_model.max_output_tokens = 2_048
    recursive_builder = _builder(
        dataset,
        cache_root,
        recursive_summary_model,
        recursive_structured_model,
        recursive_summary_model=recursive_model,
    )
    with recursive_builder.build(strategies=("recursive_summary",)):
        pass

    composed_summary_model = _SummaryModel()
    composed_summary_model.max_output_tokens = 4_096
    composed_structured_model = _StructuredModel()
    composed_structured_model.max_output_tokens = 4_096
    builder = _builder(
        dataset,
        cache_root,
        composed_summary_model,
        composed_structured_model,
        fact_model=fact_model,
        recursive_summary_model=recursive_model,
    )

    assert required_memory_strategies(("cloud_fact_recursive_hybrid",)) == (
        "recursive_summary",
        "fact_patch",
    )
    with builder.build_fact_recursive_hybrid() as snapshot:
        fingerprint = snapshot.memory_fingerprint()
        context = snapshot.resolve(
            "cloud_fact_recursive_hybrid",
            "Set Gary's usual HUD brightness.",
        )
        scripted = ScriptedAgentModel(
            [
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="set-hud-hybrid",
                            name="carcontrol_HUD_set_brightness_level",
                            arguments={"level": 8},
                        ),
                    ),
                ),
                AgentResponse(content="Applied."),
            ]
        )
        result = run_agent_evaluation(
            dataset,
            agent_model=scripted,
            profile_names=("cloud_fact_recursive_hybrid",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "hybrid-artifacts",
        )

        assert "## Query-selected Fact Memory" in context.content
        assert "## Scenario-wide Recursive Summary" in context.content
        assert "**Gary**\n- hud_brightness: 8" in context.content
        assert context.metadata["memory_strategy"] == ("fact_recursive_hybrid")
        assert context.metadata["retrieval_mode"] == (
            "fact_top_k_plus_full_recursive_summary"
        )
        assert context.metadata["query_dependent"] is True
        assert context.metadata["recall_at_k_applicable"] is True
        assert context.fact_records
        assert context.fact_query_context is not None
        assert context.trace["strategy"] == "fact_recursive_hybrid"
        assert snapshot.memory_fingerprint() == fingerprint
        assert snapshot.manifest["config"]["composition"] == (
            "fact-recursive-context-v1"
        )
        assert snapshot.manifest["config"]["components"]["fact_patch"] == (
            multi_fact_cache_key
        )

        task = result.tasks[0]
        assert task["score"]["exact_state_match"] is True
        assert task["context"]["selector_tool_names"] == [
            "carcontrol_HUD_set_brightness_level"
        ]
        assert task["context"]["tool_memory_execution_hint_count"] == 1
        assert task["memory_trace"]["strategy"] == ("fact_recursive_hybrid")
        profile_metrics = result.metrics["profiles"]["cloud_fact_recursive_hybrid"]
        assert profile_metrics["recursive_summary"]["snapshot_count"] == 1
        assert profile_metrics["fact_quality"]["active_record_count"] == 1
        system_prompt = scripted.requests[0][0][0].content
        assert "## Query-selected Fact Memory" in system_prompt
        assert "## Scenario-wide Recursive Summary" in system_prompt
        assert (
            dataset.scenario(1).history_path.read_text(encoding="utf-8").strip()
            not in system_prompt
        )

    assert len(fact_model.requests) == 2
    assert len(recursive_model.requests) == 2
    with builder.build_fact_recursive_hybrid() as cached:
        assert cached.manifest["status"] == "ready"
        assert cached.resolve(
            "cloud_fact_recursive_hybrid",
            "A second query",
        ).content
    assert len(fact_model.requests) == 2
    assert len(recursive_model.requests) == 2


def test_recursive_assisted_fact_adds_only_grounded_fact_context(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    recursive_model = _RecursiveSummaryModel()
    builder = _builder(
        dataset,
        tmp_path / "assisted-fact-cache",
        _SummaryModel(),
        _StructuredModel(),
        fact_model=_EmptyFactModel(),
        recursive_summary_model=recursive_model,
    )
    created_models = []

    def fact_model_factory(
        recursive_summary: str,
    ) -> _FactModel:
        assert "**Gary**\n- hud_brightness: 8" in recursive_summary
        assisted_model = _FactModel()
        assisted_model.prompt_version = "vehicle-recursive-assisted-fact-memory-test-v1"
        assisted_model.auxiliary_context = {"recursive_summary": recursive_summary}
        assisted_model.auxiliary_context_sha256 = "a" * 64
        created_models.append(assisted_model)
        return assisted_model

    assert required_memory_strategies(("cloud_recursive_assisted_fact_patch",)) == (
        "recursive_summary",
        "fact_patch",
    )
    with builder.build_recursive_assisted_fact(
        fact_model_factory=fact_model_factory,
    ) as snapshot:
        fingerprint = snapshot.memory_fingerprint()
        context = snapshot.resolve(
            "cloud_recursive_assisted_fact_patch",
            "Set Gary's usual HUD brightness.",
        )
        records = snapshot.fact_snapshot.repository.list_fact_memory_records(
            snapshot.fact_snapshot.fact_session_id
        )
        calls = snapshot.fact_snapshot.provider_calls()

        assert len(records) == 1
        assert records[0].predicate == "hud_brightness"
        assert records[0].value == 8
        assert "hud_brightness" in context.content
        assert "## Scenario-wide Recursive Summary" not in context.content
        assert context.metadata["memory_strategy"] == ("recursive_assisted_fact_patch")
        assert context.metadata["recursive_summary_agent_exposure"] is False
        assert context.trace["recursive_summary_agent_exposure"] is False
        assert context.fact_records[0].id == records[0].id
        assert snapshot.memory_fingerprint() == fingerprint
        assert snapshot.manifest["config"]["composition"] == (
            "recursive-assisted-fact-extraction-v1"
        )
        extraction_call = next(
            call
            for call in calls
            if call["role"] == "fact_memory_extraction"
            and call["metadata"].get("recursive_assisted")
        )
        assert extraction_call["metadata"]["recursive_assisted"] is True
        assert extraction_call["metadata"]["assisted_candidate_count"] == 1

    assert len(created_models) == 1
    assert len(created_models[0].requests) == 1


@pytest.mark.parametrize(
    "profile",
    (
        "cloud_schema_informed_recursive_assisted_fact_patch",
        "cloud_joint_planned_fact_patch",
    ),
)
def test_schema_informed_recursive_assisted_fact_uses_separate_profile(
    tmp_path: Path,
    profile: str,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    builder = _builder(
        dataset,
        tmp_path / "schema-assisted-fact-cache",
        _SummaryModel(),
        _StructuredModel(),
        fact_model=_SchemaEmptyFactModel(),
        recursive_summary_model=_RecursiveSummaryModel(),
    )

    def fact_model_factory(recursive_summary: str) -> _SchemaFactModel:
        assert "hud_brightness: 8" in recursive_summary
        model = _SchemaFactModel()
        model.auxiliary_context = {"recursive_summary": recursive_summary}
        model.auxiliary_context_sha256 = "b" * 64
        return model

    assert required_memory_strategies((profile,)) == (
        "recursive_summary",
        "fact_patch",
    )
    with builder.build_recursive_assisted_fact(
        fact_model_factory=fact_model_factory,
        profile_name=profile,
    ) as snapshot:
        context = snapshot.resolve(
            profile,
            "Set Gary's usual HUD brightness.",
        )

        assert snapshot.manifest["config"]["composition"] == (
            "post-normalized-recursive-assisted-fact-extraction-v1"
        )
        assert snapshot.manifest["config"]["profile"] == profile
        assert context.metadata["memory_strategy"] == (
            "schema_informed_recursive_assisted_fact_patch"
        )
        assert context.trace["composition_version"] == (
            "post-normalized-recursive-assisted-fact-extraction-v1"
        )


def test_recursive_summary_fails_build_after_bounded_retries(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    recursive_model = _FailingRecursiveSummaryModel()
    builder = _builder(
        dataset,
        tmp_path / "recursive-failure-cache",
        _SummaryModel(),
        _StructuredModel(),
        recursive_summary_model=recursive_model,
    )

    with pytest.raises(
        RuntimeError,
        match="Memory consolidation failed: recursive_summary:0",
    ):
        builder.build(strategies=("recursive_summary",))

    assert len(recursive_model.requests) == 2


def test_memory_build_cache_retrieval_and_agent_context_are_immutable(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    summary_model = _SummaryModel()
    structured_model = _StructuredModel()
    builder = _builder(
        dataset,
        tmp_path / "cache",
        summary_model,
        structured_model,
    )

    with builder.build() as snapshot:
        fingerprint = snapshot.memory_fingerprint()
        summary = snapshot.resolve(
            "cloud_summary",
            "Gary HUD brightness",
        )
        bm25 = snapshot.resolve(
            "cloud_structured_bm25",
            "Gary HUD brightness",
        )
        progress_before = snapshot.repository.consolidation_progress(
            snapshot.structured_session_id
        )
        scripted = ScriptedAgentModel(
            [
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="discover-hud",
                            name="list_module_tools",
                            arguments={"module_name": "HUD"},
                        ),
                    ),
                ),
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="set-hud",
                            name="carcontrol_HUD_set_brightness_level",
                            arguments={"level": 8},
                        ),
                    ),
                ),
                AgentResponse(content="Applied."),
            ]
        )
        result = run_agent_evaluation(
            dataset,
            agent_model=scripted,
            profile_names=("cloud_structured_bm25",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "artifacts",
        )
        progress_after = snapshot.repository.consolidation_progress(
            snapshot.structured_session_id
        )

        assert summary.content == "- Gary HUD brightness: 8"
        assert "Gary.hud_brightness: 8" in bm25.content
        assert bm25.trace["retrieval"]["run"]["status"] == "completed"
        assert bm25.trace["selected_memories"][0]["sources"][0]["evidence_text"]
        assert result.status == "completed"
        assert result.tasks[0]["score"]["exact_state_match"] is True
        assert result.tasks[0]["context"]["query_updates_memory"] is False
        assert result.tasks[0]["memory_trace"]["mode"] == "bm25"
        assert snapshot.memory_fingerprint() == fingerprint
        assert progress_after["message_count"] == progress_before["message_count"]
        system_prompt = scripted.requests[0][0][0].content
        assert "Gary.hud_brightness: 8" in system_prompt
        assert (
            dataset.scenario(1).history_path.read_text(encoding="utf-8").strip()
            not in system_prompt
        )

    assert len(summary_model.requests) == 1
    assert len(structured_model.requests) == 1
    manifest_path = next((tmp_path / "cache").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "building"
    for profile in manifest["profiles"].values():
        profile["status"] = "building"
        profile["completed_batches"] = 0
        profile["consolidation_run_ids"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with builder.build() as cached:
        assert cached.manifest["status"] == "ready"
        assert all(
            profile["completed_batches"] == 1
            and len(profile["consolidation_run_ids"]) == 1
            for profile in cached.manifest["profiles"].values()
        )
    assert len(summary_model.requests) == 1
    assert len(structured_model.requests) == 1


def test_memory_cache_resumes_pending_batch_without_duplicate_ingestion(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    structured_model = _StructuredModel(fail_once=True)
    builder = _builder(
        dataset,
        tmp_path / "cache",
        _SummaryModel(),
        structured_model,
    )

    with pytest.raises(RuntimeError, match="Memory consolidation failed"):
        builder.build(strategies=("structured",))

    with builder.build(strategies=("structured",)) as snapshot:
        progress = snapshot.repository.consolidation_progress(
            snapshot.structured_session_id
        )
        assert progress["message_count"] == 1
        assert progress["completed_message_count"] == 1
        assert progress["run_status_counts"] == {
            "completed": 1,
            "failed": 1,
        }
        assert snapshot.manifest["profiles"]["structured"]["completed_batches"] == 1

    assert len(structured_model.requests) == 2


def test_empty_summary_completes_batch_and_advances_cache_watermark(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    summary_model = _EmptySummaryModel()
    builder = _builder(
        dataset,
        tmp_path / "cache",
        summary_model,
        _StructuredModel(),
    )

    with builder.build(strategies=("summary",)) as snapshot:
        progress = snapshot.repository.consolidation_progress(
            snapshot.summary_session_id
        )
        context = snapshot.resolve("cloud_summary", "unrelated question")

        assert progress["message_count"] == 1
        assert progress["completed_message_count"] == 1
        assert progress["pending_message_count"] == 0
        assert progress["run_status_counts"] == {"completed": 1}
        assert context.content == ""
        assert snapshot.manifest["profiles"]["summary"]["status"] == "ready"

    assert len(summary_model.requests) == 1


def test_schema_patch_replay_retrieval_agent_and_metrics_are_isolated(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    patch_model = _PatchModel()
    builder = _builder(
        dataset,
        tmp_path / "cache",
        _SummaryModel(),
        _StructuredModel(),
        patch_model,
    )

    assert required_memory_strategies(("cloud_schema_patch",)) == ("schema_patch",)
    assert required_memory_strategies(("oracle_tool_schema_patch",)) == (
        "schema_patch",
    )
    with builder.build(strategies=("schema_patch",)) as snapshot:
        fingerprint = snapshot.memory_fingerprint()
        context = snapshot.resolve(
            "cloud_schema_patch",
            "Set Gary's usual HUD brightness.",
        )
        oracle_context = snapshot.resolve(
            "oracle_tool_schema_patch",
            "Set Gary's usual HUD brightness.",
            ("carcontrol_HUD_set_brightness_level",),
        )
        scripted = ScriptedAgentModel(
            [
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="discover-hud",
                            name="list_module_tools",
                            arguments={"module_name": "HUD"},
                        ),
                    ),
                ),
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="set-hud",
                            name="carcontrol_HUD_set_brightness_level",
                            arguments={"level": 8},
                        ),
                    ),
                ),
                AgentResponse(content="Applied."),
            ]
        )
        result = run_agent_evaluation(
            dataset,
            agent_model=scripted,
            profile_names=("cloud_schema_patch",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "artifacts",
        )

        assert "domain=hud" in context.content
        assert "topic=brightness_level" in context.content
        assert 'conditions={"person":"gary"}' in context.content
        assert "value=8" in context.content
        assert "evidence" not in context.content
        assert context.trace["strategy"] == "schema_patch"
        assert context.trace["retrieval"]["run"]["status"] == "completed"
        assert oracle_context.trace["oracle_tool_routing"] is True
        assert (
            oracle_context.trace["retrieval"]["run"]["route"]["reason"]
            == "oracle_tool_names"
        )
        quality = context.trace["patch_quality"]
        assert quality["run_count"] == 1
        assert quality["operation_counts"] == {"ADD": 1}
        assert quality["active_record_count"] == 1
        assert quality["evidence_acceptance_rate"] == 1
        assert snapshot.memory_fingerprint() == fingerprint

        task = result.tasks[0]
        profile = result.metrics["profiles"]["cloud_schema_patch"]
        assert task["score"]["exact_state_match"] is True
        assert task["argument_exact_match"] is True
        assert task["retrieval_quality"]["recall_at_k"] == 1
        assert task["context"]["query_updates_memory"] is False
        assert profile["argument_exact_match"] == 1
        assert profile["retrieval_recall_at_k"] == 1
        assert profile["patch_quality"]["proposal_count"] == 1
        system_prompt = scripted.requests[0][0][0].content
        raw_history = (
            dataset.scenario(1).history_path.read_text(encoding="utf-8").strip()
        )
        assert raw_history not in system_prompt
        assert "evidence" not in system_prompt

    assert len(patch_model.requests) == 1
    assert "2025-01-01" in patch_model.requests[0][0][0].content
    assert patch_model.requests[0][2] == ()
    assert "2025-01-02" in patch_model.requests[0][0][1].content
    with builder.build(strategies=("schema_patch",)) as cached:
        assert cached.manifest["profiles"]["schema_patch"]["status"] == "ready"
        assert cached.manifest["profiles"]["schema_patch"]["ingested_turns"] == 2
        assert cached.repository.list_tool_memory_records(cached.patch_session_id)
    assert len(patch_model.requests) == 1


def test_fact_patch_replays_vehicle_history_without_tool_ontology(
    tmp_path: Path,
):
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    fact_model = _FactModel()
    builder = _builder(
        dataset,
        tmp_path / "fact-cache",
        _SummaryModel(),
        _StructuredModel(),
        fact_model=fact_model,
    )

    with builder.build(strategies=("fact_patch",)) as snapshot:
        records = snapshot.repository.list_fact_memory_records(snapshot.fact_session_id)
        profile = snapshot.manifest["profiles"]["fact_patch"]
        calls = snapshot.provider_calls()
        context = snapshot.resolve(
            "cloud_fact_patch",
            "Gary is driving; set his usual HUD brightness.",
        )
        oracle_context = snapshot.resolve(
            "oracle_tool_fact_patch",
            "Gary is driving; set his usual HUD brightness.",
            ("carcontrol_HUD_set_brightness_level",),
        )
        retrieval_label = OracleRetrievalLabel(
            task_id="vehicle-01-00",
            record_status="record_present",
            record_keys=(
                OracleFactRecordKey(
                    entity_id="gary",
                    predicate="hud_brightness",
                    value=8,
                ),
            ),
        )
        binding_context = snapshot.resolve(
            "oracle_binding_fact_patch",
            "Gary is driving; set his usual HUD brightness.",
            oracle_retrieval_label=retrieval_label,
        )
        cumulative_context = snapshot.resolve(
            "oracle_retrieval_binding_fact_patch",
            "Gary is driving; set his usual HUD brightness.",
            oracle_retrieval_label=retrieval_label,
        )
        scripted = ScriptedAgentModel(
            [
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="set-hud-fact",
                            name="carcontrol_HUD_set_brightness_level",
                            arguments={"level": 8},
                        ),
                    ),
                ),
                AgentResponse(content="Applied."),
            ]
        )
        evaluation = run_agent_evaluation(
            dataset,
            agent_model=scripted,
            profile_names=("cloud_fact_patch",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "fact-artifacts",
        )

        assert len(records) == 1
        assert records[0].entity_id == "gary"
        assert records[0].predicate == "hud_brightness"
        assert records[0].value == 8
        assert profile["status"] == "ready"
        assert profile["ingested_turns"] == 2
        assert profile["fact_quality"]["decision_counts"] == {"ADD": 1}
        assert calls[0]["role"] == "fact_memory_extraction"
        assert calls[0]["metadata"]["tool_ontology_included"] is False
        assert context.fact_records[0].id == records[0].id
        assert context.metadata["route_is_hard_filter"] is False
        assert context.trace["strategy"] == "fact_patch"
        assert oracle_context.trace["oracle_tool_routing"] is True
        assert binding_context.metadata["oracle_retrieval_all_records_selected"] is True
        assert cumulative_context.fact_records[0].id == records[0].id
        assert required_memory_strategies(("cloud_fact_patch",)) == ("fact_patch",)
        assert required_memory_strategies(("cloud_schema_informed_fact_patch",)) == (
            "fact_patch",
        )
        task = evaluation.tasks[0]
        assert task["score"]["exact_state_match"] is True
        assert task["argument_exact_match"] is True
        assert task["context"]["tool_memory_execution_hints"][0] == {
            "record_id": records[0].id,
            "tool_name": "carcontrol_HUD_set_brightness_level",
            "arguments": {"level": 8},
            "confidence": 0.99,
            "tool_domain": "hud",
            "topic": "brightness_level",
        }
        assert task["tool_boundary"]["router_preloaded_tools"] == [
            "carcontrol_HUD_set_brightness_level"
        ]

    assert len(fact_model.requests) == 1
    assert len(fact_model.requests[0][0]) == 2
    assert fact_model.requests[0][1] == ()

    with builder.build(strategies=("fact_patch",)) as cached:
        assert cached.manifest["profiles"]["fact_patch"]["status"] == "ready"
        assert cached.repository.list_fact_memory_records(cached.fact_session_id)
    assert len(fact_model.requests) == 1


def test_recursive_summary_gated_wiki_profile_and_fake_e2e(
    tmp_path: Path,
) -> None:
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    qa_path = root / "benchmark/qa_data/qa_1.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["related_to_vehicle_preference"][0]["query"] = (
        "Set Gary's usual HUD brightness when it rains."
    )
    _write(qa_path, json.dumps(qa))
    dataset = load_vehicle_benchmark(root, strict=False)
    builder = _builder(
        dataset,
        tmp_path / "summary-wiki-cache",
        _SummaryModel(),
        _StructuredModel(),
        recursive_summary_model=_RecursiveSummaryModel(),
    )

    assert required_memory_strategies(("cloud_recursive_summary_gated_wiki",)) == (
        "recursive_summary",
    )
    with builder.build(strategies=("recursive_summary",)) as snapshot:
        memory_fingerprint = snapshot.memory_fingerprint()
        baseline = snapshot.resolve(
            "cloud_recursive_summary",
            "Set Gary's usual HUD brightness.",
        )
        gated_closed = snapshot.resolve(
            "cloud_recursive_summary_gated_wiki",
            "Set Gary's usual HUD brightness.",
        )
        assert gated_closed.content == baseline.content
        assert gated_closed.metadata["summary_wiki_gate_open"] is False
        assert gated_closed.wiki_traversal is None

        task_query = dataset.scenario(1).tasks[0].query
        gated_open = snapshot.resolve(
            "cloud_recursive_summary_gated_wiki",
            task_query,
        )
        first_cache_signature = gated_open.metadata["summary_wiki_cache_signature"]
        repeated = snapshot.resolve(
            "cloud_recursive_summary_gated_wiki",
            task_query,
        )
        assert repeated.metadata["summary_wiki_cache_signature"] == (
            first_cache_signature
        )
        assert snapshot.memory_fingerprint() == memory_fingerprint
        assert gated_open.metadata["summary_wiki_gate_open"] is True
        assert gated_open.wiki_traversal is not None
        page_id = next(
            page.page_id
            for page in gated_open.wiki_traversal.index.pages
            if page.page_type == "summary_update"
        )
        model = ScriptedAgentModel(
            [
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="wiki-search",
                            name=MEMORY_WIKI_SEARCH,
                            arguments={"query": "Gary HUD brightness"},
                        ),
                    ),
                ),
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="wiki-read",
                            name=MEMORY_WIKI_READ,
                            arguments={"page_ids": [page_id]},
                        ),
                    ),
                ),
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="discover-hud",
                            name="list_module_tools",
                            arguments={"module_name": "HUD"},
                        ),
                    ),
                ),
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="set-hud",
                            name="carcontrol_HUD_set_brightness_level",
                            arguments={"level": 8},
                        ),
                    ),
                ),
                AgentResponse(content="Applied."),
            ]
        )
        evaluation = run_agent_evaluation(
            dataset,
            agent_model=model,
            profile_names=("cloud_recursive_summary_gated_wiki",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "summary-wiki-artifacts",
        )

        task = evaluation.tasks[0]
        assert task["score"]["exact_state_match"] is True
        assert [call["name"] for call in task["predicted_calls"]] == [
            "carcontrol_HUD_set_brightness_level"
        ]
        assert task["tool_boundary"]["initial_tools"] == [
            "list_module_tools",
            MEMORY_WIKI_SEARCH,
            MEMORY_WIKI_READ,
        ]
        assert (
            task["context"]["retrieval_metadata"]["summary_wiki_termination_reason"]
            == "evidence_sufficient"
        )
        traversal = task["memory_trace"]["summary_wiki"]["traversal"]
        assert traversal["search_count"] == 1
        assert traversal["read_count"] == 1
        assert traversal["fallback_used"] is False
        assert [item["call_kind"] for item in task["tool_trace"][:2]] == [
            "memory_retrieval",
            "memory_retrieval",
        ]
        profile_metrics = evaluation.metrics["profiles"][
            "cloud_recursive_summary_gated_wiki"
        ]
        assert profile_metrics["wiki_traversal"]["search_calls"] == 1
        assert profile_metrics["wiki_traversal"]["read_pages"] == 1
        assert profile_metrics["wiki_traversal"]["termination_reasons"] == {
            "evidence_sufficient": 1
        }
        assert (
            profile_metrics["workflow_stages"]["online_memory_retrieval"][
                "wiki_read_context_tokens"
            ]
            > 0
        )
        assert evaluation.metrics["artifact_privacy_audit"]["passed"] is True
        assert task["context"]["raw_history_included"] is False
        assert snapshot.memory_fingerprint() == memory_fingerprint

        resumed_model = ScriptedAgentModel([])
        resumed = run_agent_evaluation(
            dataset,
            agent_model=resumed_model,
            profile_names=("cloud_recursive_summary_gated_wiki",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "summary-wiki-artifacts",
            resume_run_id=evaluation.run_id,
        )
        assert resumed.status == "completed"
        assert not resumed_model.requests
        assert (
            resumed.metrics["profiles"]["cloud_recursive_summary_gated_wiki"][
                "exact_state_match"
            ]
            == 1
        )


def test_post_normalized_fact_wiki_profile_and_fake_e2e(
    tmp_path: Path,
) -> None:
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    builder = _builder(
        dataset,
        tmp_path / "fact-wiki-cache",
        _SummaryModel(),
        _StructuredModel(),
        fact_model=_SchemaEmptyFactModel(),
        recursive_summary_model=_RecursiveSummaryModel(),
    )

    def fact_model_factory(recursive_summary: str) -> _SchemaFactModel:
        model = _SchemaFactModel()
        model.auxiliary_context = {"recursive_summary": recursive_summary}
        model.auxiliary_context_sha256 = "c" * 64
        return model

    assert required_memory_strategies(("cloud_post_normalized_fact_wiki",)) == (
        "recursive_summary",
        "fact_patch",
    )
    with builder.build_recursive_assisted_fact(
        fact_model_factory=fact_model_factory,
        profile_name="cloud_post_normalized_fact_wiki",
    ) as snapshot:
        memory_fingerprint = snapshot.memory_fingerprint()
        query = dataset.scenario(1).tasks[0].query
        context = snapshot.resolve(
            "cloud_post_normalized_fact_wiki",
            query,
        )
        assert context.metadata["fact_wiki_available"] is True
        assert context.metadata["memory_strategy"] == ("post_normalized_fact_wiki")
        assert context.wiki_traversal is not None
        assert context.wiki_kind == "fact_wiki"
        first_cache_signature = context.metadata["fact_wiki_cache_signature"]
        repeated = snapshot.resolve(
            "cloud_post_normalized_fact_wiki",
            query,
        )
        assert repeated.metadata["fact_wiki_cache_signature"] == (first_cache_signature)
        assert snapshot.memory_fingerprint() == memory_fingerprint
        page_id = context.metadata["fact_wiki_seed_page_ids"][0]
        model = ScriptedAgentModel(
            [
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="wiki-search",
                            name=MEMORY_WIKI_SEARCH,
                            arguments={"query": "Gary HUD brightness"},
                        ),
                    ),
                ),
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="wiki-read",
                            name=MEMORY_WIKI_READ,
                            arguments={"page_ids": [page_id]},
                        ),
                    ),
                ),
                AgentResponse(
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="set-hud",
                            name="carcontrol_HUD_set_brightness_level",
                            arguments={"level": 8},
                        ),
                    ),
                ),
                AgentResponse(content="Applied."),
            ]
        )
        evaluation = run_agent_evaluation(
            dataset,
            agent_model=model,
            profile_names=("cloud_post_normalized_fact_wiki",),
            scenario_index=1,
            memory_resolver=snapshot.resolve,
            memory_manifest=snapshot.manifest,
            output_root=tmp_path / "fact-wiki-artifacts",
        )

        task = evaluation.tasks[0]
        assert task["score"]["exact_state_match"] is True
        assert [call["name"] for call in task["predicted_calls"]] == [
            "carcontrol_HUD_set_brightness_level"
        ]
        assert task["context"]["retrieval_metadata"]["memory_tool_plan_applied"] is True
        assert (
            task["context"]["retrieval_metadata"]["fact_wiki_termination_reason"]
            == "evidence_sufficient"
        )
        assert task["memory_trace"]["fact_wiki"]["traversal"]["selected_page_ids"] == (
            page_id,
        )
        assert task["tool_boundary"]["initial_tools"] == [
            "list_module_tools",
            MEMORY_WIKI_SEARCH,
            MEMORY_WIKI_READ,
            "carcontrol_HUD_set_brightness_level",
        ]
        profile_metrics = evaluation.metrics["profiles"][
            "cloud_post_normalized_fact_wiki"
        ]
        assert profile_metrics["wiki_traversal"]["available_tasks"] == 1
        assert profile_metrics["wiki_traversal"]["read_pages"] == 1
        assert (
            profile_metrics["workflow_stages"]["online_memory_retrieval"][
                "wiki_tool_calls"
            ]
            == 2
        )
        assert evaluation.metrics["artifact_privacy_audit"]["passed"] is True
        assert task["context"]["raw_history_included"] is False
        assert snapshot.memory_fingerprint() == memory_fingerprint


def test_post_normalized_fact_wiki_failure_uses_base_fact_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _memory_benchmark(tmp_path / "VehicleMemBench")
    dataset = load_vehicle_benchmark(root, strict=False)
    builder = _builder(
        dataset,
        tmp_path / "fact-wiki-fallback-cache",
        _SummaryModel(),
        _StructuredModel(),
        fact_model=_SchemaEmptyFactModel(),
        recursive_summary_model=_RecursiveSummaryModel(),
    )

    def fact_model_factory(recursive_summary: str) -> _SchemaFactModel:
        model = _SchemaFactModel()
        model.auxiliary_context = {"recursive_summary": recursive_summary}
        model.auxiliary_context_sha256 = "d" * 64
        return model

    with builder.build_recursive_assisted_fact(
        fact_model_factory=fact_model_factory,
        profile_name="cloud_post_normalized_fact_wiki",
    ) as snapshot:
        query = dataset.scenario(1).tasks[0].query
        baseline = snapshot.fact_snapshot.resolve("cloud_fact_patch", query)

        def fail_runtime(**_kwargs):
            raise RuntimeError("planned Fact Wiki failure")

        monkeypatch.setattr(
            vehicle_memory_module,
            "build_fact_wiki_runtime",
            fail_runtime,
        )
        context = snapshot.resolve(
            "cloud_post_normalized_fact_wiki",
            query,
        )

        assert context.content == baseline.content
        assert context.fact_records == baseline.fact_records
        assert context.fact_related_records == ()
        assert context.wiki_traversal is None
        assert context.metadata["fact_wiki_fallback_used"] is True
        assert context.metadata["retrieval_mode"] == ("assisted_fact_top_k_fallback")
        assert "planned Fact Wiki failure" in context.trace["fact_wiki"]["error"]
