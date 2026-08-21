from __future__ import annotations

from datetime import datetime, timedelta

from palmclaw_ubuntu.vehicle_bench.consensus_gold import (
    CONSENSUS_GOLD_SCHEMA_VERSION,
    CombinedTurnCandidatePayload,
    CompactionCandidatePayload,
    GeneratedCombinedCandidate,
    GeneratedCompactionCandidate,
    build_compaction_input,
    build_gold_generation_input,
)
from palmclaw_ubuntu.vehicle_bench.consensus_gold_pipeline import (
    ConsensusGoldPipeline,
    GoldCheckpointStore,
)
from palmclaw_ubuntu.vehicle_bench.human_review import (
    HumanReviewQueue,
    HumanReviewSubmission,
    build_reviewed_candidate,
    build_reviewed_compaction_candidate,
)
from palmclaw_ubuntu.vehicle_bench.memory import VehicleHistoryEntry


class FakeGenerator:
    def __init__(self, *, dissent: bool = False) -> None:
        self.dissent = dissent
        self.calls: list[tuple[int, str]] = []

    def generate(self, generation_input, *, sample_id):
        self.calls.append((generation_input.turn_index, sample_id))
        color = (
            "blue"
            if self.dissent and sample_id == "C"
            else _turn_color(generation_input.current_utterance)
        )
        if not generation_input.previous_memory:
            operation = {
                "op": "add",
                "target": "",
                "content": f"### Gary\n- Ambient color: {color}",
                "identity_key": "gary.ambient_color",
                "temporal_action": "durable_upsert",
                "temporal_cue": "",
            }
        else:
            old_color = (
                "green" if "green" in generation_input.previous_memory else "blue"
            )
            operation = {
                "op": "replace",
                "target": f"- Ambient color: {old_color}",
                "content": f"- Ambient color: {color}",
                "identity_key": "gary.ambient_color",
                "temporal_action": "durable_upsert",
                "temporal_cue": "",
            }
        payload = CombinedTurnCandidatePayload.model_validate(
            {
                "decision": "UPDATE",
                "operations": [operation],
                "reason": f"Gary explicitly prefers {color} ambient lighting.",
                "evidence": [{"quote": generation_input.current_utterance}],
            }
        )
        return GeneratedCombinedCandidate(
            sample_id=sample_id,
            payload=payload,
            response_id=f"{generation_input.turn_index}-{sample_id}",
            model_id="fake-luna",
            prompt_version="test",
            schema_version=CONSENSUS_GOLD_SCHEMA_VERSION,
            input_sha256=generation_input.input_sha256,
            usage={
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "cached_tokens": 0,
            },
        )


class SelectMajorityResolver:
    def __init__(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.calls = 0

    def resolve(self, *, generation_input, gate_results, consensus):
        self.calls += 1
        selected = next(
            result for result in gate_results if result.sample_id == self.sample_id
        )
        return selected.candidate.model_copy(update={"sample_id": "TERRA"})


class NoResolution:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, *, generation_input, gate_results, consensus):
        self.calls += 1
        return None


class FakeCompactor:
    def __init__(self, *, dissent: bool = False) -> None:
        self.dissent = dissent
        self.calls: list[tuple[int, str]] = []

    def generate(self, compaction_input, *, sample_id):
        self.calls.append((compaction_input.turn_index, sample_id))
        memory = (
            "Gary: green ambient"
            if self.dissent and sample_id == "C"
            else "Gary: green"
        )
        return GeneratedCompactionCandidate(
            sample_id=sample_id,
            payload=CompactionCandidatePayload(next_memory=memory),
            response_id=f"compaction-{sample_id}",
            model_id="fake-luna",
            prompt_version="test",
            schema_version=CONSENSUS_GOLD_SCHEMA_VERSION,
            input_sha256=compaction_input.input_sha256,
            usage={
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "cached_tokens": 0,
            },
        )


class CompactionSelectMajority:
    def __init__(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.calls = 0

    def resolve(self, *, compaction_input, gate_results, consensus):
        self.calls += 1
        selected = next(
            result for result in gate_results if result.sample_id == self.sample_id
        )
        return selected.candidate.model_copy(update={"sample_id": "SOL"})


class NoCompactionResolution:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, *, compaction_input, gate_results, consensus):
        self.calls += 1
        return None


def _entry(index: int, color: str) -> VehicleHistoryEntry:
    timestamp = datetime(2025, 4, 10, 13, 0) + timedelta(minutes=index)
    content = f"I prefer {color} ambient lighting."
    rendered = timestamp.strftime("%Y-%m-%d %H:%M")
    return VehicleHistoryEntry(
        line_number=index + 1,
        timestamp=timestamp,
        speaker="Gary",
        content=content,
        raw=f"[{rendered}] Gary: {content}",
    )


def _turn_color(content: str) -> str:
    return "blue" if "blue" in content else "green"


def test_pipeline_commits_strict_turn_order_and_resumes(tmp_path) -> None:
    entries = (_entry(0, "green"), _entry(1, "blue"))
    store = GoldCheckpointStore(
        tmp_path / "gold.sqlite",
        scenario_index=1,
        entries=entries,
    )
    generator = FakeGenerator()
    no_resolution = NoResolution()
    pipeline = ConsensusGoldPipeline(
        scenario_index=1,
        entries=entries,
        store=store,
        luna=generator,
        terra=no_resolution,
        sol=no_resolution,
    )

    first = pipeline.run(max_commits=1)

    assert first.status == "RUNNING"
    assert first.processed_turns == 1
    assert store.state().next_turn_index == 1
    assert len(store.active_labels()) == 1
    assert len(generator.calls) == 3

    resumed = pipeline.run()

    assert resumed.status == "COMPLETED"
    assert resumed.processed_turns == 1
    assert store.state().next_turn_index == 2
    assert resumed.final_memory.endswith("Ambient color: blue")
    assert len(generator.calls) == 6
    labels = store.active_labels()
    assert labels[1].before_memory_sha256 == labels[0].after_memory_sha256


def test_material_two_of_three_uses_terra_before_commit(tmp_path) -> None:
    entries = (_entry(0, "green"),)
    store = GoldCheckpointStore(
        tmp_path / "gold.sqlite",
        scenario_index=1,
        entries=entries,
    )
    terra = SelectMajorityResolver("A")
    pipeline = ConsensusGoldPipeline(
        scenario_index=1,
        entries=entries,
        store=store,
        luna=FakeGenerator(dissent=True),
        terra=terra,
        sol=NoResolution(),
    )

    result = pipeline.run()

    assert result.status == "COMPLETED"
    assert terra.calls == 1
    assert store.active_labels()[0].validation_path == (
        "luna_consensus",
        "terra",
        "rule_validator",
    )
    stages = [attempt["stage"] for attempt in store.attempts()]
    assert stages.count("LUNA") == 3
    assert stages.count("TERRA") == 1


def test_unresolved_escalation_pauses_without_advancing_turn(tmp_path) -> None:
    entries = (_entry(0, "green"), _entry(1, "blue"))
    store = GoldCheckpointStore(
        tmp_path / "gold.sqlite",
        scenario_index=1,
        entries=entries,
    )
    terra = NoResolution()
    sol = NoResolution()
    pipeline = ConsensusGoldPipeline(
        scenario_index=1,
        entries=entries,
        store=store,
        luna=FakeGenerator(dissent=True),
        terra=terra,
        sol=sol,
    )

    result = pipeline.run()

    assert result.status == "PAUSED_REVIEW"
    assert result.processed_turns == 0
    assert store.state().next_turn_index == 0
    assert store.active_labels() == ()
    assert store.state().pending_review["reason"] == "ESCALATION_UNRESOLVED"
    assert terra.calls == 1
    assert sol.calls == 1


def test_human_review_selects_valid_candidate_and_resumes_atomically(tmp_path) -> None:
    entries = (_entry(0, "green"),)
    checkpoint_path = tmp_path / "gold.sqlite"
    store = GoldCheckpointStore(
        checkpoint_path,
        scenario_index=1,
        entries=entries,
    )
    pipeline = ConsensusGoldPipeline(
        scenario_index=1,
        entries=entries,
        store=store,
        luna=FakeGenerator(dissent=True),
        terra=NoResolution(),
        sol=NoResolution(),
    )
    assert pipeline.run().status == "PAUSED_REVIEW"
    paused_state = store.state()
    queue = HumanReviewQueue(tmp_path / "reviews.sqlite")
    record = queue.register_paused(
        checkpoint_path=checkpoint_path,
        state=paused_state,
        entry=entries[0],
    )

    submitted = queue.submit(
        record.review_id,
        HumanReviewSubmission(
            reviewer="test-reviewer",
            action="SELECT",
            selected_sample_id="A",
            reason="Candidate A preserves the explicit durable preference.",
        ),
        expected_request_sha256=record.request_sha256,
    )
    generation_input = build_gold_generation_input(
        scenario_index=1,
        turn_index=0,
        entry=entries[0],
        previous_memory=paused_state.approved_memory,
    )
    candidate = build_reviewed_candidate(submitted, generation_input)
    resumed_state = pipeline.resume_human_candidate(candidate)
    queue.mark_applied(record.review_id)

    assert resumed_state.status == "COMPLETED"
    assert resumed_state.next_turn_index == 1
    assert store.active_labels()[0].validation_path == (
        "human_review",
        "rule_validator",
    )
    assert queue.get(record.review_id).status == "APPLIED"


def test_human_review_rejects_stale_request_hash(tmp_path) -> None:
    entries = (_entry(0, "green"),)
    checkpoint_path = tmp_path / "gold.sqlite"
    store = GoldCheckpointStore(
        checkpoint_path,
        scenario_index=1,
        entries=entries,
    )
    pipeline = ConsensusGoldPipeline(
        scenario_index=1,
        entries=entries,
        store=store,
        luna=FakeGenerator(dissent=True),
        terra=NoResolution(),
        sol=NoResolution(),
    )
    pipeline.run()
    queue = HumanReviewQueue(tmp_path / "reviews.sqlite")
    record = queue.register_paused(
        checkpoint_path=checkpoint_path,
        state=store.state(),
        entry=entries[0],
    )

    try:
        queue.submit(
            record.review_id,
            HumanReviewSubmission(
                reviewer="test-reviewer",
                action="NO_OP",
                reason="This turn does not establish durable memory.",
            ),
            expected_request_sha256="stale",
        )
    except ValueError as error:
        assert "hash has changed" in str(error)
    else:
        raise AssertionError("Stale Human Review request was accepted")

    submitted = queue.submit(
        record.review_id,
        HumanReviewSubmission(
            reviewer="test-reviewer",
            action="NO_OP",
            reason="This turn does not establish durable memory.",
        ),
        expected_request_sha256=record.request_sha256,
    )
    rejected = queue.mark_apply_error(
        submitted.review_id,
        ValueError("rule gate rejected the correction"),
    )
    revision = queue.register_paused(
        checkpoint_path=checkpoint_path,
        state=store.state(),
        entry=entries[0],
    )

    assert rejected.status == "REJECTED"
    assert revision.status == "PENDING"
    assert revision.review_id.endswith("-r2")


def test_compaction_threshold_compacts_then_commits_atomically(tmp_path) -> None:
    entries = (_entry(0, "green"),)
    store = GoldCheckpointStore(
        tmp_path / "gold.sqlite",
        scenario_index=1,
        entries=entries,
    )
    no_resolution = NoResolution()
    no_compaction_resolution = NoCompactionResolution()
    compactor = FakeCompactor()
    pipeline = ConsensusGoldPipeline(
        scenario_index=1,
        entries=entries,
        store=store,
        luna=FakeGenerator(),
        terra=no_resolution,
        sol=no_resolution,
        compactor_luna=compactor,
        compactor_terra=no_compaction_resolution,
        compactor_sol=no_compaction_resolution,
        compaction_add_threshold=1,
    )

    result = pipeline.run()

    assert result.status == "COMPLETED"
    assert store.state().next_turn_index == 1
    assert store.state().approved_memory == "Gary: green"
    assert store.state().add_count_since_compaction == 0
    label = store.active_labels()[0]
    assert label.compaction_triggered is True
    assert label.after_memory == "Gary: green"
    assert len(compactor.calls) == 3


def test_compaction_disagreement_uses_sol_before_commit(tmp_path) -> None:
    entries = (_entry(0, "green"),)
    store = GoldCheckpointStore(
        tmp_path / "gold.sqlite",
        scenario_index=1,
        entries=entries,
    )
    sol = CompactionSelectMajority("A")
    pipeline = ConsensusGoldPipeline(
        scenario_index=1,
        entries=entries,
        store=store,
        luna=FakeGenerator(),
        terra=NoResolution(),
        sol=NoResolution(),
        compactor_luna=FakeCompactor(dissent=True),
        compactor_terra=NoCompactionResolution(),
        compactor_sol=sol,
        compaction_add_threshold=1,
    )

    result = pipeline.run()

    assert result.status == "COMPLETED"
    assert sol.calls == 1
    assert store.active_labels()[0].after_memory == "Gary: green"
    stages = [attempt["stage"] for attempt in store.attempts()]
    assert stages.count("COMPACTION_LUNA") == 3
    assert stages.count("COMPACTION_TERRA") == 0
    assert stages.count("COMPACTION_SOL") == 1


def test_unresolved_compaction_pauses_without_committing_turn(tmp_path) -> None:
    entries = (_entry(0, "green"),)
    store = GoldCheckpointStore(
        tmp_path / "gold.sqlite",
        scenario_index=1,
        entries=entries,
    )
    terra = NoCompactionResolution()
    sol = NoCompactionResolution()
    pipeline = ConsensusGoldPipeline(
        scenario_index=1,
        entries=entries,
        store=store,
        luna=FakeGenerator(),
        terra=NoResolution(),
        sol=NoResolution(),
        compactor_luna=FakeCompactor(dissent=True),
        compactor_terra=terra,
        compactor_sol=sol,
        compaction_add_threshold=1,
    )

    result = pipeline.run()

    assert result.status == "PAUSED_REVIEW"
    assert store.state().next_turn_index == 0
    assert store.active_labels() == ()
    assert store.state().pending_review["reason"] == (
        "COMPACTION_ESCALATION_UNRESOLVED"
    )
    assert terra.calls == 0
    assert sol.calls == 1

    paused_state = store.state()
    queue = HumanReviewQueue(tmp_path / "reviews.sqlite")
    record = queue.register_paused(
        checkpoint_path=tmp_path / "gold.sqlite",
        state=paused_state,
        entry=entries[0],
    )
    submitted = queue.submit(
        record.review_id,
        HumanReviewSubmission(
            reviewer="test-reviewer",
            action="SELECT",
            selected_sample_id="A",
            reason="Candidate A preserves the facts while reducing the memory.",
        ),
        expected_request_sha256=record.request_sha256,
    )
    pending_payload = paused_state.pending_review["payload"]
    compaction_input = build_compaction_input(
        scenario_index=1,
        turn_index=0,
        post_patch_memory=pending_payload["post_patch_memory"],
    )
    candidate = build_reviewed_compaction_candidate(submitted, compaction_input)
    resumed_state = pipeline.resume_human_compaction(candidate)
    queue.mark_applied(record.review_id)

    assert resumed_state.status == "COMPLETED"
    assert resumed_state.next_turn_index == 1
    assert store.active_labels()[0].validation_path[-2:] == (
        "compaction_human_review",
        "compaction_rule_validator",
    )


def test_checkpoint_rejects_different_source(tmp_path) -> None:
    path = tmp_path / "gold.sqlite"
    GoldCheckpointStore(path, scenario_index=1, entries=(_entry(0, "green"),))

    try:
        GoldCheckpointStore(path, scenario_index=1, entries=(_entry(0, "blue"),))
    except ValueError as error:
        assert "source fingerprint" in str(error)
    else:
        raise AssertionError("Checkpoint accepted a different source")
