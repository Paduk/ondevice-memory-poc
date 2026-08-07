from __future__ import annotations

import time
from dataclasses import replace

from palmclaw_ubuntu.application import create_runtime
from palmclaw_ubuntu.models import (
    FactMemoryCandidate,
    FactMemoryExtractionResponse,
    MemoryEvidence,
    ModelUsage,
    PatchMemoryResponse,
    ToolMemoryIdentity,
    ToolMemoryPatch,
)
from palmclaw_ubuntu.providers import ScriptedPatchMemoryModel
from palmclaw_ubuntu.storage import SQLiteRepository


class _TurnPatchModel:
    backend = "fake"
    model_id = "turn-patch"
    prompt_version = "tool-memory-patch-v1"
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
        source = next(message for message in messages if message.role == "user")
        return PatchMemoryResponse(
            patches=(
                ToolMemoryPatch(
                    operation="ADD",
                    identity=ToolMemoryIdentity(
                        user_id=user_id,
                        tool_domain="file",
                        topic="write",
                        scope="global",
                        scope_key=user_id,
                    ),
                    value="notes.txt",
                    memory_type="preference",
                    confidence=0.96,
                    evidence=(
                        MemoryEvidence(
                            message_id=source.id,
                            quote=source.content,
                        ),
                    ),
                    reason="durable file preference",
                ),
            ),
            usage=ModelUsage(
                input_tokens=30,
                output_tokens=12,
                total_tokens=42,
            ),
            response_id="patch-response",
            metadata={
                "privacy": {
                    "destination": "cloud",
                    "input_chars": len(source.content),
                    "sensitive_chars_after": 0,
                }
            },
        )


class _TurnFactModel:
    backend = "fake"
    model_id = "turn-fact"
    prompt_version = "fact-memory-extraction-v1"
    schema_version = "fact-memory-candidate-v1"

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
        source = next(message for message in messages if message.role == "user")
        return FactMemoryExtractionResponse(
            candidates=(
                FactMemoryCandidate(
                    entity_id="Patricia",
                    predicate="headrest height",
                    value=44,
                    identity_conditions={},
                    applicability={},
                    capability_hints=("seat",),
                    memory_type="preference",
                    confidence=0.96,
                    evidence=(
                        MemoryEvidence(
                            message_id=source.id,
                            quote=source.content,
                        ),
                    ),
                ),
            )
        )


class _SlowPatchModel:
    backend = "fake"
    model_id = "slow-patch"
    prompt_version = "tool-memory-patch-v1"
    schema_version = "tool-memory-patch-v1"

    def propose(self, *args, **kwargs):
        del args, kwargs
        time.sleep(0.1)
        return PatchMemoryResponse(patches=())


class _InvalidPartitionPatchModel(_TurnPatchModel):
    def propose(self, *args, **kwargs):
        response = super().propose(*args, **kwargs)
        patch = response.patches[0]
        assert patch.identity is not None
        return replace(
            response,
            patches=(
                replace(
                    patch,
                    identity=replace(
                        patch.identity,
                        tool_domain="shell",
                        topic="execute",
                    ),
                ),
            ),
        )


class _ReverseOrderedBatchPatchModel(_TurnPatchModel):
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
        patches = []
        for source in (message for message in messages if message.role == "user"):
            value = source.content.rsplit(" ", 1)[-1]
            patches.append(
                ToolMemoryPatch(
                    operation="ADD",
                    identity=ToolMemoryIdentity(
                        user_id=user_id,
                        tool_domain="file",
                        topic="write",
                        scope="conditional",
                        scope_key=user_id,
                        conditions={"filename": value},
                    ),
                    value=value,
                    memory_type="preference",
                    confidence=0.96,
                    evidence=(
                        MemoryEvidence(
                            message_id=source.id,
                            quote=source.content,
                        ),
                    ),
                    reason="durable file preference",
                )
            )
        return PatchMemoryResponse(patches=tuple(reversed(patches)))


class _MixedValidityBatchPatchModel(_TurnPatchModel):
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
        sources = tuple(
            message for message in messages if message.role == "user"
        )
        patches = []
        for index, source in enumerate(sources):
            value = source.content.rsplit(" ", 1)[-1]
            patches.append(
                ToolMemoryPatch(
                    operation="ADD",
                    identity=ToolMemoryIdentity(
                        user_id=user_id,
                        tool_domain="file" if index == 0 else "shell",
                        topic="write" if index == 0 else "execute",
                        scope="conditional",
                        scope_key=user_id,
                        conditions={"filename": value},
                    ),
                    value=value,
                    memory_type="preference",
                    confidence=0.96,
                    evidence=(
                        MemoryEvidence(
                            message_id=source.id,
                            quote=source.content,
                        ),
                    ),
                    reason="durable file preference",
                )
            )
        return PatchMemoryResponse(patches=tuple(patches))


def _patch_settings(settings, **changes):
    values = {
        "patch_memory_enabled": True,
        "patch_memory_backend": "fake",
        "patch_memory_user_id": "default_user",
        "patch_memory_retry_delay_seconds": 0,
        "patch_memory_lease_seconds": 3,
        "memory_trigger_messages": 100,
        **changes,
    }
    return replace(settings, **values)


def _completed_turn(
    repository: SQLiteRepository,
    session_id: str,
    content: str,
) -> str:
    turn_id = repository.create_turn(session_id)
    repository.append_message(
        session_id,
        "user",
        content,
        turn_id=turn_id,
    )
    repository.finish_turn(turn_id, "completed", 1, "test")
    return turn_id


def test_completed_turn_is_queued_and_applied_by_background_worker(settings):
    model = _TurnPatchModel()
    patched = _patch_settings(settings)
    with create_runtime(patched, patch_memory_model=model) as runtime:
        session = runtime.repository.create_session("Background patch")

        turn = runtime.turn_coordinator.run(
            session.id,
            "Always save reports to notes.txt",
        )

        assert turn.status == "completed"
        assert turn.patch_job_id is not None
        assert turn.patch_queue_error is None
        assert model.requests == []
        assert runtime.repository.memory_patch_job(turn.patch_job_id)["status"] == (
            "queued"
        )

        result = runtime.patch_worker.run_pending(limit=4)
        records = runtime.repository.list_tool_memory_records(session.id)
        trace = runtime.repository.memory_patch_trace(result.run_ids[0])
        queue = runtime.repository.memory_patch_queue_status(
            session_id=session.id
        )

    assert result.completed_count == 1
    assert records[0].value == "notes.txt"
    assert records[0].status == "active"
    assert trace["run"]["status"] == "completed"
    assert trace["job"]["source_turn_id"] == turn.turn_id
    assert trace["model_calls"][0]["role"] == "memory_patch"
    assert trace["model_calls"][0]["usage"]["total_tokens"] == 42
    assert trace["model_calls"][0]["metadata"]["patch_count"] == 1
    assert queue["ready_count"] == 0
    assert queue["processed_message_cursor"] > 0


def test_runtime_can_queue_and_apply_fact_first_worker(settings):
    model = _TurnFactModel()
    patched = replace(
        settings,
        fact_memory_enabled=True,
        fact_memory_backend="fake",
        fact_memory_user_id="default_user",
        patch_memory_retry_delay_seconds=0,
        patch_memory_lease_seconds=3,
        memory_trigger_messages=100,
    )
    with create_runtime(patched, fact_memory_model=model) as runtime:
        session = runtime.repository.create_session("Background Fact")
        turn = runtime.turn_coordinator.run(
            session.id,
            "Patricia prefers headrest height 44",
        )

        assert runtime.patch_worker is None
        assert runtime.fact_worker is not None
        assert turn.patch_job_id is not None
        assert model.requests == []

        result = runtime.fact_worker.run_pending(limit=4)
        records = runtime.repository.list_fact_memory_records(session.id)
        trace = runtime.repository.memory_patch_trace(result.run_ids[0])

    assert result.completed_count == 1
    assert records[0].entity_id == "patricia"
    assert records[0].predicate == "headrest_height"
    assert records[0].value == 44
    assert trace["run"]["status"] == "completed"
    assert trace["model_calls"][0]["role"] == "fact_memory_extraction"
    assert trace["fact_candidates"][0]["decision"] == "ADD"
    assert len(model.requests) == 1
    assert model.requests[0][1] == ()


def test_cloud_failure_retries_without_requeueing_the_turn(settings):
    model = ScriptedPatchMemoryModel(
        [
            RuntimeError("temporary provider failure"),
            PatchMemoryResponse(patches=()),
        ]
    )
    patched = _patch_settings(settings)
    with create_runtime(patched, patch_memory_model=model) as runtime:
        session = runtime.repository.create_session("Patch retry")
        turn = runtime.turn_coordinator.run(session.id, "Remember nothing")

        first = runtime.patch_worker.run_pending(limit=1)
        retrying = runtime.repository.memory_patch_job(turn.patch_job_id)
        second = runtime.patch_worker.run_pending(limit=1)
        completed = runtime.repository.memory_patch_job(turn.patch_job_id)
        runs = runtime.repository._connection.execute(
            """
            SELECT status
            FROM memory_patch_runs
            WHERE job_id = ?
            ORDER BY started_at
            """,
            (turn.patch_job_id,),
        ).fetchall()

    assert first.retryable_count == 1
    assert retrying["status"] == "retryable"
    assert second.completed_count == 1
    assert completed["status"] == "completed"
    assert completed["attempt_count"] == 2
    assert [row["status"] for row in runs] == ["failed", "completed"]
    assert len(model.requests) == 2


def test_background_patch_rejects_partition_outside_tool_ontology(settings):
    patched = _patch_settings(settings, patch_memory_max_attempts=1)
    with create_runtime(
        patched,
        patch_memory_model=_InvalidPartitionPatchModel(),
    ) as runtime:
        session = runtime.repository.create_session("Patch schema gate")
        turn = runtime.turn_coordinator.run(
            session.id,
            "Always save reports to notes.txt",
        )

        result = runtime.patch_worker.run_pending(limit=1)
        trace = runtime.repository.memory_patch_trace(result.run_ids[0])
        records = runtime.repository.list_tool_memory_records(session.id)

    assert turn.status == "completed"
    assert result.failed_count == 1
    assert records == []
    assert trace["proposals"][0]["status"] == "rejected"
    assert (
        trace["proposals"][0]["validation"]["code"]
        == "unknown_tool_schema_partition"
    )


def test_bounded_batch_uses_one_model_call_and_links_source_jobs(settings):
    patched = _patch_settings(settings)
    model = ScriptedPatchMemoryModel(
        [
            PatchMemoryResponse(
                patches=(),
                usage=ModelUsage(input_tokens=20, output_tokens=2),
            )
        ]
    )
    with create_runtime(patched, patch_memory_model=model) as runtime:
        session = runtime.repository.create_session("Patch batch")
        first = runtime.turn_coordinator.run(session.id, "first turn")
        second = runtime.turn_coordinator.run(session.id, "second turn")

        result = runtime.patch_worker.run_pending(limit=2)
        trace = runtime.repository.memory_patch_trace(result.run_ids[0])

    assert result.claimed_count == 2
    assert result.completed_count == 2
    assert len(result.run_ids) == 1
    assert len(model.requests) == 1
    assert trace["run"]["source_turn_id"] is None
    assert trace["job"] is None
    assert {job["source_turn_id"] for job in trace["jobs"]} == {
        first.turn_id,
        second.turn_id,
    }
    metadata = trace["model_calls"][0]["metadata"]
    assert metadata["batch_turn_count"] == 2
    assert metadata["batch_message_count"] == 4


def test_batch_patches_are_applied_in_source_message_order(settings):
    patched = _patch_settings(settings)
    model = _ReverseOrderedBatchPatchModel()
    with create_runtime(patched, patch_memory_model=model) as runtime:
        session = runtime.repository.create_session("Ordered patch batch")
        first = runtime.turn_coordinator.run(
            session.id,
            "Always save alpha.txt",
        )
        second = runtime.turn_coordinator.run(
            session.id,
            "Always save beta.txt",
        )

        result = runtime.patch_worker.run_pending(limit=2)
        trace = runtime.repository.memory_patch_trace(result.run_ids[0])

    proposal_message_ids = [
        proposal["evidence"][0]["message_id"]
        for proposal in trace["proposals"]
    ]
    assert len(model.requests) == 1
    assert result.completed_count == 2
    assert proposal_message_ids == sorted(proposal_message_ids)
    assert {job["source_turn_id"] for job in trace["jobs"]} == {
        first.turn_id,
        second.turn_id,
    }


def test_batch_retries_only_job_referenced_by_rejected_patch(settings):
    patched = _patch_settings(settings)
    model = _MixedValidityBatchPatchModel()
    with create_runtime(patched, patch_memory_model=model) as runtime:
        session = runtime.repository.create_session("Isolated patch batch")
        first = runtime.turn_coordinator.run(
            session.id,
            "Always save alpha.txt",
        )
        second = runtime.turn_coordinator.run(
            session.id,
            "Always save beta.txt",
        )

        result = runtime.patch_worker.run_pending(limit=2)
        trace = runtime.repository.memory_patch_trace(result.run_ids[0])
        first_job = runtime.repository.memory_patch_job(first.patch_job_id)
        second_job = runtime.repository.memory_patch_job(second.patch_job_id)
        records = runtime.repository.list_tool_memory_records(session.id)

    assert result.completed_count == 1
    assert result.retryable_count == 1
    assert result.failed_count == 0
    assert len(result.errors) == 1
    assert result.errors[0]["job_id"] == second.patch_job_id
    assert trace["run"]["status"] == "completed"
    assert [proposal["status"] for proposal in trace["proposals"]] == [
        "applied",
        "rejected",
    ]
    assert first_job["status"] == "completed"
    assert second_job["status"] == "retryable"
    assert len(records) == 1
    assert records[0].value == "alpha.txt"


def test_batch_token_limit_splits_large_turn_groups(settings):
    patched = _patch_settings(
        settings,
        patch_memory_batch_size=32,
        patch_memory_batch_tokens=256,
    )
    model = ScriptedPatchMemoryModel(
        [
            PatchMemoryResponse(patches=()),
            PatchMemoryResponse(patches=()),
        ]
    )
    with create_runtime(patched, patch_memory_model=model) as runtime:
        session = runtime.repository.create_session("Token bounded batch")
        long_text = " ".join(f"preference{i}" for i in range(300))
        runtime.turn_coordinator.run(session.id, long_text)
        runtime.turn_coordinator.run(session.id, long_text)

        result = runtime.patch_worker.run_pending(limit=2)

    assert result.completed_count == 2
    assert len(result.run_ids) == 2
    assert len(model.requests) == 2


def test_patch_timeout_does_not_fail_foreground_turn(settings):
    patched = _patch_settings(
        settings,
        model_timeout_seconds=0.01,
        patch_memory_lease_seconds=0.2,
        patch_memory_max_attempts=1,
    )
    with create_runtime(
        patched,
        patch_memory_model=_SlowPatchModel(),
    ) as runtime:
        session = runtime.repository.create_session("Patch timeout")

        turn = runtime.turn_coordinator.run(session.id, "Normal chat remains fast")
        result = runtime.patch_worker.run_pending(limit=1)
        job = runtime.repository.memory_patch_job(turn.patch_job_id)
        trace = runtime.repository.memory_patch_trace(result.run_ids[0])

    assert turn.status == "completed"
    assert turn.patch_job_id is not None
    assert result.failed_count == 1
    assert job["status"] == "failed"
    assert "timed out" in job["last_error"]
    assert "timed out" in trace["model_calls"][0]["error"]


def test_restart_resumes_running_job_and_checkpoints_completed_run(settings):
    patched = _patch_settings(settings)
    patched.ensure_directories()
    repository = SQLiteRepository(patched.database_path)
    session = repository.create_session("Patch recovery")
    interrupted_turn = _completed_turn(repository, session.id, "first")
    completed_turn = _completed_turn(repository, session.id, "second")
    interrupted_job = repository.enqueue_memory_patch_job(
        session_id=session.id,
        source_turn_id=interrupted_turn,
        max_attempts=3,
    )
    completed_job = repository.enqueue_memory_patch_job(
        session_id=session.id,
        source_turn_id=completed_turn,
        max_attempts=3,
    )
    claimed = repository.claim_memory_patch_jobs(limit=2, lease_seconds=30)
    assert {job["id"] for job in claimed} == {interrupted_job, completed_job}
    interrupted_run = repository.begin_memory_patch_run(
        session_id=session.id,
        source_turn_id=interrupted_turn,
        backend="fake",
        model_id="fake-patch",
        prompt_version="patch-v1",
        schema_version="patch-v1",
        status="running",
        job_id=interrupted_job,
    )
    completed_run = repository.begin_memory_patch_run(
        session_id=session.id,
        source_turn_id=completed_turn,
        backend="fake",
        model_id="fake-patch",
        prompt_version="patch-v1",
        schema_version="patch-v1",
        status="running",
        job_id=completed_job,
    )
    repository.finish_memory_patch_run(completed_run, status="completed")
    repository.close()

    with create_runtime(patched) as runtime:
        interrupted = runtime.repository.memory_patch_job(interrupted_job)
        completed = runtime.repository.memory_patch_job(completed_job)
        result = runtime.patch_worker.run_pending(limit=4)
        final_interrupted = runtime.repository.memory_patch_job(interrupted_job)
        final_completed = runtime.repository.memory_patch_job(completed_job)

    assert interrupted["status"] == "retryable"
    assert completed["status"] == "completed"
    assert result.claimed_count == 1
    assert result.completed_count == 1
    assert final_interrupted["status"] == "completed"
    assert final_interrupted["attempt_count"] == 2
    assert final_completed["attempt_count"] == 1
    assert runtime.recovery["memory_patch_jobs_completed"] == 1
    assert runtime.recovery["memory_patch_jobs_interrupted"] == 1
    assert runtime.recovery["memory_patch_runs_interrupted"] == 1
    assert interrupted_run != result.run_ids[0]


def test_restart_completes_all_jobs_linked_to_completed_batch_run(settings):
    patched = _patch_settings(settings)
    patched.ensure_directories()
    repository = SQLiteRepository(patched.database_path)
    session = repository.create_session("Batch recovery")
    turns = (
        _completed_turn(repository, session.id, "first"),
        _completed_turn(repository, session.id, "second"),
    )
    job_ids = tuple(
        repository.enqueue_memory_patch_job(
            session_id=session.id,
            source_turn_id=turn_id,
            max_attempts=3,
        )
        for turn_id in turns
    )
    repository.claim_memory_patch_jobs(limit=2, lease_seconds=30)
    run_id = repository.begin_memory_patch_run(
        session_id=session.id,
        source_turn_id=None,
        backend="fake",
        model_id="fake-patch",
        prompt_version="patch-batch-v2",
        schema_version="patch-v1",
        status="running",
    )
    repository.link_memory_patch_jobs_to_batch_run(
        job_ids,
        run_id=run_id,
        session_id=session.id,
    )
    repository.finish_memory_patch_run(run_id, status="completed")
    repository.close()

    with create_runtime(patched) as runtime:
        jobs = tuple(
            runtime.repository.memory_patch_job(job_id) for job_id in job_ids
        )

    assert all(job["status"] == "completed" for job in jobs)
    assert runtime.recovery["memory_patch_jobs_completed"] == 2
