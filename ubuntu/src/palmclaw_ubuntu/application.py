from __future__ import annotations

import os
from dataclasses import dataclass

from palmclaw_ubuntu.agent import AgentLoop
from palmclaw_ubuntu.background_memory import (
    BackgroundFactMemoryWorker,
    BackgroundMemoryPatchWorker,
)
from palmclaw_ubuntu.config import Settings
from palmclaw_ubuntu.context import ContextBuilder
from palmclaw_ubuntu.contracts import (
    AgentModel,
    EmbeddingModel,
    FactMemoryModel,
    MemoryModel,
    PatchMemoryModel,
    StructuredMemoryModel,
)
from palmclaw_ubuntu.coordinator import SessionTurnCoordinator
from palmclaw_ubuntu.fact_memory_retrieval import FactMemoryRetriever
from palmclaw_ubuntu.memory import MemoryEngine
from palmclaw_ubuntu.memory_router import ToolMemoryRetriever, ToolSchemaRouter
from palmclaw_ubuntu.providers import (
    EchoAgentModel,
    FakeEmbeddingModel,
    FakeFactMemoryModel,
    FakeMemoryModel,
    FakePatchMemoryModel,
    FakeStructuredMemoryModel,
    LocalMemoryModel,
    LocalStructuredMemoryModel,
    OpenAIEmbeddingModel,
    OpenAIFactMemoryModel,
    OpenAIMemoryModel,
    OpenAIPatchMemoryModel,
    OpenAIResponsesAgentModel,
    OpenAIStructuredMemoryModel,
)
from palmclaw_ubuntu.runtime_lock import RuntimeLease
from palmclaw_ubuntu.skills import SkillsLoader
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tools import (
    FileReadTool,
    FileWriteTool,
    ToolRegistry,
    WebFetchTool,
)
from palmclaw_ubuntu.validation import MemoryGatePolicy, MemoryValidationGate
from palmclaw_ubuntu.workspace import WorkspaceResolver


@dataclass
class Runtime:
    settings: Settings
    repository: SQLiteRepository
    skills_loader: SkillsLoader
    tool_registry: ToolRegistry
    memory_engine: MemoryEngine
    agent_loop: AgentLoop
    turn_coordinator: SessionTurnCoordinator
    patch_worker: BackgroundMemoryPatchWorker | None
    fact_worker: BackgroundFactMemoryWorker | None
    tool_memory_retriever: ToolMemoryRetriever | None
    fact_memory_retriever: FactMemoryRetriever | None
    recovery: dict[str, int]
    lease: RuntimeLease

    def close(self) -> None:
        try:
            self.repository.close()
        finally:
            self.lease.close()

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def create_runtime(
    settings: Settings,
    *,
    agent_model: AgentModel | None = None,
    memory_model: MemoryModel | None = None,
    structured_memory_model: StructuredMemoryModel | None = None,
    embedding_model: EmbeddingModel | None = None,
    patch_memory_model: PatchMemoryModel | None = None,
    fact_memory_model: FactMemoryModel | None = None,
) -> Runtime:
    settings.ensure_directories()
    lease = RuntimeLease(settings.data_dir / ".runtime.lock")
    try:
        repository = SQLiteRepository(settings.database_path)
    except Exception:
        lease.close()
        raise
    try:
        recovery = repository.recover_interrupted_execution()
        return _assemble_runtime(
            settings,
            repository=repository,
            lease=lease,
            recovery=recovery,
            agent_model=agent_model,
            memory_model=memory_model,
            structured_memory_model=structured_memory_model,
            embedding_model=embedding_model,
            patch_memory_model=patch_memory_model,
            fact_memory_model=fact_memory_model,
        )
    except Exception:
        repository.close()
        lease.close()
        raise


def _assemble_runtime(
    settings: Settings,
    *,
    repository: SQLiteRepository,
    lease: RuntimeLease,
    recovery: dict[str, int],
    agent_model: AgentModel | None,
    memory_model: MemoryModel | None,
    structured_memory_model: StructuredMemoryModel | None,
    embedding_model: EmbeddingModel | None,
    patch_memory_model: PatchMemoryModel | None,
    fact_memory_model: FactMemoryModel | None,
) -> Runtime:
    skills_loader = SkillsLoader(
        settings.builtin_skills_root,
        settings.workspace_skills_root,
    )
    workspace_resolver = WorkspaceResolver(
        settings.workspace_root,
        settings.shared_workspace_root,
    )
    tool_registry = ToolRegistry(
        [
            FileReadTool(
                workspace_resolver,
                timeout_seconds=settings.tool_timeout_seconds,
                max_file_bytes=settings.max_file_bytes,
            ),
            FileWriteTool(
                workspace_resolver,
                timeout_seconds=settings.tool_timeout_seconds,
                max_file_bytes=settings.max_file_bytes,
            ),
            WebFetchTool(
                timeout_seconds=settings.tool_timeout_seconds,
                max_bytes=settings.max_web_bytes,
                max_redirects=settings.max_web_redirects,
            ),
        ],
        max_result_chars=settings.max_tool_result_chars,
    )
    if settings.backend == "fake":
        agent_model = agent_model or EchoAgentModel()
    else:
        settings.require_cloud_configuration()
        agent_model = agent_model or OpenAIResponsesAgentModel(
            settings.agent_model or "",
            timeout_seconds=settings.model_timeout_seconds,
            max_output_tokens=settings.agent_max_output_tokens,
            reasoning_effort=settings.agent_reasoning_effort,
            redact_pii=settings.cloud_pii_redaction,
            pii_allowlist=settings.pii_allowlist,
        )

    resolved_memory_backend = (
        settings.backend
        if settings.memory_backend == "auto"
        else settings.memory_backend
    )
    if resolved_memory_backend == "fake":
        memory_model = memory_model or FakeMemoryModel()
        structured_memory_model = structured_memory_model or FakeStructuredMemoryModel()
    elif resolved_memory_backend == "local":
        memory_model = memory_model or LocalMemoryModel(
            settings.local_memory_model,
            base_url=settings.local_memory_base_url,
            api_key=settings.local_memory_api_key,
            timeout_seconds=settings.model_timeout_seconds,
            max_output_tokens=settings.memory_max_output_tokens,
            pii_allowlist=settings.pii_allowlist,
        )
        structured_memory_model = structured_memory_model or LocalStructuredMemoryModel(
            settings.local_memory_model,
            base_url=settings.local_memory_base_url,
            api_key=settings.local_memory_api_key,
            timeout_seconds=settings.model_timeout_seconds,
            max_output_tokens=settings.memory_max_output_tokens,
            pii_allowlist=settings.pii_allowlist,
        )
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        if not settings.memory_model:
            raise RuntimeError("PALMCLAW_MEMORY_MODEL is not set")
        memory_model = memory_model or OpenAIMemoryModel(
            settings.memory_model,
            timeout_seconds=settings.model_timeout_seconds,
            max_output_tokens=settings.memory_max_output_tokens,
            reasoning_effort=settings.memory_reasoning_effort,
            redact_pii=settings.cloud_pii_redaction,
            pii_allowlist=settings.pii_allowlist,
        )
        structured_memory_model = (
            structured_memory_model
            or OpenAIStructuredMemoryModel(
                settings.memory_model,
                timeout_seconds=settings.model_timeout_seconds,
                max_output_tokens=settings.memory_max_output_tokens,
                reasoning_effort=settings.memory_reasoning_effort,
                redact_pii=settings.cloud_pii_redaction,
                pii_allowlist=settings.pii_allowlist,
            )
        )

    if settings.backend == "fake":
        embedding_model = embedding_model or FakeEmbeddingModel(
            settings.embedding_dimensions
        )
    else:
        embedding_model = embedding_model or OpenAIEmbeddingModel(
            settings.embedding_model or "",
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.model_timeout_seconds,
            redact_pii=settings.cloud_pii_redaction,
            pii_allowlist=settings.pii_allowlist,
        )

    context_builder = ContextBuilder(
        skills_loader,
        max_history_messages=settings.max_history_messages,
        max_context_tokens=settings.max_context_tokens,
        memory_token_budget=settings.memory_context_tokens,
        skill_token_budget=settings.skill_context_tokens,
        tool_memory_token_budget=settings.tool_memory_context_tokens,
    )
    memory_engine = MemoryEngine(
        repository=repository,
        strategy=settings.memory_strategy,
        summary_model=memory_model,
        structured_model=structured_memory_model,
        embedding_model=embedding_model,
        trigger_messages=settings.memory_trigger_messages,
        retrieval_mode=settings.retrieval_mode,
        top_k=settings.memory_top_k,
        token_budget=settings.memory_context_tokens,
        model_timeout_seconds=settings.model_timeout_seconds,
        validation_gate=MemoryValidationGate(
            MemoryGatePolicy(
                enabled=settings.memory_gate_enabled,
                minimum_confidence=settings.memory_min_confidence,
                global_minimum_confidence=settings.memory_global_min_confidence,
                review_high_sensitivity=(settings.memory_review_high_sensitivity),
                pii_allowlist=settings.pii_allowlist,
            )
        ),
        local_pii_storage=settings.local_pii_storage,
        pii_allowlist=settings.pii_allowlist,
        memory_input_cost_per_million=(settings.memory_input_cost_per_million),
        memory_output_cost_per_million=(settings.memory_output_cost_per_million),
    )
    agent_loop = AgentLoop(
        repository=repository,
        context_builder=context_builder,
        tool_registry=tool_registry,
        agent_model=agent_model,
        memory_engine=memory_engine,
        max_tool_rounds=settings.max_tool_rounds,
        max_history_messages=settings.max_history_messages,
        model_timeout_seconds=settings.model_timeout_seconds,
        tool_memory_retriever=(
            ToolMemoryRetriever(
                repository=repository,
                router=ToolSchemaRouter(
                    tool_registry.memory_ontology(),
                    max_routes=settings.tool_memory_max_routes,
                    embedding_model=(
                        embedding_model
                        if settings.tool_memory_semantic_routing_enabled
                        else None
                    ),
                    model_timeout_seconds=settings.model_timeout_seconds,
                    repository=repository,
                ),
                embedding_model=embedding_model,
                user_id=settings.patch_memory_user_id,
                vehicle_id=settings.tool_memory_vehicle_id,
                mode=settings.tool_memory_retrieval_mode,
                top_k=settings.tool_memory_top_k,
                token_budget=settings.tool_memory_context_tokens,
                model_timeout_seconds=settings.model_timeout_seconds,
                embedding_input_cost_per_million=(
                    settings.embedding_input_cost_per_million
                ),
            )
            if settings.tool_memory_retrieval_enabled
            and not settings.fact_memory_enabled
            else None
        ),
        fact_memory_retriever=(
            FactMemoryRetriever(
                repository=repository,
                router=ToolSchemaRouter(
                    tool_registry.memory_ontology(),
                    max_routes=settings.tool_memory_max_routes,
                    embedding_model=(
                        embedding_model
                        if settings.tool_memory_semantic_routing_enabled
                        else None
                    ),
                    model_timeout_seconds=settings.model_timeout_seconds,
                    repository=repository,
                ),
                embedding_model=embedding_model,
                user_id=settings.fact_memory_user_id,
                mode=settings.tool_memory_retrieval_mode,
                top_k=settings.tool_memory_top_k,
                token_budget=settings.tool_memory_context_tokens,
                model_timeout_seconds=settings.model_timeout_seconds,
                embedding_input_cost_per_million=(
                    settings.embedding_input_cost_per_million
                ),
            )
            if settings.tool_memory_retrieval_enabled
            and settings.fact_memory_enabled
            else None
        ),
        joint_memory_tool_planning_enabled=(
            settings.joint_memory_tool_planning_enabled
        ),
        joint_memory_tool_max_tools=settings.joint_memory_tool_max_tools,
        joint_memory_tool_schema_tokens=(
            settings.joint_memory_tool_schema_tokens
        ),
    )
    tool_memory_retriever = agent_loop.tool_memory_retriever
    fact_memory_retriever = agent_loop.fact_memory_retriever
    patch_worker = None
    if settings.patch_memory_enabled:
        resolved_patch_backend = (
            settings.backend
            if settings.patch_memory_backend == "auto"
            else settings.patch_memory_backend
        )
        if patch_memory_model is None and resolved_patch_backend == "fake":
            patch_memory_model = FakePatchMemoryModel()
        elif patch_memory_model is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            if not settings.patch_memory_model:
                raise RuntimeError("PALMCLAW_PATCH_MEMORY_MODEL is not set")
            patch_memory_model = OpenAIPatchMemoryModel(
                settings.patch_memory_model,
                timeout_seconds=settings.model_timeout_seconds,
                max_output_tokens=settings.memory_max_output_tokens,
                reasoning_effort=settings.memory_reasoning_effort,
                redact_pii=settings.cloud_pii_redaction,
                pii_allowlist=settings.pii_allowlist,
            )
        patch_worker = BackgroundMemoryPatchWorker(
            repository=repository,
            model=patch_memory_model,
            tool_registry=tool_registry,
            user_id=settings.patch_memory_user_id,
            model_timeout_seconds=settings.model_timeout_seconds,
            max_attempts=settings.patch_memory_max_attempts,
            retry_delay_seconds=settings.patch_memory_retry_delay_seconds,
            lease_seconds=settings.patch_memory_lease_seconds,
            memory_input_cost_per_million=(
                settings.memory_input_cost_per_million
            ),
            memory_output_cost_per_million=(
                settings.memory_output_cost_per_million
            ),
            batch_turn_limit=settings.patch_memory_batch_size,
            batch_token_limit=settings.patch_memory_batch_tokens,
        )
    fact_worker = None
    if settings.fact_memory_enabled:
        resolved_fact_backend = (
            settings.backend
            if settings.fact_memory_backend == "auto"
            else settings.fact_memory_backend
        )
        if fact_memory_model is None and resolved_fact_backend == "fake":
            fact_memory_model = FakeFactMemoryModel()
        elif fact_memory_model is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            if not settings.fact_memory_model:
                raise RuntimeError("PALMCLAW_FACT_MEMORY_MODEL is not set")
            fact_memory_model = OpenAIFactMemoryModel(
                settings.fact_memory_model,
                timeout_seconds=settings.model_timeout_seconds,
                max_output_tokens=settings.memory_max_output_tokens,
                reasoning_effort=settings.memory_reasoning_effort,
                redact_pii=settings.cloud_pii_redaction,
                pii_allowlist=settings.pii_allowlist,
            )
        fact_worker = BackgroundFactMemoryWorker(
            repository=repository,
            model=fact_memory_model,
            user_id=settings.fact_memory_user_id,
            model_timeout_seconds=settings.model_timeout_seconds,
            max_attempts=settings.patch_memory_max_attempts,
            retry_delay_seconds=settings.patch_memory_retry_delay_seconds,
            lease_seconds=settings.patch_memory_lease_seconds,
            memory_input_cost_per_million=(
                settings.memory_input_cost_per_million
            ),
            memory_output_cost_per_million=(
                settings.memory_output_cost_per_million
            ),
            batch_turn_limit=settings.patch_memory_batch_size,
            batch_token_limit=settings.patch_memory_batch_tokens,
            linking_context_limit=settings.fact_memory_linking_context_limit,
            pii_allowlist=settings.pii_allowlist,
        )
    turn_coordinator = SessionTurnCoordinator(
        agent_loop,
        max_concurrent_sessions=settings.max_concurrent_sessions,
        patch_worker=patch_worker or fact_worker,
    )
    return Runtime(
        settings=settings,
        repository=repository,
        skills_loader=skills_loader,
        tool_registry=tool_registry,
        memory_engine=memory_engine,
        agent_loop=agent_loop,
        turn_coordinator=turn_coordinator,
        patch_worker=patch_worker,
        fact_worker=fact_worker,
        tool_memory_retriever=tool_memory_retriever,
        fact_memory_retriever=fact_memory_retriever,
        recovery=recovery,
        lease=lease,
    )
