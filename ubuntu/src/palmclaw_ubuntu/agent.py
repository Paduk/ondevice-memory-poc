from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

from palmclaw_ubuntu.context import ContextBuilder
from palmclaw_ubuntu.contracts import AgentModel
from palmclaw_ubuntu.fact_memory_retrieval import FactMemoryRetriever
from palmclaw_ubuntu.memory import MemoryEngine
from palmclaw_ubuntu.memory_router import ToolMemoryRetriever
from palmclaw_ubuntu.memory_tool_planning import (
    build_memory_tool_plan,
    choose_memory_tool_hint_result,
    render_joint_tool_candidate_rationale,
)
from palmclaw_ubuntu.models import (
    AgentResponse,
    ChatMessage,
    ModelUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from palmclaw_ubuntu.privacy import redact_secrets
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tool_memory_execution import (
    build_fact_memory_execution_hints,
    build_tool_memory_execution_hints,
    routed_tool_names,
)
from palmclaw_ubuntu.tools import ToolExecutionContext, ToolRegistry


@dataclass(frozen=True)
class RunResult:
    turn_id: str
    content: str
    rounds: int
    status: str
    terminal_reason: str
    consolidation_run_id: str | None = None
    patch_job_id: str | None = None
    patch_queue_error: str | None = None


class AgentLoop:
    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        context_builder: ContextBuilder,
        tool_registry: ToolRegistry,
        agent_model: AgentModel,
        memory_engine: MemoryEngine,
        max_tool_rounds: int,
        max_history_messages: int,
        model_timeout_seconds: float,
        tool_memory_retriever: ToolMemoryRetriever | None = None,
        fact_memory_retriever: FactMemoryRetriever | None = None,
        joint_memory_tool_planning_enabled: bool = False,
        joint_memory_tool_max_tools: int = 12,
        joint_memory_tool_schema_tokens: int = 1_800,
    ):
        self.repository = repository
        self.context_builder = context_builder
        self.tool_registry = tool_registry
        self.agent_model = agent_model
        self.memory_engine = memory_engine
        self.max_tool_rounds = max_tool_rounds
        self.max_history_messages = max_history_messages
        self.model_timeout_seconds = model_timeout_seconds
        self.tool_memory_retriever = tool_memory_retriever
        self.fact_memory_retriever = fact_memory_retriever
        self.joint_memory_tool_planning_enabled = (
            joint_memory_tool_planning_enabled
        )
        self.joint_memory_tool_max_tools = joint_memory_tool_max_tools
        self.joint_memory_tool_schema_tokens = joint_memory_tool_schema_tokens

    def run(self, session_id: str, user_text: str) -> RunResult:
        self.repository.require_session(session_id)
        turn_id = self.repository.create_turn(session_id)
        safe_user_text = redact_secrets(user_text)
        self.repository.append_message(
            session_id,
            "user",
            safe_user_text,
            turn_id=turn_id,
        )
        last_content = ""
        rounds = 0
        attempted_unsafe_calls: set[str] = set()
        try:
            retrieved_memory = self.memory_engine.retrieve(
                session_id,
                turn_id=turn_id,
                query=safe_user_text,
            )
            tool_memory_content = ""
            tool_memory_hints = ()
            agent_tool_definitions = self.tool_registry.definitions()
            tool_memory_metadata: dict[str, object] = {
                "tool_memory_retrieval_status": "disabled",
                "tool_memory_retrieval_selected_count": 0,
            }
            if self.fact_memory_retriever is not None:
                try:
                    fact_memory = self.fact_memory_retriever.retrieve(
                        session_id,
                        turn_id=turn_id,
                        query=safe_user_text,
                    )
                    routed = routed_tool_names(fact_memory.metadata)
                    fallback_hint_result = build_fact_memory_execution_hints(
                        fact_memory.records,
                        self.tool_registry.definitions(),
                        fact_memory.query_context,
                        query=safe_user_text,
                        routed_tools=routed,
                    )
                    plan = None
                    plan_error = None
                    if self.joint_memory_tool_planning_enabled:
                        try:
                            plan = build_memory_tool_plan(
                                query=safe_user_text,
                                records=(
                                    *fact_memory.records,
                                    *fact_memory.related_records,
                                ),
                                definitions=self.tool_registry.definitions(),
                                query_context=fact_memory.query_context,
                                routed_tools=routed,
                                max_tools=self.joint_memory_tool_max_tools,
                                schema_token_budget=(
                                    self.joint_memory_tool_schema_tokens
                                ),
                            )
                        except Exception as exc:
                            plan_error = redact_secrets(
                                f"{type(exc).__name__}: {exc}"
                            )
                    planned_hints = plan.as_hint_result() if plan is not None else None
                    hint_result = choose_memory_tool_hint_result(
                        planned_hints,
                        fallback_hint_result,
                    )
                    tool_memory_content = fact_memory.content
                    if plan is not None and plan.candidates:
                        tool_memory_content = (
                            "[Joint-ranked Tool candidates and evidence]\n"
                            + render_joint_tool_candidate_rationale(
                                plan.candidates
                            )
                            + "\n\n"
                            + fact_memory.content
                        )
                    tool_memory_hints = hint_result.hints
                    tool_memory_metadata = {
                        **dict(fact_memory.metadata),
                        **hint_result.as_metadata(),
                        **(plan.as_metadata() if plan is not None else {}),
                        "memory_tool_plan_fallback": bool(
                            self.joint_memory_tool_planning_enabled
                            and hint_result is fallback_hint_result
                        ),
                        "memory_tool_plan_applied": bool(
                            planned_hints is not None
                            and hint_result is planned_hints
                        ),
                        "memory_tool_plan_error": plan_error,
                    }
                    joint_names = (
                        tuple(item.tool_name for item in plan.candidates)
                        if plan is not None
                        else ()
                    )
                    selected_tool_names = tuple(
                        dict.fromkeys(
                            (
                                *(joint_names or routed),
                                *(hint.tool_name for hint in hint_result.hints),
                            )
                        )
                    )
                    if selected_tool_names:
                        selected_set = set(selected_tool_names)
                        agent_tool_definitions = tuple(
                            definition
                            for definition in self.tool_registry.definitions()
                            if definition.name in selected_set
                        )
                    tool_memory_metadata["fact_memory_selected_tool_names"] = (
                        list(selected_tool_names)
                    )
                except Exception as exc:
                    tool_memory_metadata = {
                        "fact_memory_retrieval_status": "failed",
                        "fact_memory_retrieval_selected_count": 0,
                        "fact_memory_retrieval_error": redact_secrets(
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
            elif self.tool_memory_retriever is not None:
                try:
                    tool_memory = self.tool_memory_retriever.retrieve(
                        session_id,
                        turn_id=turn_id,
                        query=safe_user_text,
                    )
                    tool_memory_content = tool_memory.content
                    hint_result = build_tool_memory_execution_hints(
                        tool_memory.records,
                        self.tool_registry.definitions(),
                    )
                    tool_memory_hints = hint_result.hints
                    tool_memory_metadata = {
                        **dict(tool_memory.metadata),
                        **hint_result.as_metadata(),
                    }
                except Exception as exc:
                    tool_memory_metadata = {
                        "tool_memory_retrieval_status": "failed",
                        "tool_memory_retrieval_selected_count": 0,
                        "tool_memory_retrieval_error": redact_secrets(
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
            for rounds in range(1, self.max_tool_rounds + 1):
                stored_messages = self.repository.list_messages(
                    session_id,
                    limit=self.max_history_messages,
                )
                built_context = self.context_builder.build(
                    session_id=session_id,
                    messages=stored_messages,
                    memory=retrieved_memory.content,
                    memory_metadata=dict(retrieved_memory.metadata),
                    tool_memory=tool_memory_content,
                    tool_memory_metadata=tool_memory_metadata,
                    tool_memory_hints=tool_memory_hints,
                )
                response = self._complete_agent(
                    session_id=session_id,
                    turn_id=turn_id,
                    messages=built_context.messages,
                    context_metadata=built_context.metadata,
                    tool_definitions=agent_tool_definitions,
                )
                last_content = redact_secrets(response.content)
                self.repository.append_message(
                    session_id,
                    "assistant",
                    last_content,
                    turn_id=turn_id,
                    tool_calls=response.tool_calls,
                    provider_items=response.continuation_items,
                )
                if not response.tool_calls:
                    self.repository.finish_turn(
                        turn_id,
                        "completed",
                        rounds,
                        "assistant_response",
                    )
                    consolidation_run_id = self.memory_engine.consolidate(session_id)
                    return RunResult(
                        turn_id=turn_id,
                        content=last_content,
                        rounds=rounds,
                        status="completed",
                        terminal_reason="assistant_response",
                        consolidation_run_id=consolidation_run_id,
                    )

                for call in response.tool_calls:
                    definition = self.tool_registry.definition(call.name)
                    side_effect = definition.side_effect if definition else "unknown"
                    retry_safety = definition.retry_safety if definition else "safe"
                    fingerprint = self.tool_registry.fingerprint(call)
                    self.repository.record_tool_call(
                        turn_id,
                        rounds,
                        call,
                        side_effect=side_effect,
                        retry_safety=retry_safety,
                        fingerprint=fingerprint,
                    )
                    validation_error = self.tool_registry.validate(call)
                    if validation_error is not None:
                        result = validation_error
                        duration_ms = 0
                    elif (
                        retry_safety == "unsafe"
                        and fingerprint in attempted_unsafe_calls
                    ):
                        result = ToolResult(
                            tool_call_id=call.id,
                            content=(
                                '{"error":{"code":"duplicate_side_effect",'
                                '"message":"Repeated unsafe Tool call blocked"}}'
                            ),
                            is_error=True,
                            metadata={
                                "error_code": "duplicate_side_effect",
                            },
                        )
                        duration_ms = 0
                    else:
                        self.repository.mark_tool_call_executing(call.id)
                        if retry_safety == "unsafe":
                            attempted_unsafe_calls.add(fingerprint)
                        result, duration_ms = self.tool_registry.execute(
                            call,
                            ToolExecutionContext(session_id=session_id),
                        )
                    self._persist_tool_result(
                        session_id,
                        turn_id,
                        call,
                        result,
                        duration_ms,
                    )

            last_content = (
                f"Stopped after reaching max Tool rounds ({self.max_tool_rounds})."
            )
            self.repository.append_message(
                session_id,
                "assistant",
                last_content,
                turn_id=turn_id,
            )
            self.repository.finish_turn(
                turn_id,
                "failed",
                rounds,
                "max_rounds",
            )
            return RunResult(
                turn_id=turn_id,
                content=last_content,
                rounds=rounds,
                status="failed",
                terminal_reason="max_rounds",
            )
        except KeyboardInterrupt:
            self.repository.finish_turn(
                turn_id,
                "cancelled",
                rounds,
                "user_cancelled",
            )
            raise
        except TimeoutError:
            self.repository.finish_turn(
                turn_id,
                "timed_out",
                rounds,
                "provider_timeout",
            )
            raise
        except Exception as exc:
            self.repository.finish_turn(
                turn_id,
                "failed",
                rounds,
                f"{type(exc).__name__}: {exc}",
            )
            raise

    def _complete_agent(
        self,
        *,
        session_id: str,
        turn_id: str,
        messages: Sequence[ChatMessage],
        context_metadata: dict[str, object],
        tool_definitions: Sequence[ToolDefinition],
    ) -> AgentResponse:
        started = time.monotonic()
        try:
            response = self._invoke_with_timeout(
                lambda: self.agent_model.complete(
                    messages,
                    tool_definitions,
                ),
                timeout_seconds=self.model_timeout_seconds,
                label="AgentModel",
            )
        except Exception as exc:
            self.repository.record_model_call(
                session_id=session_id,
                turn_id=turn_id,
                role="agent",
                backend=self.agent_model.backend,
                model_id=self.agent_model.model_id,
                prompt_version=self.agent_model.prompt_version,
                schema_version=None,
                latency_ms=self._elapsed_ms(started),
                usage=ModelUsage(),
                response_id=None,
                metadata=context_metadata,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self.repository.record_model_call(
            session_id=session_id,
            turn_id=turn_id,
            role="agent",
            backend=self.agent_model.backend,
            model_id=self.agent_model.model_id,
            prompt_version=self.agent_model.prompt_version,
            schema_version=None,
            latency_ms=self._elapsed_ms(started),
            usage=response.usage,
            response_id=response.response_id,
            metadata={
                **context_metadata,
                **dict(response.metadata),
            },
        )
        return response

    def _persist_tool_result(
        self,
        session_id: str,
        turn_id: str,
        call: ToolCall,
        result: ToolResult,
        duration_ms: int,
    ) -> None:
        error_code = str(result.metadata.get("error_code", ""))
        status_by_error = {
            "timeout": "timed_out",
            "invalid_arguments": "invalid",
            "unknown_tool": "denied",
            "duplicate_side_effect": "denied",
            "permission_denied": "denied",
        }
        status = (
            status_by_error.get(error_code, "failed")
            if result.is_error
            else "succeeded"
        )
        self.repository.record_tool_result(
            call.id,
            content=result.content,
            is_error=result.is_error,
            metadata=dict(result.metadata),
            duration_ms=duration_ms,
            status=status,
        )
        self.repository.append_message(
            session_id,
            "tool",
            result.content,
            turn_id=turn_id,
            tool_call_id=call.id,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, round((time.monotonic() - started) * 1000))

    @staticmethod
    def _invoke_with_timeout(
        operation,
        *,
        timeout_seconds: float,
        label: str,
    ):
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=label.lower(),
        )
        future = executor.submit(operation)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(f"{label} timed out after {timeout_seconds:g}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
