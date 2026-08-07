from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict
from typing import Any

from palmclaw_ubuntu.contracts import EmbeddingModel
from palmclaw_ubuntu.memory import bm25, cosine_similarity
from palmclaw_ubuntu.memory_router import SchemaRouteDecision, ToolSchemaRouter
from palmclaw_ubuntu.models import (
    EmbeddingResponse,
    FactMemoryRecord,
    FactMemoryRetrievalResult,
    FactQueryContext,
    ModelUsage,
)
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tokens import TokenCounter
from palmclaw_ubuntu.tool_memory_schema import normalize_identifier

_WORD = re.compile(r"[\w가-힣]+", re.UNICODE)
_CLOCK = re.compile(
    r"\b(1[0-2]|0?\d)(?::([0-5]\d))?\s*([ap])\.?m\.?\b",
    re.IGNORECASE,
)
_RUNTIME_KEYS = frozenset(
    {"driver", "light", "occupant", "passenger", "person", "seat", "side", "zone"}
)
_DRIVER_TERMS = (
    "is driving",
    "was driving",
    "will drive",
    "drove",
    "driver's seat",
    "driving today",
    "운전",
)
_PASSENGER_TERMS = ("passenger", "riding shotgun", "조수석", "동승")
_WEATHER_ALIASES = {
    "rain": ("rain", "rainy", "raining", "storm", "wet", "비", "폭우"),
    "snow": ("snow", "snowy", "icy", "눈", "빙판"),
    "hot": ("hot", "warm", "heat", "더운", "더위"),
    "cold": ("cold", "cool", "freezing", "추운", "한파"),
}


def interpret_fact_query(
    query: str,
    *,
    known_entities: Sequence[str] = (),
    condition_context: Mapping[str, Any] | None = None,
) -> FactQueryContext:
    """Extract only request-local identity and applicability information.

    This parser intentionally does not decide which Tool or Fact is correct.
    It supplies deterministic signals to retrieval and late binding; semantic
    BM25/embedding scores still participate when wording is indirect.
    """

    folded = query.casefold()
    context = {
        normalize_identifier(str(key)): value
        for key, value in (condition_context or {}).items()
    }
    mentioned: list[str] = []
    roles: dict[str, str] = {}
    requester_candidates: list[tuple[int, str]] = []
    normalized_entities = tuple(
        dict.fromkeys(normalize_identifier(item) for item in known_entities)
    )
    alias_owners: dict[str, set[str]] = {}
    for entity in normalized_entities:
        for alias in _entity_aliases(entity):
            alias_owners.setdefault(alias, set()).add(entity)
    for entity in normalized_entities:
        full_alias = entity.replace("_", " ").casefold()
        aliases = tuple(
            alias
            for alias in _entity_aliases(entity)
            if alias == full_alias or len(alias_owners[alias]) == 1
        )
        spans = [
            match.span()
            for alias in aliases
            for match in re.finditer(rf"\b{re.escape(alias)}\b", folded)
        ]
        if not spans:
            continue
        mentioned.append(entity)
        for _, end in spans:
            suffix = folded[end : min(len(folded), end + 48)]
            marker = re.search(
                r"^\W*(?:\w+\W+){0,3}"
                r"(?:asked|asks|told|tells|said|says|requested|requests)\b",
                suffix,
            )
            if marker is not None:
                requester_candidates.append((end + marker.start(), entity))
        nearest_role = _nearest_role(folded, spans)
        if nearest_role is not None:
            roles[entity] = nearest_role

    if len(mentioned) == 1 and any(
        marker in folded
        for marker in ("i'm driving", "i am driving", "i'll drive", "제가 운전")
    ):
        roles[mentioned[0]] = "driver"

    time_of_day = _time_of_day(folded, context)
    weather = _weather(folded, context)
    situation = _first_context_value(
        context,
        ("situation", "context", "use_case", "location", "place"),
    )
    return FactQueryContext(
        entities=tuple(dict.fromkeys(mentioned)),
        entity_roles=roles,
        requester_entity=(
            max(requester_candidates)[1] if requester_candidates else None
        ),
        time_of_day=time_of_day,
        weather=weather,
        situation=str(situation) if situation is not None else None,
    )


class FactMemoryRetriever:
    """Memory-first retrieval over active Fact records.

    Tool routing contributes a score and a Tool candidate union, but never
    limits the SQL Fact partition. This is the key behavioral difference from
    the legacy Tool-memory retriever.
    """

    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        router: ToolSchemaRouter,
        embedding_model: EmbeddingModel,
        user_id: str,
        mode: str,
        top_k: int,
        token_budget: int,
        model_timeout_seconds: float,
        embedding_input_cost_per_million: float = 0.0,
        token_counter: TokenCounter | None = None,
    ):
        if mode not in {"none", "full", "bm25", "embedding", "hybrid"}:
            raise ValueError(f"Unsupported Fact memory retrieval mode: {mode}")
        if top_k < 1:
            raise ValueError("Fact memory top_k must be positive")
        if token_budget < 0:
            raise ValueError("Fact memory token budget cannot be negative")
        self.repository = repository
        self.router = router
        self.embedding_model = embedding_model
        self.user_id = normalize_identifier(user_id)
        self.mode = mode
        self.top_k = top_k
        self.token_budget = token_budget
        self.model_timeout_seconds = model_timeout_seconds
        self.embedding_input_cost_per_million = embedding_input_cost_per_million
        self.token_counter = token_counter or TokenCounter()

    def retrieve(
        self,
        session_id: str,
        *,
        turn_id: str | None,
        query: str,
        mode: str | None = None,
        top_k: int | None = None,
        condition_context: Mapping[str, Any] | None = None,
        oracle_tool_names: Sequence[str] = (),
    ) -> FactMemoryRetrievalResult:
        started = time.monotonic()
        selected_mode = mode or self.mode
        selected_top_k = self.top_k if top_k is None else top_k
        if selected_mode not in {"none", "full", "bm25", "embedding", "hybrid"}:
            raise ValueError(
                f"Unsupported Fact memory retrieval mode: {selected_mode}"
            )
        records = self.repository.list_fact_memory_records(
            session_id,
            user_id=self.user_id,
        )
        query_context = interpret_fact_query(
            query,
            known_entities=tuple(record.entity_id for record in records),
            condition_context=condition_context,
        )
        route = (
            self.router.route_tool_names(oracle_tool_names)
            if oracle_tool_names
            else self.router.route(query)
        )
        run_id = self.repository.begin_fact_memory_retrieval(
            session_id=session_id,
            turn_id=turn_id,
            query=query,
            user_id=self.user_id,
            mode=selected_mode,
            top_k=selected_top_k,
            token_budget=self.token_budget,
            query_context=asdict(query_context),
            route=route.as_dict(),
        )
        try:
            self._record_route_embedding(
                route,
                session_id=session_id,
                turn_id=turn_id,
                run_id=run_id,
            )
            return self._rank(
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                query=query,
                mode=selected_mode,
                top_k=selected_top_k,
                query_context=query_context,
                condition_context=condition_context or {},
                route=route,
                records=records,
                started=started,
            )
        except Exception as exc:
            latency = _elapsed_ms(started)
            self.repository.fail_fact_memory_retrieval(
                run_id,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=latency,
            )
            return FactMemoryRetrievalResult(
                content="",
                query_context=query_context,
                run_id=run_id,
                metadata={
                    "fact_memory_retrieval_run_id": run_id,
                    "fact_memory_retrieval_status": "failed",
                    "fact_memory_retrieval_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "fact_memory_retrieval_selected_count": 0,
                    "fact_memory_retrieval_latency_ms": latency,
                    "tool_memory_route": route.as_dict(),
                },
            )

    def apply_oracle_selection(
        self,
        result: FactMemoryRetrievalResult,
        *,
        session_id: str,
        task_id: str,
        record_status: str,
        record_keys: Sequence[Mapping[str, Any]],
    ) -> FactMemoryRetrievalResult:
        """Prepend reviewed Gold-relevant active records to the normal Top-k.

        This does not change routing, the Fact database, late binding, or Tool
        arguments. A record_absent label therefore leaves normal retrieval
        unchanged and records the upstream coverage gap.
        """

        inspected, oracle_records = self.inspect_oracle_selection(
            result,
            session_id=session_id,
            task_id=task_id,
            record_status=record_status,
            record_keys=record_keys,
        )
        baseline_ids = tuple(record.id for record in result.records)
        if record_status == "record_absent":
            return inspected

        ordered = [
            *oracle_records,
            *(
                record
                for record in result.records
                if record.id not in {item.id for item in oracle_records}
            ),
        ]
        selected: list[FactMemoryRecord] = []
        used_tokens = 0
        oracle_ids = {record.id for record in oracle_records}
        for record in ordered:
            if len(selected) >= self.top_k:
                break
            rendered = self._render_context(record)
            token_count = self.token_counter.count(rendered)
            if used_tokens + token_count > self.token_budget:
                if record.id in oracle_ids:
                    raise ValueError(
                        "Oracle Fact record exceeds retrieval token budget: "
                        f"{task_id}"
                    )
                continue
            selected.append(record)
            used_tokens += token_count

        already_selected = sum(record.id in baseline_ids for record in oracle_records)
        forced = len(oracle_records) - already_selected
        selected_ids = tuple(record.id for record in selected)
        return FactMemoryRetrievalResult(
            content="\n".join(self._render_context(record) for record in selected),
            records=tuple(selected),
            query_context=result.query_context,
            run_id=result.run_id,
            metadata={
                **dict(inspected.metadata),
                "fact_memory_retrieval_selected_count": len(selected),
                "fact_memory_retrieval_selected_ids": [
                    *selected_ids
                ],
                "fact_memory_retrieval_tokens": used_tokens,
                "oracle_retrieval_forced_count": forced,
                "oracle_retrieval_all_records_selected": True,
                "oracle_retrieval_context_changed": selected_ids
                != baseline_ids,
            },
        )

    def inspect_oracle_selection(
        self,
        result: FactMemoryRetrievalResult,
        *,
        session_id: str,
        task_id: str,
        record_status: str,
        record_keys: Sequence[Mapping[str, Any]],
    ) -> tuple[FactMemoryRetrievalResult, tuple[FactMemoryRecord, ...]]:
        """Attach reviewed record availability without changing Top-k order."""

        baseline_ids = tuple(record.id for record in result.records)
        if record_status == "record_absent":
            if record_keys:
                raise ValueError(
                    "record_absent Oracle label cannot contain record keys"
                )
            return FactMemoryRetrievalResult(
                content=result.content,
                records=result.records,
                query_context=result.query_context,
                run_id=result.run_id,
                metadata={
                    **dict(result.metadata),
                    "oracle_retrieval": True,
                    "oracle_retrieval_record_status": "record_absent",
                    "oracle_retrieval_record_ids": [],
                    "oracle_retrieval_baseline_selected_ids": list(baseline_ids),
                    "oracle_retrieval_already_selected_count": 0,
                    "oracle_retrieval_forced_count": 0,
                    "oracle_retrieval_all_records_selected": False,
                    "oracle_retrieval_context_changed": False,
                },
            ), ()
        if record_status != "record_present" or not record_keys:
            raise ValueError(f"Invalid Oracle Retrieval label for {task_id}")

        active = self.repository.list_fact_memory_records(
            session_id,
            user_id=self.user_id,
        )
        oracle_records: list[FactMemoryRecord] = []
        for key in record_keys:
            matches = [
                record for record in active if _oracle_key_matches(record, key)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Oracle Fact key must match exactly one active record: "
                    f"task={task_id}, key={dict(key)}, "
                    f"matches={len(matches)}"
                )
            if matches[0].id not in {record.id for record in oracle_records}:
                oracle_records.append(matches[0])

        already_selected = sum(record.id in baseline_ids for record in oracle_records)
        all_selected = already_selected == len(oracle_records)
        return FactMemoryRetrievalResult(
            content=result.content,
            records=result.records,
            query_context=result.query_context,
            run_id=result.run_id,
            metadata={
                **dict(result.metadata),
                "oracle_retrieval": True,
                "oracle_retrieval_record_status": "record_present",
                "oracle_retrieval_record_ids": [
                    record.id for record in oracle_records
                ],
                "oracle_retrieval_baseline_selected_ids": list(baseline_ids),
                "oracle_retrieval_already_selected_count": already_selected,
                "oracle_retrieval_forced_count": (
                    len(oracle_records) - already_selected
                ),
                "oracle_retrieval_all_records_selected": all_selected,
                "oracle_retrieval_context_changed": False,
            },
        ), tuple(oracle_records)

    def _rank(
        self,
        *,
        run_id: str,
        session_id: str,
        turn_id: str | None,
        query: str,
        mode: str,
        top_k: int,
        query_context: FactQueryContext,
        condition_context: Mapping[str, Any],
        route: SchemaRouteDecision,
        records: Sequence[FactMemoryRecord],
        started: float,
    ) -> FactMemoryRetrievalResult:
        states: dict[str, dict[str, Any]] = {}
        for record in records:
            compatible, condition_score = _condition_compatibility(
                record,
                query=query,
                query_context=query_context,
                condition_context=condition_context,
            )
            rendered = self._render_context(record)
            states[record.id] = {
                "record": record,
                "bm25_score": 0.0,
                "embedding_score": 0.0,
                "entity_score": _entity_score(record, query_context),
                "condition_score": condition_score,
                "route_score": _route_score(record, route),
                "combined_score": 0.0,
                "selected": False,
                "exclusion_reason": None if compatible else "condition_mismatch",
                "token_count": self.token_counter.count(rendered),
                "rank": None,
            }
        eligible = [
            state
            for state in states.values()
            if state["exclusion_reason"] is None
        ]
        eligible_records = [state["record"] for state in eligible]
        index_texts = [
            self._render_index(record) for record in eligible_records
        ]
        if mode in {"bm25", "hybrid"}:
            for state, score in zip(
                eligible,
                bm25(query, index_texts),
                strict=True,
            ):
                state["bm25_score"] = score
        if mode in {"embedding", "hybrid"} and eligible_records:
            scores = self._embedding_scores(
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                query=query,
                records=eligible_records,
                query_vector=(
                    route.semantic_query_vector
                    if (
                        route.semantic_query_vector
                        and route.semantic_model_id == self.embedding_model.model_id
                        and len(route.semantic_query_vector)
                        == self.embedding_model.dimensions
                    )
                    else None
                ),
            )
            for state in eligible:
                state["embedding_score"] = scores.get(
                    state["record"].id,
                    0.0,
                )
        self._combine_scores(eligible, mode)
        ranked = sorted(
            eligible,
            key=lambda state: (
                -float(state["combined_score"]),
                -float(state["record"].confidence),
                -int(state["record"].version),
                state["record"].id,
            ),
        )
        selected: list[FactMemoryRecord] = []
        used_tokens = 0
        for rank, state in enumerate(ranked, start=1):
            state["rank"] = rank
            if mode == "none":
                state["exclusion_reason"] = "mode_none"
            elif (
                mode != "full"
                and float(state["combined_score"]) <= 0
            ):
                state["exclusion_reason"] = "below_score"
            elif len(selected) >= top_k:
                state["exclusion_reason"] = "top_k"
            elif used_tokens + int(state["token_count"]) > self.token_budget:
                state["exclusion_reason"] = "token_budget"
            else:
                state["selected"] = True
                selected.append(state["record"])
                used_tokens += int(state["token_count"])

        candidates = [
            {
                key: value
                for key, value in state.items()
                if key != "record"
            }
            | {"record_id": state["record"].id}
            for state in states.values()
        ]
        latency = _elapsed_ms(started)
        self.repository.complete_fact_memory_retrieval(
            run_id,
            candidates,
            selected_count=len(selected),
            selected_tokens=used_tokens,
            latency_ms=latency,
        )
        selected_ids = {record.id for record in selected}
        selected_bundle_ids = {
            record.bundle_id for record in selected if record.bundle_id is not None
        }
        related_records = tuple(
            record
            for record in eligible_records
            if record.id not in selected_ids
            and record.bundle_id in selected_bundle_ids
        )[:12]
        return FactMemoryRetrievalResult(
            content="\n".join(self._render_context(record) for record in selected),
            records=tuple(selected),
            related_records=related_records,
            query_context=query_context,
            run_id=run_id,
            metadata={
                "fact_memory_retrieval_run_id": run_id,
                "fact_memory_retrieval_status": "completed",
                "fact_memory_retrieval_mode": mode,
                "fact_memory_query_context": asdict(query_context),
                "tool_memory_route": route.as_dict(),
                "fact_memory_retrieval_candidate_count": len(records),
                "fact_memory_retrieval_ranked_count": len(eligible),
                "fact_memory_retrieval_selected_count": len(selected),
                "fact_memory_retrieval_selected_ids": [
                    record.id for record in selected
                ],
                "fact_memory_related_record_count": len(related_records),
                "fact_memory_related_record_ids": [
                    record.id for record in related_records
                ],
                "fact_memory_retrieval_tokens": used_tokens,
                "fact_memory_retrieval_latency_ms": latency,
                "memory_first": True,
                "route_is_hard_filter": False,
            },
        )

    def _embedding_scores(
        self,
        *,
        run_id: str,
        session_id: str,
        turn_id: str | None,
        query: str,
        records: Sequence[FactMemoryRecord],
        query_vector: Sequence[float] | None,
    ) -> dict[str, float]:
        stored = self.repository.fact_memory_embeddings(
            [record.id for record in records],
            model_id=self.embedding_model.model_id,
            dimensions=self.embedding_model.dimensions,
        )
        missing = [record for record in records if record.id not in stored]
        supplied_query = tuple(query_vector) if query_vector is not None else None
        texts = [
            *([] if supplied_query is not None else [query]),
            *(self._render_index(record) for record in missing),
        ]
        response = None
        if texts:
            call_started = time.monotonic()
            try:
                response = self._invoke_with_timeout(
                    lambda: self.embedding_model.embed(texts)
                )
            except Exception as exc:
                self._record_embedding_call(
                    session_id=session_id,
                    turn_id=turn_id,
                    run_id=run_id,
                    started=call_started,
                    usage=ModelUsage(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            self._record_embedding_call(
                session_id=session_id,
                turn_id=turn_id,
                run_id=run_id,
                started=call_started,
                usage=response.usage,
                response_id=response.response_id,
                metadata={
                    **dict(response.metadata),
                    "input_count": len(texts),
                    "dimensions": self.embedding_model.dimensions,
                    "reused_route_query_vector": supplied_query is not None,
                },
            )
        if supplied_query is None:
            assert response is not None
            supplied_query = tuple(response.vectors[0])
            record_vectors = response.vectors[1:]
        else:
            record_vectors = response.vectors if response is not None else ()
        self._validate_vector(supplied_query)
        for record, vector in zip(missing, record_vectors, strict=True):
            self._validate_vector(vector)
            self.repository.store_fact_memory_embedding(
                record_id=record.id,
                model_id=self.embedding_model.model_id,
                dimensions=self.embedding_model.dimensions,
                vector=vector,
            )
            stored[record.id] = tuple(vector)
        return {
            record.id: cosine_similarity(supplied_query, stored[record.id])
            for record in records
        }

    def _record_embedding_call(
        self,
        *,
        session_id: str,
        turn_id: str | None,
        run_id: str,
        started: float,
        usage: ModelUsage,
        response_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.repository.record_model_call(
            session_id=session_id,
            turn_id=turn_id,
            role="fact_memory_embedding",
            backend=self.embedding_model.backend,
            model_id=self.embedding_model.model_id,
            prompt_version="fact-memory-retrieval-v1",
            schema_version=None,
            latency_ms=_elapsed_ms(started),
            usage=usage,
            response_id=response_id,
            metadata={
                **dict(metadata or {}),
                "fact_memory_retrieval_run_id": run_id,
                "estimated_cost_usd": (
                    usage.input_tokens
                    * self.embedding_input_cost_per_million
                    / 1_000_000
                    if self.embedding_model.backend == "openai"
                    else 0.0
                ),
            },
            error=error,
        )

    def _record_route_embedding(
        self,
        decision: SchemaRouteDecision,
        *,
        session_id: str,
        turn_id: str | None,
        run_id: str,
    ) -> None:
        if not decision.semantic_attempted:
            return
        assert decision.semantic_backend is not None
        assert decision.semantic_model_id is not None
        self.repository.record_model_call(
            session_id=session_id,
            turn_id=turn_id,
            role="fact_tool_schema_embedding",
            backend=decision.semantic_backend,
            model_id=decision.semantic_model_id,
            prompt_version="fact-tool-schema-routing-v1",
            schema_version=None,
            latency_ms=decision.semantic_latency_ms,
            usage=decision.semantic_usage,
            response_id=decision.semantic_response_id,
            metadata={
                **dict(decision.semantic_metadata),
                "fact_memory_retrieval_run_id": run_id,
                "routing_query_chars": len(decision.routing_query),
                "estimated_cost_usd": (
                    decision.semantic_usage.input_tokens
                    * self.embedding_input_cost_per_million
                    / 1_000_000
                    if decision.semantic_backend == "openai"
                    else 0.0
                ),
            },
            error=decision.semantic_error,
        )

    def _invoke_with_timeout(self, operation) -> EmbeddingResponse:
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fact-memory-embedding",
        )
        future = executor.submit(operation)
        try:
            return future.result(timeout=self.model_timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(
                "Fact memory EmbeddingModel timed out after "
                f"{self.model_timeout_seconds:g}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.embedding_model.dimensions:
            raise ValueError("Fact embedding vector dimension mismatch")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Fact embedding vector contains non-finite values")

    @staticmethod
    def _combine_scores(states: Sequence[dict[str, Any]], mode: str) -> None:
        max_bm25 = max(
            (float(state["bm25_score"]) for state in states),
            default=0.0,
        )
        max_embedding = max(
            (max(0.0, float(state["embedding_score"])) for state in states),
            default=0.0,
        )
        for state in states:
            if mode == "none":
                retrieval = 0.0
            elif mode == "full":
                retrieval = 1.0
            elif mode == "bm25":
                retrieval = (
                    float(state["bm25_score"]) / max_bm25 if max_bm25 else 0.0
                )
            elif mode == "embedding":
                retrieval = (
                    max(0.0, float(state["embedding_score"])) / max_embedding
                    if max_embedding
                    else 0.0
                )
            else:
                lexical = (
                    float(state["bm25_score"]) / max_bm25 if max_bm25 else 0.0
                )
                semantic = (
                    max(0.0, float(state["embedding_score"])) / max_embedding
                    if max_embedding
                    else 0.0
                )
                retrieval = (lexical + semantic) / 2
            state["combined_score"] = (
                retrieval * 0.55
                + float(state["entity_score"]) * 0.25
                + float(state["condition_score"]) * 0.15
                + float(state["route_score"]) * 0.05
            )

    @staticmethod
    def _render_index(record: FactMemoryRecord) -> str:
        return " ".join(
            (
                record.entity_id.replace("_", " "),
                record.predicate.replace("_", " "),
                " ".join(item.replace("_", " ") for item in record.capability_hints),
                record.memory_type,
                json.dumps(
                    record.identity_conditions,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                json.dumps(
                    record.applicability,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                json.dumps(record.value, ensure_ascii=False, sort_keys=True),
            )
        )

    @staticmethod
    def _render_context(record: FactMemoryRecord) -> str:
        return (
            f"- [fact_id={record.id} entity={record.entity_id} "
            f"predicate={record.predicate} "
            f"identity={_compact_json(record.identity_conditions)} "
            f"applicability={_compact_json(record.applicability)} "
            f"capabilities={_compact_json(record.capability_hints)} "
            f"version={record.version} confidence={record.confidence:.2f}] "
            f"value={_compact_json(record.value)}"
        )


def _oracle_key_matches(
    record: FactMemoryRecord,
    key: Mapping[str, Any],
) -> bool:
    if record.entity_id != normalize_identifier(str(key["entity_id"])):
        return False
    if record.predicate != normalize_identifier(str(key["predicate"])):
        return False
    if record.value != key["value"]:
        return False
    identity_conditions = key.get("identity_conditions")
    if (
        identity_conditions is not None
        and dict(record.identity_conditions) != identity_conditions
    ):
        return False
    applicability = key.get("applicability")
    return not (
        applicability is not None
        and dict(record.applicability) != applicability
    )


def _entity_aliases(entity: str) -> tuple[str, ...]:
    full = entity.replace("_", " ").casefold()
    parts = tuple(part for part in full.split() if len(part) > 1)
    return tuple(dict.fromkeys((full, *parts)))


def _nearest_role(
    query: str,
    entity_spans: Sequence[tuple[int, int]],
) -> str | None:
    scored: list[tuple[int, int, str]] = []
    for priority, (role, terms) in enumerate(
        (("driver", _DRIVER_TERMS), ("passenger", _PASSENGER_TERMS))
    ):
        for term in terms:
            for match in re.finditer(re.escape(term), query):
                for start, end in entity_spans:
                    distance = min(
                        abs(match.start() - end),
                        abs(start - match.end()),
                    )
                    if distance <= 42:
                        before_entity_penalty = 30 if match.end() <= start else 0
                        scored.append(
                            (distance + before_entity_penalty, priority, role)
                        )
    return min(scored)[2] if scored else None


def _time_of_day(query: str, context: Mapping[str, Any]) -> str | None:
    explicit = _first_context_value(context, ("time_of_day", "time", "period"))
    if explicit is not None:
        return _canonical_time(str(explicit))
    clock = _CLOCK.search(query)
    if clock is not None:
        hour = int(clock.group(1)) % 12
        if clock.group(3).casefold() == "p":
            hour += 12
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"
    for canonical, terms in {
        "morning": ("morning", "dawn", "아침", "새벽"),
        "afternoon": ("afternoon", "noon", "낮", "오후"),
        "evening": ("evening", "dusk", "sunset", "저녁", "해질"),
        "night": ("night", "late", "밤", "야간"),
    }.items():
        if any(term in query for term in terms):
            return canonical
    return None


def _canonical_time(value: str) -> str:
    folded = value.casefold().replace("_", " ")
    if any(term in folded for term in ("dusk", "sunset", "evening", "저녁")):
        return "evening"
    if any(term in folded for term in ("night", "late", "밤")):
        return "night"
    if any(term in folded for term in ("morning", "dawn", "아침", "새벽")):
        return "morning"
    if any(term in folded for term in ("afternoon", "noon", "오후", "낮")):
        return "afternoon"
    return normalize_identifier(value)


def _weather(query: str, context: Mapping[str, Any]) -> str | None:
    explicit = _first_context_value(context, ("weather", "climate"))
    if explicit is not None:
        return _canonical_weather(str(explicit))
    for canonical, aliases in _WEATHER_ALIASES.items():
        if any(alias in query for alias in aliases):
            return canonical
    return None


def _canonical_weather(value: str) -> str:
    folded = value.casefold()
    for canonical, aliases in _WEATHER_ALIASES.items():
        if any(alias in folded for alias in aliases):
            return canonical
    return normalize_identifier(value)


def _entity_score(
    record: FactMemoryRecord,
    context: FactQueryContext,
) -> float:
    if not context.entities:
        return 0.0
    if record.entity_id not in context.entities:
        return 0.0
    if record.entity_id == context.requester_entity:
        return 1.0
    if context.entity_roles.get(record.entity_id) == "driver":
        return 0.9
    if context.entity_roles.get(record.entity_id) == "passenger":
        return 0.8
    return 0.7


def _condition_compatibility(
    record: FactMemoryRecord,
    *,
    query: str,
    query_context: FactQueryContext,
    condition_context: Mapping[str, Any],
) -> tuple[bool, float]:
    conditions = {**record.identity_conditions, **record.applicability}
    if not conditions:
        return True, 1.0
    known = {
        normalize_identifier(str(key)): value
        for key, value in condition_context.items()
    }
    if query_context.time_of_day:
        known["time_of_day"] = query_context.time_of_day
        known["time"] = query_context.time_of_day
    if query_context.weather:
        known["weather"] = query_context.weather
    matches = 0
    unknown = 0
    for key, expected in conditions.items():
        normalized_key = normalize_identifier(str(key))
        if normalized_key in _RUNTIME_KEYS:
            continue
        actual = known.get(normalized_key)
        if actual is None:
            if _value_mentions(expected, query):
                matches += 1
            else:
                unknown += 1
            continue
        expected_value = _canonical_condition(normalized_key, expected)
        actual_value = _canonical_condition(normalized_key, actual)
        if expected_value != actual_value:
            return False, 0.0
        matches += 1
    denominator = matches + unknown
    return True, (matches + unknown * 0.5) / denominator if denominator else 1.0


def _canonical_condition(key: str, value: Any) -> Any:
    if key in {"time", "time_of_day", "period"}:
        return _canonical_time(str(value))
    if key in {"weather", "climate"}:
        return _canonical_weather(str(value))
    return normalize_identifier(str(value))


def _route_score(
    record: FactMemoryRecord,
    decision: SchemaRouteDecision,
) -> float:
    record_tokens = set(_tokens(" ".join(
        (record.predicate, *record.capability_hints)
    )))
    best = 0.0
    for route in decision.routes:
        route_tokens = set(_tokens(" ".join((route.domain, *route.topics))))
        overlap = len(record_tokens & route_tokens)
        if overlap:
            best = max(best, min(1.0, overlap / max(1, len(record_tokens))))
    return best


def _value_mentions(value: Any, query: str) -> bool:
    if isinstance(value, Mapping):
        return all(_value_mentions(item, query) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_value_mentions(item, query) for item in value)
    rendered = str(value).casefold().replace("_", " ")
    return rendered in query.casefold()


def _first_context_value(
    context: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        if key in context and context[key] is not None and context[key] != "":
            return context[key]
    return None


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(value.casefold().replace("_", " ")))


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
