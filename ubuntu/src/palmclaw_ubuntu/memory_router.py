from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Protocol

from palmclaw_ubuntu.contracts import EmbeddingModel
from palmclaw_ubuntu.memory import bm25, cosine_similarity
from palmclaw_ubuntu.models import (
    EmbeddingResponse,
    ModelUsage,
    ToolMemoryRecord,
    ToolMemoryRetrievalResult,
)
from palmclaw_ubuntu.storage import SQLiteRepository
from palmclaw_ubuntu.tokens import TokenCounter
from palmclaw_ubuntu.tool_memory_schema import (
    ToolMemoryOntology,
    ToolMemoryTool,
    normalize_identifier,
)

_TOKEN_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)
_CLOCK_PATTERN = re.compile(
    r"\b(1[0-2]|0?\d)(?::([0-5]\d))?\s*([ap])\.?m\.?\b",
    re.IGNORECASE,
)
_ENTITY_CONDITION_KEYS = frozenset(
    {"driver", "occupant", "passenger", "person", "subject"}
)
_SOFT_CONTEXT_KEYS = frozenset(
    {"context", "location", "place", "situation", "use_case"}
)
_WEATHER_TERMS = {
    "hot": frozenset({"hot", "heat", "warm", "더운", "더위"}),
    "cold": frozenset({"cold", "cool", "freezing", "icy", "snow", "추운", "눈"}),
    "rain": frozenset({"rain", "rainy", "storm", "wet", "비", "폭우"}),
}
_GENERIC_ALIASES: dict[str, tuple[str, ...]] = {
    "ambient": ("atmosphere", "ambience", "cabin"),
    "air": ("climate", "ac", "aircon", "에어컨", "공조"),
    "brightness": ("bright", "dim", "glare", "visibility"),
    "close": ("shut", "닫아", "닫기"),
    "color": ("colour",),
    "circulation": ("recirculate", "smog", "air_quality"),
    "file": ("document", "report", "파일", "문서", "보고서"),
    "get": ("show", "check", "inspect", "보여", "확인"),
    "headrest": ("head_support",),
    "hud": ("projection", "head_up_display"),
    "light": ("lamp", "조명", "등"),
    "lock": ("잠가", "잠금"),
    "navigate": ("route", "directions", "길안내", "경로"),
    "navigation": ("route", "destination", "내비", "경로", "목적지"),
    "open": ("열어", "열기"),
    "read": ("show", "inspect", "읽어", "보여"),
    "reading": ("book", "document"),
    "search": ("find", "lookup", "검색", "찾아"),
    "seat": ("chair", "시트", "좌석"),
    "set": ("adjust", "change", "설정", "조정", "변경"),
    "temperature": ("temp", "온도"),
    "unlock": ("열어", "잠금해제"),
    "ventilation": ("ventilate", "airflow"),
    "volume": ("sound", "audio", "볼륨", "음량"),
    "voice": ("guidance", "spoken"),
    "web": ("website", "url", "internet", "웹", "인터넷"),
    "write": ("save", "store", "create", "작성", "저장"),
}
_ROUTING_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)
_WEAK_SLOT_TERMS = frozenset(
    {
        "enabled",
        "is_on",
        "level",
        "mode",
        "position",
        "power",
        "seat",
        "switch",
        "time",
        "zone",
    }
)
_WEAK_INTENT_TERMS = _WEAK_SLOT_TERMS | frozenset(
    {"adjust", "change", "set"}
)


@dataclass(frozen=True)
class SchemaRoute:
    domain: str
    topics: tuple[str, ...]
    tool_names: tuple[str, ...]
    score: float
    matched_terms: tuple[str, ...]
    lexical_score: float = 0.0
    semantic_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "topics": list(self.topics),
            "tool_names": list(self.tool_names),
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "lexical_score": self.lexical_score,
            "semantic_score": self.semantic_score,
        }


@dataclass(frozen=True)
class SchemaRouteDecision:
    routes: tuple[SchemaRoute, ...]
    ambiguous: bool
    classifier_used: bool
    reason: str
    routing_query: str = ""
    semantic_attempted: bool = False
    semantic_used: bool = False
    semantic_backend: str | None = None
    semantic_model_id: str | None = None
    semantic_latency_ms: int = 0
    semantic_usage: ModelUsage = field(default_factory=ModelUsage)
    semantic_response_id: str | None = None
    semantic_metadata: Mapping[str, Any] = field(default_factory=dict)
    semantic_error: str | None = None
    semantic_query_vector: tuple[float, ...] = field(
        default=(),
        repr=False,
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "routes": [route.as_dict() for route in self.routes],
            "ambiguous": self.ambiguous,
            "classifier_used": self.classifier_used,
            "reason": self.reason,
            "routing_query": self.routing_query,
            "semantic": {
                "attempted": self.semantic_attempted,
                "used": self.semantic_used,
                "backend": self.semantic_backend,
                "model_id": self.semantic_model_id,
                "latency_ms": self.semantic_latency_ms,
                "error": self.semantic_error,
            },
        }


@dataclass(frozen=True)
class _SemanticRouteResult:
    scores: Mapping[str, float]
    query_vector: tuple[float, ...]
    backend: str
    model_id: str
    latency_ms: int
    usage: ModelUsage
    response_id: str | None
    metadata: Mapping[str, Any]
    input_count: int
    cached_tool_count: int


class ToolMemoryRouteClassifier(Protocol):
    def classify(
        self,
        query: str,
        ontology: ToolMemoryOntology,
    ) -> Sequence[SchemaRoute]: ...


class ToolSchemaRouter:
    def __init__(
        self,
        ontology: ToolMemoryOntology,
        *,
        max_routes: int = 3,
        classifier: ToolMemoryRouteClassifier | None = None,
        embedding_model: EmbeddingModel | None = None,
        model_timeout_seconds: float = 60.0,
        semantic_weight: float = 0.65,
        semantic_min_score: float = 0.2,
        max_tools_per_domain: int = 2,
        max_total_tools: int = 4,
        repository: SQLiteRepository | None = None,
    ):
        if max_routes < 1:
            raise ValueError("max_routes must be positive")
        if model_timeout_seconds <= 0:
            raise ValueError("Router model timeout must be positive")
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        if not -1 <= semantic_min_score <= 1:
            raise ValueError("semantic_min_score must be between -1 and 1")
        if max_tools_per_domain < 1:
            raise ValueError("max_tools_per_domain must be positive")
        if max_total_tools < 1:
            raise ValueError("max_total_tools must be positive")
        self.ontology = ontology
        self.max_routes = max_routes
        self.classifier = classifier
        self.embedding_model = embedding_model
        self.model_timeout_seconds = model_timeout_seconds
        self.semantic_weight = semantic_weight
        self.semantic_min_score = semantic_min_score
        self.max_tools_per_domain = max_tools_per_domain
        self.max_total_tools = max_total_tools
        self.repository = repository
        self._tool_vectors: dict[str, tuple[float, ...]] = {}

    def route(self, query: str) -> SchemaRouteDecision:
        routing_query = _routing_focus(query)
        lexical = self._lexical_tool_scores(routing_query)
        semantic: _SemanticRouteResult | None = None
        semantic_error = None
        if self.embedding_model is not None:
            try:
                semantic = self._semantic_tool_scores(routing_query)
            except Exception as exc:
                semantic_error = f"{type(exc).__name__}: {exc}"
        routes = self._combined_routes(
            lexical,
            semantic.scores if semantic is not None else {},
        )
        ambiguous = self._is_ambiguous(routes)
        if self.classifier is not None and (not routes or ambiguous):
            classified = self._validated_classifier_routes(
                self.classifier.classify(query, self.ontology)
            )
            if classified:
                return self._decision(
                    routes=classified[: self.max_routes],
                    ambiguous=len(classified) > 1,
                    classifier_used=True,
                    reason="classifier_fallback",
                    routing_query=routing_query,
                    semantic=semantic,
                    semantic_error=semantic_error,
                )
        if not routes:
            return self._decision(
                routes=(),
                ambiguous=False,
                classifier_used=False,
                reason="no_schema_match",
                routing_query=routing_query,
                semantic=semantic,
                semantic_error=semantic_error,
            )
        reason = (
            "hybrid_schema_match"
            if semantic is not None and lexical
            else (
                "semantic_schema_match"
                if semantic is not None
                else (
                    "lexical_after_semantic_error"
                    if semantic_error is not None
                    else "lexical_schema_match"
                )
            )
        )
        return self._decision(
            routes=routes[: self.max_routes],
            ambiguous=ambiguous,
            classifier_used=False,
            reason=reason,
            routing_query=routing_query,
            semantic=semantic,
            semantic_error=semantic_error,
        )

    def route_tool_names(
        self,
        tool_names: Sequence[str],
    ) -> SchemaRouteDecision:
        """Route exact Tool names for a diagnostic without exposing arguments."""
        requested = tuple(dict.fromkeys(str(name) for name in tool_names))
        if not requested:
            raise ValueError("Oracle Tool routing requires at least one Tool name")
        tools_by_folded_name = {
            tool.tool_name.casefold(): tool for tool in self.ontology.tools
        }
        unknown = sorted(
            name
            for name in requested
            if name.casefold() not in tools_by_folded_name
        )
        if unknown:
            raise ValueError(f"Unknown Oracle Tool names: {unknown}")
        grouped: dict[str, dict[str, set[str]]] = {}
        for name in requested:
            tool = tools_by_folded_name[name.casefold()]
            state = grouped.setdefault(
                tool.domain,
                {"topics": set(), "tool_names": set()},
            )
            state["topics"].add(tool.topic)
            state["topics"].update(slot.name for slot in tool.slots)
            state["tool_names"].add(tool.tool_name)
        routes = tuple(
            SchemaRoute(
                domain=domain,
                topics=tuple(sorted(state["topics"])),
                tool_names=tuple(sorted(state["tool_names"])),
                score=1.0,
                matched_terms=(),
                lexical_score=1.0,
            )
            for domain, state in sorted(grouped.items())
        )
        return SchemaRouteDecision(
            routes=routes,
            ambiguous=False,
            classifier_used=False,
            reason="oracle_tool_names",
        )

    def _lexical_tool_scores(
        self,
        query: str,
    ) -> dict[str, tuple[float, tuple[str, ...]]]:
        query_terms = set(_tokens(query)) - _ROUTING_STOP_TERMS
        scores: dict[str, tuple[float, tuple[str, ...]]] = {}
        for tool in self.ontology.tools:
            weighted_fields = (
                (tool.domain, 6.0, "domain"),
                (tool.topic, 7.0, "topic"),
                (tool.action, 2.0, "action"),
                *((slot.name, 5.0, "slot") for slot in tool.slots),
                *((term, 1.5, "description") for term in _tokens(tool.description)),
                *((term, 1.0, "tool") for term in _tokens(tool.tool_name)),
            )
            score = 0.0
            matched: set[str] = set()
            for raw_term, weight, field_name in weighted_fields:
                for term in _schema_terms(raw_term):
                    if term in _ROUTING_STOP_TERMS:
                        continue
                    if field_name == "slot" and term in _WEAK_SLOT_TERMS:
                        continue
                    if term not in query_terms:
                        continue
                    score += weight
                    matched.add(term)
            if score > 0:
                scores[tool.tool_name] = (
                    score,
                    tuple(sorted(matched)),
                )
        return scores

    def _semantic_tool_scores(self, query: str) -> _SemanticRouteResult:
        assert self.embedding_model is not None
        fingerprints = {
            tool.tool_name: _tool_route_fingerprint(tool)
            for tool in self.ontology.tools
        }
        if (
            self.repository is not None
            and len(self._tool_vectors) < len(self.ontology.tools)
        ):
            self._tool_vectors.update(
                self.repository.tool_schema_embeddings(
                    fingerprints,
                    model_id=self.embedding_model.model_id,
                    dimensions=self.embedding_model.dimensions,
                )
            )
        missing = [
            tool
            for tool in self.ontology.tools
            if tool.tool_name not in self._tool_vectors
        ]
        cached_count = len(self.ontology.tools) - len(missing)
        texts = [query, *(_tool_route_document(tool) for tool in missing)]
        started = time.monotonic()
        response = self._invoke_embedding(texts)
        latency = _elapsed_ms(started)
        if len(response.vectors) != len(texts):
            raise RuntimeError(
                "Routing EmbeddingModel returned an unexpected vector count"
            )
        query_vector = tuple(response.vectors[0])
        self._validate_route_vector(query_vector)
        for tool, vector in zip(
            missing,
            response.vectors[1:],
            strict=True,
        ):
            normalized = tuple(vector)
            self._validate_route_vector(normalized)
            self._tool_vectors[tool.tool_name] = normalized
        if self.repository is not None and missing:
            self.repository.store_tool_schema_embeddings(
                {
                    tool.tool_name: (
                        fingerprints[tool.tool_name],
                        self._tool_vectors[tool.tool_name],
                    )
                    for tool in missing
                },
                model_id=self.embedding_model.model_id,
                dimensions=self.embedding_model.dimensions,
            )
        scores = {
            tool.tool_name: cosine_similarity(
                query_vector,
                self._tool_vectors[tool.tool_name],
            )
            for tool in self.ontology.tools
        }
        return _SemanticRouteResult(
            scores=scores,
            query_vector=query_vector,
            backend=self.embedding_model.backend,
            model_id=self.embedding_model.model_id,
            latency_ms=latency,
            usage=response.usage,
            response_id=response.response_id,
            metadata=response.metadata,
            input_count=len(texts),
            cached_tool_count=cached_count,
        )

    def _combined_routes(
        self,
        lexical: Mapping[str, tuple[float, tuple[str, ...]]],
        semantic: Mapping[str, float],
    ) -> tuple[SchemaRoute, ...]:
        max_lexical = max(
            (item[0] for item in lexical.values()),
            default=0.0,
        )
        positive_semantic = {
            name: max(0.0, float(score))
            for name, score in semantic.items()
        }
        max_semantic = max(positive_semantic.values(), default=0.0)
        candidates = []
        for tool in self.ontology.tools:
            lexical_score, matched = lexical.get(
                tool.tool_name,
                (0.0, ()),
            )
            semantic_score = float(semantic.get(tool.tool_name, 0.0))
            if (
                lexical_score <= 0
                and semantic_score < self.semantic_min_score
            ):
                continue
            lexical_normalized = (
                lexical_score / max_lexical if max_lexical else 0.0
            )
            semantic_normalized = (
                max(0.0, semantic_score) / max_semantic
                if max_semantic
                else 0.0
            )
            combined = (
                (
                    (1 - self.semantic_weight) * lexical_normalized
                    + self.semantic_weight * semantic_normalized
                )
                if semantic
                else lexical_score
            )
            candidates.append(
                {
                    "tool": tool,
                    "score": combined,
                    "lexical_score": lexical_score,
                    "semantic_score": semantic_score,
                    "matched_terms": matched,
                }
            )
        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["tool"].tool_name),
            )
        )
        if not candidates:
            return ()
        if not semantic and any(
            set(candidate["matched_terms"]) - _WEAK_INTENT_TERMS
            for candidate in candidates
        ):
            candidates = [
                candidate
                for candidate in candidates
                if set(candidate["matched_terms"]) - _WEAK_INTENT_TERMS
            ]
        relative_floor = float(candidates[0]["score"]) * 0.72
        domain_counts: dict[str, int] = {}
        selected = []
        for candidate in candidates:
            tool = candidate["tool"]
            assert isinstance(tool, ToolMemoryTool)
            if float(candidate["score"]) < relative_floor:
                continue
            if domain_counts.get(tool.domain, 0) >= self.max_tools_per_domain:
                continue
            selected.append(candidate)
            domain_counts[tool.domain] = domain_counts.get(tool.domain, 0) + 1
            if len(selected) >= self.max_total_tools:
                break

        grouped: dict[str, dict[str, Any]] = {}
        for candidate in selected:
            tool = candidate["tool"]
            assert isinstance(tool, ToolMemoryTool)
            state = grouped.setdefault(
                tool.domain,
                {
                    "score": 0.0,
                    "lexical_score": 0.0,
                    "semantic_score": 0.0,
                    "topics": set(),
                    "tool_names": set(),
                    "matched_terms": set(),
                },
            )
            state["score"] = max(
                float(state["score"]),
                float(candidate["score"]),
            )
            state["lexical_score"] = max(
                float(state["lexical_score"]),
                float(candidate["lexical_score"]),
            )
            state["semantic_score"] = max(
                float(state["semantic_score"]),
                float(candidate["semantic_score"]),
            )
            state["topics"].add(tool.topic)
            state["topics"].update(slot.name for slot in tool.slots)
            state["tool_names"].add(tool.tool_name)
            state["matched_terms"].update(candidate["matched_terms"])
        return tuple(
            sorted(
                (
                    SchemaRoute(
                        domain=domain,
                        topics=tuple(sorted(state["topics"])),
                        tool_names=tuple(sorted(state["tool_names"])),
                        score=float(state["score"]),
                        matched_terms=tuple(sorted(state["matched_terms"])),
                        lexical_score=float(state["lexical_score"]),
                        semantic_score=float(state["semantic_score"]),
                    )
                    for domain, state in grouped.items()
                ),
                key=lambda route: (-route.score, route.domain),
            )
        )

    def _decision(
        self,
        *,
        routes: tuple[SchemaRoute, ...],
        ambiguous: bool,
        classifier_used: bool,
        reason: str,
        routing_query: str,
        semantic: _SemanticRouteResult | None,
        semantic_error: str | None,
    ) -> SchemaRouteDecision:
        return SchemaRouteDecision(
            routes=routes,
            ambiguous=ambiguous,
            classifier_used=classifier_used,
            reason=reason,
            routing_query=routing_query,
            semantic_attempted=self.embedding_model is not None,
            semantic_used=semantic is not None,
            semantic_backend=(
                semantic.backend
                if semantic is not None
                else (
                    self.embedding_model.backend
                    if self.embedding_model is not None
                    else None
                )
            ),
            semantic_model_id=(
                semantic.model_id
                if semantic is not None
                else (
                    self.embedding_model.model_id
                    if self.embedding_model is not None
                    else None
                )
            ),
            semantic_latency_ms=(
                semantic.latency_ms if semantic is not None else 0
            ),
            semantic_usage=(
                semantic.usage if semantic is not None else ModelUsage()
            ),
            semantic_response_id=(
                semantic.response_id if semantic is not None else None
            ),
            semantic_metadata=(
                {
                    **dict(semantic.metadata),
                    "input_count": semantic.input_count,
                    "cached_tool_count": semantic.cached_tool_count,
                }
                if semantic is not None
                else {}
            ),
            semantic_error=semantic_error,
            semantic_query_vector=(
                semantic.query_vector if semantic is not None else ()
            ),
        )

    def _invoke_embedding(
        self,
        texts: Sequence[str],
    ) -> EmbeddingResponse:
        assert self.embedding_model is not None
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tool-schema-embedding",
        )
        future = executor.submit(self.embedding_model.embed, texts)
        try:
            return future.result(timeout=self.model_timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(
                "Tool schema EmbeddingModel timed out after "
                f"{self.model_timeout_seconds:g}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _validate_route_vector(self, vector: Sequence[float]) -> None:
        assert self.embedding_model is not None
        if len(vector) != self.embedding_model.dimensions:
            raise ValueError(
                "Tool schema embedding dimension does not match configuration"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Tool schema embedding contains a non-finite value")

    def _validated_classifier_routes(
        self,
        routes: Sequence[SchemaRoute],
    ) -> tuple[SchemaRoute, ...]:
        known_domains = set(self.ontology.domains)
        validated = []
        for route in routes:
            domain = normalize_identifier(route.domain)
            if domain not in known_domains:
                continue
            known_topics = {
                topic
                for tool in self.ontology.tools
                if tool.domain == domain
                for topic in (tool.topic, *(slot.name for slot in tool.slots))
            }
            topics = tuple(
                topic
                for topic in (
                    normalize_identifier(item) for item in route.topics
                )
                if topic in known_topics
            )
            validated.append(
                SchemaRoute(
                    domain=domain,
                    topics=tuple(dict.fromkeys(topics)),
                    tool_names=tuple(route.tool_names),
                    score=route.score,
                    matched_terms=tuple(route.matched_terms),
                    lexical_score=route.lexical_score,
                    semantic_score=route.semantic_score,
                )
            )
        return tuple(validated)

    @staticmethod
    def _is_ambiguous(routes: Sequence[SchemaRoute]) -> bool:
        return len(routes) > 1 and routes[1].score >= routes[0].score * 0.8


class ToolMemoryRetriever:
    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        router: ToolSchemaRouter,
        embedding_model: EmbeddingModel,
        user_id: str,
        vehicle_id: str | None,
        mode: str,
        top_k: int,
        token_budget: int,
        model_timeout_seconds: float,
        embedding_input_cost_per_million: float = 0.0,
        token_counter: TokenCounter | None = None,
    ):
        if mode not in {"none", "full", "bm25", "embedding", "hybrid"}:
            raise ValueError(f"Unsupported Tool memory retrieval mode: {mode}")
        if top_k < 1:
            raise ValueError("Tool memory top_k must be positive")
        if token_budget < 0:
            raise ValueError("Tool memory token budget cannot be negative")
        self.repository = repository
        self.router = router
        self.embedding_model = embedding_model
        self.user_id = normalize_identifier(user_id)
        self.vehicle_id = (
            normalize_identifier(vehicle_id) if vehicle_id else None
        )
        self.mode = mode
        self.top_k = top_k
        self.token_budget = token_budget
        self.model_timeout_seconds = model_timeout_seconds
        self.embedding_input_cost_per_million = (
            embedding_input_cost_per_million
        )
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
    ) -> ToolMemoryRetrievalResult:
        started = time.monotonic()
        selected_mode = mode or self.mode
        selected_top_k = self.top_k if top_k is None else top_k
        if selected_mode not in {"none", "full", "bm25", "embedding", "hybrid"}:
            raise ValueError(
                f"Unsupported Tool memory retrieval mode: {selected_mode}"
            )
        if selected_top_k < 1:
            raise ValueError("Tool memory top_k must be positive")
        decision = (
            self.router.route_tool_names(oracle_tool_names)
            if oracle_tool_names
            else self.router.route(query)
        )
        run_id = self.repository.begin_tool_memory_retrieval(
            session_id=session_id,
            turn_id=turn_id,
            query=query,
            user_id=self.user_id,
            mode=selected_mode,
            top_k=selected_top_k,
            token_budget=self.token_budget,
            route=decision.as_dict(),
            classifier_used=decision.classifier_used,
        )
        try:
            self._record_route_embedding(
                decision,
                session_id=session_id,
                turn_id=turn_id,
                run_id=run_id,
            )
            if not decision.routes:
                return self._complete_empty(
                    run_id,
                    started=started,
                    mode=selected_mode,
                    decision=decision,
                    reason="no_route",
                )
            records = self.repository.routed_tool_memory_records(
                user_id=self.user_id,
                routes=tuple(
                    (route.domain, route.topics) for route in decision.routes
                ),
            )
            return self._rank(
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                query=query,
                mode=selected_mode,
                top_k=selected_top_k,
                decision=decision,
                records=records,
                condition_context=condition_context or {},
                started=started,
            )
        except Exception as exc:
            latency = _elapsed_ms(started)
            self.repository.fail_tool_memory_retrieval(
                run_id,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=latency,
            )
            return ToolMemoryRetrievalResult(
                content="",
                run_id=run_id,
                metadata={
                    "tool_memory_retrieval_run_id": run_id,
                    "tool_memory_retrieval_status": "failed",
                    "tool_memory_retrieval_mode": selected_mode,
                    "tool_memory_retrieval_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "tool_memory_retrieval_selected_count": 0,
                    "tool_memory_retrieval_latency_ms": latency,
                },
            )

    def _rank(
        self,
        *,
        run_id: str,
        session_id: str,
        turn_id: str | None,
        query: str,
        mode: str,
        top_k: int,
        decision: SchemaRouteDecision,
        records: Sequence[ToolMemoryRecord],
        condition_context: Mapping[str, Any],
        started: float,
    ) -> ToolMemoryRetrievalResult:
        eligible = []
        trace_by_id: dict[str, dict[str, Any]] = {}
        for record in records:
            exclusion = self._partition_exclusion(
                record,
                session_id=session_id,
                query=query,
                condition_context=condition_context,
            )
            trace_by_id[record.id] = {
                "record_id": record.id,
                "route_domain": record.tool_domain,
                "route_topic": record.topic,
                "rank": None,
                "bm25_score": 0.0,
                "embedding_score": 0.0,
                "combined_score": 0.0,
                "selected": False,
                "exclusion_reason": exclusion,
                "token_count": self.token_counter.count(
                    self._render_context(record)
                ),
            }
            if exclusion is None:
                eligible.append(record)

        states = {
            record.id: {
                "record": record,
                "bm25_score": 0.0,
                "embedding_score": 0.0,
                "combined_score": 0.0,
            }
            for record in eligible
        }
        index_texts = [self._render_index(record) for record in eligible]
        if mode in {"bm25", "hybrid"}:
            for record, score in zip(
                eligible,
                bm25(query, index_texts),
                strict=True,
            ):
                states[record.id]["bm25_score"] = score
        if mode in {"embedding", "hybrid"} and eligible:
            scores = self._embedding_scores(
                run_id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                query=query,
                records=eligible,
                query_vector=(
                    decision.semantic_query_vector
                    if (
                        decision.semantic_query_vector
                        and decision.semantic_model_id
                        == self.embedding_model.model_id
                        and len(decision.semantic_query_vector)
                        == self.embedding_model.dimensions
                    )
                    else None
                ),
            )
            for record in eligible:
                states[record.id]["embedding_score"] = scores.get(record.id, 0.0)
        self._combine_scores(states, mode)
        ranked = sorted(
            states.values(),
            key=lambda state: (
                -float(state["combined_score"]),
                -float(state["record"].confidence),
                -int(state["record"].version),
                state["record"].id,
            ),
        )
        selected: list[ToolMemoryRecord] = []
        used_tokens = 0
        for rank, state in enumerate(ranked, start=1):
            record = state["record"]
            trace = trace_by_id[record.id]
            trace.update(
                {
                    "rank": rank,
                    "bm25_score": state["bm25_score"],
                    "embedding_score": state["embedding_score"],
                    "combined_score": state["combined_score"],
                }
            )
            exclusion = None
            if mode == "none":
                exclusion = "mode_none"
            elif mode != "full" and float(state["combined_score"]) <= 0:
                exclusion = "below_score"
            elif len(selected) >= top_k:
                exclusion = "top_k"
            elif used_tokens + int(trace["token_count"]) > self.token_budget:
                exclusion = "token_budget"
            else:
                selected.append(record)
                used_tokens += int(trace["token_count"])
                trace["selected"] = True
            trace["exclusion_reason"] = exclusion
        latency = _elapsed_ms(started)
        candidates = list(trace_by_id.values())
        self.repository.complete_tool_memory_retrieval(
            run_id,
            candidates,
            selected_count=len(selected),
            selected_tokens=used_tokens,
            latency_ms=latency,
        )
        return ToolMemoryRetrievalResult(
            content="\n".join(self._render_context(record) for record in selected),
            records=tuple(selected),
            run_id=run_id,
            metadata={
                "tool_memory_retrieval_run_id": run_id,
                "tool_memory_retrieval_status": "completed",
                "tool_memory_retrieval_mode": mode,
                "tool_memory_route": decision.as_dict(),
                "tool_memory_retrieval_candidate_count": len(records),
                "tool_memory_retrieval_ranked_count": len(eligible),
                "tool_memory_retrieval_selected_count": len(selected),
                "tool_memory_retrieval_selected_ids": [
                    record.id for record in selected
                ],
                "tool_memory_retrieval_tokens": used_tokens,
                "tool_memory_retrieval_latency_ms": latency,
            },
        )

    def _complete_empty(
        self,
        run_id: str,
        *,
        started: float,
        mode: str,
        decision: SchemaRouteDecision,
        reason: str,
    ) -> ToolMemoryRetrievalResult:
        latency = _elapsed_ms(started)
        self.repository.complete_tool_memory_retrieval(
            run_id,
            (),
            selected_count=0,
            selected_tokens=0,
            latency_ms=latency,
        )
        return ToolMemoryRetrievalResult(
            content="",
            run_id=run_id,
            metadata={
                "tool_memory_retrieval_run_id": run_id,
                "tool_memory_retrieval_status": "completed",
                "tool_memory_retrieval_mode": mode,
                "tool_memory_route": decision.as_dict(),
                "tool_memory_retrieval_empty_reason": reason,
                "tool_memory_retrieval_candidate_count": 0,
                "tool_memory_retrieval_selected_count": 0,
                "tool_memory_retrieval_tokens": 0,
                "tool_memory_retrieval_latency_ms": latency,
            },
        )

    def _partition_exclusion(
        self,
        record: ToolMemoryRecord,
        *,
        session_id: str,
        query: str,
        condition_context: Mapping[str, Any],
    ) -> str | None:
        normalized_session = normalize_identifier(session_id)
        if record.scope == "global" and record.scope_key != self.user_id:
            return "scope_mismatch"
        if record.scope == "session" and record.scope_key != normalized_session:
            return "scope_mismatch"
        if record.scope == "vehicle" and (
            self.vehicle_id is None or record.scope_key != self.vehicle_id
        ):
            return "scope_mismatch"
        if record.scope == "conditional" and record.scope_key not in {
            self.user_id,
            normalized_session,
            self.vehicle_id,
        }:
            return "scope_mismatch"
        if record.conditions and not _conditions_match(
            record.conditions,
            query,
            condition_context,
        ):
            return "condition_mismatch"
        return None

    def _embedding_scores(
        self,
        *,
        run_id: str,
        session_id: str,
        turn_id: str | None,
        query: str,
        records: Sequence[ToolMemoryRecord],
        query_vector: Sequence[float] | None = None,
    ) -> dict[str, float]:
        record_ids = [record.id for record in records]
        stored = self.repository.tool_memory_embeddings(
            record_ids,
            model_id=self.embedding_model.model_id,
            dimensions=self.embedding_model.dimensions,
        )
        missing = [record for record in records if record.id not in stored]
        supplied_query_vector = (
            tuple(query_vector) if query_vector is not None else None
        )
        texts = [
            *([] if supplied_query_vector is not None else [query]),
            *(self._render_index(record) for record in missing),
        ]
        response = None
        if texts:
            started = time.monotonic()
            try:
                response = self._invoke_with_timeout(
                    lambda: self.embedding_model.embed(texts)
                )
            except Exception as exc:
                self._record_embedding_call(
                    session_id=session_id,
                    turn_id=turn_id,
                    run_id=run_id,
                    started=started,
                    usage=ModelUsage(),
                    response_id=None,
                    metadata={"input_count": len(texts)},
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            self._record_embedding_call(
                session_id=session_id,
                turn_id=turn_id,
                run_id=run_id,
                started=started,
                usage=response.usage,
                response_id=response.response_id,
                metadata={
                    **dict(response.metadata),
                    "input_count": len(texts),
                    "dimensions": self.embedding_model.dimensions,
                    "reused_route_query_vector": (
                        supplied_query_vector is not None
                    ),
                },
            )
            if len(response.vectors) != len(texts):
                raise RuntimeError(
                    "EmbeddingModel returned an unexpected vector count"
                )
        if supplied_query_vector is None:
            assert response is not None
            supplied_query_vector = tuple(response.vectors[0])
            record_vectors = response.vectors[1:]
        else:
            record_vectors = response.vectors if response is not None else ()
        self._validate_vector(supplied_query_vector)
        for record, vector in zip(
            missing,
            record_vectors,
            strict=True,
        ):
            self._validate_vector(vector)
            self.repository.store_tool_memory_embedding(
                record_id=record.id,
                model_id=self.embedding_model.model_id,
                dimensions=self.embedding_model.dimensions,
                vector=vector,
            )
            stored[record.id] = vector
        return {
            record.id: cosine_similarity(
                supplied_query_vector,
                stored[record.id],
            )
            for record in records
        }

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
        metadata = {
            **dict(decision.semantic_metadata),
            "tool_memory_retrieval_run_id": run_id,
            "routing_query_chars": len(decision.routing_query),
            "dimensions": self.embedding_model.dimensions,
            "estimated_cost_usd": (
                decision.semantic_usage.input_tokens
                * self.embedding_input_cost_per_million
                / 1_000_000
                if decision.semantic_backend == "openai"
                else 0.0
            ),
        }
        self.repository.record_model_call(
            session_id=session_id,
            turn_id=turn_id,
            role="tool_schema_embedding",
            backend=decision.semantic_backend,
            model_id=decision.semantic_model_id,
            prompt_version="tool-schema-routing-v2",
            schema_version=None,
            latency_ms=decision.semantic_latency_ms,
            usage=decision.semantic_usage,
            response_id=decision.semantic_response_id,
            metadata=metadata,
            error=decision.semantic_error,
        )

    def _record_embedding_call(
        self,
        *,
        session_id: str,
        turn_id: str | None,
        run_id: str,
        started: float,
        usage: ModelUsage,
        response_id: str | None,
        metadata: dict[str, Any],
        error: str | None = None,
    ) -> None:
        self.repository.record_model_call(
            session_id=session_id,
            turn_id=turn_id,
            role="tool_memory_embedding",
            backend=self.embedding_model.backend,
            model_id=self.embedding_model.model_id,
            prompt_version="tool-memory-retrieval-v1",
            schema_version=None,
            latency_ms=_elapsed_ms(started),
            usage=usage,
            response_id=response_id,
            metadata={
                **metadata,
                "tool_memory_retrieval_run_id": run_id,
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

    def _invoke_with_timeout(self, operation) -> EmbeddingResponse:
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tool-memory-embedding",
        )
        future = executor.submit(operation)
        try:
            return future.result(timeout=self.model_timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError(
                "Tool memory EmbeddingModel timed out after "
                f"{self.model_timeout_seconds:g}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.embedding_model.dimensions:
            raise ValueError(
                "Embedding vector dimension does not match configured dimensions"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding vector contains a non-finite value")

    @staticmethod
    def _combine_scores(
        states: Mapping[str, dict[str, Any]],
        mode: str,
    ) -> None:
        if mode == "full":
            for index, state in enumerate(states.values(), start=1):
                state["combined_score"] = 1.0 / index
            return
        if mode == "bm25":
            for state in states.values():
                state["combined_score"] = state["bm25_score"]
            return
        if mode == "embedding":
            for state in states.values():
                state["combined_score"] = state["embedding_score"]
            return
        if mode == "hybrid":
            max_bm25 = max(
                (float(state["bm25_score"]) for state in states.values()),
                default=0.0,
            )
            max_embedding = max(
                (
                    max(0.0, float(state["embedding_score"]))
                    for state in states.values()
                ),
                default=0.0,
            )
            for state in states.values():
                lexical = (
                    float(state["bm25_score"]) / max_bm25
                    if max_bm25
                    else 0.0
                )
                semantic = (
                    max(0.0, float(state["embedding_score"])) / max_embedding
                    if max_embedding
                    else 0.0
                )
                state["combined_score"] = (lexical + semantic) / 2
            return
        if mode == "none":
            return
        raise ValueError(f"Unsupported Tool memory retrieval mode: {mode}")

    @staticmethod
    def _render_index(record: ToolMemoryRecord) -> str:
        return " ".join(
            (
                record.tool_domain.replace("_", " "),
                record.topic.replace("_", " "),
                record.memory_type,
                json.dumps(record.conditions, ensure_ascii=False, sort_keys=True),
                json.dumps(record.value, ensure_ascii=False, sort_keys=True),
            )
        )

    @staticmethod
    def _render_context(record: ToolMemoryRecord) -> str:
        conditions = json.dumps(
            record.conditions,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        value = json.dumps(
            record.value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (
            f"- [id={record.id} domain={record.tool_domain} "
            f"topic={record.topic} scope={record.scope} "
            f"conditions={conditions} version={record.version} "
            f"confidence={record.confidence:.2f}] value={value}"
        )


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("_", " ")
    return tuple(_TOKEN_PATTERN.findall(normalized))


def _routing_focus(query: str) -> str:
    stripped = query.strip()
    candidates = []
    for quote in ("'", '"'):
        positions = [
            index
            for index, character in enumerate(stripped)
            if character == quote
            and not (
                index > 0
                and index + 1 < len(stripped)
                and stripped[index - 1].isalnum()
                and stripped[index + 1].isalnum()
            )
        ]
        if len(positions) >= 2:
            focused = stripped[positions[0] + 1 : positions[-1]].strip()
            if focused:
                candidates.append(focused)
    if not candidates:
        return stripped
    return max(candidates, key=len)


def _tool_route_document(tool: ToolMemoryTool) -> str:
    slots = []
    for slot in tool.slots:
        description = str(slot.schema.get("description", "")).strip()
        enum = slot.schema.get("enum", ())
        enum_text = (
            " ".join(str(item) for item in enum)
            if isinstance(enum, Sequence)
            and not isinstance(enum, (str, bytes))
            else ""
        )
        slots.append(f"{slot.name} {description} {enum_text}".strip())
    return " ".join(
        (
            tool.tool_name.replace("_", " "),
            tool.domain.replace("_", " "),
            tool.topic.replace("_", " "),
            tool.action.replace("_", " "),
            tool.description,
            *slots,
        )
    )


def _tool_route_fingerprint(tool: ToolMemoryTool) -> str:
    return hashlib.sha256(
        _tool_route_document(tool).encode("utf-8")
    ).hexdigest()


def _schema_terms(value: str) -> tuple[str, ...]:
    terms = set(_tokens(value))
    for term in tuple(terms):
        terms.update(_GENERIC_ALIASES.get(term, ()))
    return tuple(terms)


def _condition_query_text(
    query: str,
    context: Mapping[str, Any],
) -> str:
    return normalize_identifier(
        " ".join(
            (
                query,
                json.dumps(context, ensure_ascii=False, sort_keys=True),
            )
        )
    )


def _conditions_match(
    conditions: Mapping[str, Any],
    query: str,
    context: Mapping[str, Any],
) -> bool:
    condition_text = _condition_query_text(query, context)
    query_tokens = set(condition_text.split("_"))
    raw_text = " ".join(
        (query, json.dumps(context, ensure_ascii=False, sort_keys=True))
    )
    time_buckets = _time_buckets(raw_text)
    for key, value in _condition_items(conditions):
        normalized_key = normalize_identifier(key)
        normalized_value = normalize_identifier(str(value))
        if not normalized_value:
            continue
        value_tokens = set(normalized_value.split("_"))
        if normalized_key in _ENTITY_CONDITION_KEYS:
            if not value_tokens & query_tokens:
                return False
            continue
        if normalized_key in {"time", "time_of_day", "period"}:
            expected = _time_value_bucket(normalized_value)
            if expected is not None and expected not in time_buckets:
                return False
            if expected is None and normalized_value not in condition_text:
                return False
            continue
        if normalized_key in {"weather", "climate"}:
            expected_weather = _weather_bucket(value_tokens)
            if expected_weather is not None:
                if not (_WEATHER_TERMS[expected_weather] & query_tokens):
                    return False
            elif not value_tokens <= query_tokens:
                return False
            continue
        if normalized_key in _SOFT_CONTEXT_KEYS:
            continue
        # Other applicability details contribute to BM25/embedding ranking.
        # They are not hard exclusions because natural-language paraphrases
        # should remain retrievable.
    return True


def _condition_items(value: Any, prefix: str = "") -> tuple[tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        return tuple(
            item
            for key, nested in value.items()
            for item in _condition_items(
                nested,
                f"{prefix}_{key}" if prefix else str(key),
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            item
            for nested in value
            for item in _condition_items(nested, prefix)
        )
    if value is None:
        return ()
    return ((prefix, value),)


def _time_buckets(value: str) -> set[str]:
    tokens = set(_tokens(value))
    buckets: set[str] = set()
    if tokens & {"day", "daytime", "morning", "noon", "sunny", "낮", "아침"}:
        buckets.add("daytime")
    if tokens & {"dark", "evening", "night", "밤", "저녁"}:
        buckets.add("night")
    for match in _CLOCK_PATTERN.finditer(value):
        hour = int(match.group(1)) % 12
        if match.group(3).casefold() == "p":
            hour += 12
        buckets.add("daytime" if 6 <= hour < 18 else "night")
    return buckets


def _time_value_bucket(value: str) -> str | None:
    tokens = set(value.split("_"))
    if tokens & {"day", "daytime", "morning", "noon"}:
        return "daytime"
    if tokens & {"dark", "evening", "night"}:
        return "night"
    return None


def _weather_bucket(tokens: set[str]) -> str | None:
    for bucket, terms in _WEATHER_TERMS.items():
        if terms & tokens:
            return bucket
    return None


def _condition_leaves(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        return tuple(
            leaf
            for item in value.values()
            for leaf in _condition_leaves(item)
        )
    if isinstance(value, (list, tuple)):
        return tuple(leaf for item in value for leaf in _condition_leaves(item))
    if value is None:
        return ()
    return (value,)


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
