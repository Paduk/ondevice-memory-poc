from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from palmclaw_ubuntu.models import ToolDefinition, ToolResult
from palmclaw_ubuntu.tokens import TokenCounter
from palmclaw_ubuntu.tools import (
    ToolExecutionContext,
    ToolRegistry,
)
from palmclaw_ubuntu.vehicle_wiki import (
    SummaryWikiVersion,
    VehicleWikiPage,
    build_summary_wiki_pages,
    vehicle_wiki_fingerprint,
)
from palmclaw_ubuntu.vehicle_wiki_retrieval import (
    VehicleWikiIndex,
    VehicleWikiTraversalBudget,
    VehicleWikiTraversalSession,
)

SUMMARY_WIKI_GATE_POLICY_VERSION = "summary-wiki-coverage-gate-v1"
SUMMARY_WIKI_RUNTIME_POLICY_VERSION = "summary-wiki-agent-tools-v1"
MEMORY_WIKI_SEARCH = "memory_wiki_search"
MEMORY_WIKI_READ = "memory_wiki_read"
MEMORY_WIKI_TOOL_NAMES = (MEMORY_WIKI_SEARCH, MEMORY_WIKI_READ)

_TOKEN_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)
_HEADING_ENTITY_PATTERN = re.compile(r"(?m)^\s*\*\*([^*\n:]+)\*\*\s*:?")
_COMPARISON_MARKERS = (
    "compare",
    "comparison",
    "difference",
    "both",
    " versus ",
    " vs ",
    "비교",
    "차이",
    "둘 다",
    "양쪽",
)
_CONDITION_MARKERS = (
    " when ",
    " whenever ",
    " while ",
    " if ",
    " at night",
    " during ",
    " depending on",
    "때 ",
    "때는",
    "경우",
    "조건",
    "밤에",
    "낮에",
    "비가",
    "날씨",
)
_CORRECTION_MARKERS = (
    "earlier",
    "previous",
    "before",
    "used to",
    "changed",
    "corrected",
    "actually",
    "old setting",
    "이전",
    "예전",
    "바꾸기 전",
    "변경",
    "정정",
    "원래",
)
_COVERAGE_STOPWORDS = frozenset(
    {
        "apply",
        "car",
        "current",
        "for",
        "from",
        "have",
        "memory",
        "please",
        "preference",
        "preferences",
        "recursive",
        "set",
        "setting",
        "settings",
        "summary",
        "that",
        "the",
        "their",
        "this",
        "use",
        "vehicle",
        "with",
        "값",
        "기억",
        "설정",
        "적용",
        "차량",
        "해줘",
    }
)


@dataclass(frozen=True)
class SummaryWikiGateDecision:
    opened: bool
    reasons: tuple[str, ...]
    matched_entities: tuple[str, ...] = ()
    missing_entities: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()
    query_features: tuple[str, ...] = ()
    policy_version: str = SUMMARY_WIKI_GATE_POLICY_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SummaryWikiRuntime:
    pages: tuple[VehicleWikiPage, ...]
    session: VehicleWikiTraversalSession
    gate: SummaryWikiGateDecision
    page_fingerprint: str
    cache_signature: str


def build_summary_wiki_versions(
    memories: Sequence[Mapping[str, Any]],
    daily_steps: Sequence[Mapping[str, Any]],
) -> tuple[SummaryWikiVersion, ...]:
    """Reconstruct immutable Summary versions without reading raw History."""

    rows = tuple(
        sorted(
            (item for item in memories if str(item.get("content", "")).strip()),
            key=lambda item: (
                int(item.get("version", 0)),
                str(item.get("created_at", "")),
                str(item.get("id", "")),
            ),
        )
    )
    steps_by_hash: dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
    for step in sorted(
        daily_steps,
        key=lambda item: int(item.get("index", 0)),
    ):
        if step.get("status") != "updated":
            continue
        steps_by_hash[str(step.get("summary_sha256", ""))].append(step)

    versions = []
    for index, row in enumerate(rows):
        content = str(row["content"]).strip()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        matching_steps = steps_by_hash.get(content_hash)
        step = matching_steps.popleft() if matching_steps else None
        row_id = str(row.get("id") or f"summary-version-{index + 1}")
        if step is None:
            created_at = str(row.get("created_at") or f"version-{index + 1:06d}")
            source_ids = (f"memory:{row_id}",)
        else:
            created_at = (
                f"{step.get('date', 'unknown-date')}T00:00:00Z-"
                f"{int(step.get('index', index)):06d}"
            )
            run_id = str(step.get("consolidation_run_id") or "").strip()
            source_ids = (
                (f"consolidation:{run_id}",)
                if run_id
                else (f"memory:{row_id}",)
            )
        versions.append(
            SummaryWikiVersion(
                version_id=row_id,
                content=content,
                created_at=created_at,
                source_ids=source_ids,
            )
        )
    return tuple(versions)


def evaluate_summary_wiki_gate(
    query: str,
    final_summary: str,
    pages: Sequence[VehicleWikiPage],
) -> SummaryWikiGateDecision:
    normalized_query = f" {query.casefold().strip()} "
    current_entities = _summary_entities(final_summary)
    historical_entities = _summary_entities(
        "\n".join(page.content for page in pages)
    )
    all_entities = tuple(
        sorted(
            {*current_entities, *historical_entities},
            key=lambda item: (item.casefold(), item),
        )
    )
    matched_entities = tuple(
        entity
        for entity in all_entities
        if _contains_phrase(normalized_query, entity)
    )
    current_entity_keys = {item.casefold() for item in current_entities}
    missing_entities = tuple(
        entity
        for entity in matched_entities
        if entity.casefold() not in current_entity_keys
    )

    features = []
    if len(matched_entities) > 1:
        features.append("multi_entity")
    if _contains_marker(normalized_query, _COMPARISON_MARKERS):
        features.append("comparison")
    if _contains_marker(normalized_query, _CONDITION_MARKERS):
        features.append("conditional")
    if _contains_marker(normalized_query, _CORRECTION_MARKERS):
        features.append("correction_or_history")

    current_terms = set(_coverage_tokens(final_summary))
    historical_terms = set(
        _coverage_tokens(
            "\n".join(
                page.content
                for page in pages
                if page.page_type == "summary_update"
            )
        )
    )
    query_terms = set(_coverage_tokens(query))
    entity_terms = {
        token for entity in matched_entities for token in _coverage_tokens(entity)
    }
    missing_terms = tuple(
        sorted(
            (query_terms & historical_terms) - current_terms - entity_terms
        )
    )

    reasons = [*features]
    if missing_entities:
        reasons.append("entity_missing_from_final_summary")
    if missing_terms:
        reasons.append("query_term_missing_from_final_summary")
    reasons = list(dict.fromkeys(reasons))
    return SummaryWikiGateDecision(
        opened=bool(reasons),
        reasons=tuple(reasons),
        matched_entities=matched_entities,
        missing_entities=missing_entities,
        missing_terms=missing_terms,
        query_features=tuple(features),
    )


def build_summary_wiki_runtime(
    *,
    namespace: str,
    query: str,
    final_summary: str,
    memories: Sequence[Mapping[str, Any]],
    daily_steps: Sequence[Mapping[str, Any]],
    base_cache_key: str,
    budget: VehicleWikiTraversalBudget | None = None,
    token_counter: TokenCounter | None = None,
) -> SummaryWikiRuntime:
    versions = build_summary_wiki_versions(memories, daily_steps)
    pages = build_summary_wiki_pages(
        namespace=namespace,
        versions=versions,
        token_counter=token_counter,
    )
    page_fingerprint = vehicle_wiki_fingerprint(pages)
    gate = evaluate_summary_wiki_gate(query, final_summary, pages)
    signature_payload = json.dumps(
        {
            "base_cache_key": base_cache_key,
            "gate_policy": SUMMARY_WIKI_GATE_POLICY_VERSION,
            "runtime_policy": SUMMARY_WIKI_RUNTIME_POLICY_VERSION,
            "page_fingerprint": page_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    cache_signature = hashlib.sha256(
        signature_payload.encode("utf-8")
    ).hexdigest()
    return SummaryWikiRuntime(
        pages=pages,
        session=VehicleWikiTraversalSession(
            VehicleWikiIndex(pages, token_counter=token_counter),
            budget=budget,
        ),
        gate=gate,
        page_fingerprint=page_fingerprint,
        cache_signature=cache_signature,
    )


class MemoryWikiSearchTool:
    def __init__(self, session: VehicleWikiTraversalSession):
        self.session = session
        self.definition = ToolDefinition(
            name=MEMORY_WIKI_SEARCH,
            description=(
                "Search bounded historical Recursive Summary pages. Use only "
                "when the supplied current memory may be incomplete."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Short entity, setting, or condition query",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            timeout_seconds=2,
            side_effect="none",
            retry_safety="safe",
        )

    def run(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        result = self.session.search(str(arguments["query"]))
        payload = {
            "query": result.query,
            "hits": [
                {
                    "page_id": hit.page_id,
                    "title": hit.title,
                    "description": hit.description,
                    "score": round(hit.score, 6),
                    "matched_fields": list(hit.matched_fields),
                }
                for hit in result.hits
            ],
            "terminated": self.session.terminated,
        }
        return ToolResult(
            tool_call_id="",
            content=json.dumps(payload, ensure_ascii=False),
            metadata={
                "wiki_action": "search",
                "hit_count": len(result.hits),
                "search_count": self.session.trace.search_count,
            },
        )


class MemoryWikiReadTool:
    def __init__(self, session: VehicleWikiTraversalSession):
        self.session = session
        self.definition = ToolDefinition(
            name=MEMORY_WIKI_READ,
            description=(
                "Read up to three Vehicle Wiki page IDs returned by search or "
                "linked from a previously read page."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "page_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "maxItems": 3,
                    }
                },
                "required": ["page_ids"],
                "additionalProperties": False,
            },
            timeout_seconds=2,
            side_effect="none",
            retry_safety="safe",
        )

    def run(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        result = self.session.read(
            tuple(str(item) for item in arguments["page_ids"])
        )
        payload = {
            "pages": [page.page_id for page in result.pages],
            "content": result.rendered_content,
            "missing_page_ids": list(result.missing_page_ids),
            "budget_excluded_page_ids": list(
                result.budget_excluded_page_ids
            ),
            "rendered_tokens": result.rendered_tokens,
            "truncated": result.truncated,
        }
        return ToolResult(
            tool_call_id="",
            content=json.dumps(payload, ensure_ascii=False),
            metadata={
                "wiki_action": "read",
                "page_count": len(result.pages),
                "rendered_tokens": result.rendered_tokens,
            },
        )


def register_vehicle_wiki_tools(
    registry: ToolRegistry,
    session: VehicleWikiTraversalSession,
) -> tuple[str, ...]:
    registry.register(MemoryWikiSearchTool(session))
    registry.register(MemoryWikiReadTool(session))
    return MEMORY_WIKI_TOOL_NAMES


register_summary_wiki_tools = register_vehicle_wiki_tools


def _summary_entities(content: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            match.group(1).strip()
            for match in _HEADING_ENTITY_PATTERN.finditer(content)
            if match.group(1).strip()
        )
    )


def _contains_phrase(normalized_query: str, phrase: str) -> bool:
    normalized_phrase = " ".join(phrase.casefold().split())
    return bool(
        re.search(
            rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)",
            normalized_query,
        )
    )


def _contains_marker(query: str, markers: Sequence[str]) -> bool:
    return any(marker.casefold() in query for marker in markers)


def _coverage_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _TOKEN_PATTERN.findall(text.casefold().replace("_", " "))
        if len(token) >= 3 and token not in _COVERAGE_STOPWORDS
    )
