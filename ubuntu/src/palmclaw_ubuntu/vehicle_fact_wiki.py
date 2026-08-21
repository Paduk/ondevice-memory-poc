from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from palmclaw_ubuntu.models import FactMemoryRecord, FactQueryContext
from palmclaw_ubuntu.tokens import TokenCounter
from palmclaw_ubuntu.tool_memory_schema import ToolMemoryOntology
from palmclaw_ubuntu.vehicle_wiki import (
    FactWikiSource,
    VehicleWikiPage,
    build_fact_wiki_pages,
    vehicle_wiki_fingerprint,
    vehicle_wiki_page_id,
)
from palmclaw_ubuntu.vehicle_wiki_retrieval import (
    VehicleWikiIndex,
    VehicleWikiTraversalBudget,
    VehicleWikiTraversalSession,
)

FACT_WIKI_PROJECTION_POLICY_VERSION = "post-normalized-fact-wiki-v2"
FACT_WIKI_EXPANSION_POLICY_VERSION = "fact-wiki-linked-expansion-v1"
FACT_WIKI_MAX_PLANNER_FACTS = 4

_TOKEN_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)
_GENERIC_TOOL_TERMS = frozenset(
    {
        "carcontrol",
        "control",
        "get",
        "set",
        "switch",
        "value",
        "vehicle",
    }
)


@dataclass(frozen=True)
class FactWikiRuntime:
    pages: tuple[VehicleWikiPage, ...]
    session: VehicleWikiTraversalSession
    seed_page_ids: tuple[str, ...]
    expanded_records: tuple[FactMemoryRecord, ...]
    page_fingerprint: str
    cache_signature: str
    semantic_score_count: int


def build_fact_wiki_runtime(
    *,
    namespace: str,
    records: Sequence[FactMemoryRecord],
    sources_by_record: Mapping[str, Sequence[FactWikiSource]],
    seed_records: Sequence[FactMemoryRecord],
    retrieval_trace: Mapping[str, Any],
    query_context: FactQueryContext,
    ontology: ToolMemoryOntology,
    base_cache_key: str,
    budget: VehicleWikiTraversalBudget | None = None,
    token_counter: TokenCounter | None = None,
) -> FactWikiRuntime:
    tools_by_capability = _tools_by_capability(records, ontology)
    pages = build_fact_wiki_pages(
        namespace=namespace,
        records=records,
        sources_by_record=sources_by_record,
        tools_by_capability=tools_by_capability,
        token_counter=token_counter,
    )
    page_fingerprint = vehicle_wiki_fingerprint(pages)
    semantic_scores = _fact_page_semantic_scores(
        namespace,
        retrieval_trace,
    )
    expanded_records = expand_linked_fact_records(
        namespace=namespace,
        pages=pages,
        records=records,
        seed_records=seed_records,
        retrieval_trace=retrieval_trace,
        query_context=query_context,
    )
    seed_page_ids = tuple(
        vehicle_wiki_page_id(namespace, "fact", record.id)
        for record in seed_records
    )
    signature_payload = json.dumps(
        {
            "base_cache_key": base_cache_key,
            "expansion_policy": FACT_WIKI_EXPANSION_POLICY_VERSION,
            "page_fingerprint": page_fingerprint,
            "projection_policy": FACT_WIKI_PROJECTION_POLICY_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return FactWikiRuntime(
        pages=pages,
        session=VehicleWikiTraversalSession(
            VehicleWikiIndex(pages, token_counter=token_counter),
            budget=budget,
            semantic_scores=semantic_scores,
        ),
        seed_page_ids=seed_page_ids,
        expanded_records=expanded_records,
        page_fingerprint=page_fingerprint,
        cache_signature=hashlib.sha256(
            signature_payload.encode("utf-8")
        ).hexdigest(),
        semantic_score_count=len(semantic_scores),
    )


def expand_linked_fact_records(
    *,
    namespace: str,
    pages: Sequence[VehicleWikiPage],
    records: Sequence[FactMemoryRecord],
    seed_records: Sequence[FactMemoryRecord],
    retrieval_trace: Mapping[str, Any],
    query_context: FactQueryContext,
    max_planner_facts: int = FACT_WIKI_MAX_PLANNER_FACTS,
) -> tuple[FactMemoryRecord, ...]:
    if max_planner_facts < 1:
        raise ValueError("max_planner_facts must be positive")
    seed_ids = {record.id for record in seed_records}
    remaining = max(0, max_planner_facts - len(seed_ids))
    if not remaining:
        return ()

    active_by_id = {
        record.id: record
        for record in records
        if record.status == "active"
    }
    eligible_ids, ranks = _eligible_candidate_ids(retrieval_trace)
    eligible_ids &= set(active_by_id)
    page_by_id = {page.page_id: page for page in pages}
    fact_record_by_page = {
        vehicle_wiki_page_id(namespace, "fact", record.id): record.id
        for record in records
    }
    reachable_fact_ids: set[str] = set()
    frontier: deque[tuple[str, int]] = deque(
        (
            vehicle_wiki_page_id(namespace, "fact", record.id),
            0,
        )
        for record in seed_records
    )
    visited: set[str] = set()
    while frontier:
        page_id, depth = frontier.popleft()
        if page_id in visited or depth > 2:
            continue
        visited.add(page_id)
        record_id = fact_record_by_page.get(page_id)
        if record_id is not None and record_id not in seed_ids:
            reachable_fact_ids.add(record_id)
        page = page_by_id.get(page_id)
        if page is None or depth == 2:
            continue
        for link in page.links:
            frontier.append((link.target_page_id, depth + 1))

    seed_entities = {record.entity_id for record in seed_records}
    seed_bundles = {
        record.bundle_id for record in seed_records if record.bundle_id
    }
    seed_capabilities = {
        capability
        for record in seed_records
        for capability in record.capability_hints
    }
    allowed_entities = set(query_context.entities)
    for record in active_by_id.values():
        if record.id in seed_ids:
            continue
        same_entity = record.entity_id in seed_entities
        same_bundle = bool(record.bundle_id and record.bundle_id in seed_bundles)
        shared_capability = bool(
            set(record.capability_hints) & seed_capabilities
        )
        if same_entity and (same_bundle or shared_capability):
            reachable_fact_ids.add(record.id)

    candidates = []
    for record_id in (reachable_fact_ids & eligible_ids) - seed_ids:
        record = active_by_id[record_id]
        if allowed_entities:
            if record.entity_id not in allowed_entities:
                continue
        elif record.entity_id not in seed_entities:
            continue
        candidates.append(record)
    candidates.sort(
        key=lambda record: (
            ranks.get(record.id, 1_000_000),
            -record.confidence,
            -record.version,
            record.id,
        )
    )
    return tuple(candidates[:remaining])


def _tools_by_capability(
    records: Sequence[FactMemoryRecord],
    ontology: ToolMemoryOntology,
) -> dict[str, tuple[str, ...]]:
    result = {}
    for capability in sorted(
        {
            hint
            for record in records
            for hint in record.capability_hints
            if hint.strip()
        }
    ):
        capability_tokens = set(_tokens(capability)) - _GENERIC_TOOL_TERMS
        matches = []
        for tool in ontology.tools:
            tool_tokens = set(
                _tokens(
                    " ".join(
                        (
                            tool.namespace,
                            tool.domain,
                            tool.topic,
                            tool.tool_name,
                        )
                    )
                )
            ) - _GENERIC_TOOL_TERMS
            if capability_tokens & tool_tokens:
                matches.append(tool.tool_name)
        result[capability] = tuple(sorted(matches))
    return result


def _fact_page_semantic_scores(
    namespace: str,
    retrieval_trace: Mapping[str, Any],
) -> dict[str, float]:
    scores = {}
    for candidate in retrieval_trace.get("candidates", ()):
        record_id = str(candidate.get("record_id", "")).strip()
        if not record_id:
            continue
        scores[vehicle_wiki_page_id(namespace, "fact", record_id)] = float(
            candidate.get("embedding_score", 0.0)
        )
    return scores


def _eligible_candidate_ids(
    retrieval_trace: Mapping[str, Any],
) -> tuple[set[str], dict[str, int]]:
    eligible = set()
    ranks = {}
    for candidate in retrieval_trace.get("candidates", ()):
        record_id = str(candidate.get("record_id", "")).strip()
        if not record_id:
            continue
        rank = candidate.get("rank")
        if rank is not None:
            ranks[record_id] = int(rank)
        if candidate.get("exclusion_reason") in {
            None,
            "top_k",
            "token_budget",
        }:
            eligible.add(record_id)
    return eligible, ranks


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        _TOKEN_PATTERN.findall(value.casefold().replace("_", " ").replace(".", " "))
    )
