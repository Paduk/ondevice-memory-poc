from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from palmclaw_ubuntu.models import FactMemoryRecord, MemoryEvidence
from palmclaw_ubuntu.tokens import TokenCounter

VEHICLE_WIKI_POLICY_VERSION = "vehicle-wiki-virtual-pages-v1"

VEHICLE_WIKI_RELATIONS = frozenset(
    {
        "about_entity",
        "applies_under",
        "has_fact",
        "implemented_by_tool",
        "next_update",
        "previous_update",
        "related_capability",
        "supported_by",
        "supersedes",
    }
)

_ACYCLIC_RELATIONS = frozenset(
    {"next_update", "previous_update", "supersedes"}
)
_PROVENANCE_REQUIRED_PAGE_TYPES = frozenset(
    {"evidence", "fact", "summary_overview", "summary_update"}
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


class VehicleWikiValidationError(ValueError):
    """Raised when a virtual Vehicle Wiki violates its structural contract."""


class _TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


@dataclass(frozen=True)
class VehicleWikiLink:
    source_page_id: str
    target_page_id: str
    relation: str
    evidence: tuple[str, ...] = ()
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_page_id": self.source_page_id,
            "target_page_id": self.target_page_id,
            "relation": self.relation,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class VehicleWikiPage:
    page_id: str
    page_type: str
    title: str
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    description: str = ""
    content: str = ""
    source_ids: tuple[str, ...] = ()
    links: tuple[VehicleWikiLink, ...] = ()
    version: int = 1
    updated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_type": self.page_type,
            "title": self.title,
            "aliases": list(self.aliases),
            "tags": list(self.tags),
            "description": self.description,
            "content": self.content,
            "source_ids": list(self.source_ids),
            "links": [
                link.as_dict()
                for link in sorted(
                    self.links,
                    key=lambda item: (
                        item.relation,
                        item.target_page_id,
                        item.source_page_id,
                    ),
                )
            ],
            "version": self.version,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class VehicleWikiSearchHit:
    page_id: str
    title: str
    description: str
    score: float
    matched_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class VehicleWikiSearchResult:
    query: str
    hits: tuple[VehicleWikiSearchHit, ...]
    latency_ms: int = 0


@dataclass(frozen=True)
class VehicleWikiReadResult:
    requested_page_ids: tuple[str, ...]
    pages: tuple[VehicleWikiPage, ...]
    rendered_content: str
    rendered_tokens: int
    missing_page_ids: tuple[str, ...] = ()
    budget_excluded_page_ids: tuple[str, ...] = ()
    truncated: bool = False
    latency_ms: int = 0


@dataclass(frozen=True)
class VehicleWikiTraversalStep:
    action: str
    query: str | None = None
    page_ids: tuple[str, ...] = ()
    result_page_ids: tuple[str, ...] = ()
    rendered_tokens: int = 0


@dataclass(frozen=True)
class VehicleWikiTraversalTrace:
    steps: tuple[VehicleWikiTraversalStep, ...] = ()
    termination_reason: str | None = None
    search_count: int = 0
    read_count: int = 0
    hop_count: int = 0
    empty_search_count: int = 0
    selected_page_ids: tuple[str, ...] = ()
    rendered_tokens: int = 0
    fallback_used: bool = False
    error: str | None = None


@dataclass(frozen=True)
class VehicleWikiTraversalResult:
    content: str
    pages: tuple[VehicleWikiPage, ...]
    trace: VehicleWikiTraversalTrace
    fallback_used: bool = False
    error: str | None = None


@dataclass(frozen=True)
class VehicleWikiValidationReport:
    page_count: int
    link_count: int
    provenance_page_count: int
    max_content_tokens: int
    fingerprint: str


@dataclass(frozen=True)
class SummaryWikiVersion:
    version_id: str
    content: str
    created_at: str
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactWikiSource:
    message_id: int
    quote: str
    start_char: int | None = None
    end_char: int | None = None

    @classmethod
    def from_evidence(cls, evidence: MemoryEvidence) -> FactWikiSource:
        return cls(
            message_id=evidence.message_id,
            quote=evidence.quote,
            start_char=evidence.start_char,
            end_char=evidence.end_char,
        )


def vehicle_wiki_page_id(
    namespace: str,
    page_type: str,
    stable_key: str,
) -> str:
    payload = _canonical_json(
        {
            "namespace": namespace,
            "page_type": page_type,
            "stable_key": stable_key,
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"wiki:{page_type}:{digest}"


def vehicle_wiki_payload(
    pages: Sequence[VehicleWikiPage],
) -> dict[str, Any]:
    return {
        "policy_version": VEHICLE_WIKI_POLICY_VERSION,
        "pages": [
            page.as_dict() for page in sorted(pages, key=lambda item: item.page_id)
        ],
    }


def vehicle_wiki_fingerprint(pages: Sequence[VehicleWikiPage]) -> str:
    payload = _canonical_json(vehicle_wiki_payload(pages))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_vehicle_wiki(
    pages: Sequence[VehicleWikiPage],
    *,
    max_page_tokens: int = 1_200,
    token_counter: _TokenCounter | None = None,
) -> VehicleWikiValidationReport:
    if max_page_tokens < 1:
        raise ValueError("max_page_tokens must be positive")
    counter = token_counter or TokenCounter()
    by_id: dict[str, VehicleWikiPage] = {}
    max_content_tokens = 0
    provenance_page_count = 0
    link_count = 0

    for page in pages:
        if not page.page_id or not page.page_type or not page.title.strip():
            raise VehicleWikiValidationError(
                "Wiki pages require page_id, page_type, and title"
            )
        if page.page_id in by_id:
            raise VehicleWikiValidationError(
                f"Duplicate Wiki page ID: {page.page_id}"
            )
        if page.version < 1:
            raise VehicleWikiValidationError(
                f"Wiki page version must be positive: {page.page_id}"
            )
        if len(page.source_ids) != len(set(page.source_ids)):
            raise VehicleWikiValidationError(
                f"Duplicate source IDs on Wiki page: {page.page_id}"
            )
        if page.page_type in _PROVENANCE_REQUIRED_PAGE_TYPES:
            provenance_page_count += 1
            if not page.source_ids:
                raise VehicleWikiValidationError(
                    f"Wiki page lacks source provenance: {page.page_id}"
                )
        content_tokens = counter.count(page.content)
        max_content_tokens = max(max_content_tokens, content_tokens)
        if content_tokens > max_page_tokens:
            raise VehicleWikiValidationError(
                f"Wiki page exceeds {max_page_tokens} tokens: {page.page_id}"
            )
        by_id[page.page_id] = page

    edges_by_relation: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for page in pages:
        seen_links: set[tuple[str, str]] = set()
        for link in page.links:
            link_count += 1
            if link.source_page_id != page.page_id:
                raise VehicleWikiValidationError(
                    f"Wiki link source mismatch on page: {page.page_id}"
                )
            if link.relation not in VEHICLE_WIKI_RELATIONS:
                raise VehicleWikiValidationError(
                    f"Unsupported Wiki relation: {link.relation}"
                )
            if link.target_page_id not in by_id:
                raise VehicleWikiValidationError(
                    f"Dangling Wiki link: {link.target_page_id}"
                )
            if link.source_page_id == link.target_page_id:
                raise VehicleWikiValidationError(
                    f"Self-referencing Wiki link: {link.source_page_id}"
                )
            if not 0 <= link.confidence <= 1:
                raise VehicleWikiValidationError(
                    f"Wiki link confidence is out of range: {link.confidence}"
                )
            identity = (link.relation, link.target_page_id)
            if identity in seen_links:
                raise VehicleWikiValidationError(
                    f"Duplicate Wiki link on page: {page.page_id}"
                )
            seen_links.add(identity)
            if link.relation in _ACYCLIC_RELATIONS:
                edges_by_relation[link.relation][link.source_page_id].add(
                    link.target_page_id
                )

    for relation, edges in edges_by_relation.items():
        if _contains_cycle(edges):
            raise VehicleWikiValidationError(
                f"Wiki relation contains a cycle: {relation}"
            )

    return VehicleWikiValidationReport(
        page_count=len(pages),
        link_count=link_count,
        provenance_page_count=provenance_page_count,
        max_content_tokens=max_content_tokens,
        fingerprint=vehicle_wiki_fingerprint(pages),
    )


def build_summary_wiki_pages(
    *,
    namespace: str,
    versions: Sequence[SummaryWikiVersion],
    max_page_tokens: int = 1_200,
    token_counter: _TokenCounter | None = None,
) -> tuple[VehicleWikiPage, ...]:
    if not namespace.strip():
        raise ValueError("namespace must not be empty")
    if max_page_tokens < 1:
        raise ValueError("max_page_tokens must be positive")
    counter = token_counter or TokenCounter()
    ordered = tuple(
        sorted(
            (version for version in versions if version.content.strip()),
            key=lambda item: (item.created_at, item.version_id),
        )
    )
    if not ordered:
        return ()
    version_ids = [version.version_id for version in ordered]
    if len(version_ids) != len(set(version_ids)):
        raise VehicleWikiValidationError("Duplicate Recursive Summary version ID")

    update_specs: list[tuple[SummaryWikiVersion, str]] = []
    previous_sentences: tuple[str, ...] = ()
    for version in ordered:
        current_sentences = _split_sentences(version.content)
        previous_keys = {_normalized_sentence(item) for item in previous_sentences}
        delta = tuple(
            sentence
            for sentence in current_sentences
            if _normalized_sentence(sentence) not in previous_keys
        )
        if delta:
            update_specs.append((version, "\n".join(delta)))
        previous_sentences = current_sentences

    overview_id = vehicle_wiki_page_id(namespace, "summary_overview", "latest")
    update_ids = {
        version.version_id: vehicle_wiki_page_id(
            namespace,
            "summary_update",
            version.version_id,
        )
        for version, _ in update_specs
    }
    pages: list[VehicleWikiPage] = []
    latest = ordered[-1]
    overview_links: tuple[VehicleWikiLink, ...] = ()
    if update_specs:
        latest_update_id = update_ids[update_specs[-1][0].version_id]
        overview_links = (
            VehicleWikiLink(
                source_page_id=overview_id,
                target_page_id=latest_update_id,
                relation="previous_update",
                evidence=(latest.version_id,),
            ),
        )
    pages.append(
        VehicleWikiPage(
            page_id=overview_id,
            page_type="summary_overview",
            title="Current recursive memory overview",
            aliases=("recursive summary", "current memory"),
            tags=("overview", "recursive_summary"),
            description="Latest bounded Recursive Summary snapshot.",
            content=counter.truncate(latest.content.strip(), max_page_tokens),
            source_ids=_source_ids(latest),
            links=overview_links,
            version=len(ordered),
            updated_at=latest.created_at,
        )
    )

    for index, (version, delta) in enumerate(update_specs):
        page_id = update_ids[version.version_id]
        links: list[VehicleWikiLink] = []
        if index:
            previous_id = update_ids[update_specs[index - 1][0].version_id]
            links.append(
                VehicleWikiLink(
                    source_page_id=page_id,
                    target_page_id=previous_id,
                    relation="previous_update",
                    evidence=(version.version_id,),
                )
            )
        if index + 1 < len(update_specs):
            next_id = update_ids[update_specs[index + 1][0].version_id]
            links.append(
                VehicleWikiLink(
                    source_page_id=page_id,
                    target_page_id=next_id,
                    relation="next_update",
                    evidence=(version.version_id,),
                )
            )
        pages.append(
            VehicleWikiPage(
                page_id=page_id,
                page_type="summary_update",
                title=f"Recursive memory update {version.created_at}",
                aliases=(version.version_id,),
                tags=(
                    "recursive_summary",
                    "summary_update",
                    version.created_at[:10],
                ),
                description="New sentences introduced by this summary version.",
                content=counter.truncate(delta, max_page_tokens),
                source_ids=_source_ids(version),
                links=tuple(links),
                version=index + 1,
                updated_at=version.created_at,
            )
        )

    result = tuple(sorted(pages, key=lambda item: item.page_id))
    validate_vehicle_wiki(
        result,
        max_page_tokens=max_page_tokens,
        token_counter=counter,
    )
    return result


def build_fact_wiki_pages(
    *,
    namespace: str,
    records: Sequence[FactMemoryRecord],
    sources_by_record: Mapping[
        str,
        Sequence[FactWikiSource | MemoryEvidence],
    ],
    tools_by_capability: Mapping[str, Sequence[str]] | None = None,
    max_page_tokens: int = 1_200,
    token_counter: _TokenCounter | None = None,
) -> tuple[VehicleWikiPage, ...]:
    if not namespace.strip():
        raise ValueError("namespace must not be empty")
    if max_page_tokens < 1:
        raise ValueError("max_page_tokens must be positive")
    counter = token_counter or TokenCounter()
    ordered_records = tuple(sorted(records, key=lambda item: item.id))
    record_ids = {record.id for record in ordered_records}
    if len(record_ids) != len(ordered_records):
        raise VehicleWikiValidationError("Duplicate Fact record ID")
    normalized_tools = {
        capability: tuple(sorted(set(tool_names)))
        for capability, tool_names in (tools_by_capability or {}).items()
    }

    page_specs: dict[str, dict[str, Any]] = {}
    links: dict[str, list[VehicleWikiLink]] = defaultdict(list)

    def add_page(
        *,
        page_id: str,
        page_type: str,
        title: str,
        aliases: Sequence[str] = (),
        tags: Sequence[str] = (),
        description: str = "",
        content: str = "",
        source_ids: Sequence[str] = (),
        version: int = 1,
        updated_at: str | None = None,
    ) -> None:
        existing = page_specs.get(page_id)
        spec = {
            "page_id": page_id,
            "page_type": page_type,
            "title": title,
            "aliases": tuple(sorted(set(aliases))),
            "tags": tuple(sorted(set(tags))),
            "description": description,
            "content": counter.truncate(content, max_page_tokens),
            "source_ids": tuple(sorted(set(source_ids))),
            "version": version,
            "updated_at": updated_at,
        }
        if existing is None:
            page_specs[page_id] = spec
            return
        immutable_fields = (
            "page_type",
            "title",
            "description",
            "content",
        )
        if any(existing[field] != spec[field] for field in immutable_fields):
            raise VehicleWikiValidationError(f"Conflicting Wiki page: {page_id}")
        existing["aliases"] = tuple(
            sorted({*existing["aliases"], *spec["aliases"]})
        )
        existing["tags"] = tuple(sorted({*existing["tags"], *spec["tags"]}))
        existing["source_ids"] = tuple(
            sorted({*existing["source_ids"], *spec["source_ids"]})
        )
        existing["version"] = max(existing["version"], spec["version"])
        existing["updated_at"] = max(
            filter(None, (existing["updated_at"], spec["updated_at"])),
            default=None,
        )

    def add_link(
        source_page_id: str,
        target_page_id: str,
        relation: str,
        evidence: Sequence[str],
        confidence: float = 1.0,
    ) -> None:
        for index, existing in enumerate(links[source_page_id]):
            if (
                existing.target_page_id == target_page_id
                and existing.relation == relation
            ):
                links[source_page_id][index] = VehicleWikiLink(
                    source_page_id=source_page_id,
                    target_page_id=target_page_id,
                    relation=relation,
                    evidence=tuple(
                        sorted({*existing.evidence, *evidence})
                    ),
                    confidence=max(existing.confidence, confidence),
                )
                return
        links[source_page_id].append(
            VehicleWikiLink(
                source_page_id=source_page_id,
                target_page_id=target_page_id,
                relation=relation,
                evidence=tuple(sorted(set(evidence))),
                confidence=confidence,
            )
        )

    for record in ordered_records:
        record_sources = tuple(
            _fact_source(item) for item in sources_by_record.get(record.id, ())
        )
        if not record_sources:
            raise VehicleWikiValidationError(
                f"Fact record lacks source evidence: {record.id}"
            )
        fact_id = vehicle_wiki_page_id(namespace, "fact", record.id)
        entity_id = vehicle_wiki_page_id(
            namespace,
            "entity",
            record.entity_id,
        )
        source_ids = tuple(
            f"message:{source.message_id}" for source in record_sources
        )
        add_page(
            page_id=entity_id,
            page_type="entity",
            title=record.entity_id,
            aliases=(record.entity_id, record.user_id),
            tags=("entity",),
            description="Entity referenced by versioned vehicle facts.",
            content=record.entity_id,
            source_ids=(record.id,),
            updated_at=record.updated_at,
        )
        add_page(
            page_id=fact_id,
            page_type="fact",
            title=f"{record.entity_id}: {record.predicate}",
            aliases=(record.predicate, *record.capability_hints),
            tags=(record.memory_type, record.status, "fact"),
            description="Versioned post-normalized vehicle fact.",
            content=_canonical_json(
                {
                    "entity_id": record.entity_id,
                    "predicate": record.predicate,
                    "value": record.value,
                    "identity_conditions": record.identity_conditions,
                    "applicability": record.applicability,
                    "status": record.status,
                    "confidence": record.confidence,
                }
            ),
            source_ids=source_ids,
            version=record.version,
            updated_at=record.updated_at,
        )
        add_link(entity_id, fact_id, "has_fact", (record.id,), record.confidence)
        add_link(fact_id, entity_id, "about_entity", (record.id,), record.confidence)

        conditions = {
            "identity_conditions": record.identity_conditions,
            "applicability": record.applicability,
        }
        if record.identity_conditions or record.applicability:
            condition_key = _canonical_json(conditions)
            condition_id = vehicle_wiki_page_id(
                namespace,
                "condition",
                condition_key,
            )
            add_page(
                page_id=condition_id,
                page_type="condition",
                title="Vehicle Fact conditions",
                tags=("condition",),
                description="Identity and applicability boundary for vehicle facts.",
                content=condition_key,
                source_ids=(record.id,),
                updated_at=record.updated_at,
            )
            add_link(
                fact_id,
                condition_id,
                "applies_under",
                (record.id,),
                record.confidence,
            )

        for source in record_sources:
            evidence_key = (
                f"{record.id}:{source.message_id}:"
                f"{source.start_char}:{source.end_char}"
            )
            evidence_id = vehicle_wiki_page_id(
                namespace,
                "evidence",
                evidence_key,
            )
            message_source_id = f"message:{source.message_id}"
            add_page(
                page_id=evidence_id,
                page_type="evidence",
                title=f"Evidence for {record.predicate}",
                tags=("evidence",),
                description="Bounded source quote supporting a vehicle fact.",
                content=source.quote.strip(),
                source_ids=(message_source_id,),
                updated_at=record.updated_at,
            )
            add_link(
                fact_id,
                evidence_id,
                "supported_by",
                (message_source_id,),
                record.confidence,
            )

        for capability in sorted(set(record.capability_hints)):
            capability_id = vehicle_wiki_page_id(
                namespace,
                "capability",
                capability,
            )
            add_page(
                page_id=capability_id,
                page_type="capability",
                title=capability,
                aliases=(capability,),
                tags=("capability",),
                description="Vehicle capability associated with stored facts.",
                content=capability,
                source_ids=(record.id,),
            )
            add_link(
                fact_id,
                capability_id,
                "related_capability",
                (record.id,),
                record.confidence,
            )
            for tool_name in normalized_tools.get(capability, ()):
                tool_id = vehicle_wiki_page_id(namespace, "tool", tool_name)
                add_page(
                    page_id=tool_id,
                    page_type="tool",
                    title=tool_name,
                    aliases=(tool_name,),
                    tags=("tool",),
                    description="Vehicle Tool linked by the existing capability map.",
                    content=tool_name,
                    source_ids=(f"tool:{tool_name}",),
                )
                add_link(
                    capability_id,
                    tool_id,
                    "implemented_by_tool",
                    (f"tool:{tool_name}",),
                )

        if record.supersedes_id in record_ids:
            previous_fact_id = vehicle_wiki_page_id(
                namespace,
                "fact",
                str(record.supersedes_id),
            )
            add_link(
                fact_id,
                previous_fact_id,
                "supersedes",
                (record.id, str(record.supersedes_id)),
                record.confidence,
            )

    pages = tuple(
        VehicleWikiPage(
            **spec,
            links=tuple(
                sorted(
                    links[page_id],
                    key=lambda item: (item.relation, item.target_page_id),
                )
            ),
        )
        for page_id, spec in sorted(page_specs.items())
    )
    validate_vehicle_wiki(
        pages,
        max_page_tokens=max_page_tokens,
        token_counter=counter,
    )
    return pages


def _source_ids(version: SummaryWikiVersion) -> tuple[str, ...]:
    return tuple(sorted(set(version.source_ids or (version.version_id,))))


def _fact_source(item: FactWikiSource | MemoryEvidence) -> FactWikiSource:
    return (
        item
        if isinstance(item, FactWikiSource)
        else FactWikiSource.from_evidence(item)
    )


def _split_sentences(content: str) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in _SENTENCE_BOUNDARY.split(content.strip())
        if part.strip()
    )


def _normalized_sentence(sentence: str) -> str:
    return " ".join(sentence.casefold().split())


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _contains_cycle(edges: Mapping[str, set[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for target in edges.get(node, set()):
            if visit(target):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in edges)
