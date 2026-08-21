from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from palmclaw_ubuntu.tokens import TokenCounter
from palmclaw_ubuntu.vehicle_wiki import (
    VehicleWikiPage,
    VehicleWikiReadResult,
    VehicleWikiSearchHit,
    VehicleWikiSearchResult,
    VehicleWikiTraversalResult,
    VehicleWikiTraversalStep,
    VehicleWikiTraversalTrace,
    validate_vehicle_wiki,
)

VEHICLE_WIKI_SEARCH_LIMIT = 2
VEHICLE_WIKI_READ_PAGE_LIMIT = 4
VEHICLE_WIKI_READ_BATCH_LIMIT = 3
VEHICLE_WIKI_HOP_LIMIT = 2
VEHICLE_WIKI_TOKEN_BUDGET = 1_200
VEHICLE_WIKI_EMPTY_SEARCH_PATIENCE = 2

_TOKEN_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)
_FIELD_WEIGHTS = {
    "title": 4.0,
    "aliases": 4.0,
    "tags": 3.0,
    "description": 2.0,
    "content": 1.0,
}


class VehicleWikiTraversalError(RuntimeError):
    """Base failure raised by the bounded virtual Wiki traversal layer."""


class VehicleWikiTraversalLimitError(VehicleWikiTraversalError):
    """Raised when an operation would exceed a hard traversal budget."""


@dataclass(frozen=True)
class VehicleWikiTraversalBudget:
    search_limit: int = VEHICLE_WIKI_SEARCH_LIMIT
    read_page_limit: int = VEHICLE_WIKI_READ_PAGE_LIMIT
    read_batch_limit: int = VEHICLE_WIKI_READ_BATCH_LIMIT
    hop_limit: int = VEHICLE_WIKI_HOP_LIMIT
    token_budget: int = VEHICLE_WIKI_TOKEN_BUDGET
    empty_search_patience: int = VEHICLE_WIKI_EMPTY_SEARCH_PATIENCE
    search_top_k: int = 5

    def __post_init__(self) -> None:
        positive = {
            "search_limit": self.search_limit,
            "read_page_limit": self.read_page_limit,
            "read_batch_limit": self.read_batch_limit,
            "token_budget": self.token_budget,
            "empty_search_patience": self.empty_search_patience,
            "search_top_k": self.search_top_k,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.hop_limit < 0:
            raise ValueError("hop_limit cannot be negative")
        if self.read_batch_limit > self.read_page_limit:
            raise ValueError(
                "read_batch_limit cannot exceed read_page_limit"
            )


class VehicleWikiIndex:
    """Immutable local index over virtual Wiki pages.

    Search is lexical by default. Callers may supply scores from already
    available embeddings, but this class never invokes an embedding provider.
    """

    def __init__(
        self,
        pages: Sequence[VehicleWikiPage],
        *,
        token_counter: TokenCounter | None = None,
        max_page_tokens: int = VEHICLE_WIKI_TOKEN_BUDGET,
    ):
        self.token_counter = token_counter or TokenCounter()
        validate_vehicle_wiki(
            pages,
            max_page_tokens=max_page_tokens,
            token_counter=self.token_counter,
        )
        self.pages = tuple(sorted(pages, key=lambda item: item.page_id))
        self._by_id = {page.page_id: page for page in self.pages}

    def page(self, page_id: str) -> VehicleWikiPage | None:
        return self._by_id.get(page_id)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        semantic_scores: Mapping[str, float] | None = None,
    ) -> VehicleWikiSearchResult:
        if not query.strip():
            raise ValueError("Wiki search query cannot be empty")
        if top_k < 1:
            raise ValueError("Wiki search top_k must be positive")
        started = time.monotonic()
        query_tokens = set(_tokenize(query))
        query_folded = " ".join(query.casefold().split())
        ranked: list[VehicleWikiSearchHit] = []
        for page in self.pages:
            lexical_score, matched_fields = _page_lexical_score(
                page,
                query_tokens=query_tokens,
                query_folded=query_folded,
            )
            semantic_score = _semantic_score(
                (semantic_scores or {}).get(page.page_id)
            )
            if lexical_score <= 0 and semantic_score <= 0:
                continue
            score = lexical_score + semantic_score * 0.25
            ranked.append(
                VehicleWikiSearchHit(
                    page_id=page.page_id,
                    title=page.title,
                    description=page.description,
                    score=score,
                    matched_fields=matched_fields,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.page_id))
        return VehicleWikiSearchResult(
            query=query,
            hits=tuple(ranked[:top_k]),
            latency_ms=_elapsed_ms(started),
        )


class VehicleWikiTraversalSession:
    """Stateful budget guard for future Agent search/read tool calls."""

    def __init__(
        self,
        index: VehicleWikiIndex,
        *,
        budget: VehicleWikiTraversalBudget | None = None,
        semantic_scores: Mapping[str, float] | None = None,
    ):
        self.index = index
        self.budget = budget or VehicleWikiTraversalBudget()
        self.semantic_scores = dict(semantic_scores or {})
        self._steps: list[VehicleWikiTraversalStep] = []
        self._search_count = 0
        self._read_page_count = 0
        self._empty_search_count = 0
        self._max_hop = 0
        self._rendered_tokens = 0
        self._selected_page_ids: list[str] = []
        self._selected_pages: dict[str, VehicleWikiPage] = {}
        self._rendered_parts: list[str] = []
        self._discovered_depth: dict[str, int] = {}
        self._termination_reason: str | None = None
        self._fallback_used = False
        self._error: str | None = None

    @property
    def terminated(self) -> bool:
        return self._termination_reason is not None

    @property
    def trace(self) -> VehicleWikiTraversalTrace:
        return VehicleWikiTraversalTrace(
            steps=tuple(self._steps),
            termination_reason=self._termination_reason,
            search_count=self._search_count,
            read_count=self._read_page_count,
            hop_count=self._max_hop,
            empty_search_count=self._empty_search_count,
            selected_page_ids=tuple(self._selected_page_ids),
            rendered_tokens=self._rendered_tokens,
            fallback_used=self._fallback_used,
            error=self._error,
        )

    def search(
        self,
        query: str,
        *,
        semantic_scores: Mapping[str, float] | None = None,
    ) -> VehicleWikiSearchResult:
        self._ensure_active()
        if self._search_count >= self.budget.search_limit:
            self._raise_limit("search_budget_exhausted")
        result = self.index.search(
            query,
            top_k=self.budget.search_top_k,
            semantic_scores=(
                semantic_scores
                if semantic_scores is not None
                else self.semantic_scores
            ),
        )
        self._search_count += 1
        result_ids = tuple(hit.page_id for hit in result.hits)
        for page_id in result_ids:
            self._remember_depth(page_id, 0)
        if result_ids:
            self._empty_search_count = 0
        else:
            self._empty_search_count += 1
        self._steps.append(
            VehicleWikiTraversalStep(
                action="search",
                query=query,
                result_page_ids=result_ids,
            )
        )
        if self._empty_search_count >= self.budget.empty_search_patience:
            self._termination_reason = "empty_search_patience"
        return result

    def read(self, page_ids: Sequence[str]) -> VehicleWikiReadResult:
        self._ensure_active()
        requested = tuple(dict.fromkeys(page_ids))
        if not requested:
            raise ValueError("Wiki read requires at least one page ID")
        if len(requested) > self.budget.read_batch_limit:
            self._raise_limit("read_batch_budget_exhausted")
        if self._read_page_count + len(requested) > self.budget.read_page_limit:
            self._raise_limit("read_page_budget_exhausted")
        for page_id in requested:
            depth = self._discovered_depth.get(page_id, 0)
            if depth > self.budget.hop_limit:
                self._raise_limit("hop_budget_exhausted")
        self._read_page_count += len(requested)

        started = time.monotonic()
        pages: list[VehicleWikiPage] = []
        missing: list[str] = []
        excluded: list[str] = []
        rendered_parts: list[str] = []
        call_tokens = 0
        truncated = False
        for index, page_id in enumerate(requested):
            page = self.index.page(page_id)
            if page is None:
                missing.append(page_id)
                continue
            rendered = render_vehicle_wiki_page(page)
            available = self.budget.token_budget - self._rendered_tokens
            if available <= 0:
                excluded.extend(requested[index:])
                truncated = True
                break
            rendered_tokens = self.index.token_counter.count(rendered)
            if rendered_tokens > available:
                rendered = self.index.token_counter.truncate(rendered, available)
                rendered_tokens = self.index.token_counter.count(rendered)
                truncated = True
            if not rendered.strip():
                excluded.append(page_id)
                continue
            pages.append(page)
            rendered_parts.append(rendered)
            call_tokens += rendered_tokens
            self._rendered_tokens += rendered_tokens
            self._rendered_parts.append(rendered)
            if page_id not in self._selected_pages:
                self._selected_pages[page_id] = page
                self._selected_page_ids.append(page_id)
            depth = self._discovered_depth.get(page_id, 0)
            self._max_hop = max(self._max_hop, depth)
            for link in page.links:
                self._remember_depth(link.target_page_id, depth + 1)

        self._steps.append(
            VehicleWikiTraversalStep(
                action="read",
                page_ids=requested,
                result_page_ids=tuple(page.page_id for page in pages),
                rendered_tokens=call_tokens,
            )
        )
        return VehicleWikiReadResult(
            requested_page_ids=requested,
            pages=tuple(pages),
            rendered_content="\n\n".join(rendered_parts),
            rendered_tokens=call_tokens,
            missing_page_ids=tuple(missing),
            budget_excluded_page_ids=tuple(excluded),
            truncated=truncated,
            latency_ms=_elapsed_ms(started),
        )

    def finish(
        self,
        *,
        evidence_sufficient: bool,
        fallback_content: str,
    ) -> VehicleWikiTraversalResult:
        if evidence_sufficient and self._rendered_parts:
            self._termination_reason = "evidence_sufficient"
            content = "\n\n".join(self._rendered_parts)
            fallback_used = False
        else:
            if self._termination_reason is None:
                self._termination_reason = "insufficient_evidence"
            content = fallback_content
            fallback_used = True
        self._fallback_used = fallback_used
        return VehicleWikiTraversalResult(
            content=content,
            pages=tuple(self._selected_pages.values()),
            trace=self.trace,
            fallback_used=fallback_used,
            error=self._error,
        )

    def fail(
        self,
        error: Exception | str,
        *,
        fallback_content: str,
    ) -> VehicleWikiTraversalResult:
        self._termination_reason = "tool_error"
        self._error = (
            error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        )
        self._fallback_used = True
        return VehicleWikiTraversalResult(
            content=fallback_content,
            pages=tuple(self._selected_pages.values()),
            trace=self.trace,
            fallback_used=True,
            error=self._error,
        )

    def _remember_depth(self, page_id: str, depth: int) -> None:
        current = self._discovered_depth.get(page_id)
        if current is None or depth < current:
            self._discovered_depth[page_id] = depth

    def _ensure_active(self) -> None:
        if self.terminated:
            raise VehicleWikiTraversalError(
                f"Wiki traversal already terminated: {self._termination_reason}"
            )

    def _raise_limit(self, reason: str) -> None:
        self._termination_reason = reason
        raise VehicleWikiTraversalLimitError(reason)


def render_vehicle_wiki_page(page: VehicleWikiPage) -> str:
    aliases = ", ".join(page.aliases) or "(none)"
    tags = ", ".join(page.tags) or "(none)"
    links = "\n".join(
        f"- {link.relation}: {link.target_page_id}"
        for link in page.links
    ) or "- (none)"
    return (
        f"[Vehicle Wiki: {page.page_type}]\n"
        f"Page ID: {page.page_id}\n"
        f"Title: {page.title}\n"
        f"Aliases: {aliases}\n"
        f"Tags: {tags}\n"
        f"Description: {page.description or '(none)'}\n"
        f"Content:\n{page.content or '(none)'}\n"
        f"Links:\n{links}"
    )


def _page_lexical_score(
    page: VehicleWikiPage,
    *,
    query_tokens: set[str],
    query_folded: str,
) -> tuple[float, tuple[str, ...]]:
    if not query_tokens:
        return 0.0, ()
    fields = {
        "title": page.title,
        "aliases": " ".join(page.aliases),
        "tags": " ".join(page.tags),
        "description": page.description,
        "content": page.content,
    }
    score = 0.0
    matched: list[str] = []
    for name, content in fields.items():
        field_tokens = set(_tokenize(content))
        overlap = query_tokens & field_tokens
        if not overlap:
            continue
        matched.append(name)
        score += _FIELD_WEIGHTS[name] * len(overlap) / len(query_tokens)
    normalized_title = " ".join(page.title.casefold().split())
    normalized_aliases = {
        " ".join(alias.casefold().split()) for alias in page.aliases
    }
    if query_folded == normalized_title or query_folded in normalized_aliases:
        score += 2.0
    return score, tuple(matched)


def _semantic_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.casefold().replace("_", " ")))


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1_000))
