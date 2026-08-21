from __future__ import annotations

import pytest

from palmclaw_ubuntu.vehicle_wiki import VehicleWikiLink, VehicleWikiPage
from palmclaw_ubuntu.vehicle_wiki_retrieval import (
    VehicleWikiIndex,
    VehicleWikiTraversalBudget,
    VehicleWikiTraversalError,
    VehicleWikiTraversalLimitError,
    VehicleWikiTraversalSession,
)


class _WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def _pages() -> tuple[VehicleWikiPage, ...]:
    entity_id = "wiki:entity:gary"
    fact_id = "wiki:fact:volume"
    evidence_id = "wiki:evidence:volume"
    tool_id = "wiki:tool:set-volume"
    return (
        VehicleWikiPage(
            page_id=entity_id,
            page_type="entity",
            title="Gary",
            aliases=("driver Gary",),
            tags=("entity",),
            content="Gary's vehicle preferences.",
            source_ids=("fact:volume",),
            links=(
                VehicleWikiLink(
                    source_page_id=entity_id,
                    target_page_id=fact_id,
                    relation="has_fact",
                ),
            ),
        ),
        VehicleWikiPage(
            page_id=fact_id,
            page_type="fact",
            title="Gary: night volume",
            aliases=("night volume",),
            tags=("active", "preference"),
            content="At night, Gary prefers volume 20.",
            source_ids=("message:20",),
            links=(
                VehicleWikiLink(
                    source_page_id=fact_id,
                    target_page_id=evidence_id,
                    relation="supported_by",
                ),
            ),
        ),
        VehicleWikiPage(
            page_id=evidence_id,
            page_type="evidence",
            title="Evidence for night volume",
            tags=("evidence",),
            content="Actually, use 20 at night.",
            source_ids=("message:20",),
            links=(
                VehicleWikiLink(
                    source_page_id=evidence_id,
                    target_page_id=tool_id,
                    relation="related_capability",
                ),
            ),
        ),
        VehicleWikiPage(
            page_id=tool_id,
            page_type="tool",
            title="carcontrol_music_set_volume",
            tags=("tool",),
            content="Set music volume.",
            source_ids=("tool:carcontrol_music_set_volume",),
        ),
        VehicleWikiPage(
            page_id="wiki:summary_update:other",
            page_type="summary_update",
            title="Unrelated update",
            aliases=("other",),
            tags=("summary_update",),
            content="The phrase night volume appears only in content.",
            source_ids=("summary:other",),
        ),
    )


def _index() -> VehicleWikiIndex:
    return VehicleWikiIndex(_pages(), token_counter=_WordCounter())


def test_search_supports_structured_fields_and_reused_semantic_score() -> None:
    index = _index()

    lexical = index.search("night volume", top_k=5)
    assert lexical.hits[0].page_id == "wiki:fact:volume"
    assert "aliases" in lexical.hits[0].matched_fields

    semantic = index.search(
        "zzzzsemantic",
        top_k=1,
        semantic_scores={"wiki:tool:set-volume": 0.9},
    )
    assert [hit.page_id for hit in semantic.hits] == [
        "wiki:tool:set-volume"
    ]
    assert semantic.hits[0].matched_fields == ()


def test_direct_search_read_and_sufficient_finish() -> None:
    session = VehicleWikiTraversalSession(_index())

    search = session.search("Gary")
    entity_id = next(
        hit.page_id for hit in search.hits if hit.page_id == "wiki:entity:gary"
    )
    read = session.read((entity_id,))
    result = session.finish(
        evidence_sufficient=True,
        fallback_content="original summary",
    )

    assert read.rendered_tokens > 0
    assert "Gary" in result.content
    assert result.fallback_used is False
    assert result.trace.termination_reason == "evidence_sufficient"
    assert result.trace.search_count == 1
    assert result.trace.read_count == 1
    assert result.trace.hop_count == 0


def test_link_following_allows_two_hops_and_rejects_third() -> None:
    session = VehicleWikiTraversalSession(
        _index(),
        budget=VehicleWikiTraversalBudget(search_top_k=1),
    )
    session.search("Gary")
    session.read(("wiki:entity:gary",))
    session.read(("wiki:fact:volume",))
    session.read(("wiki:evidence:volume",))

    with pytest.raises(
        VehicleWikiTraversalLimitError,
        match="hop_budget_exhausted",
    ):
        session.read(("wiki:tool:set-volume",))

    assert session.trace.hop_count == 2
    assert session.trace.termination_reason == "hop_budget_exhausted"


def test_empty_search_patience_terminates_and_uses_fallback() -> None:
    session = VehicleWikiTraversalSession(_index())

    assert session.search("missing alpha").hits == ()
    assert session.search("missing beta").hits == ()
    with pytest.raises(VehicleWikiTraversalError, match="already terminated"):
        session.search("Gary")
    result = session.finish(
        evidence_sufficient=False,
        fallback_content="original summary",
    )

    assert result.content == "original summary"
    assert result.fallback_used is True
    assert result.trace.termination_reason == "empty_search_patience"
    assert result.trace.empty_search_count == 2


def test_read_enforces_batch_page_and_token_budgets() -> None:
    session = VehicleWikiTraversalSession(
        _index(),
        budget=VehicleWikiTraversalBudget(
            read_page_limit=2,
            read_batch_limit=2,
            token_budget=12,
        ),
    )

    read = session.read(("wiki:entity:gary", "wiki:fact:volume"))
    assert read.truncated is True
    assert session.trace.rendered_tokens <= 12
    with pytest.raises(
        VehicleWikiTraversalLimitError,
        match="read_page_budget_exhausted",
    ):
        session.read(("wiki:evidence:volume",))
    result = session.finish(
        evidence_sufficient=False,
        fallback_content="fallback",
    )
    assert result.content == "fallback"
    assert result.trace.termination_reason == "read_page_budget_exhausted"


def test_missing_page_and_tool_error_return_bounded_fallback() -> None:
    session = VehicleWikiTraversalSession(_index())
    read = session.read(("wiki:missing",))
    result = session.fail(
        RuntimeError("planned failure"),
        fallback_content="safe fallback",
    )

    assert read.pages == ()
    assert read.missing_page_ids == ("wiki:missing",)
    assert result.content == "safe fallback"
    assert result.pages == ()
    assert result.fallback_used is True
    assert result.trace.termination_reason == "tool_error"
    assert result.error == "RuntimeError: planned failure"
