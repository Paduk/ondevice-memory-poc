from __future__ import annotations

from palmclaw_ubuntu.models import FactMemoryRecord, MemoryEvidence
from palmclaw_ubuntu.vehicle_wiki import (
    FactWikiSource,
    SummaryWikiVersion,
    VehicleWikiLink,
    VehicleWikiPage,
    VehicleWikiValidationError,
    build_fact_wiki_pages,
    build_summary_wiki_pages,
    validate_vehicle_wiki,
    vehicle_wiki_fingerprint,
)


class _WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def _record(
    record_id: str,
    *,
    value: object,
    version: int,
    status: str,
    supersedes_id: str | None = None,
) -> FactMemoryRecord:
    timestamp = f"2026-08-{version:02d}T00:00:00+00:00"
    return FactMemoryRecord(
        id=record_id,
        session_id="session",
        record_key="gary:media_volume",
        user_id="gary",
        entity_id="gary",
        predicate="media_volume",
        identity_conditions={},
        applicability={"time_of_day": "night"},
        capability_hints=("media",),
        value=value,
        memory_type="preference",
        status=status,
        confidence=0.95,
        version=version,
        supersedes_id=supersedes_id,
        merged_into_id=None,
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_summary_builder_is_deterministic_and_keeps_version_deltas() -> None:
    versions = (
        SummaryWikiVersion(
            version_id="summary:1",
            content="Gary prefers Bluetooth.",
            created_at="2026-08-01T00:00:00+00:00",
        ),
        SummaryWikiVersion(
            version_id="summary:2",
            content="Gary prefers Bluetooth. At night, volume is 20.",
            created_at="2026-08-02T00:00:00+00:00",
            source_ids=("batch:2",),
        ),
        SummaryWikiVersion(
            version_id="summary:3",
            content="Gary prefers Bluetooth. At night, volume is 20.",
            created_at="2026-08-03T00:00:00+00:00",
        ),
    )

    pages = build_summary_wiki_pages(
        namespace="scenario:6",
        versions=versions,
        token_counter=_WordCounter(),
    )
    reversed_pages = build_summary_wiki_pages(
        namespace="scenario:6",
        versions=tuple(reversed(versions)),
        token_counter=_WordCounter(),
    )

    assert vehicle_wiki_fingerprint(pages) == vehicle_wiki_fingerprint(
        reversed_pages
    )
    assert [page.page_type for page in pages].count("summary_overview") == 1
    update_pages = [page for page in pages if page.page_type == "summary_update"]
    assert len(update_pages) == 2
    assert any(page.content == "At night, volume is 20." for page in update_pages)
    assert all(page.source_ids for page in pages)
    report = validate_vehicle_wiki(pages, token_counter=_WordCounter())
    assert report.page_count == 3
    assert report.link_count == 3


def test_fact_builder_projects_provenance_versions_and_tool_links() -> None:
    old = _record("fact:old", value=30, version=1, status="superseded")
    current = _record(
        "fact:current",
        value=20,
        version=2,
        status="active",
        supersedes_id=old.id,
    )
    sources = {
        old.id: (
            MemoryEvidence(message_id=10, quote="Set the volume to 30."),
        ),
        current.id: (
            FactWikiSource(message_id=20, quote="Actually, use 20 at night."),
        ),
    }

    pages = build_fact_wiki_pages(
        namespace="scenario:6",
        records=(old, current),
        sources_by_record=sources,
        tools_by_capability={"media": ("carcontrol_music_set_volume",)},
        token_counter=_WordCounter(),
    )
    reversed_pages = build_fact_wiki_pages(
        namespace="scenario:6",
        records=(current, old),
        sources_by_record=sources,
        tools_by_capability={"media": ("carcontrol_music_set_volume",)},
        token_counter=_WordCounter(),
    )

    assert vehicle_wiki_fingerprint(pages) == vehicle_wiki_fingerprint(
        reversed_pages
    )
    assert [page.page_type for page in pages].count("entity") == 1
    assert [page.page_type for page in pages].count("fact") == 2
    assert [page.page_type for page in pages].count("evidence") == 2
    assert [page.page_type for page in pages].count("condition") == 1
    assert [page.page_type for page in pages].count("capability") == 1
    assert [page.page_type for page in pages].count("tool") == 1
    relations = {
        link.relation for page in pages for link in page.links
    }
    assert {
        "about_entity",
        "applies_under",
        "has_fact",
        "implemented_by_tool",
        "related_capability",
        "supported_by",
        "supersedes",
    } <= relations
    assert all(page.source_ids for page in pages if page.page_type == "fact")
    validate_vehicle_wiki(pages, token_counter=_WordCounter())


def test_validator_rejects_dangling_link() -> None:
    page = VehicleWikiPage(
        page_id="wiki:fact:a",
        page_type="fact",
        title="Fact A",
        source_ids=("message:1",),
        links=(
            VehicleWikiLink(
                source_page_id="wiki:fact:a",
                target_page_id="wiki:fact:missing",
                relation="supersedes",
            ),
        ),
    )

    try:
        validate_vehicle_wiki((page,), token_counter=_WordCounter())
    except VehicleWikiValidationError as exc:
        assert "Dangling Wiki link" in str(exc)
    else:
        raise AssertionError("dangling link must be rejected")


def test_validator_rejects_cycle_in_ordered_relation() -> None:
    page_a = VehicleWikiPage(
        page_id="wiki:fact:a",
        page_type="fact",
        title="Fact A",
        source_ids=("message:1",),
        links=(
            VehicleWikiLink(
                source_page_id="wiki:fact:a",
                target_page_id="wiki:fact:b",
                relation="supersedes",
            ),
        ),
    )
    page_b = VehicleWikiPage(
        page_id="wiki:fact:b",
        page_type="fact",
        title="Fact B",
        source_ids=("message:2",),
        links=(
            VehicleWikiLink(
                source_page_id="wiki:fact:b",
                target_page_id="wiki:fact:a",
                relation="supersedes",
            ),
        ),
    )

    try:
        validate_vehicle_wiki(
            (page_a, page_b),
            token_counter=_WordCounter(),
        )
    except VehicleWikiValidationError as exc:
        assert "contains a cycle" in str(exc)
    else:
        raise AssertionError("ordered relation cycle must be rejected")


def test_validator_rejects_missing_provenance_and_oversized_content() -> None:
    missing_source = VehicleWikiPage(
        page_id="wiki:summary_update:a",
        page_type="summary_update",
        title="Update",
        content="one two",
    )
    oversized = VehicleWikiPage(
        page_id="wiki:fact:b",
        page_type="fact",
        title="Fact",
        content="one two three",
        source_ids=("message:1",),
    )

    for page, expected in (
        (missing_source, "lacks source provenance"),
        (oversized, "exceeds 2 tokens"),
    ):
        try:
            validate_vehicle_wiki(
                (page,),
                max_page_tokens=2,
                token_counter=_WordCounter(),
            )
        except VehicleWikiValidationError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid page must be rejected")


def test_builders_enforce_page_token_budget() -> None:
    pages = build_summary_wiki_pages(
        namespace="scenario:6",
        versions=(
            SummaryWikiVersion(
                version_id="summary:1",
                content="one two three four five",
                created_at="2026-08-01T00:00:00+00:00",
            ),
        ),
        max_page_tokens=3,
        token_counter=_WordCounter(),
    )

    assert all(_WordCounter().count(page.content) <= 3 for page in pages)
