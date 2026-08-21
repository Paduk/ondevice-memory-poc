from __future__ import annotations

from dataclasses import replace

from palmclaw_ubuntu.models import (
    FactMemoryRecord,
    FactQueryContext,
    ToolDefinition,
)
from palmclaw_ubuntu.tool_memory_schema import build_tool_memory_ontology
from palmclaw_ubuntu.vehicle_fact_wiki import build_fact_wiki_runtime
from palmclaw_ubuntu.vehicle_wiki import FactWikiSource, build_fact_wiki_pages


def _record(
    record_id: str,
    *,
    entity: str,
    predicate: str,
    value: int,
    capability: str,
    status: str = "active",
    version: int = 1,
    supersedes_id: str | None = None,
) -> FactMemoryRecord:
    return FactMemoryRecord(
        id=record_id,
        session_id="session",
        record_key=f"key-{record_id}",
        user_id="vehicle-user",
        entity_id=entity,
        predicate=predicate,
        identity_conditions={},
        applicability={},
        capability_hints=(capability,),
        value=value,
        memory_type="preference",
        status=status,
        confidence=0.95,
        version=version,
        supersedes_id=supersedes_id,
        merged_into_id=None,
        created_at=f"2026-01-0{version}T00:00:00Z",
        updated_at=f"2026-01-0{version}T00:00:00Z",
    )


def _records() -> tuple[FactMemoryRecord, ...]:
    old = _record(
        "old-hud",
        entity="gary",
        predicate="hud_brightness",
        value=5,
        capability="hud brightness",
        status="superseded",
    )
    current = _record(
        "current-hud",
        entity="gary",
        predicate="hud_brightness",
        value=8,
        capability="hud brightness",
        version=2,
        supersedes_id=old.id,
    )
    volume = _record(
        "gary-volume",
        entity="gary",
        predicate="music_volume",
        value=20,
        capability="music volume",
    )
    other_entity = _record(
        "justin-volume",
        entity="justin",
        predicate="music_volume",
        value=30,
        capability="music volume",
    )
    return old, current, volume, other_entity


def _ontology():
    return build_tool_memory_ontology(
        (
            ToolDefinition(
                name="carcontrol_HUD_set_brightness_level",
                description="Set HUD brightness.",
                parameters={"type": "object", "properties": {}},
                timeout_seconds=2,
                side_effect="simulator_state",
                retry_safety="unsafe",
            ),
            ToolDefinition(
                name="carcontrol_music_set_volume",
                description="Set music volume.",
                parameters={"type": "object", "properties": {}},
                timeout_seconds=2,
                side_effect="simulator_state",
                retry_safety="unsafe",
            ),
        )
    )


def _trace() -> dict:
    return {
        "candidates": [
            {
                "record_id": "current-hud",
                "rank": 1,
                "embedding_score": 0.9,
                "selected": True,
                "exclusion_reason": None,
            },
            {
                "record_id": "gary-volume",
                "rank": 2,
                "embedding_score": 0.7,
                "selected": False,
                "exclusion_reason": "top_k",
            },
            {
                "record_id": "justin-volume",
                "rank": 3,
                "embedding_score": 0.6,
                "selected": False,
                "exclusion_reason": "top_k",
            },
        ]
    }


def _runtime(*, volume_exclusion: str = "top_k"):
    trace = _trace()
    trace["candidates"][1]["exclusion_reason"] = volume_exclusion
    records = _records()
    return build_fact_wiki_runtime(
        namespace="fact-wiki-test",
        records=records,
        sources_by_record={
            record.id: (
                FactWikiSource(
                    message_id=index + 1,
                    quote=f"Evidence for {record.id}",
                ),
            )
            for index, record in enumerate(records)
        },
        seed_records=(records[1],),
        retrieval_trace=trace,
        query_context=FactQueryContext(entities=("gary",)),
        ontology=_ontology(),
        base_cache_key="base-cache",
    )


def test_fact_wiki_projects_versions_evidence_capabilities_and_tools() -> None:
    runtime = _runtime()
    page_types = {page.page_type for page in runtime.pages}

    assert {
        "entity",
        "fact",
        "evidence",
        "capability",
        "tool",
    } <= page_types
    assert len(runtime.seed_page_ids) == 1
    assert runtime.semantic_score_count == 3
    assert len(runtime.page_fingerprint) == 64
    assert runtime.cache_signature == _runtime().cache_signature


def test_linked_expansion_excludes_cross_entity_and_superseded_facts() -> None:
    runtime = _runtime()

    assert [record.id for record in runtime.expanded_records] == [
        "gary-volume"
    ]
    assert all(record.status == "active" for record in runtime.expanded_records)
    assert all(
        record.entity_id == "gary" for record in runtime.expanded_records
    )


def test_condition_mismatch_is_not_added_to_planner_expansion() -> None:
    runtime = _runtime(volume_exclusion="condition_mismatch")

    assert runtime.expanded_records == ()


def test_fact_wiki_search_reuses_existing_candidate_semantic_scores() -> None:
    runtime = _runtime()
    result = runtime.session.search("zzzzsemantic")

    assert result.hits[0].page_id == runtime.seed_page_ids[0]


def test_shared_conditions_merge_across_different_fact_predicates() -> None:
    records = tuple(
        replace(
            record,
            identity_conditions={"person": "gary"},
        )
        for record in _records()[1:3]
    )
    pages = build_fact_wiki_pages(
        namespace="shared-condition-test",
        records=records,
        sources_by_record={
            record.id: (
                FactWikiSource(
                    message_id=index + 1,
                    quote=f"Evidence for {record.id}",
                ),
            )
            for index, record in enumerate(records)
        },
    )
    conditions = [page for page in pages if page.page_type == "condition"]

    assert len(conditions) == 1
    assert conditions[0].title == "Vehicle Fact conditions"
    assert conditions[0].source_ids == tuple(sorted(record.id for record in records))
