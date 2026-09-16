from memory_training.prepare_mem0_style_data import (
    AtomicFact,
    derive_scenario,
    experiment_split,
    parse_grouped_memory,
)


def _row(index: int, previous: str, operations: list[dict[str, str]]) -> dict:
    decision = "UPDATE" if operations else "NO_OP"
    return {
        "schema_version": "test",
        "sample_id": f"s015:patch:{index:05d}",
        "scenario_index": 15,
        "split": "train",
        "global_turn_index": index,
        "timestamp": "2026-01-01T00:00",
        "current_turn": {
            "speaker_id": "person-a",
            "speaker_name": "A",
            "text": f"turn {index}",
        },
        "input": {"previous_memory": previous},
        "target": {"decision": decision, "operations": operations},
    }


def test_parse_grouped_memory_supports_structured_and_natural_facts() -> None:
    memory = (
        "### A\n"
        "- [2026-01-01] tool.value; value=1\n"
        "- Prefers a quiet cabin.\n\n"
        "### shared-vehicle\n"
        "- Maintenance is due Friday."
    )
    assert parse_grouped_memory(memory) == (
        AtomicFact("A", "[2026-01-01] tool.value; value=1"),
        AtomicFact("A", "Prefers a quiet cabin."),
        AtomicFact("shared-vehicle", "Maintenance is due Friday."),
    )


def test_derive_scenario_maps_patch_to_stable_fact_crud() -> None:
    first = "### A\n- Prefers temperature 20."
    second = "### A\n- Prefers temperature 22."
    rows = [
        _row(0, "", []),
        _row(
            1,
            "",
            [{"op": "add", "target": "", "content": first}],
        ),
        _row(
            2,
            first,
            [
                {
                    "op": "replace",
                    "target": "- Prefers temperature 20.",
                    "content": "- Prefers temperature 22.",
                }
            ],
        ),
        _row(
            3,
            second,
            [
                {
                    "op": "delete",
                    "target": "### A\n- Prefers temperature 22.",
                    "content": "",
                }
            ],
        ),
    ]
    derived, versions = derive_scenario(rows)
    assert derived[0].events == []
    assert derived[1].events == [
        {"event": "ADD", "subject": "A", "text": "Prefers temperature 20."}
    ]
    assert derived[2].events == [
        {
            "event": "UPDATE",
            "id": "s015:f0001",
            "subject": "A",
            "text": "Prefers temperature 22.",
        }
    ]
    assert derived[3].events == [{"event": "DELETE", "id": "s015:f0001"}]
    assert derived[3].pure_delete_fallback
    assert derived[3].extracted_facts[0]["ephemeral_invalidation"] is True
    assert [item.version for item in versions] == [1, 2]
    assert versions[1].supersedes_fact_version_id == versions[0].fact_version_id


def test_multi_fact_add_is_split_into_atomic_records() -> None:
    content = "### A\n- Likes blue.\n- Likes green."
    rows = [
        _row(
            0,
            "",
            [{"op": "add", "target": "", "content": content}],
        )
    ]
    derived, versions = derive_scenario(rows)
    assert [event["event"] for event in derived[0].events] == ["ADD", "ADD"]
    assert [item.text for item in versions] == ["Likes blue.", "Likes green."]


def test_experiment_split_matches_current_paper_protocol() -> None:
    assert experiment_split(14) == "excluded"
    assert experiment_split(15) == "train"
    assert experiment_split(85) == "validation"
    assert experiment_split(86) == "test"
    assert experiment_split(101) == "train"
    assert experiment_split(111) == "validation"
    assert experiment_split(112) == "test"
    assert experiment_split(202) == "train"
