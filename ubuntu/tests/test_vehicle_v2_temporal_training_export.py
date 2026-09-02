from palmclaw_ubuntu.vehicle_bench.v2_temporal_training_export import (
    temporal_identity_key,
    temporalize_patch_row,
)


def _row(*, text: str, operation: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": "vehiclemembench-v2-patch-sft-v1",
        "sample_id": "s001:patch:00001",
        "scenario_index": 1,
        "split": "train",
        "turn_id": "e1-turn-001",
        "current_turn": {"text": text},
        "input": {"previous_memory": ""},
        "target": {
            "decision": "UPDATE",
            "reason_code": "NEW_VEHICLE_MEMORY",
            "reason": "Gold update.",
            "operations": [operation],
        },
        "provenance": {"after_memory_sha256": "a" * 64},
    }


def test_temporalize_conditional_gold_patch_is_lossless() -> None:
    line = (
        "- [2025-01-01 09:00] Maya Chen: "
        "carcontrol_wiper_set_speed.speed; value=3; "
        'context=(wiper="front"); condition=heavy rain'
    )
    converted = temporalize_patch_row(
        _row(
            text="In heavy rain, I prefer the front wipers at speed 3.",
            operation={"op": "add", "target": "", "content": line},
        )
    )

    operation = converted["target"]["operations"][0]
    assert operation["temporal_action"] == "conditional_upsert"
    assert operation["temporal_cue"] == ""
    assert operation["content"] == line
    assert operation["identity_key"].startswith("maya_chen.carcontrol_wiper")


def test_temporalize_explicit_until_condition_as_temporary_override() -> None:
    line = (
        "- [2025-01-01 09:00] shared-vehicle: "
        "carcontrol_airConditioner_set_circulation.circulation; value=\"inside\"; "
        "condition=until clean paved air is reached"
    )
    converted = temporalize_patch_row(
        _row(
            text="Use inside circulation until clean paved air is reached.",
            operation={"op": "add", "target": "", "content": line},
        )
    )

    operation = converted["target"]["operations"][0]
    assert operation["temporal_action"] == "temporary_override"
    assert operation["temporal_cue"] == "until clean paved air is reached"


def test_temporal_identity_excludes_condition_but_keeps_selector() -> None:
    first = (
        "- [2025-01-01 09:00] Maya: tool.setting; value=1; "
        'context=(seat="driver"); condition=rain'
    )
    second = (
        "- [2025-02-01 09:00] Maya: tool.setting; value=2; "
        'context=(seat="driver"); condition=snow'
    )

    assert temporal_identity_key(first) == temporal_identity_key(second)
