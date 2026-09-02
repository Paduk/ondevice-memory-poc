from palmclaw_ubuntu.vehicle_bench.v2_quiz_export import (
    _memory_ref,
    _normalize_gold_calls,
)


def test_memory_ref_links_all_three_training_views() -> None:
    ref = _memory_ref(
        scenario_index=21,
        global_turn_index=101,
        turn_id="v01e1-turn-022",
        memory_sha256="a" * 64,
    )

    assert ref["checkpoint_id"] == "s021:memory:00101"
    assert ref["summary_sample_id"] == "s021:summary:00101"
    assert ref["patch_sample_id"] == "s021:patch:00101"
    assert ref["delta_sample_id"] == "s021:delta:00101"


def test_normalize_gold_calls_accepts_map_and_argument_list() -> None:
    calls = _normalize_gold_calls(
        (
            {
                "name": "carcontrol_light_set_brightness",
                "arguments": {"brightness": 4},
            },
            {
                "name": "carcontrol_seat_set_temperature",
                "arguments": (
                    {"name": "seat", "value": "driver"},
                    {"name": "temperature", "value": 21},
                ),
            },
        )
    )

    assert calls[0]["arguments"] == {"brightness": 4}
    assert calls[1]["arguments"] == {"seat": "driver", "temperature": 21}
