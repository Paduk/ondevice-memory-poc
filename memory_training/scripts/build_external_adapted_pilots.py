"""Build HVP01-HVP02 and HVP05-HVP20 from disjoint licensed records."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ENVISIONED_SHA256 = (
    "dea52de35e5a30a7271aaf89c4e506419fc57464e83c201f18b93ac8d454d186"
)
AUDIO_TIER7_SHA256 = (
    "7be95ff2dd904a1cadb45afb323dde0a3b7c2574932718ed3743fcd7fc09cddc"
)
AUDIO_TIER6_SHA256 = (
    "fa5c0825d99c4f68e226480261befe9ce7aa33c9dfbb7ff1fe1fc82493bcb991"
)
AUDIO_TIER4_SHA256 = (
    "d430e801c9bba5db03b3d044ac626ebec35de1b8a4eb2e2beb4cdec0013e9271"
)
AUDIO_TIER5_SHA256 = (
    "d38e588dccb99ed3291c9467a29adcd71e4f1c4eb62c118fd482a8e9616a1850"
)


def update(
    suffix: str,
    op: str,
    slot_id: str,
    tool_name: str,
    argument_name: str,
    value: Any,
    context_arguments: dict[str, Any],
    condition: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "suffix": suffix,
        "update": {
            "op": op,
            "slot_id": slot_id,
            "tool_name": tool_name,
            "argument_name": argument_name,
            "value": value,
            "context_arguments": context_arguments,
            "condition": condition,
            "reason": reason,
        },
    }


def balanced_plan(
    scenario_index: int,
    driver: str,
    human_records: list[tuple[str, str, str]],
    noop_records: list[int | str],
    entries: list[tuple[int | str, dict[str, Any]]],
) -> dict[str, Any]:
    """Create a 25-human/25-synthetic plan with deterministic quiz coverage."""

    assert len(noop_records) == 3
    assert len(entries) == 10
    updates = {record_id: spec for record_id, spec in entries}

    def gold(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        item = spec["update"]
        arguments = dict(item["context_arguments"])
        arguments[item["argument_name"]] = item["value"]
        return item["tool_name"], arguments

    turn_quizzes = []
    for record_id, spec in entries[:4]:
        item = spec["update"]
        turn_quizzes.append(
            (
                record_id,
                f"Apply {driver}'s current remembered setting: {item['condition']}.",
                "state_shift",
                [item["slot_id"]],
                [gold(spec)],
            )
        )

    active: dict[str, dict[str, Any]] = {}
    replaced: set[str] = set()
    for _, spec in entries:
        item = spec["update"]
        if item["op"] == "replace":
            replaced.add(item["slot_id"])
        active[item["slot_id"]] = spec
    assert len(active) == 7
    active_items = list(active.items())
    final_quizzes = []
    for slot_id, spec in active_items[:5]:
        item = spec["update"]
        final_quizzes.append(
            (
                f"Apply {driver}'s current remembered setting: {item['condition']}.",
                "error_correction" if slot_id in replaced else "conditional_constraint",
                [slot_id],
                [gold(spec)],
            )
        )
    combined = active_items[5:]
    final_quizzes.append(
        (
            f"Apply both of {driver}'s remaining current settings under their stated conditions.",
            "conditional_constraint",
            [slot_id for slot_id, _ in combined],
            [gold(spec) for _, spec in combined],
        )
    )
    return {
        "scenario_index": scenario_index,
        "driver": driver,
        "human_records": human_records,
        "audio_records": noop_records + [record_id for record_id, _ in entries],
        "updates": updates,
        "turn_quizzes": turn_quizzes,
        "final_quizzes": final_quizzes,
    }


PLANS: dict[str, dict[str, Any]] = {
    "HVP01": balanced_plan(
        901,
        "Devon",
        [
            ("20", "search", "Finding information about an upcoming local event."),
            ("11", "weather", "Checking the weather before making plans."),
            ("7", "conversational", "Having a general voice-assistant conversation."),
        ],
        [1, 4, 13],
        [
            ("tier4:3959", update("Level two is my normal rear-right seat-cooling setting.", "add", "driver.normal.rear_right_seat_ventilation_speed", "carcontrol_seat_set_ventilation_speed", "speed", 2, {"seat": "rear_right"}, "for Devon's normal rear-right seat comfort", "The inferred source cooling action is made into an explicit persistent setting.")),
            ("tier6:1103", update("Eleven is my regular music volume in the car.", "add", "driver.normal.music_volume", "carcontrol_music_set_volume", "volume", 11, {}, "for Devon's normal music playback", "The corrected source volume is adapted into an explicit persistent in-car music level.")),
            ("tier5:4007", update("I normally keep my tool bag there, so open the front trunk whenever I need those tools.", "add", "driver.tools.front_trunk", "carcontrol_frontTrunk_switch", "switch", True, {}, "when Devon needs the tool bag", "The source front-trunk action is made into a recurring tool-access routine.")),
            ("tier4:4191", update("Auto is my normal cabin climate mode.", "add", "driver.normal.climate_mode", "carcontrol_airConditioner_set_mode", "mode", "auto", {"zone": "all"}, "for Devon's normal cabin climate", "The explicit source AUTO mode is made persistent.")),
            ("tier4:3994", update("Keep the driver window closed whenever rain is starting.", "add", "driver.rain.driver_window", "carcontrol_window_set_open_degree", "degree", 0, {"window": "driver"}, "when rain is starting during Devon's drive", "The source close-window action is retained under its explicit rain condition.")),
            ("tier5:4240", update("Inside-air circulation is my normal setting around restaurant exhaust.", "add", "driver.restaurant_exhaust.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "inside", {"zone": "all"}, "when restaurant exhaust bothers Devon", "The source recirculation request is retained under its restaurant-exhaust condition.")),
            ("tier5:4269", update("Keep the rear trunk closed whenever my bags need to stay out of sight.", "add", "driver.hidden_bags.rear_trunk", "carcontrol_trunk_switch", "switch", False, {}, "when Devon needs bags kept out of sight", "The source close-liftgate action is retained under its valuables condition.")),
            ("tier4:4222", update("Actually, all-zone defrost is my normal cabin climate mode now.", "replace", "driver.normal.climate_mode", "carcontrol_airConditioner_set_mode", "mode", "defrost", {"zone": "all"}, "for Devon's normal cabin climate", "The later source defrost action replaces AUTO under the same stored condition and selector.")),
            ("tier4:4226", update("Actually, use outside air when restaurant exhaust bothers me now.", "replace", "driver.restaurant_exhaust.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "outside", {"zone": "all"}, "when restaurant exhaust bothers Devon", "The later source fresh-air action replaces inside recirculation under the same condition.")),
            ("tier6:1105", update("Actually, fifty-eight is my regular in-car music volume now.", "replace", "driver.normal.music_volume", "carcontrol_music_set_volume", "volume", 58, {}, "for Devon's normal music playback", "The later corrected source volume replaces eleven under the same condition.")),
        ],
    ),
    "HVP02": balanced_plan(
        902,
        "Robin",
        [
            ("8", "music", "Choosing and playing music with a voice assistant."),
            ("28", "conversational", "Having a general voice-assistant conversation."),
            ("15", "weather", "Checking the weather before travel."),
        ],
        [17, 18, 21],
        [
            ("tier6:1081", update("Fifty-two is my regular music volume in the car.", "add", "driver.normal.music_volume", "carcontrol_music_set_volume", "volume", 52, {}, "for Robin's normal music playback", "The corrected source volume is adapted into an explicit persistent in-car music level.")),
            ("tier6:3961", update("Twenty-seven degrees is my normal rear-right temperature.", "add", "driver.normal.rear_right_temperature", "carcontrol_airConditioner_set_temperature", "temperature", 27, {"zone": "rear_right"}, "for Robin's normal rear-right climate", "The corrected source temperature and zone are made persistent.")),
            ("tier6:3973", update("Level one is my normal passenger-seat cooling setting; keep heat off only for this drive.", "add", "driver.normal.passenger_seat_ventilation_speed", "carcontrol_seat_set_ventilation_speed", "speed", 1, {"seat": "passenger"}, "for Robin's normal passenger-seat comfort", "The corrected cooling level is retained while the one-time heat request is excluded from memory.")),
            ("tier6:4011", update("I normally use it for documents, so open the front trunk whenever I need that compartment.", "add", "driver.documents.front_trunk", "carcontrol_frontTrunk_switch", "switch", True, {}, "when Robin needs stored documents", "The corrected source action is made into a recurring front-trunk routine.")),
            ("tier4:4256", update("Keep the rear trunk closed whenever it would obstruct my view.", "add", "driver.obstructed_view.rear_trunk", "carcontrol_trunk_switch", "switch", False, {}, "when the open rear trunk obstructs Robin's view", "The source close-liftgate action is retained under its visibility condition.")),
            ("tier6:4002", update("Keep all vehicle doors unlocked whenever I am ready to leave.", "add", "driver.departure.doors_locked", "carcontrol_door_set_locked", "locked", False, {"door": "all"}, "when Robin is ready to leave", "The corrected source unlock action is retained as a departure rule.")),
            ("tier6:4135", update("Detailed voice guidance is my normal navigation setting.", "add", "driver.normal.navigation_voice_mode", "carcontrol_navigation_set_voice_mode", "mode", "detailed", {}, "for Robin's normal navigation", "The corrected source unmute request is normalized to detailed VehicleMemBench guidance and made persistent.")),
            ("tier6:1083", update("Actually, twenty-five is my regular in-car music volume now.", "replace", "driver.normal.music_volume", "carcontrol_music_set_volume", "volume", 25, {}, "for Robin's normal music playback", "The later corrected source volume replaces fifty-two under the same condition.")),
            ("tier6:1089", update("Actually, seventy is my regular in-car music volume now.", "replace", "driver.normal.music_volume", "carcontrol_music_set_volume", "volume", 70, {}, "for Robin's normal music playback", "The later source volume replaces twenty-five under the same condition.")),
            ("tier6:1091", update("Actually, five is my regular in-car music volume now.", "replace", "driver.normal.music_volume", "carcontrol_music_set_volume", "volume", 5, {}, "for Robin's normal music playback", "The final source correction replaces seventy under the same condition.")),
        ],
    ),
    "HVP05": {
        "scenario_index": 905,
        "driver": "Casey",
        "human_records": [
            ("32", "open", "Planning a family holiday within a fixed budget."),
            ("44", "open", "Planning the quickest driving route to Chester Zoo."),
            ("45", "open", "Checking traffic on the commute to work."),
            ("21", "search", "Checking cinema times before deciding whether to book."),
        ],
        "audio_records": [2072, 2078, 2042, 2043, 2056, 2105, 4115, 4269, 4273],
        "updates": {
            2042: update(
                "I normally keep the music at 30, so let's make that my regular level.",
                "add",
                "driver.music.volume",
                "carcontrol_music_set_volume",
                "volume",
                30,
                {},
                "for Casey's normal music playback",
                "The source media-volume request is made persistent.",
            ),
            2043: update(
                "Actually, 75 works better for me now; use that as my regular music level.",
                "replace",
                "driver.music.volume",
                "carcontrol_music_set_volume",
                "volume",
                75,
                {},
                "for Casey's normal music playback",
                "The later source volume request replaces the stored value.",
            ),
            2056: update(
                "That's the place I normally head to on weekdays.",
                "add",
                "driver.weekday.destination",
                "carcontrol_navigation_navigate_to",
                "destination",
                "123 Main St",
                {},
                "for Casey's weekday travel",
                "The corrected source destination is made persistent.",
            ),
            2105: update(
                "I always want them locked when I park at the store.",
                "add",
                "driver.store_parking.doors_locked",
                "carcontrol_door_set_locked",
                "locked",
                True,
                {"door": "all"},
                "when Casey parks at the store",
                "The source locking request is made into a durable parking rule.",
            ),
            4115: update(
                "Work is my regular weekday destination now.",
                "replace",
                "driver.weekday.destination",
                "carcontrol_navigation_navigate_to",
                "destination",
                "Work",
                {},
                "for Casey's weekday travel",
                "The corrected source destination replaces the stored address.",
            ),
            4269: update(
                "I normally load shopping there, so that's the routine I want.",
                "add",
                "driver.shopping.front_trunk",
                "carcontrol_frontTrunk_switch",
                "switch",
                True,
                {},
                "when Casey loads shopping",
                "The corrected cargo request is made into a recurring routine.",
            ),
            4273: update(
                "That's how I like the sunroof on warm-weather drives.",
                "add",
                "driver.warm_drive.sunroof",
                "carcontrol_sunroof_set_open_degree",
                "degree",
                30,
                {},
                "for Casey's warm-weather drives",
                "The corrected source command is made into a durable sunroof preference.",
            ),
        },
        "turn_quizzes": [
            (2042, "Set the music the way Casey currently prefers it.", "state_shift", ["driver.music.volume"], [("carcontrol_music_set_volume", {"volume": 30})]),
            (2043, "Casey's regular music level has changed. Apply the current preference.", "error_correction", ["driver.music.volume"], [("carcontrol_music_set_volume", {"volume": 75})]),
            (2056, "Casey is leaving on the regular weekday trip. Start navigation.", "conditional_constraint", ["driver.weekday.destination"], [("carcontrol_navigation_navigate_to", {"destination": "123 Main St"})]),
            (2105, "Casey has parked at the store. Apply the remembered door rule.", "conditional_constraint", ["driver.store_parking.doors_locked"], [("carcontrol_door_set_locked", {"door": "all", "locked": True})]),
        ],
        "final_quizzes": [
            ("Set the music to Casey's current regular level.", "error_correction", ["driver.music.volume"], [("carcontrol_music_set_volume", {"volume": 75})]),
            ("Casey is leaving for the regular weekday destination. Start navigation.", "error_correction", ["driver.weekday.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Work"})]),
            ("Casey is parked at the store. Apply the remembered door rule.", "conditional_constraint", ["driver.store_parking.doors_locked"], [("carcontrol_door_set_locked", {"door": "all", "locked": True})]),
            ("Casey is loading shopping. Apply the front-trunk routine.", "conditional_constraint", ["driver.shopping.front_trunk"], [("carcontrol_frontTrunk_switch", {"switch": True})]),
            ("It is a warm drive. Apply Casey's sunroof preference.", "conditional_constraint", ["driver.warm_drive.sunroof"], [("carcontrol_sunroof_set_open_degree", {"degree": 30})]),
            ("Begin Casey's weekday trip with the remembered destination and music volume.", "conditional_constraint", ["driver.weekday.destination", "driver.music.volume"], [("carcontrol_navigation_navigate_to", {"destination": "Work"}), ("carcontrol_music_set_volume", {"volume": 75})]),
        ],
    },
    "HVP06": {
        "scenario_index": 906,
        "driver": "Riley",
        "human_records": [
            ("81", "open", "Booking flights, a familiar hotel, and a meal."),
            ("110", "open", "Comparing public-transport options and arrival time."),
            ("196", "open", "Finding 24-hour service stations along a long trip."),
            ("132", "open", "Checking driving time and traffic for Blackpool."),
        ],
        "audio_records": [2081, 2090, 2057, 2117, 2119, 2135, 4117, 3989, 4233],
        "updates": {
            2057: update(
                "When I start navigation without naming a place, that's where I usually mean.",
                "add",
                "driver.default.destination",
                "carcontrol_navigation_navigate_to",
                "destination",
                "221B Baker Street, London",
                {},
                "when Riley starts navigation without specifying a destination",
                "The explicit source destination is adapted into a default-destination rule.",
            ),
            2117: update(
                "On cold drives I normally only clear the front glass.",
                "add",
                "driver.cold_drive.defrost",
                "carcontrol_airConditioner_set_mode",
                "mode",
                "defrost",
                {"zone": "front"},
                "for Riley's cold drives",
                "The front-defrost request is made into a durable cold-weather setting.",
            ),
            2119: update(
                "Inside air is the setting I normally use for cabin air quality.",
                "add",
                "driver.air_quality.circulation",
                "carcontrol_airConditioner_set_circulation",
                "circulation",
                "inside",
                {"zone": "all"},
                "for Riley's normal cabin-air setting",
                "The source recirculation choice is made persistent.",
            ),
            2135: update(
                "My bags normally go in the rear, so use that one whenever I'm loading up.",
                "add",
                "driver.luggage.rear_trunk",
                "carcontrol_trunk_switch",
                "switch",
                True,
                {},
                "when Riley loads bags",
                "The source rear-liftgate action is made into a recurring routine.",
            ),
            4117: update(
                "If I don't name a place from now on, take me Home.",
                "replace",
                "driver.default.destination",
                "carcontrol_navigation_navigate_to",
                "destination",
                "Home",
                {},
                "when Riley starts navigation without specifying a destination",
                "The source correction replaces the previous default destination.",
            ),
            3989: update(
                "Fresh outside air suits me better now; make that my normal cabin-air setting.",
                "replace",
                "driver.air_quality.circulation",
                "carcontrol_airConditioner_set_circulation",
                "circulation",
                "outside",
                {"zone": "all"},
                "for Riley's normal cabin-air setting",
                "The source Fresh Air correction replaces inside recirculation.",
            ),
            4233: update(
                "From now on I want both front and rear cleared on cold drives.",
                "replace",
                "driver.cold_drive.defrost",
                "carcontrol_airConditioner_set_mode",
                "mode",
                "defrost",
                {"zone": "all"},
                "for Riley's cold drives",
                "The later source request broadens defrost from front only to all zones.",
            ),
        },
        "turn_quizzes": [
            (2057, "Riley says to start navigation without naming a place. Use the default destination.", "state_shift", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "221B Baker Street, London"})]),
            (2117, "It is a cold drive. Clear the glass the way Riley currently prefers.", "conditional_constraint", ["driver.cold_drive.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "front", "mode": "defrost"})]),
            (2119, "Apply Riley's current normal cabin-air setting.", "state_shift", ["driver.air_quality.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "inside"})]),
            (2135, "Riley is loading bags. Open the cargo area normally used for them.", "conditional_constraint", ["driver.luggage.rear_trunk"], [("carcontrol_trunk_switch", {"switch": True})]),
        ],
        "final_quizzes": [
            ("Riley starts navigation without naming a place. Use the current default.", "error_correction", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Home"})]),
            ("It is a cold drive. Clear the glass using Riley's current preference.", "error_correction", ["driver.cold_drive.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "all", "mode": "defrost"})]),
            ("Apply Riley's current normal cabin-air setting.", "error_correction", ["driver.air_quality.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"})]),
            ("Riley is loading bags. Open the cargo area normally used for them.", "conditional_constraint", ["driver.luggage.rear_trunk"], [("carcontrol_trunk_switch", {"switch": True})]),
            ("On a cold trip, start Riley's default route and clear the glass as preferred.", "conditional_constraint", ["driver.default.destination", "driver.cold_drive.defrost"], [("carcontrol_navigation_navigate_to", {"destination": "Home"}), ("carcontrol_airConditioner_set_mode", {"zone": "all", "mode": "defrost"})]),
            ("Start Riley's default route and apply the normal cabin-air setting.", "conditional_constraint", ["driver.default.destination", "driver.air_quality.circulation"], [("carcontrol_navigation_navigate_to", {"destination": "Home"}), ("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"})]),
        ],
    },
    "HVP07": {
        "scenario_index": 907,
        "driver": "Morgan",
        "human_records": [
            ("182", "open", "Booking a go-karting trip and setting a reminder."),
            ("93", "open", "Planning a bus-and-train journey with delay alerts."),
            ("134", "open", "Locating a nearby car dealership."),
        ],
        "audio_records": [2094, 2143, 2066, 2137, 2120, 2046, 4237, 4275, 4009],
        "updates": {
            2066: update(
                "That's how I want navigation to work from now on.",
                "add",
                "driver.navigation.voice_mode",
                "carcontrol_navigation_set_voice_mode",
                "mode",
                "mute",
                {},
                "for Morgan's normal navigation",
                "The explicit source mute request is made persistent.",
            ),
            2137: update(
                "That's my normal opening on warm drives.",
                "add",
                "driver.warm_drive.sunroof",
                "carcontrol_sunroof_set_open_degree",
                "degree",
                50,
                {},
                "for Morgan's warm drives",
                "The source half-open request is made persistent.",
            ),
            2120: update(
                "Using the car's own air is generally how I prefer the cabin set.",
                "add",
                "driver.cabin_air.circulation",
                "carcontrol_airConditioner_set_circulation",
                "circulation",
                "inside",
                {"zone": "all"},
                "for Morgan's normal cabin-air setting",
                "The source intent is made into a durable recirculation preference.",
            ),
            2046: update(
                "If I've selected a playlist, I normally want it playing.",
                "add",
                "driver.playlist.music",
                "carcontrol_music_switch",
                "switch",
                True,
                {},
                "when Morgan has selected a playlist",
                "The source resume-playback command is made into a recurring playlist preference.",
            ),
            4237: update(
                "Outside air suits me better now; make that my normal cabin setting.",
                "replace",
                "driver.cabin_air.circulation",
                "carcontrol_airConditioner_set_circulation",
                "circulation",
                "outside",
                {"zone": "all"},
                "for Morgan's normal cabin-air setting",
                "The source correction replaces inside recirculation with outside air.",
            ),
            4275: update(
                "Eighteen percent feels better now; that's my warm-weather position.",
                "replace",
                "driver.warm_drive.sunroof",
                "carcontrol_sunroof_set_open_degree",
                "degree",
                18,
                {},
                "for Morgan's warm drives",
                "The source parameter correction replaces the half-open setting.",
            ),
            4009: update(
                "I normally put equipment in the front trunk, so use that whenever I'm loading gear.",
                "add",
                "driver.equipment.front_trunk",
                "carcontrol_frontTrunk_switch",
                "switch",
                True,
                {},
                "when Morgan loads equipment",
                "The source tool correction is made into a recurring front-trunk routine.",
            ),
        },
        "turn_quizzes": [
            (2066, "Morgan wants to follow the map visually. Set the navigation guidance accordingly.", "state_shift", ["driver.navigation.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
            (2137, "It is a warm drive. Set the sunroof the way Morgan currently prefers.", "conditional_constraint", ["driver.warm_drive.sunroof"], [("carcontrol_sunroof_set_open_degree", {"degree": 50})]),
            (2120, "Apply Morgan's current normal cabin-air setting.", "state_shift", ["driver.cabin_air.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "inside"})]),
            (2046, "Morgan has selected a playlist. Apply the normal playback preference.", "conditional_constraint", ["driver.playlist.music"], [("carcontrol_music_switch", {"switch": True})]),
        ],
        "final_quizzes": [
            ("Morgan wants to follow the map visually. Set the navigation guidance accordingly.", "state_shift", ["driver.navigation.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
            ("It is a warm drive. Set the sunroof the way Morgan now prefers.", "error_correction", ["driver.warm_drive.sunroof"], [("carcontrol_sunroof_set_open_degree", {"degree": 18})]),
            ("Apply Morgan's normal cabin-air setting.", "error_correction", ["driver.cabin_air.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"})]),
            ("Morgan has selected a playlist. Apply the normal playback preference.", "conditional_constraint", ["driver.playlist.music"], [("carcontrol_music_switch", {"switch": True})]),
            ("Morgan is loading equipment. Apply the front-trunk routine.", "conditional_constraint", ["driver.equipment.front_trunk"], [("carcontrol_frontTrunk_switch", {"switch": True})]),
            ("On a warm drive with a playlist selected, apply Morgan's sunroof and playback preferences.", "conditional_constraint", ["driver.warm_drive.sunroof", "driver.playlist.music"], [("carcontrol_sunroof_set_open_degree", {"degree": 18}), ("carcontrol_music_switch", {"switch": True})]),
        ],
    },
}


PLANS.update(
    {
        "HVP08": balanced_plan(
            908,
            "Avery",
            [
                ("35", "search", "Choosing a film for the evening."),
                ("38", "weather", "Checking weather for a trip to Napoli."),
                ("57", "search", "Checking a local cinema showing time."),
            ],
            [2024, 2088, 2145],
            [
                ("tier5:3963", update("Eighteen degrees is my regular driver setting.", "add", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 18, {"zone": "driver"}, "for Avery's normal driving", "The explicit temperature is made persistent.")),
                ("tier5:4003", update("Sixty percent is my usual passenger-front window position.", "add", "driver.normal.passenger_window", "carcontrol_window_set_open_degree", "degree", 60, {"window": "passenger"}, "for Avery's normal driving", "The explicit window position is made persistent.")),
                ("tier5:4111", update("Use Riverside Dog Park as my default destination.", "add", "driver.default.destination", "carcontrol_navigation_navigate_to", "destination", "Riverside Dog Park on Kennel Lane", {}, "when Avery starts navigation without naming a destination", "The explicit destination becomes a default.")),
                ("tier5:4239", update("When outside food smells bother me, inside circulation is my usual choice.", "add", "driver.food_odor.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "inside", {"zone": "all"}, "when outside food odors bother Avery", "The source recirculation request becomes a persistent rule under the same odor condition.")),
                ("tier4:4096", update("Blue is my normal ambient-light color.", "add", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "blue", {}, "for Avery's normal ambient lighting", "The source color is stated explicitly and made persistent without dropping a brightness request.")),
                ("tier5:4272", update("I normally use the rear trunk when loading bulky items.", "add", "driver.bulky_items.rear_trunk", "carcontrol_trunk_switch", "switch", True, {}, "when Avery loads bulky items", "The source cargo request becomes a recurring routine.")),
                ("tier4:3988", update("Keep all doors locked whenever I park.", "add", "driver.vehicle_access.doors_locked", "carcontrol_door_set_locked", "locked", True, {"door": "all"}, "when Avery parks", "The source lock request becomes a durable parking rule.")),
                ("tier5:4005", update("Actually, ten percent is my regular passenger-front window position now.", "replace", "driver.normal.passenger_window", "carcontrol_window_set_open_degree", "degree", 10, {"window": "passenger"}, "for Avery's normal driving", "The later source value replaces the window position.")),
                ("tier5:4113", update("Actually, make Eastside Public Library my default destination now.", "replace", "driver.default.destination", "carcontrol_navigation_navigate_to", "destination", "Eastside Public Library on Maple Street", {}, "when Avery starts navigation without naming a destination", "The later source destination replaces the default.")),
                ("tier5:4002", update("Actually, keep all doors unlocked when I am about to leave now.", "replace", "driver.vehicle_access.doors_locked", "carcontrol_door_set_locked", "locked", False, {"door": "all"}, "when Avery is parked and about to leave", "The later source request replaces the parking lock rule and narrows the condition to departure.")),
            ],
        ),
        "HVP09": balanced_plan(
            909,
            "Jordan",
            [
                ("58", "search", "Planning a cinema visit."),
                ("59", "search", "Checking local film times."),
                ("57", "weather", "Choosing clothes for an Italy trip."),
                ("3", "search", "Asking what is playing at a local cinema."),
            ],
            [2123, 2125, 2127],
            [
                ("tier4:3951", update("Fan speed six is my regular cabin setting.", "add", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 6, {"zone": "all"}, "for Jordan's normal cabin setting", "The source fan level is stated explicitly and made persistent.")),
                ("tier5:4109", update("Red is the ambient-light color I want for a romantic cabin mood.", "add", "driver.romantic.ambient_color", "carcontrol_light_set_ambient_color", "color", "red", {}, "when Jordan wants a romantic cabin mood", "The source color is normalized to the equivalent VehicleMemBench color name and retained under the same romantic context.")),
                ("tier5:4271", update("I normally use the rear trunk for heavy shopping.", "add", "driver.heavy_shopping.rear_trunk", "carcontrol_trunk_switch", "switch", True, {}, "when Jordan loads heavy shopping", "The source trunk request becomes a recurring routine.")),
                ("tier4:4260", update("Fifteen percent is my normal sunroof position on bright days.", "add", "driver.bright_day.sunroof", "carcontrol_sunroof_set_open_degree", "degree", 15, {}, "on Jordan's bright-day drives", "The source sunroof value is stated explicitly and made persistent.")),
                ("tier4:4120", update("Muted guidance is my normal setting while listening to audiobooks.", "add", "driver.audiobook.voice_mode", "carcontrol_navigation_set_voice_mode", "mode", "mute", {}, "when Jordan listens to an audiobook", "The source mute request becomes conditional memory.")),
                ("tier4:3987", update("Keep all doors locked whenever I park.", "add", "driver.vehicle_access.doors_locked", "carcontrol_door_set_locked", "locked", True, {"door": "all"}, "when Jordan parks", "The source lock request becomes a durable parking rule.")),
                ("tier4:3965", update("Keep steering-wheel heat disabled as my normal setting.", "add", "driver.steering_wheel.heat_enabled", "carcontrol_steeringWheel_set_heating_enabled", "enabled", False, {}, "for Jordan's normal steering-wheel setting", "The source disable request is made persistent without inventing a weather condition.")),
                ("tier4:3953", update("Actually, fan speed eight is my regular cabin setting now.", "replace", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 8, {"zone": "all"}, "for Jordan's normal cabin setting", "The later source fan level replaces the stored value.")),
                ("tier5:4001", update("Actually, keep all doors unlocked when I am about to leave.", "replace", "driver.vehicle_access.doors_locked", "carcontrol_door_set_locked", "locked", False, {"door": "all"}, "when Jordan is parked and about to leave", "The later source request changes the stored access rule and its condition.")),
                ("tier5:3977", update("Actually, keep steering-wheel heat enabled on cold drives now.", "replace", "driver.steering_wheel.heat_enabled", "carcontrol_steeringWheel_set_heating_enabled", "enabled", True, {}, "for Jordan's cold drives", "The later cold-weather request replaces the normal disabled setting.")),
            ],
        ),
        "HVP10": balanced_plan(
            910,
            "Taylor",
            [
                ("74", "weather", "Discussing seasonal weather in Italy."),
                ("75", "search", "Finding a cinema showing for a recent film."),
                ("5", "search", "Booking seats for a Saturday cinema showing."),
            ],
            [2129, 2131, 2144],
            [
                ("tier5:3961", update("Twenty-three degrees is my regular passenger-zone setting.", "add", "driver.normal.passenger_temperature", "carcontrol_airConditioner_set_temperature", "temperature", 23, {"zone": "passenger"}, "for Taylor's normal passenger-zone climate", "The explicit temperature is made persistent.")),
                ("tier5:3965", update("Fan speed one is my regular cabin setting.", "add", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 1, {"zone": "all"}, "for Taylor's normal cabin setting", "The explicit fan level is made persistent.")),
                ("tier5:3979", update("Keep steering-wheel heat enabled whenever my hands feel cold.", "add", "driver.cold_hands.wheel_heat", "carcontrol_steeringWheel_set_heating_enabled", "enabled", True, {}, "when Taylor's hands feel cold", "The source heat request is retained under its explicit comfort condition.")),
                ("tier4:3991", update("When the kids find the rear cabin stuffy, set their rear-right window to sixty percent.", "add", "driver.kids_stuffy.rear_right_window", "carcontrol_window_set_open_degree", "degree", 60, {"window": "rear_right"}, "when Taylor's children find the rear cabin stuffy", "The source window value is attributed to the children and kept under the same stuffiness condition.")),
                ("tier5:4004", update("Sixty percent is my regular passenger-front window position.", "add", "driver.normal.passenger_window", "carcontrol_window_set_open_degree", "degree", 60, {"window": "passenger"}, "for Taylor's normal driving", "The explicit source window position is made persistent.")),
                ("tier5:4273", update("Ten percent is my normal sunroof vent position.", "add", "driver.venting.sunroof", "carcontrol_sunroof_set_open_degree", "degree", 10, {}, "when Taylor vents the cabin", "The explicit sunroof position is made persistent.")),
                ("tier5:4233", update("Use rear-only defrost whenever I need clear rear visibility.", "add", "driver.rear_visibility.defrost", "carcontrol_airConditioner_set_mode", "mode", "defrost", {"zone": "rear"}, "when Taylor needs to clear the rear glass", "The source rear-defrost request is retained as a rear-visibility rule.")),
                ("tier5:3967", update("Actually, fan speed seven is my regular cabin setting now.", "replace", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 7, {"zone": "all"}, "for Taylor's normal cabin setting", "The later source fan value replaces the stored level.")),
                ("tier5:4275", update("Actually, thirty percent is my normal sunroof vent position now.", "replace", "driver.venting.sunroof", "carcontrol_sunroof_set_open_degree", "degree", 30, {}, "when Taylor vents the cabin", "The later source position replaces the stored value.")),
                ("tier5:4006", update("Actually, ten percent is my regular passenger-front window position now.", "replace", "driver.normal.passenger_window", "carcontrol_window_set_open_degree", "degree", 10, {"window": "passenger"}, "for Taylor's normal driving", "The later source position replaces the stored passenger-window value.")),
            ],
        ),
        "HVP11": balanced_plan(
            911,
            "Quinn",
            [
                ("84", "search", "Checking cinema times in London."),
                ("85", "search", "Choosing a film at a local cinema."),
                ("81", "search", "Checking cinema opening times."),
                ("16", "search", "Opening local cinema listings."),
            ],
            [2095, 2098, 2101],
            [
                ("tier4:3961", update("Level one is my regular driver-seat heat setting.", "add", "driver.normal.seat_heat", "carcontrol_seat_set_heating_level", "level", 1, {"seat": "driver"}, "for Quinn's normal driver-seat comfort", "The source heat level is stated explicitly and made persistent.")),
                ("tier5:3962", update("Twenty-three degrees is my regular passenger-zone setting.", "add", "driver.normal.passenger_temperature", "carcontrol_airConditioner_set_temperature", "temperature", 23, {"zone": "passenger"}, "for Quinn's normal passenger-zone climate", "The explicit source temperature is made persistent.")),
                ("tier4:4095", update("Blue is my normal ambient-light color.", "add", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "blue", {}, "for Quinn's normal ambient lighting", "The source color is normalized to the equivalent VehicleMemBench color name and made persistent.")),
                ("tier4:4119", update("Muted guidance is my normal setting while listening to audiobooks.", "add", "driver.audiobook.voice_mode", "carcontrol_navigation_set_voice_mode", "mode", "mute", {}, "when Quinn listens to an audiobook", "The source mute request becomes conditional memory.")),
                ("tier4:3952", update("Fan speed six is my regular cabin setting.", "add", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 6, {"zone": "all"}, "for Quinn's normal cabin setting", "The source fan level is stated explicitly and made persistent.")),
                ("tier4:3990", update("Keep all doors locked whenever I park.", "add", "driver.vehicle_access.doors_locked", "carcontrol_door_set_locked", "locked", True, {"door": "all"}, "when Quinn parks", "The source lock request becomes a durable parking rule.")),
                ("tier4:4100", update("Use Home as my default destination.", "add", "driver.default.destination", "carcontrol_navigation_navigate_to", "destination", "Home", {}, "when Quinn starts navigation without naming a destination", "The source destination becomes a persistent default.")),
                ("tier4:3954", update("Actually, fan speed eight is my regular cabin setting now.", "replace", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 8, {"zone": "all"}, "for Quinn's normal cabin setting", "The later source fan level replaces the stored value.")),
                ("tier5:4208", update("Actually, keep all doors unlocked when I am about to leave now.", "replace", "driver.vehicle_access.doors_locked", "carcontrol_door_set_locked", "locked", False, {"door": "all"}, "when Quinn is parked and about to leave", "The later source request changes the stored access rule and its condition.")),
                ("tier5:4114", update("Actually, make Eastside Public Library my default destination now.", "replace", "driver.default.destination", "carcontrol_navigation_navigate_to", "destination", "Eastside Public Library on Maple Street", {}, "when Quinn starts navigation without naming a destination", "The later source destination replaces Home.")),
            ],
        ),
        "HVP12": balanced_plan(
            912,
            "Sydney",
            [
                ("86", "search", "Checking whether a local cinema is open."),
                ("87", "weather", "Checking a week of weather in Rome."),
                ("95", "search", "Checking local film times."),
                ("19", "search", "Checking afternoon listings at a local cinema."),
            ],
            [2110, 2112, 2114],
            [
                ("tier4:3948", update("Nineteen degrees is my regular driver-zone setting.", "add", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 19, {"zone": "driver"}, "for Sydney's normal driving", "The source temperature is stated explicitly and made persistent.")),
                ("tier4:4099", update("Home is my default destination.", "add", "driver.default.destination", "carcontrol_navigation_navigate_to", "destination", "Home", {}, "when Sydney starts navigation without naming a destination", "The source destination becomes a default.")),
                ("tier4:4081", update("Resume playback whenever I'm listening to an audiobook.", "add", "driver.audiobook.playback", "carcontrol_music_switch", "switch", True, {}, "when Sydney listens to an audiobook", "The source audiobook play request is retained under the same media condition.")),
                ("tier4:4121", update("Mute guidance whenever the navigation prompts feel too frequent.", "add", "driver.frequent_prompts.voice_mode", "carcontrol_navigation_set_voice_mode", "mode", "mute", {}, "when navigation prompts feel too frequent to Sydney", "The source mute request is retained under its explicit frequency condition.")),
                ("tier4:4219", update("Use defrost for all glass whenever frost blocks visibility.", "add", "driver.frost.defrost", "carcontrol_airConditioner_set_mode", "mode", "defrost", {"zone": "all"}, "when frost blocks Sydney's visibility", "The source front-and-rear request becomes an all-zone rule.")),
                ("tier4:3966", update("Keep steering-wheel heat disabled as my normal setting.", "add", "driver.steering_wheel.heat_enabled", "carcontrol_steeringWheel_set_heating_enabled", "enabled", False, {}, "for Sydney's normal steering-wheel setting", "The source disable request is made persistent without inventing a cold-drive condition.")),
                ("tier5:4207", update("Keep all doors unlocked as my normal parked setting.", "add", "driver.parked.doors_locked", "carcontrol_door_set_locked", "locked", False, {"door": "all"}, "when Sydney is parked", "The source unlock request is made into an explicit parked-state preference.")),
                ("tier4:3963", update("Actually, keep steering-wheel heat enabled as my normal setting now.", "replace", "driver.steering_wheel.heat_enabled", "carcontrol_steeringWheel_set_heating_enabled", "enabled", True, {}, "for Sydney's normal steering-wheel setting", "The later source request replaces the normal disabled setting under the same condition.")),
                ("tier4:3989", update("Actually, keep all doors locked as my normal parked setting now.", "replace", "driver.parked.doors_locked", "carcontrol_door_set_locked", "locked", True, {"door": "all"}, "when Sydney is parked", "The later source request replaces the parked unlock rule under the same condition.")),
                ("tier5:4112", update("Actually, make Riverside Dog Park my default destination now.", "replace", "driver.default.destination", "carcontrol_navigation_navigate_to", "destination", "Riverside Dog Park on Kennel Lane", {}, "when Sydney starts navigation without naming a destination", "The later source destination replaces Home.")),
            ],
        ),
    }
)

PLANS.update(
    {
        "HVP17": balanced_plan(
            917,
            "Blair",
            [
                ("43", "conversational", "Choosing a podcast to help with sleep."),
                ("80", "weather", "Packing for mixed weather and mountain climbing in Italy."),
                ("51", "open", "Sending an arrival-time text message."),
            ],
            [1902, 2003, 2021],
            [
                ("tier5:4107", update("Keep pink as my normal ambient-light color; the requested brightness is only for this drive.", "add", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "pink", {}, "for Blair's normal ambient lighting", "The source hex color is normalized to pink while its one-time brightness is excluded from durable memory.")),
                ("tier5:4203", update("Eighteen degrees is my regular cabin temperature; cool mode is only for this drive.", "add", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 18, {"zone": "all"}, "for Blair's normal cabin climate", "The source temperature is retained while the unsupported one-time HVAC mode is not made persistent.")),
                ("tier4:3992", update("When the children find the rear cabin stuffy, set the rear-right window to sixty percent.", "add", "driver.kids_stuffy.rear_right_window", "carcontrol_window_set_open_degree", "degree", 60, {"window": "rear_right"}, "when Blair's children find the rear cabin stuffy", "The inferred source window action is retained under its child-occupant condition.")),
                ("tier4:4101", update("Use Home as my default destination.", "add", "driver.default.destination", "carcontrol_navigation_navigate_to", "destination", "Home", {}, "when Blair starts navigation without naming a destination", "The inferred source destination becomes an explicit default.")),
                ("tier4:4221", update("Use front-and-rear defrost whenever the cabin windows are heavily fogged.", "add", "driver.heavy_fog.defrost", "carcontrol_airConditioner_set_mode", "mode", "defrost", {"zone": "all"}, "when Blair's cabin windows are heavily fogged", "The inferred source defrost scope becomes a persistent heavy-fog rule.")),
                ("tier4:4223", update("Use outside air whenever stale cabin air gives me a headache.", "add", "driver.stale_air.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "outside", {"zone": "all"}, "when stale cabin air gives Blair a headache", "The inferred fresh-air action is retained under the source stale-air condition.")),
                ("tier4:4257", update("Keep the rear trunk closed during normal driving.", "add", "driver.normal.rear_trunk", "carcontrol_trunk_switch", "switch", False, {}, "for Blair's normal driving", "The inferred source close action becomes an explicit normal-driving state.")),
                ("tier5:4110", update("Actually, red is my normal ambient-light color now.", "replace", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "red", {}, "for Blair's normal ambient lighting", "The later source color replaces pink; no unstated brightness preference is retained.")),
                ("tier6:4111", update("Actually, white is my normal ambient-light color now; the corrected brightness is only for this drive.", "replace", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "white", {}, "for Blair's normal ambient lighting", "The corrected later source color replaces red while brightness remains turn-local.")),
                ("tier5:4205", update("Actually, twenty-seven degrees is my regular cabin temperature now; heat mode is only for this drive.", "replace", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 27, {"zone": "all"}, "for Blair's normal cabin climate", "The later source temperature replaces eighteen degrees while HVAC mode remains turn-local.")),
            ],
        ),
        "HVP18": balanced_plan(
            918,
            "Harper",
            [
                ("88", "weather", "Preparing and sharing a holiday packing list."),
                ("36", "conversational", "Choosing a sleep cast to help with insomnia."),
                ("64", "alarm", "Planning a wake-up time for an early meeting."),
            ],
            [1951, 2019, 2027],
            [
                ("tier5:4108", update("Keep pink as my normal ambient-light color; the requested brightness is only for this drive.", "add", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "pink", {}, "for Harper's normal ambient lighting", "The source hex color is normalized to pink while its one-time brightness is excluded from durable memory.")),
                ("tier5:3973", update("Level two is my normal rear-right seat-cooling setting.", "add", "driver.normal.rear_right_seat_ventilation_speed", "carcontrol_seat_set_ventilation_speed", "speed", 2, {"seat": "rear_right"}, "for Harper's normal rear-right seat comfort", "The source cooling level and seat location are both retained as the persistent setting.")),
                ("tier6:4207", update("Twenty-four degrees is my regular cabin temperature; auto mode is only for this drive.", "add", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 24, {"zone": "all"}, "for Harper's normal cabin climate", "The corrected source temperature is retained while HVAC mode remains turn-local.")),
                ("tier4:4225", update("Use outside air whenever the side windows are difficult to see through because of fog.", "add", "driver.side_window_fog.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "outside", {"zone": "all"}, "when fog obscures Harper's side windows", "The inferred fresh-air action is retained under the source visibility condition.")),
                ("tier4:3995", update("Open the front trunk whenever it seems locked shut.", "add", "driver.stuck.front_trunk", "carcontrol_frontTrunk_switch", "switch", True, {}, "when Harper's front trunk seems locked shut", "The inferred source open action is retained under the same apparent-lock condition.")),
                ("tier4:4262", update("Keep the sunroof closed whenever cabin noise becomes distracting.", "add", "driver.noisy_cabin.sunroof", "carcontrol_sunroof_set_open_degree", "degree", 0, {}, "when cabin noise is distracting to Harper", "The inferred source close action is retained under its noise condition.")),
                ("tier5:4234", update("Use rear-only defrost whenever I need clear rear visibility.", "add", "driver.rear_visibility.defrost", "carcontrol_airConditioner_set_mode", "mode", "defrost", {"zone": "rear"}, "when Harper needs to clear the rear glass", "The source rear-defrost request is retained as a rear-visibility rule.")),
                ("tier6:4113", update("Actually, red is my normal ambient-light color now; the corrected brightness is only for this drive.", "replace", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "red", {}, "for Harper's normal ambient lighting", "The corrected later source color replaces pink while brightness remains turn-local.")),
                ("tier6:4209", update("Actually, twenty-one degrees is my regular cabin temperature now; the requested HVAC mode is only for this drive.", "replace", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 21, {"zone": "all"}, "for Harper's normal cabin climate", "The corrected later source temperature replaces twenty-four degrees while HVAC mode remains turn-local.")),
                ("tier4:4193", update("Actually, twenty-eight degrees is my regular cabin temperature now; heat mode is only for this drive.", "replace", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 28, {"zone": "all"}, "for Harper's normal cabin climate", "The later inferred source temperature replaces twenty-one degrees while HVAC mode remains turn-local.")),
            ],
        ),
        "HVP19": balanced_plan(
            919,
            "Emerson",
            [
                ("94", "open", "Ordering a birthday card and personalized chocolates."),
                ("67", "alarm", "Choosing a loud alarm tone for an important meeting."),
                ("55", "conversational", "Using a quiet sleep playlist to help with insomnia."),
            ],
            [1906, 2034, 2104],
            [
                ("tier4:4098", update("White is my normal ambient-light color; maximum brightness is only for finding this item.", "add", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "white", {}, "for Emerson's normal ambient lighting", "The inferred source color is made persistent while its task-specific brightness remains turn-local.")),
                ("tier5:3974", update("Level two is my normal rear-right seat-cooling setting.", "add", "driver.normal.rear_right_seat_ventilation_speed", "carcontrol_seat_set_ventilation_speed", "speed", 2, {"seat": "rear_right"}, "for Emerson's normal rear-right seat comfort", "The source cooling level and seat location are both retained as the persistent setting.")),
                ("tier6:4239", update("Keep inside-air circulation as my normal cabin-air setting.", "add", "driver.normal.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "inside", {"zone": "all"}, "for Emerson's normal cabin-air setting", "The corrected source recirculation-on request maps directly to inside-air circulation and is made persistent.")),
                ("tier5:4235", update("Use front-only defrost whenever condensation obscures the windshield.", "add", "driver.front_condensation.defrost", "carcontrol_airConditioner_set_mode", "mode", "defrost", {"zone": "front"}, "when condensation obscures Emerson's windshield", "The source front-defrost request is retained under its windshield condition.")),
                ("tier4:3997", update("Open the front trunk whenever I need to retrieve my coat from it.", "add", "driver.coat.front_trunk", "carcontrol_frontTrunk_switch", "switch", True, {}, "when Emerson needs to retrieve a coat from the front trunk", "The inferred source open action is retained under the same coat-retrieval condition.")),
                ("tier6:4003", update("Keep all doors unlocked as my normal parked-access setting.", "add", "driver.parked.doors_locked", "carcontrol_door_set_locked", "locked", False, {"door": "all"}, "when Emerson is parked", "The corrected source unlock action is made into an explicit parked-state preference.")),
                ("tier4:4075", update("Whenever I want background noise, turn on the radio.", "add", "driver.background_noise.radio", "carcontrol_radio_switch", "switch", True, {}, "when Emerson wants background noise", "The inferred Radio source selection is represented by the equivalent VehicleMemBench radio-on action.")),
                ("tier6:4114", update("Actually, red is my normal ambient-light color now; the corrected brightness is only for this drive.", "replace", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "red", {}, "for Emerson's normal ambient lighting", "The corrected later source color replaces white while brightness remains turn-local.")),
                ("tier6:4112", update("Actually, white is my normal ambient-light color again; the corrected brightness is only for this drive.", "replace", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "white", {}, "for Emerson's normal ambient lighting", "The corrected later source color replaces red while brightness remains turn-local.")),
                ("tier5:4237", update("Actually, keep outside-air circulation as my normal cabin-air setting now.", "replace", "driver.normal.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "outside", {"zone": "all"}, "for Emerson's normal cabin-air setting", "The later source recirculation-off request maps directly to outside air and replaces the prior inside-air setting.")),
            ],
        ),
        "HVP20": balanced_plan(
            920,
            "Finley",
            [
                ("102", "search", "Booking evening cinema tickets around dinner plans."),
                ("71", "conversational", "Preparing a quiet room and relaxing music for sleep."),
                ("63", "music", "Choosing upbeat music while cooking dinner."),
            ],
            [1907, 2006, 2139],
            [
                ("tier5:4204", update("Eighteen degrees is my regular cabin temperature; cool mode is only for this drive.", "add", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 18, {"zone": "all"}, "for Finley's normal cabin climate", "The source temperature is retained while HVAC mode remains turn-local.")),
                ("tier4:4097", update("White is my normal ambient-light color; maximum brightness is only for finding this item.", "add", "driver.normal.ambient_color", "carcontrol_light_set_ambient_color", "color", "white", {}, "for Finley's normal ambient lighting", "The inferred source color is made persistent while task-specific brightness remains turn-local.")),
                ("tier4:4102", update("Use Home as my default destination.", "add", "driver.default.destination", "carcontrol_navigation_navigate_to", "destination", "Home", {}, "when Finley starts navigation without naming a destination", "The inferred source destination becomes an explicit default.")),
                ("tier4:4122", update("Mute guidance whenever the navigation prompts feel too frequent.", "add", "driver.frequent_prompts.voice_mode", "carcontrol_navigation_set_voice_mode", "mode", "mute", {}, "when navigation prompts feel too frequent to Finley", "The inferred source mute action is retained under its prompt-frequency condition.")),
                ("tier6:4236", update("Use rear-only defrost whenever I need clear rear visibility.", "add", "driver.rear_visibility.defrost", "carcontrol_airConditioner_set_mode", "mode", "defrost", {"zone": "rear"}, "when Finley needs to clear the rear glass", "The corrected source rear-only request becomes a durable visibility rule.")),
                ("tier4:4224", update("Use outside air whenever stale cabin air gives me a headache.", "add", "driver.stale_air.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "outside", {"zone": "all"}, "when stale cabin air gives Finley a headache", "The inferred fresh-air action is retained under the source stale-air condition.")),
                ("tier6:4004", update("Keep all doors unlocked as my normal parked-access setting.", "add", "driver.parked.doors_locked", "carcontrol_door_set_locked", "locked", False, {"door": "all"}, "when Finley is parked", "The corrected source unlock action is made into an explicit parked-state preference.")),
                ("tier5:4206", update("Actually, twenty-seven degrees is my regular cabin temperature now; heat mode is only for this drive.", "replace", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 27, {"zone": "all"}, "for Finley's normal cabin climate", "The later source temperature replaces eighteen degrees while HVAC mode remains turn-local.")),
                ("tier6:4208", update("Actually, twenty-four degrees is my regular cabin temperature now; auto mode is only for this drive.", "replace", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 24, {"zone": "all"}, "for Finley's normal cabin climate", "The corrected later source temperature replaces twenty-seven degrees while HVAC mode remains turn-local.")),
                ("tier6:4210", update("Actually, twenty-one degrees is my regular cabin temperature now; the requested HVAC mode is only for this drive.", "replace", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 21, {"zone": "all"}, "for Finley's normal cabin climate", "The corrected later source temperature replaces twenty-four degrees while HVAC mode remains turn-local.")),
            ],
        ),
    }
)

PLANS.update(
    {
        "HVP13": balanced_plan(
            913,
            "Cameron",
            [
                ("20", "open", "Planning food and invitations for a birthday party."),
                ("49", "weather", "Packing for a four-night trip to Florence."),
                ("10", "weather", "Choosing clothes for an upcoming Italy trip."),
            ],
            [2032, 2063, 2146],
            [
                ("tier6:3966", update("Fan speed five is my regular cabin setting.", "add", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 5, {"zone": "all"}, "for Cameron's normal cabin setting", "The corrected source fan level is made persistent.")),
                ("tier6:4006", update("Fifty-five percent is my regular driver-window position.", "add", "driver.normal.driver_window", "carcontrol_window_set_open_degree", "degree", 55, {"window": "driver"}, "for Cameron's normal driving", "The corrected source window position is made persistent.")),
                ("tier6:3962", update("Twenty-seven degrees is my regular rear-right temperature.", "add", "driver.normal.rear_right_temperature", "carcontrol_airConditioner_set_temperature", "temperature", 27, {"zone": "rear_right"}, "for Cameron's normal rear-right climate", "The corrected source temperature is made persistent.")),
                ("tier6:4234", update("Use front-and-rear defrost whenever frost blocks my visibility.", "add", "driver.frost.defrost", "carcontrol_airConditioner_set_mode", "mode", "defrost", {"zone": "all"}, "when frost blocks Cameron's visibility", "The corrected source defrost scope becomes a durable visibility rule.")),
                ("tier5:4134", update("Mute guidance whenever I am telling a story.", "add", "driver.storytelling.voice_mode", "carcontrol_navigation_set_voice_mode", "mode", "mute", {}, "when Cameron is telling a story", "The source mute request is retained under its explicit storytelling context.")),
                ("tier5:4092", update("Whenever my audio playback has stopped, resume it.", "add", "driver.interrupted_audio.playback", "carcontrol_music_switch", "switch", True, {}, "when Cameron's audio playback has stopped", "The source resume request becomes a recurring interrupted-playback preference.")),
                ("tier5:3980", update("Keep steering-wheel heat enabled whenever my hands feel cold.", "add", "driver.cold_hands.wheel_heat", "carcontrol_steeringWheel_set_heating_enabled", "enabled", True, {}, "when Cameron's hands feel cold", "The source heat request is retained under its explicit comfort condition.")),
                ("tier5:3966", update("Actually, fan speed one is my regular cabin setting now.", "replace", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 1, {"zone": "all"}, "for Cameron's normal cabin setting", "The later source fan level replaces the stored value.")),
                ("tier6:3968", update("Actually, fan speed three is my regular cabin setting now.", "replace", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 3, {"zone": "all"}, "for Cameron's normal cabin setting", "The corrected later source value replaces fan level one.")),
                ("tier6:4008", update("Actually, ninety percent is my regular driver-window position now.", "replace", "driver.normal.driver_window", "carcontrol_window_set_open_degree", "degree", 90, {"window": "driver"}, "for Cameron's normal driving", "The corrected later source value replaces the stored driver-window position.")),
            ],
        ),
        "HVP14": balanced_plan(
            914,
            "Parker",
            [
                ("46", "weather", "Checking temperatures before packing for Venice."),
                ("47", "search", "Booking a middle seat at a local cinema."),
                ("11", "search", "Checking evening showings of a film."),
            ],
            [2124, 2126, 2020],
            [
                ("tier6:4274", update("Thirty percent is my normal sunroof vent position.", "add", "driver.normal.sunroof", "carcontrol_sunroof_set_open_degree", "degree", 30, {}, "for Parker's normal cabin venting", "The corrected source sunroof position is made persistent.")),
                ("tier6:4116", update("Use Work as my default destination.", "add", "driver.default.destination", "carcontrol_navigation_navigate_to", "destination", "Work", {}, "when Parker starts navigation without naming a destination", "The corrected source destination becomes a persistent default.")),
                ("tier6:4270", update("I normally use the front trunk when loading cargo.", "add", "driver.cargo.front_trunk", "carcontrol_frontTrunk_switch", "switch", True, {}, "when Parker loads cargo", "The corrected source front-trunk action becomes a recurring cargo-loading routine without inventing a cargo type.")),
                ("tier5:4131", update("Mute guidance on route segments where the directions are obvious.", "add", "driver.obvious_route.voice_mode", "carcontrol_navigation_set_voice_mode", "mode", "mute", {}, "when route directions are obvious to Parker", "The source mute request is retained under the same obvious-route condition.")),
                ("tier6:4235", update("Use rear-only defrost whenever I need clear rear visibility.", "add", "driver.rear_visibility.defrost", "carcontrol_airConditioner_set_mode", "mode", "defrost", {"zone": "rear"}, "when Parker needs to clear the rear glass", "The corrected source rear-only request becomes a durable visibility rule.")),
                ("tier6:4272", update("Keep the rear trunk closed during normal driving.", "add", "driver.normal.rear_trunk", "carcontrol_trunk_switch", "switch", False, {}, "for Parker's normal driving", "The corrected source rear-trunk state is made persistent.")),
                ("tier6:3978", update("Keep steering-wheel heat disabled as my normal setting.", "add", "driver.normal.wheel_heat", "carcontrol_steeringWheel_set_heating_enabled", "enabled", False, {}, "for Parker's normal steering-wheel setting", "The corrected source disabled state is made persistent.")),
                ("tier6:4276", update("Actually, eighteen percent is my normal sunroof vent position now.", "replace", "driver.normal.sunroof", "carcontrol_sunroof_set_open_degree", "degree", 18, {}, "for Parker's normal cabin venting", "The corrected later source position replaces thirty percent.")),
                ("tier6:4118", update("Actually, make Home my default destination now.", "replace", "driver.default.destination", "carcontrol_navigation_navigate_to", "destination", "Home", {}, "when Parker starts navigation without naming a destination", "The corrected later source destination replaces Work.")),
                ("tier6:3980", update("Actually, keep steering-wheel heat enabled as my normal setting now.", "replace", "driver.normal.wheel_heat", "carcontrol_steeringWheel_set_heating_enabled", "enabled", True, {}, "for Parker's normal steering-wheel setting", "The corrected later source state replaces the disabled setting.")),
            ],
        ),
        "HVP15": balanced_plan(
            915,
            "Rowan",
            [
                ("82", "weather", "Preparing a five-day packing list for Bologna."),
                ("40", "search", "Checking evening showings of a nearby film."),
                ("148", "search", "Booking a local cinema showing."),
            ],
            [2029, 2073, 2074],
            [
                ("tier6:4238", update("Outside air is my regular cabin-air setting.", "add", "driver.normal.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "outside", {"zone": "all"}, "for Rowan's normal cabin-air setting", "The corrected source circulation state is made persistent.")),
                ("tier5:3968", update("Fan speed seven is my regular cabin setting.", "add", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 7, {"zone": "all"}, "for Rowan's normal cabin setting", "The explicit source fan level is made persistent.")),
                ("tier4:3947", update("Nineteen degrees is my regular driver-zone setting.", "add", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 19, {"zone": "driver"}, "for Rowan's normal driving", "The inferred source temperature is stated explicitly and made persistent.")),
                ("tier5:4132", update("Mute guidance on route segments where the directions are obvious.", "add", "driver.obvious_route.voice_mode", "carcontrol_navigation_set_voice_mode", "mode", "mute", {}, "when route directions are obvious to Rowan", "The source mute request is retained under the same obvious-route condition.")),
                ("tier4:4082", update("Resume playback whenever my audiobook stops unexpectedly.", "add", "driver.audiobook.playback", "carcontrol_music_switch", "switch", True, {}, "when Rowan's audiobook stops unexpectedly", "The inferred source playback action is retained under its audiobook condition.")),
                ("tier5:3978", update("Keep steering-wheel heat enabled whenever I forget my gloves.", "add", "driver.no_gloves.wheel_heat", "carcontrol_steeringWheel_set_heating_enabled", "enabled", True, {}, "when Rowan has forgotten gloves", "The source heat request is retained under the same missing-gloves condition.")),
                ("tier4:4255", update("Keep the rear trunk closed during normal driving.", "add", "driver.normal.rear_trunk", "carcontrol_trunk_switch", "switch", False, {}, "for Rowan's normal driving", "The inferred source close action is made into an explicit normal-driving state.")),
                ("tier6:4240", update("Actually, inside circulation is my regular cabin-air setting now.", "replace", "driver.normal.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "inside", {"zone": "all"}, "for Rowan's normal cabin-air setting", "The corrected later source state replaces outside circulation.")),
                (1984, update("Fan speed eight is my regular cabin setting now.", "replace", "driver.normal.fan_speed", "carcontrol_airConditioner_set_fan_speed", "speed", 8, {"zone": "all"}, "for Rowan's normal cabin setting", "The later multi-turn source fan request replaces level seven.")),
                ("tier5:3964", update("Actually, eighteen degrees is my regular driver-zone setting now.", "replace", "driver.normal.temperature", "carcontrol_airConditioner_set_temperature", "temperature", 18, {"zone": "driver"}, "for Rowan's normal driving", "The later source temperature replaces nineteen degrees.")),
            ],
        ),
        "HVP16": balanced_plan(
            916,
            "Reese",
            [
                ("179", "joke", "Booking an evening comedy show and requesting directions."),
                ("15", "search", "Booking cinema tickets and refreshments."),
                ("22", "search", "Checking an early-evening cinema showing."),
            ],
            [2023, 2045, 2075],
            [
                ("tier5:4238", update("Whenever the windows are at risk of fogging, use outside air.", "add", "driver.fog_risk.circulation", "carcontrol_airConditioner_set_circulation", "circulation", "outside", {"zone": "all"}, "when Reese's windows are at risk of fogging", "The explicit source circulation state is retained under its window-fogging condition.")),
                ("tier5:4274", update("Ten percent is my normal sunroof position.", "add", "driver.normal.sunroof", "carcontrol_sunroof_set_open_degree", "degree", 10, {}, "for Reese's normal sunroof setting", "The source sunroof position is made persistent.")),
                ("tier5:4133", update("Mute guidance whenever I am telling a story.", "add", "driver.storytelling.voice_mode", "carcontrol_navigation_set_voice_mode", "mode", "mute", {}, "when Reese is telling a story", "The source mute request is retained under its explicit storytelling context.")),
                ("tier5:4091", update("Whenever my audio playback has stopped, resume it.", "add", "driver.interrupted_audio.playback", "carcontrol_music_switch", "switch", True, {}, "when Reese's audio playback has stopped", "The source resume request becomes a recurring interrupted-playback preference.")),
                ("tier4:4220", update("Use front-and-rear defrost whenever frost blocks my visibility.", "add", "driver.frost.defrost", "carcontrol_airConditioner_set_mode", "mode", "defrost", {"zone": "all"}, "when frost blocks Reese's visibility", "The inferred source defrost action becomes a durable visibility rule.")),
                (2001, update("Fifty percent is my regular driver-window position.", "add", "driver.normal.driver_window", "carcontrol_window_set_open_degree", "degree", 50, {"window": "driver"}, "for Reese's normal driving", "The clarified multi-turn source window request is made persistent.")),
                ("tier4:3964", update("Keep steering-wheel heat enabled whenever my hands feel cold.", "add", "driver.cold_hands.wheel_heat", "carcontrol_steeringWheel_set_heating_enabled", "enabled", True, {}, "when Reese's hands feel cold", "The inferred source heat request is retained under its cold-wheel condition.")),
                ("tier4:4259", update("Actually, fifteen percent is my normal sunroof position now.", "replace", "driver.normal.sunroof", "carcontrol_sunroof_set_open_degree", "degree", 15, {}, "for Reese's normal sunroof setting", "The inferred later source position replaces ten percent.")),
                ("tier5:4276", update("Actually, thirty percent is my normal sunroof position now.", "replace", "driver.normal.sunroof", "carcontrol_sunroof_set_open_degree", "degree", 30, {}, "for Reese's normal sunroof setting", "The later source position replaces fifteen percent.")),
                ("tier4:4261", update("Actually, fully closed is my normal sunroof position now.", "replace", "driver.normal.sunroof", "carcontrol_sunroof_set_open_degree", "degree", 0, {}, "for Reese's normal sunroof setting", "The inferred later source state replaces thirty percent.")),
            ],
        ),
    }
)

# HVP08 was reviewed turn by turn. Keep its quizzes explicit so each question
# states the activation condition instead of relying on generic factory wording.
# HVP01-HVP02 use hand-reviewed quiz wording. A close-window or close-trunk
# call can be identical to VehicleWorld's empty initial state, so those checks
# are paired with another active fact to produce a deterministic state change.
PLANS["HVP01"]["turn_quizzes"] = [
    ("tier4:3959", "Set Devon's rear-right seat ventilation to the currently remembered normal speed.", "state_shift", ["driver.normal.rear_right_seat_ventilation_speed"], [("carcontrol_seat_set_ventilation_speed", {"seat": "rear_right", "speed": 2})]),
    ("tier6:1103", "Set the music to Devon's currently remembered normal volume.", "state_shift", ["driver.normal.music_volume"], [("carcontrol_music_set_volume", {"volume": 11})]),
    ("tier5:4007", "Devon needs the tool bag. Apply the remembered front-trunk action.", "conditional_constraint", ["driver.tools.front_trunk"], [("carcontrol_frontTrunk_switch", {"switch": True})]),
    ("tier4:4191", "Apply Devon's currently remembered normal cabin climate mode.", "state_shift", ["driver.normal.climate_mode"], [("carcontrol_airConditioner_set_mode", {"zone": "all", "mode": "auto"})]),
]
PLANS["HVP01"]["final_quizzes"] = [
    ("Set Devon's rear-right seat ventilation to the remembered normal speed.", "state_shift", ["driver.normal.rear_right_seat_ventilation_speed"], [("carcontrol_seat_set_ventilation_speed", {"seat": "rear_right", "speed": 2})]),
    ("Set the music to Devon's current normal volume.", "error_correction", ["driver.normal.music_volume"], [("carcontrol_music_set_volume", {"volume": 58})]),
    ("Devon needs the tool bag. Apply the remembered front-trunk action.", "conditional_constraint", ["driver.tools.front_trunk"], [("carcontrol_frontTrunk_switch", {"switch": True})]),
    ("Apply Devon's current normal cabin climate mode.", "error_correction", ["driver.normal.climate_mode"], [("carcontrol_airConditioner_set_mode", {"zone": "all", "mode": "defrost"})]),
    ("Rain is starting and Devon needs the tool bag. Apply both remembered settings.", "conditional_constraint", ["driver.rain.driver_window", "driver.tools.front_trunk"], [("carcontrol_window_set_open_degree", {"window": "driver", "degree": 0}), ("carcontrol_frontTrunk_switch", {"switch": True})]),
    ("Restaurant exhaust is bothering Devon, and the bags must remain out of sight. Apply both remembered settings.", "conditional_constraint", ["driver.restaurant_exhaust.circulation", "driver.hidden_bags.rear_trunk"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"}), ("carcontrol_trunk_switch", {"switch": False})]),
]

PLANS["HVP02"]["turn_quizzes"] = [
    ("tier6:1081", "Set the music to Robin's currently remembered normal volume.", "state_shift", ["driver.normal.music_volume"], [("carcontrol_music_set_volume", {"volume": 52})]),
    ("tier6:3961", "Set Robin's rear-right zone to the currently remembered normal temperature.", "state_shift", ["driver.normal.rear_right_temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "rear_right", "temperature": 27})]),
    ("tier6:3973", "Set Robin's passenger-seat ventilation to the remembered normal speed.", "state_shift", ["driver.normal.passenger_seat_ventilation_speed"], [("carcontrol_seat_set_ventilation_speed", {"seat": "passenger", "speed": 1})]),
    ("tier6:4011", "Robin needs the stored documents. Apply the remembered front-trunk action.", "conditional_constraint", ["driver.documents.front_trunk"], [("carcontrol_frontTrunk_switch", {"switch": True})]),
]
PLANS["HVP02"]["final_quizzes"] = [
    ("Set the music to Robin's current normal volume.", "error_correction", ["driver.normal.music_volume"], [("carcontrol_music_set_volume", {"volume": 5})]),
    ("Set Robin's rear-right zone to the remembered normal temperature.", "state_shift", ["driver.normal.rear_right_temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "rear_right", "temperature": 27})]),
    ("Set Robin's passenger-seat ventilation to the remembered normal speed.", "state_shift", ["driver.normal.passenger_seat_ventilation_speed"], [("carcontrol_seat_set_ventilation_speed", {"seat": "passenger", "speed": 1})]),
    ("Robin needs the stored documents. Apply the remembered front-trunk action.", "conditional_constraint", ["driver.documents.front_trunk"], [("carcontrol_frontTrunk_switch", {"switch": True})]),
    ("The open rear trunk obstructs Robin's view; also restore the remembered rear-right temperature.", "conditional_constraint", ["driver.obstructed_view.rear_trunk", "driver.normal.rear_right_temperature"], [("carcontrol_trunk_switch", {"switch": False}), ("carcontrol_airConditioner_set_temperature", {"zone": "rear_right", "temperature": 27})]),
    ("Robin is ready to leave with navigation active. Apply the remembered door and guidance settings.", "conditional_constraint", ["driver.departure.doors_locked", "driver.normal.navigation_voice_mode"], [("carcontrol_door_set_locked", {"door": "all", "locked": False}), ("carcontrol_navigation_set_voice_mode", {"mode": "detailed"})]),
]

PLANS["HVP08"]["turn_quizzes"] = [
    ("tier5:3963", "Avery has started a normal drive. Set the driver-zone temperature to the remembered value.", "state_shift", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "driver", "temperature": 18})]),
    ("tier5:4003", "Avery has started a normal drive. Set the passenger-front window to the remembered position.", "state_shift", ["driver.normal.passenger_window"], [("carcontrol_window_set_open_degree", {"window": "passenger", "degree": 60})]),
    ("tier5:4111", "Avery asks to start navigation without naming a destination. Use the remembered default.", "state_shift", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Riverside Dog Park on Kennel Lane"})]),
    ("tier5:4239", "Outside restaurant odors are bothering Avery. Apply the remembered cabin-air setting.", "conditional_constraint", ["driver.food_odor.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "inside"})]),
]
PLANS["HVP08"]["final_quizzes"] = [
    ("Avery has started a normal drive. Set the driver-zone temperature to the remembered value.", "state_shift", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "driver", "temperature": 18})]),
    ("Avery has started a normal drive. Set the passenger-front window to the current remembered position.", "error_correction", ["driver.normal.passenger_window"], [("carcontrol_window_set_open_degree", {"window": "passenger", "degree": 10})]),
    ("Avery asks to start navigation without naming a destination. Use the current default.", "error_correction", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Eastside Public Library on Maple Street"})]),
    ("Outside restaurant odors are bothering Avery. Apply the remembered cabin-air setting.", "conditional_constraint", ["driver.food_odor.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "inside"})]),
    ("Set the ambient lighting to Avery's normal remembered color.", "state_shift", ["driver.normal.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "blue"})]),
    ("Avery is loading a bulky item and is about to leave. Apply the remembered rear-trunk and door settings.", "conditional_constraint", ["driver.bulky_items.rear_trunk", "driver.vehicle_access.doors_locked"], [("carcontrol_trunk_switch", {"switch": True}), ("carcontrol_door_set_locked", {"door": "all", "locked": False})]),
]

# HVP09 received the same turn-by-turn review and explicit activation queries.
PLANS["HVP09"]["turn_quizzes"] = [
    ("tier4:3951", "Set the cabin fan to Jordan's currently remembered normal speed.", "state_shift", ["driver.normal.fan_speed"], [("carcontrol_airConditioner_set_fan_speed", {"zone": "all", "speed": 6})]),
    ("tier5:4109", "Jordan wants a romantic cabin mood. Set the ambient lighting to the remembered color.", "conditional_constraint", ["driver.romantic.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "red"})]),
    ("tier5:4271", "Jordan is loading heavy shopping. Open the remembered cargo area.", "conditional_constraint", ["driver.heavy_shopping.rear_trunk"], [("carcontrol_trunk_switch", {"switch": True})]),
    ("tier4:4260", "It is a bright day. Set Jordan's sunroof to the remembered position.", "conditional_constraint", ["driver.bright_day.sunroof"], [("carcontrol_sunroof_set_open_degree", {"degree": 15})]),
]
PLANS["HVP09"]["final_quizzes"] = [
    ("Set the cabin fan to Jordan's current normal speed.", "error_correction", ["driver.normal.fan_speed"], [("carcontrol_airConditioner_set_fan_speed", {"zone": "all", "speed": 8})]),
    ("Jordan wants a romantic cabin mood. Set the ambient lighting to the remembered color.", "conditional_constraint", ["driver.romantic.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "red"})]),
    ("Jordan is loading heavy shopping. Open the remembered cargo area.", "conditional_constraint", ["driver.heavy_shopping.rear_trunk"], [("carcontrol_trunk_switch", {"switch": True})]),
    ("It is a bright day. Set Jordan's sunroof to the remembered position.", "conditional_constraint", ["driver.bright_day.sunroof"], [("carcontrol_sunroof_set_open_degree", {"degree": 15})]),
    ("Jordan is listening to an audiobook. Apply the remembered navigation voice setting.", "conditional_constraint", ["driver.audiobook.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
    ("Jordan is parked, about to leave, and it is a cold day. Apply the current door-access and steering-wheel heat settings.", "error_correction", ["driver.vehicle_access.doors_locked", "driver.steering_wheel.heat_enabled"], [("carcontrol_door_set_locked", {"door": "all", "locked": False}), ("carcontrol_steeringWheel_set_heating_enabled", {"enabled": True})]),
]

# HVP10 keeps rear and front visibility conditions separate; its third
# replacement is a true passenger-window value update rather than defrost conflation.
PLANS["HVP10"]["turn_quizzes"] = [
    ("tier5:3961", "Set the front-passenger temperature to Taylor's remembered normal value.", "state_shift", ["driver.normal.passenger_temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "passenger", "temperature": 23})]),
    ("tier5:3965", "Set the cabin fan to Taylor's currently remembered normal speed.", "state_shift", ["driver.normal.fan_speed"], [("carcontrol_airConditioner_set_fan_speed", {"zone": "all", "speed": 1})]),
    ("tier5:3979", "Taylor's hands feel cold. Apply the remembered steering-wheel setting.", "conditional_constraint", ["driver.cold_hands.wheel_heat"], [("carcontrol_steeringWheel_set_heating_enabled", {"enabled": True})]),
    ("tier4:3991", "Taylor's children find the rear cabin stuffy. Set their rear-right window to the remembered position.", "conditional_constraint", ["driver.kids_stuffy.rear_right_window"], [("carcontrol_window_set_open_degree", {"window": "rear_right", "degree": 60})]),
]
PLANS["HVP10"]["final_quizzes"] = [
    ("Set the front-passenger temperature to Taylor's remembered normal value.", "state_shift", ["driver.normal.passenger_temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "passenger", "temperature": 23})]),
    ("Set the cabin fan to Taylor's current normal speed.", "error_correction", ["driver.normal.fan_speed"], [("carcontrol_airConditioner_set_fan_speed", {"zone": "all", "speed": 7})]),
    ("Taylor's hands feel cold. Apply the remembered steering-wheel setting.", "conditional_constraint", ["driver.cold_hands.wheel_heat"], [("carcontrol_steeringWheel_set_heating_enabled", {"enabled": True})]),
    ("Taylor's children find the rear cabin stuffy. Set their rear-right window to the remembered position.", "conditional_constraint", ["driver.kids_stuffy.rear_right_window"], [("carcontrol_window_set_open_degree", {"window": "rear_right", "degree": 60})]),
    ("Set the passenger-front window to Taylor's current normal position.", "error_correction", ["driver.normal.passenger_window"], [("carcontrol_window_set_open_degree", {"window": "passenger", "degree": 10})]),
    ("Taylor wants to vent the cabin and also needs clear rear visibility. Apply the remembered sunroof and defrost settings.", "conditional_constraint", ["driver.venting.sunroof", "driver.rear_visibility.defrost"], [("carcontrol_sunroof_set_open_degree", {"degree": 30}), ("carcontrol_airConditioner_set_mode", {"zone": "rear", "mode": "defrost"})]),
]

# HVP11 uses three attribute-aligned update pairs after manual review.
PLANS["HVP11"]["turn_quizzes"] = [
    ("tier4:3961", "Set the driver-seat heat to Quinn's remembered normal level.", "state_shift", ["driver.normal.seat_heat"], [("carcontrol_seat_set_heating_level", {"seat": "driver", "level": 1})]),
    ("tier5:3962", "Set the front-passenger temperature to Quinn's remembered normal value.", "state_shift", ["driver.normal.passenger_temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "passenger", "temperature": 23})]),
    ("tier4:4095", "Set the ambient lighting to Quinn's remembered normal color.", "state_shift", ["driver.normal.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "blue"})]),
    ("tier4:4119", "Quinn is listening to an audiobook. Apply the remembered navigation voice setting.", "conditional_constraint", ["driver.audiobook.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
]
PLANS["HVP11"]["final_quizzes"] = [
    ("Set the driver-seat heat to Quinn's remembered normal level.", "state_shift", ["driver.normal.seat_heat"], [("carcontrol_seat_set_heating_level", {"seat": "driver", "level": 1})]),
    ("Set the front-passenger temperature to Quinn's remembered normal value.", "state_shift", ["driver.normal.passenger_temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "passenger", "temperature": 23})]),
    ("Set the ambient lighting to Quinn's remembered normal color.", "state_shift", ["driver.normal.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "blue"})]),
    ("Quinn is listening to an audiobook. Apply the remembered navigation voice setting.", "conditional_constraint", ["driver.audiobook.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
    ("Set the cabin fan to Quinn's current normal speed.", "error_correction", ["driver.normal.fan_speed"], [("carcontrol_airConditioner_set_fan_speed", {"zone": "all", "speed": 8})]),
    ("Quinn is parked, about to leave, and asks to start navigation without naming a destination. Apply the current door-access and destination defaults.", "error_correction", ["driver.vehicle_access.doors_locked", "driver.default.destination"], [("carcontrol_door_set_locked", {"door": "all", "locked": False}), ("carcontrol_navigation_navigate_to", {"destination": "Eastside Public Library on Maple Street"})]),
]

# HVP12's reviewed quizzes preserve the audiobook and frequent-prompt
# conditions and compare only like-for-like replacement slots.
PLANS["HVP12"]["turn_quizzes"] = [
    ("tier4:3948", "Set the driver-zone temperature to Sydney's remembered normal value.", "state_shift", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "driver", "temperature": 19})]),
    ("tier4:4099", "Sydney asks to start navigation without naming a destination. Use the remembered default.", "state_shift", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Home"})]),
    ("tier4:4081", "Sydney is listening to an audiobook that has stopped. Apply the remembered playback preference.", "conditional_constraint", ["driver.audiobook.playback"], [("carcontrol_music_switch", {"switch": True})]),
    ("tier4:4121", "The navigation prompts feel too frequent to Sydney. Apply the remembered voice setting.", "conditional_constraint", ["driver.frequent_prompts.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
]
PLANS["HVP12"]["final_quizzes"] = [
    ("Set the driver-zone temperature to Sydney's remembered normal value.", "state_shift", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "driver", "temperature": 19})]),
    ("Sydney asks to start navigation without naming a destination. Use the current default.", "error_correction", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Riverside Dog Park on Kennel Lane"})]),
    ("Sydney is listening to an audiobook that has stopped. Apply the remembered playback preference.", "conditional_constraint", ["driver.audiobook.playback"], [("carcontrol_music_switch", {"switch": True})]),
    ("The navigation prompts feel too frequent to Sydney. Apply the remembered voice setting.", "conditional_constraint", ["driver.frequent_prompts.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
    ("Frost is blocking Sydney's visibility. Apply the remembered defrost setting.", "conditional_constraint", ["driver.frost.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "all", "mode": "defrost"})]),
    ("Sydney has parked and is preparing for a cold drive. Apply the current parked-door and normal steering-wheel settings.", "error_correction", ["driver.steering_wheel.heat_enabled", "driver.parked.doors_locked"], [("carcontrol_steeringWheel_set_heating_enabled", {"enabled": True}), ("carcontrol_door_set_locked", {"door": "all", "locked": True})]),
]

# HVP13-HVP16 use explicit natural-language activation queries. Every
# replacement keeps the same slot, tool, context arguments, and condition.
PLANS["HVP13"]["turn_quizzes"] = [
    ("tier6:3966", "Set the cabin fan to Cameron's currently remembered normal speed.", "state_shift", ["driver.normal.fan_speed"], [("carcontrol_airConditioner_set_fan_speed", {"zone": "all", "speed": 5})]),
    ("tier6:4006", "Set the driver window to Cameron's currently remembered normal position.", "state_shift", ["driver.normal.driver_window"], [("carcontrol_window_set_open_degree", {"window": "driver", "degree": 55})]),
    ("tier6:3962", "Set the rear-right temperature to Cameron's remembered normal value.", "state_shift", ["driver.normal.rear_right_temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "rear_right", "temperature": 27})]),
    ("tier6:4234", "Frost is blocking Cameron's visibility. Apply the remembered defrost setting.", "conditional_constraint", ["driver.frost.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "all", "mode": "defrost"})]),
]
PLANS["HVP13"]["final_quizzes"] = [
    ("Set the cabin fan to Cameron's current normal speed.", "error_correction", ["driver.normal.fan_speed"], [("carcontrol_airConditioner_set_fan_speed", {"zone": "all", "speed": 3})]),
    ("Set the driver window to Cameron's current normal position.", "error_correction", ["driver.normal.driver_window"], [("carcontrol_window_set_open_degree", {"window": "driver", "degree": 90})]),
    ("Set the rear-right temperature to Cameron's remembered normal value.", "state_shift", ["driver.normal.rear_right_temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "rear_right", "temperature": 27})]),
    ("Frost is blocking Cameron's visibility. Apply the remembered defrost setting.", "conditional_constraint", ["driver.frost.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "all", "mode": "defrost"})]),
    ("Cameron is telling a story. Apply the remembered navigation voice setting.", "conditional_constraint", ["driver.storytelling.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
    ("Cameron's audio playback has stopped and their hands feel cold. Apply both remembered settings.", "conditional_constraint", ["driver.interrupted_audio.playback", "driver.cold_hands.wheel_heat"], [("carcontrol_music_switch", {"switch": True}), ("carcontrol_steeringWheel_set_heating_enabled", {"enabled": True})]),
]

PLANS["HVP14"]["turn_quizzes"] = [
    ("tier6:4274", "Vent the cabin using Parker's currently remembered normal sunroof position.", "state_shift", ["driver.normal.sunroof"], [("carcontrol_sunroof_set_open_degree", {"degree": 30})]),
    ("tier6:4116", "Parker starts navigation without naming a destination. Use the remembered default.", "state_shift", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Work"})]),
    ("tier6:4270", "Parker is loading cargo. Apply the remembered front-trunk routine.", "conditional_constraint", ["driver.cargo.front_trunk"], [("carcontrol_frontTrunk_switch", {"switch": True})]),
    ("tier5:4131", "The directions on this route segment are obvious to Parker. Apply the remembered voice setting.", "conditional_constraint", ["driver.obvious_route.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
]
PLANS["HVP14"]["final_quizzes"] = [
    ("Vent the cabin using Parker's current normal sunroof position.", "error_correction", ["driver.normal.sunroof"], [("carcontrol_sunroof_set_open_degree", {"degree": 18})]),
    ("Parker starts navigation without naming a destination. Use the current default.", "error_correction", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Home"})]),
    ("Parker is loading cargo. Apply the remembered front-trunk routine.", "conditional_constraint", ["driver.cargo.front_trunk"], [("carcontrol_frontTrunk_switch", {"switch": True})]),
    ("The directions on this route segment are obvious to Parker. Apply the remembered voice setting.", "conditional_constraint", ["driver.obvious_route.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
    ("Parker needs clear rear visibility. Apply the remembered defrost setting.", "conditional_constraint", ["driver.rear_visibility.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "rear", "mode": "defrost"})]),
    ("Parker has begun a normal drive. Apply the current rear-trunk and steering-wheel settings.", "error_correction", ["driver.normal.rear_trunk", "driver.normal.wheel_heat"], [("carcontrol_trunk_switch", {"switch": False}), ("carcontrol_steeringWheel_set_heating_enabled", {"enabled": True})]),
]

PLANS["HVP15"]["turn_quizzes"] = [
    ("tier6:4238", "Apply Rowan's currently remembered normal cabin-air setting.", "state_shift", ["driver.normal.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"})]),
    ("tier5:3968", "Set the cabin fan to Rowan's currently remembered normal speed.", "state_shift", ["driver.normal.fan_speed"], [("carcontrol_airConditioner_set_fan_speed", {"zone": "all", "speed": 7})]),
    ("tier4:3947", "Set the driver-zone temperature to Rowan's currently remembered normal value.", "state_shift", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "driver", "temperature": 19})]),
    ("tier5:4132", "The directions on this route segment are obvious to Rowan. Apply the remembered voice setting.", "conditional_constraint", ["driver.obvious_route.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
]
PLANS["HVP15"]["final_quizzes"] = [
    ("Apply Rowan's current normal cabin-air setting.", "error_correction", ["driver.normal.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "inside"})]),
    ("Set the cabin fan to Rowan's current normal speed.", "error_correction", ["driver.normal.fan_speed"], [("carcontrol_airConditioner_set_fan_speed", {"zone": "all", "speed": 8})]),
    ("Set the driver-zone temperature to Rowan's current normal value.", "error_correction", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "driver", "temperature": 18})]),
    ("The directions on this route segment are obvious to Rowan. Apply the remembered voice setting.", "conditional_constraint", ["driver.obvious_route.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
    ("Rowan's audiobook has stopped unexpectedly. Apply the remembered playback preference.", "conditional_constraint", ["driver.audiobook.playback"], [("carcontrol_music_switch", {"switch": True})]),
    ("Rowan has forgotten their gloves and begun a normal drive. Apply the remembered wheel-heat and rear-trunk settings.", "conditional_constraint", ["driver.no_gloves.wheel_heat", "driver.normal.rear_trunk"], [("carcontrol_steeringWheel_set_heating_enabled", {"enabled": True}), ("carcontrol_trunk_switch", {"switch": False})]),
]

PLANS["HVP16"]["turn_quizzes"] = [
    ("tier5:4238", "Reese's windows are at risk of fogging. Apply the remembered cabin-air setting.", "conditional_constraint", ["driver.fog_risk.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"})]),
    ("tier5:4274", "Set the sunroof to Reese's currently remembered normal position.", "state_shift", ["driver.normal.sunroof"], [("carcontrol_sunroof_set_open_degree", {"degree": 10})]),
    ("tier5:4133", "Reese is telling a story. Apply the remembered navigation voice setting.", "conditional_constraint", ["driver.storytelling.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
    ("tier5:4091", "Reese's audio playback has stopped. Apply the remembered playback preference.", "conditional_constraint", ["driver.interrupted_audio.playback"], [("carcontrol_music_switch", {"switch": True})]),
]
PLANS["HVP16"]["final_quizzes"] = [
    ("Reese's windows are at risk of fogging. Apply the remembered cabin-air setting.", "conditional_constraint", ["driver.fog_risk.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"})]),
    ("Set the sunroof to Reese's current normal position.", "error_correction", ["driver.normal.sunroof"], [("carcontrol_sunroof_set_open_degree", {"degree": 0})]),
    ("Reese is telling a story. Apply the remembered navigation voice setting.", "conditional_constraint", ["driver.storytelling.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
    ("Reese's audio playback has stopped. Apply the remembered playback preference.", "conditional_constraint", ["driver.interrupted_audio.playback"], [("carcontrol_music_switch", {"switch": True})]),
    ("Frost is blocking Reese's visibility. Apply the remembered defrost setting.", "conditional_constraint", ["driver.frost.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "all", "mode": "defrost"})]),
    ("Reese has begun a normal drive and their hands feel cold. Apply the remembered driver-window and steering-wheel settings.", "conditional_constraint", ["driver.normal.driver_window", "driver.cold_hands.wheel_heat"], [("carcontrol_window_set_open_degree", {"window": "driver", "degree": 50}), ("carcontrol_steeringWheel_set_heating_enabled", {"enabled": True})]),
]

# HVP17-HVP20 likewise use explicit queries that activate the exact stored
# condition and test the final value after each replacement chain.
PLANS["HVP17"]["turn_quizzes"] = [
    ("tier5:4107", "Set the ambient lighting to Blair's currently remembered normal color.", "state_shift", ["driver.normal.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "pink"})]),
    ("tier5:4203", "Set the cabin temperature to Blair's currently remembered normal value.", "state_shift", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "all", "temperature": 18})]),
    ("tier4:3992", "Blair's children find the rear cabin stuffy. Apply the remembered rear-right window setting.", "conditional_constraint", ["driver.kids_stuffy.rear_right_window"], [("carcontrol_window_set_open_degree", {"window": "rear_right", "degree": 60})]),
    ("tier4:4101", "Blair starts navigation without naming a destination. Use the remembered default.", "state_shift", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Home"})]),
]
PLANS["HVP17"]["final_quizzes"] = [
    ("Set the ambient lighting to Blair's current normal color.", "error_correction", ["driver.normal.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "white"})]),
    ("Set the cabin temperature to Blair's current normal value.", "error_correction", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "all", "temperature": 27})]),
    ("Blair's children find the rear cabin stuffy. Apply the remembered rear-right window setting.", "conditional_constraint", ["driver.kids_stuffy.rear_right_window"], [("carcontrol_window_set_open_degree", {"window": "rear_right", "degree": 60})]),
    ("Blair starts navigation without naming a destination. Use the remembered default.", "state_shift", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Home"})]),
    ("Blair's cabin windows are heavily fogged. Apply the remembered defrost setting.", "conditional_constraint", ["driver.heavy_fog.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "all", "mode": "defrost"})]),
    ("Stale cabin air is giving Blair a headache during a normal drive. Apply the remembered circulation and rear-trunk settings.", "conditional_constraint", ["driver.stale_air.circulation", "driver.normal.rear_trunk"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"}), ("carcontrol_trunk_switch", {"switch": False})]),
]

PLANS["HVP18"]["turn_quizzes"] = [
    ("tier5:4108", "Set the ambient lighting to Harper's currently remembered normal color.", "state_shift", ["driver.normal.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "pink"})]),
    ("tier5:3973", "Set Harper's rear-right seat ventilation to the remembered normal speed.", "state_shift", ["driver.normal.rear_right_seat_ventilation_speed"], [("carcontrol_seat_set_ventilation_speed", {"seat": "rear_right", "speed": 2})]),
    ("tier6:4207", "Set the cabin temperature to Harper's currently remembered normal value.", "state_shift", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "all", "temperature": 24})]),
    ("tier4:4225", "Fog is obscuring Harper's side windows. Apply the remembered circulation setting.", "conditional_constraint", ["driver.side_window_fog.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"})]),
]
PLANS["HVP18"]["final_quizzes"] = [
    ("Set the ambient lighting and rear-right seat ventilation to Harper's current normal settings.", "error_correction", ["driver.normal.ambient_color", "driver.normal.rear_right_seat_ventilation_speed"], [("carcontrol_light_set_ambient_color", {"color": "red"}), ("carcontrol_seat_set_ventilation_speed", {"seat": "rear_right", "speed": 2})]),
    ("Set the cabin temperature to Harper's current normal value.", "error_correction", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "all", "temperature": 28})]),
    ("Fog is obscuring Harper's side windows. Apply the remembered circulation setting.", "conditional_constraint", ["driver.side_window_fog.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"})]),
    ("Harper's front trunk seems locked shut. Apply the remembered front-trunk action.", "conditional_constraint", ["driver.stuck.front_trunk"], [("carcontrol_frontTrunk_switch", {"switch": True})]),
    ("Cabin noise is distracting Harper. Apply the remembered sunroof setting.", "conditional_constraint", ["driver.noisy_cabin.sunroof"], [("carcontrol_sunroof_set_open_degree", {"degree": 0})]),
    ("Harper needs clear rear visibility. Apply the remembered defrost setting.", "conditional_constraint", ["driver.rear_visibility.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "rear", "mode": "defrost"})]),
]

PLANS["HVP19"]["turn_quizzes"] = [
    ("tier4:4098", "Set the ambient lighting to Emerson's currently remembered normal color.", "state_shift", ["driver.normal.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "white"})]),
    ("tier5:3974", "Set Emerson's rear-right seat ventilation to the remembered normal speed.", "state_shift", ["driver.normal.rear_right_seat_ventilation_speed"], [("carcontrol_seat_set_ventilation_speed", {"seat": "rear_right", "speed": 2})]),
    ("tier6:4239", "Apply Emerson's currently remembered normal cabin-air setting.", "state_shift", ["driver.normal.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "inside"})]),
    ("tier5:4235", "Condensation is obscuring Emerson's windshield. Apply the remembered defrost setting.", "conditional_constraint", ["driver.front_condensation.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "front", "mode": "defrost"})]),
]
PLANS["HVP19"]["final_quizzes"] = [
    ("Set the ambient lighting to Emerson's current normal color.", "error_correction", ["driver.normal.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "white"})]),
    ("Set Emerson's rear-right seat ventilation to the remembered normal speed.", "state_shift", ["driver.normal.rear_right_seat_ventilation_speed"], [("carcontrol_seat_set_ventilation_speed", {"seat": "rear_right", "speed": 2})]),
    ("Apply Emerson's current normal cabin-air setting.", "error_correction", ["driver.normal.circulation"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"})]),
    ("Condensation is obscuring Emerson's windshield. Apply the remembered defrost setting.", "conditional_constraint", ["driver.front_condensation.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "front", "mode": "defrost"})]),
    ("Emerson needs to retrieve a coat from the front trunk. Apply the remembered action.", "conditional_constraint", ["driver.coat.front_trunk"], [("carcontrol_frontTrunk_switch", {"switch": True})]),
    ("Emerson is parked and wants background noise. Apply the remembered door-access and radio settings.", "conditional_constraint", ["driver.parked.doors_locked", "driver.background_noise.radio"], [("carcontrol_door_set_locked", {"door": "all", "locked": False}), ("carcontrol_radio_switch", {"switch": True})]),
]

PLANS["HVP20"]["turn_quizzes"] = [
    ("tier5:4204", "Set the cabin temperature to Finley's currently remembered normal value.", "state_shift", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "all", "temperature": 18})]),
    ("tier4:4097", "Set the ambient lighting to Finley's remembered normal color.", "state_shift", ["driver.normal.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "white"})]),
    ("tier4:4102", "Finley starts navigation without naming a destination. Use the remembered default.", "state_shift", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Home"})]),
    ("tier4:4122", "The navigation prompts feel too frequent to Finley. Apply the remembered voice setting.", "conditional_constraint", ["driver.frequent_prompts.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
]
PLANS["HVP20"]["final_quizzes"] = [
    ("Set the cabin temperature to Finley's current normal value.", "error_correction", ["driver.normal.temperature"], [("carcontrol_airConditioner_set_temperature", {"zone": "all", "temperature": 21})]),
    ("Set the ambient lighting to Finley's remembered normal color.", "state_shift", ["driver.normal.ambient_color"], [("carcontrol_light_set_ambient_color", {"color": "white"})]),
    ("Finley starts navigation without naming a destination. Use the remembered default.", "state_shift", ["driver.default.destination"], [("carcontrol_navigation_navigate_to", {"destination": "Home"})]),
    ("The navigation prompts feel too frequent to Finley. Apply the remembered voice setting.", "conditional_constraint", ["driver.frequent_prompts.voice_mode"], [("carcontrol_navigation_set_voice_mode", {"mode": "mute"})]),
    ("Finley needs clear rear visibility. Apply the remembered defrost setting.", "conditional_constraint", ["driver.rear_visibility.defrost"], [("carcontrol_airConditioner_set_mode", {"zone": "rear", "mode": "defrost"})]),
    ("Stale cabin air is giving Finley a headache while parked. Apply the remembered circulation and door-access settings.", "conditional_constraint", ["driver.stale_air.circulation", "driver.parked.doors_locked"], [("carcontrol_airConditioner_set_circulation", {"zone": "all", "circulation": "outside"}), ("carcontrol_door_set_locked", {"door": "all", "locked": False})]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(
            "/mnt/data/hj153lee/PalmClaw/external-sources/"
            "human-derived-vehicle-dialogue"
        ),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    return parser.parse_args()


def load_external_records(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[int, dict]]]:
    envisioned_path = root / "envisioned-va" / "perfect-va_readable-dialogues_20-09-09.csv"
    with envisioned_path.open(encoding="utf-8-sig", newline="") as handle:
        envisioned = {row["id"]: row for row in csv.DictReader(handle)}
    audio_root = root / "audio2tool" / "public"
    tier7 = {
        row["id"]: row
        for row in json.loads(
            (audio_root / "tier7_multiturn_data" / "tier7_multiturn.json").read_text()
        )
    }
    tier6 = {
        row["id"]: row
        for row in json.loads(
            (audio_root / "tier6_correction_data" / "tier6_correction.json").read_text()
        )
    }
    tier4 = {
        row["id"]: row
        for row in json.loads(
            (audio_root / "tier4_implicit_data" / "tier4_implicit.json").read_text()
        )
    }
    tier5 = {
        row["id"]: row
        for row in json.loads(
            (audio_root / "tier5_needle_data" / "tier5_needle.json").read_text()
        )
    }
    return envisioned, {"tier7": tier7, "tier6": tier6, "tier4": tier4, "tier5": tier5}


def human_turns(row: dict[str, str], field: str) -> list[tuple[str, str]]:
    turns = []
    for line in row[field].replace("\r", "").strip().splitlines():
        role, text = line.split(": ", 1)
        turns.append(("assistant" if role == "voice_assistant" else "user", text.strip()))
    return turns


def calls(values: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [{"name": name, "arguments": arguments} for name, arguments in values]


def build_source(
    scenario_id: str,
    plan: dict[str, Any],
    envisioned: dict[str, dict[str, str]],
    audio_tiers: dict[str, dict[int, dict]],
) -> dict[str, Any]:
    driver = plan["driver"]
    sessions: list[dict[str, Any]] = []
    record_cutoffs: dict[int | str, str] = {}
    start = datetime(2026, 1, 5, 8, 0)

    def add_session(context: str, source_name: str, record_id: str, raw_turns: list[tuple[str, str]], update_spec: dict[str, Any] | None = None) -> None:
        session_number = len(sessions) + 1
        session_id = f"{scenario_id.lower()}-s{session_number}"
        turns = []
        for turn_index, (speaker, original_text) in enumerate(raw_turns):
            is_update = update_spec is not None and turn_index == len(raw_turns) - 1
            if is_update:
                separator = " " if original_text.endswith((".", "?", "!")) else ". "
                text = original_text + separator + update_spec["suffix"].strip()
            else:
                text = original_text
            turn = {
                "turn_id": f"{session_id}-t{turn_index + 1:02d}",
                "speaker": speaker,
                "text": text,
                "source_trace": {
                    "dataset": source_name,
                    "record_id": record_id,
                    "record_turn_index": turn_index,
                    "origin": "human_authored" if source_name.startswith("Envisioned") else "synthetic",
                    "reuse_mode": "minimal_persistence_adaptation" if is_update else "verbatim_text_role_normalized",
                    "original_text": original_text,
                },
            }
            if is_update:
                turn["update"] = update_spec["update"]
            turns.append(turn)
        sessions.append(
            {
                "session_id": session_id,
                "timestamp": (start + timedelta(days=20 * (session_number - 1))).isoformat(),
                "context": context,
                "turns": turns,
            }
        )
        if source_name == "Audio2Tool":
            canonical_id: int | str = record_id
            if record_id.startswith(("tier7:", "tier6:")):
                canonical_id = int(record_id.split(":", 1)[1])
            record_cutoffs[canonical_id] = turns[-1]["turn_id"]
            record_cutoffs[record_id] = turns[-1]["turn_id"]

    for record_id, field, context in plan["human_records"]:
        add_session(
            context,
            "Envisioned Voice Assistant Dialogues",
            f"{record_id}:{field}",
            human_turns(envisioned[record_id], field),
        )

    for record_id in plan["audio_records"]:
        if isinstance(record_id, str):
            tier_name, numeric_id = record_id.split(":", 1)
            numeric_id = int(numeric_id)
        elif record_id in audio_tiers["tier7"]:
            tier_name, numeric_id = "tier7", record_id
        else:
            tier_name, numeric_id = "tier6", record_id
        record = audio_tiers[tier_name][numeric_id]
        if tier_name == "tier7":
            raw_turns = [
                ("assistant" if turn["role"] == "agent" else "user", turn["content"].strip())
                for turn in record["chat_history"]
            ]
        else:
            raw_turns = [("user", record["query"].strip())]
        source_record_id = f"{tier_name}:{numeric_id}"
        add_session(
            f"Externally sourced vehicle-assistant record {source_record_id}.",
            "Audio2Tool",
            source_record_id,
            raw_turns,
            plan["updates"].get(record_id),
        )

    quizzes = []
    for index, (cutoff, query, reasoning, evidence, gold_calls) in enumerate(
        plan["turn_quizzes"], 1
    ):
        quizzes.append(
            {
                "quiz_id": f"{scenario_id.lower()}-turn-{index:02d}",
                "quiz_type": "TURN",
                "cutoff_turn_id": record_cutoffs[cutoff],
                "query": query,
                "reasoning_type": reasoning,
                "evidence_slot_ids": evidence,
                "gold_calls": calls(gold_calls),
            }
        )
    final_cutoff = sessions[-1]["turns"][-1]["turn_id"]
    for index, (query, reasoning, evidence, gold_calls) in enumerate(
        plan["final_quizzes"], 1
    ):
        quizzes.append(
            {
                "quiz_id": f"{scenario_id.lower()}-final-{index:02d}",
                "quiz_type": "FINAL",
                "cutoff_turn_id": final_cutoff,
                "query": query,
                "reasoning_type": reasoning,
                "evidence_slot_ids": evidence,
                "gold_calls": calls(gold_calls),
            }
        )

    human_record_ids = [f"{record_id}:{field}" for record_id, field, _ in plan["human_records"]]
    audio_record_ids = [
        record_id
        if isinstance(record_id, str)
        else f"tier7:{record_id}"
        if record_id in audio_tiers["tier7"]
        else f"tier6:{record_id}"
        for record_id in plan["audio_records"]
    ]
    return {
        "schema_version": "vehiclemembench-human-authored-scenario-source-v1",
        "scenario_id": scenario_id,
        "scenario_index": plan["scenario_index"],
        "split": "pilot_excluded",
        "authorship": {
            "kind": "external_adapted_model_assembled_pilot",
            "eligible_for_human_test": False,
            "note": "Externally adapted pilot. Dialogue records are disjoint from every other external-adapted HVP pilot; assembly, persistence edits, memory labels, and quizzes are model-produced.",
        },
        "external_sources": [
            {
                "dataset": "Envisioned Voice Assistant Dialogues",
                "url": "https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues",
                "license": "CC BY 4.0",
                "origin": "human_authored",
                "records": human_record_ids,
                "local_file_sha256": ENVISIONED_SHA256,
            },
            {
                "dataset": "Audio2Tool",
                "url": "https://huggingface.co/datasets/RVtech/Audio2Tool",
                "license": "CC BY-NC 4.0",
                "origin": "synthetic",
                "records": audio_record_ids,
                "local_file_sha256": {
                    "tier7": AUDIO_TIER7_SHA256,
                    "tier6": AUDIO_TIER6_SHA256,
                    "tier4": AUDIO_TIER4_SHA256,
                    "tier5": AUDIO_TIER5_SHA256,
                },
            },
        ],
        "speakers": [
            {
                "speaker_id": f"external-{scenario_id.lower()}-driver",
                "short_id": "user",
                "speaker_name": f"{driver} (normalized user role)",
                "role": "driver; external user identities normalized for the pilot",
                "style": "retains the wording of each external source record",
            },
            {
                "speaker_id": f"external-{scenario_id.lower()}-assistant",
                "short_id": "assistant",
                "speaker_name": "Voice assistant (source names retained in text)",
                "role": "vehicle voice assistant",
                "style": "retains the wording of each external source record",
            },
        ],
        "sessions": sessions,
        "quizzes": quizzes,
    }


def main() -> None:
    args = parse_args()
    envisioned, audio_tiers = load_external_records(args.external_root)
    root = args.repository_root / "evaluation" / "human-authored-vehicle-memory"
    for scenario_id, plan in PLANS.items():
        directory = (
            f"pilot-hv{scenario_id[-2:]}"
            if scenario_id in {"HVP01", "HVP02"}
            else f"pilot-{scenario_id.lower()}"
        )
        output = root / directory / "scenario-source.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        source = build_source(scenario_id, plan, envisioned, audio_tiers)
        output.write_text(
            json.dumps(source, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        turns = [turn for session in source["sessions"] for turn in session["turns"]]
        human_turn_count = sum(
            turn["source_trace"]["origin"] == "human_authored" for turn in turns
        )
        adapted_turn_count = sum("update" in turn for turn in turns)
        human_records = ", ".join(
            f"`{record}`" for record in source["external_sources"][0]["records"]
        )
        audio_records = ", ".join(
            f"`{record}`" for record in source["external_sources"][1]["records"]
        )
        attribution = f"""# {scenario_id} external-adaptation record

{scenario_id} is an **externally adapted, model-assembled pilot**. It is
excluded from training and does not qualify as a human-authored evaluation
set. Its external records do not overlap with any other external-adapted HVP pilot.

| Source | Records used | Origin | License | Turns |
| --- | --- | --- | --- | ---: |
| [Envisioned Voice Assistant Dialogues](https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues) | {human_records} | Written by study participants | CC BY 4.0 | {human_turn_count} |
| [Audio2Tool](https://huggingface.co/datasets/RVtech/Audio2Tool) | {audio_records} | LLM-generated query text | CC BY-NC 4.0 | {len(turns) - human_turn_count} |

- All {len(turns)} turns carry an external source trace.
- {len(turns) - adapted_turn_count} turns preserve the source text verbatim
  with only speaker-role normalization.
- {adapted_turn_count} turns append the minimum information needed to express
  a persistent, self-contained VehicleMemBench preference.
- The V2 memory operations and quizzes are newly authored for this pilot.
- Audio2Tool's non-commercial license applies to this adaptation.

The exact record IDs, original turn text, downloaded-file SHA-256 values, and
reuse mode are preserved in `scenario-source.json`.
"""
        (output.parent / "SOURCE_ATTRIBUTION.md").write_text(
            attribution,
            encoding="utf-8",
        )
        print(output)


if __name__ == "__main__":
    main()
