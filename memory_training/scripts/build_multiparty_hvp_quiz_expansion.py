"""Build versioned multi-participant HVP scenarios and their expanded quizzes.

The frozen HVP source remains untouched.  Each extension inserts disjoint
externally sourced sessions for a second vehicle user, adds owner-scoped memory
facts, and appends method-blind quizzes grounded in deterministic gold snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_training.scripts.build_external_adapted_pilots import load_external_records
from memory_training.scripts.build_human_authored_pilot import build

HVP01_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp01-codriver",
        "short_id": "co_user",
        "speaker_name": "Morgan (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp01-s6",
    "sessions": [
        {
            "session_id": "hvp01-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:544",
            "context": "Morgan adjusts a media volume after clarifying the device and level.",
            "suffix": "Twenty-five is also my regular in-car music volume.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.music_volume",
                "tool_name": "carcontrol_music_set_volume",
                "argument_name": "volume",
                "value": 25,
                "context_arguments": {},
                "condition": "for Morgan's normal music playback",
                "reason": (
                    "The externally sourced corrected volume is adapted into "
                    "Morgan's explicit persistent in-car music preference."
                ),
            },
        },
        {
            "session_id": "hvp01-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:185",
            "context": "Morgan clarifies a preference for automatic climate control.",
            "suffix": "Auto is also my normal all-zone cabin mode in this car.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.climate_mode",
                "tool_name": "carcontrol_airConditioner_set_mode",
                "argument_name": "mode",
                "value": "auto",
                "context_arguments": {"zone": "all"},
                "condition": "for Morgan's normal cabin climate",
                "reason": (
                    "The externally sourced automatic-mode choice is adapted "
                    "into Morgan's persistent all-zone cabin preference."
                ),
            },
        },
        {
            "session_id": "hvp01-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:2136",
            "context": "Morgan clarifies that the rear liftgate is used for groceries.",
            "suffix": (
                "Opening the rear liftgate is my standing routine when I load groceries."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.groceries.rear_trunk",
                "tool_name": "carcontrol_trunk_switch",
                "argument_name": "switch",
                "value": True,
                "context_arguments": {},
                "condition": "when Morgan loads groceries",
                "reason": (
                    "The externally sourced rear-liftgate request is made into "
                    "Morgan's durable grocery-loading routine."
                ),
            },
        },
    ],
    "quizzes": [
        # Coreference resolution: six distinct referential forms.
        {
            "quiz_id": "hvp01-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s9-t01",
            "query": "Devon needs what he keeps in the compartment up front. Open that one.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.tools.front_trunk"],
            "gold_calls": [
                {"name": "carcontrol_frontTrunk_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp01-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s11-t01",
            "query": (
                "Rain has started during Devon's drive, and he needs the tools he "
                "keeps up front. Apply his rules to the pane beside the driver and "
                "that front compartment."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.rain.driver_window",
                "driver.tools.front_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 0},
                },
                {"name": "carcontrol_frontTrunk_switch", "arguments": {"switch": True}},
            ],
        },
        {
            "quiz_id": "hvp01-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s13-t01",
            "query": (
                "Devon wants the bags hidden in that rear cargo compartment and "
                "the back-right seat cooled to his usual level. Apply both settings."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.hidden_bags.rear_trunk",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": False}},
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
        {
            "quiz_id": "hvp01-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s12-t01",
            "query": "Restaurant exhaust is bothering Devon. Use the cabin air-source setting he chose for that smell.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.restaurant_exhaust.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "inside"},
                }
            ],
        },
        {
            "quiz_id": "hvp01-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s10-t01",
            "query": "Devon says to use the cabin mode he normally relies on. Apply that one to every zone.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.normal.climate_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "auto"},
                }
            ],
        },
        {
            "quiz_id": "hvp01-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s8-t01",
            "query": "While pointing to the music volume, Devon says, 'Put it back where I normally keep it.' Apply his stored level.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 11}}
            ],
        },
        # Error correction: within-turn and later correction.
        {
            "quiz_id": "hvp01-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s8-t01",
            "query": "For Devon's normal music, ignore the first late-night number and use the corrected one.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 11}}
            ],
        },
        {
            "quiz_id": "hvp01-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s16-t01",
            "query": "Devon withdrew sixty; apply the corrected regular music volume he settled on.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 58}}
            ],
        },
        # Preference conflicts: the correct owner or owner-conditioned rule matters.
        {
            "quiz_id": "hvp01-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s13-t01",
            "query": (
                "Devon needs his bags kept out of sight and wants his usual music "
                "level. Apply the two preferences stored for Devon."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.hidden_bags.rear_trunk",
                "driver.normal.music_volume",
            ],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": False}},
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 11}},
            ],
        },
        {
            "quiz_id": "hvp01-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s13-t01",
            "query": "Morgan is loading groceries. Apply Morgan's stored rear-trunk routine.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.groceries.rear_trunk"],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp01-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s14-t01",
            "query": "Morgan has taken the wheel. Apply her usual cabin mode.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.climate_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "auto"},
                }
            ],
        },
        {
            "quiz_id": "hvp01-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s14-t01",
            "query": "Devon has taken the wheel. Apply his current usual cabin mode.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.climate_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                }
            ],
        },
        {
            "quiz_id": "hvp01-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s16-t01",
            "query": "Morgan is driving and asks for her regular music level. Apply the preference saved for Morgan.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 25}}
            ],
        },
        {
            "quiz_id": "hvp01-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s16-t01",
            "query": "Devon is driving and asks for his regular music level. Apply the preference saved for Devon.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 58}}
            ],
        },
        # State shifts: latest values after explicit replacements.
        {
            "quiz_id": "hvp01-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s14-t01",
            "query": "Devon's normal cabin mode changed after the windshield fogged. Apply the latest mode.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.climate_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                }
            ],
        },
        {
            "quiz_id": "hvp01-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s15-t01",
            "query": "Use Devon's latest restaurant-exhaust air setting after his change of mind.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.restaurant_exhaust.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                }
            ],
        },
        {
            "quiz_id": "hvp01-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s16-t01",
            "query": "Devon's normal music volume used to be eleven. Apply its current value after the later change.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 58}}
            ],
        },
        {
            "quiz_id": "hvp01-turn-22",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp01-s16-t01",
            "query": "Apply all three of Devon's revised preferences: normal music volume, normal cabin mode, and restaurant-exhaust air source.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.normal.music_volume",
                "driver.normal.climate_mode",
                "driver.restaurant_exhaust.circulation",
            ],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 58}},
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
            ],
        },
        {
            "quiz_id": "hvp01-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp01-s16-t01",
            "query": (
                "Morgan is driving and has groceries to load. Set her usual "
                "music level and cabin mode, then apply her stored action for "
                "the rear cargo compartment."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.music_volume",
                "codriver.normal.climate_mode",
                "codriver.groceries.rear_trunk",
            ],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 25}},
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "auto"},
                },
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": True}},
            ],
        },
    ],
}


HVP02_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp02-codriver",
        "short_id": "co_user",
        "speaker_name": "Avery (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp02-s6",
    "sessions": [
        {
            "session_id": "hvp02-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:548",
            "context": "Avery clarifies a comfortable low background volume.",
            "suffix": "Thirty is also my regular in-car music volume.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.music_volume",
                "tool_name": "carcontrol_music_set_volume",
                "argument_name": "volume",
                "value": 30,
                "context_arguments": {},
                "condition": "for Avery's normal music playback",
                "reason": (
                    "The externally sourced background-volume choice is adapted "
                    "into Avery's explicit persistent in-car music preference."
                ),
            },
        },
        {
            "session_id": "hvp02-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:332",
            "context": "Avery clarifies which entrance to secure before leaving.",
            "suffix": (
                "Locking all the vehicle doors is also my standing routine when I leave the car."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.departure.doors_locked",
                "tool_name": "carcontrol_door_set_locked",
                "argument_name": "locked",
                "value": True,
                "context_arguments": {"door": "all"},
                "condition": "when Avery is ready to leave",
                "reason": (
                    "The externally sourced departure-lock request is adapted into "
                    "Avery's persistent all-vehicle-door routine."
                ),
            },
        },
        {
            "session_id": "hvp02-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:2067",
            "context": "Avery chooses visual navigation instead of spoken guidance.",
            "suffix": "Muted guidance is also my normal in-car navigation setting.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.navigation_voice_mode",
                "tool_name": "carcontrol_navigation_set_voice_mode",
                "argument_name": "mode",
                "value": "mute",
                "context_arguments": {},
                "condition": "for Avery's normal navigation",
                "reason": (
                    "The externally sourced mute request is normalized to the "
                    "VehicleMemBench voice mode and made persistent for Avery."
                ),
            },
        },
    ],
    "quizzes": [
        # Coreference resolution: indirect objects, pronouns, and elliptical settings.
        {
            "quiz_id": "hvp02-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s8-t01",
            "query": "Robin points toward the back-right area and says, 'Put that one where I normally keep it.' Apply the remembered temperature.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.normal.rear_right_temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "rear_right", "temperature": 27},
                }
            ],
        },
        {
            "quiz_id": "hvp02-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s9-t01",
            "query": "Robin's passenger wants the cooling they normally use in that seat. Apply its remembered level.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.normal.passenger_seat_ventilation_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "passenger", "speed": 1},
                }
            ],
        },
        {
            "quiz_id": "hvp02-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s10-t01",
            "query": "Robin needs the papers kept in the compartment up front. Open that one.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.documents.front_trunk"],
            "gold_calls": [
                {"name": "carcontrol_frontTrunk_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp02-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s11-t01",
            "query": "The open liftgate is blocking Robin's view again. Put it back the way Robin specified, and restore that back-right climate area to Robin's usual temperature.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.obstructed_view.rear_trunk",
                "driver.normal.rear_right_temperature",
            ],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": False}},
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "rear_right", "temperature": 27},
                },
            ],
        },
        {
            "quiz_id": "hvp02-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s12-t01",
            "query": "Robin is ready to get out. Set all of those entry points the way Robin requested for departure, and put the passenger-seat cooling at their usual level.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.departure.doors_locked",
                "driver.normal.passenger_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "passenger", "speed": 1},
                },
            ],
        },
        {
            "quiz_id": "hvp02-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s13-t01",
            "query": "Robin has started a route. Use the way Robin normally likes to hear it.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.normal.navigation_voice_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                }
            ],
        },
        # Error correction: the final value in each correction must win.
        {
            "quiz_id": "hvp02-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s7-t01",
            "query": "Robin first said forty-five, then corrected the normal music volume. Apply the corrected value.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 52}}
            ],
        },
        {
            "quiz_id": "hvp02-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s16-t01",
            "query": "Robin withdrew ten in the latest request. Use the corrected regular music volume instead.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 5}}
            ],
        },
        # Preference conflicts: both directions of three same-Tool owner conflicts.
        {
            "quiz_id": "hvp02-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s16-t01",
            "query": "Avery is driving. Apply the regular music volume saved for Avery, not Robin's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 30}}
            ],
        },
        {
            "quiz_id": "hvp02-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s16-t01",
            "query": "Robin is driving. Apply the regular music volume saved for Robin, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 5}}
            ],
        },
        {
            "quiz_id": "hvp02-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s12-t01",
            "query": "Avery is ready to leave. Apply Avery's usual all-door setting, not Robin's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.departure.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp02-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s12-t01",
            "query": "Robin is ready to leave. Apply Robin's usual all-door setting, not Avery's, and restore Robin's current regular music volume.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.departure.doors_locked",
                "driver.normal.music_volume",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 52}},
            ],
        },
        {
            "quiz_id": "hvp02-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s13-t01",
            "query": "Avery has taken the wheel. Apply Avery's usual navigation guidance, not Robin's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.navigation_voice_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                }
            ],
        },
        {
            "quiz_id": "hvp02-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s13-t01",
            "query": "Robin has taken the wheel. Apply Robin's usual navigation guidance, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.navigation_voice_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                }
            ],
        },
        # State shifts: successive replacements in Robin's music preference.
        {
            "quiz_id": "hvp02-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s14-t01",
            "query": "Robin changed the regular music volume from fifty-two. Apply its current value after that replacement.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 25}}
            ],
        },
        {
            "quiz_id": "hvp02-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s15-t01",
            "query": "After Robin's next change, restore only the latest regular music volume.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 70}}
            ],
        },
        {
            "quiz_id": "hvp02-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s16-t01",
            "query": "Robin changed the regular music volume again after seventy. Apply the newest value.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 5}}
            ],
        },
        {
            "quiz_id": "hvp02-turn-22",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp02-s16-t01",
            "query": "Apply Robin's latest regular music volume together with the unchanged normal rear-right temperature.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.normal.music_volume",
                "driver.normal.rear_right_temperature",
            ],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 5}},
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "rear_right", "temperature": 27},
                },
            ],
        },
        {
            "quiz_id": "hvp02-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp02-s16-t01",
            "query": "Avery takes the wheel and is ready to leave. Put the music where they usually keep it, secure every door their way, and let them follow the route from the screen instead of spoken directions.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.music_volume",
                "codriver.departure.doors_locked",
                "codriver.normal.navigation_voice_mode",
            ],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 30}},
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
            ],
        },
    ],
}


HVP03_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp03-codriver",
        "short_id": "co_user",
        "speaker_name": "Jordan (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp03-s2",
    "sessions": [
        {
            "session_id": "hvp03-mp01",
            "timestamp": "2026-02-20T09:00:00",
            "record_id": "tier7:254",
            "context": "Jordan clarifies a strong but non-maximum fan speed.",
            "suffix": "Seven is also my usual front-zone fan speed in the car.",
            "update": {
                "op": "add",
                "slot_id": "codriver.front.fan_speed",
                "tool_name": "carcontrol_airConditioner_set_fan_speed",
                "argument_name": "speed",
                "value": 7,
                "context_arguments": {"zone": "front"},
                "condition": "for Jordan's normal warm-weather driving",
                "reason": (
                    "The externally sourced explicit fan speed is adapted into "
                    "Jordan's persistent front-zone vehicle preference."
                ),
            },
        },
        {
            "session_id": "hvp03-mp02",
            "timestamp": "2026-03-01T09:00:00",
            "record_id": "tier7:542",
            "context": "Jordan asks to lower an uncomfortable speaker volume.",
            "suffix": "After 10pm, twenty is also my regular in-car music volume.",
            "update": {
                "op": "add",
                "slot_id": "codriver.after_10pm.music_volume",
                "tool_name": "carcontrol_music_set_volume",
                "argument_name": "volume",
                "value": 20,
                "context_arguments": {},
                "condition": "after 10pm for Jordan",
                "reason": (
                    "The externally sourced explicit volume is adapted into "
                    "Jordan's persistent late-night in-car preference."
                ),
            },
        },
        {
            "session_id": "hvp03-mp03",
            "timestamp": "2026-03-10T09:00:00",
            "record_id": "tier6:3974",
            "context": "Jordan corrects the passenger-seat cooling level.",
            "suffix": "Level one is also my usual passenger-seat ventilation speed.",
            "update": {
                "op": "add",
                "slot_id": "codriver.passenger.seat_ventilation",
                "tool_name": "carcontrol_seat_set_ventilation_speed",
                "argument_name": "speed",
                "value": 1,
                "context_arguments": {"seat": "passenger"},
                "condition": "when Jordan uses the front passenger seat in hot weather",
                "reason": (
                    "The externally sourced corrected cooling value is adapted "
                    "into Jordan's persistent passenger-seat preference."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp03-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s3-t05",
            "query": "Avery's children are riding in the seat behind the driver. Put that area at the temperature Avery keeps for them.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.rear_left.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "rear_left", "temperature": 22},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s4-t08",
            "query": "The cabin feels warm and stuffy to Avery. Use the airflow strength and air source they normally choose for that drive.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.front.fan_speed",
                "driver.all.circulation",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 5},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
            ],
        },
        {
            "quiz_id": "hvp03-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s5-t03",
            "query": "Avery enters a hot cabin and points to all four panes. Crack those the amount they normally use.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.hot_cabin.all_windows"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "all", "degree": 15},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s5-t09",
            "query": "Avery is sitting up front beside the driver on a hot day. Apply the latest cooling speed they chose for that seat.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.passenger.seat_ventilation"],
            "gold_calls": [
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "passenger", "speed": 2},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s6-t05",
            "query": "It is late enough for Avery's quiet-hours rule. Put the music at the level tied to this time.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.after_10pm.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 4}}
            ],
        },
        {
            "quiz_id": "hvp03-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s9-t01",
            "query": "It is a cold morning for Avery. Turn on the thing warming what their hands are holding, as they most recently requested.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.steering_wheel.heating"],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s5-t09",
            "query": "Avery rejected maximum passenger-seat cooling. Apply the corrected remembered speed.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.passenger.seat_ventilation"],
            "gold_calls": [
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "passenger", "speed": 2},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s9-t01",
            "query": "Avery first said to turn the steering-wheel heat off, then reversed that request. Apply the correction.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.steering_wheel.heating"],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s7-t01",
            "query": "Jordan is driving in warm weather. Apply Jordan's usual front fan speed, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.front.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 7},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s7-t01",
            "query": "Avery is driving in warm weather. Apply Avery's latest front fan speed, not Jordan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.front.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 3},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s6-t05",
            "query": "It is after 10pm and Jordan is driving. Use Jordan's late-night music volume, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.after_10pm.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 20}}
            ],
        },
        {
            "quiz_id": "hvp03-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s6-t05",
            "query": "It is after 10pm and Avery is driving. Use Avery's late-night music volume, not Jordan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.after_10pm.music_volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 4}}
            ],
        },
        {
            "quiz_id": "hvp03-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s5-t09",
            "query": "Jordan is using the front passenger seat on a hot day. Apply Jordan's cooling speed, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.passenger.seat_ventilation"],
            "gold_calls": [
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "passenger", "speed": 1},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s5-t09",
            "query": "Avery is using the front passenger seat on a hot day. Apply Avery's latest cooling speed, not Jordan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.passenger.seat_ventilation"],
            "gold_calls": [
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "passenger", "speed": 2},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s5-t06",
            "query": "Apply Avery's passenger-seat ventilation before the later reduction, while the remembered setting is still at maximum.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.passenger.seat_ventilation"],
            "gold_calls": [
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "passenger", "speed": 3},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s5-t09",
            "query": "Avery later lowered the remembered passenger-seat ventilation. Apply the replacement value.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.passenger.seat_ventilation"],
            "gold_calls": [
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "passenger", "speed": 2},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s7-t01",
            "query": "Avery's front fan used to be five. Apply the newest normal speed after the later correction.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.front.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 3},
                }
            ],
        },
        {
            "quiz_id": "hvp03-turn-22",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp03-s9-t01",
            "query": "Avery changed the cold-morning steering-wheel setting from off to on. Apply its current state together with the latest front fan speed.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.steering_wheel.heating",
                "driver.front.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 3},
                },
            ],
        },
        {
            "quiz_id": "hvp03-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp03-s9-t01",
            "query": "Jordan is the active user after 10pm and is sitting in the front passenger seat. Set the fan and music where they normally keep them, and apply the cooling level they chose for that seat.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.front.fan_speed",
                "codriver.after_10pm.music_volume",
                "codriver.passenger.seat_ventilation",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 7},
                },
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 20}},
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "passenger", "speed": 1},
                },
            ],
        },
    ],
}


HVP04_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp04-codriver",
        "short_id": "co_user",
        "speaker_name": "Casey (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp04-s4",
    "sessions": [
        {
            "session_id": "hvp04-mp01",
            "timestamp": "2026-04-17T08:05:00",
            "record_id": "tier7:2102",
            "context": "Casey chooses a warm driver-area temperature before selecting cooling mode.",
            "suffix": "Twenty-eight is also my usual driver-zone temperature in this car.",
            "update": {
                "op": "add",
                "slot_id": "codriver.driver_zone.temperature",
                "tool_name": "carcontrol_airConditioner_set_temperature",
                "argument_name": "temperature",
                "value": 28,
                "context_arguments": {"zone": "driver"},
                "condition": "for Casey's normal driving",
                "reason": (
                    "The externally sourced vehicle temperature is scoped to "
                    "Casey's persistent driver-zone preference."
                ),
            },
        },
        {
            "session_id": "hvp04-mp02",
            "timestamp": "2026-04-22T08:05:00",
            "record_id": "tier7:241",
            "context": "Casey selects a strong but non-maximum fan speed.",
            "suffix": "Eight is also my usual front-zone fan speed in the car.",
            "update": {
                "op": "add",
                "slot_id": "codriver.front.fan_speed",
                "tool_name": "carcontrol_airConditioner_set_fan_speed",
                "argument_name": "speed",
                "value": 8,
                "context_arguments": {"zone": "front"},
                "condition": "for Casey's normal driving",
                "reason": (
                    "The externally sourced explicit fan speed is adapted into "
                    "Casey's persistent front-zone vehicle preference."
                ),
            },
        },
        {
            "session_id": "hvp04-mp03",
            "timestamp": "2026-04-27T08:05:00",
            "record_id": "tier7:340",
            "context": "Casey confirms that the entrance should be secured before leaving.",
            "suffix": "I also keep every vehicle door locked when I park at home.",
            "update": {
                "op": "add",
                "slot_id": "codriver.home_parking.doors_locked",
                "tool_name": "carcontrol_door_set_locked",
                "argument_name": "locked",
                "value": True,
                "context_arguments": {"door": "all"},
                "condition": "when Casey is parked at home",
                "reason": (
                    "The externally sourced security action is adapted into "
                    "Casey's persistent all-door home-parking rule."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp04-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s5-t05",
            "query": "Jordan points to the climate area around the driver's seat. Put that zone where Jordan normally keeps it.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.driver_zone.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 22},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s6-t03",
            "query": "Jordan has groceries and asks to open the compartment up front used for that load. Open that one.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.grocery.front_trunk"],
            "gold_calls": [
                {"name": "carcontrol_frontTrunk_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp04-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s6-t06",
            "query": "Jordan's hands are cold during a chilly drive. Apply the remembered setting for what they are holding.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.chilly_drive.steering_heat"],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s7-t09",
            "query": "Jordan wants fresh air. Use their usual source for the whole cabin and leave the pane beside them at its remembered opening.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.all.circulation",
                "driver.driver.window_position",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 20},
                },
            ],
        },
        {
            "quiz_id": "hvp04-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s8-t02",
            "query": "Jordan says the front airflow should go back to the strength they settled on. Apply that one.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.front.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 5},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s9-t03",
            "query": "Jordan is parked at home. Set every entry point their way, then restore the temperature around the driver's seat.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.home_parking.doors_locked",
                "driver.driver_zone.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 22},
                },
            ],
        },
        {
            "quiz_id": "hvp04-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s8-t02",
            "query": "Jordan withdrew fan level nine. Apply the corrected normal front-zone value.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.front.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 5},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s9-t03",
            "query": "For Jordan's driver window, ignore the suspension and parking-assist requests and use the final corrected opening.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.driver.window_position"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 90},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s5-t05",
            "query": "Casey is driving. Apply Casey's usual driver-zone temperature, not Jordan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.driver_zone.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 28},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s5-t05",
            "query": "Jordan is driving. Apply Jordan's usual driver-zone temperature, not Casey's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.driver_zone.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 22},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s8-t02",
            "query": "Casey is driving. Apply Casey's usual front fan speed, not Jordan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.front.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 8},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s8-t02",
            "query": "Jordan is driving. Apply Jordan's usual front fan speed, not Casey's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.front.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 5},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s9-t03",
            "query": "Casey is parked at home. Apply Casey's all-door rule, not Jordan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.home_parking.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s9-t03",
            "query": "Jordan is parked at home. Apply Jordan's all-door rule, not Casey's, and restore Jordan's front fan speed.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.home_parking.doors_locked",
                "driver.front.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 5},
                },
            ],
        },
        {
            "quiz_id": "hvp04-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s7-t09",
            "query": "Before Jordan's later changes, put the driver window at its first remembered normal opening.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.driver.window_position"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 20},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s8-t02",
            "query": "Jordan next changed the usual driver window to fully closed. Apply that checkpoint together with the new front fan setting.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.driver.window_position",
                "driver.front.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 0},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 5},
                },
            ],
        },
        {
            "quiz_id": "hvp04-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s9-t02",
            "query": "After the next replacement, apply Jordan's intermediate driver-window opening.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.driver.window_position"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 55},
                }
            ],
        },
        {
            "quiz_id": "hvp04-turn-22",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp04-s9-t03",
            "query": "Jordan replaced the driver-window opening once more. Apply only its newest remembered position.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.driver.window_position"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 90},
                }
            ],
        },
        {
            "quiz_id": "hvp04-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp04-s9-t03",
            "query": "Casey is driving and then parks at home. Put the driver's climate area and front airflow where they usually keep them, then secure every door their way.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.driver_zone.temperature",
                "codriver.front.fan_speed",
                "codriver.home_parking.doors_locked",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 28},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "front", "speed": 8},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                },
            ],
        },
    ],
}


HVP05_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp05-codriver",
        "short_id": "co_user",
        "speaker_name": "Taylor (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp05-s6",
    "sessions": [
        {
            "session_id": "hvp05-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:543",
            "context": "Taylor clarifies a comfortable media volume.",
            "suffix": "Fifty is also my regular in-car music volume.",
            "update": {
                "op": "add",
                "slot_id": "codriver.music.volume",
                "tool_name": "carcontrol_music_set_volume",
                "argument_name": "volume",
                "value": 50,
                "context_arguments": {},
                "condition": "for Taylor's normal music playback",
                "reason": "The explicit source volume is adapted into Taylor's persistent in-car music preference.",
            },
        },
        {
            "session_id": "hvp05-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:359",
            "context": "Taylor confirms that an entrance should be unlocked.",
            "suffix": "I also keep all vehicle doors unlocked when I park at the store.",
            "update": {
                "op": "add",
                "slot_id": "codriver.store_parking.doors_locked",
                "tool_name": "carcontrol_door_set_locked",
                "argument_name": "locked",
                "value": False,
                "context_arguments": {"door": "all"},
                "condition": "when Taylor parks at the store",
                "reason": "The source unlock action is adapted into Taylor's persistent all-door store-parking rule.",
            },
        },
        {
            "session_id": "hvp05-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:2138",
            "context": "Taylor reasons through and selects a halfway-open sunroof.",
            "suffix": "Fifty percent is also my usual sunroof opening on warm drives.",
            "update": {
                "op": "add",
                "slot_id": "codriver.warm_drive.sunroof",
                "tool_name": "carcontrol_sunroof_set_open_degree",
                "argument_name": "degree",
                "value": 50,
                "context_arguments": {},
                "condition": "for Taylor's warm-weather drives",
                "reason": "The explicit source sunroof position is made into Taylor's durable warm-drive preference.",
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp05-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s8-t03",
            "query": "Casey says the music should return to where they currently keep it. Apply that level.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.music.volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 75}}
            ],
        },
        {
            "quiz_id": "hvp05-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s11-t01",
            "query": "It is a weekday for Casey. Start guidance to the place they now mean by their regular trip.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.weekday.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Work"},
                }
            ],
        },
        {
            "quiz_id": "hvp05-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s10-t03",
            "query": "Casey has arrived at the store. Secure every entry point the way they requested there.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.store_parking.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp05-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s12-t01",
            "query": "Casey is loading shopping into the compartment under the hood. Apply their routine for that one.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.shopping.front_trunk"],
            "gold_calls": [
                {"name": "carcontrol_frontTrunk_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp05-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s13-t01",
            "query": "On this warm drive, open the roof panel above Casey to the amount they like.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.warm_drive.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 30},
                }
            ],
        },
        {
            "quiz_id": "hvp05-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s11-t01",
            "query": "Begin Casey's usual weekday journey and put the music where they currently keep it.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.weekday.destination", "driver.music.volume"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Work"},
                },
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 75}},
            ],
        },
        {
            "quiz_id": "hvp05-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s8-t03",
            "query": "Casey replaced thirty with a louder regular music level. Apply the replacement.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.music.volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 75}}
            ],
        },
        {
            "quiz_id": "hvp05-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s11-t01",
            "query": "For Casey's weekday destination, discard both the old street address and the withdrawn headquarters wording. Navigate to the corrected place.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.weekday.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Work"},
                }
            ],
        },
        {
            "quiz_id": "hvp05-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s8-t03",
            "query": "Taylor is listening. Apply Taylor's regular music volume, not Casey's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.music.volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 50}}
            ],
        },
        {
            "quiz_id": "hvp05-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s8-t03",
            "query": "Casey is listening. Apply Casey's current regular music volume, not Taylor's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.music.volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 75}}
            ],
        },
        {
            "quiz_id": "hvp05-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s10-t03",
            "query": "Taylor is parked at the store. Use Taylor's all-door setting, not Casey's, and restore Taylor's music volume.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.store_parking.doors_locked",
                "codriver.music.volume",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 50}},
            ],
        },
        {
            "quiz_id": "hvp05-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s10-t03",
            "query": "Casey is parked at the store. Use Casey's all-door setting, not Taylor's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.store_parking.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp05-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s13-t01",
            "query": "Taylor is taking a warm drive. Apply Taylor's sunroof opening, not Casey's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.warm_drive.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 50},
                }
            ],
        },
        {
            "quiz_id": "hvp05-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s13-t01",
            "query": "Casey is taking a warm drive. Apply Casey's sunroof opening, not Taylor's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.warm_drive.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 30},
                }
            ],
        },
        {
            "quiz_id": "hvp05-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s7-t03",
            "query": "Before Casey changed the music preference, apply its first remembered value.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.music.volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 30}}
            ],
        },
        {
            "quiz_id": "hvp05-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s8-t03",
            "query": "After Casey's volume replacement, apply only the newest regular music level.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.music.volume"],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 75}}
            ],
        },
        {
            "quiz_id": "hvp05-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp05-s11-t01",
            "query": "Casey's weekday destination used to be a street address. Navigate to the current destination after its later replacement.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.weekday.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Work"},
                }
            ],
        },
        {
            "quiz_id": "hvp05-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp05-s13-t01",
            "query": "Taylor is taking a warm drive and then parks at the store. Put the music and roof panel where they like them, then leave every door in their stored state.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.music.volume",
                "codriver.warm_drive.sunroof",
                "codriver.store_parking.doors_locked",
            ],
            "gold_calls": [
                {"name": "carcontrol_music_set_volume", "arguments": {"volume": 50}},
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 50},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
            ],
        },
    ],
}


HVP06_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp06-codriver",
        "short_id": "co_user",
        "speaker_name": "Quinn (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp06-s6",
    "sessions": [
        {
            "session_id": "hvp06-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier5:4236",
            "context": "Quinn asks to clear condensation from only the front windshield.",
            "suffix": "Front-only defrost is also my regular setting on cold drives.",
            "update": {
                "op": "add",
                "slot_id": "codriver.cold_drive.defrost",
                "tool_name": "carcontrol_airConditioner_set_mode",
                "argument_name": "mode",
                "value": "defrost",
                "context_arguments": {"zone": "front"},
                "condition": "for Quinn's cold drives",
                "reason": "The vehicle-specific front-defrost request is made into Quinn's durable cold-drive setting.",
            },
        },
        {
            "session_id": "hvp06-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier5:3989",
            "context": "Quinn distinguishes an emergency filtration request from their usual cabin-air preference.",
            "suffix": "For ordinary driving, though, just recirculate the cabin air; make that my usual setting.",
            "update": {
                "op": "add",
                "slot_id": "codriver.air_quality.circulation",
                "tool_name": "carcontrol_airConditioner_set_circulation",
                "argument_name": "circulation",
                "value": "inside",
                "context_arguments": {"zone": "all"},
                "condition": "for Quinn's normal cabin-air setting",
                "reason": "Quinn separately states an explicit, durable recirculation preference for ordinary driving.",
            },
        },
        {
            "session_id": "hvp06-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier6:4271",
            "context": "Quinn corrects an unrelated door request to closing the rear trunk.",
            "suffix": "Keeping it closed is also my regular routine when my bags are loaded.",
            "update": {
                "op": "add",
                "slot_id": "codriver.luggage.rear_trunk",
                "tool_name": "carcontrol_trunk_switch",
                "argument_name": "switch",
                "value": False,
                "context_arguments": {},
                "condition": "when Quinn's bags are loaded",
                "reason": "The corrected rear-trunk close action is made into Quinn's recurring luggage routine.",
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp06-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s11-t01",
            "query": "Riley starts guidance but leaves the place unstated. Take them to the location that now fills that blank.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                }
            ],
        },
        {
            "quiz_id": "hvp06-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s13-t01",
            "query": "It is a cold drive for Riley. Clear both pieces of glass they most recently included.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.cold_drive.defrost"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                }
            ],
        },
        {
            "quiz_id": "hvp06-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s12-t01",
            "query": "Riley wants the cabin air drawn the way they now prefer it. Apply that source to every zone and start their default route.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.air_quality.circulation",
                "driver.default.destination",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                },
            ],
        },
        {
            "quiz_id": "hvp06-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s10-t03",
            "query": "Riley is loading bags. Open the cargo compartment at the back that they use for those.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.luggage.rear_trunk"],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp06-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s11-t01",
            "query": "Riley did not name a place and has bags to load. Start the route they mean and open the compartment they use.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.default.destination",
                "driver.luggage.rear_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                },
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": True}},
            ],
        },
        {
            "quiz_id": "hvp06-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s13-t01",
            "query": "On Riley's cold trip, clear the glass in their latest scope and use their current cabin-air source.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.cold_drive.defrost",
                "driver.air_quality.circulation",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
            ],
        },
        {
            "quiz_id": "hvp06-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s11-t01",
            "query": "Riley canceled the old London address and corrected the unnamed destination. Navigate to the replacement.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                }
            ],
        },
        {
            "quiz_id": "hvp06-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s12-t01",
            "query": "Riley replaced inside air after rejecting Bioweapon Defense. Apply the corrected normal circulation.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.air_quality.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                }
            ],
        },
        {
            "quiz_id": "hvp06-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s13-t01",
            "query": "Quinn is on a cold drive. Apply Quinn's stored defrost scope, not Riley's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.cold_drive.defrost"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "front", "mode": "defrost"},
                }
            ],
        },
        {
            "quiz_id": "hvp06-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s13-t01",
            "query": "Riley is on a cold drive. Apply Riley's current stored defrost scope, not Quinn's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.cold_drive.defrost"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                }
            ],
        },
        {
            "quiz_id": "hvp06-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s12-t01",
            "query": "Quinn is using the cabin. Apply Quinn's normal air source, not Riley's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.air_quality.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "inside"},
                }
            ],
        },
        {
            "quiz_id": "hvp06-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s12-t01",
            "query": "Riley is using the cabin. Apply Riley's normal air source, not Quinn's, and start Riley's default route.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.air_quality.circulation",
                "driver.default.destination",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                },
            ],
        },
        {
            "quiz_id": "hvp06-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s10-t03",
            "query": "Quinn's bags are loaded. Apply Quinn's rear-trunk routine, not Riley's, and use Quinn's cabin-air source.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.luggage.rear_trunk",
                "codriver.air_quality.circulation",
            ],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": False}},
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "inside"},
                },
            ],
        },
        {
            "quiz_id": "hvp06-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s10-t03",
            "query": "Riley is loading bags. Apply Riley's rear-trunk routine, not Quinn's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.luggage.rear_trunk"],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp06-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s8-t03",
            "query": "At this earlier cutoff, apply Riley's cold-drive defrost scope that was active then.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.cold_drive.defrost"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "front", "mode": "defrost"},
                }
            ],
        },
        {
            "quiz_id": "hvp06-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s12-t01",
            "query": "Riley's cabin-air preference has changed. Apply the value active at this cutoff and start the latest default route.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.air_quality.circulation",
                "driver.default.destination",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                },
            ],
        },
        {
            "quiz_id": "hvp06-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp06-s13-t01",
            "query": "Riley revised the cold-drive defrost scope. Apply the value active at this later cutoff.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.cold_drive.defrost"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                }
            ],
        },
        {
            "quiz_id": "hvp06-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp06-s13-t01",
            "query": "Quinn is on a cold trip with the bags already loaded. Clear only the glass they specified, use their normal cabin-air source, and leave the rear cargo compartment in their stored state.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.cold_drive.defrost",
                "codriver.air_quality.circulation",
                "codriver.luggage.rear_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "front", "mode": "defrost"},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "inside"},
                },
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": False}},
            ],
        },
    ],
}


HVP07_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp07-codriver",
        "short_id": "co_user",
        "speaker_name": "Avery (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp07-s6",
    "sessions": [
        {
            "session_id": "hvp07-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier6:4137",
            "context": "Avery corrects a request to mute guidance and keeps spoken directions enabled.",
            "suffix": "Keeping detailed spoken guidance on is also my normal navigation setting.",
            "update": {
                "op": "add",
                "slot_id": "codriver.navigation.voice_mode",
                "tool_name": "carcontrol_navigation_set_voice_mode",
                "argument_name": "mode",
                "value": "detailed",
                "context_arguments": {},
                "condition": "for Avery's normal navigation",
                "reason": "The corrected source request to keep voice guidance on is normalized to detailed guidance and made durable for Avery.",
            },
        },
        {
            "session_id": "hvp07-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier6:3990",
            "context": "Avery corrects an emergency filtration request to fresh outside air.",
            "suffix": "Fresh outside air is also my usual cabin setting.",
            "update": {
                "op": "add",
                "slot_id": "codriver.cabin_air.circulation",
                "tool_name": "carcontrol_airConditioner_set_circulation",
                "argument_name": "circulation",
                "value": "outside",
                "context_arguments": {"zone": "all"},
                "condition": "for Avery's normal cabin-air setting",
                "reason": "The explicit correction to fresh air is normalized to outside circulation and made durable for Avery.",
            },
        },
        {
            "session_id": "hvp07-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier6:4097",
            "context": "Avery clarifies that the selected playlist should be paused.",
            "suffix": "When I have selected a playlist, I normally want playback paused until I ask to resume.",
            "update": {
                "op": "add",
                "slot_id": "codriver.playlist.music",
                "tool_name": "carcontrol_music_switch",
                "argument_name": "switch",
                "value": False,
                "context_arguments": {},
                "condition": "when Avery has selected a playlist",
                "reason": "Avery explicitly makes paused playback the durable preference after selecting a playlist.",
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp07-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s11-t01",
            "query": "It is a warm drive for Morgan. Put the roof opening at the value they most recently settled on.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.warm_drive.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s10-t01",
            "query": "Set every cabin zone to the air source Morgan now uses as normal.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.cabin_air.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s12-t01",
            "query": "Morgan has chosen a playlist and is loading equipment. Apply the remembered playback and cargo routines.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.playlist.music",
                "driver.equipment.front_trunk",
            ],
            "gold_calls": [
                {"name": "carcontrol_music_switch", "arguments": {"switch": True}},
                {"name": "carcontrol_frontTrunk_switch", "arguments": {"switch": True}},
            ],
        },
        {
            "quiz_id": "hvp07-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s12-t01",
            "query": "Morgan is loading their usual gear. Open the compartment where they normally put it.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.equipment.front_trunk"],
            "gold_calls": [
                {"name": "carcontrol_frontTrunk_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp07-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s9-t03",
            "query": "Avery starts navigation with a playlist selected. Apply Avery's usual guidance and playback choices.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.navigation.voice_mode",
                "codriver.playlist.music",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                },
                {"name": "carcontrol_music_switch", "arguments": {"switch": False}},
            ],
        },
        {
            "quiz_id": "hvp07-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s11-t01",
            "query": "Morgan corrected the warm-drive roof opening. Apply the value active after that correction.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.warm_drive.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s10-t01",
            "query": "Morgan corrected the normal cabin-air source. Apply the value active after that correction to every zone.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.cabin_air.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-mp01-t01",
            "query": "Avery withdrew the request for silent directions and kept voice guidance on. Apply that correction.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["codriver.navigation.voice_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s9-t03",
            "query": "Avery is navigating. Apply Avery's normal guidance setting, not Morgan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.navigation.voice_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s9-t03",
            "query": "Morgan is navigating. Apply Morgan's normal guidance setting, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.navigation.voice_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s9-t03",
            "query": "Avery is using the cabin. Apply Avery's normal cabin-air source, not Morgan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.cabin_air.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s9-t03",
            "query": "Morgan is using the cabin. Apply Morgan's normal cabin-air source, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.cabin_air.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "inside"},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s9-t03",
            "query": "Avery has selected a playlist. Apply Avery's playback and cabin-air preferences, not Morgan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.playlist.music",
                "codriver.cabin_air.circulation",
            ],
            "gold_calls": [
                {"name": "carcontrol_music_switch", "arguments": {"switch": False}},
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
            ],
        },
        {
            "quiz_id": "hvp07-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s9-t03",
            "query": "Morgan has selected a playlist. Apply Morgan's playback preference, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.playlist.music"],
            "gold_calls": [
                {"name": "carcontrol_music_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp07-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s8-t03",
            "query": "At this earlier cutoff, apply Morgan's normal cabin-air source that was active then.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.cabin_air.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "inside"},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s10-t01",
            "query": "At this later cutoff, apply Morgan's revised normal cabin-air source.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.cabin_air.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                }
            ],
        },
        {
            "quiz_id": "hvp07-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp07-s11-t01",
            "query": "At this later cutoff, apply Morgan's revised warm-drive sunroof opening.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.warm_drive.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp07-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp07-s12-t01",
            "query": "Avery begins navigation with a playlist selected. Use their normal spoken-guidance, cabin-air, and playback settings.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.navigation.voice_mode",
                "codriver.cabin_air.circulation",
                "codriver.playlist.music",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
                {"name": "carcontrol_music_switch", "arguments": {"switch": False}},
            ],
        },
    ],
}


HVP08_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp08-codriver",
        "short_id": "co_user",
        "speaker_name": "Blake (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp08-s6",
    "sessions": [
        {
            "session_id": "hvp08-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:2103",
            "context": "Blake resolves an underspecified climate request to 22 degrees in automatic mode.",
            "suffix": "Keep twenty-two degrees as my normal driver-zone temperature.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.temperature",
                "tool_name": "carcontrol_airConditioner_set_temperature",
                "argument_name": "temperature",
                "value": 22,
                "context_arguments": {"zone": "driver"},
                "condition": "for Blake's normal driving",
                "reason": "Blake explicitly makes the source temperature their durable driver-zone setting.",
            },
        },
        {
            "session_id": "hvp08-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:2055",
            "context": "Blake resolves an underspecified ambient-light request to bright red lighting.",
            "suffix": "Red is also the ambient-light color I normally want in this car.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.ambient_color",
                "tool_name": "carcontrol_light_set_ambient_color",
                "argument_name": "color",
                "value": "red",
                "context_arguments": {},
                "condition": "for Blake's normal ambient lighting",
                "reason": "The source explicitly selects red, which Blake makes into a durable in-car ambient-light preference.",
            },
        },
        {
            "session_id": "hvp08-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier5:4270",
            "context": "Blake asks to close the rear trunk so loaded bags stay out of sight.",
            "suffix": "When I load bulky items, don't open it automatically; keep it closed until I explicitly ask.",
            "update": {
                "op": "add",
                "slot_id": "codriver.bulky_items.rear_trunk",
                "tool_name": "carcontrol_trunk_switch",
                "argument_name": "switch",
                "value": False,
                "context_arguments": {},
                "condition": "when Blake loads bulky items",
                "reason": "The explicit close request is extended into Blake's durable preference against automatically opening the rear trunk for bulky items.",
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp08-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s14-t01",
            "query": "Set Avery's passenger-front window to the position they most recently chose for normal driving.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.normal.passenger_window"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "passenger", "degree": 10},
                }
            ],
        },
        {
            "quiz_id": "hvp08-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s15-t01",
            "query": "Avery starts navigation without naming the place. Use the destination that now completes the request.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {
                        "destination": "Eastside Public Library on Maple Street"
                    },
                }
            ],
        },
        {
            "quiz_id": "hvp08-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s16-t01",
            "query": "Avery is parked and about to leave, then starts navigation without a destination. Apply the remembered door state and route.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.vehicle_access.doors_locked",
                "driver.default.destination",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {
                        "destination": "Eastside Public Library on Maple Street"
                    },
                },
            ],
        },
        {
            "quiz_id": "hvp08-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s13-t01",
            "query": "Avery begins a normal drive and wants their usual cabin mood. Apply the remembered driver temperature and light color.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.normal.ambient_color",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 18},
                },
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
            ],
        },
        {
            "quiz_id": "hvp08-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s13-t01",
            "query": "Blake begins a normal drive and wants their usual cabin mood. Apply Blake's temperature and light color.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.temperature",
                "codriver.normal.ambient_color",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 22},
                },
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                },
            ],
        },
        {
            "quiz_id": "hvp08-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s14-t01",
            "query": "Avery corrected the normal passenger-window position. Apply the value active after that correction.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.passenger_window"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "passenger", "degree": 10},
                }
            ],
        },
        {
            "quiz_id": "hvp08-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s15-t01",
            "query": "Avery corrected the unnamed default destination. Navigate to the value active after that correction.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {
                        "destination": "Eastside Public Library on Maple Street"
                    },
                }
            ],
        },
        {
            "quiz_id": "hvp08-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s16-t01",
            "query": "Avery corrected the parked-car departure rule. Apply its current state and start the current default route.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": [
                "driver.vehicle_access.doors_locked",
                "driver.default.destination",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {
                        "destination": "Eastside Public Library on Maple Street"
                    },
                },
            ],
        },
        {
            "quiz_id": "hvp08-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s13-t01",
            "query": "Blake is driving normally. Apply Blake's driver-zone temperature, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 22},
                }
            ],
        },
        {
            "quiz_id": "hvp08-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s13-t01",
            "query": "Avery is driving normally. Apply Avery's driver-zone temperature, not Blake's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp08-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s13-t01",
            "query": "Blake wants their usual ambient lighting. Apply Blake's stored color, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                }
            ],
        },
        {
            "quiz_id": "hvp08-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s13-t01",
            "query": "Avery wants their usual ambient lighting. Apply Avery's stored color, not Blake's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                }
            ],
        },
        {
            "quiz_id": "hvp08-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s13-t01",
            "query": "Blake is preparing to load bulky items. Apply Blake's stored rear-trunk routine and normal temperature, not Avery's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.bulky_items.rear_trunk",
                "codriver.normal.temperature",
            ],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": False}},
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 22},
                },
            ],
        },
        {
            "quiz_id": "hvp08-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s13-t01",
            "query": "Avery is loading a bulky item. Apply Avery's stored rear-trunk routine, not Blake's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.bulky_items.rear_trunk"],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp08-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s8-t01",
            "query": "At this earlier cutoff, apply Avery's normal passenger-window position that was active then.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.passenger_window"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "passenger", "degree": 60},
                }
            ],
        },
        {
            "quiz_id": "hvp08-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s14-t01",
            "query": "At this later cutoff, apply Avery's revised normal passenger-window position.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.passenger_window"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "passenger", "degree": 10},
                }
            ],
        },
        {
            "quiz_id": "hvp08-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp08-s15-t01",
            "query": "At this later cutoff, start guidance to Avery's revised unnamed default destination.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {
                        "destination": "Eastside Public Library on Maple Street"
                    },
                }
            ],
        },
        {
            "quiz_id": "hvp08-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp08-s16-t01",
            "query": "Blake starts a normal drive and is preparing to load a bulky item without asking to open the trunk. Apply their usual temperature, lighting, and stored trunk state.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.temperature",
                "codriver.normal.ambient_color",
                "codriver.bulky_items.rear_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 22},
                },
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                },
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": False}},
            ],
        },
        {
            "quiz_id": "hvp08-final-08",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp08-s16-t01",
            "query": "Avery, not Blake, starts a normal drive and loads a bulky item. Apply Avery's temperature, lighting, and rear-trunk routine.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.normal.ambient_color",
                "driver.bulky_items.rear_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 18},
                },
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": True}},
            ],
        },
    ],
}


HVP09_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp09-codriver",
        "short_id": "co_user",
        "speaker_name": "Cameron (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp09-s6",
    "sessions": [
        {
            "session_id": "hvp09-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:2054",
            "context": "Cameron resolves an invalid brightness request and selects soft blue ambient lighting.",
            "suffix": "Soft blue is also the ambient color I prefer for a romantic cabin mood.",
            "update": {
                "op": "add",
                "slot_id": "codriver.romantic.ambient_color",
                "tool_name": "carcontrol_light_set_ambient_color",
                "argument_name": "color",
                "value": "blue",
                "context_arguments": {},
                "condition": "when Cameron wants a romantic cabin mood",
                "reason": "Cameron explicitly makes the source blue color their durable romantic-cabin preference; source brightness remains non-durable.",
            },
        },
        {
            "session_id": "hvp09-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier4:4258",
            "context": "Cameron implicitly asks to close a rear trunk that was left open.",
            "suffix": "When I load heavy shopping, don't open it automatically; keep it closed until I explicitly ask.",
            "update": {
                "op": "add",
                "slot_id": "codriver.heavy_shopping.rear_trunk",
                "tool_name": "carcontrol_trunk_switch",
                "argument_name": "switch",
                "value": False,
                "context_arguments": {},
                "condition": "when Cameron loads heavy shopping",
                "reason": "The source close intent is extended into Cameron's durable preference against automatic rear-trunk opening for heavy shopping.",
            },
        },
        {
            "session_id": "hvp09-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier6:4138",
            "context": "Cameron corrects muted navigation instructions to voice guidance on.",
            "suffix": "When I'm listening to an audiobook, keep detailed spoken guidance on as my normal setting.",
            "update": {
                "op": "add",
                "slot_id": "codriver.audiobook.voice_mode",
                "tool_name": "carcontrol_navigation_set_voice_mode",
                "argument_name": "mode",
                "value": "detailed",
                "context_arguments": {},
                "condition": "when Cameron listens to an audiobook",
                "reason": "The corrected unmute request is explicitly normalized to Cameron's durable detailed-guidance preference during audiobooks.",
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp09-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s15-t01",
            "query": "Set every cabin zone to the fan speed Jordan most recently chose as normal.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                }
            ],
        },
        {
            "quiz_id": "hvp09-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s17-t01",
            "query": "Jordan is parked, about to leave on a cold day. Apply the remembered access and steering-wheel settings.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.vehicle_access.doors_locked",
                "driver.steering_wheel.heat_enabled",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                },
            ],
        },
        {
            "quiz_id": "hvp09-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s12-t01",
            "query": "Jordan wants a romantic cabin on a bright day. Apply the remembered light color and roof opening.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.romantic.ambient_color",
                "driver.bright_day.sunroof",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                },
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 15},
                },
            ],
        },
        {
            "quiz_id": "hvp09-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s12-t01",
            "query": "Jordan is listening to an audiobook while loading heavy shopping. Apply the remembered guidance and cargo choices.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.audiobook.voice_mode",
                "driver.heavy_shopping.rear_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": True}},
            ],
        },
        {
            "quiz_id": "hvp09-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s12-t01",
            "query": "Cameron wants a romantic cabin while listening to an audiobook. Apply Cameron's remembered lighting and guidance.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.romantic.ambient_color",
                "codriver.audiobook.voice_mode",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                },
            ],
        },
        {
            "quiz_id": "hvp09-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s15-t01",
            "query": "Jordan corrected the normal cabin fan speed. Apply the value active after that correction.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                }
            ],
        },
        {
            "quiz_id": "hvp09-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s16-t01",
            "query": "Jordan corrected the parked-car departure rule. Apply its current state and their current fan speed.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": [
                "driver.vehicle_access.doors_locked",
                "driver.normal.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                },
            ],
        },
        {
            "quiz_id": "hvp09-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s17-t01",
            "query": "Jordan corrected the cold-drive steering-wheel setting. Apply the state active after that correction.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.steering_wheel.heat_enabled"],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                }
            ],
        },
        {
            "quiz_id": "hvp09-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s12-t01",
            "query": "Cameron wants a romantic cabin. Apply Cameron's stored lighting color, not Jordan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.romantic.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                }
            ],
        },
        {
            "quiz_id": "hvp09-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s12-t01",
            "query": "Jordan wants a romantic cabin. Apply Jordan's stored lighting color, not Cameron's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.romantic.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                }
            ],
        },
        {
            "quiz_id": "hvp09-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s12-t01",
            "query": "Cameron is preparing to load heavy shopping. Apply Cameron's stored rear-trunk and cabin-color preferences, not Jordan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.heavy_shopping.rear_trunk",
                "codriver.romantic.ambient_color",
            ],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": False}},
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
            ],
        },
        {
            "quiz_id": "hvp09-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s12-t01",
            "query": "Jordan is loading heavy shopping. Apply Jordan's stored rear-trunk routine, not Cameron's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.heavy_shopping.rear_trunk"],
            "gold_calls": [
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": True}}
            ],
        },
        {
            "quiz_id": "hvp09-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s12-t01",
            "query": "Cameron is listening to an audiobook. Apply Cameron's guidance preference, not Jordan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.audiobook.voice_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                }
            ],
        },
        {
            "quiz_id": "hvp09-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s12-t01",
            "query": "Jordan is listening to an audiobook. Apply Jordan's guidance preference, not Cameron's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.audiobook.voice_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                }
            ],
        },
        {
            "quiz_id": "hvp09-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s13-t01",
            "query": "At this earlier cutoff, apply Jordan's parked-car door rule that was active then.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.vehicle_access.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp09-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s16-t01",
            "query": "At this later cutoff, apply Jordan's revised departure-door state and current fan speed.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.vehicle_access.doors_locked",
                "driver.normal.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                },
            ],
        },
        {
            "quiz_id": "hvp09-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp09-s17-t01",
            "query": "At this later cutoff, apply Jordan's revised cold-drive steering-wheel state.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.steering_wheel.heat_enabled"],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                }
            ],
        },
        {
            "quiz_id": "hvp09-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp09-s17-t01",
            "query": "Cameron wants a romantic cabin while listening to an audiobook and is preparing to load heavy shopping without asking to open the trunk. Apply Cameron's three remembered choices.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.romantic.ambient_color",
                "codriver.audiobook.voice_mode",
                "codriver.heavy_shopping.rear_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                },
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": False}},
            ],
        },
        {
            "quiz_id": "hvp09-final-08",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp09-s17-t01",
            "query": "Jordan, not Cameron, wants a romantic cabin while listening to an audiobook and loading heavy shopping. Apply Jordan's three preferences.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.romantic.ambient_color",
                "driver.audiobook.voice_mode",
                "driver.heavy_shopping.rear_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
                {"name": "carcontrol_trunk_switch", "arguments": {"switch": True}},
            ],
        },
    ],
}


HVP10_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp10-codriver",
        "short_id": "co_user",
        "speaker_name": "Dakota (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp10-s6",
    "sessions": [
        {
            "session_id": "hvp10-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier4:4194",
            "context": "Dakota reports that the vehicle is much too cold; the source label resolves the implicit request to 28°C heat.",
            "suffix": "Set the front-passenger zone to twenty-eight degrees, and remember that as my normal setting there.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.passenger_temperature",
                "tool_name": "carcontrol_airConditioner_set_temperature",
                "argument_name": "temperature",
                "value": 28,
                "context_arguments": {"zone": "passenger"},
                "condition": "for Dakota's normal passenger-zone climate",
                "reason": "The source-labeled heat intent is made explicit, scoped to the passenger zone, and restated as Dakota's durable 28-degree preference.",
            },
        },
        {
            "session_id": "hvp10-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:2118",
            "context": "Dakota clarifies that both front and rear defrosters should be used.",
            "suffix": "When I need clear rear visibility, I still want both front and rear defrosted.",
            "update": {
                "op": "add",
                "slot_id": "codriver.rear_visibility.defrost",
                "tool_name": "carcontrol_airConditioner_set_mode",
                "argument_name": "mode",
                "value": "defrost",
                "context_arguments": {"zone": "all"},
                "condition": "when Dakota needs to clear the rear glass",
                "reason": "The source explicitly resolves defrost to both panes, which Dakota makes durable for rear-visibility needs.",
            },
        },
        {
            "session_id": "hvp10-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:1990",
            "context": "Dakota chooses steering-wheel heat to warm cold hands without heating the cabin.",
            "suffix": "Whenever my hands feel cold, keep the steering-wheel heater enabled for me.",
            "update": {
                "op": "add",
                "slot_id": "codriver.cold_hands.wheel_heat",
                "tool_name": "carcontrol_steeringWheel_set_heating_enabled",
                "argument_name": "enabled",
                "value": True,
                "context_arguments": {},
                "condition": "when Dakota's hands feel cold",
                "reason": "Dakota's explicit source choice is restated as a durable cold-hands steering-wheel preference.",
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp10-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s14-t01",
            "query": "Set every cabin zone to the fan speed Taylor most recently chose as normal.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 7},
                }
            ],
        },
        {
            "quiz_id": "hvp10-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s16-t01",
            "query": "Taylor begins a normal drive while the children find the rear cabin stuffy. Apply both remembered window positions.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.passenger_window",
                "driver.kids_stuffy.rear_right_window",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "passenger", "degree": 10},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "rear_right", "degree": 60},
                },
            ],
        },
        {
            "quiz_id": "hvp10-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s15-t01",
            "query": "Taylor wants to vent the cabin and needs clear rear visibility. Apply the current roof and defrost choices.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.venting.sunroof",
                "driver.rear_visibility.defrost",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 30},
                },
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "rear", "mode": "defrost"},
                },
            ],
        },
        {
            "quiz_id": "hvp10-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s13-t01",
            "query": "Dakota takes the passenger seat and needs clear rear visibility. Apply their remembered temperature and defrost scope.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.passenger_temperature",
                "codriver.rear_visibility.defrost",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "passenger", "temperature": 28},
                },
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                },
            ],
        },
        {
            "quiz_id": "hvp10-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s13-t01",
            "query": "Dakota is driving with cold hands and is also setting the front-passenger zone. Apply their remembered wheel-heat and passenger-temperature settings.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.cold_hands.wheel_heat",
                "codriver.normal.passenger_temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "passenger", "temperature": 28},
                },
            ],
        },
        {
            "quiz_id": "hvp10-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s14-t01",
            "query": "Taylor corrected the normal cabin fan speed. Apply the value active after that correction.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 7},
                }
            ],
        },
        {
            "quiz_id": "hvp10-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s15-t01",
            "query": "Taylor corrected the sunroof vent position. Apply the value active after that correction.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.venting.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 30},
                }
            ],
        },
        {
            "quiz_id": "hvp10-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s16-t01",
            "query": "Taylor corrected the normal passenger-window position. Apply the value active after that correction.",
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.passenger_window"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "passenger", "degree": 10},
                }
            ],
        },
        {
            "quiz_id": "hvp10-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s13-t01",
            "query": "Dakota is in the front passenger seat. Apply Dakota's normal temperature, not Taylor's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.passenger_temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "passenger", "temperature": 28},
                }
            ],
        },
        {
            "quiz_id": "hvp10-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s13-t01",
            "query": "Taylor controls the front passenger climate. Apply Taylor's normal temperature, not Dakota's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.passenger_temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "passenger", "temperature": 23},
                }
            ],
        },
        {
            "quiz_id": "hvp10-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s13-t01",
            "query": "Dakota needs clear rear visibility. Apply Dakota's stored defrost scope, not Taylor's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.rear_visibility.defrost"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                }
            ],
        },
        {
            "quiz_id": "hvp10-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s13-t01",
            "query": "Taylor needs clear rear visibility. Apply Taylor's stored defrost scope, not Dakota's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.rear_visibility.defrost"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "rear", "mode": "defrost"},
                }
            ],
        },
        {
            "quiz_id": "hvp10-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s13-t01",
            "query": "Dakota takes the passenger seat and needs rear visibility. Apply Dakota's temperature and defrost preferences, not Taylor's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.passenger_temperature",
                "codriver.rear_visibility.defrost",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "passenger", "temperature": 28},
                },
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                },
            ],
        },
        {
            "quiz_id": "hvp10-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s13-t01",
            "query": "Apply Taylor's passenger-climate and rear-visibility preferences, not Dakota's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.passenger_temperature",
                "driver.rear_visibility.defrost",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "passenger", "temperature": 23},
                },
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "rear", "mode": "defrost"},
                },
            ],
        },
        {
            "quiz_id": "hvp10-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s12-t01",
            "query": "At this earlier cutoff, apply Taylor's sunroof vent position that was active then.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.venting.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 10},
                }
            ],
        },
        {
            "quiz_id": "hvp10-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s15-t01",
            "query": "At this later cutoff, apply Taylor's revised sunroof vent position.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.venting.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 30},
                }
            ],
        },
        {
            "quiz_id": "hvp10-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp10-s16-t01",
            "query": "At this later cutoff, apply Taylor's revised normal passenger-window position.",
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.passenger_window"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "passenger", "degree": 10},
                }
            ],
        },
        {
            "quiz_id": "hvp10-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp10-s16-t01",
            "query": "Dakota is driving with cold hands, is setting the front-passenger zone, and needs clear rear visibility. Apply all three of Dakota's remembered settings.",
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.passenger_temperature",
                "codriver.rear_visibility.defrost",
                "codriver.cold_hands.wheel_heat",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "passenger", "temperature": 28},
                },
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                },
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                },
            ],
        },
        {
            "quiz_id": "hvp10-final-08",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp10-s16-t01",
            "query": "For the front passenger and rear visibility, apply Taylor's stored settings rather than Dakota's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.passenger_temperature",
                "driver.rear_visibility.defrost",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "passenger", "temperature": 23},
                },
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "rear", "mode": "defrost"},
                },
            ],
        },
    ],
}


HVP11_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp11-codriver",
        "short_id": "co_user",
        "speaker_name": "Emery (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp11-s6",
    "sessions": [
        {
            "session_id": "hvp11-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier6:4136",
            "context": (
                "Emery corrects a route-preference request into a request to turn "
                "navigation voice guidance back on."
            ),
            "suffix": (
                "When I listen to an audiobook, detailed guidance is my usual "
                "navigation setting."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.audiobook.voice_mode",
                "tool_name": "carcontrol_navigation_set_voice_mode",
                "argument_name": "mode",
                "value": "detailed",
                "context_arguments": {},
                "condition": "when Emery listens to an audiobook",
                "reason": (
                    "The vehicle source's corrected guidance-on action is "
                    "normalized to detailed guidance and made durable for Emery's "
                    "audiobook listening."
                ),
            },
        },
        {
            "session_id": "hvp11-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier4:125",
            "context": (
                "Emery expresses a red lighting mood; the source's explicit red "
                "HSB action is adapted to vehicle ambient lighting."
            ),
            "suffix": "Red is also my normal in-car ambient-light color.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.ambient_color",
                "tool_name": "carcontrol_light_set_ambient_color",
                "argument_name": "color",
                "value": "red",
                "context_arguments": {},
                "condition": "for Emery's normal ambient lighting",
                "reason": (
                    "The source explicitly encodes a red light color; only the "
                    "controlled light is adapted from a room light to in-car ambient "
                    "lighting before the preference is made durable."
                ),
            },
        },
        {
            "session_id": "hvp11-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:243",
            "context": (
                "Emery and the assistant clarify a moderate fan speed; the exact "
                "source value is adapted to the vehicle cabin fan."
            ),
            "suffix": (
                "Level five is also my normal all-zone cabin fan setting in this car."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.fan_speed",
                "tool_name": "carcontrol_airConditioner_set_fan_speed",
                "argument_name": "speed",
                "value": 5,
                "context_arguments": {"zone": "all"},
                "condition": "for Emery's normal cabin setting",
                "reason": (
                    "The source dialogue explicitly settles on fan speed five; the "
                    "device is adapted to the all-zone cabin fan and made durable."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp11-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s10-t01",
            "query": (
                "Quinn takes the driver's seat while a companion uses the front "
                "passenger seat. Apply Quinn's remembered driver-seat heat and "
                "front-passenger temperature settings."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.seat_heat",
                "driver.normal.passenger_temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_seat_set_heating_level",
                    "arguments": {"seat": "driver", "level": 1},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "passenger", "temperature": 23},
                },
            ],
        },
        {
            "quiz_id": "hvp11-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s12-t01",
            "query": (
                "Quinn is listening to an audiobook during a normal drive. Apply "
                "Quinn's remembered cabin-light, navigation-voice, and airflow choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.audiobook.voice_mode",
                "driver.normal.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 6},
                },
            ],
        },
        {
            "quiz_id": "hvp11-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-mp03-t05",
            "query": (
                "Emery starts a normal drive with an audiobook playing. Apply all "
                "three settings Emery established for that situation."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.audiobook.voice_mode",
                "codriver.normal.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 5},
                },
            ],
        },
        {
            "quiz_id": "hvp11-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s13-t01",
            "query": (
                "Quinn has parked after a normal drive. Apply Quinn's current cabin "
                "fan and vehicle-access rules."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.fan_speed",
                "driver.vehicle_access.doors_locked",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 6},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                },
            ],
        },
        {
            "quiz_id": "hvp11-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s17-t01",
            "query": (
                "Quinn is parked and about to leave, then starts navigation without "
                "naming a place. Apply Quinn's current access and destination defaults."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.vehicle_access.doors_locked",
                "driver.default.destination",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {
                        "destination": "Eastside Public Library on Maple Street"
                    },
                },
            ],
        },
        {
            "quiz_id": "hvp11-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s15-t01",
            "query": (
                "Quinn corrected the normal all-cabin fan setting. Apply the value "
                "active after that correction."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                }
            ],
        },
        {
            "quiz_id": "hvp11-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s16-t01",
            "query": (
                "Quinn is using the normal cabin setting while parked and about to "
                "leave. Apply the current fan choice and the corrected access rule."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": [
                "driver.normal.fan_speed",
                "driver.vehicle_access.doors_locked",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
            ],
        },
        {
            "quiz_id": "hvp11-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s17-t01",
            "query": (
                "Quinn changed the default used when navigation begins without a "
                "named place. Start navigation using the corrected default."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {
                        "destination": "Eastside Public Library on Maple Street"
                    },
                }
            ],
        },
        {
            "quiz_id": "hvp11-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s12-t01",
            "query": (
                "Emery is listening to an audiobook. Use Emery's navigation-voice "
                "preference rather than Quinn's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.audiobook.voice_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                }
            ],
        },
        {
            "quiz_id": "hvp11-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s12-t01",
            "query": (
                "Quinn is listening to an audiobook. Use Quinn's navigation-voice "
                "preference rather than Emery's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.audiobook.voice_mode"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                }
            ],
        },
        {
            "quiz_id": "hvp11-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s12-t01",
            "query": ("Set the cabin's normal ambient lighting for Emery, not Quinn."),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                }
            ],
        },
        {
            "quiz_id": "hvp11-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s12-t01",
            "query": ("Set the cabin's normal ambient lighting for Quinn, not Emery."),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                }
            ],
        },
        {
            "quiz_id": "hvp11-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s15-t01",
            "query": ("Apply Emery's normal cabin fan preference rather than Quinn's."),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 5},
                }
            ],
        },
        {
            "quiz_id": "hvp11-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s15-t01",
            "query": (
                "Apply Quinn's current normal cabin fan preference rather than Emery's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                }
            ],
        },
        {
            "quiz_id": "hvp11-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s14-t01",
            "query": (
                "At this earlier cutoff, apply Quinn's normal cabin fan setting "
                "that was active then."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 6},
                }
            ],
        },
        {
            "quiz_id": "hvp11-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s15-t01",
            "query": (
                "At this later cutoff, apply Quinn's revised normal cabin fan setting."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                }
            ],
        },
        {
            "quiz_id": "hvp11-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp11-s15-t01",
            "query": (
                "Quinn has parked at this cutoff, before revising the later "
                "departure rule. Apply the vehicle-access action stored then."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.vehicle_access.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp11-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp11-s17-t01",
            "query": (
                "Emery starts a normal drive with an audiobook playing. Restore "
                "Emery's remembered lighting, voice-guidance, and cabin-fan choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.audiobook.voice_mode",
                "codriver.normal.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "detailed"},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 5},
                },
            ],
        },
        {
            "quiz_id": "hvp11-final-08",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp11-s17-t01",
            "query": (
                "Quinn, not Emery, starts a normal drive while listening to an "
                "audiobook. Apply Quinn's current lighting, voice, and fan preferences."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.audiobook.voice_mode",
                "driver.normal.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                },
            ],
        },
    ],
}


HVP12_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp12-codriver",
        "short_id": "co_user",
        "speaker_name": "Rowan (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp12-s6",
    "sessions": [
        {
            "session_id": "hvp12-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:221",
            "context": (
                "Rowan and the assistant settle on a 23-degree thermostat setting; "
                "the exact value is adapted to the vehicle's driver zone."
            ),
            "suffix": (
                "Twenty-three degrees is also my normal driver-zone temperature "
                "in this car."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.temperature",
                "tool_name": "carcontrol_airConditioner_set_temperature",
                "argument_name": "temperature",
                "value": 23,
                "context_arguments": {"zone": "driver"},
                "condition": "for Rowan's normal driving",
                "reason": (
                    "The source dialogue explicitly settles on 23 degrees; the "
                    "thermostat target is adapted to the vehicle's driver zone and "
                    "made durable."
                ),
            },
        },
        {
            "session_id": "hvp12-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:523",
            "context": (
                "Rowan clarifies a media-control request into an explicit pause "
                "action, which is adapted to in-car audiobook playback."
            ),
            "suffix": (
                "When I listen to an audiobook in the car, I normally keep playback "
                "paused."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.audiobook.playback",
                "tool_name": "carcontrol_music_switch",
                "argument_name": "switch",
                "value": False,
                "context_arguments": {},
                "condition": "when Rowan listens to an audiobook",
                "reason": (
                    "The source's explicit Pause action is normalized to the "
                    "VehicleMemBench music-off state and made durable for audiobook "
                    "listening."
                ),
            },
        },
        {
            "session_id": "hvp12-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:335",
            "context": (
                "Rowan returns from a distraction to an explicit unlock action; the "
                "entry lock is adapted to all vehicle doors."
            ),
            "suffix": (
                "When I am parked, I normally keep every vehicle door unlocked."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.parked.doors_locked",
                "tool_name": "carcontrol_door_set_locked",
                "argument_name": "locked",
                "value": False,
                "context_arguments": {"door": "all"},
                "condition": "when Rowan is parked",
                "reason": (
                    "The source explicitly ends with an unlock action; the front-door "
                    "target is adapted to all vehicle doors and made into Rowan's "
                    "parked-state preference."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp12-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s10-t01",
            "query": (
                "Sydney begins a normal drive and an audiobook has stopped unexpectedly. "
                "Apply Sydney's remembered driver-climate and playback choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.audiobook.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 19},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
        {
            "quiz_id": "hvp12-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s12-t01",
            "query": (
                "Frost blocks Sydney's view while navigation prompts become too "
                "frequent. Apply Sydney's two remembered responses."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.frequent_prompts.voice_mode",
                "driver.frost.defrost",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                },
            ],
        },
        {
            "quiz_id": "hvp12-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-mp03-t05",
            "query": (
                "Rowan is parked after a normal drive with an audiobook playing. "
                "Apply Rowan's remembered climate, playback, and access choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.temperature",
                "codriver.audiobook.playback",
                "codriver.parked.doors_locked",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 23},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
            ],
        },
        {
            "quiz_id": "hvp12-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s16-t01",
            "query": (
                "Sydney is parked and preparing for a normal cold drive. Apply "
                "Sydney's current steering-wheel, door-access, and temperature settings."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.steering_wheel.heat_enabled",
                "driver.parked.doors_locked",
                "driver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 19},
                },
            ],
        },
        {
            "quiz_id": "hvp12-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s17-t01",
            "query": (
                "Sydney begins a normal drive and starts navigation without naming "
                "a place. Apply the current driver temperature and destination default."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.default.destination",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 19},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Riverside Dog Park on Kennel Lane"},
                },
            ],
        },
        {
            "quiz_id": "hvp12-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s15-t01",
            "query": (
                "Sydney corrected the normal steering-wheel setting. Apply the "
                "action active after the correction."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.steering_wheel.heat_enabled"],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                }
            ],
        },
        {
            "quiz_id": "hvp12-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s16-t01",
            "query": (
                "Sydney is parked after correcting the vehicle-access rule. Apply "
                "the currently stored action."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.parked.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp12-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s17-t01",
            "query": (
                "Sydney changed the default used when navigation starts without a "
                "named place. Navigate using the corrected default."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Riverside Dog Park on Kennel Lane"},
                }
            ],
        },
        {
            "quiz_id": "hvp12-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s10-t01",
            "query": (
                "Apply Rowan's normal driver-zone temperature rather than Sydney's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 23},
                }
            ],
        },
        {
            "quiz_id": "hvp12-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s10-t01",
            "query": (
                "Apply Sydney's normal driver-zone temperature rather than Rowan's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 19},
                }
            ],
        },
        {
            "quiz_id": "hvp12-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s10-t01",
            "query": (
                "Rowan begins a normal drive with an audiobook playing. Apply Rowan's "
                "temperature and playback preferences, not Sydney's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.temperature",
                "codriver.audiobook.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 23},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp12-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s10-t01",
            "query": (
                "Sydney's audiobook has stopped. Use Sydney's playback preference "
                "rather than Rowan's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.audiobook.playback"],
            "gold_calls": [
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                }
            ],
        },
        {
            "quiz_id": "hvp12-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s16-t01",
            "query": (
                "Rowan has parked after a normal drive. Apply Rowan's temperature "
                "and vehicle-access preferences rather than Sydney's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.temperature",
                "codriver.parked.doors_locked",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 23},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
            ],
        },
        {
            "quiz_id": "hvp12-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s16-t01",
            "query": (
                "Sydney is parked. Apply Sydney's vehicle-access preference rather "
                "than Rowan's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.parked.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp12-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s14-t01",
            "query": (
                "Sydney prepares for a normal drive at this earlier cutoff. Apply "
                "the driver temperature and steering-wheel setting active then."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.steering_wheel.heat_enabled",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 19},
                },
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": False},
                },
            ],
        },
        {
            "quiz_id": "hvp12-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s15-t01",
            "query": (
                "At this later cutoff, apply Sydney's revised normal steering-wheel "
                "setting."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.steering_wheel.heat_enabled"],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                }
            ],
        },
        {
            "quiz_id": "hvp12-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s15-t01",
            "query": (
                "Sydney is parked and preparing for a normal drive at this cutoff, "
                "before the later access-rule revision. Apply the normal driver "
                "temperature and stored door action."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.parked.doors_locked",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 19},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
            ],
        },
        {
            "quiz_id": "hvp12-turn-22",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp12-s16-t01",
            "query": (
                "Sydney is parked at this later cutoff. Apply the revised "
                "vehicle-access action."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.parked.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp12-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp12-s17-t01",
            "query": (
                "Rowan, not Sydney, is parked after a normal drive with an audiobook "
                "playing. Apply Rowan's climate, playback, and access preferences."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.temperature",
                "codriver.audiobook.playback",
                "codriver.parked.doors_locked",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 23},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
            ],
        },
    ],
}


HVP13_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp13-codriver",
        "short_id": "co_user",
        "speaker_name": "Parker (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp13-s6",
    "sessions": [
        {
            "session_id": "hvp13-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:270",
            "context": (
                "Parker explicitly selects a continuously running fan at speed 8; "
                "the exact speed is adapted to the vehicle cabin fan."
            ),
            "suffix": (
                "Level eight is also my normal all-zone cabin fan setting in this car."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.fan_speed",
                "tool_name": "carcontrol_airConditioner_set_fan_speed",
                "argument_name": "speed",
                "value": 8,
                "context_arguments": {"zone": "all"},
                "condition": "for Parker's normal cabin setting",
                "reason": (
                    "The source dialogue explicitly settles on fan speed eight; the "
                    "device is adapted to the all-zone cabin fan and made durable."
                ),
            },
        },
        {
            "session_id": "hvp13-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:220",
            "context": (
                "Parker explicitly raises a thermostat target to 24°C; the exact "
                "value is adapted to the vehicle's rear-right zone."
            ),
            "suffix": (
                "Twenty-four degrees is also my normal rear-right temperature in "
                "this car."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.rear_right_temperature",
                "tool_name": "carcontrol_airConditioner_set_temperature",
                "argument_name": "temperature",
                "value": 24,
                "context_arguments": {"zone": "rear_right"},
                "condition": "for Parker's normal rear-right climate",
                "reason": (
                    "The source dialogue explicitly settles on 24 degrees; the "
                    "thermostat target is adapted to the vehicle's rear-right zone "
                    "and made durable."
                ),
            },
        },
        {
            "session_id": "hvp13-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:529",
            "context": (
                "Parker corrects a stop request into an explicit pause action, which "
                "is adapted to in-car audio playback."
            ),
            "suffix": (
                "If my in-car audio stops, normally leave it paused rather than "
                "restarting it."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.interrupted_audio.playback",
                "tool_name": "carcontrol_music_switch",
                "argument_name": "switch",
                "value": False,
                "context_arguments": {},
                "condition": "when Parker's audio playback has stopped",
                "reason": (
                    "The source's explicit Pause action is normalized to the "
                    "VehicleMemBench music-off state and made into a durable rule for "
                    "interrupted audio."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp13-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s8-t01",
            "query": (
                "Cameron starts a normal drive. Apply Cameron's remembered cabin fan "
                "and driver-window settings active at this point."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.fan_speed",
                "driver.normal.driver_window",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 5},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 55},
                },
            ],
        },
        {
            "quiz_id": "hvp13-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s10-t01",
            "query": (
                "Frost blocks Cameron's view while a rear-right passenger needs the "
                "normal climate. Apply Cameron's two remembered settings."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.rear_right_temperature",
                "driver.frost.defrost",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "rear_right", "temperature": 27},
                },
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                },
            ],
        },
        {
            "quiz_id": "hvp13-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-mp03-t03",
            "query": (
                "Parker is using the normal cabin and rear-right climate when the "
                "audio stops. Apply all three of Parker's remembered choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.fan_speed",
                "codriver.normal.rear_right_temperature",
                "codriver.interrupted_audio.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "rear_right", "temperature": 24},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp13-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s12-t01",
            "query": (
                "Cameron is telling a story when the audio playback stops. Apply "
                "Cameron's remembered navigation-voice and playback responses."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.storytelling.voice_mode",
                "driver.interrupted_audio.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
        {
            "quiz_id": "hvp13-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s16-t01",
            "query": (
                "Cameron's hands are cold during a normal drive. Apply Cameron's "
                "current wheel-heat, fan, and driver-window settings."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.cold_hands.wheel_heat",
                "driver.normal.fan_speed",
                "driver.normal.driver_window",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 3},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 90},
                },
            ],
        },
        {
            "quiz_id": "hvp13-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s14-t01",
            "query": (
                "Cameron first corrected the normal cabin fan setting. Apply the "
                "value active immediately after that correction."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 1},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s15-t01",
            "query": (
                "Cameron corrected the normal cabin fan a second time. Apply the "
                "newest stored value."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 3},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s16-t01",
            "query": (
                "Cameron revised the normal driver-window position. Apply the "
                "currently stored setting."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.driver_window"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 90},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s15-t01",
            "query": (
                "Apply Parker's normal cabin fan preference rather than Cameron's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s15-t01",
            "query": (
                "Apply Cameron's current normal cabin fan preference rather than "
                "Parker's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 3},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s12-t01",
            "query": (
                "Use Parker's normal rear-right temperature rather than Cameron's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.rear_right_temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "rear_right", "temperature": 24},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s12-t01",
            "query": (
                "Use Cameron's normal rear-right temperature rather than Parker's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.rear_right_temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "rear_right", "temperature": 27},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s12-t01",
            "query": (
                "Parker's audio playback has stopped. Apply Parker's audio and "
                "normal rear-right climate preferences rather than Cameron's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.interrupted_audio.playback",
                "codriver.normal.rear_right_temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "rear_right", "temperature": 24},
                },
            ],
        },
        {
            "quiz_id": "hvp13-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s12-t01",
            "query": (
                "Cameron's audio playback has stopped. Apply Cameron's response "
                "rather than Parker's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.interrupted_audio.playback"],
            "gold_calls": [
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s13-t01",
            "query": (
                "At this earlier cutoff, apply Cameron's normal cabin fan setting "
                "that was active then."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 5},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s14-t01",
            "query": (
                "After Cameron's first fan revision but before the second, apply the "
                "normal cabin setting stored at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 1},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s15-t01",
            "query": (
                "At this later cutoff, apply Cameron's twice-revised normal cabin "
                "fan setting."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 3},
                }
            ],
        },
        {
            "quiz_id": "hvp13-turn-22",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp13-s16-t01",
            "query": (
                "At the final cutoff, apply Cameron's revised normal driver-window "
                "position."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.driver_window"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 90},
                }
            ],
        },
        {
            "quiz_id": "hvp13-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp13-s16-t01",
            "query": (
                "Parker, not Cameron, is using the normal cabin and rear-right climate "
                "when audio playback stops. Apply Parker's three preferences."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.fan_speed",
                "codriver.normal.rear_right_temperature",
                "codriver.interrupted_audio.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "rear_right", "temperature": 24},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
    ],
}


HVP14_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp14-codriver",
        "short_id": "co_user",
        "speaker_name": "Sawyer (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the wording of each external source record",
    },
    "insert_after_session_id": "hvp14-s6",
    "sessions": [
        {
            "session_id": "hvp14-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:458",
            "context": (
                "A three-turn open-and-confirm exchange is rewritten from a garage "
                "door to the vehicle sunroof while preserving its dialogue acts."
            ),
            "turn_overrides": [
                "Open the sunroof.",
                "Just to confirm, do you want the sunroof fully opened now?",
                "Yes. Open it fully now.",
            ],
            "suffix": (
                "Fully open is also my normal sunroof position for cabin venting."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.sunroof",
                "tool_name": "carcontrol_sunroof_set_open_degree",
                "argument_name": "degree",
                "value": 100,
                "context_arguments": {},
                "condition": "for Sawyer's normal cabin venting",
                "reason": (
                    "The source's Open state and confirmation structure are preserved "
                    "in a disclosed garage-to-sunroof rewrite; fully open is "
                    "deterministically normalized to 100 percent."
                ),
            },
        },
        {
            "session_id": "hvp14-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier6:4119",
            "context": (
                "Sawyer corrects a navigation stop from a generic strip mall to "
                "Suburban Mall."
            ),
            "suffix": (
                "Suburban Mall is also my default destination whenever I start "
                "navigation without naming a place."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.default.destination",
                "tool_name": "carcontrol_navigation_navigate_to",
                "argument_name": "destination",
                "value": "Suburban Mall",
                "context_arguments": {},
                "condition": (
                    "when Sawyer starts navigation without naming a destination"
                ),
                "reason": (
                    "The vehicle source explicitly corrects the route location to "
                    "Suburban Mall; the stop is promoted to Sawyer's durable unnamed-"
                    "navigation default."
                ),
            },
        },
        {
            "session_id": "hvp14-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:454",
            "context": (
                "A three-turn close-and-clarify exchange is rewritten from a garage "
                "door to the vehicle's front trunk while preserving its dialogue acts."
            ),
            "turn_overrides": [
                "Ugh, can you do something about the front trunk? It's still open.",
                (
                    "I'd be happy to help with the cargo area. Do you mean the front "
                    "trunk or the rear liftgate?"
                ),
                "The front one. Just close it already!",
            ],
            "suffix": (
                "When I load cargo in this car, keep the front trunk closed until I "
                "explicitly ask to open it."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.cargo.front_trunk",
                "tool_name": "carcontrol_frontTrunk_switch",
                "argument_name": "switch",
                "value": False,
                "context_arguments": {},
                "condition": "when Sawyer loads cargo",
                "reason": (
                    "The source's explicit Closed state and clarification structure "
                    "are preserved in a disclosed garage-to-front-trunk rewrite before "
                    "the durable cargo rule is added."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp14-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s8-t01",
            "query": (
                "Parker begins normal cabin venting and starts navigation without "
                "naming a place. Apply Parker's remembered roof and destination defaults."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.default.destination",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 30},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Work"},
                },
            ],
        },
        {
            "quiz_id": "hvp14-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s10-t01",
            "query": (
                "Parker is preparing to load cargo while the route directions are "
                "obvious. Apply Parker's remembered front-compartment and guidance "
                "rules."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.cargo.front_trunk",
                "driver.obvious_route.voice_mode",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_frontTrunk_switch",
                    "arguments": {"switch": True},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
            ],
        },
        {
            "quiz_id": "hvp14-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-mp03-t03",
            "query": (
                "Sawyer begins normal cabin venting, starts unnamed navigation, and "
                "prepares to load cargo without explicitly asking to open a "
                "compartment. Apply Sawyer's three remembered choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.sunroof",
                "codriver.default.destination",
                "codriver.cargo.front_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 100},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Suburban Mall"},
                },
                {
                    "name": "carcontrol_frontTrunk_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp14-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s12-t01",
            "query": (
                "Parker begins a normal drive and needs clear rear visibility. Apply "
                "Parker's remembered rear-visibility climate response and rear "
                "cargo-door setting."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.rear_visibility.defrost",
                "driver.normal.rear_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "rear", "mode": "defrost"},
                },
                {
                    "name": "carcontrol_trunk_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp14-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s16-t01",
            "query": (
                "Parker begins a normal drive with cabin venting, then starts "
                "navigation without naming a place. Apply Parker's current roof, "
                "wheel-heat, and destination defaults."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.normal.wheel_heat",
                "driver.default.destination",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 18},
                },
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                },
            ],
        },
        {
            "quiz_id": "hvp14-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s14-t01",
            "query": (
                "Parker corrected the normal cabin-venting roof position. Apply the "
                "value active after that correction."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp14-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s15-t01",
            "query": (
                "Parker changed the default used when navigation starts without a "
                "named place. Navigate using the corrected default."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                }
            ],
        },
        {
            "quiz_id": "hvp14-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s16-t01",
            "query": ("Apply Sawyer's normal sunroof preference rather than Parker's."),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 100},
                }
            ],
        },
        {
            "quiz_id": "hvp14-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s16-t01",
            "query": (
                "Apply Parker's current normal sunroof preference rather than Sawyer's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp14-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s15-t01",
            "query": (
                "Sawyer starts navigation without naming a place. Use Sawyer's "
                "default rather than Parker's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Suburban Mall"},
                }
            ],
        },
        {
            "quiz_id": "hvp14-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s15-t01",
            "query": (
                "Parker starts navigation without naming a place. Use Parker's "
                "current default rather than Sawyer's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                }
            ],
        },
        {
            "quiz_id": "hvp14-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s15-t01",
            "query": (
                "Sawyer is preparing to load cargo without explicitly asking to "
                "open a compartment, and starts navigation without naming a place. "
                "Apply Sawyer's front-compartment and destination rules, not "
                "Parker's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.cargo.front_trunk",
                "codriver.default.destination",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_frontTrunk_switch",
                    "arguments": {"switch": False},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Suburban Mall"},
                },
            ],
        },
        {
            "quiz_id": "hvp14-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s15-t01",
            "query": (
                "Parker is preparing to load cargo. Apply Parker's "
                "front-compartment rule rather than Sawyer's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.cargo.front_trunk"],
            "gold_calls": [
                {
                    "name": "carcontrol_frontTrunk_switch",
                    "arguments": {"switch": True},
                }
            ],
        },
        {
            "quiz_id": "hvp14-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s16-t01",
            "query": (
                "Sawyer, not Parker, begins normal cabin venting, starts unnamed "
                "navigation, and prepares to load cargo without explicitly asking "
                "to open a compartment. Apply Sawyer's three preferences."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.sunroof",
                "codriver.default.destination",
                "codriver.cargo.front_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 100},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Suburban Mall"},
                },
                {
                    "name": "carcontrol_frontTrunk_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp14-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s13-t01",
            "query": (
                "At this earlier cutoff, apply Parker's normal cabin-venting roof "
                "position that was active then."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 30},
                }
            ],
        },
        {
            "quiz_id": "hvp14-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s14-t01",
            "query": (
                "At this later cutoff, apply Parker's revised cabin-venting roof "
                "position."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp14-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s15-t01",
            "query": (
                "Parker begins normal cabin venting at this cutoff, before the later "
                "steering-wheel revision. Apply the roof and wheel settings stored "
                "then."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.normal.wheel_heat",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 18},
                },
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": False},
                },
            ],
        },
        {
            "quiz_id": "hvp14-turn-22",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp14-s16-t01",
            "query": (
                "At the final cutoff, apply Parker's revised normal steering-wheel "
                "setting."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.wheel_heat"],
            "gold_calls": [
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                }
            ],
        },
        {
            "quiz_id": "hvp14-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp14-s16-t01",
            "query": (
                "Parker, not Sawyer, begins normal cabin venting, starts navigation "
                "without naming a place, and prepares to load cargo. Apply Parker's "
                "three current preferences."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.default.destination",
                "driver.cargo.front_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 18},
                },
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                },
                {
                    "name": "carcontrol_frontTrunk_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
    ],
}


HVP15_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp15-codriver",
        "short_id": "co_user",
        "speaker_name": "Morgan (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the dialogue acts and values of each external record",
    },
    "insert_after_session_id": "hvp15-s6",
    "sessions": [
        {
            "session_id": "hvp15-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:248",
            "context": (
                "Morgan discusses connected controls, is distracted by ambient "
                "lighting, and returns to an explicit fan-speed-three request."
            ),
            "turn_overrides": [
                (
                    "I was just thinking about how much I rely on connected controls "
                    "these days. My car's climate, lights, even phone integration are "
                    "all linked. It's kind of wild when you stop and think about it."
                ),
                (
                    "It really is impressive how much vehicle technology has evolved. "
                    "Do you normally let the cabin climate run automatically, or do "
                    "you prefer to adjust it yourself?"
                ),
                (
                    "Honestly, a mix of both. Speaking of which, I keep forgetting to "
                    "adjust the cabin fan, and it's been running a bit too strong. "
                    "Anyway, I got distracted looking at ambient-light colors online."
                ),
                (
                    "Those colors can change the cabin mood. If the fan is too strong, "
                    "we can dial it back; just tell me the speed you want."
                ),
                "Right, let's fix that: keep the cabin fan on, but only at speed 3.",
            ],
            "suffix": "Speed three is also my regular all-cabin fan setting.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.fan_speed",
                "tool_name": "carcontrol_airConditioner_set_fan_speed",
                "argument_name": "speed",
                "value": 3,
                "context_arguments": {"zone": "all"},
                "condition": "for Morgan's normal cabin setting",
                "reason": (
                    "The source's distraction-and-return structure and explicit speed "
                    "3 are preserved in a disclosed home-to-vehicle fan rewrite."
                ),
            },
        },
        {
            "session_id": "hvp15-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:225",
            "context": (
                "Morgan asks to warm the space and explicitly chooses 23 degrees."
            ),
            "suffix": (
                "Use twenty-three degrees for my driver zone during normal driving."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.temperature",
                "tool_name": "carcontrol_airConditioner_set_temperature",
                "argument_name": "temperature",
                "value": 23,
                "context_arguments": {"zone": "driver"},
                "condition": "for Morgan's normal driving",
                "reason": (
                    "The source explicitly selects 23 degrees; the driver-zone scope "
                    "and durable normal-driving condition are added for the pilot."
                ),
            },
        },
        {
            "session_id": "hvp15-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:526",
            "context": (
                "A three-turn Pause confirmation is rewritten as an unexpectedly "
                "stopped in-car audiobook while preserving the requested action."
            ),
            "turn_overrides": [
                "The audiobook stopped unexpectedly. Don't restart it yet.",
                "Understood. Would you like me to leave the audiobook paused?",
                "Yeah, leave it paused.",
            ],
            "suffix": (
                "Whenever my audiobook stops unexpectedly, leave it paused instead "
                "of resuming it."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.audiobook.playback",
                "tool_name": "carcontrol_music_switch",
                "argument_name": "switch",
                "value": False,
                "context_arguments": {},
                "condition": "when Morgan's audiobook stops unexpectedly",
                "reason": (
                    "The source explicitly selects Pause; the disclosed rewrite makes "
                    "the stopped-audiobook condition identical to the competing "
                    "driver preference."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp15-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s8-t01",
            "query": (
                "Rowan begins a normal drive. Apply the cabin-air and fan settings "
                "remembered at this point."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.circulation",
                "driver.normal.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 7},
                },
            ],
        },
        {
            "quiz_id": "hvp15-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s11-t01",
            "query": (
                "Rowan begins a normal drive and the audiobook stops unexpectedly. "
                "Apply Rowan's remembered climate and playback choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.audiobook.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 19},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
        {
            "quiz_id": "hvp15-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-mp03-t03",
            "query": (
                "Morgan begins a normal drive with the usual cabin fan, and the "
                "audiobook stops unexpectedly. Apply Morgan's three choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.fan_speed",
                "codriver.normal.temperature",
                "codriver.audiobook.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 3},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 23},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp15-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s13-t01",
            "query": (
                "Rowan starts a normal drive without gloves on a route segment with "
                "obvious directions. Apply the remembered guidance, wheel, and rear "
                "cargo-door settings."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.obvious_route.voice_mode",
                "driver.no_gloves.wheel_heat",
                "driver.normal.rear_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                },
                {
                    "name": "carcontrol_trunk_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp15-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s16-t01",
            "query": (
                "Rowan begins a normal drive. Apply the current cabin-air, fan, and "
                "driver-temperature settings."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.circulation",
                "driver.normal.fan_speed",
                "driver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "inside"},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 18},
                },
            ],
        },
        {
            "quiz_id": "hvp15-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s15-t05",
            "query": (
                "Rowan revised the regular cabin fan setting. Apply the corrected "
                "speed."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                }
            ],
        },
        {
            "quiz_id": "hvp15-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s16-t01",
            "query": (
                "Rowan revised the regular driver-zone temperature. Apply the "
                "corrected setting."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp15-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s16-t01",
            "query": "Apply Morgan's normal cabin-fan preference rather than Rowan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 3},
                }
            ],
        },
        {
            "quiz_id": "hvp15-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s16-t01",
            "query": "Apply Rowan's normal cabin-fan preference rather than Morgan's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                }
            ],
        },
        {
            "quiz_id": "hvp15-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s16-t01",
            "query": (
                "Morgan begins a normal drive. Apply Morgan's driver-temperature "
                "preference rather than Rowan's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 23},
                }
            ],
        },
        {
            "quiz_id": "hvp15-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s16-t01",
            "query": (
                "Rowan begins a normal drive. Apply Rowan's driver-temperature "
                "preference rather than Morgan's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp15-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s16-t01",
            "query": (
                "Morgan begins a normal drive and the audiobook stops unexpectedly. "
                "Apply Morgan's climate and playback preferences rather than Rowan's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.temperature",
                "codriver.audiobook.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 23},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp15-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s16-t01",
            "query": (
                "Rowan's audiobook stops unexpectedly. Apply Rowan's playback "
                "preference rather than Morgan's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.audiobook.playback"],
            "gold_calls": [
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                }
            ],
        },
        {
            "quiz_id": "hvp15-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s16-t01",
            "query": (
                "Morgan, not Rowan, begins a normal drive with the usual cabin fan, "
                "and the audiobook stops unexpectedly. Apply Morgan's three "
                "preferences."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.fan_speed",
                "codriver.normal.temperature",
                "codriver.audiobook.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 3},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 23},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp15-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s13-t01",
            "query": (
                "At this earlier cutoff, apply Rowan's normal cabin-air setting "
                "that was active then."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                }
            ],
        },
        {
            "quiz_id": "hvp15-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s14-t01",
            "query": (
                "After Rowan's cabin-air revision but before the fan revision, apply "
                "both normal settings stored at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.normal.circulation",
                "driver.normal.fan_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "inside"},
                },
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 7},
                },
            ],
        },
        {
            "quiz_id": "hvp15-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s14-t01",
            "query": (
                "Before Rowan revised the regular cabin fan, apply the fan setting "
                "active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 7},
                }
            ],
        },
        {
            "quiz_id": "hvp15-turn-22",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp15-s15-t05",
            "query": (
                "After Rowan revised the regular cabin fan, apply the setting active "
                "at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.fan_speed"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                }
            ],
        },
        {
            "quiz_id": "hvp15-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp15-s16-t01",
            "query": (
                "Rowan, not Morgan, begins a normal drive with the usual cabin fan, "
                "and the audiobook stops unexpectedly. Apply Rowan's three current "
                "preferences."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.fan_speed",
                "driver.normal.temperature",
                "driver.audiobook.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_fan_speed",
                    "arguments": {"zone": "all", "speed": 8},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 18},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
    ],
}


HVP16_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp16-codriver",
        "short_id": "co_user",
        "speaker_name": "Avery (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the dialogue acts and explicit values of each external record",
    },
    "insert_after_session_id": "hvp16-s6",
    "sessions": [
        {
            "session_id": "hvp16-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:459",
            "context": (
                "Avery distinguishes the main sunroof from other openings and "
                "clarifies that it should be fully, rather than partly, open."
            ),
            "turn_overrides": [
                (
                    "Um, I really don't want the big opening above us to stop "
                    "halfway like last time, and I'd rather not move any of the "
                    "windows. I'm kind of nervous about it."
                ),
                (
                    "No worries, I'll only work with the main sunroof. When you "
                    "say you don't want it halfway, do you want it fully open or "
                    "fully closed?"
                ),
                (
                    "Oh, right, the main sunroof. I want it all the way open, not "
                    "half-open or anything weird."
                ),
            ],
            "suffix": "Fully open is also my normal sunroof position.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.sunroof",
                "tool_name": "carcontrol_sunroof_set_open_degree",
                "argument_name": "degree",
                "value": 100,
                "context_arguments": {},
                "condition": "for Avery's normal sunroof setting",
                "reason": (
                    "The source explicitly resolves a partly open alternative to "
                    "fully Open; the disclosed object rewrite preserves that "
                    "polarity and full extent as 100 percent."
                ),
            },
        },
        {
            "session_id": "hvp16-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:537",
            "context": (
                "Avery checks that stopped in-car audio will remain paused during "
                "an important call."
            ),
            "turn_overrides": [
                (
                    "I don't want any music playing in the car right now. Are you "
                    "sure it isn't going to start automatically?"
                ),
                (
                    "I understand. The audio will not start unless you ask for it. "
                    "Is there a particular situation you're concerned about?"
                ),
                (
                    "I'm still not convinced. Can you verify that the car audio is "
                    "paused? I need to be certain it won't start during my "
                    "important call."
                ),
            ],
            "suffix": (
                "Whenever my audio playback has stopped, leave it paused instead "
                "of resuming it."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.interrupted_audio.playback",
                "tool_name": "carcontrol_music_switch",
                "argument_name": "switch",
                "value": False,
                "context_arguments": {},
                "condition": "when Avery's audio playback has stopped",
                "reason": (
                    "The source explicitly requests Pause; the vehicle rewrite "
                    "makes its stopped-playback condition identical to Reese's "
                    "competing resume rule."
                ),
            },
        },
        {
            "session_id": "hvp16-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:240",
            "context": (
                "Avery asks for a balanced driver-zone cabin temperature and "
                "explicitly settles on 20 degrees."
            ),
            "turn_overrides": [
                (
                    "I don't want the cabin to feel stuffy or overly hot, and I "
                    "definitely don't want the temperature cranked up to something "
                    "extreme."
                ),
                (
                    "Understood. Are you aiming for a cooler cabin or a mild, "
                    "balanced temperature? And should I adjust the driver zone?"
                ),
                (
                    "I'd like a comfortably cool but still cozy driver zone, "
                    "nothing too cold either. Let's keep it at 20 degrees."
                ),
            ],
            "suffix": ("Twenty degrees is also my normal driver-zone temperature."),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.temperature",
                "tool_name": "carcontrol_airConditioner_set_temperature",
                "argument_name": "temperature",
                "value": 20,
                "context_arguments": {"zone": "driver"},
                "condition": "for Avery's normal driving",
                "reason": (
                    "The source explicitly selects 20 degrees; the driver-zone "
                    "scope and durable normal-driving condition are added for the "
                    "pilot."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp16-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s8-t01",
            "query": (
                "Reese begins a normal drive as the panes start clouding. Restore "
                "the remembered overhead opening and cabin-air source."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.fog_risk.circulation",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 10},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s10-t01",
            "query": (
                "Reese is midway through a story when the audio falls silent. "
                "Apply the remembered choices for those two interruptions."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.storytelling.voice_mode",
                "driver.interrupted_audio.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-mp03-t03",
            "query": (
                "Avery takes the driver's seat for a routine trip after the audio "
                "has stopped. Restore all three of Avery's usual choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.sunroof",
                "codriver.interrupted_audio.playback",
                "codriver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 100},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 20},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s13-t01",
            "query": (
                "Reese starts a routine drive with cold hands. Restore the setting "
                "for the pane beside the driver and the one for the wheel."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.driver_window",
                "driver.cold_hands.wheel_heat",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 50},
                },
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": (
                "During Reese's normal drive, the audio goes silent while Reese is "
                "telling a story. Restore the current overhead, guidance, and "
                "playback choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.storytelling.voice_mode",
                "driver.interrupted_audio.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 0},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s14-t01",
            "query": (
                "Reese revised the normal sunroof setting for the first time. "
                "Apply the corrected position active at this cutoff."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 15},
                }
            ],
        },
        {
            "quiz_id": "hvp16-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": (
                "After Reese's later sunroof revisions, apply the corrected current "
                "position together with Reese's regular driver-window setting."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.normal.driver_window",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 0},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 50},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": "Apply Avery's normal sunroof preference rather than Reese's.",
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 100},
                }
            ],
        },
        {
            "quiz_id": "hvp16-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": (
                "Reese begins a normal drive. Use Reese's overhead preference "
                "rather than Avery's, and restore Reese's usual driver window."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.normal.driver_window",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 0},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 50},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": (
                "Avery begins a normal drive after the audio has stopped. Keep the "
                "audio in Avery's preferred state rather than Reese's and restore "
                "Avery's temperature."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.interrupted_audio.playback",
                "codriver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 20},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": (
                "Reese's audio has stopped. Apply Reese's playback preference "
                "rather than Avery's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.interrupted_audio.playback"],
            "gold_calls": [
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                }
            ],
        },
        {
            "quiz_id": "hvp16-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": (
                "Avery, not Reese, takes the driver's seat for a normal trip after "
                "the audio has stopped. Apply Avery's three stored choices."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.sunroof",
                "codriver.interrupted_audio.playback",
                "codriver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 100},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "driver", "temperature": 20},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": (
                "Reese, not Avery, begins a normal drive after the audio has "
                "stopped. Apply Reese's current overhead, playback, and "
                "driver-window choices."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.interrupted_audio.playback",
                "driver.normal.driver_window",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 0},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 50},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": (
                "The current user is Avery. During a normal drive with stopped "
                "audio, choose Avery's overhead and playback rules instead of "
                "Reese's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.sunroof",
                "codriver.interrupted_audio.playback",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 100},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp16-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s8-t01",
            "query": (
                "At this early cutoff, apply the normal sunroof position Reese had "
                "stored before any revision."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 10},
                }
            ],
        },
        {
            "quiz_id": "hvp16-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s14-t01",
            "query": (
                "After Reese's first sunroof revision, apply the normal position "
                "active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 15},
                }
            ],
        },
        {
            "quiz_id": "hvp16-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s15-t01",
            "query": (
                "After Reese's next sunroof revision, apply the normal position "
                "active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.sunroof"],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 30},
                }
            ],
        },
        {
            "quiz_id": "hvp16-turn-22",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": (
                "After Reese's final sunroof revision, apply the current normal "
                "position and the regular driver-window setting."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.normal.driver_window",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 0},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 50},
                },
            ],
        },
        {
            "quiz_id": "hvp16-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp16-s16-t01",
            "query": (
                "Reese, rather than Avery, begins a normal drive with cold hands "
                "after the audio has stopped. Apply Reese's current sunroof, "
                "playback, driver-window, and wheel settings."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.sunroof",
                "driver.interrupted_audio.playback",
                "driver.normal.driver_window",
                "driver.cold_hands.wheel_heat",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 0},
                },
                {
                    "name": "carcontrol_music_switch",
                    "arguments": {"switch": True},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "driver", "degree": 50},
                },
                {
                    "name": "carcontrol_steeringWheel_set_heating_enabled",
                    "arguments": {"enabled": True},
                },
            ],
        },
    ],
}


HVP17_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp17-codriver",
        "short_id": "co_user",
        "speaker_name": "Jordan (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the dialogue acts and explicit values of each external record",
    },
    "insert_after_session_id": "hvp17-s6",
    "sessions": [
        {
            "session_id": "hvp17-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:68",
            "context": (
                "Jordan rejects red lighting and settles on a soft blue color for "
                "the vehicle dash."
            ),
            "suffix": (
                "Blue is also my normal ambient-light color; keep the requested "
                "soft brightness only for this drive."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.ambient_color",
                "tool_name": "carcontrol_light_set_ambient_color",
                "argument_name": "color",
                "value": "blue",
                "context_arguments": {},
                "condition": "for Jordan's normal ambient lighting",
                "reason": (
                    "The source explicitly requests soft blue for the dash; its "
                    "HSB color is normalized to blue while brightness remains "
                    "turn-local."
                ),
            },
        },
        {
            "session_id": "hvp17-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:231",
            "context": (
                "Jordan says the cabin is too hot and explicitly chooses 19 "
                "degrees for all zones."
            ),
            "turn_overrides": [
                "It's too hot in this cabin.",
                (
                    "I can lower the cabin temperature. What temperature would you "
                    "prefer, and should it apply to every zone?"
                ),
                "Yeah, cooler, like 19 degrees in every zone.",
            ],
            "suffix": (
                "Nineteen degrees is also my normal all-zone cabin temperature."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.temperature",
                "tool_name": "carcontrol_airConditioner_set_temperature",
                "argument_name": "temperature",
                "value": 19,
                "context_arguments": {"zone": "all"},
                "condition": "for Jordan's normal cabin climate",
                "reason": (
                    "The source explicitly selects 19 degrees; the all-zone scope "
                    "and durable normal-climate condition are added for the pilot."
                ),
            },
        },
        {
            "session_id": "hvp17-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:480",
            "context": (
                "Jordan revises a vague half-open request and explicitly chooses a "
                "fully open rear-right window for a stuffy rear cabin."
            ),
            "turn_overrides": [
                (
                    "The kids say the back feels stuffy. I was thinking the "
                    "rear-right window could stay halfway open."
                ),
                (
                    "I can set that window precisely. Would you like a percentage, "
                    "or should I open or close it fully?"
                ),
                (
                    "Let's keep it simple; fully open the rear-right window when "
                    "they say it is stuffy."
                ),
            ],
            "suffix": "That is my standing preference for that situation.",
            "update": {
                "op": "add",
                "slot_id": "codriver.kids_stuffy.rear_right_window",
                "tool_name": "carcontrol_window_set_open_degree",
                "argument_name": "degree",
                "value": 100,
                "context_arguments": {"window": "rear_right"},
                "condition": "when Jordan's children find the rear cabin stuffy",
                "reason": (
                    "The source resolves a halfway request to fully Open; the "
                    "disclosed aperture rewrite preserves the full-open target and "
                    "adds the matched child-occupant condition."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp17-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s8-t01",
            "query": (
                "Blair begins a routine drive. Restore the remembered cabin "
                "warmth and the color used for the interior glow."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.normal.ambient_color",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 18},
                },
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "pink"},
                },
            ],
        },
        {
            "quiz_id": "hvp17-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-mp03-t03",
            "query": (
                "Jordan begins a normal trip while the children complain that the "
                "rear cabin is stuffy. Restore Jordan's three stored choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.normal.temperature",
                "codriver.kids_stuffy.rear_right_window",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 19},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "rear_right", "degree": 100},
                },
            ],
        },
        {
            "quiz_id": "hvp17-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s10-t01",
            "query": (
                "Blair asks to begin the usual trip without naming a place. Use "
                "the destination Blair meant."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": ["driver.default.destination"],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                }
            ],
        },
        {
            "quiz_id": "hvp17-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s12-t01",
            "query": (
                "The glass is badly clouded and stale cabin air is giving Blair a "
                "headache. Apply the two remembered climate responses."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.heavy_fog.defrost",
                "driver.stale_air.circulation",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "all", "mode": "defrost"},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
            ],
        },
        {
            "quiz_id": "hvp17-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s16-t01",
            "query": (
                "Blair begins a routine drive while the children say the back feels "
                "stuffy. Restore Blair's current cabin warmth, interior glow, rear "
                "opening, and cargo-door state."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.normal.ambient_color",
                "driver.kids_stuffy.rear_right_window",
                "driver.normal.rear_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 27},
                },
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "rear_right", "degree": 60},
                },
                {
                    "name": "carcontrol_trunk_switch",
                    "arguments": {"switch": False},
                },
            ],
        },
        {
            "quiz_id": "hvp17-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s15-t01",
            "query": (
                "Blair revised the normal ambient color twice. Apply the corrected "
                "color active at this cutoff together with the current cabin "
                "temperature."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 18},
                },
            ],
        },
        {
            "quiz_id": "hvp17-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s16-t01",
            "query": (
                "Blair later replaced the original normal cabin temperature. "
                "Apply the corrected current setting."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 27},
                }
            ],
        },
        {
            "quiz_id": "hvp17-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s16-t01",
            "query": (
                "Apply Jordan's normal ambient-light preference rather than Blair's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                }
            ],
        },
        {
            "quiz_id": "hvp17-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s16-t01",
            "query": (
                "Blair, not Jordan, begins a normal drive. Apply Blair's current "
                "ambient color and cabin temperature."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 27},
                },
            ],
        },
        {
            "quiz_id": "hvp17-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s16-t01",
            "query": (
                "Jordan begins a normal drive. Apply Jordan's cabin temperature "
                "rather than Blair's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 19},
                }
            ],
        },
        {
            "quiz_id": "hvp17-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s16-t01",
            "query": (
                "Blair begins a normal drive. Apply Blair's cabin temperature "
                "rather than Jordan's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 27},
                }
            ],
        },
        {
            "quiz_id": "hvp17-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s16-t01",
            "query": (
                "Jordan's children find the rear cabin stuffy. Use Jordan's "
                "rear-right opening preference rather than Blair's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.kids_stuffy.rear_right_window"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "rear_right", "degree": 100},
                }
            ],
        },
        {
            "quiz_id": "hvp17-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s16-t01",
            "query": (
                "Blair's children find the rear cabin stuffy. Use Blair's "
                "rear-right opening preference rather than Jordan's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.kids_stuffy.rear_right_window"],
            "gold_calls": [
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "rear_right", "degree": 60},
                }
            ],
        },
        {
            "quiz_id": "hvp17-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s16-t01",
            "query": (
                "Jordan, not Blair, begins a normal drive while the children find "
                "the rear cabin stuffy. Apply Jordan's three preferences."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.normal.temperature",
                "codriver.kids_stuffy.rear_right_window",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 19},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "rear_right", "degree": 100},
                },
            ],
        },
        {
            "quiz_id": "hvp17-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s7-t01",
            "query": (
                "Before Blair changed the normal ambient lighting, apply the color "
                "active at this early cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "pink"},
                }
            ],
        },
        {
            "quiz_id": "hvp17-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s8-t01",
            "query": (
                "Before Blair revised the normal cabin temperature, apply the "
                "setting active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp17-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s14-t01",
            "query": (
                "After Blair's first ambient-color revision, apply the normal color "
                "active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                }
            ],
        },
        {
            "quiz_id": "hvp17-turn-22",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp17-s15-t01",
            "query": (
                "After Blair's next ambient-color revision, apply the normal color "
                "and cabin temperature active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 18},
                },
            ],
        },
        {
            "quiz_id": "hvp17-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp17-s16-t01",
            "query": (
                "Blair, rather than Jordan, begins a normal drive while the children "
                "find the rear cabin stuffy. Apply Blair's current ambient color, "
                "cabin temperature, and rear-right window preference."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.temperature",
                "driver.kids_stuffy.rear_right_window",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 27},
                },
                {
                    "name": "carcontrol_window_set_open_degree",
                    "arguments": {"window": "rear_right", "degree": 60},
                },
            ],
        },
    ],
}


HVP18_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp18-codriver",
        "short_id": "co_user",
        "speaker_name": "Riley (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the dialogue acts and explicit values of each external record",
    },
    "insert_after_session_id": "hvp18-s6",
    "sessions": [
        {
            "session_id": "hvp18-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:70",
            "context": (
                "Riley says the interior lighting is too bright and explicitly "
                "chooses blue."
            ),
            "turn_overrides": [
                "Mmm, the cabin lights are too bright. Make the ambient color blue.",
                (
                    "I can change the cabin ambient color to blue. Would you like "
                    "me to set it now?"
                ),
                "Yeah, whatever, just make the ambient lighting blue.",
            ],
            "suffix": (
                "Blue is also my normal ambient-light color; brightness is only "
                "for this drive."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.ambient_color",
                "tool_name": "carcontrol_light_set_ambient_color",
                "argument_name": "color",
                "value": "blue",
                "context_arguments": {},
                "condition": "for Riley's normal ambient lighting",
                "reason": (
                    "The source explicitly requests blue; HSB 240 is normalized "
                    "to blue while brightness remains turn-local."
                ),
            },
        },
        {
            "session_id": "hvp18-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:227",
            "context": (
                "Riley says the cabin is too hot and explicitly chooses 20 degrees "
                "for all zones."
            ),
            "turn_overrides": [
                "I don't want the cabin to be too hot, okay?",
                "What temperature would you prefer? It is currently 21 degrees.",
                (
                    "Around 20 degrees would be nice, but not just one seat. Use "
                    "every cabin zone, please."
                ),
            ],
            "suffix": ("Twenty degrees is also my normal all-zone cabin temperature."),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.temperature",
                "tool_name": "carcontrol_airConditioner_set_temperature",
                "argument_name": "temperature",
                "value": 20,
                "context_arguments": {"zone": "all"},
                "condition": "for Riley's normal cabin climate",
                "reason": (
                    "The source explicitly selects 20 degrees; the all-zone scope "
                    "and durable normal-climate condition are added for the pilot."
                ),
            },
        },
        {
            "session_id": "hvp18-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:560",
            "context": (
                "Riley lowers in-car music after it masks traffic, briefly checks "
                "the map, and returns to an explicit volume of 20."
            ),
            "suffix": "Twenty is also my normal in-car music volume.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.music_volume",
                "tool_name": "carcontrol_music_set_volume",
                "argument_name": "volume",
                "value": 20,
                "context_arguments": {},
                "condition": "for Riley's normal music playback",
                "reason": (
                    "The source is already situated in traffic and explicitly "
                    "returns from a map interruption to volume 20; only persistence "
                    "is appended."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp18-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s8-t01",
            "query": (
                "Harper starts a routine drive. Restore the remembered interior "
                "glow and the cooling level for the back-right seat."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "pink"},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
        {
            "quiz_id": "hvp18-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-mp03-t05",
            "query": (
                "Riley begins a normal drive and starts the usual music. Restore "
                "Riley's three stored settings."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.normal.temperature",
                "codriver.normal.music_volume",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 20},
                },
                {
                    "name": "carcontrol_music_set_volume",
                    "arguments": {"volume": 20},
                },
            ],
        },
        {
            "quiz_id": "hvp18-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s11-t01",
            "query": (
                "The panes beside Harper are clouded, and the compartment up front "
                "seems stuck shut. Apply Harper's two remembered responses."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.side_window_fog.circulation",
                "driver.stuck.front_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
                {
                    "name": "carcontrol_frontTrunk_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
        {
            "quiz_id": "hvp18-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s13-t01",
            "query": (
                "Cabin noise is distracting Harper, who also needs to clear the "
                "glass behind the passengers. Apply both remembered actions."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.noisy_cabin.sunroof",
                "driver.rear_visibility.defrost",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_sunroof_set_open_degree",
                    "arguments": {"degree": 0},
                },
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "rear", "mode": "defrost"},
                },
            ],
        },
        {
            "quiz_id": "hvp18-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s16-t01",
            "query": (
                "Harper begins a routine drive. Restore the current cabin warmth, "
                "interior glow, and cooling for the seat behind the passenger."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.normal.ambient_color",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 28},
                },
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
        {
            "quiz_id": "hvp18-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s14-t01",
            "query": (
                "Harper replaced the original normal ambient color. Apply the "
                "corrected color active at this cutoff."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                }
            ],
        },
        {
            "quiz_id": "hvp18-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s16-t01",
            "query": (
                "After Harper's two temperature revisions, apply the corrected "
                "current cabin setting."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 28},
                }
            ],
        },
        {
            "quiz_id": "hvp18-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s16-t01",
            "query": (
                "Apply Riley's normal ambient-light preference rather than Harper's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                }
            ],
        },
        {
            "quiz_id": "hvp18-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s16-t01",
            "query": (
                "Harper begins a normal drive. Apply Harper's current ambient color "
                "rather than Riley's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                }
            ],
        },
        {
            "quiz_id": "hvp18-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s16-t01",
            "query": (
                "Riley begins a normal drive. Apply Riley's cabin temperature "
                "rather than Harper's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 20},
                }
            ],
        },
        {
            "quiz_id": "hvp18-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s16-t01",
            "query": (
                "Harper begins a normal drive. Apply Harper's cabin temperature "
                "rather than Riley's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 28},
                }
            ],
        },
        {
            "quiz_id": "hvp18-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s16-t01",
            "query": (
                "Riley, not Harper, begins a normal drive. Apply Riley's ambient "
                "color and cabin temperature."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 20},
                },
            ],
        },
        {
            "quiz_id": "hvp18-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s16-t01",
            "query": (
                "Harper, not Riley, begins a normal drive. Apply Harper's ambient "
                "color and cabin temperature."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 28},
                },
            ],
        },
        {
            "quiz_id": "hvp18-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s16-t01",
            "query": (
                "The current user is Riley, beginning a normal drive with the usual "
                "music. Choose Riley's ambient color and temperature rather than "
                "Harper's, and also restore Riley's music volume."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.normal.temperature",
                "codriver.normal.music_volume",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 20},
                },
                {
                    "name": "carcontrol_music_set_volume",
                    "arguments": {"volume": 20},
                },
            ],
        },
        {
            "quiz_id": "hvp18-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s7-t01",
            "query": (
                "Before Harper changed the normal ambient lighting, apply the color "
                "active at this early cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "pink"},
                }
            ],
        },
        {
            "quiz_id": "hvp18-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s9-t01",
            "query": (
                "Before Harper revised the normal cabin temperature, apply the "
                "setting active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 24},
                }
            ],
        },
        {
            "quiz_id": "hvp18-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp18-s15-t01",
            "query": (
                "After Harper's first temperature revision but before the final "
                "one, apply the setting active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 21},
                }
            ],
        },
        {
            "quiz_id": "hvp18-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp18-s16-t01",
            "query": (
                "Harper, rather than Riley, begins a normal drive. Apply Harper's "
                "current ambient color, cabin temperature, and rear-right seat "
                "cooling preference."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.temperature",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 28},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
    ],
}


HVP19_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp19-codriver",
        "short_id": "co_user",
        "speaker_name": "Taylor (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the dialogue acts and explicit values of each external record",
    },
    "insert_after_session_id": "hvp19-s6",
    "sessions": [
        {
            "session_id": "hvp19-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:67",
            "context": (
                "Taylor rejects red ambient lighting and explicitly chooses blue "
                "at a restrained brightness."
            ),
            "turn_overrides": [
                "I don't want the cabin ambient light to be red.",
                "What color would you like the cabin ambient light to be?",
                "I want it blue, but not too bright.",
            ],
            "suffix": (
                "Blue is also my normal ambient-light color; the brightness request "
                "is only for this drive."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.ambient_color",
                "tool_name": "carcontrol_light_set_ambient_color",
                "argument_name": "color",
                "value": "blue",
                "context_arguments": {},
                "condition": "for Taylor's normal ambient lighting",
                "reason": (
                    "The source explicitly requests blue; HSB hue 240 is normalized "
                    "to blue while brightness remains turn-local."
                ),
            },
        },
        {
            "session_id": "hvp19-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:345",
            "context": (
                "Taylor is leaving a parked vehicle and confirms that all vehicle "
                "doors should be locked."
            ),
            "turn_overrides": [
                "I'm leaving the parked car and want to make sure it is secure.",
                (
                    "I can lock all the vehicle doors before you walk away. Is that "
                    "what you would like?"
                ),
                "Yes, please lock all of them.",
            ],
            "suffix": (
                "Keeping all doors locked is my normal preference when I am parked."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.parked.doors_locked",
                "tool_name": "carcontrol_door_set_locked",
                "argument_name": "locked",
                "value": True,
                "context_arguments": {"door": "all"},
                "condition": "when Taylor is parked",
                "reason": (
                    "The source explicitly locks an entrance when leaving; the "
                    "disclosed vehicle rewrite preserves the security intent and "
                    "adds all-door parked scope."
                ),
            },
        },
        {
            "session_id": "hvp19-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:547",
            "context": (
                "Taylor says the music should not be too loud and explicitly "
                "selects volume 50."
            ),
            "turn_overrides": [
                "I don't want the car music to be too loud.",
                "What volume level would you prefer?",
                "Set the in-car music volume to 50.",
            ],
            "suffix": "Fifty is also my normal in-car music volume.",
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.music_volume",
                "tool_name": "carcontrol_music_set_volume",
                "argument_name": "volume",
                "value": 50,
                "context_arguments": {},
                "condition": "for Taylor's normal music playback",
                "reason": (
                    "The source explicitly selects volume 50; the speaker scope is "
                    "rewritten as in-car music and made persistent."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp19-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s8-t01",
            "query": (
                "Emerson begins a routine drive. Restore the interior glow and the "
                "cooling level for the seat behind the passenger."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-mp03-t03",
            "query": (
                "Taylor parks after listening to music during a normal drive. "
                "Restore Taylor's three stored settings."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.parked.doors_locked",
                "codriver.normal.music_volume",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                },
                {
                    "name": "carcontrol_music_set_volume",
                    "arguments": {"volume": 50},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s11-t01",
            "query": (
                "Mist is hiding the glass ahead of Emerson, who also needs the coat "
                "stored in the compartment up front. Apply both remembered actions."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.front_condensation.defrost",
                "driver.coat.front_trunk",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "front", "mode": "defrost"},
                },
                {
                    "name": "carcontrol_frontTrunk_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s13-t01",
            "query": (
                "Emerson has parked and asks for something to fill the silence. "
                "Apply the remembered access and audio choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.parked.doors_locked",
                "driver.background_noise.radio",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_radio_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s16-t01",
            "query": (
                "Emerson begins a routine drive. Restore the current interior glow, "
                "cabin-air source, and cooling for the rear-right seat."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.circulation",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s15-t01",
            "query": (
                "Emerson changed the normal ambient color and then changed it "
                "again. Apply the corrected color active at this cutoff together "
                "with the regular rear-right seat cooling."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s16-t01",
            "query": (
                "Emerson later replaced the original normal cabin-air setting. "
                "Apply the corrected current source."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.circulation"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                }
            ],
        },
        {
            "quiz_id": "hvp19-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s16-t01",
            "query": (
                "Apply Taylor's normal ambient-light preference rather than Emerson's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                }
            ],
        },
        {
            "quiz_id": "hvp19-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s16-t01",
            "query": (
                "Emerson, not Taylor, begins a normal drive. Apply Emerson's current "
                "ambient color and cabin-air source."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.circulation",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s16-t01",
            "query": (
                "Taylor is parked. Apply Taylor's door-access preference rather "
                "than Emerson's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.parked.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp19-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s16-t01",
            "query": (
                "Emerson is parked and wants background noise. Apply Emerson's door "
                "preference rather than Taylor's and restore Emerson's audio choice."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.parked.doors_locked",
                "driver.background_noise.radio",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_radio_switch",
                    "arguments": {"switch": True},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s16-t01",
            "query": (
                "Taylor, not Emerson, parks after a normal drive with music. Apply "
                "Taylor's three stored preferences."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.parked.doors_locked",
                "codriver.normal.music_volume",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                },
                {
                    "name": "carcontrol_music_set_volume",
                    "arguments": {"volume": 50},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s16-t01",
            "query": (
                "Emerson, not Taylor, parks after a normal drive. Apply Emerson's "
                "ambient color, door-access preference, and rear-right seat cooling."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.parked.doors_locked",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s16-t01",
            "query": (
                "The current user is Taylor, who has parked after a normal drive "
                "with music playing. Choose Taylor's ambient and door preferences "
                "rather than Emerson's, then restore Taylor's usual volume."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.parked.doors_locked",
                "codriver.normal.music_volume",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "blue"},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                },
                {
                    "name": "carcontrol_music_set_volume",
                    "arguments": {"volume": 50},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s8-t01",
            "query": (
                "Before Emerson's first ambient-color revision, apply the normal "
                "color and rear-right seat cooling active at this early cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
        {
            "quiz_id": "hvp19-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s14-t01",
            "query": (
                "After Emerson's first ambient-color revision, apply the normal "
                "color active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "red"},
                }
            ],
        },
        {
            "quiz_id": "hvp19-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp19-s15-t01",
            "query": (
                "After Emerson changed the normal ambient color back, apply the "
                "current color and rear-right seat cooling at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
        {
            "quiz_id": "hvp19-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp19-s16-t01",
            "query": (
                "Emerson, rather than Taylor, parks after a normal drive. Apply "
                "Emerson's current ambient color, cabin-air source, door-access "
                "preference, and rear-right seat cooling."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.circulation",
                "driver.parked.doors_locked",
                "driver.normal.rear_right_seat_ventilation_speed",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_seat_set_ventilation_speed",
                    "arguments": {"seat": "rear_right", "speed": 2},
                },
            ],
        },
    ],
}


HVP20_EXTENSION: dict[str, Any] = {
    "secondary_user": {
        "speaker_id": "external-hvp20-codriver",
        "short_id": "co_user",
        "speaker_name": "Quinn (normalized user role)",
        "role": "co-driver; external user identities normalized for the pilot",
        "style": "retains the dialogue acts and explicit values of each external record",
    },
    "insert_after_session_id": "hvp20-s6",
    "sessions": [
        {
            "session_id": "hvp20-mp01",
            "timestamp": "2026-04-20T08:00:00",
            "record_id": "tier7:73",
            "context": (
                "Quinn rejects bright blue lighting and settles on a soft yellow "
                "ambient color."
            ),
            "turn_overrides": [
                (
                    "Um, I don't want the cabin ambient light too bright or too "
                    "blue. Could you make it something warmer?"
                ),
                (
                    "You would prefer a warmer cabin light that is not too bright. "
                    "Would orange or amber at moderate brightness work?"
                ),
                (
                    "Yes, but not too orange. Make the ambient color more like a "
                    "soft yellow, and not too dim either."
                ),
            ],
            "suffix": (
                "Yellow is also my normal ambient-light color; brightness is only "
                "for this drive."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.ambient_color",
                "tool_name": "carcontrol_light_set_ambient_color",
                "argument_name": "color",
                "value": "yellow",
                "context_arguments": {},
                "condition": "for Quinn's normal ambient lighting",
                "reason": (
                    "The source explicitly selects soft yellow; HSB hue 60 is "
                    "normalized to yellow while brightness remains turn-local."
                ),
            },
        },
        {
            "session_id": "hvp20-mp02",
            "timestamp": "2026-04-25T08:00:00",
            "record_id": "tier7:229",
            "context": (
                "Quinn says the cabin feels very cold and explicitly chooses 25 "
                "degrees for all zones."
            ),
            "turn_overrides": [
                "I'm really cold in this cabin. Can you make it warmer?",
                (
                    "I understand. Would you like me to raise the temperature in "
                    "every cabin zone?"
                ),
                "Yes, please make every zone 25 degrees. I'm shivering.",
            ],
            "suffix": (
                "Twenty-five degrees is also my normal all-zone cabin temperature."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.normal.temperature",
                "tool_name": "carcontrol_airConditioner_set_temperature",
                "argument_name": "temperature",
                "value": 25,
                "context_arguments": {"zone": "all"},
                "condition": "for Quinn's normal cabin climate",
                "reason": (
                    "The source explicitly selects 25 degrees; the all-zone scope "
                    "and durable normal-climate condition are added for the pilot."
                ),
            },
        },
        {
            "session_id": "hvp20-mp03",
            "timestamp": "2026-04-30T08:00:00",
            "record_id": "tier7:337",
            "context": (
                "Quinn is leaving a parked vehicle, worries about security, and "
                "confirms that all vehicle doors should be locked."
            ),
            "turn_overrides": [
                (
                    "I'm leaving the parked car, but I'm worried about security. I "
                    "don't want anyone getting in."
                ),
                (
                    "I understand. You want the vehicle secure when you leave. "
                    "Should I lock all its doors?"
                ),
                "Yes, I'm sure. Lock all the vehicle doors.",
            ],
            "suffix": (
                "Keeping all doors locked is my normal preference when I am parked."
            ),
            "update": {
                "op": "add",
                "slot_id": "codriver.parked.doors_locked",
                "tool_name": "carcontrol_door_set_locked",
                "argument_name": "locked",
                "value": True,
                "context_arguments": {"door": "all"},
                "condition": "when Quinn is parked",
                "reason": (
                    "The source explicitly locks an entrance when leaving; the "
                    "vehicle rewrite preserves the security intent and adds "
                    "all-door parked scope."
                ),
            },
        },
    ],
    "quizzes": [
        {
            "quiz_id": "hvp20-turn-05",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s8-t01",
            "query": (
                "Finley begins a routine drive. Restore the remembered cabin warmth "
                "and the color used for the interior glow."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.normal.ambient_color",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 18},
                },
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
            ],
        },
        {
            "quiz_id": "hvp20-turn-06",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-mp03-t03",
            "query": (
                "Quinn finishes a normal drive and parks. Restore Quinn's three "
                "stored choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.normal.temperature",
                "codriver.parked.doors_locked",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "yellow"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 25},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                },
            ],
        },
        {
            "quiz_id": "hvp20-turn-07",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s10-t01",
            "query": (
                "Finley asks to begin the usual trip without naming a place, but "
                "the spoken prompts feel excessive. Apply both remembered choices."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.default.destination",
                "driver.frequent_prompts.voice_mode",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_navigation_navigate_to",
                    "arguments": {"destination": "Home"},
                },
                {
                    "name": "carcontrol_navigation_set_voice_mode",
                    "arguments": {"mode": "mute"},
                },
            ],
        },
        {
            "quiz_id": "hvp20-turn-08",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s12-t01",
            "query": (
                "Finley needs to see clearly through the glass behind the passengers, "
                "and stale cabin air is causing a headache. Apply both remembered "
                "climate responses."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.rear_visibility.defrost",
                "driver.stale_air.circulation",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_mode",
                    "arguments": {"zone": "rear", "mode": "defrost"},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
            ],
        },
        {
            "quiz_id": "hvp20-turn-09",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s16-t01",
            "query": (
                "Finley begins a routine drive. Restore the current cabin warmth "
                "and interior glow."
            ),
            "reasoning_type": "coreference_resolution",
            "evidence_slot_ids": [
                "driver.normal.temperature",
                "driver.normal.ambient_color",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 21},
                },
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
            ],
        },
        {
            "quiz_id": "hvp20-turn-10",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s15-t01",
            "query": (
                "After Finley's first two temperature revisions, apply the "
                "corrected normal value active at this cutoff."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 24},
                }
            ],
        },
        {
            "quiz_id": "hvp20-turn-11",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s16-t01",
            "query": (
                "Finley revised the normal temperature once more. Apply the "
                "corrected current cabin setting."
            ),
            "reasoning_type": "error_correction",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 21},
                }
            ],
        },
        {
            "quiz_id": "hvp20-turn-12",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s16-t01",
            "query": (
                "Apply Quinn's normal ambient-light preference rather than Finley's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.ambient_color"],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "yellow"},
                }
            ],
        },
        {
            "quiz_id": "hvp20-turn-13",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s16-t01",
            "query": (
                "Finley, not Quinn, begins a normal drive. Apply Finley's current "
                "ambient color and cabin temperature."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.temperature",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 21},
                },
            ],
        },
        {
            "quiz_id": "hvp20-turn-14",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s16-t01",
            "query": (
                "Quinn begins a normal drive. Apply Quinn's cabin temperature "
                "rather than Finley's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 25},
                }
            ],
        },
        {
            "quiz_id": "hvp20-turn-15",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s16-t01",
            "query": (
                "Finley begins a normal drive. Apply Finley's cabin temperature "
                "rather than Quinn's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 21},
                }
            ],
        },
        {
            "quiz_id": "hvp20-turn-16",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s16-t01",
            "query": (
                "Quinn is parked. Apply Quinn's door-access preference rather than "
                "Finley's."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": ["codriver.parked.doors_locked"],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                }
            ],
        },
        {
            "quiz_id": "hvp20-turn-17",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s16-t01",
            "query": (
                "Finley is parked while stale air causes a headache. Apply Finley's "
                "door preference rather than Quinn's and restore the cabin-air "
                "source."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.parked.doors_locked",
                "driver.stale_air.circulation",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
                {
                    "name": "carcontrol_airConditioner_set_circulation",
                    "arguments": {"zone": "all", "circulation": "outside"},
                },
            ],
        },
        {
            "quiz_id": "hvp20-turn-18",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s16-t01",
            "query": (
                "Quinn, not Finley, finishes a normal drive and parks. Apply Quinn's "
                "three preferences."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "codriver.normal.ambient_color",
                "codriver.normal.temperature",
                "codriver.parked.doors_locked",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "yellow"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 25},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": True},
                },
            ],
        },
        {
            "quiz_id": "hvp20-turn-19",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s7-t01",
            "query": (
                "Before Finley revised the normal cabin temperature, apply the "
                "setting active at this early cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 18},
                }
            ],
        },
        {
            "quiz_id": "hvp20-turn-20",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s14-t01",
            "query": (
                "After Finley's first temperature revision, apply the normal value "
                "active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 27},
                }
            ],
        },
        {
            "quiz_id": "hvp20-turn-21",
            "quiz_type": "TURN",
            "cutoff_turn_id": "hvp20-s15-t01",
            "query": (
                "After Finley's next temperature revision, apply the normal value "
                "active at this cutoff."
            ),
            "reasoning_type": "state_shift",
            "evidence_slot_ids": ["driver.normal.temperature"],
            "gold_calls": [
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 24},
                }
            ],
        },
        {
            "quiz_id": "hvp20-final-07",
            "quiz_type": "FINAL",
            "cutoff_turn_id": "hvp20-s16-t01",
            "query": (
                "Finley, rather than Quinn, finishes a normal drive and parks. Apply "
                "Finley's current ambient color, cabin temperature, and door-access "
                "preference."
            ),
            "reasoning_type": "preference_conflict",
            "evidence_slot_ids": [
                "driver.normal.ambient_color",
                "driver.normal.temperature",
                "driver.parked.doors_locked",
            ],
            "gold_calls": [
                {
                    "name": "carcontrol_light_set_ambient_color",
                    "arguments": {"color": "white"},
                },
                {
                    "name": "carcontrol_airConditioner_set_temperature",
                    "arguments": {"zone": "all", "temperature": 21},
                },
                {
                    "name": "carcontrol_door_set_locked",
                    "arguments": {"door": "all", "locked": False},
                },
            ],
        },
    ],
}


EXTENSIONS = {
    "HVP01": HVP01_EXTENSION,
    "HVP02": HVP02_EXTENSION,
    "HVP03": HVP03_EXTENSION,
    "HVP04": HVP04_EXTENSION,
    "HVP05": HVP05_EXTENSION,
    "HVP06": HVP06_EXTENSION,
    "HVP07": HVP07_EXTENSION,
    "HVP08": HVP08_EXTENSION,
    "HVP09": HVP09_EXTENSION,
    "HVP10": HVP10_EXTENSION,
    "HVP11": HVP11_EXTENSION,
    "HVP12": HVP12_EXTENSION,
    "HVP13": HVP13_EXTENSION,
    "HVP14": HVP14_EXTENSION,
    "HVP15": HVP15_EXTENSION,
    "HVP16": HVP16_EXTENSION,
    "HVP17": HVP17_EXTENSION,
    "HVP18": HVP18_EXTENSION,
    "HVP19": HVP19_EXTENSION,
    "HVP20": HVP20_EXTENSION,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(EXTENSIONS), default="HVP01")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(
            "/mnt/data/hj153lee/PalmClaw/external-sources/"
            "human-derived-vehicle-dialogue"
        ),
    )
    parser.add_argument(
        "--source-output-root",
        type=Path,
        default=Path(
            "evaluation/human-authored-vehicle-memory/multiparty-quiz-expansion-v1"
        ),
    )
    parser.add_argument(
        "--derived-output-root",
        type=Path,
        default=Path(
            "memory_training/data/human-authored-v2-multiparty-quiz-expansion-v1"
        ),
    )
    parser.add_argument(
        "--vehiclemembench-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_source_path(repository_root: Path, scenario_id: str) -> Path:
    index = int(scenario_id[-2:])
    directory = f"pilot-hv{index:02d}" if index <= 4 else f"pilot-hvp{index:02d}"
    return (
        repository_root
        / "evaluation"
        / "human-authored-vehicle-memory"
        / directory
        / "scenario-source.json"
    )


def used_audio_records(repository_root: Path) -> set[str]:
    root = repository_root / "evaluation" / "human-authored-vehicle-memory"
    result: set[str] = set()
    for path in sorted(root.glob("pilot-hv*/scenario-source.json")):
        source = json.loads(path.read_text(encoding="utf-8"))
        for catalog in source.get("external_sources", []):
            if catalog.get("dataset") == "Audio2Tool":
                result.update(str(record) for record in catalog.get("records", []))
    return result


def external_session(
    spec: dict[str, Any],
    secondary_short_id: str,
    audio_tiers: dict[str, dict[int, dict]],
) -> dict[str, Any]:
    tier_name, numeric_text = spec["record_id"].split(":", 1)
    numeric_id = int(numeric_text)
    record = audio_tiers[tier_name][numeric_id]
    if tier_name == "tier7":
        raw_turns = [
            (
                "assistant" if turn["role"] == "agent" else secondary_short_id,
                turn["content"].strip(),
            )
            for turn in record["chat_history"]
        ]
    else:
        raw_turns = [(secondary_short_id, record["query"].strip())]

    turn_overrides = spec.get("turn_overrides")
    if turn_overrides is not None and len(turn_overrides) != len(raw_turns):
        raise ValueError(
            f"{spec['session_id']} has {len(turn_overrides)} turn overrides for "
            f"{len(raw_turns)} source turns"
        )

    last_user = max(
        index
        for index, (speaker, _) in enumerate(raw_turns)
        if speaker == secondary_short_id
    )
    turns = []
    for index, (speaker, original_text) in enumerate(raw_turns):
        is_update = index == last_user
        rewritten = (
            turn_overrides[index] if turn_overrides is not None else original_text
        )
        text = rewritten
        if is_update:
            separator = " " if rewritten.endswith((".", "?", "!")) else ". "
            text += separator + spec["suffix"].strip()
        turn = {
            "turn_id": f"{spec['session_id']}-t{index + 1:02d}",
            "speaker": speaker,
            "text": text,
            "source_trace": {
                "dataset": "Audio2Tool",
                "record_id": spec["record_id"],
                "record_turn_index": index,
                "origin": "synthetic",
                "reuse_mode": (
                    "vehicle_domain_rewrite_with_persistence"
                    if turn_overrides is not None and is_update
                    else "vehicle_domain_rewrite"
                    if turn_overrides is not None
                    else "minimal_persistence_adaptation"
                    if is_update
                    else "verbatim_text_role_normalized"
                ),
                "original_text": original_text,
            },
        }
        if is_update:
            turn["update"] = deepcopy(spec["update"])
        turns.append(turn)
    return {
        "session_id": spec["session_id"],
        "timestamp": spec["timestamp"],
        "context": spec["context"],
        "turns": turns,
    }


def extend_source(
    base_path: Path,
    extension: dict[str, Any],
    audio_tiers: dict[str, dict[int, dict]],
    globally_used_records: set[str],
) -> dict[str, Any]:
    source = json.loads(base_path.read_text(encoding="utf-8"))
    base_hash = file_sha256(base_path)
    secondary = deepcopy(extension["secondary_user"])
    if secondary["short_id"] in {speaker["short_id"] for speaker in source["speakers"]}:
        raise ValueError(f"duplicate speaker short ID: {secondary['short_id']}")
    source["speakers"].insert(-1, secondary)

    record_ids = [spec["record_id"] for spec in extension["sessions"]]
    overlap = sorted(set(record_ids) & globally_used_records)
    if overlap:
        raise ValueError(f"extension reuses original HVP Audio2Tool records: {overlap}")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("extension repeats an Audio2Tool record")

    sessions = [
        external_session(spec, secondary["short_id"], audio_tiers)
        for spec in extension["sessions"]
    ]
    insert_after = extension["insert_after_session_id"]
    insertion_index = (
        next(
            index
            for index, row in enumerate(source["sessions"])
            if row["session_id"] == insert_after
        )
        + 1
    )
    source["sessions"][insertion_index:insertion_index] = sessions
    timestamps = [
        datetime.fromisoformat(row["timestamp"]) for row in source["sessions"]
    ]
    if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
        raise ValueError("extended sessions are not strictly chronological")

    audio_catalog = next(
        row for row in source["external_sources"] if row["dataset"] == "Audio2Tool"
    )
    audio_catalog["records"].extend(record_ids)
    source["quizzes"].extend(deepcopy(extension["quizzes"]))
    source["authorship"]["note"] = (
        "Versioned multi-participant external adaptation. Dialogue records are "
        "source-traced; participant normalization, persistence edits, memory "
        "labels, and quizzes remain model-assembled pending human review."
    )
    source["multi_participant_extension"] = {
        "schema_version": "vehiclemembench-hvp-multiparty-extension-v1",
        "base_source_path": str(base_path.resolve()),
        "base_source_sha256": base_hash,
        "secondary_speaker_id": secondary["speaker_id"],
        "inserted_audio2tool_records": record_ids,
        "inserted_session_count": len(sessions),
        "inserted_turn_count": sum(len(row["turns"]) for row in sessions),
        "inserted_update_count": len(sessions),
        "appended_quiz_count": len(extension["quizzes"]),
        "method_blind": True,
        "human_review_status": "PENDING",
    }
    return source


def write_attribution(path: Path, source: dict[str, Any]) -> None:
    extension = source["multi_participant_extension"]
    records = ", ".join(
        f"`{value}`" for value in extension["inserted_audio2tool_records"]
    )
    payload = f"""# {source["scenario_id"]} multi-participant extension

This is a versioned extension of the frozen external-adapted pilot. The base
scenario is unchanged at `{extension["base_source_sha256"]}`.

- Added participant: `{extension["secondary_speaker_id"]}`
- Added externally traced Audio2Tool records: {records}
- Added sessions/turns/updates: {extension["inserted_session_count"]} /
  {extension["inserted_turn_count"]} / {extension["inserted_update_count"]}
- Added quizzes: {extension["appended_quiz_count"]}
- Selection was blind to Summary, Patch, and Delta-v3 predictions.
- Audio2Tool is CC BY-NC 4.0 and its query text is synthetic.
- This extension remains model-assembled and is not human-authored until the
  planned independent human rewrite and validation are complete.
"""
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    scenario_id = args.scenario
    extension = EXTENSIONS[scenario_id]
    base_path = original_source_path(repository_root, scenario_id).resolve(strict=True)
    _, audio_tiers = load_external_records(args.external_root.resolve(strict=True))
    other_extension_records = {
        session["record_id"]
        for other_scenario_id, other_extension in EXTENSIONS.items()
        if other_scenario_id != scenario_id
        for session in other_extension["sessions"]
    }
    source = extend_source(
        base_path,
        extension,
        audio_tiers,
        used_audio_records(repository_root) | other_extension_records,
    )

    source_dir = args.source_output_root / scenario_id
    derived_dir = args.derived_output_root / scenario_id
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "scenario-source.json"
    source_path.write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_attribution(source_dir / "SOURCE_ATTRIBUTION.md", source)
    manifest = build(source_path, derived_dir, args.vehiclemembench_root)
    print(
        json.dumps(
            {
                "scenario_id": scenario_id,
                "source": str(source_path),
                "derived": str(derived_dir),
                "statistics": manifest["statistics"],
                "validation": manifest["validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
