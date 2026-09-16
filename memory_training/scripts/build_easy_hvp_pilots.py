"""Build an explicitly easy, multi-participant HVP evaluation stratum.

The generated sources preserve the HVP source-trace contract, but deliberately
make every durable vehicle preference and every Quiz request explicit.  These
are model-assembled pilots pending independent human rewriting/review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ENVISIONED_REL = "envisioned-va/perfect-va_readable-dialogues_20-09-09.csv"
TIER7_REL = "audio2tool/public/tier7_multiturn_data/tier7_multiturn.json"
TIER5_REL = "audio2tool/public/tier5_needle_data/tier5_needle.json"

PEOPLE = [
    ("Avery", "Blake"),
    ("Casey", "Drew"),
    ("Emery", "Frankie"),
    ("Harper", "Jordan"),
    ("Morgan", "Riley"),
    ("Parker", "Quinn"),
    ("Reese", "Sawyer"),
    ("Taylor", "Alex"),
    ("Cameron", "Dakota"),
    ("Hayden", "Jamie"),
    ("Logan", "Marley"),
    ("Finley", "Kendall"),
    ("River", "Bailey"),
    ("Sage", "Robin"),
    ("Ellis", "Sydney"),
    ("Lane", "Micah"),
    ("Shiloh", "Remy"),
    ("Noel", "Arden"),
    ("Kit", "Devon"),
    ("Milan", "Jules"),
]

TEMPERATURES = [
    (20, 22), (21, 23), (19, 21), (22, 24), (20, 23),
    (21, 22), (19, 20), (22, 23), (20, 21), (21, 24),
    (18, 21), (23, 20), (20, 24), (21, 19), (22, 20),
    (19, 23), (24, 21), (18, 22), (23, 21), (20, 22),
]
PASSENGER_TEMPERATURES = [
    24, 22, 24, 21, 22, 24, 25, 20, 22, 19,
    24, 22, 21, 23, 24, 21, 20, 25, 19, 24,
]
MUSIC_VOLUMES = [
    (18, 24), (20, 26), (22, 28), (24, 30), (26, 32),
    (16, 22), (19, 25), (21, 27), (23, 29), (25, 31),
    (17, 23), (21, 29), (24, 31), (18, 27), (22, 30),
    (26, 34), (15, 24), (20, 28), (23, 32), (19, 26),
]
AMBIENT_COLORS = [
    ("blue", "white"),
    ("orange", "blue"),
    ("green", "white"),
    ("purple", "orange"),
    ("white", "blue"),
    ("blue", "green"),
    ("orange", "white"),
    ("green", "blue"),
    ("purple", "white"),
    ("white", "orange"),
    ("cyan", "white"),
    ("yellow", "blue"),
    ("pink", "white"),
    ("red", "yellow"),
    ("cyan", "purple"),
    ("yellow", "green"),
    ("pink", "blue"),
    ("red", "white"),
    ("cyan", "orange"),
    ("yellow", "purple"),
]
DESTINATIONS = [
    "Home", "Office", "Gym", "Airport", "Library", "School", "Hospital",
    "Station", "Hotel", "Market", "Museum", "Pharmacy", "University",
    "Theater", "Bakery", "Clinic", "Marina", "Stadium", "Workshop", "Cafe",
]

LOW_QUALITY_SOURCE_MARKERS = re.compile(
    r"\.\.\.|----|\([^)]*\)|\[[^]]*\]|\*|films showing at .* is now showing|"
    r"\b(?:lists?|placeholder|reads out|name and author|film [a-z]|x, y and z)\b",
    re.IGNORECASE,
)
AMBIGUOUS_UPDATE_SOURCE_MARKERS = re.compile(
    r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|temperature|degree|"
    r"volume|music|heat|heating|cool|cooling|climate|air conditioning|"
    r"air conditioner|seat|permanent(?:ly)?|always|default|custom profiles?|usually|"
    r"prefer|preference|remember|saved?|stored?)\b",
    re.IGNORECASE,
)
EXCLUDED_ENVISIONED_RECORDS = {"100:conversational"}


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
    parser.add_argument("--start-number", type=int, default=1)
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def human_turns(text: str) -> list[tuple[str, str]]:
    turns = []
    for line in text.replace("\r", "").strip().splitlines():
        if not line.strip():
            continue
        # Exclude malformed records instead of guessing a speaker, preserving
        # an exact and auditable mapping to the external source.
        if ": " not in line:
            return []
        role, content = line.split(": ", 1)
        if role not in {"user", "voice_assistant"} or not content.strip():
            return []
        turns.append(("assistant" if role == "voice_assistant" else "human", content.strip()))
    return turns


def used_records(
    repository_root: Path, excluded_scenarios: set[str] | None = None
) -> tuple[set[str], set[str]]:
    human: set[str] = set()
    audio: set[str] = set()
    source_root = repository_root / "evaluation/human-authored-vehicle-memory"
    paths = list(source_root.glob("pilot-hv*/scenario-source.json"))
    paths += list(source_root.glob("multiparty-quiz-expansion-v1/HVP*/scenario-source.json"))
    paths += list(source_root.glob("easy-explicit-v1/HVE*/scenario-source.json"))
    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        if source.get("scenario_id") in (excluded_scenarios or set()):
            continue
        for catalog in source.get("external_sources", []):
            target = audio if catalog.get("dataset") == "Audio2Tool" else human
            target.update(str(value) for value in catalog.get("records", []))
        audio.update(
            str(value)
            for value in source.get("multi_participant_extension", {}).get(
                "inserted_audio2tool_records", []
            )
        )
    return human, audio


def allocate_human_records(
    rows: list[dict[str, str]], used: set[str], count: int
) -> list[list[tuple[str, str, list[tuple[str, str]]]]]:
    candidates = []
    for row in sorted(rows, key=lambda value: int(value["id"])):
        for field in ("search", "conversational"):
            record_id = f"{row['id']}:{field}"
            if (
                record_id in used
                or record_id in EXCLUDED_ENVISIONED_RECORDS
                or not row.get(field, "").strip()
            ):
                continue
            turns = human_turns(row[field])
            roles = [role for role, _ in turns]
            longest_role_run = max(
                (
                    len(match.group(0))
                    for match in re.finditer(
                        r"(h+|a+)",
                        "".join("h" if role == "human" else "a" for role in roles),
                    )
                ),
                default=0,
            )
            if (
                4 <= len(turns) <= 7
                and set(roles) == {"human", "assistant"}
                and longest_role_run <= 2
                and not LOW_QUALITY_SOURCE_MARKERS.search(row[field])
            ):
                candidates.append((record_id, field, turns))

    allocations = []
    cursor = 0
    for _ in range(count):
        chosen: list[tuple[str, str, list[tuple[str, str]]]] | None = None

        def search(position: int, remaining: int, current: list[Any]) -> bool:
            nonlocal chosen
            if remaining == 0 and len(current) == 3:
                chosen = list(current)
                return True
            if remaining <= 0 or position >= len(candidates) or len(current) >= 3:
                return False
            for index in range(position, min(len(candidates), position + 80)):
                candidate = candidates[index]
                if candidate[0] in used:
                    continue
                current.append(candidate)
                if search(index + 1, remaining - len(candidate[2]), current):
                    return True
                current.pop()
            return False

        if not search(cursor, 15, []):
            if not search(0, 15, []):
                raise RuntimeError("could not allocate 15 disjoint human-authored turns")
        assert chosen is not None
        allocations.append(chosen)
        used.update(value[0] for value in chosen)
        cursor = candidates.index(chosen[-1]) + 1
    return allocations


def select_tier7_noops(rows: list[dict[str, Any]], used: set[str], total: int) -> list[dict[str, Any]]:
    selected = []
    seen_text: set[str] = set()
    for row in sorted(rows, key=lambda value: int(value["id"])):
        record_id = f"tier7:{row['id']}"
        history = row.get("chat_history", [])
        signature = normalized(" ".join(str(turn.get("content", "")) for turn in history))
        if record_id in used or len(history) != 3 or signature in seen_text:
            continue
        # Low-numbered Tier-7 examples are unambiguously residential-lighting
        # conversations, making them safe vehicle-memory NO_OP examples.
        if int(row["id"]) >= 1000:
            continue
        selected.append(row)
        used.add(record_id)
        seen_text.add(signature)
        if len(selected) == total:
            return selected
    raise RuntimeError(f"needed {total} disjoint Tier-7 NO_OP records")


def select_tier5_updates(rows: list[dict[str, Any]], used: set[str], total: int) -> list[dict[str, Any]]:
    selected = []
    seen_text: set[str] = set()
    for row in sorted(rows, key=lambda value: int(value["id"])):
        record_id = f"tier5:{row['id']}"
        signature = normalized(str(row.get("query", "")))
        if (
            record_id in used
            or row.get("domain") != "smart_car"
            or signature in seen_text
            or AMBIGUOUS_UPDATE_SOURCE_MARKERS.search(str(row.get("query", "")))
        ):
            continue
        selected.append(row)
        used.add(record_id)
        seen_text.add(signature)
        if len(selected) == total:
            return selected
    raise RuntimeError(f"needed {total} disjoint Tier-5 vehicle records")


def slot_definitions(index: int, primary: str, secondary: str) -> list[dict[str, Any]]:
    temperature = TEMPERATURES[index]
    music = MUSIC_VOLUMES[index]
    color = AMBIENT_COLORS[index]
    return [
        {
            "slot_id": "driver.default.temperature",
            "owner_short": "user",
            "owner_name": primary,
            "tool_name": "carcontrol_airConditioner_set_temperature",
            "argument_name": "temperature",
            "context_arguments": {"zone": "driver"},
            "initial": temperature[0],
            "replacement": temperature[1],
            "setting": "driver-zone cabin temperature",
            "value_text": lambda value: f"{value} degrees",
            "quiz": lambda name: f"Set the driver-zone cabin temperature to {name}'s saved value.",
        },
        {
            "slot_id": "codriver.default.music_volume",
            "owner_short": "co_user",
            "owner_name": secondary,
            "tool_name": "carcontrol_music_set_volume",
            "argument_name": "volume",
            "context_arguments": {},
            "initial": music[0],
            "replacement": music[1],
            "setting": "music volume",
            "value_text": lambda value: str(value),
            "quiz": lambda name: f"Set the music volume to {name}'s saved value.",
        },
        {
            "slot_id": "driver.default.ambient_color",
            "owner_short": "user",
            "owner_name": primary,
            "tool_name": "carcontrol_light_set_ambient_color",
            "argument_name": "color",
            "context_arguments": {},
            "initial": color[0],
            "replacement": color[1],
            "setting": "ambient-light color",
            "value_text": lambda value: str(value),
            "quiz": lambda name: f"Set the ambient lighting to {name}'s saved color.",
        },
        {
            "slot_id": "codriver.default.temperature",
            "owner_short": "co_user",
            "owner_name": secondary,
            "tool_name": "carcontrol_airConditioner_set_temperature",
            "argument_name": "temperature",
            "context_arguments": {"zone": "passenger"},
            "initial": PASSENGER_TEMPERATURES[index],
            "replacement": None,
            "setting": "passenger-zone cabin temperature",
            "value_text": lambda value: f"{value} degrees",
            "quiz": lambda name: f"Set the passenger-zone cabin temperature to {name}'s saved value.",
        },
        {
            "slot_id": "driver.default.destination",
            "owner_short": "user",
            "owner_name": primary,
            "tool_name": "carcontrol_navigation_navigate_to",
            "argument_name": "destination",
            "context_arguments": {},
            "initial": DESTINATIONS[index],
            "replacement": None,
            "setting": "default navigation destination",
            "value_text": lambda value: str(value),
            "quiz": lambda name: f"Navigate to {name}'s saved default destination.",
        },
        {
            "slot_id": "codriver.default.seat_heating",
            "owner_short": "co_user",
            "owner_name": secondary,
            "tool_name": "carcontrol_seat_set_heating_level",
            "argument_name": "level",
            "context_arguments": {"seat": "passenger"},
            "initial": 2 + index % 2,
            "replacement": None,
            "setting": "passenger-seat heating level",
            "value_text": lambda value: f"level {value}",
            "quiz": lambda name: f"Set the passenger-seat heating to {name}'s saved level.",
        },
        {
            "slot_id": "driver.default.radio_volume",
            "owner_short": "user",
            "owner_name": primary,
            "tool_name": "carcontrol_radio_set_volume",
            "argument_name": "volume",
            "context_arguments": {},
            "initial": 28 + 2 * index,
            "replacement": None,
            "setting": "radio volume",
            "value_text": lambda value: str(value),
            "quiz": lambda name: f"Set the radio volume to {name}'s saved value.",
        },
    ]


def build_source(
    scenario_number: int,
    human_records: list[tuple[str, str, list[tuple[str, str]]]],
    tier7_records: list[dict[str, Any]],
    tier5_records: list[dict[str, Any]],
    hashes: dict[str, str],
) -> dict[str, Any]:
    scenario_id = f"HVE{scenario_number:02d}"
    primary, secondary = PEOPLE[scenario_number - 1]
    primary_id = f"external-{scenario_id.lower()}-driver"
    secondary_id = f"external-{scenario_id.lower()}-codriver"
    slots = slot_definitions(scenario_number - 1, primary, secondary)
    update_order = [
        (0, "add"),
        (1, "add"),
        (2, "add"),
        (0, "replace"),
        (3, "add"),
        (1, "replace"),
    ]

    noop_sessions: list[dict[str, Any]] = []
    for record_position, (record_id, field, turns) in enumerate(human_records):
        owner_short = "user" if record_position % 2 == 0 else "co_user"
        rendered = []
        for turn_index, (role, text) in enumerate(turns):
            rendered.append(
                {
                    "speaker": "assistant" if role == "assistant" else owner_short,
                    "text": text,
                    "source_trace": {
                        "dataset": "Envisioned Voice Assistant Dialogues",
                        "record_id": record_id,
                        "record_turn_index": turn_index,
                        "origin": "human_authored",
                        "reuse_mode": "verbatim_text_role_normalized",
                        "original_text": text,
                    },
                }
            )
        noop_sessions.append(
            {
                "context": "Externally sourced non-vehicle voice-assistant conversation; no durable vehicle preference is expressed.",
                "turns": rendered,
            }
        )

    for record_position, row in enumerate(tier7_records):
        owner_short = "co_user" if record_position % 2 == 0 else "user"
        rendered = []
        for turn_index, turn in enumerate(row["chat_history"]):
            text = str(turn["content"]).strip()
            rendered.append(
                {
                    "speaker": "assistant" if turn["role"] == "agent" else owner_short,
                    "text": text,
                    "source_trace": {
                        "dataset": "Audio2Tool",
                        "record_id": f"tier7:{row['id']}",
                        "record_turn_index": turn_index,
                        "origin": "synthetic",
                        "reuse_mode": "verbatim_text_role_normalized",
                        "original_text": text,
                    },
                }
            )
        noop_sessions.append(
            {
                "context": "Externally sourced residential smart-light conversation; explicitly outside the vehicle-memory domain.",
                "turns": rendered,
            }
        )

    update_sessions: list[dict[str, Any]] = []
    update_cutoffs: list[str] = []
    active_values: dict[str, Any] = {}
    for update_position, ((slot_index, op), row) in enumerate(zip(update_order, tier5_records)):
        slot = slots[slot_index]
        value = slot["initial"] if op == "add" else slot["replacement"]
        assert value is not None
        owner_short = slot["owner_short"]
        owner_name = slot["owner_name"]
        original = str(row["query"]).strip()
        if op == "add":
            explicit = (
                f"Also, my name is {owner_name}; please save this exact setting to my "
                f"permanent vehicle profile: set the {slot['setting']} to "
                f"{slot['value_text'](value)}."
            )
        else:
            explicit = (
                f"Also, my name is {owner_name}; please replace my earlier saved setting "
                f"in my permanent vehicle profile: set the {slot['setting']} to "
                f"{slot['value_text'](value)}."
            )
        separator = " " if original.endswith((".", "!", "?")) else ". "
        text = f"{original}{separator}{explicit}"
        update_sessions.append(
            {
                "context": f"{owner_name} explicitly states one durable vehicle-profile setting.",
                "turns": [
                    {
                        "speaker": owner_short,
                        "text": text,
                        "source_trace": {
                            "dataset": "Audio2Tool",
                            "record_id": f"tier5:{row['id']}",
                            "record_turn_index": 0,
                            "origin": "synthetic",
                            "reuse_mode": "minimal_persistence_adaptation",
                            "original_text": original,
                        },
                        "update": {
                            "op": op,
                            "slot_id": slot["slot_id"],
                            "owner": primary_id if owner_short == "user" else secondary_id,
                            "tool_name": slot["tool_name"],
                            "argument_name": slot["argument_name"],
                            "value": value,
                            "context_arguments": deepcopy(slot["context_arguments"]),
                            "condition": f"{owner_name}'s explicitly selected default vehicle profile",
                            "reason": (
                                f"{owner_name} explicitly asks to store this exact durable setting."
                                if op == "add"
                                else f"{owner_name} explicitly replaces the prior saved value."
                            ),
                        },
                    }
                ],
                "slot_index": slot_index,
                "op": op,
            }
        )
        active_values[slot["slot_id"]] = value

    # Alternate sourced distractor sessions and explicit one-turn updates so
    # memory updates are spread over the full 30-turn timeline.
    ordered: list[dict[str, Any]] = []
    for position in range(max(len(noop_sessions), len(update_sessions))):
        if position < len(noop_sessions):
            ordered.append(noop_sessions[position])
        if position < len(update_sessions):
            ordered.append(update_sessions[position])
    sessions = []
    start = datetime(2026, 2, 1, 9, 0) + timedelta(days=scenario_number)
    for session_index, session in enumerate(ordered, 1):
        session_id = f"{scenario_id.lower()}-s{session_index:02d}"
        turns = []
        for turn_index, turn in enumerate(session["turns"], 1):
            copied = deepcopy(turn)
            copied["turn_id"] = f"{session_id}-t{turn_index:02d}"
            turns.append(copied)
        sessions.append(
            {
                "session_id": session_id,
                "timestamp": (start + timedelta(days=7 * (session_index - 1))).isoformat(),
                "context": session["context"],
                "turns": turns,
            }
        )
        if "slot_index" in session:
            update_cutoffs.append(turns[-1]["turn_id"])

    quizzes: list[dict[str, Any]] = []
    # One direct Turn Quiz after each ADD/REPLACE, plus four non-duplicated
    # confirmations after early ADDs.
    for index, ((slot_index, op), cutoff) in enumerate(zip(update_order, update_cutoffs), 1):
        slot = slots[slot_index]
        quizzes.append(
            {
                "quiz_id": f"{scenario_id.lower()}-turn-{index:02d}",
                "quiz_type": "TURN",
                "cutoff_turn_id": cutoff,
                "query": slot["quiz"](slot["owner_name"]),
                "reasoning_type": "error_correction" if op == "replace" else "state_shift",
                "evidence_slot_ids": [slot["slot_id"]],
                "gold_calls": [
                    {
                        "name": slot["tool_name"],
                        "arguments": {
                            slot["argument_name"]: slot["initial"] if op == "add" else slot["replacement"],
                            **deepcopy(slot["context_arguments"]),
                        },
                    }
                ],
            }
        )
    for extra_index, update_index in enumerate((0, 1), 7):
        slot_index, _ = update_order[update_index]
        slot = slots[slot_index]
        quizzes.append(
            {
                "quiz_id": f"{scenario_id.lower()}-turn-{extra_index:02d}",
                "quiz_type": "TURN",
                "cutoff_turn_id": update_cutoffs[update_index],
                "query": f"Apply {slot['owner_name']}'s explicitly saved {slot['setting']} now.",
                "reasoning_type": "state_shift",
                "evidence_slot_ids": [slot["slot_id"]],
                "gold_calls": [
                    {
                        "name": slot["tool_name"],
                        "arguments": {
                            slot["argument_name"]: slot["initial"],
                            **deepcopy(slot["context_arguments"]),
                        },
                    }
                ],
            }
        )

    final_cutoff = sessions[-1]["turns"][-1]["turn_id"]
    final_index = 1
    active_slot_ids = {slots[index]["slot_id"] for index, _ in update_order}
    for slot in slots:
        if slot["slot_id"] not in active_slot_ids:
            continue
        value = active_values[slot["slot_id"]]
        arguments = {
            slot["argument_name"]: value,
            **deepcopy(slot["context_arguments"]),
        }
        for query in (
            f"Load {slot['owner_name']}'s currently saved {slot['setting']}.",
            f"Apply the latest {slot['setting']} stored for {slot['owner_name']}.",
        ):
            quizzes.append(
                {
                    "quiz_id": f"{scenario_id.lower()}-final-{final_index:02d}",
                    "quiz_type": "FINAL",
                    "cutoff_turn_id": final_cutoff,
                    "query": query,
                    "reasoning_type": "state_shift",
                    "evidence_slot_ids": [slot["slot_id"]],
                    "gold_calls": [{"name": slot["tool_name"], "arguments": arguments}],
                }
            )
            final_index += 1

    human_ids = [value[0] for value in human_records]
    audio_ids = [
        *[f"tier7:{row['id']}" for row in tier7_records],
        *[f"tier5:{row['id']}" for row in tier5_records],
    ]
    return {
        "schema_version": "vehiclemembench-human-authored-scenario-source-v1",
        "scenario_id": scenario_id,
        "scenario_index": 920 + scenario_number,
        "split": "pilot_excluded",
        "difficulty_stratum": "easy_explicit_v1",
        "difficulty_contract": {
            "durable_updates_are_explicit": True,
            "owner_named_in_every_update": True,
            "owner_named_in_every_quiz": True,
            "single_fact_single_tool_quizzes_only": True,
            "implicit_coreference": False,
            "multi_hop_reasoning": False,
            "conditional_inference": False,
        },
        "authorship": {
            "kind": "external_adapted_model_assembled_pilot",
            "eligible_for_human_test": False,
            "note": "Easy external-adapted pilot. External wording is source-traced; persistence edits, labels, and quizzes are model-assembled pending independent human rewriting and review.",
        },
        "external_sources": [
            {
                "dataset": "Envisioned Voice Assistant Dialogues",
                "url": "https://github.com/SarahTheres/Envisioned-Voice-Assistant-Dialogues",
                "license": "CC BY 4.0",
                "origin": "human_authored",
                "records": human_ids,
                "local_file_sha256": hashes["envisioned"],
            },
            {
                "dataset": "Audio2Tool",
                "url": "https://huggingface.co/datasets/RVtech/Audio2Tool",
                "license": "CC BY-NC 4.0",
                "origin": "synthetic",
                "records": audio_ids,
                "local_file_sha256": {
                    "tier7": hashes["tier7"],
                    "tier5": hashes["tier5"],
                },
            },
        ],
        "speakers": [
            {
                "speaker_id": primary_id,
                "short_id": "user",
                "speaker_name": primary,
                "role": "primary driver",
                "style": "source-record wording with explicit durable vehicle preferences",
            },
            {
                "speaker_id": secondary_id,
                "short_id": "co_user",
                "speaker_name": secondary,
                "role": "co-driver",
                "style": "source-record wording with explicit durable vehicle preferences",
            },
            {
                "speaker_id": f"external-{scenario_id.lower()}-assistant",
                "short_id": "assistant",
                "speaker_name": "Voice assistant",
                "role": "voice assistant",
                "style": "retains external source wording",
            },
        ],
        "sessions": sessions,
        "quizzes": quizzes,
    }


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    external_root = args.external_root.resolve(strict=True)
    if args.start_number < 1 or args.count < 1:
        raise ValueError("start-number and count must be positive")
    if args.start_number + args.count - 1 > len(PEOPLE):
        raise ValueError(f"scenario range exceeds configured maximum {len(PEOPLE)}")
    target_scenarios = {
        f"HVE{number:02d}"
        for number in range(args.start_number, args.start_number + args.count)
    }
    human_used, audio_used = used_records(repository_root, target_scenarios)

    envisioned_path = external_root / ENVISIONED_REL
    tier7_path = external_root / TIER7_REL
    tier5_path = external_root / TIER5_REL
    with envisioned_path.open(encoding="utf-8-sig", newline="") as handle:
        envisioned = list(csv.DictReader(handle))
    tier7 = json.loads(tier7_path.read_text(encoding="utf-8"))
    tier5 = json.loads(tier5_path.read_text(encoding="utf-8"))
    human_allocations = allocate_human_records(envisioned, human_used, args.count)
    tier7_selected = select_tier7_noops(tier7, audio_used, args.count * 3)
    tier5_selected = select_tier5_updates(tier5, audio_used, args.count * 6)
    hashes = {
        "envisioned": sha256(envisioned_path),
        "tier7": sha256(tier7_path),
        "tier5": sha256(tier5_path),
    }

    output_root = (
        repository_root
        / "evaluation/human-authored-vehicle-memory/easy-explicit-v1"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    for position in range(args.count):
        scenario_number = args.start_number + position
        source = build_source(
            scenario_number,
            human_allocations[position],
            tier7_selected[position * 3 : (position + 1) * 3],
            tier5_selected[position * 6 : (position + 1) * 6],
            hashes,
        )
        scenario_root = output_root / source["scenario_id"]
        scenario_root.mkdir(parents=True, exist_ok=True)
        source_path = scenario_root / "scenario-source.json"
        source_path.write_text(
            json.dumps(source, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        attribution = (
            f"# {source['scenario_id']} source attribution\n\n"
            "This is an external-adapted, model-assembled **Easy pilot**, not yet an "
            "eligible human-authored test item. Every dialogue turn retains a record-level "
            "source trace. Durable vehicle clauses, Patch labels, and Quiz items require "
            "independent human rewriting and approval.\n\n"
            f"- Envisioned records: {', '.join(source['external_sources'][0]['records'])}\n"
            f"- Audio2Tool records: {', '.join(source['external_sources'][1]['records'])}\n"
            "- License note: Audio2Tool is CC BY-NC 4.0 and therefore constrains this pilot.\n"
        )
        (scenario_root / "SOURCE_ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
        print(source_path)


if __name__ == "__main__":
    main()
