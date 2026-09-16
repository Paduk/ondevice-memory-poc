"""Naturalize the explicit Easy HVE corpus without changing its Gold semantics.

The source-derived clause of every UPDATE remains verbatim.  Only the appended
durable-preference sentence and the templated Quiz wording are rewritten.  The
result is still a model-assisted pilot and remains excluded from training and
from any claim of independently human-authored evaluation data.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


TRANSITIONS = (
    "One more thing before I forget.",
    "By the way, I have a preference to add.",
    "While we're here, I'd like to save one setting.",
    "There's one preference I'd like you to remember.",
    "Please keep one regular setting for me.",
    "Since this car is shared, I want to identify my own setting.",
    "There's a regular setting I'd like ready the next time I get in.",
    "I also want to sort out my usual setup.",
    "Let me update my profile too.",
    "Here's how I normally like the car.",
    "Could you remember one thing for me?",
    "Before we finish, I have one preference to add.",
    "I'd like this kept as my regular setting.",
    "Here's another regular setting for my profile.",
    "Please make a note of my usual choice.",
    "There's also a setting I use every time.",
    "While I'm thinking about my profile, here's one preference.",
    "I want the car to remember this.",
    "Please keep this for my future drives.",
    "I have another preference for my profile.",
)

IDENTITIES = (
    "This is {name} speaking.",
    "{name} here.",
    "This next preference is for {name}'s profile.",
    "I'm {name}.",
    "For clarity, this is {name}.",
)

CORES = {
    ("driver.default.temperature", "add"): (
        "I'm most comfortable with the driver's side at {value} degrees. Please remember that as my usual temperature.",
        "Set my normal driver-side temperature to {value} degrees and keep it for future drives.",
        "I usually want {value} degrees on the driver's side. Remember that preference for next time.",
        "The driver's side should normally be {value} degrees for me. Please save that as my regular choice.",
        "From now on, use {value} degrees as my usual driver-zone temperature.",
    ),
    ("driver.default.temperature", "replace"): (
        "My preferred driver's-side temperature is now {value} degrees. Use that from now on instead of the temperature saved before.",
        "Change my regular driver-zone temperature to {value} degrees and forget the old value.",
        "I want to revise my saved driver's-side temperature: make it {value} degrees from now on.",
        "Replace my previous driver-temperature preference with {value} degrees. That's the value I want remembered now.",
        "The old driver's-side temperature no longer suits me; keep {value} degrees as my new usual setting.",
    ),
    ("codriver.default.music_volume", "add"): (
        "I normally listen at music volume {value}. Please remember that for future rides.",
        "Keep the music at volume {value} as my usual setting whenever I'm riding along.",
        "My regular music-volume choice is {value}. Save that for next time.",
        "Volume {value} is where I like the music. Please remember it as my normal level.",
        "For future trips, use {value} as my usual music volume.",
    ),
    ("codriver.default.music_volume", "replace"): (
        "My usual music volume has changed to {value}. Replace the old saved level and remember this one.",
        "Change my saved music volume to {value}; I don't want the previous level anymore.",
        "I want to revise my music setting. Use volume {value} from now on instead of what was saved before.",
        "Replace my earlier music-volume preference with {value}. That's my new regular level.",
        "The old music volume no longer works for me, so keep {value} as my usual setting now.",
    ),
    ("driver.default.ambient_color", "add"): (
        "I like the cabin ambient lights set to {value}. Keep that as my regular lighting choice.",
        "Make {value} my usual ambient-light color and remember it for future drives.",
        "My preferred cabin-light color is {value}. Please save that as my normal setting.",
        "I normally want the ambient lighting in {value}. Remember that choice for next time.",
        "For my usual setup, keep the cabin ambient lights {value}.",
    ),
    ("codriver.default.temperature", "add"): (
        "On the passenger side, {value} degrees is comfortable for me. Remember that as my regular temperature.",
        "Set my usual passenger-zone temperature to {value} degrees and keep it for future rides.",
        "I normally want {value} degrees on the passenger side. Please save that preference.",
        "The passenger side should usually be {value} degrees for me. Remember it for next time.",
        "For future trips, use {value} degrees as my regular passenger-zone temperature.",
    ),
}

QUIZ_TEMPLATES = {
    "driver.default.temperature": (
        "Please set the driver's side to the temperature {name} has saved.",
        "Use {name}'s remembered temperature for the driver's side.",
        "Could you apply {name}'s saved driver-side temperature now?",
        "Set the driver zone to {name}'s current preferred temperature.",
    ),
    "codriver.default.music_volume": (
        "Please put the music at {name}'s saved volume.",
        "Use the music volume remembered for {name}.",
        "Could you apply {name}'s current music-volume preference?",
        "Set the music to the level {name} has saved.",
    ),
    "driver.default.ambient_color": (
        "Please set the cabin ambient lights to {name}'s saved color.",
        "Use the ambient-light color remembered for {name}.",
        "Could you apply {name}'s current cabin-light color?",
        "Set the ambient lighting to the color {name} has saved.",
    ),
    "codriver.default.temperature": (
        "Please set the passenger side to the temperature {name} has saved.",
        "Use {name}'s remembered temperature for the passenger side.",
        "Could you apply {name}'s saved passenger-zone temperature now?",
        "Set the passenger zone to {name}'s current preferred temperature.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("evaluation/human-authored-vehicle-memory/easy-explicit-v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation/human-authored-vehicle-memory/easy-natural-explicit-v1"),
    )
    parser.add_argument("--count", type=int, default=20)
    return parser.parse_args()


def value_text(update: dict[str, Any]) -> str:
    return str(update["value"])


def natural_clause(
    scenario_number: int,
    update_position: int,
    owner_name: str,
    update: dict[str, Any],
) -> str:
    key = (str(update["slot_id"]), str(update["op"]))
    templates = CORES.get(key)
    if templates is None:
        raise ValueError(f"unsupported Easy update: {key}")
    transition = TRANSITIONS[(scenario_number + update_position * 3 - 1) % len(TRANSITIONS)]
    if update["op"] == "replace":
        transition = transition.replace("preference to add", "preference to change")
    identity = IDENTITIES[(scenario_number + update_position - 1) % len(IDENTITIES)].format(
        name=owner_name
    )
    core = templates[(scenario_number + 2 * update_position - 1) % len(templates)].format(
        value=value_text(update)
    )
    return f"{transition} {identity} {core}"


def naturalize(source: dict[str, Any], scenario_number: int) -> dict[str, Any]:
    result = deepcopy(source)
    result["difficulty_stratum"] = "easy_natural_explicit_v1"
    result["authorship"]["note"] = (
        "Easy natural-explicit external-adapted pilot. External source clauses, "
        "durable-preference rewrites, labels, and quizzes remain model-assisted "
        "pending independent human rewriting and review."
    )
    names = {speaker["short_id"]: speaker["speaker_name"] for speaker in result["speakers"]}
    for speaker in result["speakers"]:
        if speaker["short_id"] != "assistant":
            speaker["style"] = "source-record wording with natural, explicit durable preferences"

    update_position = 0
    slot_owners: dict[str, str] = {}
    for session in result["sessions"]:
        for turn in session["turns"]:
            update = turn.get("update")
            if update is None:
                continue
            original = str(turn["source_trace"]["original_text"]).strip()
            owner_name = names[turn["speaker"]]
            separator = " " if original.endswith((".", "!", "?")) else ". "
            clause = natural_clause(scenario_number, update_position, owner_name, update)
            turn["text"] = f"{original}{separator}{clause}"
            turn["source_trace"]["reuse_mode"] = "minimal_persistence_adaptation"
            session["context"] = (
                f"{owner_name} naturally states one explicit, durable vehicle-profile setting."
            )
            slot_owners[str(update["slot_id"])] = owner_name
            update_position += 1

    slot_occurrences: dict[str, int] = {}
    for quiz in result["quizzes"]:
        slot_id = str(quiz["evidence_slot_ids"][0])
        occurrence = slot_occurrences.get(slot_id, 0)
        quiz["query"] = QUIZ_TEMPLATES[slot_id][occurrence % 4].format(
            name=slot_owners[slot_id]
        )
        slot_occurrences[slot_id] = occurrence + 1
    return result


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for scenario_number in range(1, args.count + 1):
        scenario_id = f"HVE{scenario_number:02d}"
        source_path = source_root / scenario_id / "scenario-source.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        result = naturalize(source, scenario_number)
        scenario_root = output_root / scenario_id
        scenario_root.mkdir(parents=True, exist_ok=True)
        (scenario_root / "scenario-source.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        attribution = (
            f"# {scenario_id} source attribution\n\n"
            "This natural-explicit variant preserves every record-level source trace from "
            "Easy-explicit v1. Only the model-assisted durable-preference clause and Quiz "
            "wording were rewritten. It is not independently human-authored and remains "
            "ineligible for the final human test until human approval.\n\n"
            f"- Parent source: `easy-explicit-v1/{scenario_id}`\n"
            "- External records and licenses: unchanged from the parent source.\n"
        )
        (scenario_root / "SOURCE_ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
        print(scenario_root / "scenario-source.json")


if __name__ == "__main__":
    main()
