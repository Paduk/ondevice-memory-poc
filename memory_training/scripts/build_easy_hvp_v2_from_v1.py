"""Create a strictly easier HVE calibration set from Easy v1.

V2 keeps the same owners, timeline, state transitions, and Tool-call golds, but
removes compound intents from UPDATE turns and makes every human NO_OP explicitly
non-vehicle and one-time. The original external text remains in source_trace.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "evaluation/human-authored-vehicle-memory/easy-explicit-v1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "evaluation/human-authored-vehicle-memory/easy-explicit-v2"
        ),
    )
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


def setting(update: dict[str, Any]) -> tuple[str, str]:
    tool = update["tool_name"]
    value = update["value"]
    if tool == "carcontrol_airConditioner_set_temperature":
        zone = update["context_arguments"]["zone"]
        return f"{zone}-zone cabin temperature", f"{value} degrees"
    if tool == "carcontrol_music_set_volume":
        return "music volume", str(value)
    if tool == "carcontrol_light_set_ambient_color":
        return "ambient-light color", str(value)
    raise ValueError(f"unsupported Easy-v2 memory Tool: {tool}")


def update_text(name: str, update: dict[str, Any]) -> str:
    label, value = setting(update)
    if update["op"] == "add":
        return (
            f"I'm {name}. Please store exactly one permanent vehicle-profile "
            f"preference for me: set my {label} to {value}."
        )
    return (
        f"I'm {name}. Please replace my previous permanent vehicle-profile value. "
        f"My new {label} is {value}. Use this latest value from now on and do not "
        "keep the old value."
    )


def quiz_text(owner: str, slot_id: str, quiz_id: str) -> str:
    labels = {
        "driver.default.temperature": "driver-zone cabin temperature",
        "codriver.default.music_volume": "music volume",
        "driver.default.ambient_color": "ambient-light color",
        "codriver.default.temperature": "passenger-zone cabin temperature",
    }
    label = labels[slot_id]
    kind, ordinal = quiz_id.rsplit("-", 2)[-2:]
    number = int(ordinal)
    if kind == "turn" and number <= 6:
        return f"Set the {label} to {owner}'s saved vehicle-profile value now."
    if kind == "turn":
        return f"Apply the saved {label} for {owner} now."
    if number % 2:
        return f"Apply {owner}'s latest saved {label} now."
    return f"Set the {label} to the current value in {owner}'s vehicle profile."


def transform(source: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(source)
    result["difficulty_stratum"] = "easy_explicit_v2"
    result["difficulty_contract"].update(
        {
            "single_intent_update_turns": True,
            "human_noops_explicitly_one_time": True,
            "human_noops_explicitly_nonvehicle": True,
            "source_action_competes_with_memory_fact": False,
        }
    )
    result["authorship"]["note"] = (
        "Easy-v2 calibration pilot derived from Easy-v1. External original text is "
        "retained in source_trace, while UPDATEs are single-intent rewrites and "
        "human NO_OPs receive explicit non-vehicle wrappers. Independent human "
        "rewriting and approval remain pending."
    )
    speakers = {speaker["short_id"]: speaker for speaker in result["speakers"]}
    for session in result["sessions"]:
        for turn in session["turns"]:
            trace = turn["source_trace"]
            if "update" in turn:
                name = speakers[turn["speaker"]]["speaker_name"]
                turn["text"] = update_text(name, turn["update"])
                trace["reuse_mode"] = "source_inspired_single_intent_rewrite"
                trace["adaptation_note"] = (
                    "The external record supplies conversational style only; the "
                    "evaluation utterance is an explicit single-intent calibration rewrite."
                )
            elif turn["speaker"] != "assistant":
                original = trace["original_text"].strip()
                turn["text"] = (
                    "This is a one-time request outside the vehicle, not a saved "
                    f"vehicle preference: {original}"
                )
                trace["reuse_mode"] = "explicit_nonvehicle_calibration_wrapper"
                trace["adaptation_note"] = (
                    "A label-transparent domain wrapper is added for the Easy calibration stratum."
                )

    owners = {
        "driver": speakers["user"]["speaker_name"],
        "codriver": speakers["co_user"]["speaker_name"],
    }
    for quiz in result["quizzes"]:
        slot_id = quiz["evidence_slot_ids"][0]
        owner = owners[slot_id.split(".", 1)[0]]
        quiz["query"] = quiz_text(owner, slot_id, quiz["quiz_id"])
    return result


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for number in range(1, args.count + 1):
        scenario_id = f"HVE{number:02d}"
        source_path = source_root / scenario_id / "scenario-source.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        result = transform(source)
        scenario_root = output_root / scenario_id
        scenario_root.mkdir(parents=True, exist_ok=True)
        (scenario_root / "scenario-source.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        attribution = (
            f"# {scenario_id} Easy-v2 provenance\n\n"
            "Derived from the corresponding Easy-v1 pilot. External `original_text` "
            "and record IDs remain available in every turn's `source_trace`. UPDATE "
            "utterances are single-intent calibration rewrites; human NO_OP turns "
            "are explicitly marked as one-time, non-vehicle requests. This remains "
            "model-assembled and ineligible for a human-authored-test claim.\n"
        )
        (scenario_root / "SOURCE_ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
        print(scenario_root / "scenario-source.json")


if __name__ == "__main__":
    main()
