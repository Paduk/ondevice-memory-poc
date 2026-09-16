"""Lower HVE11-HVE20 difficulty while preserving natural explicit language.

HVE01-HVE10 remain text-identical to easy-natural-explicit-v1.  In the newly
added half, UPDATE turns are reduced to the already-reviewed durable preference
only, and the first human turn of each NO_OP session gets a natural home/one-off
frame.  Gold states, update labels, Quiz cutoffs, and Tool calls never change.
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
        default=Path("evaluation/human-authored-vehicle-memory/easy-natural-explicit-v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation/human-authored-vehicle-memory/easy-natural-balanced-v1"),
    )
    parser.add_argument("--count", type=int, default=20)
    return parser.parse_args()


def natural_noop_frame(text: str, dataset: str) -> str:
    lower = text.lower()
    if dataset == "Audio2Tool":
        return f"For the living-room lights at home, just for this evening: {text}"
    if any(word in lower for word in ("sleep", "asleep", "bed", "relaxing", "calming")):
        return f"I'm getting ready for bed at home tonight. {text}"
    if any(word in lower for word in ("film", "cinema", "movie")):
        return f"I'm at home planning an outing for tonight. {text}"
    return f"I'm at home, and this is only for today. {text}"


def transform(source: dict[str, Any], scenario_number: int) -> dict[str, Any]:
    result = deepcopy(source)
    result["difficulty_stratum"] = "easy_natural_balanced_v1"
    result["authorship"]["note"] = (
        "Natural-explicit Easy calibration pilot. HVE11-HVE20 use single-intent "
        "durable updates and naturally framed one-off home requests. Rewrites and "
        "labels remain model-assisted pending independent human approval."
    )
    if scenario_number <= 10:
        return result

    for session in result["sessions"]:
        update_turns = [turn for turn in session["turns"] if "update" in turn]
        if update_turns:
            if len(update_turns) != 1 or len(session["turns"]) != 1:
                raise ValueError(f"unexpected UPDATE session structure: {session['session_id']}")
            turn = update_turns[0]
            original = str(turn["source_trace"]["original_text"]).strip()
            if not turn["text"].startswith(original):
                raise ValueError(f"source prefix mismatch: {turn['turn_id']}")
            natural_clause = turn["text"][len(original):].lstrip(" .")
            if not natural_clause:
                raise ValueError(f"missing natural preference clause: {turn['turn_id']}")
            turn["text"] = natural_clause
            turn["source_trace"]["reuse_mode"] = "source_inspired_single_intent_rewrite"
            session["context"] = (
                "One owner naturally states exactly one explicit, durable vehicle preference."
            )
            continue

        first_human = next(
            (turn for turn in session["turns"] if turn["speaker"] != "assistant"),
            None,
        )
        if first_human is None:
            raise ValueError(f"NO_OP session has no human turn: {session['session_id']}")
        original = str(first_human["source_trace"]["original_text"]).strip()
        first_human["text"] = natural_noop_frame(
            original, str(first_human["source_trace"]["dataset"])
        )
        first_human["source_trace"]["reuse_mode"] = (
            "explicit_nonvehicle_calibration_wrapper"
        )
        session["context"] = (
            "A naturally framed one-off home request; no durable vehicle preference is expressed."
        )
    return result


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for number in range(1, args.count + 1):
        scenario_id = f"HVE{number:02d}"
        source = json.loads(
            (source_root / scenario_id / "scenario-source.json").read_text(encoding="utf-8")
        )
        result = transform(source, number)
        scenario_root = output_root / scenario_id
        scenario_root.mkdir(parents=True, exist_ok=True)
        (scenario_root / "scenario-source.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (scenario_root / "SOURCE_ATTRIBUTION.md").write_text(
            f"# {scenario_id} source attribution\n\n"
            "This balanced Easy variant preserves the record-level sources and Gold "
            "semantics of `easy-natural-explicit-v1`. HVE11-HVE20 remove unrelated "
            "one-time vehicle commands from UPDATE text and naturally frame the first "
            "human turn of each home NO_OP session. It remains model-assisted and "
            "ineligible for a human-authored claim pending independent approval.\n",
            encoding="utf-8",
        )
        print(scenario_root / "scenario-source.json")


if __name__ == "__main__":
    main()
