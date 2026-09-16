#!/usr/bin/env python3
"""Classify Patch/Delta quiz disagreements from captured memory snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_SLUGS = {
    "Granite 350M": "granite4-350m",
    "Qwen 0.8B": "qwen3.5-0.8b",
    "Granite 1B": "granite4-1b",
    "Llama 3.2 1B": "llama3.2-1b",
    "Qwen 2B": "qwen3.5-2b",
    "Llama 3.2 3B": "llama3.2-3b",
}
METHOD_DIRS = {"Patch": "patch", "Delta-v3": "delta_v3_compact_k5"}
METHOD_CAUSES = (
    "MISSING_FACT", "STALE_VALUE", "CONFLICTING_FACTS", "WRONG_OWNER",
    "CONDITION_LOSS", "EVIDENCE_IGNORED",
)
QC_CAUSES = ("ORACLE_MEMORY_MISMATCH",)
FACT_RE = re.compile(r"^- \[(?P<timestamp>[^]]+)] (?P<body>.+)$")
PAYLOAD_RE = re.compile(
    r"^(?P<tool>[^.;]+)\.(?P<field>[^;]+); value=(?P<value>[^;]+)"
    r"(?:; context=\((?P<context>[^)]*)\))?"
    r"(?:; condition=(?P<condition>.*))?$"
)

# Human semantic adjudication of the 22 unique samples whose tool arguments
# matched multiple Gold-memory facts. Values identify the request-relevant
# condition; owner is included where equal-valued facts belong to two owners.
MANUAL_GOLD_SELECTORS: dict[str, dict[str, str]] = {
    "s086:final_quiz:quiz-01": {"carcontrol_music_set_volume": "Senator Eleanor Reed|when Eleanor is on a call"},
    "s086:final_quiz:quiz-09": {"carcontrol_music_set_volume": "shared-vehicle|traveling between work discussions"},
    "s086:turn_quiz:turn-quiz-delayed-V09E1-09-at-V09E4": {"carcontrol_music_set_volume": "shared-vehicle|traveling between work discussions"},
    "s086:turn_quiz:turn-quiz-V01E3-003": {"carcontrol_music_set_volume": "Senator Eleanor Reed|when Eleanor is on a call"},
    "s086:turn_quiz:turn-quiz-delayed-V01E3-11-at-V01E4": {"carcontrol_music_set_volume": "Senator Eleanor Reed|when Eleanor is on a call"},
    "s088:final_quiz:quiz-01": {"carcontrol_navigation_set_voice_mode": "Dr. Adrian Cole|hospital commute"},
    "s088:final_quiz:quiz-08": {"carcontrol_navigation_set_voice_mode": "Dr. Leila Hassan|unfamiliar teaching site"},
    "s088:turn_quiz:turn-quiz-V01E3-025": {"carcontrol_navigation_set_voice_mode": "Dr. Adrian Cole|hospital commute"},
    "s088:turn_quiz:turn-quiz-delayed-V01E3-12-at-V01E5": {"carcontrol_navigation_set_voice_mode": "Dr. Adrian Cole|hospital commute"},
    "s088:turn_quiz:turn-quiz-delayed-V08E1-08-at-V08E4": {"carcontrol_navigation_set_voice_mode": "Dr. Leila Hassan|unfamiliar teaching site"},
    "s088:turn_quiz:turn-quiz-composite-01-at-V01E3": {"carcontrol_navigation_set_voice_mode": "Dr. Adrian Cole|hospital commute"},
    "s113:final_quiz:quiz-01": {"carcontrol_music_set_volume": "Margaret Sanger|"},
    "s113:turn_quiz:turn-quiz-composite-02-at-vc09e2": {"carcontrol_music_set_volume": "Margaret Sanger|"},
    "s113:turn_quiz:turn-quiz-vc01e2-012": {"carcontrol_music_set_volume": "Margaret Sanger|"},
    "s117:final_quiz:quiz-01": {"carcontrol_wiper_set_speed": "shared-vehicle|<empty>"},
    "s117:turn_quiz:turn-quiz-delayed-v01-e1-01-at-v01-e4": {"carcontrol_wiper_set_speed": "shared-vehicle|<empty>"},
    "s117:turn_quiz:turn-quiz-v01-e4-013": {"carcontrol_wiper_set_speed": "shared-vehicle|<empty>"},
    "s119:final_quiz:quiz-08": {"carcontrol_music_set_volume": "Dr. Elena Marquez|at present on the audio display"},
    "s119:turn_quiz:turn-quiz-composite-01-at-v08e4": {
        "carcontrol_airConditioner_set_temperature": "Dr. Elena Marquez|at present on the passenger climate panel",
        "carcontrol_music_set_volume": "Dr. Elena Marquez|at present on the audio display",
    },
    "s119:final_quiz:quiz-05": {"carcontrol_seat_set_headrest_height": "Dr. Maya Rahman|at present"},
    "s119:turn_quiz:turn-quiz-v02e4-011": {"carcontrol_airConditioner_set_temperature": "Dr. Elena Marquez|at present on the passenger climate panel"},
    "s120:turn_quiz:turn-quiz-composite-03-at-v10e2": {"carcontrol_light_set_auto_headlight": "shared-vehicle|whenever rain reduces visibility"},
}

# Every original CONDITION_LOSS case was read manually. Semantically equivalent
# rewrites are reader failures; genuine deletions/substitutions remain condition
# loss. All other original CONDITION_LOSS cases are composite MISSING_FACT cases.
SEMANTICALLY_EQUIVALENT_CASES = {
    "Granite 350M::s090:turn_quiz:turn-quiz-composite-02-at-V04E4",
    "Granite 350M::s092:final_quiz:quiz-02",
    "Granite 350M::s092:turn_quiz:turn-quiz-V02E1-007",
    "Granite 350M::s092:turn_quiz:turn-quiz-delayed-V02E1-02-at-V02E4",
    "Granite 1B::s089:turn_quiz:turn-quiz-delayed-veh-07-e01-07-at-veh-07-e02",
    "Granite 1B::s095:final_quiz:quiz-04",
    "Granite 1B::s095:turn_quiz:turn-quiz-delayed-veh04-e2-13-at-veh04-e2",
    "Llama 3.2 1B::s095:final_quiz:quiz-03",
    "Llama 3.2 1B::s095:turn_quiz:turn-quiz-delayed-veh03-e1-03-at-veh03-e2",
    "Llama 3.2 3B::s091:turn_quiz:turn-quiz-delayed-e074-09-at-e076",
    # Both Gold and prediction describe the same current observation; the Gold
    # memory redundantly stores two differently worded observations.
    "Granite 350M::s119:final_quiz:quiz-08",
    "Granite 350M::s119:turn_quiz:turn-quiz-composite-01-at-v08e4",
}
CONFIRMED_CONDITION_LOSS_CASES = {
    "Granite 350M::s090:final_quiz:quiz-05",
    "Granite 350M::s098:turn_quiz:turn-quiz-delayed-v06-e1-04-at-v06-e2",
    "Granite 350M::s098:turn_quiz:turn-quiz-v06-e1-014",
    "Granite 350M::s100:final_quiz:quiz-09",
    "Granite 350M::s100:turn_quiz:turn-quiz-delayed-v09-e1-10-at-v09-e4",
    "Granite 350M::s113:turn_quiz:turn-quiz-delayed-vc02e2-16-at-vc02e2",
    "Granite 350M::s113:turn_quiz:turn-quiz-vc02e2-014",
    "Qwen 0.8B::s091:turn_quiz:turn-quiz-e066-014",
    "Granite 1B::s091:final_quiz:quiz-07",
    "Granite 1B::s091:turn_quiz:turn-quiz-e066-014",
    "Granite 1B::s092:final_quiz:quiz-04",
    "Granite 1B::s092:turn_quiz:turn-quiz-V04E3-013",
    "Granite 1B::s092:turn_quiz:turn-quiz-delayed-V04E3-13-at-V04E4",
    "Llama 3.2 1B::s086:turn_quiz:turn-quiz-composite-02-at-V05E3",
    "Llama 3.2 1B::s087:turn_quiz:turn-quiz-veh04-e01-007",
    "Llama 3.2 1B::s091:turn_quiz:turn-quiz-e066-014",
    "Llama 3.2 1B::s113:turn_quiz:turn-quiz-delayed-vc05e1-05-at-vc05e3",
    "Qwen 2B::s092:turn_quiz:turn-quiz-delayed-V05E1-05-at-V05E4",
    "Qwen 2B::s098:turn_quiz:turn-quiz-v05-e2-011",
    "Llama 3.2 3B::s113:turn_quiz:turn-quiz-delayed-vc01e1-01-at-vc08e2",
}

# The correct evidence is present, but a spurious fact with the same
# request condition and a different value was added under the wrong owner.
# It therefore competes with the answer rather than constituting a pure
# reader-side evidence-utilization failure.
MANUAL_CONFLICT_OVERRIDES = {
    "Granite 350M::s098:turn_quiz:turn-quiz-v03-e2-013",
}


@dataclass(frozen=True)
class Fact:
    owner: str
    timestamp: str
    body: str
    tool: str
    field: str
    value: Any
    context: tuple[tuple[str, Any], ...]
    condition: str

    @property
    def key(self) -> tuple[str, str, tuple[tuple[str, Any], ...]]:
        return self.tool, self.field, self.context


def _json_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text.strip()


def _parse_context(text: str | None) -> tuple[tuple[str, Any], ...]:
    if not text:
        return ()
    pairs = []
    for part in text.split(","):
        key, separator, value = part.strip().partition("=")
        if not separator:
            continue
        pairs.append((key.strip(), _json_value(value.strip())))
    return tuple(sorted(pairs))


def parse_memory(memory: str) -> tuple[Fact, ...]:
    owner = ""
    facts = []
    for raw in memory.splitlines():
        line = raw.strip()
        if line.startswith("### "):
            owner = line[4:].strip()
            continue
        match = FACT_RE.fullmatch(line)
        if not match or not owner:
            continue
        payload = PAYLOAD_RE.fullmatch(match.group("body"))
        if not payload:
            continue
        facts.append(Fact(
            owner=owner,
            timestamp=match.group("timestamp"),
            body=match.group("body"),
            tool=payload.group("tool"),
            field=payload.group("field"),
            value=_json_value(payload.group("value")),
            context=_parse_context(payload.group("context")),
            condition=(payload.group("condition") or "").strip(),
        ))
    return tuple(facts)


def fact_matches_call(fact: Fact, call: Mapping[str, Any]) -> bool:
    if fact.tool != call["name"]:
        return False
    arguments = dict(call.get("arguments", {}))
    if arguments.get(fact.field) != fact.value:
        return False
    context = dict(fact.context)
    return all(arguments.get(key) == value for key, value in context.items())


def gold_evidence(case: Mapping[str, Any]) -> tuple[Fact, ...]:
    facts = parse_memory(str(case["gold_memory"]))
    selected = []
    for call in case["gold_calls"]:
        matches = [fact for fact in facts if fact_matches_call(fact, call)]
        if len(matches) != 1:
            return ()
        selected.append(matches[0])
    return tuple(selected)


def _same_fact(left: Fact, right: Fact) -> bool:
    return (
        left.owner, left.tool, left.field, left.value, left.context, left.condition
    ) == (
        right.owner, right.tool, right.field, right.value, right.context, right.condition
    )


def classify_facts(
    expected: Sequence[Fact], materialized: Sequence[Fact], pending: Sequence[Fact]
) -> tuple[str, str]:
    """Return exclusive primary cause and a compact deterministic rationale."""
    exact = [any(_same_fact(gold, pred) for pred in materialized) for gold in expected]
    conflicts = [
        any(pred.owner == gold.owner and pred.key == gold.key and pred.value != gold.value
            for pred in materialized)
        for gold in expected
    ]
    if all(exact) and any(conflicts):
        return "CONFLICTING_FACTS", "gold evidence and a competing value coexist"
    if all(exact):
        return "EVIDENCE_IGNORED", "all gold evidence is present in reader memory"
    missing = [gold for gold, present in zip(expected, exact) if not present]
    if any(any(_same_fact(gold, pred) for pred in pending) for gold in missing):
        return "PENDING_READ_FAILURE", "missing materialized evidence exists in pending state"
    if any(any(pred.owner != gold.owner and pred.tool == gold.tool and pred.field == gold.field
               and pred.value == gold.value and pred.context == gold.context
               for pred in materialized) for gold in missing):
        return "WRONG_OWNER", "gold value/key is attached to another owner"
    if any(any(pred.owner == gold.owner and pred.key == gold.key and pred.value != gold.value
               for pred in materialized) for gold in missing):
        return "STALE_VALUE", "expected owner/key retains a different value"
    if any(any(pred.owner == gold.owner and pred.tool == gold.tool and pred.field == gold.field
               and pred.value == gold.value and pred.context == gold.context
               and pred.condition != gold.condition for pred in materialized)
           for gold in missing):
        return "CONDITION_LOSS", "gold value/key exists but condition differs"
    return "MISSING_FACT", "no usable gold evidence was found in reader memory"


def _snapshot(root: Path, case: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    slug = MODEL_SLUGS[str(case["model"])]
    scenario = f"s{int(case['scenario_index']):03d}"
    path = root / "memory-snapshots" / slug / METHOD_DIRS[method] / "scenarios" / f"{scenario}.json"
    return json.loads(path.read_text(encoding="utf-8"))["quiz_snapshots"][case["sample_id"]]


def classify_case(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    all_gold = parse_memory(str(case["gold_memory"]))
    per_call = [[fact for fact in all_gold if fact_matches_call(fact, call)] for call in case["gold_calls"]]
    manually_adjudicated = any(not matches or len(matches) > 1 for matches in per_call)
    if any(not matches for matches in per_call):
        cause = "ORACLE_MEMORY_MISMATCH"
        rationale = "manually reviewed: supplied Gold Memory lacks the quiz target value"
        review = False
    elif any(len(matches) > 1 for matches in per_call):
        selectors = MANUAL_GOLD_SELECTORS.get(str(case["sample_id"]), {})
        expected = []
        for call, matches in zip(case["gold_calls"], per_call):
            if len(matches) == 1:
                expected.append(matches[0])
                continue
            selector = selectors.get(str(call["name"]), "")
            owner, separator, condition = selector.partition("|")
            selected = [
                fact for fact in matches
                if fact.owner == owner and (
                    (condition == "<empty>" and not fact.condition)
                    or (condition != "<empty>" and condition in fact.condition)
                )
            ]
            if not separator or len(selected) != 1:
                raise ValueError(f"Unresolved manual Gold selector: {case['case_id']} {call['name']}")
            expected.append(selected[0])
        loser = str(case["loser"])
        snapshot = _snapshot(root, case, loser)
        cause, rationale = classify_facts(
            tuple(expected),
            parse_memory(str(snapshot.get("materialized_memory", ""))),
            (),
        )
        rationale = "manually selected request-relevant Gold fact; " + rationale
        if str(case["case_id"]) in SEMANTICALLY_EQUIVALENT_CASES and cause == "CONDITION_LOSS":
            cause = "EVIDENCE_IGNORED"
            rationale = "manually reviewed: alternate current-observation wording is semantically equivalent"
        review = False
    else:
        expected = tuple(matches[0] for matches in per_call)
        loser = str(case["loser"])
        snapshot = _snapshot(root, case, loser)
        materialized = parse_memory(str(snapshot.get("materialized_memory", "")))
        pending_text = "\n".join(str(item) for item in snapshot.get("pending_updates", []))
        pending = parse_memory(pending_text)
        cause, rationale = classify_facts(expected, materialized, pending)
        review = False
        if cause == "CONDITION_LOSS":
            manually_adjudicated = True
            case_id = str(case["case_id"])
            if case_id in SEMANTICALLY_EQUIVALENT_CASES:
                cause = "EVIDENCE_IGNORED"
                rationale = "manually reviewed: condition is semantically equivalent"
            elif case_id in CONFIRMED_CONDITION_LOSS_CASES:
                rationale = "manually reviewed: applicability condition is deleted or materially changed"
            else:
                cause = "MISSING_FACT"
                rationale = "manually reviewed composite request: another required fact is absent"
    if str(case["case_id"]) in MANUAL_CONFLICT_OVERRIDES:
        cause = "CONFLICTING_FACTS"
        rationale = (
            "manually reviewed: a spurious wrong-owner value competes with "
            "the correct answer-bearing fact"
        )
        manually_adjudicated = True
    result = dict(case)
    result["coding"] = dict(case["coding"])
    result["coding"].update({
        "primary_cause": cause,
        "primary_cause_status": "MANUALLY_ADJUDICATED" if manually_adjudicated else "AUTO_CODED",
        "review_required": review,
        "review_notes": rationale,
    })
    return result


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def summarize(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_model_direction: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for case in cases:
        by_model_direction[str(case["model"])][str(case["direction"])][str(case["coding"]["primary_cause"])] += 1
    main = {}
    for direction in ("PATCH_ONLY_CORRECT", "DELTA_ONLY_CORRECT"):
        rows = []
        for cause in METHOD_CAUSES:
            per_model = []
            total_n = total_den = 0
            for model in MODEL_SLUGS:
                counts = by_model_direction[model][direction]
                denominator = sum(counts[c] for c in METHOD_CAUSES)
                per_model.append(_pct(counts[cause], denominator))
                total_n += counts[cause]; total_den += denominator
            rows.append({
                "cause": cause,
                "count": total_n,
                "pooled_percent": _pct(total_n, total_den),
                "macro_percent": sum(per_model) / len(per_model),
            })
        main[direction] = rows
    return {
        "schema_version": "palmclaw-patch-delta-causal-coding-v1",
        "cases": len(cases),
        "method_causes": list(METHOD_CAUSES),
        "quality_control_causes": list(QC_CAUSES),
        "overall_counts": dict(Counter(str(c["coding"]["primary_cause"]) for c in cases)),
        "review_required": sum(bool(c["coding"]["review_required"]) for c in cases),
        "main_table": main,
        "appendix_by_model_direction": {
            model: {direction: dict(counts) for direction, counts in directions.items()}
            for model, directions in by_model_direction.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    cases = [json.loads(line) for line in (root / "disagreement-cases.jsonl").read_text(encoding="utf-8").splitlines()]
    coded = [classify_case(root, case) for case in cases]
    with (root / "coded-cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in coded:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    summary = summarize(coded)
    (root / "causal-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ("case_id", "model", "sample_id", "direction", "reasoning_type", "primary_cause", "review_required", "review_notes")
    with (root / "coded-cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for case in coded:
            coding = case["coding"]
            writer.writerow({**{key: case[key] for key in fields[:5]}, **{key: coding[key] for key in fields[5:]}})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
