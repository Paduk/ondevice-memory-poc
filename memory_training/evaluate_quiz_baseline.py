"""Resumable evaluation of fixed Quiz-only readers under memory baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_WORKSPACE_ROOT, MODEL_BY_KEY
from .quiz_sft import IndexedQuizSFTDataset, VehicleToolSchemaStore
from .validation import evaluate_quiz_tool_calling, quiz_indices_for_scenarios


TEST_SCENARIOS = tuple(range(86, 101)) + tuple(range(112, 121))
HVP_SCENARIOS = tuple(range(901, 941))
PROFILES = (
    "no_memory",
    "recent_window",
    "kv_lww",
    "gold_memory",
    "cloud_patch",
)
_WORD = re.compile(r"[a-z0-9]+")
_TOPIC_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("navigation_voice", ("navigation voice", "voice guidance", "spoken directions")),
    ("navigation_map", ("map view", "map display", "north up", "heading up")),
    ("navigation_traffic", ("traffic display", "show traffic", "traffic visible")),
    ("seat_heating", ("seat heat", "seat heating", "heated seat")),
    ("seat_ventilation", ("seat ventilation", "ventilated seat", "seat cooling")),
    ("reading_light", ("reading light", "reading lamp")),
    ("fog_light", ("fog light", "fog lamp")),
    ("low_beam", ("low beam", "headlight")),
    ("air_temperature", ("temperature", "degrees", "cabin warmer", "cabin cooler")),
    ("air_circulation", ("circulation", "outside air", "fresh air", "recircul")),
    ("air_fan", ("fan speed", "airflow")),
    ("air_mode", ("dehumid", "defrost", "air conditioning mode", "climate mode")),
    ("music_volume", ("music volume", "music quiet", "music louder", "music setting")),
    ("radio_volume", ("radio volume", "radio louder", "radio quiet")),
    ("music_playback", ("music off", "music on", "stop the music", "play music")),
    ("window_open", ("window", "windows")),
    ("door_lock", ("door lock", "doors locked", "lock the doors")),
    ("door_warning", ("door warning", "open warning")),
    ("wiper", ("wiper", "windscreen")),
    ("hud", ("head-up", "head up", "hud")),
    ("display", ("display brightness", "center display", "information display")),
    ("bluetooth", ("bluetooth", "phone connection", "connect my phone")),
    ("mirror", ("mirror", "reverse tilt")),
    ("sunroof", ("sunroof", "moonroof")),
    ("steering_wheel", ("steering wheel",)),
)
_TOPIC_FIELD = {
    "navigation_voice": "carcontrol_navigation_set_voice_mode.mode",
    "navigation_map": "carcontrol_navigation_set_map_view.view",
    "navigation_traffic": "carcontrol_navigation_set_traffic_display.enabled",
    "seat_heating": "carcontrol_seat_set_heating_level.level",
    "seat_ventilation": "carcontrol_seat_set_ventilation_speed.speed",
    "reading_light": "carcontrol_light_set_reading_light_brightness.brightness",
    "fog_light": "carcontrol_light_set_fog_light.enabled",
    "low_beam": "carcontrol_light_set_low_beam_level.level",
    "air_temperature": "carcontrol_airConditioner_set_temperature.temperature",
    "air_circulation": "carcontrol_airConditioner_set_circulation.circulation",
    "air_fan": "carcontrol_airConditioner_set_fan_speed.speed",
    "air_mode": "carcontrol_airConditioner_set_mode.mode",
    "music_volume": "carcontrol_music_set_volume.volume",
    "radio_volume": "carcontrol_radio_set_volume.volume",
    "music_playback": "carcontrol_music_switch.switch",
    "window_open": "carcontrol_window_set_open_degree.degree",
    "door_lock": "carcontrol_door_set_locked.locked",
    "door_warning": "carcontrol_door_set_open_warning.enabled",
    "wiper": "carcontrol_wiper_set_speed.speed",
    "hud": "carcontrol_HUD_set_brightness_level.level",
    "display": "carcontrol_centerInformationDisplay_set_brightness_level.level",
    "bluetooth": "carcontrol_bluetooth_set_connection.connected",
    "mirror": "carcontrol_rearviewMirror_set_auto_reverse_tilt.enabled",
    "sunroof": "carcontrol_sunroof_set_open_degree.degree",
    "steering_wheel": "carcontrol_steeringWheel_set_heating_level.level",
}
_ENUM_VALUES = (
    "north_up",
    "heading_up",
    "three_d",
    "mute",
    "simple",
    "detailed",
    "outside",
    "inside",
    "auto",
    "dehumidify",
    "defrost",
    "cool",
    "heat",
    "high",
    "medium",
    "low",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument(
        "--vehiclemembench-root",
        type=Path,
        default=Path("/home/hj153lee/VehicleMemBench"),
    )
    parser.add_argument("--scenarios", nargs="+", type=int, default=TEST_SCENARIOS)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--quiz-batch-size", type=int, default=16)
    parser.add_argument(
        "--recent-window-tokens",
        type=int,
        default=768,
        help="Tokenizer-specific history budget; identical numeric budget across models.",
    )
    parser.add_argument("--kv-retrieval-tokens", type=int, default=768)
    parser.add_argument(
        "--memory-steps-root",
        type=Path,
        help=(
            "Root containing sNNN/memory_steps/NNNNN.json Cloud Patch states; "
            "required by the cloud_patch profile."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = tuple(dict.fromkeys(args.scenarios))
    selected = set(scenarios)
    if not scenarios or not any(
        selected.issubset(allowed)
        for allowed in (set(TEST_SCENARIOS), set(HVP_SCENARIOS))
    ):
        raise ValueError(
            "Scenarios must come from either the fixed 24 Test scenarios "
            "or evaluation-only HVP/HVE aliases S901-S940"
        )
    if args.quiz_batch_size < 1:
        raise ValueError("quiz-batch-size must be positive")
    if args.profile == "cloud_patch" and args.memory_steps_root is None:
        raise ValueError("cloud_patch requires --memory-steps-root")

    checkpoint = args.checkpoint.resolve()
    adapter = checkpoint / "adapter"
    if not adapter.is_dir():
        raise FileNotFoundError(adapter)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    scenario_dir = output_dir / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    _configure_runtime_paths(args.workspace.resolve())

    source = IndexedQuizSFTDataset(data_root / "quiz_sft.jsonl", split="test")
    tools = VehicleToolSchemaStore(data_root / "vehicle_tools.json")
    quiz_counts = {
        scenario: len(quiz_indices_for_scenarios(source, [scenario]))
        for scenario in scenarios
    }
    missing = [scenario for scenario, count in quiz_counts.items() if count == 0]
    if missing:
        raise ValueError(f"Quiz data is missing for scenarios: {missing}")
    signature = _signature(args, checkpoint, scenarios, tools.sha256)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature:
            if any(scenario_dir.glob("s*.json")):
                raise ValueError(f"Stale output directory: {output_dir}")
            manifest_path.unlink()
            manifest = {}
    if not manifest_path.is_file():
        _write_json(
            manifest_path,
            {
                "schema_version": "quiz-only-memory-baseline-eval-v1",
                "signature": signature,
                "model": args.model,
                "checkpoint": str(checkpoint),
                "profile": args.profile,
                "scenarios": list(scenarios),
                "quiz_rows_per_scenario": quiz_counts,
                "recent_window_tokens": (
                    args.recent_window_tokens
                    if args.profile == "recent_window"
                    else None
                ),
                "kv_variant": (
                    "raw-dialogue deterministic topic-key LWW with lexical retrieval"
                    if args.profile == "kv_lww"
                    else None
                ),
                "gold_patch_labels_used": args.profile == "gold_memory",
                "gold_memory_evaluation": args.profile == "gold_memory",
                "memory_steps_root": (
                    str(args.memory_steps_root.resolve())
                    if args.memory_steps_root is not None
                    else None
                ),
                "created_at": _utc_now(),
            },
        )

    from accelerate import Accelerator

    from .models import load_peft_bundle

    accelerator = Accelerator()
    bundle = load_peft_bundle(
        args.model,
        adapter_path=adapter,
        gradient_checkpointing=False,
        cache_dir=args.workspace.resolve() / "cache" / "huggingface" / "hub",
    )
    model = accelerator.prepare(bundle.model)
    model.eval()

    turn_rows = (
        _load_turns(data_root / "patch.jsonl", scenarios)
        if args.profile in {"recent_window", "kv_lww"}
        else {}
    )
    completed: dict[int, dict[str, Any]] = {}
    for position, scenario in enumerate(scenarios, start=1):
        path = scenario_dir / f"s{scenario:03d}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != signature:
                raise ValueError(f"Stale scenario result: {path}")
            completed[scenario] = payload
            print(f"[{position}/{len(scenarios)}] S{scenario} restored", flush=True)
            continue

        indices = quiz_indices_for_scenarios(source, [scenario])
        if scenario in TEST_SCENARIOS and len(indices) != 40:
            raise ValueError(f"S{scenario} has {len(indices)} Quiz rows, expected 40")
        if args.profile == "gold_memory":
            overrides = None
        elif args.profile == "cloud_patch":
            assert args.memory_steps_root is not None
            overrides = _cloud_patch_overrides(
                source,
                indices,
                args.memory_steps_root.resolve(),
            )
        else:
            overrides = _memory_overrides(
                args.profile,
                source,
                indices,
                turn_rows,
                tokenizer=bundle.tokenizer,
                recent_window_tokens=args.recent_window_tokens,
                kv_retrieval_tokens=args.kv_retrieval_tokens,
            )
        result = evaluate_quiz_tool_calling(
            model,
            bundle.tokenizer,
            source,
            tools,
            indices,
            dataset_root=args.vehiclemembench_root.resolve(),
            accelerator=accelerator,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            memory_overrides=overrides,
            batch_size=args.quiz_batch_size,
        )
        for record in result["records"]:
            record["memory_mode"] = args.profile
        payload = {
            "schema_version": "quiz-only-memory-baseline-scenario-v1",
            "signature": signature,
            "scenario_index": scenario,
            "profile": args.profile,
            "metrics": {key: value for key, value in result.items() if key != "records"},
            "records": result["records"],
            "completed_at": _utc_now(),
        }
        _write_json(path, payload)
        completed[scenario] = payload
        _write_progress(
            output_dir,
            scenarios,
            completed,
            quiz_counts,
            status="RUNNING",
        )
        print(
            f"[{position}/{len(scenarios)}] S{scenario} complete "
            f"ESM={result['esm']:.4f} F1={result['tool_f1']:.4f}",
            flush=True,
        )

    records = [
        record
        for scenario in scenarios
        for record in completed[scenario]["records"]
    ]
    summary = _aggregate(records)
    summary.update(
        {
            "schema_version": "quiz-only-memory-baseline-summary-v1",
            "signature": signature,
            "model": args.model,
            "checkpoint": str(checkpoint),
            "profile": args.profile,
            "scenarios": list(scenarios),
            "completed_at": _utc_now(),
        }
    )
    _write_json(output_dir / "summary.json", summary)
    _write_progress(
        output_dir,
        scenarios,
        completed,
        quiz_counts,
        status="COMPLETE",
    )
    print(json.dumps({key: summary[key] for key in ("tasks", "esm", "tool_f1", "arg_exact")}), flush=True)
    return summary


def _memory_overrides(
    profile: str,
    source: IndexedQuizSFTDataset,
    indices: Sequence[int],
    turns: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    tokenizer: Any,
    recent_window_tokens: int,
    kv_retrieval_tokens: int,
) -> dict[str, str]:
    if profile == "kv_lww":
        return _kv_lww_overrides(
            source,
            indices,
            turns,
            tokenizer=tokenizer,
            budget=kv_retrieval_tokens,
        )
    result: dict[str, str] = {}
    for index in indices:
        row = source[index]
        sample_id = str(row["sample_id"])
        if profile == "no_memory":
            result[sample_id] = ""
            continue
        scenario = int(row["scenario_index"])
        cutoff = int(row["memory_ref"]["global_turn_index"])
        result[sample_id] = _recent_window(
            turns[scenario], cutoff, tokenizer, recent_window_tokens
        )
    return result


def _cloud_patch_overrides(
    source: IndexedQuizSFTDataset,
    indices: Sequence[int],
    memory_steps_root: Path,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for index in indices:
        row = source[index]
        scenario = int(row["scenario_index"])
        cutoff = int(row["memory_ref"]["global_turn_index"])
        path = (
            memory_steps_root
            / f"s{scenario:03d}"
            / "memory_steps"
            / f"{cutoff:05d}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Cloud Patch memory step is missing for {row['sample_id']}: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        state_after = payload.get("state_after")
        if not isinstance(state_after, Mapping) or "previous_memory" not in state_after:
            raise ValueError(f"Invalid Cloud Patch state in {path}")
        result[str(row["sample_id"])] = str(state_after["previous_memory"])
    return result


def _load_turns(
    path: Path, scenarios: Sequence[int]
) -> dict[int, list[dict[str, Any]]]:
    selected = set(scenarios)
    by_scenario: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scenario = int(row["scenario_index"])
            if scenario in selected:
                by_scenario[scenario].append(
                    {
                        "global_turn_index": int(row["global_turn_index"]),
                        "timestamp": str(row["timestamp"]),
                        "event_id": str(row["event_id"]),
                        "current_turn": dict(row["current_turn"]),
                    }
                )
    for scenario, rows in by_scenario.items():
        rows.sort(key=lambda row: int(row["global_turn_index"]))
        observed = [int(row["global_turn_index"]) for row in rows]
        if observed != list(range(len(rows))):
            raise ValueError(f"S{scenario} turn indices are not contiguous")
    return dict(by_scenario)


def _recent_window(
    turns: Sequence[Mapping[str, Any]], cutoff: int, tokenizer: Any, budget: int
) -> str:
    if budget < 1:
        raise ValueError("recent-window-tokens must be positive")
    selected: list[str] = []
    used = 0
    for row in reversed(turns[: cutoff + 1]):
        turn = row["current_turn"]
        rendered = (
            f"- [{row['timestamp']}] {turn['speaker_name']} said: {turn['text']}"
        )
        count = len(tokenizer.encode(rendered, add_special_tokens=False))
        if selected and used + count > budget:
            break
        if not selected and count > budget:
            ids = tokenizer.encode(rendered, add_special_tokens=False)[-budget:]
            rendered = tokenizer.decode(ids, skip_special_tokens=True)
            count = len(ids)
        selected.append(rendered)
        used += count
    return "### Recent dialogue\n" + "\n".join(reversed(selected))


def _kv_lww_overrides(
    source: IndexedQuizSFTDataset,
    indices: Sequence[int],
    turns: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    tokenizer: Any,
    budget: int,
) -> dict[str, str]:
    """Build a label-free, deliberately simple deterministic KV/LWW baseline."""
    if budget < 1:
        raise ValueError("kv-retrieval-tokens must be positive")
    result: dict[str, str] = {}
    for index in indices:
        row = source[index]
        scenario = int(row["scenario_index"])
        cutoff = int(row["memory_ref"]["global_turn_index"])
        request = str(row["messages"][1]["content"]).split(
            "\n\n[Current request]\n", 1
        )[1]
        store = _build_kv_lww(turns[scenario], cutoff)
        result[str(row["sample_id"])] = _retrieve_kv(
            store, request, tokenizer=tokenizer, budget=budget
        )
    return result


def _build_kv_lww(
    turns: Sequence[Mapping[str, Any]], cutoff: int
) -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    for _, _, current in _replay_kv_lww(turns, cutoff):
        store = current
    return store


def _replay_kv_lww(
    turns: Sequence[Mapping[str, Any]], cutoff: int | None = None
):
    """Yield each turn, its semantic UPDATE decision, and the evolving KV store."""
    store: dict[str, dict[str, Any]] = {}
    previous_event = ""
    previous_topic: str | None = None
    event_context: list[str] = []
    stop = len(turns) if cutoff is None else cutoff + 1
    for row in turns[:stop]:
        event_id = str(row["event_id"])
        if event_id != previous_event:
            previous_event = event_id
            previous_topic = None
            event_context = []
        current = row["current_turn"]
        text = str(current["text"])
        topic = _detect_topic(text) or previous_topic
        rendered = f"{current['speaker_name']}: {text}"
        event_context.append(rendered)
        event_context = event_context[-3:]
        if topic is None:
            yield row, False, store
            continue
        previous_topic = topic
        evidence = " | ".join(event_context)
        value = _extract_value(topic, evidence)
        if value is None:
            yield row, False, store
            continue
        speaker = _slug(str(current["speaker_name"]))
        key = f"{speaker}::{topic}"
        record = {
            "speaker": str(current["speaker_name"]),
            "topic": topic,
            "field": _field_for(topic, evidence),
            "timestamp": str(row["timestamp"]),
            "value": value,
            "evidence": evidence,
        }
        previous = store.get(key)
        changed = previous is None or (
            previous["field"], previous["value"]
        ) != (record["field"], record["value"])
        store[key] = record
        yield row, changed, store


def _retrieve_kv(
    store: Mapping[str, Mapping[str, Any]],
    request: str,
    *,
    tokenizer: Any,
    budget: int,
) -> str:
    query_topics = set(_detect_topics(request))
    query_words = set(_WORD.findall(request.lower()))
    ranked: list[tuple[int, str, Mapping[str, Any]]] = []
    for key, record in store.items():
        topic = str(record["topic"])
        words = set(_WORD.findall((key + " " + record["evidence"]).lower()))
        score = 100 * int(topic in query_topics) + len(query_words & words)
        if score > 0:
            ranked.append((score, key, record))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[tuple[str, str]] = []
    used = 0
    for _, _, record in ranked:
        rendered = _render_kv_record(record)
        count = len(tokenizer.encode(rendered, add_special_tokens=False))
        if selected and used + count > budget:
            continue
        if not selected and count > budget:
            ids = tokenizer.encode(rendered, add_special_tokens=False)[-budget:]
            rendered = tokenizer.decode(ids, skip_special_tokens=True)
            count = len(ids)
        selected.append((str(record["speaker"]), rendered))
        used += count
    grouped: dict[str, list[str]] = defaultdict(list)
    for speaker, rendered in selected:
        grouped[speaker].append(rendered)
    sections = [
        f"### {speaker}\n" + "\n".join(rows)
        for speaker, rows in grouped.items()
    ]
    return "\n\n".join(sections)


def _render_kv_record(record: Mapping[str, Any]) -> str:
    evidence = str(record["evidence"])
    value = record["value"]
    parts = [
        f"- [{record['timestamp']}] {record['field']}",
        f"value={json.dumps(value, ensure_ascii=False) if value is not None else 'unspecified'}",
        f"condition=derived from recent dialogue: {evidence}",
    ]
    return "; ".join(parts)


def _materialize_kv_store(store: Mapping[str, Mapping[str, Any]]) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in store.values():
        grouped[str(record["speaker"])].append(_render_kv_record(record))
    return "\n\n".join(
        f"### {speaker}\n" + "\n".join(rows)
        for speaker, rows in grouped.items()
    )


def _extract_value(topic: str, evidence: str) -> int | str | bool | None:
    lowered = evidence.lower().replace("north-up", "north_up").replace(
        "heading-up", "heading_up"
    ).replace("3d", "three_d")
    numeric_topics = {
        "seat_heating",
        "seat_ventilation",
        "reading_light",
        "low_beam",
        "air_temperature",
        "air_fan",
        "music_volume",
        "radio_volume",
        "window_open",
        "wiper",
        "hud",
        "display",
        "sunroof",
        "steering_wheel",
    }
    if topic in numeric_topics:
        values = re.findall(r"(?<![\w-])-?\d+(?:\.\d+)?(?![\w-])", evidence)
        if values:
            number = float(values[-1])
            return int(number) if number.is_integer() else str(number)
        if topic in {"window_open", "sunroof"} and any(
            cue in lowered for cue in ("close", "closed", "shut")
        ):
            return 0
    for value in _ENUM_VALUES:
        if re.search(rf"\b{re.escape(value)}\b", lowered):
            return value
    if topic == "door_lock":
        if "unlock" in lowered:
            return False
        if "lock" in lowered:
            return True
    false_cues = ("turn off", "switch off", "disable", "disabled", "disconnect")
    true_cues = ("turn on", "switch on", "enable", "enabled", "connect")
    if any(cue in lowered for cue in false_cues):
        return False
    if any(cue in lowered for cue in true_cues):
        return True
    return None


def _field_for(topic: str, evidence: str) -> str:
    lowered = evidence.lower()
    if topic == "reading_light" and not any(
        cue in lowered for cue in ("brightness", "level")
    ) and not re.search(r"\b\d+\b", lowered):
        return "carcontrol_light_set_reading_light.enabled"
    if topic == "hud":
        if "height" in lowered:
            return "carcontrol_HUD_set_height_level.level"
        if any(cue in lowered for cue in ("turn on", "turn off", "switch")):
            return "carcontrol_HUD_switch.switch"
    if topic == "display" and "auto" in lowered:
        return "carcontrol_centerInformationDisplay_set_auto_brightness.enabled"
    if topic == "window_open" and "auto close" in lowered:
        return "carcontrol_window_set_auto_close_on_lock.enabled"
    return _TOPIC_FIELD[topic]


def _detect_topic(text: str) -> str | None:
    topics = _detect_topics(text)
    return topics[0] if topics else None


def _detect_topics(text: str) -> list[str]:
    lowered = text.lower()
    topics = [
        topic
        for topic, aliases in _TOPIC_ALIASES
        if any(alias in lowered for alias in aliases)
    ]
    if "music" in lowered:
        playback_cues = ("turn off", "turn on", "switch off", "switch on", "stop", "play")
        inferred = (
            "music_playback"
            if any(cue in lowered for cue in playback_cues)
            else "music_volume"
        )
        topics.append(inferred)
    if "radio" in lowered:
        topics.append("radio_volume")
    return list(dict.fromkeys(topics))


def _slug(value: str) -> str:
    return "_".join(_WORD.findall(value.lower())) or "unknown"


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        mean = lambda key: sum(float(row[key]) for row in rows) / max(count, 1)
        return {
            "tasks": count,
            "esm": mean("exact_state_match"),
            "tool_f1": mean("tool_f1"),
            "arg_exact": mean("arg_exact"),
            "parse_success_rate": mean("parse_success"),
            "execution_success_rate": mean("execution_success"),
            "prefill_tokens": sum(int(row["prefill_tokens"]) for row in rows),
            "decode_tokens": sum(int(row["decode_tokens"]) for row in rows),
            "latency_seconds": sum(float(row["latency_seconds"]) for row in rows),
        }

    result = metrics(records)
    result["by_quiz_type"] = {
        key: metrics([row for row in records if row["quiz_type"] == key])
        for key in sorted({str(row["quiz_type"]) for row in records})
    }
    result["by_reasoning_type"] = {
        key: metrics([row for row in records if row["reasoning_type"] == key])
        for key in sorted({str(row["reasoning_type"]) for row in records})
    }
    return result


def _write_progress(
    output_dir: Path,
    scenarios: Sequence[int],
    completed: Mapping[int, Mapping[str, Any]],
    quiz_counts: Mapping[int, int],
    *,
    status: str,
) -> None:
    done = [scenario for scenario in scenarios if scenario in completed]
    _write_json(
        output_dir / "progress.json",
        {
            "status": status,
            "completed_scenarios": done,
            "total_scenarios": len(scenarios),
            "completed_quizzes": sum(
                int(completed[scenario]["metrics"]["tasks"]) for scenario in done
            ),
            "total_quizzes": sum(quiz_counts.values()),
            "updated_at": _utc_now(),
        },
    )


def _signature(
    args: argparse.Namespace,
    checkpoint: Path,
    scenarios: Sequence[int],
    tools_sha256: str,
) -> str:
    payload = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "profile": args.profile,
        "scenarios": list(scenarios),
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "recent_window_tokens": args.recent_window_tokens,
        "kv_retrieval_tokens": args.kv_retrieval_tokens,
        "memory_steps_root": (
            str(args.memory_steps_root.resolve())
            if args.memory_steps_root is not None
            else None
        ),
        "tools_sha256": tools_sha256,
        "vehiclemembench_root": str(args.vehiclemembench_root.resolve()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _configure_runtime_paths(workspace: Path) -> None:
    paths = {
        "XDG_CACHE_HOME": workspace / "cache",
        "HF_HOME": workspace / "cache" / "huggingface",
        "HF_HUB_CACHE": workspace / "cache" / "huggingface" / "hub",
        "HF_XET_CACHE": workspace / "cache" / "huggingface" / "xet",
        "HF_DATASETS_CACHE": workspace / "cache" / "huggingface" / "datasets",
        "TORCH_HOME": workspace / "cache" / "torch",
        "TMPDIR": workspace / "tmp",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
