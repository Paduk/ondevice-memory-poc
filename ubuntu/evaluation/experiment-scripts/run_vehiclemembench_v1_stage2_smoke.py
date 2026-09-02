#!/usr/bin/env python3
"""Run one resumable Terra Persona/Event canary for V1 reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.dataset import (
    OFFICIAL_UPSTREAM_COMMIT,
    load_vehicle_benchmark,
)
from palmclaw_ubuntu.vehicle_bench.scoring import VehicleWorldRuntime
from palmclaw_ubuntu.vehicle_bench.v1_generation import (
    V1_DEFAULT_GENERATION_MODEL,
    V1_EVENT_CHAIN_STATE_EVOLUTION_PROMPT_VERSION,
    OpenAIV1EventChainGenerationModel,
    OpenAIV1PersonaGenerationModel,
    V1GeneratedEventChains,
    V1GeneratedPersonaGroup,
    V1Stage2Artifact,
    build_persona_seed_groups,
    build_reasoning_type_plan,
    build_vehicle_attribute_catalog,
    generate_stage2_artifact,
    load_persona_seeds,
    validate_stage2_contract,
    validate_stage2_simulator_arguments,
    validate_state_evolution_coverage,
    write_stage2_artifact,
)

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_PERSONA_SEEDS = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "sources/personahub-elite-sample-v1/persona_seeds.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "stage2-smoke-terra-s1-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--persona-seeds", type=Path, default=DEFAULT_PERSONA_SEEDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--persona-checkpoint",
        type=Path,
        help="Reuse a previously validated Persona checkpoint.",
    )
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--candidate-group", type=int, default=1)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--persona-only", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--style-blueprint",
        type=Path,
        help=(
            "Optional source-selection.json. Uses its pilot reasoning order and "
            "Tool family without exposing source dialogue, people, or values."
        ),
    )
    parser.add_argument(
        "--state-evolution",
        action="store_true",
        help="Require the project-defined S21+ ADD/REPLACE/DELETE coverage.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.candidate_group <= 200:
        raise ValueError("--candidate-group must be in [1, 200]")
    if not 1 <= args.scenario <= 100:
        raise ValueError("--scenario must be in [1, 100]")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    seeds = load_persona_seeds(args.persona_seeds)
    seed_groups = build_persona_seed_groups(seeds, candidate_group_count=200)
    selected_seeds = seed_groups[args.candidate_group - 1]
    style_blueprint = None
    if args.style_blueprint is not None:
        selection_path = args.style_blueprint.expanduser().resolve(strict=True)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        style_blueprint = selection["pilot_blueprint"]
        reasoning_types = tuple(style_blueprint["reasoning_types_in_quiz_order"])
    else:
        selection_path = None
        reasoning_types = build_reasoning_type_plan(100)[args.scenario - 1]
    catalog = build_vehicle_attribute_catalog(dataset.tool_schemas)
    if style_blueprint is not None:
        allowed_tools = set(style_blueprint["tool_counts"])
        catalog = tuple(item for item in catalog if item.tool_name in allowed_tools)
        if not catalog:
            raise ValueError("style blueprint Tool allowlist produced an empty catalog")
    runtime = VehicleWorldRuntime(dataset.root)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if style_blueprint is not None:
        _write_json(
            output_root / "style-provenance.json",
            {
                "schema_version": "vehiclemembench-v1-style-provenance-v1",
                "selection_path": str(selection_path),
                "source_scenario": int(style_blueprint["scenario"]),
                "source_history_sha256": style_blueprint["history_sha256"],
                "source_qa_sha256": style_blueprint["qa_sha256"],
                "copied_source_content": False,
                "transferred_structure": {
                    "reasoning_types_in_quiz_order": list(reasoning_types),
                    "allowed_tool_names": sorted(allowed_tools),
                    "target_turns": int(style_blueprint["turns"]),
                    "reference_update_count": int(style_blueprint["cloud_trace_updates"]),
                    "reference_noop_count": int(style_blueprint["cloud_trace_noops"]),
                },
            },
        )
    persona_path = output_root / "persona.json"
    stage2_path = output_root / "stage2.json"
    if stage2_path.exists() and not args.persona_only:
        stored = json.loads(stage2_path.read_text(encoding="utf-8"))
        stored_artifact = V1Stage2Artifact.model_validate(stored)
        try:
            validate_stage2_contract(
                personas=stored_artifact.persona_group.payload.personas,
                event_payload=stored_artifact.event_chains.payload,
                planned_reasoning_types=reasoning_types,
                vehicle_attributes=catalog,
            )
            validate_stage2_simulator_arguments(
                stored_artifact.event_chains.payload,
                runtime=runtime,
            )
            if args.state_evolution:
                if (
                    stored_artifact.event_chains.prompt_version
                    != V1_EVENT_CHAIN_STATE_EVOLUTION_PROMPT_VERSION
                ):
                    raise ValueError("stored Stage 2 uses the legacy event profile")
                validate_state_evolution_coverage(
                    stored_artifact.event_chains.payload
                )
        except ValueError:
            invalid_path = stage2_path.with_name(
                f"stage2.invalid-{stored_artifact.artifact_sha256[:8]}.json"
            )
            stage2_path.replace(invalid_path)
        else:
            coverage = (
                _write_state_evolution_coverage(output_root, stored_artifact)
                if args.state_evolution
                else None
            )
            return {
                "status": "already_completed",
                "stage2_path": str(stage2_path),
                "artifact_sha256": stored["artifact_sha256"],
                "state_evolution_coverage": coverage,
            }

    persona_model = OpenAIV1PersonaGenerationModel(
        args.model,
        timeout_seconds=args.timeout_seconds,
    )
    if persona_path.exists():
        persona_group = V1GeneratedPersonaGroup.model_validate_json(
            persona_path.read_text(encoding="utf-8")
        )
    elif args.persona_checkpoint is not None:
        persona_group = V1GeneratedPersonaGroup.model_validate_json(
            args.persona_checkpoint.expanduser()
            .resolve(strict=True)
            .read_text(encoding="utf-8")
        )
        expected_group_id = f"candidate-group-{args.candidate_group:03d}"
        if persona_group.candidate_group_id != expected_group_id:
            raise ValueError(
                "--persona-checkpoint candidate group does not match request"
            )
        _write_json(persona_path, persona_group.model_dump(mode="json"))
    else:
        persona_group = None
        persona_error = None
        for _persona_attempt in range(1, args.max_attempts + 1):
            try:
                persona_group = persona_model.generate(
                    selected_seeds,
                    candidate_group_id=(
                        f"candidate-group-{args.candidate_group:03d}"
                    ),
                )
                break
            except ValueError as exc:
                persona_error = exc
        if persona_group is None:
            assert persona_error is not None
            raise persona_error
        persona_group = persona_group.model_copy(
            update={
                "usage": {
                    **persona_group.usage,
                    "generation_attempts": _persona_attempt,
                },
            }
        )
        _write_json(persona_path, persona_group.model_dump(mode="json"))
    if args.persona_only:
        return {
            "status": "persona_completed",
            "persona_path": str(persona_path),
            "usage": persona_group.usage,
        }

    event_path = output_root / "event_chains.json"
    event_chains = None
    if event_path.exists():
        stored_event_chains = V1GeneratedEventChains.model_validate_json(
            event_path.read_text(encoding="utf-8")
        )
        try:
            validate_stage2_contract(
                personas=persona_group.payload.personas,
                event_payload=stored_event_chains.payload,
                planned_reasoning_types=reasoning_types,
                vehicle_attributes=catalog,
            )
            validate_stage2_simulator_arguments(
                stored_event_chains.payload,
                runtime=runtime,
            )
            if args.state_evolution:
                if (
                    stored_event_chains.prompt_version
                    != V1_EVENT_CHAIN_STATE_EVOLUTION_PROMPT_VERSION
                ):
                    raise ValueError("stored events use the legacy event profile")
                validate_state_evolution_coverage(stored_event_chains.payload)
            event_chains = stored_event_chains
        except ValueError:
            invalid_path = event_path.with_name(
                f"event_chains.invalid-{stored_event_chains.input_sha256[:8]}.json"
            )
            event_path.replace(invalid_path)
    if event_chains is None:
        event_model = OpenAIV1EventChainGenerationModel(
            args.model,
            timeout_seconds=args.timeout_seconds,
            state_evolution=args.state_evolution,
        )
        last_error = None
        for attempt in range(1, args.max_attempts + 1):
            event_chains = event_model.generate(
                persona_group.payload.personas,
                scenario_candidate_id=f"scenario-candidate-{args.scenario:03d}",
                reasoning_types=reasoning_types,
                vehicle_attributes=catalog,
            )
            try:
                validate_stage2_contract(
                    personas=persona_group.payload.personas,
                    event_payload=event_chains.payload,
                    planned_reasoning_types=reasoning_types,
                    vehicle_attributes=catalog,
                )
                validate_stage2_simulator_arguments(
                    event_chains.payload,
                    runtime=runtime,
                )
                if args.state_evolution:
                    validate_state_evolution_coverage(event_chains.payload)
                event_chains = event_chains.model_copy(
                    update={
                        "usage": {
                            **event_chains.usage,
                            "generation_attempts": attempt,
                        },
                    }
                )
                break
            except ValueError as exc:
                last_error = exc
                event_chains = None
        if event_chains is None:
            assert last_error is not None
            raise last_error
        _write_json(event_path, event_chains.model_dump(mode="json"))

    class _StoredPersonaModel:
        def generate(self, seeds, *, candidate_group_id):
            del seeds, candidate_group_id
            return persona_group

    class _StoredEventModel:
        def generate(
            self,
            personas,
            *,
            scenario_candidate_id,
            reasoning_types,
            vehicle_attributes,
        ):
            del personas, scenario_candidate_id, reasoning_types, vehicle_attributes
            return event_chains

    artifact = generate_stage2_artifact(
        candidate_group_id=persona_group.candidate_group_id,
        scenario_candidate_id=event_chains.scenario_candidate_id,
        seeds=selected_seeds,
        reasoning_types=reasoning_types,
        vehicle_attributes=catalog,
        persona_model=_StoredPersonaModel(),
        event_model=_StoredEventModel(),
        require_state_evolution=args.state_evolution,
    )
    destination = write_stage2_artifact(output_root, artifact)
    coverage = (
        _write_state_evolution_coverage(output_root, artifact)
        if args.state_evolution
        else None
    )
    return {
        "status": "completed",
        "stage2_path": str(destination),
        "artifact_sha256": artifact.artifact_sha256,
        "audit": artifact.audit.model_dump(mode="json"),
        "state_evolution_coverage": coverage,
        "usage": {
            "persona": persona_group.usage,
            "event_chain": event_chains.usage,
        },
    }


def _write_state_evolution_coverage(
    output_root: Path,
    artifact: V1Stage2Artifact,
) -> dict[str, Any]:
    coverage = validate_state_evolution_coverage(
        artifact.event_chains.payload
    ).model_dump(mode="json")
    payload = {
        "schema_version": "vehiclemembench-v2-state-evolution-coverage-v1",
        "source_stage2_sha256": artifact.artifact_sha256,
        "event_prompt_version": artifact.event_chains.prompt_version,
        "coverage": coverage,
    }
    _write_json(output_root / "state-evolution-coverage.json", payload)
    return coverage


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    payload = run(build_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
