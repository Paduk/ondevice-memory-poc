#!/usr/bin/env python3
"""Generate one resumable V2 Temporal/Current Stage 2 scenario."""

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
    OpenAIV1PersonaGenerationModel,
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
from palmclaw_ubuntu.vehicle_bench.v2_temporal_anchor import (
    OpenAIV2TemporalAnchorGenerationModel,
    V2GeneratedTemporalAnchors,
    load_temporal_anchor_plan,
    validate_temporal_anchors,
    write_temporal_anchor_plan,
)
from palmclaw_ubuntu.vehicle_bench.v2_temporal_scenario import (
    OpenAIV2TemporalEventChainGenerationModel,
    V2GeneratedTemporalEventChains,
    build_temporal_plan_artifact,
    normalize_temporal_event_payload,
    temporal_scenario_profile,
    validate_temporal_event_payload,
    write_temporal_plan,
)

DEFAULT_DATASET_ROOT = Path("/home/hj153lee/VehicleMemBench")
DEFAULT_PERSONA_SEEDS = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction/"
    "sources/personahub-elite-sample-v1/persona_seeds.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-temporal/"
    "stage2-temporal-terra-t01-r1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--persona-seeds", type=Path, default=DEFAULT_PERSONA_SEEDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--model", default=V1_DEFAULT_GENERATION_MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    profile = temporal_scenario_profile(args.scenario)
    dataset = load_vehicle_benchmark(
        args.dataset_root,
        expected_commit=OFFICIAL_UPSTREAM_COMMIT,
        strict=True,
    )
    seeds = load_persona_seeds(args.persona_seeds)
    seed_groups = build_persona_seed_groups(seeds, candidate_group_count=200)
    selected_seeds = seed_groups[profile.candidate_group - 1]
    reasoning_types = build_reasoning_type_plan(20)[profile.scenario_index - 1]
    catalog = build_vehicle_attribute_catalog(dataset.tool_schemas)
    runtime = VehicleWorldRuntime(dataset.root)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage2_path = output_root / "stage2.json"
    plan_path = output_root / "temporal-plan.json"
    if stage2_path.exists() and plan_path.exists():
        stage2 = V1Stage2Artifact.model_validate_json(
            stage2_path.read_text(encoding="utf-8")
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("source_stage2_sha256") != stage2.artifact_sha256:
            raise ValueError("Stored temporal plan does not match Stage 2")
        anchors = _load_or_generate_anchors(
            stage2=stage2,
            temporal_plan=plan,
            output_root=output_root,
            model_id=args.model,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
        )
        return {
            "status": "already_completed",
            "stage2_path": str(stage2_path),
            "temporal_plan_path": str(plan_path),
            "temporal_anchor_path": str(
                output_root / "temporal-anchor-plan.json"
            ),
            "artifact_sha256": stage2.artifact_sha256,
            "audit": plan.get("audit"),
            "anchor_count": len(anchors.payload.anchors),
        }

    persona_path = output_root / "persona.json"
    if persona_path.exists():
        persona_group = V1GeneratedPersonaGroup.model_validate_json(
            persona_path.read_text(encoding="utf-8")
        )
    else:
        persona_model = OpenAIV1PersonaGenerationModel(
            args.model,
            timeout_seconds=args.timeout_seconds,
        )
        persona_group = None
        last_error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                generated = persona_model.generate(
                    selected_seeds,
                    candidate_group_id=f"candidate-group-{profile.candidate_group:03d}",
                )
                persona_group = generated.model_copy(
                    update={
                        "usage": {**generated.usage, "generation_attempts": attempt}
                    }
                )
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
        if persona_group is None:
            assert last_error is not None
            raise last_error
        _write_json(persona_path, persona_group.model_dump(mode="json"))

    temporal_path = output_root / "temporal-event-chains.json"
    if temporal_path.exists():
        generated_temporal = V2GeneratedTemporalEventChains.model_validate_json(
            temporal_path.read_text(encoding="utf-8")
        )
        if generated_temporal.profile != profile:
            raise ValueError("Stored temporal event chains use a different profile")
        validate_temporal_event_payload(generated_temporal.payload, profile=profile)
    else:
        event_model = OpenAIV2TemporalEventChainGenerationModel(
            args.model,
            timeout_seconds=args.timeout_seconds,
        )
        generated_temporal = None
        last_error = None
        for attempt_path in sorted(
            output_root.glob("temporal-event-chains.attempt-*.json"),
            reverse=True,
        ):
            try:
                candidate = V2GeneratedTemporalEventChains.model_validate_json(
                    attempt_path.read_text(encoding="utf-8")
                )
                generated_temporal = _prepare_generated_temporal(
                    candidate,
                    profile=profile,
                    personas=persona_group.payload.personas,
                    reasoning_types=reasoning_types,
                    catalog=catalog,
                    runtime=runtime,
                )
                break
            except ValueError as exc:
                last_error = exc
        for attempt in range(1, args.max_attempts + 1):
            if generated_temporal is not None:
                break
            try:
                generated = event_model.generate(
                    persona_group.payload.personas,
                    scenario_candidate_id=profile.scenario_id,
                    reasoning_types=reasoning_types,
                    vehicle_attributes=catalog,
                    profile=profile,
                )
                _write_json(
                    output_root / f"temporal-event-chains.attempt-{attempt}.json",
                    generated.model_dump(mode="json"),
                )
                prepared = _prepare_generated_temporal(
                    generated,
                    profile=profile,
                    personas=persona_group.payload.personas,
                    reasoning_types=reasoning_types,
                    catalog=catalog,
                    runtime=runtime,
                )
                generated_temporal = prepared.model_copy(
                    update={
                        "usage": {**prepared.usage, "generation_attempts": attempt}
                    }
                )
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
        if generated_temporal is None:
            assert last_error is not None
            raise last_error
        _write_json(temporal_path, generated_temporal.model_dump(mode="json"))

    event_chains = generated_temporal.as_v1_event_chains()

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

    stage2 = generate_stage2_artifact(
        candidate_group_id=persona_group.candidate_group_id,
        scenario_candidate_id=profile.scenario_id,
        seeds=selected_seeds,
        reasoning_types=reasoning_types,
        vehicle_attributes=catalog,
        persona_model=_StoredPersonaModel(),
        event_model=_StoredEventModel(),
        require_state_evolution=False,
    )
    write_stage2_artifact(output_root, stage2)
    plan = build_temporal_plan_artifact(
        generated_temporal,
        source_stage2_sha256=stage2.artifact_sha256,
    )
    write_temporal_plan(plan_path, plan)
    anchors = _load_or_generate_anchors(
        stage2=stage2,
        temporal_plan=plan,
        output_root=output_root,
        model_id=args.model,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    )
    return {
        "status": "completed",
        "stage2_path": str(stage2_path),
        "temporal_plan_path": str(plan_path),
        "temporal_anchor_path": str(output_root / "temporal-anchor-plan.json"),
        "artifact_sha256": stage2.artifact_sha256,
        "profile": profile.model_dump(mode="json"),
        "audit": plan["audit"],
        "anchor_count": len(anchors.payload.anchors),
        "usage": {
            "persona": persona_group.usage,
            "event_chain": generated_temporal.usage,
        },
    }


def _load_or_generate_anchors(
    *,
    stage2: V1Stage2Artifact,
    temporal_plan: dict[str, Any],
    output_root: Path,
    model_id: str,
    timeout_seconds: float,
    max_attempts: int,
) -> V2GeneratedTemporalAnchors:
    anchor_path = output_root / "temporal-anchor-plan.json"
    if anchor_path.exists():
        artifact, _ = load_temporal_anchor_plan(
            anchor_path,
            source_stage2_sha256=stage2.artifact_sha256,
            source_temporal_plan_sha256=str(temporal_plan["artifact_sha256"]),
        )
        assert artifact is not None
        validate_temporal_anchors(stage2, temporal_plan, artifact.payload)
        return artifact

    candidates = []
    for attempt_path in sorted(
        output_root.glob("temporal-anchors.attempt-*.json"), reverse=True
    ):
        try:
            candidates.append(
                V2GeneratedTemporalAnchors.model_validate_json(
                    attempt_path.read_text(encoding="utf-8")
                )
            )
        except ValueError:
            continue
    model = OpenAIV2TemporalAnchorGenerationModel(
        model_id,
        timeout_seconds=timeout_seconds,
    )
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            validate_temporal_anchors(stage2, temporal_plan, candidate.payload)
            write_temporal_anchor_plan(anchor_path, candidate)
            return candidate
        except ValueError as exc:
            last_error = exc
    for attempt in range(1, max_attempts + 1):
        try:
            candidate = model.generate(stage2=stage2, temporal_plan=temporal_plan)
            write_temporal_anchor_plan(
                output_root / f"temporal-anchors.attempt-{attempt}.json",
                candidate,
            )
            validate_temporal_anchors(stage2, temporal_plan, candidate.payload)
            write_temporal_anchor_plan(anchor_path, candidate)
            return candidate
        except (RuntimeError, ValueError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_generated_temporal(
    generated: V2GeneratedTemporalEventChains,
    *,
    profile,
    personas,
    reasoning_types,
    catalog,
    runtime,
) -> V2GeneratedTemporalEventChains:
    normalized, notes = normalize_temporal_event_payload(
        generated.payload,
        personas=personas,
    )
    prepared = generated.model_copy(
        update={
            "payload": normalized,
            "normalization_notes": tuple(
                (*generated.normalization_notes, *notes)
            ),
        }
    )
    validate_temporal_event_payload(prepared.payload, profile=profile)
    event_chains = prepared.as_v1_event_chains()
    validate_stage2_contract(
        personas=personas,
        event_payload=event_chains.payload,
        planned_reasoning_types=reasoning_types,
        vehicle_attributes=catalog,
    )
    validate_stage2_simulator_arguments(event_chains.payload, runtime=runtime)
    validate_state_evolution_coverage(
        event_chains.payload,
        minimum_replace_count=1,
    )
    return prepared


def main() -> int:
    payload = run(build_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
