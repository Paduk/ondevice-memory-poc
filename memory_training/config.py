"""Shared paths and model matrix for the memory-training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_STORAGE_ROOT = Path("/mnt/data/hj153lee/PalmClaw")
DEFAULT_WORKSPACE_ROOT = DEFAULT_STORAGE_ROOT / "on-device-memory-training"
DEFAULT_DATA_ROOT = (
    DEFAULT_STORAGE_ROOT
    / "evaluation"
    / "vehiclemembench-v2-training"
    / "hybrid-s1-s100-summary-patch-delta-v2"
)


@dataclass(frozen=True)
class TargetModel:
    key: str
    family: str
    parameters_b: float
    hf_id: str
    ollama_tag: str


TARGET_MODELS = (
    TargetModel(
        "granite4-350m",
        "granite4",
        0.35,
        "ibm-granite/granite-4.0-350m",
        "granite4:350m",
    ),
    TargetModel(
        "granite4-1b",
        "granite4",
        1,
        "ibm-granite/granite-4.0-1b",
        "granite4:1b",
    ),
    TargetModel(
        "qwen3.5-0.8b",
        "qwen3.5",
        0.8,
        "Qwen/Qwen3.5-0.8B",
        "qwen3.5:0.8b",
    ),
    TargetModel("qwen3.5-2b", "qwen3.5", 2, "Qwen/Qwen3.5-2B", "qwen3.5:2b"),
    TargetModel("qwen3.5-4b", "qwen3.5", 4, "Qwen/Qwen3.5-4B", "qwen3.5:4b"),
    TargetModel("qwen3.5-9b", "qwen3.5", 9, "Qwen/Qwen3.5-9B", "qwen3.5:9b"),
    TargetModel(
        "granite4.1-3b",
        "granite4.1",
        3,
        "ibm-granite/granite-4.1-3b",
        "granite4.1:3b",
    ),
    TargetModel(
        "granite4.1-8b",
        "granite4.1",
        8,
        "ibm-granite/granite-4.1-8b",
        "granite4.1:8b",
    ),
)

MODEL_BY_KEY = {model.key: model for model in TARGET_MODELS}

WORKSPACE_DIRECTORIES = (
    "cache/huggingface/hub",
    "cache/huggingface/datasets",
    "cache/torch",
    "checkpoints",
    "exports/ollama",
    "logs",
    "mlflow/artifacts",
    "reports/preflight",
    "runs",
    "tmp",
)


def split_for_scenario(scenario_index: int) -> str:
    # S1-S100 are the original Hybrid V2 scenarios.  T1-T20 are stored in the
    # same indexed files as 101-120 so the integer catalog remains backwards
    # compatible while the two scenario families never collide.
    if 101 <= scenario_index <= 110:
        return "train"
    if scenario_index == 111:
        return "validation"
    if 112 <= scenario_index <= 120:
        return "test"
    # Original VehicleMemBench V1 scenarios used only as auxiliary training
    # data are encoded as 201-250 to avoid colliding with Hybrid V2 S1-S100.
    # They intentionally have no validation/test partition in this pipeline.
    if 201 <= scenario_index <= 250:
        return "train"
    # Training-only V1-style structural clones.  These keep the original
    # V1 source split disjoint from the generated S301-S320 identities.
    if 301 <= scenario_index <= 320:
        return "train"
    if not 1 <= scenario_index <= 100:
        raise ValueError(
            "scenario_index must be S1-S100, encoded T1-T20, "
            "training-only encoded V1 S1-S50, or V1-style S301-S320: "
            f"{scenario_index}"
        )
    if scenario_index <= 80:
        return "train"
    if scenario_index <= 90:
        return "validation"
    return "test"
