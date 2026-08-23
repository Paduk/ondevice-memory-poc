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
    parameters_b: int
    hf_id: str
    ollama_tag: str


TARGET_MODELS = (
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
    if not 1 <= scenario_index <= 100:
        raise ValueError(f"scenario_index must be 1..100: {scenario_index}")
    if scenario_index <= 80:
        return "train"
    if scenario_index <= 90:
        return "validation"
    return "test"
