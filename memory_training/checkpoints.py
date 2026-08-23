"""Adapter-only checkpoints with optimizer, scheduler, and exact resume state."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class TrainingProgress:
    epoch: int = 0
    next_batch: int = 0
    global_step: int = 0
    best_score: float | None = None
    best_checkpoint: str | None = None
    best_metric: str | None = None


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    progress: TrainingProgress,
) -> None:
    if not accelerator.is_main_process:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / "adapter"
    accelerator.unwrap_model(model).save_pretrained(
        adapter_dir, safe_serialization=True
    )
    tokenizer.save_pretrained(adapter_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "python_rng": random.getstate(),
        },
        checkpoint_dir / "training_state.pt",
    )
    (checkpoint_dir / "progress.json").write_text(
        json.dumps(asdict(progress), indent=2) + "\n", encoding="utf-8"
    )


def load_progress(checkpoint_dir: Path) -> TrainingProgress:
    payload = json.loads((checkpoint_dir / "progress.json").read_text(encoding="utf-8"))
    return TrainingProgress(**payload)


def restore_training_state(
    checkpoint_dir: Path, *, optimizer: Any, scheduler: Any
) -> None:
    payload = torch.load(
        checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False
    )
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    random.setstate(payload["python_rng"])
