"""Target-model loading and language-only PEFT configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import MODEL_BY_KEY, TargetModel


@dataclass(frozen=True)
class ModelBundle:
    spec: TargetModel
    model: Any
    tokenizer: Any
    lora_targets: tuple[str, ...]


def select_language_lora_targets(model: Any) -> list[str]:
    import torch

    preferred = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    found = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1] in preferred
        and not any(part in name.lower() for part in ("visual", "vision"))
    }
    if not found:
        raise RuntimeError("No supported language LoRA targets were found")
    return sorted(found)


def load_peft_bundle(
    model_key: str,
    *,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    adapter_path: Path | None = None,
    gradient_checkpointing: bool = True,
    cache_dir: Path | None = None,
) -> ModelBundle:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoConfig, AutoTokenizer

    spec = MODEL_BY_KEY[model_key]
    config = AutoConfig.from_pretrained(
        spec.hf_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "cache_dir": cache_dir,
    }
    if config.model_type == "qwen3_5":
        from transformers import AutoModelForImageTextToText

        loader = AutoModelForImageTextToText
    else:
        from transformers import AutoModelForCausalLM

        loader = AutoModelForCausalLM
    model = loader.from_pretrained(spec.hf_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    for name, parameter in model.named_parameters():
        if any(part in name.lower() for part in ("visual", "vision")):
            parameter.requires_grad = False
    targets = select_language_lora_targets(model)
    if adapter_path is None:
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=targets,
                task_type="CAUSAL_LM",
            ),
        )
    else:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return ModelBundle(spec, model, tokenizer, tuple(targets))
