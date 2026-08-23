"""Run metadata, forward, or one-step LoRA backward checks for a target model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import DEFAULT_WORKSPACE_ROOT, MODEL_BY_KEY


def _select_linear_targets(model: object) -> list[str]:
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
        for name, module in model.named_modules()  # type: ignore[attr-defined]
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1] in preferred
        and not any(part in name.lower() for part in ("visual", "vision"))
    }
    if not found:
        raise RuntimeError("No supported language LoRA target modules were discovered")
    return sorted(found)


def run_canary(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    model_spec = MODEL_BY_KEY[args.model]
    config = AutoConfig.from_pretrained(model_spec.hf_id, trust_remote_code=True)
    report: dict[str, object] = {
        "schema_version": "palmclaw-memory-model-canary-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_key": model_spec.key,
        "hf_id": model_spec.hf_id,
        "mode": args.mode,
        "transformers_version": transformers.__version__,
        "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None),
        "status": "PASS",
    }
    if args.mode == "metadata":
        AutoTokenizer.from_pretrained(model_spec.hf_id, trust_remote_code=True)
        report["tokenizer_loaded"] = True
        return report

    kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": args.device}
    if config.model_type == "qwen3_5":
        from transformers import AutoModelForImageTextToText

        model_loader = AutoModelForImageTextToText
    else:
        from transformers import AutoModelForCausalLM

        model_loader = AutoModelForCausalLM
    model = model_loader.from_pretrained(model_spec.hf_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_spec.hf_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = (
        "Previous memory:\n- Alex prefers cabin temperature 22 C.\n\n"
        "Current turn:\nAlex: Set my usual temperature.\n\n"
        'Output JSON: {"decision":"NO_OP"}'
    )
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    model.config.use_cache = False

    if args.mode == "backward":
        from peft import LoraConfig, get_peft_model

        for name, parameter in model.named_parameters():
            if any(part in name.lower() for part in ("visual", "vision")):
                parameter.requires_grad = False
        targets = _select_linear_targets(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules=targets,
                task_type="CAUSAL_LM",
            ),
        )
        report["lora_targets"] = targets

    labels = batch["input_ids"].clone()
    output = model(**batch, labels=labels)
    report["loss"] = float(output.loss.detach().cpu())
    report["input_tokens"] = int(batch["input_ids"].shape[-1])
    if args.mode == "backward":
        output.loss.backward()
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        report["trainable_parameters"] = trainable
    if torch.cuda.is_available():
        report["peak_cuda_memory_mib"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 2
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument(
        "--mode", choices=("metadata", "forward", "backward"), default="metadata"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    report_dir = workspace / "reports" / "preflight"
    report_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(workspace / "cache" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(workspace / "cache" / "torch"))
    try:
        report = run_canary(args)
    except Exception as exc:  # noqa: BLE001 - persist all canary failures as reports.
        report = {
            "schema_version": "palmclaw-memory-model-canary-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_key": args.model,
            "mode": args.mode,
            "status": "FAIL",
            "error": str(exc),
        }
    path = report_dir / f"model-{args.model}-{args.mode}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
