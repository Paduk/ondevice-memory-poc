"""Merge a best PEFT adapter, export GGUF, and register it with Ollama."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_WORKSPACE_ROOT, MODEL_BY_KEY
from .ollama_client import OllamaClient

ISOLATED_LLAMA_CPP = (
    DEFAULT_WORKSPACE_ROOT / "tools" / "llama.cpp-qwen35"
)
DEFAULT_LLAMA_CPP = (
    ISOLATED_LLAMA_CPP
    if (ISOLATED_LLAMA_CPP / "convert_hf_to_gguf.py").is_file()
    else Path("/home/hj153lee/llama.cpp")
)
ISOLATED_OLLAMA = (
    DEFAULT_WORKSPACE_ROOT / "tools" / "ollama-v0.32.15" / "bin" / "ollama"
)
DEFAULT_OLLAMA_BINARY = (
    ISOLATED_OLLAMA if ISOLATED_OLLAMA.is_file() else Path("ollama")
)


@dataclass(frozen=True)
class ExportPlan:
    run_id: str
    model_key: str
    method: str
    hf_id: str
    architecture: str
    adapter_dir: str
    export_dir: str
    merged_dir: str
    f16_gguf: str
    quantized_gguf: str
    modelfile: str
    ollama_tag: str
    ollama_binary: str
    gguf_outtype: str
    quantization: str
    context_length: int
    converter: str
    quantizer: str
    converter_supports_architecture: bool
    converter_arguments: tuple[str, ...]
    estimated_peak_bytes: int
    free_bytes: int
    disk_space_sufficient: bool
    ollama_version: str | None


def build_export_plan(args: argparse.Namespace) -> ExportPlan:
    from transformers import AutoConfig

    run_dir = args.run_dir.resolve()
    config = _read_json(run_dir / "config.json")
    model_key = args.model or str(config.get("model", {}).get("key", ""))
    method = args.method or str(config.get("method", ""))
    if model_key not in MODEL_BY_KEY:
        raise ValueError(f"Unknown target model: {model_key}")
    if not method:
        raise ValueError("Method is missing from run config and --method")
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
    else:
        best = _read_json(run_dir / "best-checkpoint.json")
        checkpoint = Path(best.get("checkpoint", "")).resolve()
    adapter_dir = checkpoint / "adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter not found: {adapter_dir}")
    spec = MODEL_BY_KEY[model_key]
    architecture = str(
        (
            AutoConfig.from_pretrained(spec.hf_id, trust_remote_code=True).architectures
            or [""]
        )[0]
    )
    export_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (args.workspace / "exports" / "ollama" / run_dir.name).resolve()
    )
    merged_dir = export_dir / "merged-hf"
    gguf_outtype = args.gguf_outtype.lower()
    quantization = args.quantization.upper()
    f16_gguf = export_dir / f"model-{gguf_outtype}.gguf"
    quantized_gguf = (
        f16_gguf
        if quantization == "NONE"
        else export_dir / f"model-{quantization.lower()}.gguf"
    )
    converter = (args.llama_cpp / "convert_hf_to_gguf.py").resolve()
    quantizer = (args.llama_cpp / "build" / "bin" / "llama-quantize").resolve()
    if not converter.is_file():
        raise FileNotFoundError("llama.cpp converter is missing")
    if quantization != "NONE" and not quantizer.is_file():
        raise FileNotFoundError("llama.cpp quantizer is missing")
    converter_supports = _converter_supports(args.llama_cpp, architecture)
    converter_arguments = _converter_arguments(architecture)
    parameter_bytes = spec.parameters_b * 1_000_000_000
    estimated_peak = int(parameter_bytes * 4.8)
    free = shutil.disk_usage(
        export_dir.parent if export_dir.parent.exists() else args.workspace
    ).free
    ollama_version = None
    try:
        client = OllamaClient(args.ollama_url, timeout_seconds=5)
        ollama_version = client.version()
        client.close()
    except Exception as exc:  # noqa: BLE001 - plan remains useful offline.
        ollama_version = f"unavailable:{type(exc).__name__}"
    artifact_type = gguf_outtype if quantization == "NONE" else quantization
    tag = args.ollama_tag or _default_tag(
        run_dir.name, model_key, method, artifact_type
    )
    return ExportPlan(
        run_id=run_dir.name,
        model_key=model_key,
        method=method,
        hf_id=spec.hf_id,
        architecture=architecture,
        adapter_dir=str(adapter_dir),
        export_dir=str(export_dir),
        merged_dir=str(merged_dir),
        f16_gguf=str(f16_gguf),
        quantized_gguf=str(quantized_gguf),
        modelfile=str(export_dir / "Modelfile"),
        ollama_tag=tag,
        ollama_binary=str(args.ollama_bin),
        gguf_outtype=gguf_outtype,
        quantization=quantization,
        context_length=args.context_length,
        converter=str(converter),
        quantizer=str(quantizer),
        converter_supports_architecture=converter_supports,
        converter_arguments=converter_arguments,
        estimated_peak_bytes=estimated_peak,
        free_bytes=free,
        disk_space_sufficient=free >= estimated_peak,
        ollama_version=ollama_version,
    )


def execute_export(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_export_plan(args)
    export_dir = Path(plan.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    _write_json(export_dir / "export-plan.json", asdict(plan))
    if args.stage == "plan":
        return {"status": "PLANNED", "plan": asdict(plan)}
    if not plan.disk_space_sufficient and not args.ignore_disk_check:
        raise RuntimeError(
            f"Insufficient free space: need about {plan.estimated_peak_bytes / 1024**3:.1f} GiB, "
            f"have {plan.free_bytes / 1024**3:.1f} GiB"
        )
    stages = ("merge", "gguf", "register") if args.stage == "all" else (args.stage,)
    if "merge" in stages:
        _merge_adapter(plan)
    if "gguf" in stages:
        if not plan.converter_supports_architecture:
            raise RuntimeError(
                f"Current llama.cpp does not support {plan.architecture}; update the isolated "
                "converter before GGUF export"
            )
        _convert_and_quantize(plan, threads=args.threads)
    if "register" in stages:
        _register_ollama(plan, args.ollama_url)
    manifest = _build_manifest(plan, args.ollama_url)
    _write_json(export_dir / "export-manifest.json", manifest)
    if args.cleanup_intermediate:
        _cleanup_intermediate(plan)
    return manifest


def _merge_adapter(plan: ExportPlan) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(plan.hf_id, trust_remote_code=True)
    kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if config.model_type == "qwen3_5":
        from transformers import AutoModelForImageTextToText

        loader = AutoModelForImageTextToText
    else:
        from transformers import AutoModelForCausalLM

        loader = AutoModelForCausalLM
    base = loader.from_pretrained(plan.hf_id, **kwargs)
    merged = PeftModel.from_pretrained(base, plan.adapter_dir).merge_and_unload()
    merged.save_pretrained(
        plan.merged_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    AutoTokenizer.from_pretrained(
        plan.adapter_dir, trust_remote_code=True
    ).save_pretrained(plan.merged_dir)


def _convert_and_quantize(plan: ExportPlan, *, threads: int) -> None:
    subprocess.run(
        [
            sys.executable,
            plan.converter,
            plan.merged_dir,
            "--outfile",
            plan.f16_gguf,
            "--outtype",
            plan.gguf_outtype,
            *plan.converter_arguments,
        ],
        check=True,
    )
    if plan.quantization != "NONE":
        subprocess.run(
            [
                plan.quantizer,
                plan.f16_gguf,
                plan.quantized_gguf,
                plan.quantization,
                str(threads),
            ],
            check=True,
        )
    modelfile = (
        f"FROM {plan.quantized_gguf}\n"
        "PARAMETER temperature 0\n"
        "PARAMETER seed 42\n"
        f"PARAMETER num_ctx {plan.context_length}\n"
    )
    Path(plan.modelfile).write_text(modelfile, encoding="utf-8")


def _converter_supports(llama_cpp: Path, architecture: str) -> bool:
    candidates = [llama_cpp / "convert_hf_to_gguf.py"]
    conversion = llama_cpp / "conversion"
    if conversion.is_dir():
        candidates.extend(conversion.rglob("*.py"))
    needle = re.compile(rf"[\"']{re.escape(architecture)}[\"']")
    return any(
        needle.search(path.read_text(encoding="utf-8", errors="ignore"))
        for path in candidates
        if path.is_file()
    )


def _converter_arguments(architecture: str) -> tuple[str, ...]:
    # Some merged Qwen3.5 exports advertise a NextN/MTP block without storing
    # its tensors. Omitting MTP prevents the runtime from expecting blk.N+1.
    return ("--no-mtp",) if architecture.startswith("Qwen3_5") else ()


def _register_ollama(plan: ExportPlan, ollama_url: str) -> None:
    if not Path(plan.quantized_gguf).is_file() or not Path(plan.modelfile).is_file():
        raise FileNotFoundError("Final GGUF or Modelfile is missing")
    environment = os.environ.copy()
    environment["OLLAMA_HOST"] = ollama_url
    subprocess.run(
        [plan.ollama_binary, "create", plan.ollama_tag, "-f", plan.modelfile],
        check=True,
        env=environment,
    )


def _build_manifest(plan: ExportPlan, ollama_url: str) -> dict[str, Any]:
    show = None
    show_error = None
    try:
        client = OllamaClient(ollama_url, timeout_seconds=30)
        show = client.show(plan.ollama_tag)
        client.close()
    except Exception as exc:  # noqa: BLE001 - partial stages are expected.
        show_error = f"{type(exc).__name__}: {exc}"
    gguf = Path(plan.quantized_gguf)
    return {
        "schema_version": "palmclaw-ollama-export-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "REGISTERED" if show else "PARTIAL",
        "plan": asdict(plan),
        "final_gguf_size_bytes": gguf.stat().st_size if gguf.is_file() else None,
        "final_gguf_sha256": _sha256(gguf) if gguf.is_file() else None,
        "ollama_show": show,
        "ollama_show_error": show_error,
    }


def _cleanup_intermediate(plan: ExportPlan) -> None:
    merged = Path(plan.merged_dir).resolve()
    export = Path(plan.export_dir).resolve()
    if merged.parent != export:
        raise RuntimeError("Refusing to clean an unexpected merged directory")
    if merged.is_dir():
        shutil.rmtree(merged)
    f16 = Path(plan.f16_gguf).resolve()
    final_gguf = Path(plan.quantized_gguf).resolve()
    if f16 != final_gguf and f16.parent == export and f16.is_file():
        f16.unlink()


def _default_tag(run_id: str, model: str, method: str, quantization: str) -> str:
    value = f"palmclaw-{model}-{method}-{run_id}".lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value).strip("-.")
    return f"{value}:{quantization.lower()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY))
    parser.add_argument("--method")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--llama-cpp", type=Path, default=DEFAULT_LLAMA_CPP)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-bin", type=Path, default=DEFAULT_OLLAMA_BINARY)
    parser.add_argument("--ollama-tag")
    parser.add_argument(
        "--gguf-outtype",
        choices=("f32", "f16", "bf16", "q8_0", "auto"),
        default="bf16",
        help="GGUF converter output type (default: bf16)",
    )
    parser.add_argument(
        "--quantization",
        default="NONE",
        help="llama.cpp quantization, or NONE to register the GGUF directly",
    )
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--stage", choices=("plan", "merge", "gguf", "register", "all"), default="plan"
    )
    parser.add_argument("--cleanup-intermediate", action="store_true")
    parser.add_argument("--ignore-disk-check", action="store_true")
    return parser


def main() -> None:
    result = execute_export(build_parser().parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
