"""Run a local HF/PEFT checkpoint as the LongMemEval answer reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_WORKSPACE_ROOT, MODEL_BY_KEY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "adapter").is_dir():
        raise FileNotFoundError(checkpoint / "adapter")
    rows = [
        json.loads(line)
        for line in args.inputs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    output = args.output.resolve()
    records = output / "records"
    records.mkdir(parents=True, exist_ok=True)
    pending = [
        row for row in rows if not (records / f"{row['question_id']}.json").is_file()
    ]

    if pending:
        _configure_runtime_paths(args.workspace.resolve())
        from accelerate import Accelerator
        from .models import load_peft_bundle

        accelerator = Accelerator()
        bundle = load_peft_bundle(
            args.model,
            adapter_path=checkpoint / "adapter",
            gradient_checkpointing=False,
            cache_dir=args.workspace.resolve() / "cache" / "huggingface" / "hub",
        )
        model = accelerator.prepare(bundle.model)
        model.eval()
        tokenizer = bundle.tokenizer
        tokenizer.padding_side = "left"
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Answer the question using only the supplied memory. "
                                "If the memory is insufficient, say so concisely."
                            ),
                        },
                        {"role": "user", "content": row["reader_prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in chunk
            ]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(accelerator.device)
                for key, value in encoded.items()
                if key != "token_type_ids"
            }
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_width = encoded["input_ids"].shape[1]
            answers = tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
            for row, answer in zip(chunk, answers, strict=True):
                payload = {
                    "question_id": str(row["question_id"]),
                    "hypothesis": answer.strip(),
                    "reader_model": args.model,
                    "reader_checkpoint": str(checkpoint),
                }
                (records / f"{row['question_id']}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(f"completed {min(start + len(chunk), len(pending))}/{len(pending)}", flush=True)

    by_id = {
        row["question_id"]: json.loads(
            (records / f"{row['question_id']}.json").read_text(encoding="utf-8")
        )
        for row in rows
    }
    ordered = [by_id[str(row["question_id"])] for row in rows]
    hypotheses = output / "hypotheses.jsonl"
    hypotheses.write_text(
        "".join(
            json.dumps(
                {"question_id": row["question_id"], "hypothesis": row["hypothesis"]},
                ensure_ascii=False,
            )
            + "\n"
            for row in ordered
        ),
        encoding="utf-8",
    )
    summary = {
        "questions": len(ordered),
        "reader_model": args.model,
        "reader_checkpoint": str(checkpoint),
        "empty_answers": sum(not row["hypothesis"] for row in ordered),
        "hypotheses": str(hypotheses),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _configure_runtime_paths(workspace: Path) -> None:
    import os

    huggingface = workspace / "cache" / "huggingface"
    for name, path in {
        "XDG_CACHE_HOME": workspace / "cache",
        "HF_HOME": huggingface,
        "HF_HUB_CACHE": huggingface / "hub",
        "HF_XET_CACHE": huggingface / "xet",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
