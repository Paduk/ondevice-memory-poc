"""Run a Patch checkpoint turn-wise over the LongMemEval 200/300 test split.

This stage produces final memories and reader prompts.  It intentionally does
not bake in a reader model, so the same fixed CloudLLM can compare every memory
method without rerunning the on-device memory pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DEFAULT_WORKSPACE_ROOT, MODEL_BY_KEY
from .longmemeval import DATE_FORMAT, load_json_list, sha256_file
from .methods import GeneralPatchMethod
from .training_data import ChatExampleEncoder
from .validation import generate_outputs_batch


@dataclass
class Stream:
    question: dict[str, Any]
    turns: list[tuple[str, str, str]]
    state: str = ""
    next_turn: int = 0
    invalid_outputs: int = 0
    errors: list[str] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_BY_KEY), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--full-data", type=Path, required=True)
    parser.add_argument("--test-questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--question-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-turns-per-question", type=int)
    return parser


def full_user_turns(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    sessions = row["haystack_sessions"]
    dates = row["haystack_dates"]
    session_ids = row["haystack_session_ids"]
    if not (len(sessions) == len(dates) == len(session_ids)):
        raise ValueError(f"Unaligned history arrays for {row['question_id']}")
    ordered = sorted(
        zip(dates, session_ids, sessions, strict=True),
        key=lambda item: (item[0], item[1]),
    )
    return [
        (str(date), str(session_id), " ".join(str(message["content"]).split()))
        for date, session_id, messages in ordered
        for message in messages
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.question_batch_size < 1:
        raise ValueError("question-batch-size must be positive")
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "adapter").is_dir():
        raise FileNotFoundError(f"Checkpoint adapter not found: {checkpoint}")
    output = args.output.resolve()
    result_dir = output / "memories"
    result_dir.mkdir(parents=True, exist_ok=True)
    full_data = load_json_list(args.full_data)
    full_by_id = {str(row["question_id"]): row for row in full_data}
    questions = [
        json.loads(line)
        for line in args.test_questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        questions = questions[: args.limit]
    missing = [row["question_id"] for row in questions if row["question_id"] not in full_by_id]
    if missing:
        raise ValueError(f"Missing {len(missing)} questions from full data")

    signature = _signature(args, checkpoint)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise ValueError(f"Stale output directory: {output}")
    else:
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "palmclaw-longmemeval-memory-eval-v1",
                    "signature": signature,
                    "model": args.model,
                    "checkpoint": str(checkpoint),
                    "history_policy": "all user turns, chronological, question hidden",
                    "reader_policy": "separate fixed reader over final_memory + question",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    pending = [
        row for row in questions if not (result_dir / f"{row['question_id']}.json").is_file()
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
        method = GeneralPatchMethod()
        encoder = ChatExampleEncoder(bundle.tokenizer, max_length=args.max_length)
        for start in range(0, len(pending), args.question_batch_size):
            chunk = pending[start : start + args.question_batch_size]
            streams = []
            for question in chunk:
                turns = full_user_turns(full_by_id[str(question["question_id"])])
                if args.max_turns_per_question is not None:
                    turns = turns[: args.max_turns_per_question]
                streams.append(Stream(question, turns))
            _run_streams(
                streams,
                model=model,
                tokenizer=bundle.tokenizer,
                method=method,
                encoder=encoder,
                accelerator=accelerator,
                max_new_tokens=args.max_new_tokens,
            )
            for stream in streams:
                question_id = str(stream.question["question_id"])
                payload = {
                    **stream.question,
                    "final_memory": stream.state,
                    "processed_user_turns": stream.next_turn,
                    "invalid_outputs": stream.invalid_outputs,
                    "errors": stream.errors,
                    "signature": signature,
                }
                (result_dir / f"{question_id}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(f"completed {min(start + len(chunk), len(pending))}/{len(pending)} pending questions", flush=True)

    results = [
        json.loads((result_dir / f"{row['question_id']}.json").read_text(encoding="utf-8"))
        for row in questions
    ]
    reader_rows = []
    for result in results:
        prompt = (
            "Answer the question using only the final user memory below. "
            "If the memory is insufficient, say so briefly.\n\n"
            f"Final user memory:\n{result['final_memory'] or '(empty)'}\n\n"
            f"Current date: {result['question_date']}\n"
            f"Question: {result['question']}\nAnswer:"
        )
        reader_rows.append(
            {
                "question_id": result["question_id"],
                "question_type": result["question_type"],
                "question": result["question"],
                "answer": result["answer"],
                "reader_prompt": prompt,
            }
        )
    (output / "reader_inputs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reader_rows),
        encoding="utf-8",
    )
    summary = {
        "questions": len(results),
        "processed_user_turns": sum(row["processed_user_turns"] for row in results),
        "invalid_outputs": sum(row["invalid_outputs"] for row in results),
        "empty_memories": sum(not row["final_memory"].strip() for row in results),
        "reader_inputs": str(output / "reader_inputs.jsonl"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _run_streams(
    streams: list[Stream],
    *,
    model: Any,
    tokenizer: Any,
    method: GeneralPatchMethod,
    encoder: ChatExampleEncoder,
    accelerator: Any,
    max_new_tokens: int,
) -> None:
    while True:
        active = [stream for stream in streams if stream.next_turn < len(stream.turns)]
        if not active:
            return
        rows = []
        for stream in active:
            date, session_id, text = stream.turns[stream.next_turn]
            rows.append(
                {
                    "turn_id": f"lme:{stream.question['question_id']}:{session_id}:{stream.next_turn}",
                    "timestamp": date,
                    "current_turn": {
                        "speaker_id": "user",
                        "speaker_name": "User",
                        "text": text,
                    },
                    "input": {"previous_memory": stream.state},
                }
            )
        generated = generate_outputs_batch(
            model,
            tokenizer,
            method,
            rows,
            encoder=encoder,
            accelerator=accelerator,
            max_new_tokens=max_new_tokens,
        )
        for stream, (text, _prompt_tokens, _generated_tokens, _elapsed) in zip(
            active, generated, strict=True
        ):
            try:
                parsed = method.parse_output(text)
                stream.state = method.apply_output(stream.state, parsed)
            except Exception as exc:  # invalid generation preserves prior memory
                stream.invalid_outputs += 1
                if len(stream.errors) < 20:
                    stream.errors.append(f"turn {stream.next_turn}: {type(exc).__name__}: {exc}")
            stream.next_turn += 1


def _signature(args: argparse.Namespace, checkpoint: Path) -> str:
    value = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "adapter_config": sha256_file(checkpoint / "adapter" / "adapter_config.json"),
        "full_data": sha256_file(args.full_data),
        "test_questions": sha256_file(args.test_questions),
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "max_turns_per_question": args.max_turns_per_question,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _configure_runtime_paths(workspace: Path) -> None:
    import os

    huggingface = workspace / "cache" / "huggingface"
    values = {
        "XDG_CACHE_HOME": workspace / "cache",
        "HF_HOME": huggingface,
        "HF_HUB_CACHE": huggingface / "hub",
        "HF_XET_CACHE": huggingface / "xet",
        "HF_DATASETS_CACHE": huggingface / "datasets",
    }
    for name, path in values.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
