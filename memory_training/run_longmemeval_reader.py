"""Run one fixed OpenAI-compatible reader over prepared LongMemEval memories."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    from openai import OpenAI

    rows = [
        json.loads(line)
        for line in args.inputs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = args.output.resolve()
    record_dir = output / "records"
    record_dir.mkdir(parents=True, exist_ok=True)

    def answer(row: dict[str, Any]) -> dict[str, str]:
        path = record_dir / f"{row['question_id']}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer from the supplied memory. Be concise and do not invent missing facts.",
                },
                {"role": "user", "content": row["reader_prompt"]},
            ],
            max_completion_tokens=args.max_tokens,
        )
        result = {
            "question_id": str(row["question_id"]),
            "hypothesis": str(completion.choices[0].message.content or "").strip(),
            "reader_model": args.model,
        }
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(answer, row): row["question_id"] for row in rows}
        for future in as_completed(futures):
            results.append(future.result())
    by_id = {row["question_id"]: row for row in results}
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
        "hypotheses": str(hypotheses),
        "next_step": "score hypotheses.jsonl with the official LongMemEval evaluate_qa.py",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
