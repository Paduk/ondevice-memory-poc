"""Prepare a label-free LongMemEval KU/TR Patch adaptation dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import build_catalog
from .quiz_sft import SCHEMA_VERSION as QUIZ_SCHEMA_VERSION
from .longmemeval import (
    SCHEMA_VERSION,
    aligned_views,
    build_question_rows,
    dataset_statistics,
    load_json_list,
    select_memalpha_split,
    sha256_file,
    split_development,
    validate_replay,
)

DEFAULT_SPLIT = Path(__file__).parent / "configs" / "longmemeval_memalpha_200_train_ids.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--question-metadata", type=Path, required=True)
    parser.add_argument("--full-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [args.oracle, args.question_metadata, args.full_data, args.split_manifest]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"Output is not empty: {output}; pass --force")
    output.mkdir(parents=True, exist_ok=True)

    oracle = load_json_list(args.oracle)
    metadata = load_json_list(args.question_metadata)
    full_data = load_json_list(args.full_data)
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    development, test = select_memalpha_split(
        oracle, split_manifest["train_question_ids"]
    )
    training, validation, abstention = split_development(
        development,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    # This pilot intentionally uses every deterministic development example for
    # adaptation; model selection is performed only on the held-out 120 questions.
    training = sorted([*training, *validation], key=lambda row: str(row["question_id"]))
    validation = []
    metadata_by_id = {str(row["question_id"]): row for row in metadata}
    full_by_id = {str(row["question_id"]): row for row in full_data}
    official_ids = {str(row["question_id"]) for row in oracle}
    if set(full_by_id) != official_ids:
        raise ValueError("Full and oracle LongMemEval question IDs differ")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    quiz_rows: list[dict[str, Any]] = []
    conversion_counts: Counter[str] = Counter()
    for split, questions, scenario_start, scenario_count in (
        ("train", training, 1, 80),
        ("validation", validation, 81, 10),
    ):
        for ordinal, question in enumerate(questions):
            question_id = str(question["question_id"])
            scenario = scenario_start + ordinal % scenario_count
            generated, counts = build_question_rows(
                question,
                metadata_by_id[question_id],
                split=split,
                scenario_index=scenario,
                question_ordinal=ordinal + 1,
                include_noop=True,
            )
            rows_by_split[split].extend(generated)
            conversion_counts.update(counts)
            final_memory = generated[-1]["target"]["next_memory"]
            quiz_rows.append(
                {
                    "schema_version": QUIZ_SCHEMA_VERSION,
                    "sample_id": f"lme:{question_id}:final-qa",
                    "scenario_index": scenario,
                    "source_split": split,
                    "sft_split": split,
                    "quiz_type": "FINAL",
                    "quiz_id": f"lme:{question_id}",
                    "reasoning_type": question["question_type"],
                    "memory_ref": {
                        "question_id": question_id,
                        "final_memory_sample_id": generated[-1]["sample_id"],
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer the question using only the supplied memory. "
                                "If the memory is insufficient, say so concisely."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question date: {question['question_date']}\n"
                                f"Memory:\n{final_memory}\n\n"
                                f"Question:\n{question['question']}"
                            ),
                        },
                        {"role": "assistant", "content": str(question["answer"])},
                    ],
                    "target": {"answer": str(question["answer"])},
                    "provenance": {
                        "dataset": "LongMemEval",
                        "question_id": question_id,
                        "question_type": question["question_type"],
                        "label_source": "official_answer",
                    },
                }
            )
    all_rows = [*rows_by_split["train"], *rows_by_split["validation"]]
    validate_replay(all_rows)

    handles = {
        view: (output / f"{view}.jsonl").open("w", encoding="utf-8")
        for view in ("summary", "patch", "delta")
    }
    try:
        for row in all_rows:
            for view, view_row in aligned_views(row).items():
                handles[view].write(json.dumps(view_row, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    question_records = {"train": training, "validation": validation, "test": test}
    stats = dataset_statistics(
        rows_by_split,
        question_records,
        audit_abstentions=abstention,
        conversion_counts=conversion_counts,
    )
    stats["full_test_user_turns"] = sum(
        message.get("role") == "user"
        for question in test
        for session in full_by_id[str(question["question_id"])]["haystack_sessions"]
        for message in session
    )
    stats["patch_replay"] = {"checked": len(all_rows), "failed": 0}

    test_rows = []
    for row in test:
        question_id = str(row["question_id"])
        full = full_by_id[question_id]
        test_rows.append(
            {
                "question_id": question_id,
                "question_type": row["question_type"],
                "question": row["question"],
                "answer": row["answer"],
                "question_date": row["question_date"],
                "full_data_question_id": full["question_id"],
            }
        )
    (output / "test_questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in test_rows),
        encoding="utf-8",
    )
    (output / "audit_abstentions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in abstention),
        encoding="utf-8",
    )
    (output / "quiz_sft.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quiz_rows),
        encoding="utf-8",
    )
    quiz_counts = Counter(row["sft_split"] for row in quiz_rows)
    (output / "quiz_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": QUIZ_SCHEMA_VERSION,
                "rows": len(quiz_rows),
                "by_split": dict(sorted(quiz_counts.items())),
                "target": "natural_language_answer",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "general_patch",
        "label_policy": "has_answer=true becomes deterministic ADD/REPLACE; has_answer=false becomes NO_OP",
        "split_policy": "Mem-alpha public 200/300 split, restricted to KU/TR; 10% stratified validation from non-abstention development questions",
        "abstention_policy": "excluded from SFT and retained for audit",
        "negative_policy": "all has_answer=false user turns are labeled NO_OP for this pilot",
        "quiz_policy": "one official final question/answer row per development question, conditioned on deterministic final memory",
        "training_note": "Patch/general_patch memory SFT plus natural-language final-QA SFT",
        "sources": {
            "oracle": {"path": str(args.oracle.resolve()), "sha256": sha256_file(args.oracle)},
            "question_metadata": {"path": str(args.question_metadata.resolve()), "sha256": sha256_file(args.question_metadata)},
            "full_data": {"path": str(args.full_data.resolve()), "sha256": sha256_file(args.full_data)},
            "split_manifest": {"path": str(args.split_manifest.resolve()), "sha256": sha256_file(args.split_manifest)},
        },
        "statistics": stats,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    catalog = build_catalog(output, output / "catalog.sqlite")
    manifest["catalog"] = catalog
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # The catalog fingerprints manifest.json, so rebuild after adding catalog metadata.
    catalog = build_catalog(output, output / "catalog.sqlite")
    result = {"output": str(output), "statistics": stats, "catalog": catalog}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()
