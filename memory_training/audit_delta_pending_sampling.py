"""Audit Delta-v3 training exposures for pending-depth sampling conditions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEFAULT_WORKSPACE_ROOT
from .dataset import IndexedMemoryDataset, ensure_catalog
from .methods import METHODS, DeltaV3Method
from .pending_depth_ablation import (
    merge_exposure_counts,
    pending_depth,
    plan_exposure_counts,
)
from .sampling import EpochSampler, SamplingConfig

CURRENT_MAIN_TRAIN_SCENARIOS = (
    *range(15, 81),
    *range(101, 111),
    202,
    205,
    206,
    214,
    217,
    223,
    231,
    232,
    233,
    236,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--method", default="delta_v3_compact_k5")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("unstratified", "depth-weighted"),
        default=("unstratified", "depth-weighted"),
    )
    parser.add_argument("--pending-weights", nargs="+", type=int, default=(40, 30, 15, 10, 5))
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=(45,))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--noop-per-update", type=int, default=5)
    parser.add_argument("--adjacent-noop-fraction", type=float, default=0.30)
    parser.add_argument("--trajectory-fraction", type=float, default=0.20)
    parser.add_argument("--train-scenarios", nargs="+", type=int)
    parser.add_argument(
        "--model-key",
        help="Optionally count exact encoded Memory-SFT tokens for this tokenizer.",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    method = METHODS[args.method]()
    if not isinstance(method, DeltaV3Method):
        raise TypeError("Pending-depth audit requires a Delta-v3 method")
    interval = method.compaction_interval
    if len(args.pending_weights) != interval:
        raise ValueError(
            f"pending-weights must provide {interval} values for {args.method}"
        )
    train_scenarios = tuple(args.train_scenarios or CURRENT_MAIN_TRAIN_SCENARIOS)
    catalog_path = args.catalog_path or args.data_root / "catalog.sqlite"
    catalog = ensure_catalog(args.data_root.resolve(), catalog_path.resolve())
    allowed_row_ids = [
        row_id
        for scenario in train_scenarios
        for segment in catalog.scenario_eligible_segments(scenario)
        for row_id in segment
    ]
    source = IndexedMemoryDataset(catalog, method.source_view, row_ids=allowed_row_ids)
    depths: dict[int, int] = {}
    decisions: dict[int, str] = {}
    corpus = Counter()
    for position, row_id in enumerate(allowed_row_ids):
        row = source[position]
        depth = pending_depth(row, compaction_interval=interval)
        decision = str(row.get("target", {}).get("decision"))
        if decision not in {"NO_OP", "UPDATE"}:
            raise ValueError(f"Row {row_id} has invalid decision: {decision}")
        depths[row_id] = depth
        decisions[row_id] = decision
        corpus[(decision, depth)] += 1

    encoder = None
    token_cache: dict[int, tuple[int, int, int, bool]] = {}
    if args.model_key:
        from transformers import AutoTokenizer

        from .config import MODEL_BY_KEY
        from .models import configure_tokenizer_for_model
        from .training_data import ChatExampleEncoder

        spec = MODEL_BY_KEY[args.model_key]
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id,
            trust_remote_code=True,
            cache_dir=args.workspace / "cache" / "huggingface" / "hub",
        )
        configure_tokenizer_for_model(spec, tokenizer)
        encoder = ChatExampleEncoder(tokenizer, max_length=args.max_length)

    condition_reports: dict[str, Any] = {}
    for condition in args.conditions:
        weights = tuple(args.pending_weights) if condition == "depth-weighted" else ()
        seed_reports = []
        all_exposures = []
        all_token_counts: Counter[str] = Counter()
        for seed in args.sampling_seeds:
            config = SamplingConfig(
                noop_per_update=args.noop_per_update,
                adjacent_noop_fraction=args.adjacent_noop_fraction,
                trajectory_fraction=args.trajectory_fraction,
                train_scenarios=train_scenarios,
                noop_stratum_weights=weights,
                seed=seed,
                trajectory_seed=seed,
            )
            sampler = EpochSampler(
                catalog,
                config,
                noop_strata=depths if weights else None,
            )
            epochs = []
            for epoch in range(args.epochs):
                plan = sampler.build(epoch)
                exposures = plan_exposure_counts(plan, depths, decisions)
                all_exposures.append(exposures)
                token_counts = (
                    _plan_token_counts(
                        plan,
                        source,
                        method,
                        encoder,
                        token_cache,
                        max_length=args.max_length,
                    )
                    if encoder is not None
                    else None
                )
                if token_counts is not None:
                    all_token_counts.update(token_counts)
                epochs.append(
                    {
                        "epoch": epoch + 1,
                        "plan": plan.summary(),
                        "exposures": exposures,
                        "memory_sft_tokens": token_counts,
                    }
                )
            seed_reports.append({"seed": seed, "epochs": epochs})
        condition_reports[condition] = {
            "aggregate_exposures": merge_exposure_counts(all_exposures),
            "aggregate_memory_sft_tokens": (
                dict(all_token_counts) if encoder is not None else None
            ),
            "seeds": seed_reports,
        }

    result = {
        "schema_version": "palmclaw-delta-pending-sampling-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "compaction_interval": interval,
        "conditions": list(args.conditions),
        "pending_weights": list(args.pending_weights),
        "sampling_seeds": list(args.sampling_seeds),
        "epochs": args.epochs,
        "token_accounting": {
            "model_key": args.model_key,
            "max_length": args.max_length if args.model_key else None,
            "scope": "Memory-SFT examples only; identical Quiz exposures are excluded",
        },
        "training": {
            "scenarios": list(train_scenarios),
            "noop_per_update": args.noop_per_update,
            "adjacent_noop_fraction": args.adjacent_noop_fraction,
            "trajectory_fraction": args.trajectory_fraction,
            "trajectory_sampling": "paired by seed across conditions",
        },
        "eligible_corpus": _nested_counts(corpus, interval),
        "condition_reports": condition_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_render_markdown(result), encoding="utf-8")
    return result


def _nested_counts(counter: Counter[tuple[str, int]], interval: int) -> dict[str, dict[str, int]]:
    return {
        decision: {
            str(depth): int(counter[(decision, depth)])
            for depth in range(interval)
        }
        for decision in ("NO_OP", "UPDATE")
    }


def _plan_token_counts(
    plan: Any,
    source: IndexedMemoryDataset,
    method: DeltaV3Method,
    encoder: Any,
    cache: dict[int, tuple[int, int, int, bool]],
    *,
    max_length: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    row_ids = [*plan.independent_row_ids]
    row_ids.extend(
        row_id for window in plan.trajectory_windows for row_id in window.row_ids
    )
    for row_id in row_ids:
        encoded_counts = cache.get(row_id)
        if encoded_counts is None:
            row = source[source.position_for_row_id(row_id)]
            encoded = encoder.encode(row, method, row_id=row_id)
            target_tokens = sum(label != -100 for label in encoded.labels)
            total_tokens = len(encoded.input_ids)
            encoded_counts = (
                total_tokens - target_tokens,
                target_tokens,
                total_tokens,
                total_tokens == max_length,
            )
            cache[row_id] = encoded_counts
        input_tokens, target_tokens, total_tokens, at_max_length = encoded_counts
        counts.update(
            {
                "examples": 1,
                "input_tokens": input_tokens,
                "target_tokens": target_tokens,
                "total_tokens": total_tokens,
                "examples_at_max_length": int(at_max_length),
            }
        )
    return counts


def _render_markdown(result: dict[str, Any]) -> str:
    interval = int(result["compaction_interval"])
    conditions = result["conditions"]
    lines = [
        "# Delta-v3 pending-depth sampling audit",
        "",
        (
            "Counts include independent examples and trajectory-window exposures. "
            "The JSON artifact retains per-seed and per-epoch details."
        ),
        "",
        "| Condition | Decision | "
        + " | ".join(f"d={depth}" for depth in range(interval))
        + " | Total |",
        "|---|---|" + "---:|" * (interval + 1),
    ]
    for condition in conditions:
        total = result["condition_reports"][condition]["aggregate_exposures"]["total"]
        for decision in ("NO_OP", "UPDATE"):
            values = [int(total[decision][str(depth)]) for depth in range(interval)]
            lines.append(
                f"| {condition} | {decision} | "
                + " | ".join(str(value) for value in values)
                + f" | {sum(values)} |"
            )
    if result["token_accounting"]["model_key"]:
        lines.extend(("", "## Encoded Memory-SFT token exposures", ""))
        lines.append(
            "| Condition | Examples | Input tokens | Target tokens | Total tokens | At max length |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for condition in conditions:
            tokens = result["condition_reports"][condition][
                "aggregate_memory_sft_tokens"
            ]
            lines.append(
                f"| {condition} | {tokens['examples']} | {tokens['input_tokens']} | "
                f"{tokens['target_tokens']} | {tokens['total_tokens']} | "
                f"{tokens['examples_at_max_length']} |"
            )
        unstratified_tokens = result["condition_reports"]["unstratified"][
            "aggregate_memory_sft_tokens"
        ]["total_tokens"]
        weighted_tokens = result["condition_reports"]["depth-weighted"][
            "aggregate_memory_sft_tokens"
        ]["total_tokens"]
        relative_change = (
            (weighted_tokens - unstratified_tokens) / unstratified_tokens
            if unstratified_tokens
            else 0.0
        )
        lines.extend(
            (
                "",
                (
                    "Depth-weighted total tokens differ by "
                    f"{relative_change:+.2%} from unstratified sampling; "
                    "the example count is held fixed."
                ),
            )
        )
    lines.extend(
        (
            "",
            (
                "The two conditions use identical UPDATE, total independent NO_OP, "
                "adjacent-NO_OP, trajectory, and Quiz budgets; only independent NO_OP "
                "selection uses pending depth in the depth-weighted condition."
            ),
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    result = run(build_parser().parse_args())
    summary = {
        condition: report["aggregate_exposures"]["total"]
        for condition, report in result["condition_reports"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
