"""Combine compatible HF prefix-cache benchmark summaries into one report."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "palmclaw-hf-prefix-cache-comparison-v1"
VIEWS = {
    "all_turns": ("all_turns",),
    "gold_update": ("by_gold_decision", "UPDATE"),
    "predicted_update": ("by_predicted_decision", "UPDATE"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Benchmark summary.json; pass once per method/cache mode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--projected-prefill-tokens-per-second",
        type=float,
        help="Optional on-device prefill rate used only for a projected latency column.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.projected_prefill_tokens_per_second is not None
        and args.projected_prefill_tokens_per_second <= 0
    ):
        raise ValueError("Projected prefill throughput must be positive")
    summaries = [_read_json(path.resolve()) for path in args.input]
    comparison = build_comparison(
        summaries,
        projected_prefill_tokens_per_second=(
            args.projected_prefill_tokens_per_second
        ),
    )
    output_dir = args.output_dir.resolve()
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    if not args.force and (json_path.exists() or markdown_path.exists()):
        raise FileExistsError(f"Comparison output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        json_path,
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
    )
    _write_text(markdown_path, render_markdown(comparison))
    return comparison


def build_comparison(
    summaries: Sequence[Mapping[str, Any]],
    *,
    projected_prefill_tokens_per_second: float | None = None,
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("At least one benchmark summary is required")
    compatibility = _compatibility_signature(summaries[0])
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for summary in summaries:
        if _compatibility_signature(summary) != compatibility:
            raise ValueError("Benchmark summaries do not share one test configuration")
        identity = (str(summary["method"]), _cache_variant(summary))
        if identity in indexed:
            raise ValueError(f"Duplicate benchmark summary: {identity}")
        indexed[identity] = summary

    rows_by_view = {}
    for view_name, path in VIEWS.items():
        rows = []
        for (method, cache_variant), summary in sorted(indexed.items()):
            metrics = _nested(summary["metrics"], path)
            row = _comparison_row(
                method,
                cache_variant,
                metrics,
                projected_prefill_tokens_per_second=(
                    projected_prefill_tokens_per_second
                ),
            )
            off = indexed.get((method, "off"))
            if str(summary["cache_mode"]) == "on" and off is not None:
                off_metrics = _nested(off["metrics"], path)
                row.update(_savings(metrics, off_metrics))
            rows.append(row)
        rows_by_view[view_name] = rows

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        **compatibility,
        "projected_prefill_tokens_per_second": (
            projected_prefill_tokens_per_second
        ),
        "views": rows_by_view,
    }


def render_markdown(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# HF Prefix-cache Efficiency Comparison",
        "",
        f"- Model: `{comparison['model']}`",
        f"- Replay: `{comparison['replay_mode']}`",
        f"- Manifest: `{comparison['turn_manifest_signature']}`",
    ]
    rate = comparison.get("projected_prefill_tokens_per_second")
    if rate is not None:
        lines.append(f"- Projected on-device prefill: `{rate:g} token/s`")
    for view_name in VIEWS:
        lines.extend(("", f"## {view_name}", ""))
        headers = [
            "Method",
            "Cache",
            "Turns",
            "Prefill tok",
            "BG prefill tok",
            "Decode tok",
            "Cost tok",
            "Reuse",
            "Prefill ms",
            "Decode ms",
            "TTFT ms",
            "E2E ms",
            "Token saving",
        ]
        if rate is not None:
            headers.append("Projected prefill ms")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        for row in comparison["views"][view_name]:
            values = [
                row["method"],
                row["cache_mode"],
                str(row["turns"]),
                _number(row["evaluated_prefill_tokens_mean"]),
                _number(row["background_prefill_tokens_mean"]),
                _number(row["decode_tokens_mean"]),
                _number(row["model_cost_tokens_mean"]),
                _percent(row["cache_reuse_ratio"]),
                _milliseconds(row["prefill_seconds_mean"]),
                _milliseconds(row["decode_seconds_mean"]),
                _milliseconds(row["ttft_seconds_mean"]),
                _milliseconds(row["end_to_end_seconds_mean"]),
                _optional_percent(row.get("model_cost_token_reduction_vs_off")),
            ]
            if rate is not None:
                values.append(
                    _milliseconds(row["projected_prefill_seconds_mean"])
                )
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _comparison_row(
    method: str,
    cache_mode: str,
    metrics: Mapping[str, Any],
    *,
    projected_prefill_tokens_per_second: float | None,
) -> dict[str, Any]:
    prefill_tokens = float(metrics["evaluated_prefill_tokens"]["mean"])
    background_prefill_tokens = float(
        metrics.get("background_prefill_tokens", {"mean": 0.0})["mean"]
    )
    decode_tokens = float(metrics["decode_tokens"]["mean"])
    total_model_tokens = float(
        metrics.get(
            "total_model_tokens",
            {"mean": prefill_tokens + decode_tokens},
        )["mean"]
    )
    row = {
        "method": method,
        "cache_mode": cache_mode,
        "turns": int(metrics["turns"]),
        "evaluated_prefill_tokens_mean": prefill_tokens,
        "background_prefill_tokens_mean": background_prefill_tokens,
        "decode_tokens_mean": decode_tokens,
        "model_cost_tokens_mean": total_model_tokens,
        "cache_reuse_ratio": float(metrics["cache_reuse_ratio"]),
        "prefill_seconds_mean": float(metrics["prefill_seconds"]["mean"]),
        "decode_seconds_mean": float(metrics["decode_seconds"]["mean"]),
        "ttft_seconds_mean": float(metrics["ttft_seconds"]["mean"]),
        "end_to_end_seconds_mean": float(metrics["end_to_end_seconds"]["mean"]),
        "kv_cache_bytes_mean": float(metrics["kv_cache_bytes"]["mean"]),
        "errors": int(metrics["errors"]),
    }
    if projected_prefill_tokens_per_second is not None:
        row["projected_prefill_seconds_mean"] = (
            prefill_tokens / projected_prefill_tokens_per_second
        )
    return row


def _savings(
    cache_on: Mapping[str, Any], cache_off: Mapping[str, Any]
) -> dict[str, float]:
    on_prefill = float(cache_on["evaluated_prefill_tokens"]["mean"])
    off_prefill = float(cache_off["evaluated_prefill_tokens"]["mean"])
    on_cost = float(
        cache_on.get(
            "total_model_tokens",
            {"mean": on_prefill + float(cache_on["decode_tokens"]["mean"])},
        )["mean"]
    )
    off_cost = float(
        cache_off.get(
            "total_model_tokens",
            {"mean": off_prefill + float(cache_off["decode_tokens"]["mean"])},
        )["mean"]
    )
    return {
        "prefill_token_reduction_vs_off": _reduction(on_prefill, off_prefill),
        "model_cost_token_reduction_vs_off": _reduction(on_cost, off_cost),
        "prefill_latency_reduction_vs_off": _reduction(
            float(cache_on["prefill_seconds"]["mean"]),
            float(cache_off["prefill_seconds"]["mean"]),
        ),
        "end_to_end_latency_reduction_vs_off": _reduction(
            float(cache_on["end_to_end_seconds"]["mean"]),
            float(cache_off["end_to_end_seconds"]["mean"]),
        ),
    }


def _compatibility_signature(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "model",
            "turn_manifest_signature",
            "replay_mode",
            "max_length",
            "max_new_tokens",
            "repetitions",
        )
    }


def _cache_variant(summary: Mapping[str, Any]) -> str:
    if summary["cache_mode"] == "on" and summary.get("background_prefill"):
        return "on+background"
    return str(summary["cache_mode"])


def _nested(value: Any, path: Sequence[str]) -> Mapping[str, Any]:
    for key in path:
        value = value[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected metric object at {'.'.join(path)}")
    return value


def _reduction(candidate: float, baseline: float) -> float:
    return 1.0 - candidate / baseline if baseline else 0.0


def _number(value: float) -> str:
    return f"{value:.2f}"


def _milliseconds(seconds: float) -> str:
    return f"{seconds * 1000:.2f}"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _optional_percent(value: float | None) -> str:
    return "-" if value is None else _percent(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "model": result["model"],
                "views": list(result["views"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
