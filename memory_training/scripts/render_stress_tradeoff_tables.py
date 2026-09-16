#!/usr/bin/env python3
"""Render compact cache ON/OFF stress trade-off tables from the aggregate JSON."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models.json"
OUTPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models-tables.md"
WORKLOADS = ("base", "20", "40", "60", "80")


def workload_label(workload: str) -> str:
    return "Base (13.4)" if workload == "base" else f"U{workload}"


def main() -> None:
    data = json.loads(INPUT.read_text())
    lines = [
        "# Stress trade-off 통합 테이블",
        "",
        "한 **update-cycle**은 `UPDATE turn의 decode 생성 + 직후 turn의 prefill`이다. "
        "NO-OP 등 다른 turn 비용은 포함하지 않는다. Prefill OFF는 직후 turn의 logical "
        "prompt 전체를 다시 계산하고, ON은 실제 cache replay의 evaluated prefill만 "
        "계산한다고 가정한다. 평균과 p95는 각 update-cycle에서 직접 계산했다.",
        "",
    ]
    for model, model_data in data["models"].items():
        methods = model_data["methods"]
        lines.extend(
            [
                f"## {model}",
                "",
                f"Throughput: prefill `{model_data['prefill_tokens_per_second']:.1f}` tok/s, "
                f"decode `{model_data['decode_tokens_per_second']:.1f}` tok/s",
                "",
                "| 방법 | 부하 | Composite | Update decode | Post-prefill OFF | Post-prefill ON | "
                "Decode lat | Prefill OFF lat | Prefill ON lat | Cycle OFF | Cycle ON | ON p95 | 절감 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                "| | | | tok/cycle | tok/cycle | tok/cycle | sec/cycle | sec/cycle | sec/cycle | sec/cycle | sec/cycle | sec/cycle | % |",
            ]
        )
        for method, method_data in methods.items():
            for workload in WORKLOADS:
                value = method_data[workload]
                cycle = value["update_cycle"]
                total_off_seconds = cycle["cache_off_seconds_mean"]
                total_on_seconds = cycle["cache_on_seconds_mean"]
                saving = 100 * (1 - total_on_seconds / total_off_seconds) if total_off_seconds else 0
                lines.append(
                    f"| {method} | {workload_label(workload)} | {value['composite']:.2f} | "
                    f"{cycle['update_decode_tokens_mean']:.1f} | "
                    f"{cycle['post_update_logical_prefill_tokens_mean']:.1f} | "
                    f"{cycle['post_update_evaluated_prefill_tokens_mean']:.1f} | "
                    f"{cycle['update_decode_seconds_mean']:.2f} | "
                    f"{cycle['post_update_cache_off_prefill_seconds_mean']:.2f} | "
                    f"{cycle['post_update_cache_on_prefill_seconds_mean']:.2f} | "
                    f"{total_off_seconds:.2f} | {total_on_seconds:.2f} | "
                    f"{cycle['cache_on_seconds_p95']:.2f} | {saving:.1f} |"
                )
        lines.append("")

    OUTPUT.write_text("\n".join(lines) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
