#!/usr/bin/env python3
"""Render same-condition Delta latency savings relative to Patch."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models.json"
OUTPUT = REPO_ROOT / "docs/engineering/results/patch-vs-delta-update-cycle-latency-savings.md"
MODELS = ("Granite 350M", "Qwen 0.8B", "Granite 1B", "Qwen 2B")
WORKLOADS = ("base", "20", "40", "60", "80")
DELTAS = ("k=2", "k=5", "k=10")
LABELS = {"base": "Base (13.4)", "20": "U20", "40": "U40", "60": "U60", "80": "U80"}


def saving(delta_latency: float, patch_latency: float) -> float:
    return 100 * (1 - delta_latency / patch_latency)


def format_saving(value: float, best: bool) -> str:
    if abs(value) < 0.05:
        value = 0.0
    rendered = f"{value:+.1f}%"
    return f"**{rendered}**" if best else rendered


def render_table(lines: list[str], data: dict, cache: str) -> None:
    metric = f"cache_{cache.lower()}_seconds_mean"
    lines.extend(
        [
            f"## Cache {cache}",
            "",
            "| 모델 | UPDATE | k=2 | k=5 | k=10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        methods = data[model]["methods"]
        for workload in WORKLOADS:
            patch = methods["Patch"][workload]["update_cycle"][metric]
            values = {
                method: saving(methods[method][workload]["update_cycle"][metric], patch)
                for method in DELTAS
            }
            best = max(values, key=values.get)
            lines.append(
                f"| {model} | {LABELS[workload]} | "
                + " | ".join(format_saving(values[method], method == best) for method in DELTAS)
                + " |"
            )
    lines.append("")


def main() -> None:
    data = json.loads(INPUT.read_text())["models"]
    lines = [
        "# Patch 대비 Delta Update-cycle Latency 절감률",
        "",
        "동일 모델·동일 UPDATE workload·동일 Cache 조건의 Patch를 기준으로 계산한다.",
        "",
        "```text",
        "Latency saving (%) = 100 × (1 - Delta update-cycle latency / Patch update-cycle latency)",
        "```",
        "",
        "양수는 Delta가 Patch보다 빠름, 음수는 느림을 뜻한다. 각 행의 최대 절감률을 "
        "굵게 표시했다. Update-cycle은 `UPDATE turn decode + 직후 turn prefill`이다.",
        "",
    ]
    render_table(lines, data, "OFF")
    render_table(lines, data, "ON")
    lines.extend(
        [
            "## 사용 원칙",
            "",
            "- 이 표는 latency만 비교한다. 최종 operating point는 별도의 Composite와 함께 고른다.",
            "- Cache OFF는 구조적 baseline, Cache ON은 실제 KV-cache 이득을 보여준다.",
            "- 결과 범위는 held-out stress subset S86–S90 5개 시나리오다.",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
