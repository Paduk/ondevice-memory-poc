#!/usr/bin/env python3
"""Render paper-facing Composite/update-cycle latency tables."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-four-models.json"
OUTPUT = REPO_ROOT / "docs/engineering/results/stress-tradeoff-composite-latency-tables.md"
WORKLOADS = ("base", "20", "40", "60", "80")
LABELS = {"base": "Base (13.4)", "20": "U20", "40": "U40", "60": "U60", "80": "U80"}


def render_condition(lines: list[str], methods: dict, cache: str) -> None:
    metric = f"cache_{cache.lower()}_seconds_mean"
    lines.extend(
        [
            f"### Cache {cache}",
            "",
            "각 cell은 `Composite / update-cycle latency (sec)`이다.",
            "",
            "| 방법 | " + " | ".join(LABELS[w] for w in WORKLOADS) + " |",
            "|---|" + "---:|" * len(WORKLOADS),
        ]
    )
    for method, method_data in methods.items():
        cells = [
            f"{method_data[w]['composite']:.2f} / {method_data[w]['update_cycle'][metric]:.2f}"
            for w in WORKLOADS
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")


def main() -> None:
    data = json.loads(INPUT.read_text())
    lines = [
        "# Composite–Update-cycle Latency 요약",
        "",
        "`Update-cycle = UPDATE turn decode + 직후 turn prefill`이며, 오른쪽 값이 작을수록 "
        "빠르다. Cache OFF/ON의 차이는 직후 turn prefill에서만 발생한다. 모든 결과는 "
        "held-out stress subset S86–S90 5개 시나리오의 cycle 평균이다.",
        "",
    ]
    for model, model_data in data["models"].items():
        lines.extend(
            [
                f"## {model}",
                "",
                f"Prefill `{model_data['prefill_tokens_per_second']:.1f}` tok/s · "
                f"Decode `{model_data['decode_tokens_per_second']:.1f}` tok/s",
                "",
            ]
        )
        render_condition(lines, model_data["methods"], "OFF")
        render_condition(lines, model_data["methods"], "ON")

    OUTPUT.write_text("\n".join(lines) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
