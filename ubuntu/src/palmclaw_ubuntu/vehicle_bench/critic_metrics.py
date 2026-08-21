from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from palmclaw_ubuntu.vehicle_bench.critic_pipeline import (
    CriticCheckpointStore,
)
from palmclaw_ubuntu.vehicle_bench.critic_trace import CriticSourceTurn

CRITIC_METRICS_VERSION = "vehiclemembench-v2-critic-metrics-v1"

MODEL_PRICES_PER_MILLION = {
    "luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    "terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
}


def build_critic_metrics(
    store: CriticCheckpointStore,
    *,
    turns: tuple[CriticSourceTurn, ...],
) -> dict[str, Any]:
    state = store.state()
    active = store.active_results()
    attempts = store.attempts()
    source_by_index = {turn.turn_index: turn for turn in turns}

    verdicts: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    for label in active:
        response = label["result"]["response"]
        verdict = str(response["verdict"])
        decision = str(response["decision"])
        reason_code = str(response["reason_code"])
        verdicts[verdict] += 1
        decisions[decision] += 1
        reasons[reason_code] += 1
        stages[str(label["stage"])] += 1
        original = source_by_index[int(label["turn_index"])].original_decision
        confusion[f"{original}->{decision}"] += 1

    attempt_stages: Counter[str] = Counter()
    model_aggregates = {
        model: {
            "calls": 0,
            "logical_attempts": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
        }
        for model in ("luna", "terra")
    }
    for attempt in attempts:
        stage = str(attempt["stage"])
        attempt_stages[stage] += 1
        model = "luna" if stage == "luna" else "terra"
        usage, latency_ms, api_calls = _attempt_usage_and_latency(attempt)
        aggregate = model_aggregates[model]
        aggregate["calls"] += api_calls
        aggregate["logical_attempts"] += 1
        aggregate["input_tokens"] += usage["input_tokens"]
        aggregate["cached_input_tokens"] += usage["cached_tokens"]
        aggregate["output_tokens"] += usage["output_tokens"]
        aggregate["total_tokens"] += usage["total_tokens"]
        aggregate["latency_ms"] += latency_ms

    total_cost = 0.0
    for model, aggregate in model_aggregates.items():
        prices = MODEL_PRICES_PER_MILLION[model]
        cached = min(
            aggregate["cached_input_tokens"], aggregate["input_tokens"]
        )
        uncached = aggregate["input_tokens"] - cached
        cost = (
            uncached * prices["input"]
            + cached * prices["cached_input"]
            + aggregate["output_tokens"] * prices["output"]
        ) / 1_000_000
        aggregate["estimated_cost_usd"] = round(cost, 6)
        aggregate["latency_seconds"] = round(
            aggregate["latency_ms"] / 1_000,
            3,
        )
        total_cost += cost

    total_usage = {
        key: sum(int(item[key]) for item in model_aggregates.values())
        for key in (
            "calls",
            "logical_attempts",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "total_tokens",
            "latency_ms",
        )
    }
    total_usage["latency_seconds"] = round(total_usage["latency_ms"] / 1_000, 3)
    total_usage["estimated_cost_usd"] = round(total_cost, 6)
    processed = state.next_turn_index
    scale = len(turns) / processed if processed else None
    extrapolation = None
    if scale is not None:
        extrapolation = {
            "scale_from_processed_turns": round(scale, 6),
            "estimated_full_calls": round(total_usage["calls"] * scale),
            "estimated_full_input_tokens": round(
                total_usage["input_tokens"] * scale
            ),
            "estimated_full_output_tokens": round(
                total_usage["output_tokens"] * scale
            ),
            "estimated_full_cost_usd": round(total_cost * scale, 4),
            "estimated_full_provider_latency_seconds": round(
                total_usage["latency_ms"] / 1_000 * scale,
                1,
            ),
            "warning": "Linear smoke extrapolation; REJECT/rollback rates may vary.",
        }

    return {
        "version": CRITIC_METRICS_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "method": state.method,
        "scenario_index": state.scenario_index,
        "status": state.status,
        "turn_count": state.turn_count,
        "processed_turns": processed,
        "active_label_count": len(active),
        "replay_generation": state.replay_generation,
        "final_memory_chars": len(state.approved_memory),
        "final_memory_sha256": hashlib.sha256(
            state.approved_memory.encode("utf-8")
        ).hexdigest(),
        "labels": {
            "verdicts": dict(sorted(verdicts.items())),
            "decisions": dict(sorted(decisions.items())),
            "reason_codes": dict(sorted(reasons.items())),
            "stages": dict(sorted(stages.items())),
            "original_to_final": dict(sorted(confusion.items())),
        },
        "recovery": {
            "terra_escalations": attempt_stages["terra_initial"],
            "context_recoveries": attempt_stages["terra_recovered"],
            "rollback_repairs": attempt_stages["terra_rollback_repair"],
            "rollback_events": len(store.rollback_events()),
            "unresolved_events": len(store.unresolved_events()),
            "unresolved_history": len(store.unresolved_history()),
            "defer_keep_labels": verdicts["DEFER_KEEP"],
        },
        "attempt_stages": dict(sorted(attempt_stages.items())),
        "models": model_aggregates,
        "total": total_usage,
        "full_run_extrapolation": extrapolation,
    }


def _attempt_usage_and_latency(
    attempt: Mapping[str, Any],
) -> tuple[dict[str, int], int, int]:
    stage = str(attempt["stage"])
    payload = attempt["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("Critic attempt payload is not an object")
    if stage in {"luna", "terra_rollback_repair"}:
        response = payload.get("response")
        if not isinstance(response, Mapping):
            raise ValueError("Critic result attempt is missing response")
        usage = response.get("usage")
        metadata = response.get("metadata")
    else:
        adjudication = (
            payload.get("adjudication")
            if stage == "terra_recovered"
            else payload
        )
        if not isinstance(adjudication, Mapping):
            raise ValueError("Terra attempt is missing adjudication")
        usage = adjudication.get("usage")
        metadata = adjudication.get("metadata")
    if not isinstance(usage, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("Critic attempt usage/metadata is missing")
    normalized_usage = {
        key: int(usage.get(key, 0) or 0)
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
        )
    }
    semantic_retries = int(metadata.get("semantic_retry_count", 0) or 0)
    return (
        normalized_usage,
        int(metadata.get("latency_ms", 0) or 0),
        1 + semantic_retries,
    )
