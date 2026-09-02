from argparse import Namespace
from types import SimpleNamespace

from memory_training.evaluate_cloud_closed_loop import (
    _cost,
    _flatten_tools,
    _merge_usage,
    _usage,
)


def test_usage_splits_cached_input_tokens() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=40),
        )
    )

    assert _usage(response) == {
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "uncached_input_tokens": 60,
        "output_tokens": 20,
        "total_tokens": 120,
    }


def test_cost_uses_separate_cached_rate() -> None:
    args = Namespace(
        input_price_per_million=0.20,
        cached_input_price_per_million=0.02,
        output_price_per_million=1.20,
    )
    usage = {
        "uncached_input_tokens": 1_000_000,
        "cached_input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }

    assert _cost(usage, args) == 1.42


def test_merge_usage_includes_semantic_compaction_call() -> None:
    main = {
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "uncached_input_tokens": 80,
        "output_tokens": 10,
        "total_tokens": 110,
    }
    compaction = {
        "input_tokens": 50,
        "cached_input_tokens": 0,
        "uncached_input_tokens": 50,
        "output_tokens": 25,
        "total_tokens": 75,
    }

    assert _merge_usage(main, compaction) == {
        "input_tokens": 150,
        "cached_input_tokens": 20,
        "uncached_input_tokens": 130,
        "output_tokens": 35,
        "total_tokens": 185,
    }


def test_flatten_tools_preserves_on_device_schema() -> None:
    source = [
        {
            "type": "function",
            "function": {
                "name": "set_temperature",
                "description": "Set temperature",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    assert _flatten_tools(source) == [
        {
            "type": "function",
            "name": "set_temperature",
            "description": "Set temperature",
            "parameters": source[0]["function"]["parameters"],
            "strict": False,
        }
    ]
