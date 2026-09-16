from __future__ import annotations

from memory_training.scripts.prepare_patch_delta_disagreement_analysis import (
    _output_symptoms,
)


def _call(name: str, **arguments: object) -> dict:
    return {"name": name, "arguments": arguments}


def test_output_symptoms_distinguish_wrong_argument() -> None:
    gold = [_call("set_temperature", temperature=22)]
    predicted = [_call("set_temperature", temperature=24)]
    assert _output_symptoms(predicted, gold) == ["WRONG_ARGUMENT"]


def test_output_symptoms_are_multilabel_for_extra_wrong_tool() -> None:
    gold = [_call("set_temperature", temperature=22)]
    predicted = [
        _call("set_temperature", temperature=24),
        _call("set_volume", volume=10),
    ]
    assert _output_symptoms(predicted, gold) == [
        "WRONG_TOOL",
        "OVER_ACTION",
        "WRONG_ARGUMENT",
    ]
