import json

import pytest

from memory_training.methods.temporal_patch import TemporalPatchMethod


def _operation(**overrides: str) -> dict[str, str]:
    value = {
        "op": "add",
        "target": "",
        "content": "- [2025-01-01 09:00] Maya: tool.setting; value=2",
        "identity_key": "maya.tool_setting",
        "temporal_action": "durable_upsert",
        "temporal_cue": "",
    }
    value.update(overrides)
    return value


def test_temporal_patch_method_applies_only_exact_block_fields() -> None:
    method = TemporalPatchMethod()
    encoded = json.dumps(
        {"decision": "UPDATE", "operations": [_operation()]},
        separators=(",", ":"),
    )

    parsed = method.parse_output(encoded)

    assert method.apply_output("", parsed) == _operation()["content"]


def test_temporal_patch_rejects_temporary_override_without_cue() -> None:
    method = TemporalPatchMethod()
    encoded = json.dumps(
        {
            "decision": "UPDATE",
            "operations": [
                _operation(temporal_action="temporary_override")
            ],
        }
    )

    with pytest.raises(ValueError, match="requires temporal_cue"):
        method.parse_output(encoded)


def test_temporal_patch_rejects_deleting_conditional_upsert() -> None:
    method = TemporalPatchMethod()
    encoded = json.dumps(
        {
            "decision": "UPDATE",
            "operations": [
                _operation(
                    op="delete",
                    target=_operation()["content"],
                    content="",
                    temporal_action="conditional_upsert",
                )
            ],
        }
    )

    with pytest.raises(ValueError, match="conditional_upsert cannot use delete"):
        method.parse_output(encoded)
