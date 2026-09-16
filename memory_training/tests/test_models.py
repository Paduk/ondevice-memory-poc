from __future__ import annotations

from types import SimpleNamespace

import pytest

from memory_training.config import MODEL_BY_KEY
from memory_training.models import (
    _LLAMA_SINGLE_TOOL_BLOCK,
    configure_tokenizer_for_model,
)


def test_llama32_chat_template_supports_multiple_palmclaw_tool_calls() -> None:
    tokenizer = SimpleNamespace(
        chat_template=(
            'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.'
            + _LLAMA_SINGLE_TOOL_BLOCK
        )
    )

    configure_tokenizer_for_model(MODEL_BY_KEY["llama3.2-1b"], tokenizer)

    assert "This model only supports single tool-calls at once!" not in tokenizer.chat_template
    assert "for tool_call in message.tool_calls" in tokenizer.chat_template
    assert "<tool_call>" in tokenizer.chat_template
    assert '"arguments"' in tokenizer.chat_template
    assert '"parameters"' not in tokenizer.chat_template


def test_llama32_rejects_unknown_chat_template() -> None:
    tokenizer = SimpleNamespace(chat_template="upstream template changed")

    with pytest.raises(ValueError, match="unsupported Llama tool chat template"):
        configure_tokenizer_for_model(MODEL_BY_KEY["llama3.2-1b"], tokenizer)


def test_other_model_chat_template_is_unchanged() -> None:
    tokenizer = SimpleNamespace(chat_template="template")

    configure_tokenizer_for_model(MODEL_BY_KEY["granite4-1b"], tokenizer)

    assert tokenizer.chat_template == "template"
