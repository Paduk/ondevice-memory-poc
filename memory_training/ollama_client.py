"""Ollama chat client for structured memory generation and Tool Calling."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from .ubuntu_bridge import enable_ubuntu_runtime


@dataclass(frozen=True)
class OllamaChatResult:
    content: str
    tool_calls: tuple[dict[str, Any], ...]
    prompt_tokens: int
    output_tokens: int
    total_duration_ns: int
    load_duration_ns: int
    prompt_duration_ns: int
    output_duration_ns: int
    raw: Mapping[str, Any]

    @property
    def latency_seconds(self) -> float:
        return self.total_duration_ns / 1_000_000_000


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        timeout_seconds: float = 300,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def version(self) -> str:
        response = self.client.get("/api/version")
        response.raise_for_status()
        return str(response.json().get("version", "unknown"))

    def tags(self) -> list[dict[str, Any]]:
        response = self.client.get("/api/tags")
        response.raise_for_status()
        return list(response.json().get("models", []))

    def show(self, model: str) -> dict[str, Any]:
        response = self.client.post("/api/show", json={"model": model})
        response.raise_for_status()
        return dict(response.json())

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        json_mode: bool = False,
        think: bool | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        context_length: int = 4096,
        max_new_tokens: int = 768,
        keep_alive: str = "10m",
    ) -> OllamaChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_ctx": context_length,
                "num_predict": max_new_tokens,
            },
        }
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
        if json_mode:
            payload["format"] = "json"
        if think is not None:
            payload["think"] = think
        response = self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        value = response.json()
        message = value.get("message") or {}
        tool_calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            tool_calls.append(
                {
                    "id": str(
                        call.get("id") or f"ollama-{index}-{uuid.uuid4().hex[:8]}"
                    ),
                    "name": str(function.get("name") or ""),
                    "arguments": arguments,
                }
            )
        return OllamaChatResult(
            content=str(message.get("content") or ""),
            tool_calls=tuple(tool_calls),
            prompt_tokens=int(value.get("prompt_eval_count") or 0),
            output_tokens=int(value.get("eval_count") or 0),
            total_duration_ns=int(value.get("total_duration") or 0),
            load_duration_ns=int(value.get("load_duration") or 0),
            prompt_duration_ns=int(value.get("prompt_eval_duration") or 0),
            output_duration_ns=int(value.get("eval_duration") or 0),
            raw=value,
        )


class OllamaAgentModel:
    """PalmClaw AgentModel adapter backed by Ollama `/api/chat`."""

    backend = "ollama"
    prompt_version = "ollama-agent-v1"

    def __init__(
        self,
        client: OllamaClient,
        model_id: str,
        *,
        context_length: int = 8192,
        max_output_tokens: int = 1024,
        seed: int = 42,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.seed = seed

    def complete(self, messages: Sequence[Any], tools: Sequence[Any]) -> Any:
        enable_ubuntu_runtime()
        from palmclaw_ubuntu.models import AgentResponse, ModelUsage, ToolCall

        provider_messages = _ollama_messages(messages)
        provider_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                },
            }
            for tool in tools
        ]
        result = self.client.chat(
            model=self.model_id,
            messages=provider_messages,
            tools=provider_tools,
            think=False,
            context_length=self.context_length,
            max_new_tokens=self.max_output_tokens,
            seed=self.seed,
        )
        return AgentResponse(
            content=result.content,
            tool_calls=tuple(
                ToolCall(
                    id=str(call["id"]),
                    name=str(call["name"]),
                    arguments=dict(call["arguments"]),
                )
                for call in result.tool_calls
            ),
            usage=ModelUsage(
                input_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.prompt_tokens + result.output_tokens,
            ),
            response_id=str(result.raw.get("created_at") or "") or None,
            metadata={
                "model": self.model_id,
                "total_duration_ns": result.total_duration_ns,
                "load_duration_ns": result.load_duration_ns,
                "prompt_duration_ns": result.prompt_duration_ns,
                "output_duration_ns": result.output_duration_ns,
            },
        )


def _ollama_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    converted = []
    tool_names: dict[str, str] = {}
    for message in messages:
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = []
            for call in message.tool_calls:
                tool_names[call.id] = call.name
                item["tool_calls"].append(
                    {
                        "function": {
                            "name": call.name,
                            "arguments": dict(call.arguments),
                        }
                    }
                )
        if message.role == "tool" and message.tool_call_id:
            item["tool_name"] = tool_names.get(message.tool_call_id, "")
        converted.append(item)
    return converted
