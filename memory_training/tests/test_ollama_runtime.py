from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from memory_training.dataset import DatasetCatalog, build_catalog
from memory_training.export_ollama import (
    _convert_and_quantize,
    _converter_arguments,
    _converter_supports,
)
from memory_training.methods import PatchMethod
from memory_training.ollama_client import (
    OllamaAgentModel,
    OllamaChatResult,
    OllamaClient,
)
from memory_training.ollama_test import run_memory_scenario
from memory_training.ubuntu_bridge import enable_ubuntu_runtime


def test_ollama_client_and_agent_tool_adapter() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "1.0.0"})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "created_at": "now",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "set_volume",
                                    "arguments": {"volume": 10},
                                }
                            }
                        ],
                    },
                    "prompt_eval_count": 20,
                    "eval_count": 5,
                    "total_duration": 1_000_000_000,
                },
            )
        raise AssertionError(request.url.path)

    client = OllamaClient(transport=httpx.MockTransport(handler))
    assert client.version() == "1.0.0"
    enable_ubuntu_runtime()
    from palmclaw_ubuntu.models import ChatMessage, ToolDefinition

    agent = OllamaAgentModel(client, "test:latest")
    response = agent.complete(
        [ChatMessage(role="user", content="set it")],
        [ToolDefinition("set_volume", "Set volume", {"type": "object"}, 5)],
    )
    assert response.tool_calls[0].arguments == {"volume": 10}
    assert response.usage.total_tokens == 25
    payload = json.loads(requests[-1].content)
    assert payload["options"]["temperature"] == 0
    assert payload["think"] is False
    assert payload["tools"][0]["function"]["name"] == "set_volume"


class FakeMemoryClient:
    def __init__(self) -> None:
        self.outputs = iter(
            (
                '{"decision":"NO_OP"}',
                '{"decision":"UPDATE","operations":[{"op":"add","target":"","content":"- fact"}]}',
            )
        )

    def chat(self, **kwargs) -> OllamaChatResult:
        del kwargs
        return OllamaChatResult(
            content=next(self.outputs),
            tool_calls=(),
            prompt_tokens=10,
            output_tokens=5,
            total_duration_ns=100,
            load_duration_ns=0,
            prompt_duration_ns=40,
            output_duration_ns=60,
            raw={},
        )


def test_closed_loop_memory_creates_quiz_snapshot(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for turn, decision in enumerate(("NO_OP", "UPDATE")):
        common = {
            "sample_id": f"s091:sample:{turn}",
            "scenario_index": 91,
            "split": "test",
            "global_turn_index": turn,
            "turn_id": f"turn-{turn}",
            "timestamp": f"2026-01-01T00:0{turn}",
            "current_turn": {"speaker_name": "Alex", "text": f"turn {turn}"},
        }
        rows.append((common, decision))
    for view in ("summary", "patch", "delta"):
        with (data / f"{view}.jsonl").open("w") as handle:
            for common, decision in rows:
                turn = common["global_turn_index"]
                target = {"decision": decision}
                if view == "summary":
                    target["next_memory"] = "" if turn == 0 else "- fact"
                    memory_input = {"previous_memory": ""}
                elif view == "patch":
                    target["operations"] = (
                        []
                        if turn == 0
                        else [{"op": "add", "target": "", "content": "- fact"}]
                    )
                    memory_input = {"previous_memory": ""}
                else:
                    target["operations"] = []
                    memory_input = {
                        "base_summary": "",
                        "pending_deltas": [],
                        "updates_since_compaction": 0,
                    }
                handle.write(
                    json.dumps({**common, "input": memory_input, "target": target})
                    + "\n"
                )
    (data / "manifest.json").write_text("{}")
    (data / "quiz_manifest.json").write_text("{}")
    catalog_path = tmp_path / "catalog.sqlite"
    build_catalog(data, catalog_path)
    catalog = DatasetCatalog(catalog_path, data)
    quizzes = [
        {
            "sample_id": "quiz-sample",
            "memory_ref": {"global_turn_index": 1},
        }
    ]
    summary = run_memory_scenario(
        FakeMemoryClient(),
        PatchMethod(),
        catalog,
        91,
        quizzes,
        output_dir=tmp_path / "output",
        model="fake",
        signature="signature",
        seed=42,
        context_length=1024,
        max_new_tokens=64,
        checkpoint_interval=1,
        turn_limit=None,
        force=False,
    )
    assert summary["final_state"]["exact"] == 1.0
    assert summary["quiz_snapshots"]["quiz-sample"] == "- fact"
    assert summary["metrics"]["prefill_tokens"] == 20


def test_qwen35_uses_isolated_converter_without_mtp(tmp_path) -> None:
    llama_cpp = tmp_path / "llama.cpp"
    conversion = llama_cpp / "conversion"
    conversion.mkdir(parents=True)
    (llama_cpp / "convert_hf_to_gguf.py").write_text("# modular converter\n")
    (conversion / "qwen.py").write_text(
        '@register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")\n'
    )
    assert _converter_supports(llama_cpp, "Qwen3_5ForConditionalGeneration")
    assert _converter_supports(llama_cpp, "Qwen3_5ForCausalLM")
    assert _converter_arguments("Qwen3_5ForConditionalGeneration") == ("--no-mtp",)
    assert _converter_arguments("GraniteForCausalLM") == ()


def test_bf16_export_skips_quantizer(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(command, *, check):
        assert check
        calls.append(command)

    monkeypatch.setattr("memory_training.export_ollama.subprocess.run", fake_run)
    gguf = tmp_path / "model-bf16.gguf"
    plan = SimpleNamespace(
        converter="convert_hf_to_gguf.py",
        merged_dir=str(tmp_path / "merged"),
        f16_gguf=str(gguf),
        gguf_outtype="bf16",
        converter_arguments=("--no-mtp",),
        quantization="NONE",
        quantizer="llama-quantize",
        quantized_gguf=str(gguf),
        modelfile=str(tmp_path / "Modelfile"),
        context_length=8192,
    )

    _convert_and_quantize(plan, threads=8)

    assert len(calls) == 1
    assert calls[0][calls[0].index("--outtype") + 1] == "bf16"
    assert Path(plan.modelfile).read_text().startswith(f"FROM {gguf}\n")
