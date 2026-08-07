import pytest

from palmclaw_ubuntu.config import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MEMORY_MODEL,
    Settings,
)


def test_phase_one_cloud_models_are_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("PALMCLAW_AGENT_MODEL", raising=False)
    monkeypatch.delenv("PALMCLAW_MEMORY_MODEL", raising=False)
    settings = Settings.from_env(data_dir=tmp_path)
    assert settings.agent_model == DEFAULT_AGENT_MODEL
    assert settings.memory_model == DEFAULT_MEMORY_MODEL
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.memory_strategy == "structured"
    assert settings.retrieval_mode == "hybrid"


def test_cloud_models_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("PALMCLAW_AGENT_MODEL", "agent-override")
    monkeypatch.setenv("PALMCLAW_MEMORY_MODEL", "memory-override")
    settings = Settings.from_env(data_dir=tmp_path)
    assert settings.agent_model == "agent-override"
    assert settings.memory_model == "memory-override"


def test_local_memory_backend_can_be_selected_independently(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("PALMCLAW_BACKEND", "openai")
    monkeypatch.setenv("PALMCLAW_MEMORY_BACKEND", "local")
    monkeypatch.setenv("PALMCLAW_LOCAL_MEMORY_MODEL", "qwen-memory")
    monkeypatch.setenv(
        "PALMCLAW_LOCAL_MEMORY_BASE_URL",
        "http://localhost:8080/v1",
    )

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.backend == "openai"
    assert settings.memory_backend == "local"
    assert settings.local_memory_model == "qwen-memory"
    assert settings.local_memory_base_url == "http://localhost:8080/v1"


def test_fact_memory_cloud_poc_can_be_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("PALMCLAW_FACT_MEMORY_ENABLED", "1")
    monkeypatch.setenv("PALMCLAW_FACT_MEMORY_MODEL", "fact-model")

    settings = Settings.from_env(data_dir=tmp_path)

    assert settings.fact_memory_enabled is True
    assert settings.fact_memory_model == "fact-model"
    assert settings.fact_memory_linking_context_limit == 24


def test_joint_memory_tool_planning_is_opt_in_and_bounded(tmp_path, monkeypatch):
    defaults = Settings.from_env(data_dir=tmp_path / "defaults")
    assert defaults.joint_memory_tool_planning_enabled is False

    monkeypatch.setenv("PALMCLAW_JOINT_MEMORY_TOOL_PLANNING_ENABLED", "1")
    monkeypatch.setenv("PALMCLAW_JOINT_MEMORY_TOOL_MAX_TOOLS", "7")
    monkeypatch.setenv("PALMCLAW_JOINT_MEMORY_TOOL_SCHEMA_TOKENS", "900")
    enabled = Settings.from_env(data_dir=tmp_path / "enabled")

    assert enabled.joint_memory_tool_planning_enabled is True
    assert enabled.joint_memory_tool_max_tools == 7
    assert enabled.joint_memory_tool_schema_tokens == 900


def test_fact_and_schema_patch_workers_are_mutually_exclusive(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("PALMCLAW_FACT_MEMORY_ENABLED", "1")
    monkeypatch.setenv("PALMCLAW_PATCH_MEMORY_ENABLED", "1")

    with pytest.raises(ValueError, match="mutually exclusive"):
        Settings.from_env(data_dir=tmp_path)
