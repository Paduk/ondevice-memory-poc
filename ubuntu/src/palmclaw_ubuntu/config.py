from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_AGENT_MODEL = "gpt-5.6-terra"
DEFAULT_MEMORY_MODEL = "gpt-5.6-luna"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_MEMORY_MODEL = "local-memory"
DEFAULT_LOCAL_MEMORY_BASE_URL = "http://127.0.0.1:8080/v1"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_path: Path
    workspace_root: Path
    shared_workspace_root: Path
    builtin_skills_root: Path
    workspace_skills_root: Path
    backend: str = "openai"
    agent_model: str | None = None
    memory_model: str | None = None
    embedding_model: str | None = None
    memory_backend: str = "auto"
    local_memory_model: str = DEFAULT_LOCAL_MEMORY_MODEL
    local_memory_base_url: str = DEFAULT_LOCAL_MEMORY_BASE_URL
    local_memory_api_key: str = "local"
    memory_strategy: str = "structured"
    retrieval_mode: str = "hybrid"
    memory_top_k: int = 5
    embedding_dimensions: int = 256
    agent_reasoning_effort: str = "low"
    memory_reasoning_effort: str = "low"
    agent_max_output_tokens: int = 2_048
    memory_max_output_tokens: int = 1_024
    model_timeout_seconds: float = 60.0
    tool_timeout_seconds: float = 15.0
    max_tool_rounds: int = 8
    max_concurrent_sessions: int = 4
    max_history_messages: int = 40
    max_context_tokens: int = 16_000
    memory_context_tokens: int = 2_000
    skill_context_tokens: int = 3_000
    max_tool_result_chars: int = 20_000
    memory_trigger_messages: int = 6
    max_file_bytes: int = 1_000_000
    max_web_bytes: int = 500_000
    max_web_redirects: int = 3
    memory_gate_enabled: bool = True
    memory_min_confidence: float = 0.65
    memory_global_min_confidence: float = 0.85
    memory_review_high_sensitivity: bool = True
    cloud_pii_redaction: bool = True
    local_pii_storage: str = "redacted"
    pii_allowlist: tuple[str, ...] = ()
    agent_input_cost_per_million: float = 0.0
    agent_output_cost_per_million: float = 0.0
    memory_input_cost_per_million: float = 0.0
    memory_output_cost_per_million: float = 0.0
    embedding_input_cost_per_million: float = 0.0
    patch_memory_enabled: bool = False
    patch_memory_backend: str = "auto"
    patch_memory_model: str | None = None
    patch_memory_user_id: str = "default_user"
    patch_memory_batch_size: int = 32
    patch_memory_batch_tokens: int = 4_096
    patch_memory_max_attempts: int = 3
    patch_memory_retry_delay_seconds: float = 30.0
    patch_memory_lease_seconds: float = 180.0
    fact_memory_enabled: bool = False
    fact_memory_backend: str = "auto"
    fact_memory_model: str | None = None
    fact_memory_user_id: str = "default_user"
    fact_memory_linking_context_limit: int = 24
    tool_memory_retrieval_enabled: bool = False
    tool_memory_retrieval_mode: str = "hybrid"
    tool_memory_top_k: int = 5
    tool_memory_context_tokens: int = 1_000
    tool_memory_max_routes: int = 3
    tool_memory_vehicle_id: str | None = None
    tool_memory_semantic_routing_enabled: bool = True
    joint_memory_tool_planning_enabled: bool = False
    joint_memory_tool_max_tools: int = 12
    joint_memory_tool_schema_tokens: int = 1_800

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> Settings:
        package_root = Path(__file__).resolve().parent
        default_data_dir = Path(__file__).resolve().parents[2] / ".data"
        resolved_data_dir = (
            Path(data_dir or os.getenv("PALMCLAW_DATA_DIR", default_data_dir))
            .expanduser()
            .resolve()
        )
        workspace_root = (
            Path(os.getenv("PALMCLAW_WORKSPACE_ROOT", resolved_data_dir / "workspaces"))
            .expanduser()
            .resolve()
        )
        shared_workspace_root = (
            Path(
                os.getenv(
                    "PALMCLAW_SHARED_WORKSPACE_ROOT",
                    resolved_data_dir / "shared",
                )
            )
            .expanduser()
            .resolve()
        )
        workspace_skills_root = (
            Path(
                os.getenv(
                    "PALMCLAW_SKILLS_ROOT",
                    resolved_data_dir / "skills",
                )
            )
            .expanduser()
            .resolve()
        )

        backend = os.getenv("PALMCLAW_BACKEND", "openai").strip().lower()
        if backend not in {"openai", "fake"}:
            raise ValueError("PALMCLAW_BACKEND must be 'openai' or 'fake'")

        agent_model = os.getenv(
            "PALMCLAW_AGENT_MODEL",
            DEFAULT_AGENT_MODEL,
        ).strip()
        memory_model = os.getenv(
            "PALMCLAW_MEMORY_MODEL",
            DEFAULT_MEMORY_MODEL,
        ).strip()
        embedding_model = os.getenv(
            "PALMCLAW_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        ).strip()
        patch_memory_enabled = _env_bool(
            "PALMCLAW_PATCH_MEMORY_ENABLED",
            False,
        )
        fact_memory_enabled = _env_bool(
            "PALMCLAW_FACT_MEMORY_ENABLED",
            False,
        )

        settings = cls(
            data_dir=resolved_data_dir,
            database_path=resolved_data_dir / "palmclaw.db",
            workspace_root=workspace_root,
            shared_workspace_root=shared_workspace_root,
            builtin_skills_root=package_root / "builtin_skills",
            workspace_skills_root=workspace_skills_root,
            backend=backend,
            agent_model=agent_model,
            memory_model=memory_model,
            embedding_model=embedding_model,
            memory_backend=os.getenv(
                "PALMCLAW_MEMORY_BACKEND",
                backend,
            )
            .strip()
            .lower(),
            local_memory_model=os.getenv(
                "PALMCLAW_LOCAL_MEMORY_MODEL",
                DEFAULT_LOCAL_MEMORY_MODEL,
            ).strip(),
            local_memory_base_url=os.getenv(
                "PALMCLAW_LOCAL_MEMORY_BASE_URL",
                DEFAULT_LOCAL_MEMORY_BASE_URL,
            ).strip(),
            local_memory_api_key=os.getenv(
                "PALMCLAW_LOCAL_MEMORY_API_KEY",
                "local",
            ).strip(),
            memory_strategy=os.getenv(
                "PALMCLAW_MEMORY_STRATEGY",
                "structured",
            )
            .strip()
            .lower(),
            retrieval_mode=os.getenv(
                "PALMCLAW_RETRIEVAL_MODE",
                "hybrid",
            )
            .strip()
            .lower(),
            memory_top_k=_env_int("PALMCLAW_MEMORY_TOP_K", 5),
            embedding_dimensions=_env_int(
                "PALMCLAW_EMBEDDING_DIMENSIONS",
                256,
            ),
            agent_reasoning_effort=os.getenv(
                "PALMCLAW_AGENT_REASONING_EFFORT",
                "low",
            ).strip(),
            memory_reasoning_effort=os.getenv(
                "PALMCLAW_MEMORY_REASONING_EFFORT",
                "low",
            ).strip(),
            agent_max_output_tokens=_env_int(
                "PALMCLAW_AGENT_MAX_OUTPUT_TOKENS",
                2_048,
            ),
            memory_max_output_tokens=_env_int(
                "PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS",
                1_024,
            ),
            model_timeout_seconds=_env_float(
                "PALMCLAW_MODEL_TIMEOUT_SECONDS",
                60.0,
            ),
            tool_timeout_seconds=_env_float(
                "PALMCLAW_TOOL_TIMEOUT_SECONDS",
                15.0,
            ),
            max_tool_rounds=_env_int("PALMCLAW_MAX_TOOL_ROUNDS", 8),
            max_concurrent_sessions=_env_int(
                "PALMCLAW_MAX_CONCURRENT_SESSIONS",
                4,
            ),
            max_history_messages=_env_int(
                "PALMCLAW_MAX_HISTORY_MESSAGES",
                40,
            ),
            max_context_tokens=_env_int(
                "PALMCLAW_MAX_CONTEXT_TOKENS",
                16_000,
            ),
            memory_context_tokens=_env_int(
                "PALMCLAW_MEMORY_CONTEXT_TOKENS",
                2_000,
            ),
            skill_context_tokens=_env_int(
                "PALMCLAW_SKILL_CONTEXT_TOKENS",
                3_000,
            ),
            max_tool_result_chars=_env_int(
                "PALMCLAW_MAX_TOOL_RESULT_CHARS",
                20_000,
            ),
            memory_trigger_messages=_env_int(
                "PALMCLAW_MEMORY_TRIGGER_MESSAGES",
                6,
            ),
            max_file_bytes=_env_int("PALMCLAW_MAX_FILE_BYTES", 1_000_000),
            max_web_bytes=_env_int("PALMCLAW_MAX_WEB_BYTES", 500_000),
            max_web_redirects=_env_int("PALMCLAW_MAX_WEB_REDIRECTS", 3),
            memory_gate_enabled=_env_bool(
                "PALMCLAW_MEMORY_GATE_ENABLED",
                True,
            ),
            memory_min_confidence=_env_float(
                "PALMCLAW_MEMORY_MIN_CONFIDENCE",
                0.65,
            ),
            memory_global_min_confidence=_env_float(
                "PALMCLAW_MEMORY_GLOBAL_MIN_CONFIDENCE",
                0.85,
            ),
            memory_review_high_sensitivity=_env_bool(
                "PALMCLAW_MEMORY_REVIEW_HIGH_SENSITIVITY",
                True,
            ),
            cloud_pii_redaction=_env_bool(
                "PALMCLAW_CLOUD_PII_REDACTION",
                True,
            ),
            local_pii_storage=os.getenv(
                "PALMCLAW_LOCAL_PII_STORAGE",
                "redacted",
            )
            .strip()
            .lower(),
            pii_allowlist=tuple(
                item.strip()
                for item in os.getenv("PALMCLAW_PII_ALLOWLIST", "").split(",")
                if item.strip()
            ),
            agent_input_cost_per_million=_env_float(
                "PALMCLAW_AGENT_INPUT_COST_PER_MILLION",
                0.0,
            ),
            agent_output_cost_per_million=_env_float(
                "PALMCLAW_AGENT_OUTPUT_COST_PER_MILLION",
                0.0,
            ),
            memory_input_cost_per_million=_env_float(
                "PALMCLAW_MEMORY_INPUT_COST_PER_MILLION",
                0.0,
            ),
            memory_output_cost_per_million=_env_float(
                "PALMCLAW_MEMORY_OUTPUT_COST_PER_MILLION",
                0.0,
            ),
            embedding_input_cost_per_million=_env_float(
                "PALMCLAW_EMBEDDING_INPUT_COST_PER_MILLION",
                0.0,
            ),
            patch_memory_enabled=patch_memory_enabled,
            patch_memory_backend=os.getenv(
                "PALMCLAW_PATCH_MEMORY_BACKEND",
                "auto",
            )
            .strip()
            .lower(),
            patch_memory_model=os.getenv(
                "PALMCLAW_PATCH_MEMORY_MODEL",
                memory_model,
            ).strip(),
            patch_memory_user_id=os.getenv(
                "PALMCLAW_PATCH_MEMORY_USER_ID",
                "default_user",
            ).strip(),
            patch_memory_batch_size=_env_int(
                "PALMCLAW_PATCH_MEMORY_BATCH_SIZE",
                32,
            ),
            patch_memory_batch_tokens=_env_int(
                "PALMCLAW_PATCH_MEMORY_BATCH_TOKENS",
                4_096,
            ),
            patch_memory_max_attempts=_env_int(
                "PALMCLAW_PATCH_MEMORY_MAX_ATTEMPTS",
                3,
            ),
            patch_memory_retry_delay_seconds=_env_float(
                "PALMCLAW_PATCH_MEMORY_RETRY_DELAY_SECONDS",
                30.0,
            ),
            patch_memory_lease_seconds=_env_float(
                "PALMCLAW_PATCH_MEMORY_LEASE_SECONDS",
                180.0,
            ),
            fact_memory_enabled=fact_memory_enabled,
            fact_memory_backend=os.getenv(
                "PALMCLAW_FACT_MEMORY_BACKEND",
                "auto",
            )
            .strip()
            .lower(),
            fact_memory_model=os.getenv(
                "PALMCLAW_FACT_MEMORY_MODEL",
                memory_model,
            ).strip(),
            fact_memory_user_id=os.getenv(
                "PALMCLAW_FACT_MEMORY_USER_ID",
                "default_user",
            ).strip(),
            fact_memory_linking_context_limit=_env_int(
                "PALMCLAW_FACT_MEMORY_LINKING_CONTEXT_LIMIT",
                24,
            ),
            tool_memory_retrieval_enabled=_env_bool(
                "PALMCLAW_TOOL_MEMORY_RETRIEVAL_ENABLED",
                patch_memory_enabled or fact_memory_enabled,
            ),
            tool_memory_retrieval_mode=os.getenv(
                "PALMCLAW_TOOL_MEMORY_RETRIEVAL_MODE",
                "hybrid",
            )
            .strip()
            .lower(),
            tool_memory_top_k=_env_int(
                "PALMCLAW_TOOL_MEMORY_TOP_K",
                5,
            ),
            tool_memory_context_tokens=_env_int(
                "PALMCLAW_TOOL_MEMORY_CONTEXT_TOKENS",
                1_000,
            ),
            tool_memory_max_routes=_env_int(
                "PALMCLAW_TOOL_MEMORY_MAX_ROUTES",
                3,
            ),
            tool_memory_semantic_routing_enabled=_env_bool(
                "PALMCLAW_TOOL_MEMORY_SEMANTIC_ROUTING_ENABLED",
                True,
            ),
            tool_memory_vehicle_id=(
                os.getenv("PALMCLAW_TOOL_MEMORY_VEHICLE_ID", "").strip()
                or None
            ),
            joint_memory_tool_planning_enabled=_env_bool(
                "PALMCLAW_JOINT_MEMORY_TOOL_PLANNING_ENABLED",
                False,
            ),
            joint_memory_tool_max_tools=_env_int(
                "PALMCLAW_JOINT_MEMORY_TOOL_MAX_TOOLS",
                12,
            ),
            joint_memory_tool_schema_tokens=_env_int(
                "PALMCLAW_JOINT_MEMORY_TOOL_SCHEMA_TOKENS",
                1_800,
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.memory_backend not in {"auto", "openai", "local", "fake"}:
            raise ValueError(
                "memory_backend must be 'auto', 'openai', 'local', or 'fake'"
            )
        if self.patch_memory_backend not in {"auto", "openai", "fake"}:
            raise ValueError(
                "patch_memory_backend must be 'auto', 'openai', or 'fake'"
            )
        if self.fact_memory_backend not in {"auto", "openai", "fake"}:
            raise ValueError(
                "fact_memory_backend must be 'auto', 'openai', or 'fake'"
            )
        if self.patch_memory_enabled and self.fact_memory_enabled:
            raise ValueError(
                "patch_memory_enabled and fact_memory_enabled are mutually exclusive"
            )
        if self.memory_backend == "local":
            if not self.local_memory_model:
                raise ValueError("local_memory_model cannot be empty")
            if not self.local_memory_base_url:
                raise ValueError("local_memory_base_url cannot be empty")
        if self.memory_strategy not in {"summary", "structured"}:
            raise ValueError("memory_strategy must be 'summary' or 'structured'")
        if self.retrieval_mode not in {
            "none",
            "full",
            "bm25",
            "embedding",
            "hybrid",
        }:
            raise ValueError("Invalid retrieval_mode")
        if self.tool_memory_retrieval_mode not in {
            "none",
            "full",
            "bm25",
            "embedding",
            "hybrid",
        }:
            raise ValueError("Invalid tool_memory_retrieval_mode")
        if self.memory_top_k < 1:
            raise ValueError("memory_top_k must be at least 1")
        if self.tool_memory_top_k < 1:
            raise ValueError("tool_memory_top_k must be at least 1")
        if self.tool_memory_max_routes < 1:
            raise ValueError("tool_memory_max_routes must be at least 1")
        if self.joint_memory_tool_max_tools < 1:
            raise ValueError("joint_memory_tool_max_tools must be at least 1")
        if self.joint_memory_tool_schema_tokens < 1:
            raise ValueError("joint_memory_tool_schema_tokens must be positive")
        if self.embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be at least 1")
        if self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if self.max_history_messages < 1:
            raise ValueError("max_history_messages must be at least 1")
        if self.max_concurrent_sessions < 1:
            raise ValueError("max_concurrent_sessions must be at least 1")
        if self.memory_trigger_messages < 1:
            raise ValueError("memory_trigger_messages must be at least 1")
        if self.model_timeout_seconds <= 0 or self.tool_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_file_bytes < 1 or self.max_tool_result_chars < 1:
            raise ValueError("size limits must be positive")
        if self.max_web_bytes < 1 or self.max_web_redirects < 0:
            raise ValueError("web limits must be positive")
        if self.max_context_tokens < 256:
            raise ValueError("max_context_tokens must be at least 256")
        if (
            self.memory_context_tokens < 0
            or self.skill_context_tokens < 0
            or self.tool_memory_context_tokens < 0
        ):
            raise ValueError("component token budgets cannot be negative")
        if (
            self.memory_context_tokens
            + self.skill_context_tokens
            + self.tool_memory_context_tokens
            >= self.max_context_tokens
        ):
            raise ValueError(
                "memory and Skill budgets must leave room for policy and history"
            )
        if self.agent_max_output_tokens < 1 or self.memory_max_output_tokens < 1:
            raise ValueError("max output token limits must be positive")
        if self.patch_memory_batch_size < 1:
            raise ValueError("patch_memory_batch_size must be positive")
        if self.patch_memory_batch_tokens < 256:
            raise ValueError("patch_memory_batch_tokens must be at least 256")
        if self.patch_memory_max_attempts < 1:
            raise ValueError("patch_memory_max_attempts must be positive")
        if self.patch_memory_retry_delay_seconds < 0:
            raise ValueError("patch_memory_retry_delay_seconds cannot be negative")
        if self.patch_memory_lease_seconds <= self.model_timeout_seconds:
            raise ValueError(
                "patch_memory_lease_seconds must exceed model_timeout_seconds"
            )
        if not self.patch_memory_user_id:
            raise ValueError("patch_memory_user_id cannot be empty")
        if not self.fact_memory_user_id:
            raise ValueError("fact_memory_user_id cannot be empty")
        if self.fact_memory_linking_context_limit < 1:
            raise ValueError(
                "fact_memory_linking_context_limit must be positive"
            )
        if not 0 <= self.memory_min_confidence <= 1:
            raise ValueError("memory_min_confidence must be between 0 and 1")
        if not 0 <= self.memory_global_min_confidence <= 1:
            raise ValueError("memory_global_min_confidence must be between 0 and 1")
        if self.local_pii_storage not in {"raw", "redacted"}:
            raise ValueError("local_pii_storage must be 'raw' or 'redacted'")
        if any(
            rate < 0
            for rate in (
                self.agent_input_cost_per_million,
                self.agent_output_cost_per_million,
                self.memory_input_cost_per_million,
                self.memory_output_cost_per_million,
                self.embedding_input_cost_per_million,
            )
        ):
            raise ValueError("model cost rates cannot be negative")
        allowed_efforts = {"none", "low", "medium", "high", "xhigh", "max"}
        if self.agent_reasoning_effort not in allowed_efforts:
            raise ValueError("Invalid agent reasoning effort")
        if self.memory_reasoning_effort not in allowed_efforts:
            raise ValueError("Invalid memory reasoning effort")

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.workspace_root,
            self.shared_workspace_root,
            self.workspace_skills_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def require_cloud_configuration(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        if not self.agent_model:
            raise RuntimeError("PALMCLAW_AGENT_MODEL is not set")
        if self.memory_backend == "openai" and not self.memory_model:
            raise RuntimeError("PALMCLAW_MEMORY_MODEL is not set")
        resolved_patch_backend = (
            self.backend
            if self.patch_memory_backend == "auto"
            else self.patch_memory_backend
        )
        if (
            self.patch_memory_enabled
            and resolved_patch_backend == "openai"
            and not self.patch_memory_model
        ):
            raise RuntimeError("PALMCLAW_PATCH_MEMORY_MODEL is not set")
        resolved_fact_backend = (
            self.backend
            if self.fact_memory_backend == "auto"
            else self.fact_memory_backend
        )
        if (
            self.fact_memory_enabled
            and resolved_fact_backend == "openai"
            and not self.fact_memory_model
        ):
            raise RuntimeError("PALMCLAW_FACT_MEMORY_MODEL is not set")
        if (
            self.memory_strategy == "structured"
            and self.retrieval_mode in {"embedding", "hybrid"}
            and not self.embedding_model
        ):
            raise RuntimeError("PALMCLAW_EMBEDDING_MODEL is not set")
