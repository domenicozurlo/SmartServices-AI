import os
from dataclasses import dataclass
from typing import Literal

from handlers.structured_qa.types import StructuredQaTool


DEFAULT_ALLOWED_TOOLS = tuple(tool.value for tool in StructuredQaTool)
SUPPORTED_MODES = {"mock_db", "mcp"}


@dataclass(frozen=True)
class StructuredQaConfig:
    mode: Literal["mock_db", "mcp"]
    mcp_url: str
    mcp_toolset: str
    mcp_timeout_seconds: float
    mcp_enabled: bool
    allowed_tools: tuple[str, ...]

    @property
    def data_source(self) -> str:
        if self.mode == "mcp":
            return "mcp_toolbox"
        return "mock_db"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _allowed_tools_from_env() -> tuple[str, ...]:
    raw = os.getenv("MCP_STRUCTURED_QA_ALLOWED_TOOLS", "")
    if not raw.strip():
        return DEFAULT_ALLOWED_TOOLS
    requested = tuple(item.strip() for item in raw.split(",") if item.strip())
    known = set(DEFAULT_ALLOWED_TOOLS)
    allowed = tuple(item for item in requested if item in known)
    return allowed or DEFAULT_ALLOWED_TOOLS


def load_config() -> StructuredQaConfig:
    raw_mode = os.getenv("STRUCTURED_QA_MODE", "mock_db").strip().lower()
    mode = raw_mode if raw_mode in SUPPORTED_MODES else "mock_db"
    return StructuredQaConfig(
        mode=mode,  # type: ignore[arg-type]
        mcp_url=os.getenv("MCP_STRUCTURED_QA_URL", "").strip(),
        mcp_toolset=os.getenv("MCP_STRUCTURED_QA_TOOLSET", "senshistory_readonly").strip(),
        mcp_timeout_seconds=_env_float("MCP_STRUCTURED_QA_TIMEOUT_SECONDS", 15.0),
        mcp_enabled=_env_bool("MCP_STRUCTURED_QA_ENABLED"),
        allowed_tools=_allowed_tools_from_env(),
    )
