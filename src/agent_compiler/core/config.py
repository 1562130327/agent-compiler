"""Centralized configuration — reads from env vars and optional config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "openai_compat"   # claude | openai | openai_compat
    api_key: str = ""
    api_base: str = ""
    model: str = ""

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    @property
    def summary(self) -> dict:
        masked = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
        return {
            "provider": self.provider,
            "model": self.model or "(default)",
            "api_base": self.api_base or "(default)",
            "api_key": masked if self.api_key else "(not set)",
        }


@dataclass
class AgentConfig:
    """Full agent configuration.

    Priority: constructor kwargs > config.yaml > env vars > defaults

    Environment variables:
        LLM_PROVIDER, LLM_API_KEY, LLM_API_BASE, LLM_MODEL
        AGENT_CACHE_DIR, AGENT_SIMILARITY_THRESHOLD
        AGENT_MAX_TURNS, AGENT_MAX_SESSION_MSGS
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    cache_dir: str = "./agent_cache"
    similarity_threshold: float = 0.55
    embedding_mode: str = "lightweight"   # lightweight | neural
    ram_max_entries: int = 1000
    ram_max_memory_mb: float = 200.0
    max_turns: int = 10                  # ReAct loop max iterations
    max_session_messages: int = 50       # conversation context window
    cache_seeds_path: str | None = None  # path to YAML file for cache pre-seeding
    mcp_servers: list[dict] = field(default_factory=list)  # MCP server configs
    skills_dirs: list[str] = field(default_factory=list)   # extra skill directories
    context_max_tokens: int = 100_000    # context window token budget

    @classmethod
    def from_env(cls, **overrides) -> AgentConfig:
        """Build config from environment variables, with optional overrides."""
        llm = LLMConfig(
            provider=overrides.pop("llm_provider", None)
                     or os.environ.get("LLM_PROVIDER", "openai_compat"),
            api_key=overrides.pop("llm_api_key", None)
                    or os.environ.get("LLM_API_KEY", ""),
            api_base=overrides.pop("llm_api_base", None)
                     or os.environ.get("LLM_API_BASE", ""),
            model=overrides.pop("llm_model", None)
                  or os.environ.get("LLM_MODEL", ""),
        )
        if not llm.model:
            llm.model = {
                "claude": "claude-sonnet-4-6",
                "openai": "gpt-4o-mini",
                "openai_compat": "gpt-4o-mini",
            }.get(llm.provider, "")

        return cls(
            llm=llm,
            cache_dir=overrides.pop("cache_dir", None) or os.environ.get("AGENT_CACHE_DIR", "./agent_cache"),
            similarity_threshold=float(overrides.pop("similarity_threshold", None) or os.environ.get("AGENT_SIMILARITY_THRESHOLD", "0.55")),
            max_turns=int(overrides.pop("max_turns", None) or os.environ.get("AGENT_MAX_TURNS", "10")),
            max_session_messages=int(overrides.pop("max_session_messages", None) or os.environ.get("AGENT_MAX_SESSION_MSGS", "50")),
            cache_seeds_path=overrides.pop("cache_seeds_path", None) or os.environ.get("AGENT_CACHE_SEEDS"),
            **overrides,
        )

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> AgentConfig:
        """Build config from a YAML file, with overrides."""
        import yaml
        cfg_data = {}
        if Path(path).exists():
            with open(path, encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}

        llm_data = cfg_data.get("llm", {})
        cache_data = cfg_data.get("cache", {})
        agent_data = cfg_data.get("agent", {})

        llm = LLMConfig(
            provider=overrides.pop("llm_provider", None)
                     or llm_data.get("provider", "mock"),
            api_key=overrides.pop("llm_api_key", None)
                    or llm_data.get("api_key", ""),
            api_base=overrides.pop("llm_api_base", None)
                     or llm_data.get("api_base", ""),
            model=overrides.pop("llm_model", None)
                  or llm_data.get("model", ""),
        )
        if not llm.model:
            llm.model = {
                "claude": "claude-sonnet-4-6",
                "openai": "gpt-4o-mini",
                "openai_compat": "gpt-4o-mini",
            }.get(llm.provider, "")

        return cls(
            llm=llm,
            cache_dir=overrides.pop("cache_dir", None) or cache_data.get("dir", "./agent_cache"),
            similarity_threshold=float(overrides.pop("similarity_threshold", None) or cache_data.get("similarity_threshold", 0.55)),
            max_turns=int(overrides.pop("max_turns", None) or agent_data.get("max_turns", 10)),
            max_session_messages=int(overrides.pop("max_session_messages", None) or agent_data.get("max_session_messages", 50)),
            cache_seeds_path=overrides.pop("cache_seeds_path", None) or cache_data.get("seeds_path"),
            mcp_servers=overrides.pop("mcp_servers", None) or cfg_data.get("mcp_servers", []),
            skills_dirs=overrides.pop("skills_dirs", None) or cfg_data.get("skills_dirs", []),
            context_max_tokens=int(overrides.pop("context_max_tokens", None) or agent_data.get("context_max_tokens", 100_000)),
            **overrides,
        )
