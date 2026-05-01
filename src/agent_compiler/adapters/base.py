"""Base adapter interface — all adapters extend this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProxyConfig:
    """Configuration for the agent-compiler proxy."""
    listen_host: str = "127.0.0.1"
    listen_port: int = 8100
    upstream_base_url: str = "https://api.deepseek.com"
    upstream_api_key: str = ""
    upstream_model: str = "deepseek-chat"
    cache_dir: str = "./proxy_cache"
    similarity_threshold: float = 0.30
    rules_path: str | None = "rules.yaml"


class BaseAdapter(ABC):
    """Base class for agent-framework-specific adapters.

    Each adapter:
      1. Knows how the target framework formats its LLM requests
      2. Can extract the user's intent from the request
      3. Can format cached results back into the expected response format
    """

    @abstractmethod
    def extract_user_message(self, request_body: dict) -> str:
        """Extract the user's message from an LLM chat completion request."""

    @abstractmethod
    def format_tool_result(self, tool_results: list[dict]) -> str:
        """Format executed tool results into an LLM-style text response.

        This is what the agent sees — it should look like the LLM
        performed the tool calls and summarized the results.
        """
