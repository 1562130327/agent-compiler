"""Layer abstract base class — defines the interface for each dispatch layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_compiler.core.types import AgentResult


class Layer(ABC):
    """Abstract base for a dispatch layer.

    Each layer implements a `match` method that returns an AgentResult
    on success or None to pass through to the next layer.
    """

    @abstractmethod
    def match(self, user_input: str, tool_executor) -> AgentResult | None:
        """Try to handle the user input. Return result or None to fall through."""

    def stats(self) -> dict:
        return {}
