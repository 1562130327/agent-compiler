"""Shared type definitions for the agent compiler framework."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np


@dataclass
class ActionStep:
    """A single executable step in a workflow."""
    tool_name: str
    params: dict[str, Any]
    is_generic: bool = False
    description: str = ""


@dataclass
class WorkflowTemplate:
    """A cached workflow pattern that can be reused."""
    id: str
    intent: str
    steps: list[ActionStep]
    keywords: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    hit_count: int = 0
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)
    params_schema: dict[str, Any] = field(default_factory=dict)
    original_input: str = ""  # original user input in the user's language

    @staticmethod
    def generate_id(intent: str) -> str:
        h = hashlib.sha256(intent.encode()).hexdigest()[:12]
        return f"wf_{h}"

    def bump(self):
        self.hit_count += 1
        self.last_hit_at = time.time()


@dataclass
class AgentResult:
    """Result from the agent processing pipeline."""
    success: bool
    data: Any
    source: Literal["cache", "llm"]
    confidence: float
    latency_ms: float
    workflow_id: str | None = None
    error: str | None = None
    text: str = ""  # conversational response for display


@dataclass
class Message:
    """A single message in a conversation session."""
    role: str  # "user" | "assistant" | "tool_result"
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    """Multi-turn conversation session."""
    id: str
    messages: list[Message] = field(default_factory=list)
    system_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append(Message(role=role, content=content, **kwargs))
        self.last_active_at = time.time()


@dataclass
class ToolDefinition:
    """Tool metadata for LLM tool-use APIs."""
    name: str
    description: str
    fn: Callable[..., dict]
    params_schema: dict[str, Any]  # JSON Schema
