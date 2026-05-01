"""Shared type definitions for the agent compiler framework."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Literal

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
    embedding: np.ndarray | None = None
    hit_count: int = 0
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)
    params_schema: dict[str, Any] = field(default_factory=dict)

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
    source: Literal["rule", "cache", "llm"]
    confidence: float
    latency_ms: float
    workflow_id: str | None = None
    error: str | None = None


@dataclass
class Rule:
    """A single rule in the rule engine."""
    name: str
    keywords: list[str]
    patterns: list[str]
    tool_name: str
    params: dict[str, Any]
