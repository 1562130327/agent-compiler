"""Agent Compiler — AI agent with automatic workflow caching.

Architecture:
    Cache check → hit: execute cached workflow, pure CPU
                 → miss: LLM ReAct loop → compile → cache → execute

The system gets faster over time as the cache grows.

Quick start:
    from agent_compiler import Agent
    agent = Agent()
    result = agent.process("check server status")
    print(agent.efficiency_report())
"""

from agent_compiler.core.agent import Agent
from agent_compiler.core.config import AgentConfig, LLMConfig
from agent_compiler.core.types import (
    ActionStep, AgentResult, Message, Session, ToolDefinition, WorkflowTemplate,
)
from agent_compiler.tools.registry import ToolRegistry

__all__ = [
    "Agent", "AgentConfig", "LLMConfig",
    "ActionStep", "AgentResult", "Message", "Session", "ToolDefinition",
    "WorkflowTemplate", "ToolRegistry",
]
