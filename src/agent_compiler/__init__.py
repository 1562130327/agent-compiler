"""Agent Compiler — LLM as compiler, not interpreter.

Three-layer dispatch:
    L1: Rule Engine    -> keyword/regex match, pure CPU
    L2: Pattern Cache  -> embedding similarity, pure CPU
    L3: LLM Fallback   -> GPU reasoning, with result caching

Quick start:
    from agent_compiler import Agent
    agent = Agent()
    result = agent.process("check server status")
    print(agent.efficiency_report())
"""

from agent_compiler.core.agent import Agent
from agent_compiler.core.config import AgentConfig, LLMConfig
from agent_compiler.core.types import ActionStep, AgentResult, Rule, WorkflowTemplate
from agent_compiler.tools.registry import ToolRegistry

__all__ = [
    "Agent", "AgentConfig", "LLMConfig",
    "ActionStep", "AgentResult", "Rule", "WorkflowTemplate",
    "ToolRegistry",
]
