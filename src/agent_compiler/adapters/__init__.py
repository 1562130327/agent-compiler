"""Adapters — agent-compiler as a token-saving plugin for other agent frameworks.

Each adapter intercepts LLM calls from a specific agent framework,
checks the three-layer cache (Rule → Pattern → LLM), and returns
cached results without calling the LLM when possible.

Architecture:
    Agent (OpenClaw / Hermes / ...) → Proxy (localhost:8100) → Cache check
                                                                ├─ Hit → execute tools → return
                                                                └─ Miss → forward to LLM → cache → return
"""

from agent_compiler.adapters.proxy import AgentCompilerProxy

__all__ = ["AgentCompilerProxy"]
