"""Backward compatibility — re-exports from agent_compiler.tools."""

from agent_compiler.tools.registry import ToolRegistry

execute_step = ToolRegistry.execute

__all__ = ["execute_step"]
