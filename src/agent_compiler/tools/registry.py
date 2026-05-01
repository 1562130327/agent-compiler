"""Tool registry — pluggable tool system.

Tools are the actual functions the agent can call. To add custom tools,
register them with ToolRegistry.register() or extend the _BUILTIN_TOOLS dict.
"""

from __future__ import annotations

from typing import Any, Callable

from agent_compiler.core.types import ActionStep


class ToolRegistry:
    """Global tool registry. Tools are pure functions that take params and return dicts.

    Usage:
        # Register a custom tool
        ToolRegistry.register("my_tool", my_function)

        # Execute a step
        result = ToolRegistry.execute(ActionStep(tool_name="my_tool", params={}))
    """

    _tools: dict[str, Callable] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str, fn: Callable[..., dict]):
        """Register a tool function."""
        cls._tools[name] = fn

    @classmethod
    def unregister(cls, name: str):
        cls._tools.pop(name, None)

    @classmethod
    def list_tools(cls) -> list[str]:
        cls._ensure_init()
        return list(cls._tools.keys())

    @classmethod
    def execute(cls, step: ActionStep) -> dict:
        """Execute a single action step and return the result."""
        cls._ensure_init()
        tool_name = step.tool_name
        fn = cls._tools.get(tool_name)
        if fn is None:
            return {"tool": tool_name, "error": f"Unknown tool: {tool_name}"}

        # Resolve parameter placeholders (${...}) — use None for unresolved
        resolved = {}
        for k, v in step.params.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                resolved[k] = None
            else:
                resolved[k] = v

        try:
            result = fn(**resolved)
            return {"tool": tool_name, "success": True, "data": result}
        except Exception as e:
            return {"tool": tool_name, "success": False, "error": str(e)}

    @classmethod
    def _ensure_init(cls):
        if not cls._initialized:
            from agent_compiler.tools.demo_tools import _BUILTIN_TOOLS
            cls._tools.update(_BUILTIN_TOOLS)
            cls._initialized = True
