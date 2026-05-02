"""Tool registry — pluggable tool system.

Tools are the actual functions the agent can call. To add custom tools,
register them with ToolRegistry.register() or extend the _BUILTIN_TOOLS dict.
"""

from __future__ import annotations

from typing import Any, Callable

from agent_compiler.core.types import ActionStep, ToolDefinition


class ToolRegistry:
    """Global tool registry. Tools are pure functions that take params and return dicts.

    Usage:
        # Register a custom tool
        ToolRegistry.register("my_tool", my_function)

        # Register with LLM metadata
        ToolRegistry.register_with_def("my_tool", my_function,
            description="Does something useful",
            params_schema={"type": "object", "properties": {...}},
        )

        # Execute a step
        result = ToolRegistry.execute(ActionStep(tool_name="my_tool", params={}))
    """

    _tools: dict[str, Callable] = {}
    _defs: dict[str, dict] = {}
    _initialized = False

    @classmethod
    def register(cls, name: str, fn: Callable[..., dict]):
        """Register a tool function."""
        cls._tools[name] = fn

    @classmethod
    def register_with_def(cls, name: str, fn: Callable[..., dict],
                          description: str = "", params_schema: dict | None = None):
        """Register a tool with full LLM metadata."""
        cls._tools[name] = fn
        cls._defs[name] = {
            "name": name,
            "description": description,
            "fn": fn,
            "params_schema": params_schema or {"type": "object", "properties": {}},
        }

    @classmethod
    def unregister(cls, name: str):
        cls._tools.pop(name, None)
        cls._defs.pop(name, None)

    @classmethod
    def list_tools(cls) -> list[str]:
        cls._ensure_init()
        return list(cls._tools.keys())

    @classmethod
    def list_tool_definitions(cls) -> list[dict]:
        """Return tool metadata for LLM tool-use APIs.

        Returns list of {name, description, params_schema} dicts.
        """
        cls._ensure_init()
        return [{
            "name": name,
            "description": defn["description"],
            "params_schema": defn["params_schema"],
        } for name, defn in cls._defs.items()]

    @classmethod
    def get_tool_definition(cls, name: str) -> dict | None:
        """Get a single tool's definition dict."""
        cls._ensure_init()
        return cls._defs.get(name)

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
            from agent_compiler.tools.demo_tools import _BUILTIN_TOOLS, _BUILTIN_TOOL_DEFS
            cls._tools.update(_BUILTIN_TOOLS)
            cls._defs.update(_BUILTIN_TOOL_DEFS)
            # Load real system tools (override mock list_directory with real one)
            from agent_compiler.tools.system_tools import _SYSTEM_TOOLS, _SYSTEM_TOOL_DEFS
            cls._tools.update(_SYSTEM_TOOLS)
            cls._defs.update(_SYSTEM_TOOL_DEFS)
            cls._initialized = True
