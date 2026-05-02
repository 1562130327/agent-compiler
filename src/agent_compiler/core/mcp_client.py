"""MCP client manager — connects to MCP servers, discovers tools, routes calls.

Supports stdio (subprocess) and HTTP transports. MCP servers expose tools
that are automatically registered in ToolRegistry.

Config example (config.yaml):
    mcp_servers:
      - name: filesystem
        transport: stdio
        command: npx
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"]
      - name: github
        transport: http
        url: http://localhost:8080/sse
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_compiler.tools.registry import ToolRegistry


@dataclass
class MCPServerConfig:
    """Configuration for one MCP server connection."""
    name: str
    transport: str = "stdio"          # "stdio" | "http" | "sse"
    # stdio:
    command: str | None = None        # e.g. "npx", "python"
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # http/sse:
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> MCPServerConfig:
        return cls(
            name=d["name"],
            transport=d.get("transport", "stdio"),
            command=d.get("command"),
            args=d.get("args", []),
            env=d.get("env", {}),
            url=d.get("url"),
            headers=d.get("headers", {}),
        )


class MCPClientManager:
    """Manages connections to one or more MCP servers.

    On connect(), each server's tools are discovered and registered
    in ToolRegistry with the prefix "mcp/{server_name}/{tool_name}".

    Usage:
        mgr = MCPClientManager()
        mgr.configure([MCPServerConfig(...), ...])
        mgr.connect_all()    # sync wrapper around async connect
        # Tools are now registered and callable via ToolRegistry
    """

    def __init__(self):
        self._servers: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, Any] = {}  # MCP ClientSession objects
        self._tools: dict[str, tuple[str, str]] = {}  # tool_name -> (server_name, orig_name)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._connected = False

    def configure(self, servers: list[MCPServerConfig]):
        """Set the MCP servers to connect to."""
        for srv in servers:
            self._servers[srv.name] = srv

    def configure_from_dicts(self, servers: list[dict]):
        """Configure from list of dicts (e.g. from config.yaml)."""
        self.configure([MCPServerConfig.from_dict(d) for d in servers])

    def connect_all(self):
        """Connect to all configured MCP servers synchronously.

        Starts a background event loop thread for MCP communication.
        """
        if self._connected:
            return
        if not self._servers:
            return

        # Start async event loop in a background thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        for name, srv in self._servers.items():
            try:
                self._connect_sync(name, srv)
            except Exception as e:
                print(f"[MCP] Failed to connect to server '{name}': {e}", file=sys.stderr)

        self._connected = True

    def _connect_sync(self, name: str, srv: MCPServerConfig):
        """Connect to one MCP server synchronously."""
        async def _connect():
            if srv.transport == "stdio":
                return await self._connect_stdio(srv)
            elif srv.transport in ("http", "sse"):
                return await self._connect_http(srv)
            else:
                raise ValueError(f"Unknown transport: {srv.transport}")

        future = asyncio.run_coroutine_threadsafe(_connect(), self._loop)
        future.result(timeout=30)

    async def _connect_stdio(self, srv: MCPServerConfig):
        """Connect to an MCP server over stdio."""
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession

        cmd = srv.command or "npx"
        args = srv.args or []
        env = {**__import__('os').environ, **srv.env}

        # Start the subprocess
        proc = subprocess.Popen(
            [cmd] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        async with stdio_client(
            (proc.stdin, proc.stdout)
        ) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._sessions[srv.name] = (session, proc)

                # Discover tools
                result = await session.list_tools()
                for tool in result.tools:
                    mcp_name = f"mcp/{srv.name}/{tool.name}"
                    self._tools[mcp_name] = (srv.name, tool.name)

                    # Build a synchronous wrapper for this tool
                    def make_wrapper(srv_name, tool_name, tool_schema):
                        def wrapper(**kwargs):
                            return self._call_tool_sync(srv_name, tool_name, kwargs)
                        return wrapper

                    wrapper = make_wrapper(srv.name, tool.name, tool)
                    ToolRegistry.register_with_def(
                        mcp_name,
                        wrapper,
                        description=f"[MCP:{srv.name}] {tool.description or tool.name}",
                        params_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {"type": "object", "properties": {}},
                    )

    async def _connect_http(self, srv: MCPServerConfig):
        """Connect to an MCP server over HTTP/SSE."""
        # For HTTP transport, use the streamable HTTP client
        try:
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            # Fall back to SSE client
            from mcp.client.sse import sse_client

            async with sse_client(srv.url, headers=srv.headers) as (read, write):
                from mcp import ClientSession
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._sessions[srv.name] = (session, None)

                    result = await session.list_tools()
                    for tool in result.tools:
                        mcp_name = f"mcp/{srv.name}/{tool.name}"
                        self._tools[mcp_name] = (srv.name, tool.name)

                        def make_wrapper(srv_name, tool_name):
                            def wrapper(**kwargs):
                                return self._call_tool_sync(srv_name, tool_name, kwargs)
                            return wrapper

                        wrapper = make_wrapper(srv.name, tool.name)
                        ToolRegistry.register_with_def(
                            mcp_name, wrapper,
                            description=f"[MCP:{srv.name}] {tool.description or tool.name}",
                            params_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {"type": "object", "properties": {}},
                        )
            return

        # streamable HTTP path
        async with streamablehttp_client(srv.url, headers=srv.headers) as (read, write, _):
            from mcp import ClientSession
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._sessions[srv.name] = (session, None)

                result = await session.list_tools()
                for tool in result.tools:
                    mcp_name = f"mcp/{srv.name}/{tool.name}"
                    self._tools[mcp_name] = (srv.name, tool.name)

                    def make_wrapper(srv_name, tool_name):
                        def wrapper(**kwargs):
                            return self._call_tool_sync(srv_name, tool_name, kwargs)
                        return wrapper

                    wrapper = make_wrapper(srv.name, tool.name)
                    ToolRegistry.register_with_def(
                        mcp_name, wrapper,
                        description=f"[MCP:{srv.name}] {tool.description or tool.name}",
                        params_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {"type": "object", "properties": {}},
                    )

    def _call_tool_sync(self, server_name: str, tool_name: str, params: dict) -> dict:
        """Call an MCP tool synchronously from the background thread."""
        async def _call():
            session, _ = self._sessions.get(server_name, (None, None))
            if session is None:
                return {"error": f"MCP server '{server_name}' not connected"}
            try:
                result = await session.call_tool(tool_name, params)
                # Convert result to dict
                if hasattr(result, 'content'):
                    texts = []
                    for c in result.content:
                        if hasattr(c, 'text'):
                            texts.append(c.text)
                        elif hasattr(c, 'data'):
                            texts.append(str(c.data))
                        else:
                            texts.append(str(c))
                    return {"success": True, "data": "\n".join(texts)}
                return {"success": True, "data": str(result)}
            except Exception as e:
                return {"success": False, "error": str(e)}

        future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
        try:
            return future.result(timeout=60)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_mcp_tools(self) -> list[str]:
        """Return all registered MCP tool names."""
        return list(self._tools.keys())

    def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for name in list(self._sessions.keys()):
            try:
                session, proc = self._sessions[name]
                # Unregister tools
                for mcp_name, (srv_name, _) in list(self._tools.items()):
                    if srv_name == name:
                        ToolRegistry.unregister(mcp_name)
                        del self._tools[mcp_name]
                del self._sessions[name]
            except Exception:
                pass
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._connected = False
