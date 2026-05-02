"""HTTP proxy server — transparent cache layer between agent and LLM.

Usage:
    python -m agent_compiler.adapters.proxy
    python -m agent_compiler.adapters.proxy --port 8100 --upstream https://api.deepseek.com

Then configure your agent's LLM API base to: http://127.0.0.1:8100/v1
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from agent_compiler.core.agent import Agent
from agent_compiler.core.config import AgentConfig, LLMConfig
from agent_compiler.core.types import AgentResult


class AgentCompilerProxy:
    """Transparent HTTP proxy that adds three-layer caching to any agent's LLM calls.

    Works with OpenClaw, Hermes Agent, LangChain, AutoGPT, or any framework
    that speaks OpenAI-compatible chat completions API.

    Architecture:
        Agent → POST /v1/chat/completions → Proxy
                                              ├─ L1: rule match → execute → return
                                              ├─ L2: cache match → execute → return
                                              └─ L3: forward to upstream LLM → cache → return
    """

    def __init__(
        self,
        upstream_base_url: Optional[str] = None,
        upstream_api_key: Optional[str] = None,
        upstream_model: Optional[str] = None,
        listen_host: str = "127.0.0.1",
        listen_port: int = 8100,
        cache_dir: str = "./proxy_cache",
        similarity_threshold: float = 0.30,
        rules_path: Optional[str] = None,
    ):
        # Upstream LLM config
        self.upstream_base_url = upstream_base_url or os.environ.get("LLM_API_BASE", "https://api.deepseek.com")
        self.upstream_api_key = upstream_api_key or os.environ.get("LLM_API_KEY", "")
        self.upstream_model = upstream_model or os.environ.get("LLM_MODEL", "deepseek-chat")

        # Server config
        self.listen_host = listen_host
        self.listen_port = listen_port

        # Agent-compiler core (same three-layer engine)
        self.agent = Agent(
            config=AgentConfig(
                llm=LLMConfig(
                    provider="openai_compat",
                    api_key=self.upstream_api_key,
                    api_base=self.upstream_base_url,
                    model=self.upstream_model,
                ),
                cache_dir=cache_dir,
                similarity_threshold=similarity_threshold,
                cache_seeds_path=rules_path,
            )
        )
        self._http = httpx.Client(timeout=httpx.Timeout(120.0))
        self._total_requests = 0
        self._cache_hits = 0

    # ── Request extraction ─────────────────────────────────────────

    def extract_user_message(self, body: dict) -> str:
        """Extract the last user message from a chat completion request body."""
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Vision format: [{"type": "text", "text": "..."}, ...]
                    parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    return " ".join(parts)
                return str(content)
        return ""

    def extract_system_prompt(self, body: dict) -> str:
        """Extract the system prompt from the request."""
        messages = body.get("messages", [])
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list) and len(content) > 0:
                    return str(content[0].get("text", ""))
                return str(content)
        return ""

    # ── Response formatting ────────────────────────────────────────

    def format_chat_response(self, content: str, model: Optional[str] = None) -> dict:
        """Format a plain text response as an OpenAI-compatible chat completion."""
        return {
            "id": f"proxy-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model or "agent-compiler",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "note": "agent-compiler cache hit — 0 LLM tokens used",
            },
        }

    def format_tool_result_message(self, results: list[dict]) -> str:
        """Format executed tool results into a human-readable text response.

        This makes the agent think the LLM analyzed the situation and
        produced a natural-language summary with the tool results embedded.
        """
        parts = []

        for i, r in enumerate(results, 1):
            if not r.get("success"):
                parts.append(f"{i}. {r['tool']}: 执行失败 ({r.get('error', 'unknown')})")
                continue

            data = r.get("data", {})
            tool = r["tool"]

            if tool == "get_system_status":
                parts.append(f"**系统状态**\n"
                           f"- 主机: {data.get('hostname', 'N/A')}\n"
                           f"- CPU: {data.get('cpu_percent', 'N/A')}%\n"
                           f"- 内存: {data.get('memory_used_gb', 'N/A')}/{data.get('memory_total_gb', 'N/A')} GB\n"
                           f"- 运行时间: {data.get('uptime_days', 'N/A')} 天\n"
                           f"- 活跃服务: {', '.join(data.get('active_services', []))}")

            elif tool == "get_disk_usage":
                parts.append(f"**磁盘空间**\n"
                           f"- 总容量: {data.get('total_gb', 'N/A')} GB\n"
                           f"- 已用: {data.get('used_gb', 'N/A')} GB\n"
                           f"- 剩余: {data.get('free_gb', 'N/A')} GB\n"
                           f"- 使用率: {data.get('use_percent', 'N/A')}%")

            elif tool == "search_logs":
                entries = data.get("entries", [])
                lines = [f"**日志搜索结果** (pattern: {data.get('pattern', '')}, "
                        f"共 {data.get('total_hits', len(entries))} 条)"]
                for e in entries[:5]:
                    lines.append(f"- [{e.get('timestamp', '')}] [{e.get('level', '')}] "
                               f"{e.get('service', '')}: {e.get('message', '')}")
                if len(entries) > 5:
                    lines.append(f"... 还有 {len(entries) - 5} 条")
                parts.append("\n".join(lines))

            elif tool == "generate_report":
                parts.append(data.get("body", f"报告: {data.get('title', '')}"))

            elif tool == "list_directory":
                files = data.get("files", [])
                lines = [f"**目录内容** ({data.get('path', '.')}, 共 {data.get('total', len(files))} 个文件)"]
                for f in files[:10]:
                    lines.append(f"- {f['name']} ({f.get('size_kb', '?')} KB)")
                parts.append("\n".join(lines))

            elif tool == "get_current_time":
                parts.append(f"**当前时间**\n- {data.get('date', '')} {data.get('time', '')} "
                           f"({data.get('weekday', '')})")

            elif tool == "chat_reply":
                parts.append(data.get("message", ""))

            elif tool == "find_large_files":
                files = data.get("files", [])
                lines = [f"**大文件** ({data.get('path', '')}, top {data.get('top_n', '')})"]
                for f in files[:10]:
                    lines.append(f"- {f['name']}: {f.get('size_mb', '?')} MB")
                parts.append("\n".join(lines))

            else:
                parts.append(f"**{tool}**: {json.dumps(data, ensure_ascii=False, indent=2)}")

        return "\n\n".join(parts)

    # ── Core proxy logic ────────────────────────────────────────────

    def process_request(self, body: dict) -> dict:
        """Process a chat completion request through agent caching.

        Returns an OpenAI-compatible response dict.
        """
        self._total_requests += 1
        user_input = self.extract_user_message(body)
        requested_model = body.get("model", self.upstream_model)

        if not user_input.strip():
            return self._forward(body, requested_model)

        # Run through agent-compiler
        result = self.agent.process(user_input)

        if result.source == "cache":
            # Cache hit: return formatted tool result
            self._cache_hits += 1
            steps = result.data.get("steps", []) if result.success else []
            message = self.format_tool_result_message(steps)
            latency_info = f"({result.latency_ms:.1f}ms)"
            full_message = f"[agent-compiler -> cache {latency_info}]\n\n{message}"
            return self.format_chat_response(full_message, model="agent-compiler")

        if result.source == "llm" and result.text:
            # LLM response with conversational text
            return self.format_chat_response(result.text, model="agent-compiler")

        # Fallback: forward to upstream LLM
        response = self._forward(body, requested_model)
        return response

    def _forward(self, body: dict, model: str) -> dict:
        """Forward the request to the upstream LLM provider."""
        url = f"{self.upstream_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.upstream_api_key}",
            "Content-Type": "application/json",
        }
        if not body.get("model"):
            body = {**body, "model": model}

        resp = self._http.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # ── Stats ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": f"{self._cache_hits / self._total_requests * 100:.1f}%"
                if self._total_requests > 0 else "0%",
            "agent_stats": self.agent.stats(),
        }


# ── FastAPI app ────────────────────────────────────────────────────

def create_app(proxy: Optional[AgentCompilerProxy] = None):
    """Create a FastAPI app backed by an AgentCompilerProxy."""
    from fastapi import FastAPI
    from fastapi import Request as FastAPIRequest
    from fastapi.responses import JSONResponse

    if proxy is None:
        proxy = AgentCompilerProxy()

    app = FastAPI(title="Agent Compiler Proxy", version="0.2.0")

    @app.get("/health")
    async def health():
        return {"status": "ok", "stats": proxy.stats}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: FastAPIRequest):
        body = await request.json()
        try:
            response = proxy.process_request(body)
            return JSONResponse(content=response)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "proxy_error"}},
            )

    @app.get("/stats")
    async def stats():
        return proxy.stats

    return app


# ── CLI entry ──────────────────────────────────────────────────────

def main():
    """Start the proxy server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Agent Compiler Proxy — token-saving cache layer for any AI agent")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address")
    parser.add_argument("--port", type=int, default=8100, help="Listen port")
    parser.add_argument("--upstream", help="Upstream LLM API base URL")
    parser.add_argument("--api-key", help="Upstream LLM API key")
    parser.add_argument("--model", help="Upstream model name")
    parser.add_argument("--cache-dir", default="./proxy_cache", help="Cache directory")
    parser.add_argument("--threshold", type=float, default=0.30, help="Similarity threshold")
    parser.add_argument("--rules", default="rules.yaml", help="Rules file path")

    args = parser.parse_args()

    proxy = AgentCompilerProxy(
        upstream_base_url=args.upstream,
        upstream_api_key=args.api_key,
        upstream_model=args.model,
        listen_host=args.host,
        listen_port=args.port,
        cache_dir=args.cache_dir,
        similarity_threshold=args.threshold,
        rules_path=args.rules,
    )

    print(f"""
╔══════════════════════════════════════════════════════╗
║     Agent Compiler Proxy v0.2.0                      ║
╠══════════════════════════════════════════════════════╣
║  上游: {proxy.upstream_base_url:<38} ║
║  模型: {proxy.upstream_model:<38} ║
║  监听: http://{proxy.listen_host}:{proxy.listen_port:<36} ║
║                                                      ║
║  用法:                                                ║
║  将 Agent 的 LLM API Base 改为:                       ║
║    http://127.0.0.1:{args.port}/v1                     ║
║                                                      ║
║  OpenClaw: 设置 models.providers.*.baseUrl            ║
║  Hermes:   设置 LLM_API_BASE 环境变量                 ║
╚══════════════════════════════════════════════════════╝
""")

    app = create_app(proxy)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
