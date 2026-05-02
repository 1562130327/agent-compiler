"""Agent Compiler Web UI — FastAPI 后端 + 静态页面服务."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent_compiler.core.agent import Agent
from agent_compiler.core.config import AgentConfig

STATIC_DIR = Path(__file__).parent / "static"

# ── FastAPI app ──────────────────────────────────────────────────────

app_web = FastAPI(title="Agent Compiler", version="0.3.0")

# Global agent instance (initialized on startup)
_agent: Agent | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
        print(f"[启动] Agent 初始化完成, provider={_agent.llm.provider}, model={_agent.llm.model}")
    return _agent


# ── Routes ──────────────────────────────────────────────────────────


@app_web.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main chat interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>Agent Compiler</h1><p>Frontend not found.</p>"


@app_web.post("/api/chat")
async def chat(request: Request):
    """Chat endpoint — sends user message to agent, returns response."""
    try:
        body = await request.json()
    except Exception as exc:
        print(f"[ERROR] JSON 解析失败: {exc}")
        return JSONResponse({"error": f"请求格式错误: {exc}"}, status_code=400)

    user_input = body.get("message", "").strip()
    session_id = body.get("session_id")
    if not user_input:
        return {"error": "empty message"}

    print(f"[API] 收到消息: '{user_input[:60]}...' (session={session_id})")

    agent = get_agent()
    t0 = time.perf_counter()
    try:
        result = agent.process(user_input, session_id=session_id)
    except Exception as exc:
        traceback.print_exc()
        print(f"[ERROR] agent.process 异常: {exc}")
        return JSONResponse({"error": f"处理请求时出错: {exc}"}, status_code=500)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[API] 响应完成: source={result.source}, latency={elapsed:.1f}ms, tokens={result.tokens}")

    return {
        "text": result.text or "",
        "source": result.source,
        "latency_ms": round(elapsed, 1),
        "tokens": result.tokens or {},
        "session_id": session_id,
        "tot_used": getattr(result, "tot_used", False),
    }


@app_web.post("/api/chat/stream")
async def chat_stream(request: Request):
    """Streaming chat — sends SSE events as agent processes."""
    try:
        body = await request.json()
    except Exception:
        return StreamingResponse(
            _sse_event({"error": "请求格式错误"}),
            media_type="text/event-stream",
        )
    user_input = body.get("message", "").strip()
    if not user_input:
        return StreamingResponse(
            _sse_event({"error": "empty message"}),
            media_type="text/event-stream",
        )

    async def generate():
        agent = get_agent()
        yield _sse_event({"type": "status", "text": "thinking..."})
        try:
            result = agent.process(user_input)
        except Exception as exc:
            yield _sse_event({"type": "error", "text": str(exc)})
            return
        yield _sse_event({
            "type": "done",
            "text": result.text or "",
            "source": result.source,
            "latency_ms": round(result.latency_ms, 1),
            "tokens": result.tokens or {},
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app_web.get("/api/stats")
async def stats():
    """Get agent statistics."""
    agent = get_agent()
    return agent.stats()


@app_web.get("/api/memory")
async def memory():
    """Get memory system status."""
    agent = get_agent()
    mem = agent.memory
    mem_stats = mem.stats()
    recent = []
    for m in sorted(mem.all(), key=lambda x: -x.updated_at)[:10]:
        recent.append({
            "id": m.id,
            "category": m.category,
            "tier": m.tier.value,
            "title": m.title,
            "content": m.content[:200],
            "confidence": m.confidence,
            "created_at": m.created_at,
        })
    return {"stats": mem_stats, "recent": recent}


@app_web.get("/api/report")
async def report():
    """Get efficiency report."""
    agent = get_agent()
    return {"report": agent.efficiency_report()}


@app_web.post("/api/clear")
async def clear():
    """Clear cache and reset agent."""
    global _agent
    if _agent:
        _agent.cache.ram._cache.clear()
        _agent.cache.embeddings._faiss = None
        _agent.cache.embeddings._index.clear()
        _agent.cache.embeddings._next_id = 0
        _agent._metrics = {"cache": 0, "llm": 0, "total": 0}
        _agent._total_tokens = {"prompt": 0, "completion": 0, "total": 0}
    return {"ok": True}


# ── SSE helper ──────────────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Startup ─────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    if STATIC_DIR.exists():
        app_web.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app_web


def start_server(host: str = "127.0.0.1", port: int = 8220,
                 open_browser: bool = True):
    """Start the web server and optionally open browser."""
    import threading
    import webbrowser

    create_app()

    if open_browser:
        def _open():
            import time as _time
            _time.sleep(0.8)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=_open, daemon=True).start()

    import uvicorn
    uvicorn.run(app_web, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
