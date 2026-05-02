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


# ── ComfyUI API ─────────────────────────────────────────────────────

from agent_compiler.adapters.comfyui import ComfyUIAdapter, ComfyUIConfig

_comfyui: ComfyUIAdapter | None = None


def _get_comfyui() -> ComfyUIAdapter:
    global _comfyui
    if _comfyui is None:
        _comfyui = ComfyUIAdapter(ComfyUIConfig())
    return _comfyui


async def _build_gen_params(body: dict, style: dict, wf_id: str, adapter, prompt: str, neg: str = "") -> dict:
    """Build ComfyUI generation params from user-friendly keys (steps/cfg/seed).

    Maps simple keys like 'steps', 'cfg', 'seed' to node-specific keys
    (e.g. '99:steps', '99:cfg', '99:seed') by inspecting the workflow's
    enabledParams for the correct node ID prefix.
    """
    params = {}
    # Start with fixed params from locked style
    params.update(style.get("fixed_params", {}))

    # Determine the node-specific key prefix from workflow config
    text_key = None
    steps_key = None
    cfg_key = None
    seed_key = None
    neg_key = None

    if wf_id:
        try:
            config = await adapter.get_workflow_config(wf_id)
            template = config.get("workflow_template", config)
            api_cfg = template.get("_api_config", template.get("api_config", {}))
            enabled = api_cfg.get("enabledParams", {})
            for k in enabled:
                if k.endswith(":text") and not text_key:
                    text_key = k
                elif k.endswith(":steps") and not steps_key:
                    steps_key = k
                elif k.endswith(":cfg") and not cfg_key:
                    cfg_key = k
                elif k.endswith(":seed") and not seed_key:
                    seed_key = k
                elif k.endswith(":negative") and not neg_key:
                    neg_key = k
                elif k.endswith(":negative_prompt") and not neg_key:
                    neg_key = k
        except Exception:
            pass  # If workflow fetch fails, params will just use simple keys

    # Apply user params — accept both simple and node-specific keys
    user_params = body.get("params", {})
    for k, v in user_params.items():
        params[k] = v

    # Map simple keys if node-specific key not already set
    if text_key and not any(k.endswith(":text") for k in params):
        params[text_key] = prompt
    if steps_key and "steps" in body and not any(k.endswith(":steps") for k in params):
        params[steps_key] = body["steps"]
    if cfg_key and "cfg" in body and not any(k.endswith(":cfg") for k in params):
        params[cfg_key] = body["cfg"]
    if seed_key and "seed" in body and not any(k.endswith(":seed") for k in params):
        params[seed_key] = body["seed"]
    if neg_key and neg and not any(k.endswith(":negative") or k.endswith(":negative_prompt") for k in params):
        params[neg_key] = neg

    # Ensure text prompt is set — always add it even if no node mapping found
    if text_key and text_key not in params:
        params[text_key] = prompt
    if not text_key:
        # Fallback: try to find any text key or add one
        params["prompt"] = prompt

    return params


@app_web.get("/api/comfyui/workflows")
async def comfyui_workflows():
    """List ComfyUI workflow templates."""
    adapter = _get_comfyui()
    workflows = await adapter.list_workflows()
    return {"workflows": workflows}


@app_web.get("/api/comfyui/workflows/{workflow_id:path}")
async def comfyui_workflow_config(workflow_id: str):
    """Get a workflow's config including editable params."""
    adapter = _get_comfyui()
    config = await adapter.get_workflow_config(workflow_id)
    return config


@app_web.post("/api/comfyui/generate")
async def comfyui_generate(request: Request):
    """Submit a ComfyUI generation task.

    Body: {"workflow_id": "...", "params": {"93:text": "...", "99:seed": 42}, "wait": true}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    workflow_id = body.get("workflow_id", "")
    params = body.get("params")
    wait = body.get("wait", False)

    if not workflow_id:
        return JSONResponse({"error": "workflow_id is required"}, status_code=400)

    print(f"[ComfyUI] 生成请求: workflow={workflow_id}, params={params}, wait={wait}")

    adapter = _get_comfyui()
    result = await adapter.generate(workflow_id, params)

    if not result.get("success"):
        return JSONResponse({"error": result.get("error", "生成失败"), "raw": result}, status_code=500)

    prompt_id = result.get("prompt_id")
    response = {
        "success": True,
        "workflow_id": workflow_id,
        "job_id": result.get("id"),
        "prompt_id": prompt_id,
        "status": "submitted",
    }

    if wait and prompt_id:
        output = await adapter.wait_for_result(prompt_id)
        response["status"] = output["status"]
        response["images"] = output.get("image_urls", [])
        response["error"] = output.get("error")

    return response


@app_web.get("/api/comfyui/status/{prompt_id}")
async def comfyui_status(prompt_id: str):
    """Check status of a generation task."""
    adapter = _get_comfyui()
    result = await adapter.get_prompt_result(prompt_id)
    if result is None:
        queue = await adapter.get_queue_status()
        return {"prompt_id": prompt_id, "found": False,
                "queue_busy": queue.get("busy"),
                "queue_pending": queue.get("pending_count", 0)}
    parsed = adapter._parse_prompt_result(result)
    return {"prompt_id": prompt_id, "found": True, **parsed}


@app_web.get("/api/comfyui/queue")
async def comfyui_queue():
    """Get ComfyUI queue status."""
    adapter = _get_comfyui()
    return await adapter.get_queue_status()


@app_web.get("/api/comfyui/images")
async def comfyui_images():
    """List output images."""
    adapter = _get_comfyui()
    images = await adapter.list_images()
    return {"images": images}


@app_web.get("/api/comfyui/models")
async def comfyui_models():
    """List available models."""
    adapter = _get_comfyui()
    models = await adapter.list_models()
    return {"models": models}


# ── Story Engine ────────────────────────────────────────────────────

import uuid
import copy
from dataclasses import dataclass, field as dc_field

from agent_compiler.web.models import ProjectStore, StyleConfig

# Persistent project store
_store = ProjectStore()

# Legacy aliases for quick in-memory access (synced to disk by _store)
_projects: dict[str, dict] = {}
_panels: dict[str, list[dict]] = {}
_characters: dict[str, list[dict]] = {}
_environments: dict[str, list[dict]] = {}

def _sync_store_to_ram(project_id: str):
    """Sync ProjectStore data into RAM dicts for backward compat during transition."""
    proj = _store.get_project(project_id)
    if proj:
        _projects[project_id] = proj
    _panels[project_id] = _store.get_panels(project_id)
    _characters[project_id] = _store.get_characters(project_id)
    _environments[project_id] = _store.get_environments(project_id)

STORY_BREAKDOWN_PROMPT = """You are a comic/manga storyboarding expert. Given a story premise, break it down into individual panels.

Output ONLY valid JSON array. Each panel object:
{
  "panel_num": number,
  "scene_description": "what the image shows — detailed, visual, in Chinese",
  "characters": ["character names appearing"],
  "dialogue": "spoken text or null",
  "emotion": "emotional tone",
  "camera": "shot type (wide/close-up/medium/overhead/etc)",
  "action": "what is happening in this moment"
}

Generate 6-12 panels. Make them visually descriptive — these will be used as image generation prompts."""


@app_web.post("/api/story/breakdown")
async def story_breakdown(request: Request):
    """Use LLM to break a story premise into comic panels."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    premise = body.get("premise", "").strip()
    style = body.get("style", "日漫风格")
    panel_count = body.get("panel_count", 8)

    if not premise:
        return JSONResponse({"error": "premise is required"}, status_code=400)

    prompt = f"{STORY_BREAKDOWN_PROMPT}\n\nStory premise: {premise}\nStyle: {style}\nTarget panels: {panel_count}"

    agent = get_agent()
    try:
        result = agent.llm.chat(prompt)
        text = result if isinstance(result, str) else str(result)

        # try to extract JSON from response
        json_start = text.find("[")
        json_end = text.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            panels = json.loads(text[json_start:json_end])
        else:
            # fallback: wrap as single panel
            panels = [{"panel_num": 1, "scene_description": text,
                       "characters": [], "dialogue": None, "emotion": "neutral",
                       "camera": "medium", "action": ""}]

        return {"success": True, "panels": panels, "count": len(panels)}
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": f"故事拆解失败: {exc}"}, status_code=500)


# ── Projects ────────────────────────────────────────────────────────

@app_web.get("/api/projects")
async def list_projects():
    """List all projects."""
    return {"projects": _store.list_projects()}


@app_web.post("/api/projects")
async def create_project(request: Request):
    """Create a new project."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    project_id = body.get("id") or str(uuid.uuid4())[:8]
    name = body.get("name", "未命名项目")
    style = body.get("style", "日漫风格")
    workflow_id = body.get("workflow_id", "")

    proj = _store.create_project(project_id, name=name, style_name=style, workflow_id=workflow_id)
    _sync_store_to_ram(proj["id"])
    return {"success": True, "project": proj}


@app_web.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and its panels/characters."""
    _store.delete_project(project_id)
    _projects.pop(project_id, None)
    _panels.pop(project_id, None)
    _characters.pop(project_id, None)
    _environments.pop(project_id, None)
    return {"success": True}


# ── Panels ───────────────────────────────────────────────────────────

@app_web.get("/api/projects/{project_id}/panels")
async def list_panels(project_id: str):
    """Get all panels for a project."""
    return {"panels": _store.get_panels(project_id)}


@app_web.post("/api/projects/{project_id}/panels")
async def save_panels(project_id: str, request: Request):
    """Save/replace all panels for a project."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    panels_data = body.get("panels", [])
    for i, p in enumerate(panels_data):
        if "id" not in p:
            p["id"] = f"panel_{i+1}"
    _store.save_panels(project_id, panels_data)
    _panels[project_id] = panels_data
    return {"success": True, "count": len(panels_data)}


@app_web.put("/api/projects/{project_id}/panels/{panel_id}")
async def update_panel(project_id: str, panel_id: str, request: Request):
    """Update a single panel."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    result = _store.update_panel(project_id, panel_id, body)
    if result:
        _panels[project_id] = _store.get_panels(project_id)
        return {"success": True, "panel": result}
    return JSONResponse({"error": "panel not found"}, status_code=404)


@app_web.post("/api/projects/{project_id}/panels/{panel_id}/regenerate")
async def regenerate_panel(project_id: str, panel_id: str, request: Request):
    """Regenerate a single panel image with adjustable steps/cfg/seed.

    Uses character ref images + environment ref image as img2img input
    for character-consistent panel generation.

    Body: {"steps": 20, "cfg": 7.0, "seed": -1, "workflow_id": "..."}
    Simple keys (steps/cfg/seed) auto-map to node-specific keys.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    style = _store.get_style(project_id) or {}
    if not style.get("is_locked"):
        return JSONResponse({"error": "请先在审图面板锁定风格后再生成分镜图"}, status_code=400)

    panels = _store.get_panels(project_id)
    panel = next((p for p in panels if p.get("id") == panel_id), None)
    if not panel:
        return JSONResponse({"error": f"分镜未找到: {panel_id}"}, status_code=404)

    # Use the workflow from the style or from the request
    wf_id = body.get("workflow_id") or style.get("workflow_id", "")
    if not wf_id:
        return JSONResponse({"error": "请先在风格设置中配置 workflow_id"}, status_code=400)

    # Assemble prompt for this panel
    chars = {c.get("id"): c for c in _store.get_characters(project_id)}
    envs = {e.get("id"): e for e in _store.get_environments(project_id)}
    prompt = _assemble_single_prompt(panel, chars, envs, style)
    neg = style.get("negative_prompt", "低质量, 模糊, 畸形手, 畸形脸")

    adapter = _get_comfyui()
    params = await _build_gen_params(body, style, wf_id, adapter, prompt, neg)

    # Inject character reference images as img2img input
    _inject_ref_images(params, panel, chars, envs)

    result = await adapter.generate(wf_id, params)
    prompt_id = result.get("prompt_id")

    if prompt_id:
        panel["prompt_positive"] = prompt
        panel["prompt_negative"] = neg
        panel["prompt_id"] = prompt_id
        panel["status"] = "generating"
        _store.save_panels(project_id, panels)
        _panels[project_id] = panels

    return {
        "success": True, "prompt_id": prompt_id, "panel_id": panel_id,
        "panel_num": panel.get("panel_num"), "prompt": prompt,
    }


@app_web.get("/api/projects/{project_id}/panels/{panel_id}/generation-status")
async def panel_generation_status(project_id: str, panel_id: str):
    """Poll for a panel's generation result. Returns status + image URLs."""
    panels = _store.get_panels(project_id)
    panel = next((p for p in panels if p.get("id") == panel_id), None)
    if not panel:
        return JSONResponse({"error": "分镜未找到"}, status_code=404)

    prompt_id = panel.get("prompt_id", "")
    if not prompt_id:
        return {"found": False, "status": panel.get("status", "draft")}

    adapter = _get_comfyui()
    result = await adapter.get_prompt_result(prompt_id)
    if result is None:
        queue = await adapter.get_queue_status()
        return {"prompt_id": prompt_id, "found": False, "status": "generating",
                "queue_busy": queue.get("busy"), "queue_pending": queue.get("pending_count", 0)}

    parsed = adapter._parse_prompt_result(result)
    status = parsed.get("status", "pending")
    if status == "done":
        images = parsed.get("image_urls", [])
        if images:
            panel["generated_images"] = images
        panel["status"] = "done"
        _store.save_panels(project_id, panels)
        _panels[project_id] = panels
    elif status == "error":
        panel["status"] = "failed"
        _store.save_panels(project_id, panels)

    return {"prompt_id": prompt_id, "found": True, **parsed}


def _assemble_single_prompt(panel: dict, chars: dict, envs: dict, style: dict) -> str:
    """Assemble a single panel's positive prompt."""
    parts = []
    if style.get("style_name"):
        parts.append(style["style_name"])
    if style.get("positive_prefix"):
        parts.append(style["positive_prefix"])

    scene = panel.get("scene_description", "")
    if scene:
        parts.append(scene)

    # Character details
    char_ids = panel.get("characters", [])
    for cid in char_ids:
        c = chars.get(cid, {})
        if c.get("appearance_detail"):
            parts.append(f"character {c.get('name', cid)}: {c['appearance_detail']}")
        if c.get("clothing_details"):
            parts.append(f"outfit: {c['clothing_details']}")

    # Environment
    env_id = panel.get("environment", "")
    env = envs.get(env_id, {})
    if env.get("description"):
        parts.append(f"background: {env['description']}")

    # Atmosphere & camera
    emotion = panel.get("emotion", "")
    if emotion:
        parts.append(f"{emotion} atmosphere")
    camera = panel.get("camera", "")
    if camera:
        parts.append(f"{camera} shot")

    return ", ".join(parts)


def _inject_ref_images(params: dict, panel: dict, chars: dict, envs: dict):
    """Inject character/environment reference images into params for img2img."""
    char_refs = []
    for cid in panel.get("characters", []):
        c = chars.get(cid, {})
        for fld in ["ref_image_full", "ref_image_front", "ref_image_url"]:
            url = c.get(fld, "")
            if url:
                char_refs.append(url)
                break

    env_id = panel.get("environment", "")
    env = envs.get(env_id, {})
    env_ref = env.get("ref_image_url", "")

    # Find img2img image slots in params
    for k in list(params.keys()):
        if ":image" in k or ":input_image" in k or ":ref_image" in k:
            if char_refs:
                params[k] = char_refs[0]  # primary char ref
            continue
        if ":image2" in k or ":ref_image2" in k:
            if len(char_refs) > 1:
                params[k] = char_refs[1]
            continue
        if ":env_image" in k or ":bg_image" in k:
            if env_ref:
                params[k] = env_ref
            continue


# ── Characters ──────────────────────────────────────────────────────

@app_web.get("/api/projects/{project_id}/characters")
async def list_characters(project_id: str):
    """Get all characters for a project."""
    return {"characters": _store.get_characters(project_id)}


@app_web.post("/api/projects/{project_id}/characters")
async def save_characters(project_id: str, request: Request):
    """Save/replace all characters for a project."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    chars = body.get("characters", [])
    for i, c in enumerate(chars):
        if "id" not in c:
            c["id"] = f"char_{i+1}"
    _store.save_characters(project_id, chars)
    _characters[project_id] = chars
    return {"success": True, "count": len(chars)}


# ── Comprehensive Breakdown ──────────────────────────────────────────

COMPREHENSIVE_BREAKDOWN_PROMPT = """你是一位专业的漫画/影视编剧大师。根据故事梗概，输出完整的漫剧制作方案。

输出格式必须是严格的 JSON，按以下结构：

{{
  "story_title": "故事标题",
  "story_summary": "200字以内的故事简介",
  "story_themes": ["主题1", "主题2"],

  "characters": [
    {{
      "id": "char_1",
      "name": "角色名",
      "description": "角色背景和性格描述",
      "appearance_detail": "三视图级别的外形描述：身高、体型、发型、脸型、肤色、眼睛颜色、年龄感等。用于生成角色参考图。",
      "clothing_details": "详细的服装描述：款式、颜色、材质、纹理、配饰、鞋履等。用于提示词保证角色穿着一致性。",
      "voice_style": "声音特点描述(用于配音)",
      "personality": "性格特点"
    }}
  ],

  "environments": [
    {{
      "id": "env_1",
      "name": "场景名",
      "description": "环境的详细视觉描述，含光影、氛围、色彩、建筑/自然元素等"
    }}
  ],

  "panels": [
    {{
      "panel_num": 1,
      "scene_description": "画面描述——用作图片生成的提示词基础，详细视觉化，含构图、光影、色彩",
      "characters": ["char_1"],
      "environment": "env_1",
      "action": "角色在这个分镜中的具体动作",
      "dialogue": "角色台词",
      "voice_line": "配音文本（含语气指示，如(愤怒地)(轻声)）",
      "emotion": "情绪/氛围",
      "camera": "镜头类型(wide/close-up/medium/overhead/dutch angle/POV等)",
      "video_motion": "视频运动描述（镜头移动方向、物体运动轨迹等，用于图生视频）"
    }}
  ]
}}

生成规则：
1. {target_characters} 个主要角色，每个角色必须包含 appearance_detail 和 clothing_details（用于保持角色一致性）
2. 至少 {min_environments} 个场景环境
3. 约 {target_panels} 个分镜，每个分镜必须引用已有的角色 ID 和环境 ID
4. scene_description 要详细到可以直接作为 Stable Diffusion 提示词
5. voice_line 要包含语气指示，用于配音演员参考
6. video_motion 用于后续图生视频工作流的附加运动描述
7. **重要：以下字段必须用英文输出（用于 ComfyUI 生图提示词）**：
   - characters[].appearance_detail
   - characters[].clothing_details
   - environments[].description
   - panels[].scene_description
   - panels[].emotion
   - panels[].camera
   - panels[].video_motion
   其他字段（name, description, dialogue, voice_line 等）用中文

故事前提：{premise}
风格/类型：{genre}"""


@app_web.post("/api/story/comprehensive-breakdown")
async def comprehensive_breakdown(request: Request):
    """一次 LLM 调用生成：故事、详细角色（三视图级）、环境、分镜（含角色/环境映射、台词、配音、动作、视频运动）"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    premise = body.get("premise", "").strip()
    genre = body.get("genre", "日漫风格")
    target_characters = body.get("target_characters", 3)
    target_panels = body.get("target_panels", 8)
    project_id = body.get("project_id", "")

    if not premise:
        return JSONResponse({"error": "premise is required"}, status_code=400)

    min_environments = max(1, target_characters)
    prompt = COMPREHENSIVE_BREAKDOWN_PROMPT.format(
        premise=premise, genre=genre,
        target_characters=target_characters,
        target_panels=target_panels,
        min_environments=min_environments,
    )

    agent = get_agent()
    try:
        result = agent.llm.chat(prompt, system="你是一个专业的漫画编剧。只输出 JSON，不要 markdown，不要解释。")
        text = result if isinstance(result, str) else str(result)

        # Extract JSON from response
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(text[json_start:json_end])
        else:
            return JSONResponse({"error": "LLM 未返回有效 JSON", "raw": text[:500]}, status_code=500)

        # Auto-save to project store if project_id provided
        if project_id:
            chars = data.get("characters", [])
            envs = data.get("environments", [])
            panels = data.get("panels", [])
            for i, c in enumerate(chars):
                if "id" not in c:
                    c["id"] = f"char_{i+1}"
            for i, e in enumerate(envs):
                if "id" not in e:
                    e["id"] = f"env_{i+1}"
            for i, p in enumerate(panels):
                if "id" not in p:
                    p["id"] = f"panel_{p.get('panel_num', i+1)}"
            _store.save_characters(project_id, chars)
            _store.save_environments(project_id, envs)

            # Auto-assemble English positive/negative prompts for each panel
            style = _store.get_style(project_id) or {}
            chars_map = {c["id"]: c for c in chars}
            envs_map = {e["id"]: e for e in envs}
            for p in panels:
                positive = _assemble_single_prompt(p, chars_map, envs_map, style)
                negative = style.get("negative_prompt", "low quality, blurry, deformed hands, deformed face, text, watermark")
                p["prompt_positive"] = positive
                p["prompt_negative"] = negative

            _store.save_panels(project_id, panels)
            _characters[project_id] = chars
            _environments[project_id] = envs
            _panels[project_id] = panels
            data["saved_to_project"] = project_id

        return {"success": True, **data}
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": f"JSON 解析失败: {exc}", "raw": text[:500]}, status_code=500)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": f"综合拆解失败: {exc}"}, status_code=500)


# ── Style Lock ───────────────────────────────────────────────────────

@app_web.post("/api/projects/{project_id}/style")
async def save_style_config(project_id: str, request: Request):
    """Save/update project style configuration."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    cfg = StyleConfig(
        project_id=project_id,
        style_name=body.get("style_name", "日漫风格"),
        workflow_id=body.get("workflow_id", ""),
        positive_prefix=body.get("positive_prefix", "masterpiece, best quality"),
        negative_prompt=body.get("negative_prompt", "低质量, 模糊, 畸形手, 畸形脸, 文字, 水印"),
        fixed_params=body.get("fixed_params", {}),
        character_ref_workflow=body.get("character_ref_workflow", body.get("workflow_id", "")),
        environment_ref_workflow=body.get("environment_ref_workflow", body.get("workflow_id", "")),
        video_workflow=body.get("video_workflow", ""),
    )
    _store.save_style(project_id, cfg)
    return {"success": True, "style": _store.get_style(project_id)}


@app_web.get("/api/projects/{project_id}/style")
async def get_style_config(project_id: str):
    """Get current style configuration."""
    cfg = _store.get_style(project_id)
    if not cfg:
        return {"style": None, "message": "请先设置风格配置"}
    return {"style": cfg}


@app_web.post("/api/projects/{project_id}/style/lock")
async def lock_style(project_id: str):
    """Lock style — after lock, all panels use this config."""
    cfg = _store.lock_style(project_id)
    return {"success": True, "style": cfg, "message": "风格已锁定，所有分镜将使用此配置"}


@app_web.post("/api/projects/{project_id}/style/unlock")
async def unlock_style(project_id: str):
    """Unlock style to allow changes."""
    cfg = _store.unlock_style(project_id)
    return {"success": True, "style": cfg}


@app_web.post("/api/projects/{project_id}/style/preview")
async def generate_style_preview(project_id: str, request: Request):
    """Generate N preview images for style review (审图). Returns prompt_ids to poll."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    count = body.get("count", 10)
    seeds = body.get("seeds", [-1] * count)
    style = _store.get_style(project_id) or {}
    wf_id = style.get("workflow_id", "")

    if not wf_id:
        return JSONResponse({"error": "请先选择生图工作流"}, status_code=400)

    adapter = _get_comfyui()
    prompt_ids = []

    for i in range(count):
        seed = seeds[i] if i < len(seeds) else -1
        prompt = (
            f"{style.get('positive_prefix', 'masterpiece, high quality')}, "
            f"a kitten sitting next to a flower pot, "
            f"soft lighting, high quality illustration"
        )
        params = await _build_gen_params(
            {"steps": 20, "cfg": 7, "seed": seed},
            style, wf_id, adapter, prompt
        )
        result = await adapter.generate(wf_id, params)
        if result.get("prompt_id"):
            prompt_ids.append(result["prompt_id"])

    return {"success": True, "prompt_ids": prompt_ids, "count": len(prompt_ids)}


@app_web.post("/api/projects/{project_id}/style/preview-image")
async def save_preview_image(project_id: str, request: Request):
    """Save a completed preview image URL to the style config."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    image_url = body.get("image_url", "")
    if image_url:
        style = _store.get_style(project_id) or {}
        previews = style.get("preview_images", [])
        if image_url not in previews:
            previews.append(image_url)
            style["preview_images"] = previews
            _store.save_style(project_id, _dict_to_style(style))

    return {"success": True}


@app_web.post("/api/projects/{project_id}/style/clear-previews")
async def clear_preview_images(project_id: str):
    """Clear all preview images for this project."""
    style = _store.get_style(project_id) or {}
    style["preview_images"] = []
    style["selected_preview"] = -1
    _store.save_style(project_id, _dict_to_style(style))
    return {"success": True, "message": "预览图已清理"}


@app_web.get("/api/projects/{project_id}/style/workflow-params")
async def get_workflow_params(project_id: str):
    """Extract model, LoRA, sampler info from the current workflow config.

    Returns details about the workflow nodes so the user can see what
    model/LoRA/sampler will be used for all subsequent generations.
    """
    style = _store.get_style(project_id) or {}
    wf_id = style.get("workflow_id", "")
    if not wf_id:
        return JSONResponse({"error": "请先选择生图工作流"}, status_code=400)

    adapter = _get_comfyui()
    try:
        config = await adapter.get_workflow_config(wf_id)
    except Exception as e:
        return JSONResponse({"error": f"获取工作流配置失败: {e}"}, status_code=400)

    nodes = config.get("nodes", [])
    params_info = {
        "workflow_id": wf_id,
        "checkpoints": [],   # 大模型/底模
        "loras": [],         # LoRA 模型
        "sampler": "",       # 采样器
        "scheduler": "",     # 调度器
        "steps": "",         # 步数
        "cfg": "",           # CFG scale
        "resolution": "",    # 分辨率
        "positive_prompt": "",  # 正向提示词节点
        "negative_prompt": "",  # 负向提示词节点
    }

    for node in nodes:
        ntype = node.get("type", "")
        title = node.get("title", "")
        widget_values = node.get("widget_values", {})
        inputs = node.get("inputs", [])

        # Checkpoint/Model loader
        if "CheckpointLoader" in ntype or "UNETLoader" in ntype:
            for inp in inputs:
                if inp.get("name") == "ckpt_name" or inp.get("name") == "unet_name":
                    params_info["checkpoints"].append(inp.get("default", ""))
            if title and title not in params_info["checkpoints"]:
                params_info["checkpoints"].append(title)

        # LoRA loader
        if "LoraLoader" in ntype:
            lora_info = {}
            for inp in inputs:
                if inp.get("name") == "lora_name":
                    lora_info["name"] = inp.get("default", "")
                if inp.get("name") == "strength_model" or inp.get("name") == "strength":
                    lora_info["strength"] = inp.get("default", 1.0)
            if lora_info:
                params_info["loras"].append(lora_info)

        # Sampler
        if "KSampler" in ntype or "Sampler" in ntype:
            for inp in inputs:
                if inp.get("name") == "sampler_name":
                    params_info["sampler"] = inp.get("default", "")
                if inp.get("name") == "scheduler":
                    params_info["scheduler"] = inp.get("default", "")
                if inp.get("name") == "steps":
                    params_info["steps"] = inp.get("default", "")
                if inp.get("name") == "cfg":
                    params_info["cfg"] = inp.get("default", "")

        # CLIP Text Encode (positive/negative prompts)
        if "CLIPTextEncode" in ntype:
            widget_text = widget_values.get("text", "")
            if "negative" in title.lower():
                if not params_info["negative_prompt"]:
                    params_info["negative_prompt"] = widget_text
            else:
                if not params_info["positive_prompt"]:
                    params_info["positive_prompt"] = widget_text

        # Empty Latent Image (resolution)
        if "EmptyLatentImage" in ntype:
            w = h = ""
            for inp in inputs:
                if inp.get("name") == "width":
                    w = inp.get("default", "")
                if inp.get("name") == "height":
                    h = inp.get("default", "")
            if w and h:
                params_info["resolution"] = f"{w}x{h}"

    return {"success": True, "params": params_info}


def _dict_to_style(d: dict) -> "StyleConfig":
    from agent_compiler.web.models import StyleConfig
    return StyleConfig(
        project_id=d.get("project_id", ""),
        style_name=d.get("style_name", "日漫风格"),
        workflow_id=d.get("workflow_id", ""),
        positive_prefix=d.get("positive_prefix", ""),
        negative_prompt=d.get("negative_prompt", ""),
        fixed_params=d.get("fixed_params", {}),
        character_ref_workflow=d.get("character_ref_workflow", ""),
        environment_ref_workflow=d.get("environment_ref_workflow", ""),
        video_workflow=d.get("video_workflow", ""),
        preview_images=d.get("preview_images", []),
        selected_preview=d.get("selected_preview", -1),
        is_locked=d.get("is_locked", False),
        locked_at=d.get("locked_at", 0),
    )


# ── Environments ─────────────────────────────────────────────────────

@app_web.get("/api/projects/{project_id}/environments")
async def list_environments(project_id: str):
    """Get all environments for a project."""
    return {"environments": _store.get_environments(project_id)}


@app_web.post("/api/projects/{project_id}/environments")
async def save_environments(project_id: str, request: Request):
    """Save/replace all environments for a project."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    envs = body.get("environments", [])
    for i, e in enumerate(envs):
        if "id" not in e:
            e["id"] = f"env_{i+1}"
    _store.save_environments(project_id, envs)
    _environments[project_id] = envs
    return {"success": True, "count": len(envs)}


@app_web.post("/api/projects/{project_id}/environments/generate-ref")
async def generate_environment_ref(project_id: str, request: Request):
    """Generate reference image for an environment.

    Body: {"environment_id": "env_1", "steps": 20, "cfg": 7.0, "seed": -1}
    Simple keys (steps/cfg/seed) are auto-mapped to node-specific keys via
    the workflow's enabledParams, so the user doesn't need to know node IDs.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    env_id = body.get("environment_id", "")
    style = _store.get_style(project_id) or {}
    wf_id = body.get("workflow_id") or style.get("environment_ref_workflow", "")

    if not wf_id:
        return JSONResponse({"error": "请先在风格设置中配置 environment_ref_workflow"}, status_code=400)

    envs = _store.get_environments(project_id)
    env = next((e for e in envs if e.get("id") == env_id), None)
    if not env:
        return JSONResponse({"error": f"环境未找到: {env_id}"}, status_code=404)

    prompt = (
        f"{style.get('style_name', '')}, {env.get('description', '')}, "
        f"detailed environment, high quality, "
        f"{style.get('positive_prefix', 'masterpiece, high quality')}"
    )

    adapter = _get_comfyui()
    params = await _build_gen_params(body, style, wf_id, adapter, prompt)

    result = await adapter.generate(wf_id, params)
    prompt_id = result.get("prompt_id")

    if prompt_id:
        env["ref_prompt"] = prompt
        env["ref_prompt_id"] = prompt_id
        env["status"] = "generating"
        _store.save_environments(project_id, envs)
        _environments[project_id] = envs

    return {"success": True, "prompt_id": prompt_id, "environment_id": env_id, "prompt": prompt}


@app_web.get("/api/projects/{project_id}/environments/{env_id}/ref-status")
async def environment_ref_status(project_id: str, env_id: str):
    """Poll for environment reference image generation result."""
    envs = _store.get_environments(project_id)
    env = next((e for e in envs if e.get("id") == env_id), None)
    if not env:
        return JSONResponse({"error": "环境未找到"}, status_code=404)

    prompt_id = env.get("ref_prompt_id", "")
    if not prompt_id:
        return {"found": False, "status": env.get("status", "draft")}

    adapter = _get_comfyui()
    result = await adapter.get_prompt_result(prompt_id)
    if result is None:
        queue = await adapter.get_queue_status()
        return {"prompt_id": prompt_id, "found": False, "status": "generating",
                "queue_busy": queue.get("busy"), "queue_pending": queue.get("pending_count", 0)}

    parsed = adapter._parse_prompt_result(result)
    status = parsed.get("status", "pending")
    if status == "done":
        images = parsed.get("image_urls", [])
        if images:
            env["ref_image_url"] = images[0]
        env["status"] = "done"
        _store.save_environments(project_id, envs)
        _environments[project_id] = envs
    elif status == "error":
        env["status"] = "failed"
        _store.save_environments(project_id, envs)

    return {"prompt_id": prompt_id, "found": True, **parsed}


@app_web.post("/api/projects/{project_id}/characters/generate-ref")
async def generate_character_ref(project_id: str, request: Request):
    """Generate three-view reference images for a character.

    Body: {"character_id": "char_1", "steps": 20, "cfg": 7.0, "seed": -1}
    Simple keys (steps/cfg/seed) are auto-mapped to node-specific keys via
    the workflow's enabledParams, so the user doesn't need to know node IDs.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    char_id = body.get("character_id", "")
    style = _store.get_style(project_id) or {}
    wf_id = body.get("workflow_id") or style.get("character_ref_workflow", "")

    if not wf_id:
        return JSONResponse({"error": "请先在风格设置中配置 character_ref_workflow"}, status_code=400)

    chars = _store.get_characters(project_id)
    char = next((c for c in chars if c.get("id") == char_id), None)
    if not char:
        return JSONResponse({"error": f"角色未找到: {char_id}"}, status_code=404)

    # Build character reference prompt with three-view + white background
    prompt = (
        f"{style.get('style_name', '')}, character design sheet, full body, "
        f"front view, {char.get('appearance_detail', '')}, "
        f"{char.get('clothing_details', '')}, "
        f"white background, turnaround reference, character reference sheet, solo, "
        f"{style.get('positive_prefix', 'masterpiece, high quality')}"
    )
    neg = style.get("negative_prompt", "")

    adapter = _get_comfyui()
    params = await _build_gen_params(body, style, wf_id, adapter, prompt, neg)

    result = await adapter.generate(wf_id, params)
    prompt_id = result.get("prompt_id")

    if prompt_id:
        char["ref_prompt"] = prompt
        char["ref_prompt_id"] = prompt_id
        char["status"] = "generating"
        _store.save_characters(project_id, chars)
        _characters[project_id] = chars

    return {"success": True, "prompt_id": prompt_id, "character_id": char_id, "prompt": prompt}


@app_web.get("/api/projects/{project_id}/characters/{char_id}/ref-status")
async def character_ref_status(project_id: str, char_id: str):
    """Poll for character reference image generation result."""
    chars = _store.get_characters(project_id)
    char = next((c for c in chars if c.get("id") == char_id), None)
    if not char:
        return JSONResponse({"error": "角色未找到"}, status_code=404)

    prompt_id = char.get("ref_prompt_id", "")
    if not prompt_id:
        return {"found": False, "status": char.get("status", "draft")}

    adapter = _get_comfyui()
    result = await adapter.get_prompt_result(prompt_id)
    if result is None:
        queue = await adapter.get_queue_status()
        return {"prompt_id": prompt_id, "found": False, "status": "generating",
                "queue_busy": queue.get("busy"), "queue_pending": queue.get("pending_count", 0)}

    parsed = adapter._parse_prompt_result(result)
    status = parsed.get("status", "pending")
    if status == "done":
        images = parsed.get("image_urls", [])
        if images:
            char["ref_image_full"] = images[0]
            if len(images) > 1:
                char["ref_image_front"] = images[0]
                char["ref_image_side"] = images[1] if len(images) > 1 else images[0]
                char["ref_image_back"] = images[2] if len(images) > 2 else images[0]
        char["status"] = "done"
        _store.save_characters(project_id, chars)
        _characters[project_id] = chars
    elif status == "error":
        char["status"] = "failed"
        _store.save_characters(project_id, chars)

    return {"prompt_id": prompt_id, "found": True, **parsed}


# ── Table View ───────────────────────────────────────────────────────

@app_web.get("/api/projects/{project_id}/table-view")
async def table_view(project_id: str):
    """Get table mapping: panel ↔ character ↔ environment ↔ dialogue ↔ voice."""
    return _store.build_table_view(project_id)


# ── Video Generation ─────────────────────────────────────────────────

@app_web.post("/api/projects/{project_id}/video")
async def generate_video(project_id: str, request: Request):
    """Submit panels to video generation workflow (img2video)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    panel_ids = body.get("panel_ids", [])
    style = _store.get_style(project_id) or {}
    video_wf = body.get("video_workflow_id") or style.get("video_workflow", "")

    if not video_wf:
        return JSONResponse({"error": "请先在风格设置中配置 video_workflow"}, status_code=400)

    panels = _store.get_panels(project_id)
    target_panels = [p for p in panels if not panel_ids or p.get("id") in panel_ids]

    if not target_panels:
        return JSONResponse({"error": "没有找到可生成视频的分镜"}, status_code=400)

    adapter = _get_comfyui()
    results = []
    for p in target_panels:
        images = p.get("generated_images", [])
        if not images:
            results.append({"panel_id": p.get("id"), "error": "该分镜没有生成图片"})
            continue

        params = body.get("params", {})
        params.update({k: v for k, v in style.get("fixed_params", {}).items() if k not in params})
        # Set the input image to the panel's generated image
        for k in params:
            if ":image" in k and images:
                params[k] = images[0]
                break

        result = await adapter.generate(video_wf, params)
        if result.get("success"):
            p["video_prompt_id"] = result.get("prompt_id")
            p["video_status"] = "generating"
        results.append({
            "panel_id": p.get("id"),
            "panel_num": p.get("panel_num"),
            "prompt_id": result.get("prompt_id"),
            "success": result.get("success", False),
        })

    _store.save_panels(project_id, panels)
    _panels[project_id] = panels

    return {"success": True, "video_workflow": video_wf, "total": len(results), "results": results}


# ── Enhanced Prompt Assembly ─────────────────────────────────────────

@app_web.post("/api/projects/{project_id}/assemble-prompts")
async def assemble_all_prompts(project_id: str, request: Request):
    """Assemble prompts for all panels using style config + character refs + environment refs."""
    style = _store.get_style(project_id) or {}
    panels = _store.get_panels(project_id)
    chars = {c.get("id"): c for c in _store.get_characters(project_id)}
    envs = {e.get("id"): e for e in _store.get_environments(project_id)}

    if not style:
        return JSONResponse({"error": "请先设置风格配置"}, status_code=400)

    assembled = []
    for p in panels:
        positive = _assemble_single_prompt(p, chars, envs, style)
        negative = style.get("negative_prompt", "低质量, 模糊, 畸形")

        # Collect character ref image URLs for img2img
        char_ref_urls = []
        for cid in p.get("characters", []):
            c = chars.get(cid, {})
            for k in ("ref_image_full", "ref_image_front"):
                url = c.get(k, "")
                if url:
                    char_ref_urls.append({"character_id": cid, "image_url": url, "name": c.get("name", "")})
                    break

        env = envs.get(p.get("environment", ""), {})
        env_ref_url = env.get("ref_image_url", "")

        p["prompt_positive"] = positive
        p["prompt_negative"] = negative
        p["character_refs"] = [cr["image_url"] for cr in char_ref_urls]
        p["environment_ref"] = env_ref_url if env_ref_url else ""
        p["status"] = "prompt_ready"

        assembled.append({
            "panel_id": p.get("id"),
            "panel_num": p.get("panel_num"),
            "positive": positive,
            "negative": negative,
            "img2img_char_refs": char_ref_urls,
            "img2img_env_ref": env_ref_url if env_ref_url else None,
        })

    _store.save_panels(project_id, panels)
    _panels[project_id] = panels

    return {"success": True, "count": len(assembled), "panels": assembled}


# ── Skill Generation ─────────────────────────────────────────────────

@app_web.post("/api/skills/auto-generate")
async def auto_generate_skill(request: Request):
    """Auto-generate a reusable skill from a successful pipeline."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    name = body.get("name", "").strip()
    description = body.get("description", "").strip()
    steps = body.get("pipeline_steps", [])
    project_id = body.get("source_project_id", "")

    if not name or not description:
        return JSONResponse({"error": "name and description are required"}, status_code=400)

    from agent_compiler.web.models import SkillTemplate
    template = SkillTemplate(
        name=name, description=description,
        pipeline_steps=steps, source_project_id=project_id,
        created_at=time.time(),
    )
    data = _store.save_skill(template)

    # Reload skills in the agent
    agent = get_agent()
    if hasattr(agent, 'skills'):
        agent.skills.discover()

    return {"success": True, "skill": data, "message": f"技能 '{name}' 已生成，下次说 '{description.split(',')[0]}' 自动触发"}


@app_web.get("/api/skills")
async def list_skills():
    """List auto-generated comic skills."""
    return {"skills": _store.list_skills()}


@app_web.delete("/api/skills/{skill_id}")
async def delete_skill(skill_id: str):
    """Delete a skill template by ID."""
    skill_file = _store._skills_dir / f"{skill_id}.json"
    if skill_file.exists():
        skill_file.unlink()
        return {"success": True, "message": f"技能 '{skill_id}' 已删除"}
    return JSONResponse({"error": "skill not found"}, status_code=404)


@app_web.get("/api/memory/list")
async def list_memories():
    """List all agent memories with full content."""
    agent = get_agent()
    mem = agent.memory
    all_memories = []
    for m in sorted(mem.all(), key=lambda x: -x.updated_at):
        all_memories.append({
            "id": m.id,
            "category": m.category,
            "tier": m.tier.value,
            "title": m.title,
            "content": m.content,
            "confidence": m.confidence,
            "created_at": m.created_at,
            "updated_at": m.updated_at,
        })
    return {"memories": all_memories}


@app_web.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    agent = get_agent()
    mem = agent.memory
    all_ids = [m.id for m in mem.all()]
    if memory_id not in all_ids:
        return JSONResponse({"error": "memory not found"}, status_code=404)
    mem.delete(memory_id)
    return {"success": True, "message": f"记忆 '{memory_id}' 已删除"}


# ── Batch Generation ────────────────────────────────────────────────

@app_web.post("/api/comfyui/batch")
async def comfyui_batch_generate(request: Request):
    """Batch generate images for multiple panels or parameter grid.

    Body: {
        "workflow_id": "...",
        "panels": [{"params": {...}}, {"params": {...}}],  // one per panel
        OR
        "grid": {"param": "99:seed", "values": [1,2,3], "base_params": {...}}
    }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    workflow_id = body.get("workflow_id", "")
    panels = body.get("panels")
    grid = body.get("grid")
    wait = body.get("wait", False)

    if not workflow_id:
        return JSONResponse({"error": "workflow_id is required"}, status_code=400)

    adapter = _get_comfyui()

    # Build param sets
    param_sets = []
    if panels:
        param_sets = [p.get("params", {}) for p in panels]
    elif grid:
        param_name = grid.get("param", "")
        values = grid.get("values", [])
        base_params = copy.deepcopy(grid.get("base_params", {}))
        for val in values:
            p = copy.deepcopy(base_params)
            p[param_name] = val
            param_sets.append(p)

    if not param_sets:
        return JSONResponse({"error": "no panels or grid specified"}, status_code=400)

    results = []
    for i, ps in enumerate(param_sets):
        print(f"[ComfyUI Batch] {i+1}/{len(param_sets)}: params={ps}")
        result = await adapter.generate(workflow_id, ps)
        if result.get("success") and wait:
            output = await adapter.wait_for_result(result.get("prompt_id"))
            result["output"] = output
        results.append({
            "index": i,
            "params": ps,
            "prompt_id": result.get("prompt_id"),
            "status": result.get("output", {}).get("status", "submitted") if wait else "submitted",
            "images": result.get("output", {}).get("image_urls", []) if wait else [],
        })

    return {"success": True, "workflow_id": workflow_id, "total": len(results), "results": results}


# ── Prompt Assembly ─────────────────────────────────────────────────

@app_web.post("/api/story/assemble-prompt")
async def assemble_prompt(request: Request):
    """Assemble a ComfyUI-ready prompt from a panel + character info."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    panel = body.get("panel", {})
    characters = body.get("characters", [])
    style = body.get("style", "日漫风格")

    scene = panel.get("scene_description", "")
    emotion = panel.get("emotion", "")
    camera = panel.get("camera", "")

    # Build character descriptions
    char_descs = []
    for c in characters:
        name = c.get("name", "")
        desc = c.get("description", "")
        if name:
            char_descs.append(f"{name}({desc})" if desc else name)

    positive = f"{style}, {scene}"
    if char_descs:
        positive += f", 角色: {', '.join(char_descs)}"
    if emotion:
        positive += f", {emotion}氛围"
    if camera:
        positive += f", {camera}镜头"

    negative = body.get("negative", "低质量, 模糊, 畸形手, 畸形脸, 文字, 水印")

    return {
        "positive": positive,
        "negative": negative,
        "scene": scene,
        "characters": char_descs,
    }


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
