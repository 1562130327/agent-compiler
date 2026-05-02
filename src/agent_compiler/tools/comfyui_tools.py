"""ComfyUI tools — registered into Agent tool registry for ReAct loop access.

Each tool is a pure function (params in, dict out). The ComfyUIAdapter
singleton is configured on first use.
"""

from __future__ import annotations

from typing import Any

from agent_compiler.adapters.comfyui import ComfyUIAdapter, ComfyUIConfig

_adapter: ComfyUIAdapter | None = None


def _get_adapter() -> ComfyUIAdapter:
    global _adapter
    if _adapter is None:
        _adapter = ComfyUIAdapter(ComfyUIConfig())
    return _adapter


def configure_adapter(base_url: str) -> None:
    global _adapter
    _adapter = ComfyUIAdapter(ComfyUIConfig(base_url=base_url))


# ── Tool functions ─────────────────────────────────────────────────

def comfyui_list_workflows() -> dict:
    """List all ComfyUI workflow templates."""
    import asyncio
    adapter = _get_adapter()
    workflows = asyncio.run(adapter.list_workflows())
    return {"count": len(workflows), "workflows": workflows}


def comfyui_get_workflow_config(workflow_id: str) -> dict:
    """Get a workflow's full configuration including editable parameters.

    Returns the workflow template and the _api_config which shows:
    - enabledParams: which parameters can be changed
    - formValues: current/default parameter values
    - customLabels: human-readable labels
    """
    import asyncio
    adapter = _get_adapter()
    config = asyncio.run(adapter.get_workflow_config(workflow_id))
    # extract just the API-relevant parts for the LLM
    api_config = config.get("api_config") or config.get("_api_config") or {}
    editable = {k: v for k, v in api_config.get("enabledParams", {}).items() if v}
    return {
        "workflow_id": workflow_id,
        "editable_params": editable,
        "current_values": api_config.get("formValues", {}),
        "labels": api_config.get("customLabels", {}),
    }


def comfyui_generate(workflow_id: str, params: dict[str, Any] | None = None,
                     wait: bool = False) -> dict:
    """Generate an image/video using a ComfyUI workflow.

    Args:
        workflow_id: ID of the workflow (e.g. "C07-文生图-Zimage-Nunchaku加速")
        params: parameter overrides as {node_id:param: value}
                e.g. {"93:text": "a cat", "99:seed": 42}
        wait: if True, wait for generation to complete before returning
    """
    import asyncio
    adapter = _get_adapter()
    result = asyncio.run(adapter.generate(workflow_id, params))
    if not result.get("success"):
        return {"error": result.get("error", "Unknown error"), "workflow_id": workflow_id}

    prompt_id = result.get("prompt_id")

    if wait and prompt_id:
        output = asyncio.run(adapter.wait_for_result(prompt_id))
        return {
            "workflow_id": workflow_id,
            "prompt_id": prompt_id,
            "status": output["status"],
            "images": output.get("image_urls", []),
            "image_count": len(output.get("images", [])),
        }

    return {
        "workflow_id": workflow_id,
        "prompt_id": prompt_id,
        "job_id": result.get("id"),
        "status": "submitted",
    }


def comfyui_check_status(prompt_id: str) -> dict:
    """Check the status of a submitted generation job."""
    import asyncio
    adapter = _get_adapter()
    result = asyncio.run(adapter.get_prompt_result(prompt_id))
    if result is None:
        queue = asyncio.run(adapter.get_queue_status())
        return {"prompt_id": prompt_id, "found": False,
                "queue_busy": queue.get("busy"),
                "queue_pending": queue.get("pending_count", 0)}
    parsed = adapter._parse_prompt_result(result)
    return {"prompt_id": prompt_id, "found": True, **parsed}


def comfyui_list_images() -> dict:
    """List all output images on the ComfyUI instance."""
    import asyncio
    adapter = _get_adapter()
    images = asyncio.run(adapter.list_images())
    return {"count": len(images), "images": images[:20]}


def comfyui_queue_status() -> dict:
    """Get ComfyUI queue status (busy/running/pending counts)."""
    import asyncio
    adapter = _get_adapter()
    return asyncio.run(adapter.get_queue_status())


# ── Registration maps ──────────────────────────────────────────────

_COMFYUI_TOOLS: dict = {
    "comfyui_list_workflows": comfyui_list_workflows,
    "comfyui_get_workflow_config": comfyui_get_workflow_config,
    "comfyui_generate": comfyui_generate,
    "comfyui_check_status": comfyui_check_status,
    "comfyui_list_images": comfyui_list_images,
    "comfyui_queue_status": comfyui_queue_status,
}

_COMFYUI_TOOL_DEFS: dict = {
    "comfyui_list_workflows": {
        "name": "comfyui_list_workflows",
        "description": "列出 ComfyUI 上所有可用的工作流模板。用于了解有哪些生图/视频/音频工作流可用。",
        "fn": comfyui_list_workflows,
        "params_schema": {"type": "object", "properties": {}},
    },
    "comfyui_get_workflow_config": {
        "name": "comfyui_get_workflow_config",
        "description": "获取指定工作流的完整配置，包括可修改的参数列表和当前默认值。在生图之前先用此工具了解可以改哪些参数（提示词/尺寸/种子等）。",
        "fn": comfyui_get_workflow_config,
        "params_schema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "description": "工作流 ID，如 C07-文生图-Zimage-Nunchaku加速"},
            },
            "required": ["workflow_id"],
        },
    },
    "comfyui_generate": {
        "name": "comfyui_generate",
        "description": "使用指定工作流和参数提交生图/视频任务。先调用 comfyui_get_workflow_config 了解参数，再构建 params 字典提交。",
        "fn": comfyui_generate,
        "params_schema": {
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "description": "工作流 ID"},
                "params": {"type": "object", "description": "参数覆盖，如 {\"93:text\": \"一只猫\", \"99:seed\": 42}"},
                "wait": {"type": "boolean", "description": "是否等待生成完成（默认 false，建议对单张图设为 true）", "default": False},
            },
            "required": ["workflow_id", "params"],
        },
    },
    "comfyui_check_status": {
        "name": "comfyui_check_status",
        "description": "查询已提交生成任务的状态和结果。如果任务完成，会返回生成的图片链接。",
        "fn": comfyui_check_status,
        "params_schema": {
            "type": "object",
            "properties": {
                "prompt_id": {"type": "string", "description": "提交任务时返回的 prompt_id"},
            },
            "required": ["prompt_id"],
        },
    },
    "comfyui_list_images": {
        "name": "comfyui_list_images",
        "description": "列出 ComfyUI 输出目录中的所有图片和视频文件。",
        "fn": comfyui_list_images,
        "params_schema": {"type": "object", "properties": {}},
    },
    "comfyui_queue_status": {
        "name": "comfyui_queue_status",
        "description": "查询 ComfyUI 队列状态：是否忙碌、正在运行的任务数、排队等待的任务数。",
        "fn": comfyui_queue_status,
        "params_schema": {"type": "object", "properties": {}},
    },
}
