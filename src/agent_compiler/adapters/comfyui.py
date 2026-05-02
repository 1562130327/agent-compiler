"""ComfyUI (Zealman) API adapter — wraps the cloud ComfyUI instance's API.

Provides:
  - list_workflows / get_workflow_config
  - generate (submit a workflow with overridden params)
  - queue status / history / image retrieval
  - polling-based wait_for_result (Zealman has no WebSocket)

All methods are async so they can be called directly from FastAPI routes.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError
from urllib.parse import quote


@dataclass
class ComfyUIConfig:
    """Connection settings for a Zealman-wrapped ComfyUI instance."""
    base_url: str = "https://uu244618-7777e2782b02.bjb1.seetacloud.com:8443"
    timeout: int = 120          # seconds for HTTP requests
    poll_interval: float = 2.0  # seconds between status polls


class ComfyUIAdapter:
    """Adapter for the Zealman ComfyUI Control Panel API.

    Usage:
        adapter = ComfyUIAdapter(ComfyUIConfig(base_url="https://..."))
        workflows = await adapter.list_workflows()
        result = await adapter.generate("C07-...", {"93:text": "a cat", "99:seed": 42})
        outputs = await adapter.wait_for_result(result["prompt_id"])
    """

    def __init__(self, config: ComfyUIConfig | None = None):
        self.config = config or ComfyUIConfig()
        self._prompt_cache: dict[str, dict] = {}

    # ── low-level HTTP ──────────────────────────────────────────────

    def _build_url(self, path: str) -> str:
        """Build a full URL, properly encoding non-ASCII path segments."""
        # quote each path segment individually, preserving /
        parts = path.split("/")
        encoded = "/".join(quote(p, safe="") for p in parts)
        return self.config.base_url.rstrip("/") + "/" + encoded.lstrip("/")

    async def _get(self, path: str) -> dict | list:
        """GET JSON from the Zealman API."""
        url = self._build_url(path)
        return await asyncio.to_thread(_sync_get_json, url, self.config.timeout)

    async def _post(self, path: str, body: dict) -> dict:
        """POST JSON to the Zealman API."""
        url = self._build_url(path)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        return await asyncio.to_thread(_sync_post_json, url, data, self.config.timeout)

    # ── workflow operations ─────────────────────────────────────────

    async def list_workflows(self) -> list[dict[str, Any]]:
        """List all saved workflow templates.

        Returns list of {id, name, mtime} dicts.
        """
        resp = await self._get("/api/workflow/list")
        if isinstance(resp, dict) and resp.get("success"):
            return resp["workflows"]
        return []

    async def get_workflow_config(self, workflow_id: str) -> dict:
        """Get a workflow's full JSON template including _api_config.

        The _api_config block defines which parameters are exposed via API
        (enabledParams) and their current/default values (formValues).
        """
        return await self._get(f"/api/workflow/config/{workflow_id}")

    # ── generation ──────────────────────────────────────────────────

    async def generate(self, workflow_id: str, params: dict[str, Any] | None = None) -> dict:
        """Submit a workflow for generation with optional parameter overrides.

        Args:
            workflow_id: e.g. "C07-文生图-Zimage-Nunchaku加速"
            params: dict of {nodeId:param: value} overrides, e.g.
                    {"93:text": "a cat in rain", "99:seed": 12345}

        Returns:
            {"success": true, "id": "workflow-xxx", "prompt_id": "xxx", "prompt": {...}}
        """
        config = await self.get_workflow_config(workflow_id)
        if not config.get("success", True):
            return config

        template = config.get("workflow_template", config)

        # apply parameter overrides into formValues
        if params:
            api_cfg = template.get("_api_config", {})
            form_vals = api_cfg.get("formValues", {})
            for key, value in params.items():
                form_vals[key] = value
            api_cfg["formValues"] = form_vals
            template["_api_config"] = api_cfg

        result = await self._post("/api/workflow/generate", {"workflow_template": template})
        if result.get("success") and result.get("prompt_id"):
            self._prompt_cache[result["prompt_id"]] = result
        return result

    async def generate_from_template(self, template: dict, params: dict[str, Any] | None = None) -> dict:
        """Submit an arbitrary workflow template (without fetching by ID)."""
        if params:
            api_cfg = template.get("_api_config", {})
            form_vals = api_cfg.get("formValues", {})
            for key, value in params.items():
                form_vals[key] = value
            api_cfg["formValues"] = form_vals
            template["_api_config"] = api_cfg

        result = await self._post("/api/workflow/generate", {"workflow_template": template})
        if result.get("success") and result.get("prompt_id"):
            self._prompt_cache[result["prompt_id"]] = result
        return result

    # ── status & results ────────────────────────────────────────────

    async def get_queue_status(self) -> dict:
        """Get current ComfyUI queue status.

        Returns {"busy": bool, "running_count": int, "pending_count": int}
        """
        return await self._get("/api/comfy/queue-status")

    async def get_history(self) -> dict:
        """Get ComfyUI prompt history (maps prompt_id -> execution info)."""
        return await self._get("/api/comfy/proxy/history")

    async def get_prompt_result(self, prompt_id: str) -> dict | None:
        """Get execution result for a specific prompt_id from history."""
        history = await self.get_history()
        return history.get(prompt_id)

    def get_image_url(self, filename: str) -> str:
        """Build a view URL for an output image."""
        return f"{self.config.base_url.rstrip('/')}/api/comfy/view?filename={quote(filename)}"

    def get_image_download_url(self, filename: str) -> str:
        """Build a download URL (same as view for this proxy)."""
        return self.get_image_url(filename)

    # ── polling helper ──────────────────────────────────────────────

    async def wait_for_result(self, prompt_id: str, timeout: int = 300) -> dict:
        """Poll queue + history until the prompt completes or times out.

        Args:
            prompt_id: the ComfyUI prompt_id from generate()
            timeout: max seconds to wait (default 5 min)

        Returns:
            {"status": "success"|"error"|"timeout", "images": [...], "raw": {...}}
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            # first check if queue is done
            queue = await self.get_queue_status()
            if not queue.get("busy") and queue.get("pending_count", 0) == 0:
                # queue idle — check history for our prompt
                result = await self.get_prompt_result(prompt_id)
                if result is not None:
                    return self._parse_prompt_result(result)

            await asyncio.sleep(self.config.poll_interval)

        # timeout — check one last time
        result = await self.get_prompt_result(prompt_id)
        if result is not None:
            return self._parse_prompt_result(result)

        return {"status": "timeout", "images": [], "raw": {"prompt_id": prompt_id}}

    def _parse_prompt_result(self, raw: dict) -> dict:
        """Extract output images from a ComfyUI history entry."""
        status = raw.get("status", {})
        status_str = status.get("status_str", "unknown")
        completed = status.get("completed", False)
        messages = status.get("messages", [])

        error_msg = None
        if status_str == "error":
            for msg in messages:
                if msg[0] == "execution_error":
                    error_msg = msg[1].get("exception_message", "Unknown error")
                    break

        images: list[str] = []
        outputs = raw.get("outputs", {})
        for node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                images.append(img["filename"])

        # resolve full image URLs
        image_urls = [self.get_image_url(f) for f in images]

        return {
            "status": status_str,
            "completed": completed,
            "error": error_msg,
            "images": images,
            "image_urls": image_urls,
            "raw": raw,
        }

    # ── model info ──────────────────────────────────────────────────

    async def list_models(self) -> list[dict]:
        """List available AI models."""
        return await self._get("/api/models/list")

    async def list_images(self) -> list[dict]:
        """List output images."""
        return await self._get("/api/images")


# ── synchronous HTTP helpers ────────────────────────────────────────

def _sync_get_json(url: str, timeout: int) -> dict | list:
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except URLError as e:
        return {"success": False, "error": str(e)}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}"}


def _sync_post_json(url: str, data: bytes, timeout: int) -> dict:
    req = Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    }, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except URLError as e:
        return {"success": False, "error": str(e)}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON parse error: {e}"}
