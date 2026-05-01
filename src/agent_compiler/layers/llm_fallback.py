"""Layer 3: LLM fallback — invoked only when L1 (rules) and L2 (cache) both miss.

Supports:
  - mock:           zero-dependency demo mode (default)
  - claude:         Anthropic Claude API
  - openai:         OpenAI API (GPT-4o, etc.)
  - openai_compat:  any OpenAI-compatible API (OpenRouter, Ollama, vLLM,
                    DeepSeek, local models, etc.)

Configuration priority: constructor args > env vars > defaults

Env vars:
  LLM_PROVIDER   = mock | claude | openai | openai_compat
  LLM_API_KEY    = your api key
  LLM_API_BASE   = custom endpoint URL (for openai_compat)
  LLM_MODEL      = model name override
"""

from __future__ import annotations

import json
import os
import re
import time

from agent_compiler.core.types import AgentResult

# ── Mock responses for demo ─────────────────────────────────────────

_MOCK_RESPONSES = [
    {
        "intent": "查询错误日志并生成报告",
        "steps": [
            {"tool_name": "search_logs", "params": {"pattern": "ERROR|CRITICAL", "days": 1, "level": "ERROR"},
             "description": "搜索最近1天的错误日志"},
            {"tool_name": "generate_report", "params": {"format": "markdown", "title": "错误日志报告", "include_timeline": True},
             "description": "生成汇总报告"},
        ],
    },
    {
        "intent": "分析磁盘空间使用",
        "steps": [
            {"tool_name": "get_disk_usage", "params": {},
             "description": "获取磁盘空间概况"},
            {"tool_name": "find_large_files", "params": {"top_n": 10, "path": "/var/log"},
             "description": "查找最大的文件"},
            {"tool_name": "generate_report", "params": {"format": "text", "title": "磁盘空间分析"},
             "description": "生成分析报告"},
        ],
    },
    {
        "intent": "查看系统状态",
        "steps": [
            {"tool_name": "get_system_status", "params": {"format": "detailed"},
             "description": "获取系统状态详情"},
        ],
    },
    {
        "intent": "搜索指定内容的日志",
        "steps": [
            {"tool_name": "search_logs", "params": {"pattern": "${pattern}", "days": 7, "level": "INFO"},
             "description": "搜索日志中的关键词"},
            {"tool_name": "generate_report", "params": {"format": "markdown", "title": "日志搜索结果"},
             "description": "生成搜索结果报告"},
        ],
    },
]

SYSTEM_PROMPT = """You are a workflow compiler. Given a user's task, output a JSON array of action steps.

Each step must have:
  - "tool_name": one of [search_logs, generate_report, get_disk_usage, find_large_files, get_system_status, list_directory, get_current_time]
  - "params": object with tool-specific parameters
  - "description": short description of what this step does

Output ONLY valid JSON inside ```json ... ``` code block. No other text."""


class LLMFallback:
    """Layer 3: LLM reasoning for novel tasks.

    Usage:
        # Mock mode (no API needed)
        llm = LLMFallback()

        # Claude API
        llm = LLMFallback(provider="claude", api_key="sk-ant-...")

        # OpenAI API
        llm = LLMFallback(provider="openai", api_key="sk-...")

        # Any OpenAI-compatible API (OpenRouter, Ollama, DeepSeek, etc.)
        llm = LLMFallback(
            provider="openai_compat",
            api_key="your-key",
            api_base="https://your-endpoint/v1",
            model="your-model-name",
        )

        # From env vars
        # export LLM_PROVIDER=openai_compat
        # export LLM_API_KEY=sk-xxx
        # export LLM_API_BASE=https://api.openrouter.ai/v1
        # export LLM_MODEL=anthropic/claude-sonnet-4
        llm = LLMFallback()  # reads from env
    """

    def __init__(self,
                 provider: str | None = None,
                 api_key: str | None = None,
                 api_base: str | None = None,
                 model: str | None = None):
        # Resolve config: args > env > defaults
        self.provider = provider or os.environ.get("LLM_PROVIDER", "mock")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.api_base = api_base or os.environ.get("LLM_API_BASE", "")
        self.model = model or os.environ.get("LLM_MODEL", "")

        # Set default models per provider if not specified
        if not self.model:
            self.model = {
                "claude": "claude-sonnet-4-6",
                "openai": "gpt-4o-mini",
                "openai_compat": "gpt-4o-mini",
            }.get(self.provider, "")

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    @property
    def config_summary(self) -> dict:
        """Return current config (masking API key)."""
        masked = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
        return {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base or "(default)",
            "api_key": masked if self.api_key else "(not set)",
        }

    def reason(self, user_input: str) -> AgentResult:
        """Run LLM reasoning and return structured result."""
        t0 = time.perf_counter()

        try:
            intent, steps_data = self._call_llm(user_input)
            latency = (time.perf_counter() - t0) * 1000
            return AgentResult(
                success=True,
                data={"intent": intent, "steps_data": steps_data, "raw_input": user_input},
                source="llm",
                confidence=0.85,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return AgentResult(
                success=False, data=None, source="llm",
                confidence=0.0, latency_ms=latency, error=str(e),
            )

    def _call_llm(self, user_input: str) -> tuple[str, list[dict]]:
        if self.provider == "mock":
            return self._mock_call(user_input)
        elif self.provider == "claude":
            return self._claude_call(user_input)
        elif self.provider in ("openai", "openai_compat"):
            return self._openai_compat_call(user_input)
        else:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                f"Valid options: mock, claude, openai, openai_compat"
            )

    # ── Mock (zero-dependency demo) ──────────────────────────────────

    def _mock_call(self, user_input: str) -> tuple[str, list[dict]]:
        inp = user_input.lower()
        if any(w in inp for w in ("磁盘", "disk", "空间", "硬盘")):
            resp = _MOCK_RESPONSES[1]
        elif any(w in inp for w in ("状态", "status")):
            resp = _MOCK_RESPONSES[2]
        elif any(w in inp for w in ("搜索", "search", "查找")):
            resp = _MOCK_RESPONSES[3]
        else:
            resp = _MOCK_RESPONSES[0]
        return resp["intent"], resp["steps"]

    # ── Claude API ───────────────────────────────────────────────────

    def _claude_call(self, user_input: str) -> tuple[str, list[dict]]:
        import anthropic
        kwargs = {"api_key": self.api_key}
        client = anthropic.Anthropic(**kwargs)
        msg = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_input}],
        )
        return _parse_response(msg.content[0].text)

    # ── OpenAI / OpenAI-compatible ────────────────────────────────────
    #   Covers: OpenAI, OpenRouter, Ollama, vLLM, DeepSeek, Groq,
    #           local LLaMA.cpp server, etc.

    def _openai_compat_call(self, user_input: str) -> tuple[str, list[dict]]:
        from openai import OpenAI
        kwargs = {"api_key": self.api_key}
        if self.api_base:
            kwargs["base_url"] = self.api_base
        client = OpenAI(**kwargs)

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        return _parse_response(resp.choices[0].message.content)


def _parse_response(text: str) -> tuple[str, list[dict]]:
    """Parse LLM JSON response, extracting from code blocks if present."""
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        text = m.group(1)
    steps_data = json.loads(text)
    if isinstance(steps_data, dict):
        intent = steps_data.get("intent", "")
        steps = steps_data.get("steps", [])
    elif isinstance(steps_data, list):
        intent = "multi-step task"
        steps = steps_data
    else:
        intent = ""
        steps = []
    return intent, steps
