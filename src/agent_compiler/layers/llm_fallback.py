"""LLM Provider — primary reasoning engine with ReAct loop.

Supports:
  - mock:           zero-dependency demo mode (simulated ReAct)
  - claude:         Anthropic Claude API
  - openai:         OpenAI API (GPT-4o, etc.)
  - openai_compat:  any OpenAI-compatible API (OpenRouter, Ollama, vLLM,
                    DeepSeek, local models, etc.)

Configuration priority: constructor args > env vars > defaults
"""

from __future__ import annotations

import json
import os
import re
import time

from agent_compiler.core.types import AgentResult


# ── System prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Agent Compiler, a helpful AI assistant with access to system tools.

## Your capabilities
You can use the available tools to:
- Check system status (CPU, memory, services)
- Check disk usage
- Search and analyze logs
- Generate reports
- List directory contents
- Find large files
- Get current time

## Rules
- If the user greets you or chats casually, reply naturally in the user's language.
- If the user asks who you are or what you can do, introduce yourself.
- If the user asks you to do something, use the appropriate tool(s) to help them.
- After receiving tool results, report them clearly in natural language.
- Never make up tool results — only report what you actually received.
- Be conversational and helpful. Reply in the same language the user uses."""


# ── Mock responses for demo ─────────────────────────────────────────

_MOCK_RESPONSES = [
    {
        "intent": "search logs and generate report",
        "steps": [
            {"tool_name": "search_logs", "params": {"pattern": "ERROR|CRITICAL", "days": 1, "level": "ERROR"},
             "description": "Search recent error logs"},
            {"tool_name": "generate_report", "params": {"format": "markdown", "title": "Error Log Report", "include_timeline": True},
             "description": "Generate summary report"},
        ],
    },
    {
        "intent": "analyze disk space",
        "steps": [
            {"tool_name": "get_disk_usage", "params": {},
             "description": "Get disk space overview"},
            {"tool_name": "find_large_files", "params": {"top_n": 10, "path": "/var/log"},
             "description": "Find largest files"},
            {"tool_name": "generate_report", "params": {"format": "text", "title": "Disk Analysis"},
             "description": "Generate disk analysis report"},
        ],
    },
    {
        "intent": "check system status",
        "steps": [
            {"tool_name": "get_system_status", "params": {"format": "detailed"},
             "description": "Get detailed system status"},
        ],
    },
    {
        "intent": "search logs by keyword",
        "steps": [
            {"tool_name": "search_logs", "params": {"pattern": "${pattern}", "days": 7, "level": "INFO"},
             "description": "Search logs for keyword"},
            {"tool_name": "generate_report", "params": {"format": "markdown", "title": "Log Search Results"},
             "description": "Generate search results report"},
        ],
    },
]


class LLMProvider:
    """Primary reasoning engine with ReAct loop.

    Usage:
        # Mock mode (no API needed)
        llm = LLMProvider()

        # Claude API
        llm = LLMProvider(provider="claude", api_key="sk-ant-...")

        # Any OpenAI-compatible API
        llm = LLMProvider(
            provider="openai_compat",
            api_key="your-key",
            api_base="https://your-endpoint/v1",
            model="your-model-name",
        )
    """

    def __init__(self,
                 provider: str | None = None,
                 api_key: str | None = None,
                 api_base: str | None = None,
                 model: str | None = None,
                 max_turns: int = 10):
        self.provider = provider or os.environ.get("LLM_PROVIDER", "mock")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.api_base = api_base or os.environ.get("LLM_API_BASE", "")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.max_turns = max_turns

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
        masked = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
        return {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base or "(default)",
            "api_key": masked if self.api_key else "(not set)",
        }

    # ── ReAct loop ─────────────────────────────────────────────────

    def react(self, user_input: str, context: list[dict],
              tool_defs: list[dict]) -> dict:
        """Run the ReAct loop. Returns {text, tool_calls, intent}.

        context:  list of {"role": "...", "content": "..."} dicts
        tool_defs: list of {"name", "description", "params_schema"} dicts
        """
        t0 = time.perf_counter()
        messages = list(context) + [{"role": "user", "content": user_input}]
        all_tool_calls: list[dict] = []
        final_text = ""
        final_intent = user_input

        for turn in range(self.max_turns):
            if self.provider == "mock":
                response = self._mock_react_turn(user_input, messages, tool_defs, turn)
            elif self.provider == "claude":
                response = self._claude_turn(messages, tool_defs)
            elif self.provider in ("openai", "openai_compat"):
                response = self._openai_turn(messages, tool_defs)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # Check for tool calls
            tool_calls = response.get("tool_calls", [])

            if tool_calls:
                all_tool_calls.extend(tool_calls)
                final_intent = response.get("intent", final_intent)

                # Add assistant message with tool_calls (required by OpenAI API)
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.get("tool_call_id", ""),
                            "type": "function",
                            "function": {
                                "name": tc["tool_name"],
                                "arguments": json.dumps(tc.get("params", {}), ensure_ascii=False),
                            }
                        }
                        for tc in tool_calls
                    ],
                })

                # Execute tools and append results
                for tc in tool_calls:
                    from agent_compiler.tools.registry import ToolRegistry
                    from agent_compiler.core.types import ActionStep
                    step = ActionStep(
                        tool_name=tc["tool_name"],
                        params=tc.get("params", {}),
                        description=tc.get("description", ""),
                    )
                    result = ToolRegistry.execute(step)
                    messages.append({
                        "role": "tool_result",
                        "content": json.dumps(result, ensure_ascii=False),
                        "tool_call_id": tc.get("tool_call_id", ""),
                    })
                continue  # next ReAct turn

            # No tool calls — this is the final text reply
            final_text = response.get("reply", response.get("text", ""))
            final_intent = response.get("intent", final_intent)
            break

        # Max turns reached
        if not final_text:
            final_text = "I have completed the requested actions."
            if all_tool_calls:
                tools_used = [tc["tool_name"] for tc in all_tool_calls]
                final_text = f"Done. Executed: {', '.join(tools_used)}."

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "text": final_text,
            "tool_calls": all_tool_calls,
            "intent": final_intent,
            "latency_ms": latency_ms,
        }

    # ── Backward compat: reason() ────────────────────────────────────

    def reason(self, user_input: str) -> AgentResult:
        """Legacy single-turn API. Returns AgentResult directly."""
        t0 = time.perf_counter()
        try:
            intent, steps_data = self._call_llm_legacy(user_input)
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

    def _call_llm_legacy(self, user_input: str) -> tuple[str, list[dict]]:
        """Legacy: single-turn JSON workflow compilation."""
        if self.provider == "mock":
            return self._mock_call(user_input)
        elif self.provider == "claude":
            return self._claude_call_legacy(user_input)
        elif self.provider in ("openai", "openai_compat"):
            return self._openai_compat_call_legacy(user_input)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ── Mock ────────────────────────────────────────────────────────

    def _mock_react_turn(self, user_input: str, messages: list[dict],
                         tool_defs: list[dict], turn: int) -> dict:
        """Mock ReAct: first turn returns tools + text, or just text."""
        # Check if we already got tool results
        has_tool_results = any(m["role"] == "tool_result" for m in messages)

        if has_tool_results:
            # Generate a text reply based on what happened
            results_text = []
            for m in messages:
                if m["role"] == "tool_result":
                    try:
                        data = json.loads(m["content"])
                        if isinstance(data, dict):
                            status = "OK" if data.get("success") else f"FAIL: {data.get('error', 'unknown')}"
                            results_text.append(f"  - {data.get('tool', '?')}: {status}")
                    except Exception:
                        pass
            reply = "Here is what I found:\n" + "\n".join(results_text) if results_text else "All done!"
            return {"intent": "results summary", "reply": reply, "tool_calls": []}

        # First turn: determine intent
        inp = user_input.lower()

        # ── Determine mock response ──────────────────────────────
        if any(w in inp for w in ("磁盘", "disk", "硬盘", "空间还剩", "空间够",
                                   "内存", "c盘", "d盘", "e盘", "盘空间", "盘内存")):
            resp = _MOCK_RESPONSES[1]  # disk analysis
        elif any(w in inp for w in ("服务器状态", "系统状态", "运行状态", "server status")):
            resp = _MOCK_RESPONSES[2]  # system status
        elif any(w in inp for w in ("日志", "log", "错误", "error", "报错", "搜索",
                                     "search", "查找", "timeout", "connection")):
            resp = _MOCK_RESPONSES[3]  # log search
        elif any(w in inp for w in ("几点", "时间", "日期", "今天几号")):
            resp = _MOCK_RESPONSES[2]  # system status (covers time)
        elif any(w in inp for w in ("文件列表", "列出文件", "目录", "大文件")):
            resp = _MOCK_RESPONSES[1]  # disk + files
        else:
            # Conversational — no tools
            return {
                "intent": "chat reply",
                "reply": self._chat_reply(user_input),
                "tool_calls": [],
            }

        return {
            "intent": resp["intent"],
            "tool_calls": resp["steps"],
            "reply": "",
        }

    def _mock_call(self, user_input: str) -> tuple[str, list[dict]]:
        """Legacy mock: returns intent + steps for backward compat."""
        result = self._mock_react_turn(user_input, [], [], 0)
        return result["intent"], result["tool_calls"]

    @staticmethod
    def _chat_reply(user_input: str) -> str:
        """Generate a natural chat reply for non-task inputs."""
        inp = user_input.lower()

        def has(*words: str) -> bool:
            return any(w in inp for w in words)

        if has("你好", "hi", "hello", "嗨", "hey", "在吗"):
            return ("你好！我是 Agent Compiler，一个智能任务执行引擎。\n\n"
                    "我可以帮你：\n"
                    "- 查看服务器状态、磁盘空间\n"
                    "- 搜索和分析日志\n"
                    "- 生成报告、查找文件\n"
                    "- 查看当前时间\n\n"
                    "直接告诉我你想做什么就行！")

        if has("你是谁", "你是什么", "介绍", "功能", "做什么", "干嘛",
               "会什么", "能做什么", "who are you", "what can you",
               "你能帮我", "帮什么", "哪些", "有什么能力"):
            return ("我是 **Agent Compiler**，一个智能任务执行引擎。\n\n"
                    "**工作原理：**\n"
                    "当你告诉我一个任务时，我会通过 LLM 分析意图，\n"
                    "然后执行相应的工具来完成它。\n\n"
                    "执行过的任务会被缓存，下次相似的问题就直接执行，\n"
                    "不再调用 LLM，速度更快，还省 Token。\n\n"
                    "试试这些：\n"
                    "- 查看服务器状态\n"
                    "- 帮我查错误日志\n"
                    "- 磁盘空间还剩多少\n"
                    "- 现在几点")

        if has("谢谢", "感谢", "thanks", "thank", "thx"):
            return "不客气！还有什么需要做的吗？"

        if has("再见", "bye", "拜拜"):
            return "再见！有问题随时找我。"

        if has("回答", "问题", "能不能", "可以回答", "能否", "能回答", "可以问"):
            return ("当然可以！你可以问我：\n\n"
                    "- 查看服务器状态 — 获取系统运行信息\n"
                    "- 帮我查错误日志 — 搜索最近的错误\n"
                    "- 磁盘空间还剩多少 — 查看磁盘使用\n"
                    "- 现在几点 — 获取当前时间\n"
                    "- 列出文件 — 查看目录内容\n\n"
                    "这些我都会直接执行对应的工具并返回结果。试试看？")

        if has("怎么用", "怎么操作", "怎么说话", "使用方法", "how to use"):
            return ("很简单，直接告诉我你想做什么就行。比如：\n\n"
                    "  - 查看服务器状态 — 读取 CPU、内存、运行时间\n"
                    "  - 帮我查错误日志 — 搜索并汇总错误\n"
                    "  - 磁盘空间还剩多少 — 查看磁盘使用\n\n"
                    "不需要记命令，正常说话就行。")

        # Dynamic fallback — acknowledge the query instead of being dismissive
        if len(user_input) > 30:
            return (f"好的，我理解你想做的是：\"{user_input[:50]}{'...' if len(user_input) > 50 else ''}\"\n\n"
                    f"目前 mock 模式下我内置了以下工具：\n"
                    f"- 查看服务器状态\n"
                    f"- 搜索和分析日志\n"
                    f"- 查看磁盘空间\n"
                    f"- 查找大文件\n"
                    f"- 列出目录文件\n"
                    f"- 查看当前时间\n\n"
                    f"切换到真实 LLM 模式可以获得更智能的回复：\n"
                    f"  agent-compiler   (不带 --provider mock)")
        return ("收到！我目前运行在 mock 演示模式，内置了以下能力：\n\n"
                "- 查看服务器状态\n"
                "- 帮我查错误日志\n"
                "- 磁盘空间还剩多少\n"
                "- 现在几点\n"
                "- 列出文件、查找大文件\n\n"
                "试试这些，或者直接运行 `agent-compiler` 用真实 AI 模式！")

    # ── Claude API ──────────────────────────────────────────────────

    def _claude_turn(self, messages: list[dict], tool_defs: list[dict]) -> dict:
        """Single turn of ReAct via Claude Messages API."""
        import anthropic

        # Build tool schemas in Anthropic format
        tools = []
        for td in tool_defs:
            tools.append({
                "name": td["name"],
                "description": td["description"],
                "input_schema": td["params_schema"],
            })

        kwargs = {"api_key": self.api_key}
        client = anthropic.Anthropic(**kwargs)

        # Convert tool_result roles for Anthropic format
        anthropic_messages = []
        for m in messages:
            role = m["role"]
            if role == "tool_result":
                role = "user"
                content = f"<tool_result>{m['content']}</tool_result>"
            else:
                content = m["content"]
            anthropic_messages.append({"role": role, "content": content})

        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=anthropic_messages,
            tools=tools,
        )

        # Parse response
        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "tool_name": block.name,
                    "params": dict(block.input),
                    "description": f"Call {block.name}",
                    "tool_call_id": block.id,
                })

        if tool_calls:
            return {"tool_calls": tool_calls, "intent": "task"}

        return {"reply": "\n".join(text_parts), "intent": "reply"}

    def _claude_call_legacy(self, user_input: str) -> tuple[str, list[dict]]:
        """Legacy: single-turn JSON via Claude."""
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

    # ── OpenAI / OpenAI-compatible ──────────────────────────────────

    def _openai_turn(self, messages: list[dict], tool_defs: list[dict]) -> dict:
        """Single turn of ReAct via OpenAI-compatible API."""
        from openai import OpenAI

        # Build tool schemas in OpenAI format
        tools = []
        for td in tool_defs:
            tools.append({
                "type": "function",
                "function": {
                    "name": td["name"],
                    "description": td["description"],
                    "parameters": td["params_schema"],
                },
            })

        kwargs = {"api_key": self.api_key}
        if self.api_base:
            kwargs["base_url"] = self.api_base
        client = OpenAI(**kwargs)

        openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in messages:
            role = m["role"]
            if role == "tool_result":
                openai_messages.append({
                    "role": "tool",
                    "content": m["content"],
                    "tool_call_id": m.get("tool_call_id", ""),
                })
            elif role == "assistant" and m.get("tool_calls"):
                openai_messages.append({
                    "role": "assistant",
                    "content": m.get("content"),
                    "tool_calls": m["tool_calls"],
                })
            else:
                openai_messages.append({"role": role, "content": m["content"]})

        resp = client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=tools if tools else None,
        )

        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    params = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    params = {}
                tool_calls.append({
                    "tool_name": tc.function.name,
                    "params": params,
                    "description": f"Call {tc.function.name}",
                    "tool_call_id": tc.id,
                })

        text = msg.content or ""

        if tool_calls:
            return {"tool_calls": tool_calls, "intent": "task"}

        # Try to parse JSON from text (for providers without native tool support)
        parsed = self._try_parse_json(text)
        if parsed:
            intent = parsed.get("intent", "task")
            if parsed.get("tool_calls"):
                return {"tool_calls": parsed["tool_calls"], "intent": intent}
            if parsed.get("reply"):
                return {"reply": parsed["reply"], "intent": intent}

        return {"reply": text, "intent": "reply"}

    def _openai_compat_call_legacy(self, user_input: str) -> tuple[str, list[dict]]:
        """Legacy: single-turn JSON via OpenAI."""
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

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Try to extract and parse JSON from LLM text output."""
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            text = m.group(1)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None


# ── Backward compat alias ────────────────────────────────────────────

LLMFallback = LLMProvider


# ── Response parsing ────────────────────────────────────────────────

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
