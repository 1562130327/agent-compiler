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


# ── System prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Agent Compiler, a helpful AI assistant. Reply in the user's language. Be concise.

When using tools: read files before editing, verify results, report tool output accurately.
For coding: plan first, edit existing files rather than create new ones, run tests after changes."""






class LLMProvider:
    """Primary reasoning engine with ReAct loop.

    Usage:
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

    @staticmethod
    def _is_chat_query(user_input: str) -> bool:
        """Check if this is a simple conversation, not a tool-using task."""
        task_patterns = [
            "查看", "搜索", "查找", "列出", "读取", "读取文件",
            "写", "写入", "创建文件", "编辑", "修改",
            "执行", "运行", "安装", "测试", "部署",
            "git", "commit", "push", "pull", "diff", "log",
            "磁盘", "内存", "进程", "日志", "log",
            "read", "write", "edit", "run", "execute", "install",
            "search", "find", "list", "glob", "grep",
            "状态", "status", "系统", "报告",
            "http", "curl", "fetch", "下载",
        ]
        inp = user_input.lower()
        return not any(p in inp for p in task_patterns)

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
        # For simple chat questions, skip tools entirely — saves 5000+ tokens
        if self._is_chat_query(user_input):
            tool_defs = []

        t0 = time.perf_counter()
        messages = list(context) + [{"role": "user", "content": user_input}]
        msg_chars = sum(len(str(m.get("content", ""))) for m in messages)
        print(f"[LLM] react: {len(messages)} 消息, {msg_chars} 字符, tools={len(tool_defs)}, is_chat={self._is_chat_query(user_input)}")
        all_tool_calls: list[dict] = []
        final_text = ""
        final_intent = user_input
        total_tokens = {"prompt": 0, "completion": 0, "total": 0}

        for turn in range(self.max_turns):
            if self.provider == "mock":
                raise RuntimeError("Mock 模式已禁用。请配置真实的 LLM API（参见 config.yaml）")
            elif self.provider == "claude":
                response = self._claude_turn(messages, tool_defs)
            elif self.provider in ("openai", "openai_compat"):
                response = self._openai_turn(messages, tool_defs)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # Accumulate token usage from each turn
            tokens = response.get("tokens", {})
            for k in ("prompt", "completion", "total"):
                total_tokens[k] += tokens.get(k, 0)

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

        # ── Reflexion: self-critique and revise (non-mock only) ──
        reflexion_rounds = 0
        reflexion_log: list[dict] = []
        if not self.is_mock and final_text:
            for rev_round in range(2):  # max 2 revision rounds
                score = self._evaluate_output(user_input, final_text, all_tool_calls)
                reflexion_log.append({
                    "round": rev_round + 1,
                    "score": score.get("score", 0),
                    "feedback": score.get("feedback", ""),
                })
                if score.get("score", 0) >= 4:
                    break
                # Revise based on feedback
                revised = self._revise_output(
                    user_input, final_text,
                    score.get("feedback", "改进输出质量"),
                    messages,
                )
                if revised and revised != final_text:
                    final_text = revised
                    reflexion_rounds += 1
                else:
                    break

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "text": final_text,
            "tool_calls": all_tool_calls,
            "intent": final_intent,
            "latency_ms": latency_ms,
            "tokens": total_tokens,
            "reflexion_rounds": reflexion_rounds,
            "reflexion_log": reflexion_log,
        }

    # ── Memory extraction ───────────────────────────────────────────

    def extract_memories(self, user_input: str, assistant_reply: str) -> list[dict]:
        """Extract key facts from a conversation turn for the memory system.

        Returns a list of fact dicts with keys:
          - category: user_profile | project | pattern | feedback | knowledge
          - title: short summary
          - content: the fact
          - keywords: list of search keywords
          - confidence: 0.0-1.0
        """
        if self.is_mock:
            return self._mock_extract_memories(user_input, assistant_reply)

        prompt = f"""Analyze this conversation turn and extract any key facts worth remembering about the user or project.

User message: {user_input[:600]}
Assistant reply: {assistant_reply[:600]}

Output ONLY a JSON array of facts. Each fact must have:
- "category": one of "user_profile", "project", "pattern", "feedback", "knowledge"
- "title": short label in CHINESE (<60 chars)
- "content": the fact itself in CHINESE (concise, <300 chars)
- "keywords": array of 3-6 search keywords (MUST include both Chinese and English keywords)
- "confidence": number 0.0-1.0 (how certain you are this is worth keeping)

Categories:
- user_profile: user's name, role, preferences, skills, habits
- project: project names, repos, branches, tech stack, architecture
- pattern: task patterns the user repeats
- feedback: user corrections or preferences about how you should work
- knowledge: other useful facts the user wants you to remember

IMPORTANT: Write ALL text fields (title, content, keywords) in CHINESE so they can be searched by Chinese queries. Include English translations in keywords only.

If nothing is worth remembering, output an empty array [].
Do NOT wrap in markdown. Output ONLY valid JSON."""

        try:
            result = self._llm_json_call(prompt)
            if isinstance(result, list):
                return [f for f in result if isinstance(f, dict) and f.get("content")]
            return []
        except Exception:
            return []

    def _llm_json_call(self, prompt: str) -> list | dict:
        """Make a lightweight LLM call expecting JSON output."""
        if self.provider in ("openai", "openai_compat"):
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.api_base:
                kwargs["base_url"] = self.api_base
            client = OpenAI(**kwargs)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a fact extraction system. Output ONLY valid JSON, no markdown, no explanation."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            text = resp.choices[0].message.content or ""
            return self._try_parse_json(text) or []
        elif self.provider == "claude":
            import anthropic
            kwargs = {"api_key": self.api_key}
            client = anthropic.Anthropic(**kwargs)
            msg = client.messages.create(
                model=self.model,
                max_tokens=500,
                system="You are a fact extraction system. Output ONLY valid JSON, no markdown, no explanation.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text if msg.content else ""
            return self._try_parse_json(text) or []
        return []

    # ── Reflexion: self-evaluation and revision ────────────────────────

    def _evaluate_output(self, user_input: str, output: str,
                         tool_calls: list[dict]) -> dict:
        """LLM self-evaluates its output on a 1-5 scale.

        Returns dict with 'score' (int) and 'feedback' (str in Chinese).
        """
        tools_summary = ""
        if tool_calls:
            tools_used = [tc["tool_name"] for tc in tool_calls]
            tools_summary = f"\n工具执行: {', '.join(tools_used)}"

        prompt = f"""评估以下 AI 助手回复的质量。

用户请求: {user_input[:500]}

助手回复:
{output[:1000]}{tools_summary}

请按 1-5 分打分:
- 1: 完全错误或无关
- 2: 有重大问题 — 遗漏关键信息、令人困惑、部分错误
- 3: 基本可接受但可大幅改进 — 模糊、不完整、略有偏差
- 4: 良好 — 较好地回应了请求，只有小改进空间
- 5: 优秀 — 全面、准确、结构清晰，直接回应请求

输出 ONLY JSON:
{{"score": <整数1-5>, "feedback": "<1-2句中评语，说明哪里需要改进或哪里做得好>"}}

4-5分时 feedback 简要说明哪里做得好。1-3分时 feedback 简明指出问题及改进方向。"""

        try:
            result = self._llm_json_call(prompt)
            if isinstance(result, dict) and "score" in result:
                return {
                    "score": int(result.get("score", 3)),
                    "feedback": str(result.get("feedback", "")),
                }
        except Exception:
            pass
        return {"score": 4, "feedback": "无法评估，假定质量合格。"}

    def _revise_output(self, user_input: str, original: str,
                       feedback: str, messages: list[dict]) -> str | None:
        """Ask the LLM to revise its output based on evaluation feedback.

        Returns revised text, or None if revision fails.
        """
        recent_context = []
        for m in messages[-6:]:
            role = m.get("role", "")
            content = m.get("content", "")
            if content and role in ("user", "assistant", "tool_result"):
                recent_context.append(f"[{role}] {str(content)[:300]}")
        context_str = "\n".join(recent_context) if recent_context else "(无上下文)"

        prompt = f"""根据质量反馈修订以下 AI 回复。

用户请求: {user_input[:500]}

近期上下文:
{context_str[:800]}

原始回复:
{original[:1000]}

改进反馈:
{feedback}

请生成修订后的回复，解决反馈中提到的问题。
- 如果遗漏了信息，请补充
- 如果有错误，请更正
- 如果过于模糊，请添加具体内容
- 保持与原始回复相同的语言和语气
- 仅输出修订后的回复文本，不要 JSON，不要 markdown 包裹"""

        try:
            if self.provider in ("openai", "openai_compat"):
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.api_base:
                    kwargs["base_url"] = self.api_base
                client = OpenAI(**kwargs)
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个根据反馈修订回复的助手。只输出修订后的文本，无需解释。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                text = resp.choices[0].message.content or ""
                return text.strip() if text.strip() else None

            elif self.provider == "claude":
                import anthropic
                kwargs = {"api_key": self.api_key}
                client = anthropic.Anthropic(**kwargs)
                msg = client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system="你是一个根据反馈修订回复的助手。只输出修订后的文本，无需解释。",
                    messages=[{"role": "user", "content": prompt}],
                )
                text = msg.content[0].text if msg.content else ""
                return text.strip() if text.strip() else None

        except Exception:
            pass

        return None

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

        # Extract token usage
        tokens = {}
        if resp.usage:
            tokens = {
                "prompt": resp.usage.prompt_tokens or 0,
                "completion": resp.usage.completion_tokens or 0,
                "total": resp.usage.total_tokens or 0,
            }

        text = msg.content or ""

        result: dict = {"intent": "reply", "tokens": tokens}

        if tool_calls:
            result["tool_calls"] = tool_calls
            result["intent"] = "task"
            return result

        # Try to parse JSON from text (for providers without native tool support)
        parsed = self._try_parse_json(text)
        if parsed:
            intent = parsed.get("intent", "task")
            if parsed.get("tool_calls"):
                result["tool_calls"] = parsed["tool_calls"]
                result["intent"] = intent
                return result
            if parsed.get("reply"):
                result["reply"] = parsed["reply"]
                result["intent"] = intent
                return result

        result["reply"] = text
        return result

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
