"""Main agent: if-else dispatch (Cache -> LLM ReAct).

Architecture:
    Cache check → hit:  execute cached workflow, pure CPU, milliseconds
                 → miss: LLM ReAct loop → compile → cache → execute

The system gets faster over time as the cache grows naturally
from LLM interactions. No hand-written rules needed.
"""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from agent_compiler.compiler.compiler import Compiler
from agent_compiler.core.config import AgentConfig, LLMConfig
from agent_compiler.core.types import (
    ActionStep, AgentResult, Session, WorkflowTemplate,
)
from agent_compiler.embeddings.lightweight import LightweightEmbedding
from agent_compiler.layers.llm_fallback import LLMProvider
from agent_compiler.layers.pattern_cache import PatternCache
from agent_compiler.tools.registry import ToolRegistry


class SessionManager:
    """Tracks multi-turn conversation sessions in memory."""

    def __init__(self, ttl_minutes: int = 60, max_messages: int = 50):
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl_minutes
        self._max_msgs = max_messages

    def get_or_create(self, session_id: str | None = None) -> Session:
        self._cleanup()
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        new_id = session_id or f"sess_{uuid4().hex[:12]}"
        sess = Session(id=new_id)
        self._sessions[new_id] = sess
        return sess

    def _cleanup(self):
        now = time.time()
        cutoff = now - self._ttl * 60
        stale = [sid for sid, s in self._sessions.items()
                 if s.last_active_at < cutoff]
        for sid in stale:
            del self._sessions[sid]
        for s in self._sessions.values():
            if len(s.messages) > self._max_msgs:
                s.messages = s.messages[-self._max_msgs:]


class Agent:
    """LLM Agent with automatic workflow caching.

    Usage:
        # Simplest: mock mode, no API needed
        agent = Agent()

        # With config
        config = AgentConfig(
            llm=LLMConfig(provider="claude", api_key="sk-ant-..."),
            similarity_threshold=0.45,
        )
        agent = Agent(config)

        # Or just pass individual params
        agent = Agent(
            llm_provider="openai_compat",
            llm_api_key="sk-xxx",
            llm_api_base="https://api.deepseek.com",
            llm_model="deepseek-chat",
        )
    """

    @staticmethod
    def _find_upwards(filename: str) -> str | None:
        """Search CWD, parent dirs, and common locations for a file."""
        # 1. CWD and parent directories (up to 5 levels)
        d = Path.cwd().resolve()
        for _ in range(5):
            p = d / filename
            if p.exists():
                return str(p)
            if d.parent == d:
                break
            d = d.parent
        # 2. Package source tree (for dev installs)
        src = Path(__file__).resolve().parent.parent.parent.parent
        p = src / filename
        if p.exists():
            return str(p)
        # 3. User home
        home = Path.home()
        for sub in (".agent-compiler", ".config/agent-compiler", "agent-compiler"):
            p = home / sub / filename
            if p.exists():
                return str(p)
        return None

    def __init__(self, config: AgentConfig | None = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            found = self._find_upwards("config.yaml")
            if found:
                self.config = AgentConfig.from_yaml(found, **kwargs)
            else:
                self.config = AgentConfig.from_env(**kwargs)

        cfg = self.config

        # Embedding provider
        self.embeddings = LightweightEmbedding(
            similarity_threshold=cfg.similarity_threshold,
        )

        # Cache (the "if" in if-else)
        self.cache = PatternCache(self.embeddings, cfg.cache_dir)

        # LLM provider (the "else" in if-else)
        self.llm = LLMProvider(
            provider=cfg.llm.provider,
            api_key=cfg.llm.api_key,
            api_base=cfg.llm.api_base,
            model=cfg.llm.model,
            max_turns=cfg.max_turns,
        )

        # Compiler
        self.compiler = Compiler()

        # Session manager
        self._sessions = SessionManager(
            ttl_minutes=60,
            max_messages=cfg.max_session_messages,
        )

        # Metrics
        self._metrics = {"cache": 0, "llm": 0, "total": 0}

        # Pre-load cache seeds from rules.yaml (if present)
        seeds_path = cfg.cache_seeds_path
        if seeds_path is None:
            seeds_path = self._find_upwards("rules.yaml")
        if seeds_path:
            self._load_cache_seeds(seeds_path)

    # ── Main dispatch ──────────────────────────────────────────────

    def process(self, user_input: str,
                session_id: str | None = None) -> AgentResult:
        """Process user input through cache-check or LLM ReAct loop.

        Args:
            user_input: the user's natural language input
            session_id: optional conversation session ID for multi-turn

        Returns AgentResult with:
          - source: "cache" | "llm"
          - text: conversational reply
          - data: steps executed
          - latency_ms: processing time
        """
        self._metrics["total"] += 1
        t0 = time.perf_counter()

        # Get or create session
        session = self._sessions.get_or_create(session_id)
        session.add_message("user", user_input)

        # ── IF: Cache check ──
        result = self.cache.match(user_input, ToolRegistry.execute)
        if result is not None:
            self._metrics["cache"] += 1
            result.text = self._format_cache_text(result)
            result.source = "cache"
            session.add_message("assistant", result.text)
            return result

        # ── ELSE: LLM ReAct loop ──
        react_result = self._run_react_loop(user_input, session)
        self._metrics["llm"] += 1

        latency_ms = (time.perf_counter() - t0) * 1000

        agent_result = AgentResult(
            success=True,
            data={
                "steps": react_result.get("executed_steps", []),
                "workflow": react_result.get("workflow"),
            },
            source="llm",
            confidence=0.85,
            latency_ms=latency_ms,
            workflow_id=react_result.get("workflow_id"),
            text=react_result["text"],
        )

        session.add_message("assistant", react_result["text"])
        return agent_result

    # ── ReAct loop ─────────────────────────────────────────────────

    def _run_react_loop(self, user_input: str, session: Session) -> dict:
        """Run LLM ReAct loop, compile result, cache, and execute.

        Returns: {text, executed_steps, workflow, workflow_id}
        """
        # Build context from session history
        context = []
        for msg in session.messages[-20:]:  # last 20 messages
            if msg.role == "user":
                context.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant" and msg.content:
                context.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool_result" and msg.content:
                context.append({"role": "tool_result", "content": msg.content})

        # Get tool definitions
        tool_defs = ToolRegistry.list_tool_definitions()

        # Run ReAct
        react_result = self.llm.react(user_input, context, tool_defs)

        # Collect executed steps
        executed_steps: list[dict] = []

        # Execute tool calls from ReAct
        for tc in react_result.get("tool_calls", []):
            step = ActionStep(
                tool_name=tc["tool_name"],
                params=tc.get("params", {}),
                description=tc.get("description", ""),
            )
            exec_result = ToolRegistry.execute(step)
            executed_steps.append(exec_result)

        workflow_id = None
        wf = None

        # Compile and cache if there were tool calls
        if react_result.get("tool_calls"):
            intent = react_result.get("intent", user_input)
            steps_data = react_result["tool_calls"]
            wf = self.compiler.compile(intent, steps_data,
                                       original_input=user_input)
            self.cache.cache(wf)
            workflow_id = wf.id

        # For cache hits or chat-only responses, check if reply is from chat_reply
        if not react_result.get("text"):
            if executed_steps:
                text = self._format_tool_results(executed_steps)
            else:
                text = "Done."
            react_result["text"] = text

        return {
            "text": react_result["text"],
            "executed_steps": executed_steps,
            "workflow": wf,
            "workflow_id": workflow_id,
        }

    # ── Formatting ──────────────────────────────────────────────────

    def _format_cache_text(self, result: AgentResult) -> str:
        """Format a cache hit result as conversational text."""
        steps = result.data.get("steps", [])
        if not steps:
            return "Done."

        # Single chat reply
        if len(steps) == 1 and steps[0].get("tool") == "chat_reply":
            msg = steps[0].get("data", {}).get("message", "")
            if msg:
                return msg

        return self._format_tool_results(steps)

    @staticmethod
    def _format_tool_results(steps: list[dict]) -> str:
        """Format tool execution results into natural language."""
        if not steps:
            return "Done."

        parts: list[str] = []
        for r in steps:
            tool = r.get("tool", "unknown")
            if not r.get("success"):
                parts.append(f"- {tool}: 执行失败 ({r.get('error', 'unknown')})")
                continue

            data = r.get("data", {})

            if tool == "get_system_status":
                parts.append(
                    f"**系统状态**\n"
                    f"- 主机: {data.get('hostname', 'N/A')}\n"
                    f"- CPU: {data.get('cpu_percent', 'N/A')}%\n"
                    f"- 内存: {data.get('memory_used_gb', 'N/A')}/{data.get('memory_total_gb', 'N/A')} GB\n"
                    f"- 运行时间: {data.get('uptime_days', 'N/A')} 天\n"
                    f"- 活跃服务: {', '.join(data.get('active_services', []))}"
                )
            elif tool == "get_disk_usage":
                parts.append(
                    f"**磁盘空间**\n"
                    f"- 总容量: {data.get('total_gb', 'N/A')} GB\n"
                    f"- 已用: {data.get('used_gb', 'N/A')} GB\n"
                    f"- 剩余: {data.get('free_gb', 'N/A')} GB\n"
                    f"- 使用率: {data.get('use_percent', 'N/A')}%"
                )
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
            elif tool == "find_large_files":
                files = data.get("files", [])
                lines = [f"**大文件** ({data.get('path', '')}, top {data.get('top_n', '')})"]
                for f in files[:10]:
                    lines.append(f"- {f['name']}: {f.get('size_mb', '?')} MB")
                parts.append("\n".join(lines))
            elif tool == "chat_reply":
                parts.append(data.get("message", ""))
            else:
                parts.append(f"**{tool}**: OK")

        return "\n\n".join(parts)

    # ── Cache seeds ─────────────────────────────────────────────────

    def _load_cache_seeds(self, path: str):
        """Pre-load initial workflows from a rules YAML file into cache."""
        import yaml
        p = Path(path)
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for r in data.get("rules", []):
            keywords = r.get("keywords", [])
            wf = WorkflowTemplate(
                id=WorkflowTemplate.generate_id(r["name"]),
                intent=r["name"],
                steps=[ActionStep(
                    tool_name=r["tool_name"],
                    params=r.get("params", {}),
                )],
                keywords=keywords,
                # Join keywords without delimiter to avoid space
                # n-grams polluting the FAISS vocabulary
                original_input="".join(keywords) if keywords else r["name"],
            )
            self.cache.cache(wf)

    # ── Stats ───────────────────────────────────────────────────────

    def load_cache_from_disk(self):
        """Pre-load cached workflows from disk into RAM and FAISS on startup."""
        for wf in self.cache.disk.load_all():
            if wf.hit_count > 0:
                self.cache.ram.put(wf)
                self.cache.embeddings.add(wf)

    def efficiency_report(self) -> str:
        """Generate a human-readable efficiency report."""
        total = self._metrics["total"]
        cache_pct = self._metrics["cache"] / total * 100 if total > 0 else 0
        llm_pct = self._metrics["llm"] / total * 100 if total > 0 else 0
        gpu_saved = 100 - llm_pct

        lines = [
            "=" * 40,
            "  Agent Compiler — 效率报告",
            "=" * 40,
            f"  总请求:       {total}",
            f"  缓存命中:     {self._metrics['cache']} ({cache_pct:.1f}%)",
            f"  LLM调用:      {self._metrics['llm']} ({llm_pct:.1f}%)",
            f"  LLM节省:      {gpu_saved:.1f}%",
            "=" * 40,
        ]
        cache_stats = self.cache.stats()
        lines += [
            f"  缓存命中率:   {cache_stats['hit_rate']*100:.1f}%",
            f"  RAM 模板数:   {cache_stats['ram']['entries']}",
            f"  磁盘模板数:   {cache_stats['disk_count']}",
            f"  向量索引总数:  {cache_stats['embeddings']['total_vectors']}",
            "=" * 40,
        ]
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            **self._metrics,
            "cache": self.cache.stats(),
            "llm_config": self.llm.config_summary,
        }
