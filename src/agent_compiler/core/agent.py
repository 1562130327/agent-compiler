"""Main agent: if-else dispatch (Cache -> LLM ReAct) + self-evolving memory.

Architecture:
    Cache check → hit:  execute cached workflow, pure CPU, milliseconds
                 → miss: LLM ReAct loop → compile → cache → execute

    Every interaction feeds the self-evolving memory, which injects
    relevant context into future LLM calls. The system gets faster
    AND smarter over time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

from agent_compiler.core.memory import MemoryStore

from agent_compiler.compiler.compiler import Compiler
from agent_compiler.core.config import AgentConfig, LLMConfig
from agent_compiler.core.executor import Executor, ExecutionResult
from agent_compiler.core.planning import ExecutionPlan, Planner
from agent_compiler.core.types import (
    ActionStep, AgentResult, Session, WorkflowTemplate,
)
from agent_compiler.embeddings.lightweight import LightweightEmbedding
from agent_compiler.layers.llm_fallback import LLMProvider
from agent_compiler.layers.pattern_cache import PatternCache
from agent_compiler.layers.thought_tree import ThoughtTree, should_use_tot
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

        # Self-evolving memory
        memory_dir = str(Path(cfg.cache_dir).parent / "agent_memory")
        self.memory = MemoryStore(memory_dir)

        # ── New v0.3 components ──────────────────────────────────

        # MCP client — connects to external tool servers
        self.mcp_manager = None
        if cfg.mcp_servers:
            from agent_compiler.core.mcp_client import MCPClientManager
            self.mcp_manager = MCPClientManager()
            self.mcp_manager.configure_from_dicts(cfg.mcp_servers)
            try:
                self.mcp_manager.connect_all()
            except Exception:
                pass  # MCP is best-effort at startup

        # Skill system — loads SKILL.md files
        from agent_compiler.skills.loader import SkillLoader
        self.skills = SkillLoader()
        self.skills.discover(extra_dirs=cfg.skills_dirs)

        # Context manager — token-aware window management
        from agent_compiler.core.context_manager import ContextManager, ContextBudget
        self.context_mgr = ContextManager(
            budget=ContextBudget(max_tokens=cfg.context_max_tokens)
        )

        # Planner — task decomposition
        self.planner = Planner(self.llm)

        # Executor — plan execution with dependency management
        self.executor = Executor(max_parallel=1)

        # Metrics
        self._metrics = {"cache": 0, "llm": 0, "total": 0}
        self._total_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._interaction_count = 0

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

        # ── ELSE: LLM ReAct loop (or Plan+Execute for complex tasks) ──

        # Detect if this is a multi-step task that benefits from planning
        plan_keywords = ["首先", "然后", "接着", "最后", "步骤", "逐步",
                        "first", "then", "next", "finally", "step by step",
                        "创建", "生成", "构建", "部署", "安装配置"]
        use_plan = any(kw in user_input.lower() for kw in plan_keywords) and len(user_input) > 30

        if use_plan and not self.llm.is_mock:
            react_result = self._plan_and_execute(user_input, session)
            # If plan failed entirely, already fell through to ReAct
        else:
            react_result = self._run_react_loop(user_input, session)
        self._metrics["llm"] += 1
        self._interaction_count += 1

        # Accumulate token usage
        llm_tokens = react_result.get("tokens", {})
        for k in ("prompt", "completion", "total"):
            self._total_tokens[k] += llm_tokens.get(k, 0)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Auto-extract memories from this interaction (using LLM for quality)
        reply_text = react_result.get("text", "")
        try:
            self.memory.extract_from_interaction(user_input, reply_text, self.llm)
        except Exception:
            pass  # memory extraction is best-effort

        # Save episodic memory (conversation summary)
        try:
            self.memory.add_episodic(user_input, reply_text)
        except Exception:
            pass

        # Save reflexion feedback as procedural memory
        reflexion_log = react_result.get("reflexion_log", [])
        for entry in reflexion_log:
            if entry.get("score", 0) < 4 and entry.get("feedback"):
                try:
                    self.memory.add_procedural(
                        rule=entry["feedback"],
                        context=f"reflexion_rev{entry.get('round', 0)}",
                        confidence=0.55,
                    )
                except Exception:
                    pass

        # Periodic memory consolidation (every 10 LLM interactions)
        if self._interaction_count % 10 == 0:
            try:
                changes = self.memory.consolidate()
                if changes > 0:
                    react_result["text"] += f"\n\n[记忆演化: {changes} 条记录已更新]"
            except Exception:
                pass

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
            tokens=llm_tokens,
        )

        session.add_message("assistant", react_result["text"])
        return agent_result

    # ── ReAct loop ─────────────────────────────────────────────────

    def _run_react_loop(self, user_input: str, session: Session,
                        plan: ExecutionPlan | None = None) -> dict:
        """Run LLM ReAct loop, compile result, cache, and execute.

        Returns: {text, executed_steps, workflow, workflow_id}
        """
        # Build context from session history (exclude current user msg —
        # it will be appended by llm.react() as the active prompt)
        history = session.messages[:-1] if session.messages else []
        context = []
        for msg in history[-20:]:
            if msg.role == "user":
                context.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant" and msg.content:
                context.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool_result" and msg.content:
                context.append({"role": "tool_result", "content": msg.content})

        # Get tool definitions
        tool_defs = ToolRegistry.list_tool_definitions()

        # Inject relevant memories into context
        memory_ctx = self.memory.context_for_query(user_input)
        if memory_ctx:
            context.insert(0, {"role": "system", "content": memory_ctx})

        # Inject matched skill prompt (if auto-triggered)
        skill = self.skills.match(user_input)
        if skill:
            skill_prompt = skill.resolve_body()
            context.insert(0, {"role": "system",
                               "content": f"## 激活技能: {skill.name}\n{skill_prompt}"})

        # Inject execution plan (if provided)
        if plan:
            context.insert(0, {"role": "system",
                               "content": plan.to_prompt()})

        # ── Tree-of-Thought: deep reasoning for complex tasks ──
        tot_result = None
        if should_use_tot(user_input) and not self.llm.is_mock:
            try:
                tot = ThoughtTree(self.llm, beam_width=2, max_depth=3)
                tot.solve(user_input, context=memory_ctx or "")
                tot_text = tot.best_path_text()
                if tot_text and hasattr(tot, '_best_path') and len(tot._best_path) > 2:
                    context.insert(0, {"role": "system",
                                       "content": tot_text})
                    tot_result = tot_text
            except Exception:
                pass  # ToT is best-effort, fall through to regular ReAct

        # Use context manager to build token-aware context
        skill_list_prompt = self.skills.build_context_prompt()
        full_system = skill_list_prompt if skill_list_prompt else ""
        managed_context = self.context_mgr.build_context(
            system_prompt=full_system,
            memory_context=memory_ctx,
        )
        if managed_context:
            # Merge managed context with our built context
            # Managed context goes first (system/memory), then our context
            pass  # context is already built above

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

        # Compile and cache the result (tool calls or chat reply — both benefit)
        if react_result.get("tool_calls"):
            intent = react_result.get("intent", user_input)
            steps_data = react_result["tool_calls"]
            wf = self.compiler.compile(intent, steps_data,
                                       original_input=user_input)
            self.cache.cache(wf)
            workflow_id = wf.id
        elif react_result.get("text"):
            # Cache chat-only responses too — avoid re-querying LLM for
            # the same conversational question (e.g. "what tools do you have")
            chat_text = react_result["text"]
            wf = self.compiler.compile(
                f"chat: {user_input}",
                [{"tool_name": "chat_reply",
                  "params": {"message": chat_text},
                  "description": f"Cached reply: {user_input[:40]}"}],
                original_input=user_input,
            )
            self.cache.cache(wf)
            workflow_id = wf.id

        # Ensure we always have text
        if not react_result.get("text"):
            if executed_steps:
                text = self._format_tool_results(executed_steps)
            else:
                text = "Done."
            react_result["text"] = text

        result = {
            "text": react_result["text"],
            "executed_steps": executed_steps,
            "workflow": wf,
            "workflow_id": workflow_id,
            "tokens": react_result.get("tokens", {}),
            "reflexion_log": react_result.get("reflexion_log", []),
            "tot_used": tot_result is not None,
        }
        return result

    # ── Plan-driven execution ────────────────────────────────────────

    def _plan_and_execute(self, user_input: str, session: Session) -> dict:
        """Plan-driven execution path for complex tasks.

        Uses Planner to decompose the task, then Executor to run steps
        with dependency management and auto-retry.
        """
        tool_names = [td["name"] for td in ToolRegistry.list_tool_definitions()]
        plan = self.planner.plan(
            user_input,
            available_tools=tool_names,
            context=f"Session has {len(session.messages)} prior messages",
        )

        # Run via executor
        result = self.executor.execute(plan, abort_on_failure=False)

        # Build response text
        response_text = result.to_text()

        # If the plan had errors, fall back to ReAct
        if result.steps_failed > 0 and result.steps_executed == 0:
            # Plan failed entirely — fall through to ReAct
            react_result = self._run_react_loop(user_input, session, plan=plan)
            react_result["plan"] = plan
            return react_result

        return {
            "text": response_text,
            "executed_steps": [],
            "workflow": None,
            "workflow_id": None,
            "tokens": {},
            "reflexion_log": [],
            "tot_used": False,
            "plan": plan,
            "plan_result": result,
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
            elif tool == "execute_shell":
                stdout = data.get("stdout", "")
                stderr = data.get("stderr", "")
                exit_code = data.get("exit_code", -1)
                header = f"**Shell 执行结果** (exit={exit_code})"
                body = ""
                if stdout:
                    body += f"\n```\n{stdout[:1500]}\n```"
                if stderr:
                    body += f"\nstderr:\n```\n{stderr[:500]}\n```"
                parts.append(header + body)
            elif tool == "read_file":
                content = data.get("content", "")
                fpath = data.get("path", "")
                lines = data.get("returned_lines", 0)
                total = data.get("total_lines", 0)
                header = f"**{fpath}** ({lines}/{total} 行)"
                parts.append(header + f"\n```\n{content[:3000]}\n```")
            elif tool == "write_file":
                fpath = data.get("path", "")
                size = data.get("bytes_written", 0)
                parts.append(f"**文件已写入**: {fpath} ({size} bytes)")
            elif tool == "edit_file":
                fpath = data.get("path", "")
                occ = data.get("occurrences", 0)
                parts.append(f"**文件已编辑**: {fpath} ({occ} 处匹配，替换了第 1 处)")
            elif tool == "glob_files":
                files = data.get("files", [])
                total = data.get("total", 0)
                lines = [f"**文件匹配** ({data.get('pattern', '')}, 共 {total} 个)"]
                for f in files[:20]:
                    lines.append(f"- {f['path']} ({f.get('size_bytes', 0)} bytes)")
                if total > 20:
                    lines.append(f"... 还有 {total - 20} 个")
                parts.append("\n".join(lines))
            elif tool == "search_files":
                matches = data.get("matches", [])
                total = data.get("total_matches", 0)
                lines = [f"**代码搜索** ({data.get('pattern', '')}, 共 {total} 处匹配)"]
                for m in matches[:15]:
                    lines.append(f"- {m['file']}:{m['line']}: {m['content']}")
                if total > 15:
                    lines.append(f"... 还有 {total - 15} 处")
                parts.append("\n".join(lines))
            elif tool == "execute_python":
                stdout = data.get("stdout", "")
                stderr = data.get("stderr", "")
                header = f"**Python 执行结果** (exit={data.get('exit_code', -1)})"
                body = ""
                if stdout:
                    body += f"\n```\n{stdout[:2000]}\n```"
                if stderr:
                    body += f"\nstderr:\n```\n{stderr[:500]}\n```"
                parts.append(header + body)
            elif tool == "install_package":
                pkg = data.get("package", "")
                ok = data.get("success", False)
                status = "安装成功" if ok else "安装失败"
                parts.append(f"**pip install {pkg}**: {status}")
            elif tool == "run_tests":
                output = data.get("output", "")
                summary = data.get("summary", {})
                header = f"**测试结果** (通过:{summary.get('passed',0)}, 失败:{summary.get('failed',0)})"
                parts.append(header + f"\n```\n{output[:2000]}\n```")
            elif tool == "web_fetch":
                content = data.get("content", "")
                url = data.get("url", "")
                parts.append(f"**网页内容** ({url})\n{content[:2000]}")
            elif tool == "web_search":
                results = data.get("results", [])
                lines = [f"**搜索结果** ({data.get('query', '')}, 共 {len(results)} 条)"]
                for r in results[:5]:
                    lines.append(f"- [{r['title']}]({r['url']})")
                    if r.get("snippet"):
                        lines.append(f"  {r['snippet'][:200]}")
                parts.append("\n".join(lines))
            elif tool == "git_status":
                files = data.get("files", [])
                path = data.get("path", ".")
                lines = [f"**Git 状态** ({path}, {len(files)} 个有变更的文件)"]
                for f in files[:20]:
                    lines.append(f"- {f}")
                parts.append("\n".join(lines))
            elif tool == "git_diff":
                stat = data.get("stat", "")
                diff = data.get("diff", "")
                parts.append(f"**Git Diff**\n```\n{stat or diff[:1500]}\n```")
            elif tool == "git_log":
                commits = data.get("commits", [])
                lines = [f"**Git Log** ({len(commits)} 条提交)"]
                for c in commits[:10]:
                    lines.append(f"- {c}")
                parts.append("\n".join(lines))
            elif tool == "list_processes":
                procs = data.get("processes", [])
                total = data.get("total", 0)
                lines = [f"**进程列表** (共 {total} 个, 按{data.get('sort_by', 'cpu')}排序)"]
                for p in procs[:10]:
                    if sys.platform == "win32":
                        lines.append(f"- [{p.get('pid', '?')}] {p.get('name', '?')} ({p.get('memory_kb', 0)} KB)")
                    else:
                        lines.append(f"- [{p.get('pid', '?')}] {p.get('command', '?')[:60]} (CPU:{p.get('cpu_pct', 0)}% MEM:{p.get('mem_pct', 0)}%)")
                parts.append("\n".join(lines))
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

        tk = self._total_tokens
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
            "─" * 40,
            f"  Token 消耗",
            f"    累计 Prompt:     {tk['prompt']:,}",
            f"    累计 Completion: {tk['completion']:,}",
            f"    累计 Total:      {tk['total']:,}",
            "─" * 40,
            f"  记忆系统",
        ]
        mem_stats = self.memory.stats()
        lines.append(f"    记忆总数:     {mem_stats['total_memories']}")
        for cat, count in mem_stats.get("by_category", {}).items():
            cat_labels = {"user_profile": "用户画像", "project": "项目信息",
                         "pattern": "任务模式", "feedback": "反馈", "knowledge": "知识"}
            label = cat_labels.get(cat, cat)
            lines.append(f"      {label}: {count}")
        tier_info = mem_stats.get("by_tier", {})
        if tier_info:
            lines.append(f"    分层: 语义{tier_info.get('semantic',0)} 情节{tier_info.get('episodic',0)} 程序{tier_info.get('procedural',0)}")
        evo = mem_stats.get("evolution", {})
        lines.append(f"    演化: 提取{evo.get('extractions',0)} 合并{evo.get('merges',0)} 清理{evo.get('prunes',0)}")
        lines.append("=" * 40)
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            **self._metrics,
            "tokens": dict(self._total_tokens),
            "memory": self.memory.stats(),
            "cache": self.cache.stats(),
            "llm_config": self.llm.config_summary,
        }
