#!/usr/bin/env python3
"""CLI entry point for Agent Compiler.

Usage:
    agent-compiler                    # interactive chat mode (mock)
    agent-compiler --query "..."      # single query
    agent-compiler --report           # show cache stats and exit

Environment variables for real LLM:
    LLM_PROVIDER=openai_compat
    LLM_API_KEY=sk-xxx
    LLM_API_BASE=https://api.deepseek.com
    LLM_MODEL=deepseek-chat
"""

import json
import sys
import time
from pathlib import Path

from agent_compiler.core.agent import Agent
from agent_compiler.core.config import AgentConfig


def _print_result(result):
    """Display an agent result nicely."""
    source_label = {"cache": "CACHE", "llm": "LLM"}.get(result.source, result.source)

    # Token info line
    token_info = ""
    if result.tokens and result.tokens.get("total"):
        tk = result.tokens
        token_info = f" | {tk['total']:,} tokens (P:{tk['prompt']:,} C:{tk['completion']:,})"

    # Show conversational text if available
    if result.text:
        print(f"\n{result.text}")
        print(f"\n  [{source_label}] {result.latency_ms:.1f}ms{token_info}")
    elif result.success and "steps" in result.data:
        print(f"\n  [{source_label}] {result.latency_ms:.1f}ms{token_info}")
        for s in result.data["steps"]:
            status = "OK" if s.get("success") else "FAIL"
            err = f" ({s.get('error', '')})" if not s.get("success") else ""
            print(f"    {s['tool']}: {status}{err}")
    elif not result.success:
        print(f"\n  [{source_label}] Error: {result.error}")


def interactive(agent: Agent):
    """Interactive chat mode with multi-turn conversation."""

    print("\n" + "=" * 55)
    print("  Agent Compiler — 对话模式")
    print("=" * 55)
    print("  直接打字即可，Agent 会分析意图并执行。")
    print("  交互越多，缓存越丰富，响应越快。")
    print("=" * 55)
    print(f"  LLM: {agent.llm.config_summary['provider']} "
          f"({agent.llm.config_summary['model']})")
    print("=" * 55)
    print()
    print("  输入「报告」看统计，「帮助」看用法，「退出」结束。")
    print()

    session_id = None

    while True:
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见!")
            break

        if not user_input:
            continue

        inp_lower = user_input.lower()
        inp_len = len(user_input)

        # ── Built-in commands (short input only) ─────────────────
        def _is_cmd(*keywords: str) -> bool:
            if inp_len > 6:
                return False
            return any(k in inp_lower for k in keywords)

        if _is_cmd("退出", "再见", "quit", "exit", "q"):
            print(f"\n{agent.efficiency_report()}")
            print("\n  再见!")
            break

        if _is_cmd("帮助", "help", "?", "？"):
            print()
            print("  Agent Compiler — 使用说明")
            print("  " + "-" * 40)
            print("  直接输入你想做的事，比如：")
            print("    - 查看服务器状态")
            print("    - 帮我查一下错误日志")
            print("    - 磁盘空间还剩多少")
            print("    - 现在几点")
            print()
            print("  内置指令（短输入）：")
            print("    报告 / 统计  — 查看效率报告")
            print("    记忆          — 查看自演化记忆系统状态")
            print("    清除 / 清空  — 清除缓存重新开始")
            print("    帮助 / help  — 显示本帮助")
            print("    退出 / quit  — 退出程序")
            print()
            continue

        if _is_cmd("报告", "统计", "report", "stats"):
            print(f"\n{agent.efficiency_report()}")
            continue

        if _is_cmd("记忆", "memory", "mem"):
            mem = agent.memory
            mem_stats = mem.stats()
            print(f"\n{'=' * 40}")
            print("  自演化记忆系统")
            print(f"{'=' * 40}")
            print(f"  记忆总数: {mem_stats['total_memories']}")
            print(f"  分类:")
            cat_labels = {"user_profile": "用户画像", "project": "项目信息",
                         "pattern": "任务模式", "feedback": "反馈", "knowledge": "知识"}
            for cat, count in mem_stats.get("by_category", {}).items():
                print(f"    {cat_labels.get(cat, cat)}: {count}")
            evo = mem_stats.get("evolution", {})
            print(f"  演化统计:")
            print(f"    自动提取: {evo.get('extractions', 0)} 次")
            print(f"    合并相似: {evo.get('merges', 0)} 次")
            print(f"    清理过期: {evo.get('prunes', 0)} 次")
            print(f"    巩固轮次: {evo.get('consolidations', 0)} 次")
            print(f"{'=' * 40}")
            # Show recent memories
            all_mems = mem.all()
            if all_mems:
                print("\n  最近记忆:")
                for m in sorted(all_mems, key=lambda x: -x.updated_at)[:5]:
                    age = (time.time() - m.created_at) / 3600
                    age_str = f"{age:.1f}h前" if age < 48 else f"{age/24:.1f}d前"
                    print(f"    [{m.category}] {m.title} (置信度:{m.confidence:.0%}, {age_str})")
            print()
            continue

        if _is_cmd("清除", "清空", "重置", "clear", "reset"):
            agent.cache.ram._cache.clear()
            agent.cache.embeddings._faiss = None
            agent.cache.embeddings._index.clear()
            agent.cache.embeddings._next_id = 0
            agent._metrics = {"cache": 0, "llm": 0, "total": 0}
            agent._total_tokens = {"prompt": 0, "completion": 0, "total": 0}
            session_id = None
            print("  缓存已清空，计数器和 Token 统计已归零。")
            continue

        # ── Normal query ───────────────────────────────────────
        result = agent.process(user_input, session_id=session_id)
        if session_id is None and result.workflow_id:
            # Start tracking session for multi-turn
            session_id = f"sess_{result.workflow_id}"
        _print_result(result)
        sys.stdout.flush()  # 强制刷新，确保 Windows 下立即显示

        if agent._metrics["total"] % 5 == 0:
            print(f"\n  [自动报告 — 每 5 次查询触发]")
            print(agent.efficiency_report())


def main():
    """Main CLI entry point."""
    # 强制行缓冲，解决 Windows 下输出不立即显示的问题
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    import argparse

    parser = argparse.ArgumentParser(
        description="Agent Compiler - AI Agent with automatic caching")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive chat mode (default)")
    parser.add_argument("--query", "-q", type=str,
                        help="Single query mode")
    parser.add_argument("--web", "-w", action="store_true",
                        help="Start web UI (http://localhost:8220)")
    parser.add_argument("--port", "-p", type=int, default=8220,
                        help="Web UI port (default: 8220)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser in web mode")
    parser.add_argument("--report", "-r", action="store_true",
                        help="Show cache stats and exit")
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config.yaml")
    parser.add_argument("--provider", type=str,
                        help="LLM provider: mock|claude|openai|openai_compat")
    parser.add_argument("--api-key", type=str, help="LLM API key")
    parser.add_argument("--api-base", type=str, help="Custom API base URL")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--cache-dir", type=str, default="./agent_cache",
                        help="Cache directory")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Similarity threshold (0-1)")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="ReAct loop max iterations")
    parser.add_argument("--seeds", type=str, default="rules.yaml",
                        help="Cache seeds YAML file")

    args = parser.parse_args()

    kwargs = dict(
        llm_provider=args.provider,
        llm_api_key=args.api_key,
        llm_api_base=args.api_base,
        llm_model=args.model,
        cache_dir=args.cache_dir,
        similarity_threshold=args.threshold,
        max_turns=args.max_turns,
        cache_seeds_path=args.seeds,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def _find_upwards(filename: str) -> str | None:
        """Search CWD, parent dirs, and common locations for a file."""
        # 1. CWD and parent directories
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

    if args.config:
        config = AgentConfig.from_yaml(args.config, **kwargs)
    else:
        found = _find_upwards("config.yaml")
        if found:
            config = AgentConfig.from_yaml(found, **kwargs)
        else:
            config = AgentConfig.from_env(**kwargs)

    agent = Agent(config)

    if args.report:
        agent.load_cache_from_disk()
        print(agent.efficiency_report())
        return

    if args.web:
        from agent_compiler.web.server import start_server
        print(f"\n  Agent Compiler Web UI 启动中...")
        print(f"  浏览器打开后即可使用: http://127.0.0.1:{args.port}")
        print(f"  按 Ctrl+C 退出\n")
        start_server(host="127.0.0.1", port=args.port,
                    open_browser=not args.no_browser)
        return

    if args.query:
        result = agent.process(args.query)
        _print_result(result)
        print(f"\n{agent.efficiency_report()}")
        return

    interactive(agent)


def web_main():
    """Entry point for 'agent-compiler-web' command — directly opens web UI."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    from agent_compiler.web.server import start_server
    print("Agent Compiler Web UI")
    print("浏览器打开后即可使用: http://127.0.0.1:8220")
    print("按 Ctrl+C 退出\n")
    start_server()


if __name__ == "__main__":
    main()
