#!/usr/bin/env python3
"""CLI entry point for Agent Compiler.

Usage:
    python -m agent_compiler.cli.app           # interactive mode (mock)
    python -m agent_compiler.cli.app --cli     # single-query CLI mode
    python -m agent_compiler.cli.app --report  # show cache stats and exit

Environment variables for real LLM:
    LLM_PROVIDER=openai_compat
    LLM_API_KEY=sk-xxx
    LLM_API_BASE=https://your-endpoint/v1
    LLM_MODEL=your-model
"""

import json
import sys
from pathlib import Path

from agent_compiler.core.agent import Agent
from agent_compiler.core.config import AgentConfig

# ── Demo queries for interactive mode ────────────────────────────────

_QUERIES = [
    ("查看服务器状态",       "L1: keyword rule hit"),
    ("帮我查看昨天的错误日志并生成报告", "L3: LLM, first time -> compiled & cached"),
    ("查看最近的错误日志并写个汇总报告", "L2: cache hit, pure CPU"),
    ("看看磁盘空间还剩多少",  "L1: keyword rule hit"),
    ("分析 /var/log 下最大的 10 个文件", "L3: LLM, new task -> compiled"),
    ("帮我看看硬盘空间够不够", "L2: cache hit (similar to disk query)"),
    ("现在几点？",            "L1: keyword rule hit"),
    ("搜索日志里包含 timeout 的内容", "L3: LLM, log search -> compiled"),
    ("搜索日志里包含 connection 的内容", "L2: cache hit (similar to above)"),
]


# ── Output formatting ────────────────────────────────────────────────

def _print_result(result):
    source = result.source.upper()
    labels = {"RULE": "L1", "CACHE": "L2", "LLM": "L3"}

    print(f"\n  [{labels.get(source, source)}] {source} "
          f"| latency: {result.latency_ms:.2f}ms "
          f"| confidence: {result.confidence}")

    if result.workflow_id:
        print(f"       workflow: {result.workflow_id}")

    if not result.success:
        print(f"       error: {result.error}")
        return

    data = result.data
    if "steps" in data:
        print(f"       steps executed:")
        for i, s in enumerate(data["steps"], 1):
            status = "OK" if s.get("success") else "FAIL"
            err = f" ({s.get('error', '')})" if not s.get("success") else ""
            print(f"         {i}. [{s['tool']}] {status}{err}")
    elif "tool" in data:
        status = "OK" if data.get("success") else "FAIL"
        print(f"       tool: {data['tool']} {status}")


# ── Entry points ─────────────────────────────────────────────────────

def interactive(agent: Agent):
    """Interactive numbered-query demo."""
    print("\n" + "=" * 55)
    print("  Agent Compiler -- Three-Layer Dispatch Demo")
    print("=" * 55)
    print("  L1: Rule Engine   (keyword/regex, CPU, microseconds)")
    print("  L2: Pattern Cache (embedding similarity, CPU, millisec)")
    print("  L3: LLM Fallback  (GPU inference, seconds)")
    print("=" * 55)
    print(f"  LLM: {agent.llm.config_summary['provider']} "
          f"({agent.llm.config_summary['model']})")
    print("=" * 55)

    while True:
        print()
        for i, (q, hint) in enumerate(_QUERIES, 1):
            print(f"  [{i}] {q}")
            print(f"      -> {hint}")
        print(f"\n  [r] efficiency report")
        print(f"  [s] detailed stats (JSON)")
        print(f"  [c] custom input")
        print(f"  [q] quit")

        choice = input("\n  Select: ").strip()

        if choice == "q":
            print("  Goodbye!")
            break
        elif choice == "r":
            print(f"\n{agent.efficiency_report()}")
        elif choice == "s":
            print(json.dumps(agent.stats(), indent=2, ensure_ascii=False))
        elif choice == "c":
            user_input = input("  Query: ").strip()
            if user_input:
                result = agent.process(user_input)
                _print_result(result)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(_QUERIES):
                    query, hint = _QUERIES[idx]
                    print(f'\n  Query: "{query}"')
                    print(f"  Expected: {hint}")
                    result = agent.process(query)
                    _print_result(result)
                    if agent._metrics["total"] % 3 == 0:
                        print(f"\n{agent.efficiency_report()}")
                else:
                    print("  Invalid choice")
            except ValueError:
                print("  Invalid choice")


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Agent Compiler - Three-layer dispatch agent")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive demo mode (default)")
    parser.add_argument("--query", "-q", type=str,
                        help="Single query mode")
    parser.add_argument("--report", "-r", action="store_true",
                        help="Show cache stats and exit")
    parser.add_argument("--config", "-c", type=str,
                        help="Path to config.yaml")
    parser.add_argument("--provider", type=str,
                        help="LLM provider: mock|claude|openai|openai_compat")
    parser.add_argument("--api-key", type=str,
                        help="LLM API key")
    parser.add_argument("--api-base", type=str,
                        help="Custom API base URL")
    parser.add_argument("--model", type=str,
                        help="Model name")
    parser.add_argument("--cache-dir", type=str, default="./agent_cache",
                        help="Cache directory")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Similarity threshold (0-1)")

    args = parser.parse_args()

    # Build config
    kwargs = dict(
        llm_provider=args.provider,
        llm_api_key=args.api_key,
        llm_api_base=args.api_base,
        llm_model=args.model,
        cache_dir=args.cache_dir,
        similarity_threshold=args.threshold,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if args.config:
        config = AgentConfig.from_yaml(args.config, **kwargs)
    elif Path("config.yaml").exists():
        config = AgentConfig.from_yaml("config.yaml", **kwargs)
    else:
        config = AgentConfig.from_env(**kwargs)

    agent = Agent(config)

    if args.report:
        # Load existing cache and show stats
        agent.load_cache_from_disk()
        print(agent.efficiency_report())
        return

    if args.query:
        result = agent.process(args.query)
        _print_result(result)
        print(f"\n{agent.efficiency_report()}")
        return

    # Default: interactive mode
    interactive(agent)


if __name__ == "__main__":
    main()
