#!/usr/bin/env python3
"""Quick demo script — programmatic usage example.

Run:
    python examples/demo_cli.py

Or use the CLI entry point:
    python -m agent_compiler.cli.app -i
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_compiler import Agent


def main():
    agent = Agent(llm_provider="mock", similarity_threshold=0.30)

    queries = [
        ("查看服务器状态",          "L1: rule hit"),
        ("帮我查看昨天的错误日志并生成报告", "L3: LLM -> cached"),
        ("查看最近错误日志并汇总报告",      "L2: cache hit"),
        ("看看磁盘空间还剩多少",     "L1: rule hit"),
    ]

    print("=" * 50)
    print("  Agent Compiler Demo")
    print(f"  LLM: {agent.llm.config_summary['provider']}")
    print("=" * 50)

    for i, (q, expected) in enumerate(queries, 1):
        print(f"\n[{i}] \"{q}\"")
        print(f"    expected: {expected}")
        r = agent.process(q)
        print(f"    result: L{r.source.upper()} "
              f"| {r.latency_ms:.2f}ms "
              f"| confidence={r.confidence}")

    print(f"\n{agent.efficiency_report()}")


if __name__ == "__main__":
    main()
