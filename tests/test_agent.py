"""End-to-end tests for the Agent three-layer dispatch."""

import pytest
from agent_compiler.agent import Agent


class TestAgent:
    @pytest.fixture
    def agent(self, tmp_path):
        return Agent(
            rules_path=None,  # no rules file, test programmatic rules
            cache_dir=str(tmp_path / "agent_test_cache"),
            llm_provider="mock",
            similarity_threshold=0.4,
        )

    def test_layer1_rule_hit(self, agent):
        from agent_compiler.types import Rule
        agent.rules.add_rule(Rule(
            name="status", keywords=["服务器状态", "系统状态", "运行状态"],
            patterns=[], tool_name="get_system_status", params={"format": "summary"},
        ))
        result = agent.process("查看服务器状态")
        assert result.source == "rule"
        assert result.success
        assert "cpu_percent" in str(result.data)

    def test_layer3_llm_fallback(self, agent):
        result = agent.process("帮我查看昨天的错误日志并生成报告")
        assert result.source == "llm"
        assert result.success
        assert "steps" in result.data
        assert result.workflow_id is not None

    def test_layer2_cache_hit(self, agent):
        # First call: LLM → compiles and caches
        r1 = agent.process("帮我查看昨天的错误日志并生成报告")
        assert r1.source == "llm"
        assert r1.success

        # Second call: similar → cache hit
        r2 = agent.process("查看最近错误日志并汇总报告")
        assert r2.source == "cache"
        assert r2.success

    def test_efficiency_report(self, agent):
        from agent_compiler.types import Rule
        agent.rules.add_rule(Rule(
            name="status", keywords=["服务器状态"],
            patterns=[], tool_name="get_system_status", params={},
        ))
        agent.process("查看服务器状态")
        agent.process("新任务A")
        agent.process("新任务B")

        report = agent.efficiency_report()
        assert "效率报告" in report

    def test_metrics_accumulate(self, agent):
        from agent_compiler.types import Rule
        agent.rules.add_rule(Rule(
            name="t", keywords=["test"],
            patterns=[], tool_name="get_current_time", params={},
        ))
        agent.process("test")
        agent.process("unknown task")

        s = agent.stats()
        assert s["total"] == 2
        assert s["rule"] == 1
        assert s["llm"] == 1
