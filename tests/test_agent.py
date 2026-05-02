"""End-to-end tests for the Agent if-else dispatch."""

import pytest
from agent_compiler.agent import Agent


class TestAgent:
    @pytest.fixture
    def agent(self, tmp_path):
        return Agent(
            cache_seeds_path=None,  # no seed rules
            cache_dir=str(tmp_path / "agent_test_cache"),
            llm_provider="mock",
            similarity_threshold=0.4,
        )

    def test_llm_task_executes_tools(self, agent):
        """New task goes to LLM, executes tools, returns text."""
        result = agent.process("帮我分析网络连接状态")
        assert result.source == "llm"
        assert result.success
        assert result.text  # should have conversational reply

    def test_cache_hit_after_llm(self, agent):
        """First call goes to LLM, second similar call hits cache."""
        r1 = agent.process("帮我查看昨天的错误日志并生成报告")
        assert r1.source == "llm"
        assert r1.success
        assert r1.workflow_id is not None

        r2 = agent.process("查看最近错误日志并汇总报告")
        assert r2.source == "cache"
        assert r2.success

    def test_llm_handles_chat(self, agent):
        """Conversational input gets a text reply without tools."""
        result = agent.process("你好")
        assert result.success
        assert result.text

    def test_llm_handles_self_intro(self, agent):
        """Self-intro question gets a descriptive reply."""
        result = agent.process("你能做什么")
        assert result.success
        assert result.text

    def test_efficiency_report(self, agent):
        """Report should work with new if-else architecture."""
        agent.process("新任务A")
        report = agent.efficiency_report()
        assert "效率报告" in report
        assert "缓存命中" in report

    def test_metrics_accumulate(self, agent):
        agent.process("unknown task")
        agent.process("another task")
        s = agent.stats()
        assert s["total"] == 2
        assert s["llm"] == 2

    def test_cache_seeds_produce_hit(self, tmp_path):
        """Agent with cache seeds hits cache on first matching query."""
        import yaml, os
        seeds_path = str(tmp_path / "seeds.yaml")
        seeds = {
            "rules": [{
                "name": "system_status",
                "keywords": ["服务器状态", "系统状态"],
                "patterns": [],
                "tool_name": "get_system_status",
                "params": {"format": "summary"},
            }]
        }
        with open(seeds_path, "w", encoding="utf-8") as f:
            yaml.dump(seeds, f)

        agent = Agent(
            cache_seeds_path=seeds_path,
            cache_dir=str(tmp_path / "cache"),
            llm_provider="mock",
        )
        result = agent.process("查看服务器状态")
        assert result.source == "cache"
        assert result.success

    def test_session_messages_accumulate(self, agent):
        """Multi-turn: session retains message history."""
        sid = "test_sess"
        agent.process("查看服务器状态", session_id=sid)
        agent.process("现在几点", session_id=sid)
        sess = agent._sessions.get_or_create(sid)
        assert len(sess.messages) >= 4  # 2 user + 2 assistant
