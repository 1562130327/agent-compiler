"""Tests for Layer 1: Rule engine."""

import pytest
from agent_compiler.layers.rule_engine import RuleEngine
from agent_compiler.types import Rule


class TestRuleEngine:
    def test_keyword_match(self):
        engine = RuleEngine()
        engine.add_rule(Rule(
            name="test", keywords=["服务器状态", "server status"],
            patterns=[], tool_name="get_status", params={"format": "summary"},
        ))
        result = engine.match("查看服务器状态")
        assert result is not None
        assert result.source == "rule"
        assert result.confidence == 1.0
        assert result.data["tool"] == "get_status"

    def test_regex_match(self):
        engine = RuleEngine()
        engine.add_rule(Rule(
            name="disk", keywords=[],
            patterns=[r"查看.*磁盘", r"磁盘.*空间"],
            tool_name="get_disk", params={},
        ))
        result = engine.match("帮我查看一下磁盘空间")
        assert result is not None
        assert result.source == "rule"
        assert result.workflow_id == "disk"

    def test_no_match(self):
        engine = RuleEngine()
        engine.add_rule(Rule(
            name="time", keywords=["时间", "几点"],
            patterns=[], tool_name="get_time", params={},
        ))
        result = engine.match("帮我写一篇关于黑洞的文章")
        assert result is None

    def test_latency(self):
        engine = RuleEngine()
        engine.add_rule(Rule(
            name="fast", keywords=["test"],
            patterns=[], tool_name="noop", params={},
        ))
        result = engine.match("test something")
        assert result is not None
        assert result.latency_ms < 10  # microseconds expected

    def test_stats(self):
        engine = RuleEngine()
        engine.add_rule(Rule(
            name="r1", keywords=["a"], patterns=[r"b"], tool_name="t", params={},
        ))
        s = engine.stats()
        assert s["total_rules"] == 1
        assert s["compiled_patterns"] == 1
        assert s["keyword_entries"] == 1
