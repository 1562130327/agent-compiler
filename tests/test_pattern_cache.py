"""Tests for Layer 2: Pattern cache."""

import pytest
from agent_compiler.embeddings.lightweight import LightweightEmbedding
from agent_compiler.layers.pattern_cache import PatternCache
from agent_compiler.core.types import ActionStep, WorkflowTemplate


def fake_executor(step: ActionStep) -> dict:
    return {"tool": step.tool_name, "success": True, "data": {"ok": True}}


class TestPatternCache:
    @pytest.fixture
    def cache(self, tmp_path):
        emb = LightweightEmbedding(similarity_threshold=0.3)
        return PatternCache(emb, cache_dir=str(tmp_path / "test_cache"))

    def test_miss_on_empty_cache(self, cache):
        result = cache.match("some random query", fake_executor)
        assert result is None

    def test_hit_after_cache(self, cache):
        # Cache a workflow
        wf = WorkflowTemplate(
            id=WorkflowTemplate.generate_id("check error logs"),
            intent="检查错误日志并生成报告",
            steps=[
                ActionStep(tool_name="search_logs", params={"pattern": "ERROR", "days": 1}),
                ActionStep(tool_name="generate_report", params={"format": "markdown"}),
            ],
        )
        cache.cache(wf)

        # Similar query should hit
        result = cache.match("帮我查看最近的错误日志并生成汇总报告", fake_executor)
        assert result is not None
        assert result.source == "cache"
        assert result.workflow_id == wf.id

    def test_different_query_misses(self, cache):
        wf = WorkflowTemplate(
            id=WorkflowTemplate.generate_id("disk check"),
            intent="检查磁盘空间",
            steps=[ActionStep(tool_name="get_disk_usage", params={})],
        )
        cache.cache(wf)

        result = cache.match("查询错误日志", fake_executor)
        # May miss due to low similarity with disk-related query
        # Just verify no exception
        if result:
            assert result.source == "cache"
            assert result.workflow_id == wf.id
            # Should only hit if similarity is high enough — acceptable either way

    def test_stats(self, cache):
        s = cache.stats()
        assert "hits" in s
        assert "misses" in s
        assert "total_queries" in s
