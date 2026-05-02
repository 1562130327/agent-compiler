"""Tests for loading cache seeds from YAML files."""

import pytest
import yaml
from agent_compiler.core.agent import Agent


class TestCacheSeeds:
    @pytest.fixture
    def seeds_path(self, tmp_path):
        p = tmp_path / "test_seeds.yaml"
        seeds = {
            "rules": [
                {
                    "name": "server_status",
                    "keywords": ["服务器状态", "系统状态"],
                    "patterns": [],
                    "tool_name": "get_system_status",
                    "params": {"format": "summary"},
                },
                {
                    "name": "disk_check",
                    "keywords": ["磁盘空间", "硬盘空间"],
                    "patterns": [],
                    "tool_name": "get_disk_usage",
                    "params": {},
                },
            ]
        }
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(seeds, f)
        return p

    def test_seeds_loaded_as_workflows(self, seeds_path, tmp_path):
        agent = Agent(
            cache_seeds_path=str(seeds_path),
            cache_dir=str(tmp_path / "cache"),
            llm_provider="mock",
        )
        stats = agent.cache.stats()
        assert stats["embeddings"]["total_vectors"] >= 2

    def test_seed_produces_cache_hit(self, seeds_path, tmp_path):
        agent = Agent(
            cache_seeds_path=str(seeds_path),
            cache_dir=str(tmp_path / "cache_hit"),
            llm_provider="mock",
        )
        result = agent.process("帮我查看服务器状态")
        assert result.source == "cache"
        assert result.success

    def test_seed_produces_disk_hit(self, seeds_path, tmp_path):
        agent = Agent(
            cache_seeds_path=str(seeds_path),
            cache_dir=str(tmp_path / "cache_disk"),
            llm_provider="mock",
        )
        result = agent.process("看看磁盘空间还剩多少")
        assert result.source == "cache"
        assert result.success

    def test_seed_missing_path_ok(self, tmp_path):
        agent = Agent(
            cache_seeds_path=str(tmp_path / "nonexistent.yaml"),
            cache_dir=str(tmp_path / "cache_noseed"),
            llm_provider="mock",
        )
        assert agent.cache.stats()["embeddings"]["total_vectors"] == 0
