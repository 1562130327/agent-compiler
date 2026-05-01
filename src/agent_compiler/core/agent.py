"""Main agent: three-layer dispatch (Rule -> Cache -> LLM).

Architecture:
    L1: Rule Engine    -> keyword/regex match, pure CPU, microseconds
    L2: Pattern Cache  -> embedding similarity, pure CPU, milliseconds
    L3: LLM Fallback   -> GPU inference, seconds

The system gets faster over time as L2 accumulates cached patterns
and L1 can optionally promote high-frequency workflows into rules.
"""

from __future__ import annotations

from pathlib import Path

from agent_compiler.compiler.compiler import Compiler
from agent_compiler.core.config import AgentConfig, LLMConfig
from agent_compiler.core.types import ActionStep, AgentResult
from agent_compiler.embeddings.lightweight import LightweightEmbedding
from agent_compiler.layers.llm_fallback import LLMFallback
from agent_compiler.layers.pattern_cache import PatternCache
from agent_compiler.layers.rule_engine import RuleEngine
from agent_compiler.tools.registry import ToolRegistry


class Agent:
    """LLM-as-Compiler agent with three-layer dispatch.

    Usage:
        # Simplest: mock mode, no API needed
        agent = Agent()

        # With config
        from agent_compiler.core.config import AgentConfig, LLMConfig
        config = AgentConfig(
            llm=LLMConfig(provider="claude", api_key="sk-ant-..."),
            similarity_threshold=0.45,
        )
        agent = Agent(config)

        # Or just pass individual params
        agent = Agent(
            llm_provider="openai_compat",
            llm_api_key="sk-xxx",
            llm_api_base="https://api.openrouter.ai/v1",
            llm_model="anthropic/claude-sonnet-4",
        )
    """

    def __init__(self, config: AgentConfig | None = None, **kwargs):
        # Build config: passed config > config.yaml > env vars > kwargs > defaults
        if config is not None:
            self.config = config
        else:
            yaml_path = Path("config.yaml")
            if yaml_path.exists():
                self.config = AgentConfig.from_yaml(str(yaml_path), **kwargs)
            else:
                self.config = AgentConfig.from_env(**kwargs)

        cfg = self.config

        # Resolve rules path
        rules_path = cfg.rules_path
        if rules_path is None:
            default = Path(__file__).parent.parent.parent.parent / "rules.yaml"
            if default.exists():
                rules_path = str(default)

        # Layer 1: Rule engine
        self.rules = RuleEngine(rules_path)

        # Embedding provider (pluggable)
        self.embeddings = LightweightEmbedding(
            similarity_threshold=cfg.similarity_threshold,
        )

        # Layer 2: Pattern cache
        self.cache = PatternCache(self.embeddings, cfg.cache_dir)

        # Layer 3: LLM fallback
        self.llm = LLMFallback(
            provider=cfg.llm.provider,
            api_key=cfg.llm.api_key,
            api_base=cfg.llm.api_base,
            model=cfg.llm.model,
        )

        # Compiler for workflow compilation
        self.compiler = Compiler()

        # Metrics
        self._metrics = {"rule": 0, "cache": 0, "llm": 0, "total": 0}

    def process(self, user_input: str) -> AgentResult:
        """Process user input through the three-layer dispatch pipeline.

        Returns an AgentResult with:
          - source: "rule" | "cache" | "llm"
          - data: execution results
          - confidence: match confidence
          - latency_ms: processing time in milliseconds
        """
        self._metrics["total"] += 1

        # ── L1: Rule engine ──
        result = self.rules.match(user_input)
        if result is not None:
            self._metrics["rule"] += 1
            step = ActionStep(
                tool_name=result.data["tool"],
                params=result.data.get("params", {}),
                description="Rule match",
            )
            exec_result = ToolRegistry.execute(step)
            result.data = exec_result
            return result

        # ── L2: Pattern cache ──
        result = self.cache.match(user_input, ToolRegistry.execute)
        if result is not None:
            self._metrics["cache"] += 1
            return result

        # ── L3: LLM fallback ──
        llm_result = self.llm.reason(user_input)
        self._metrics["llm"] += 1

        if llm_result.success:
            intent = llm_result.data.get("intent", user_input)
            steps_data = llm_result.data.get("steps_data", [])
            wf = self.compiler.compile(intent, steps_data)

            # Cache for future reuse
            self.cache.cache(wf)

            # Execute
            steps = self.compiler.instantiate(wf, {})
            results = [ToolRegistry.execute(s) for s in steps]
            llm_result.data = {"steps": results, "workflow": wf}
            llm_result.workflow_id = wf.id

        return llm_result

    def efficiency_report(self) -> str:
        """Generate a human-readable efficiency report."""
        total = self._metrics["total"]
        rule_pct = self._metrics["rule"] / total * 100 if total > 0 else 0
        cache_pct = self._metrics["cache"] / total * 100 if total > 0 else 0
        llm_pct = self._metrics["llm"] / total * 100 if total > 0 else 0
        gpu_saved = 100 - llm_pct

        lines = [
            "=" * 40,
            "  Agent Compiler — 效率报告",
            "=" * 40,
            f"  总请求:       {total}",
            f"  L1 规则命中:  {self._metrics['rule']} ({rule_pct:.1f}%)",
            f"  L2 缓存命中:  {self._metrics['cache']} ({cache_pct:.1f}%)",
            f"  L3 LLM回退:   {self._metrics['llm']} ({llm_pct:.1f}%)",
            f"  GPU调用节省:  {gpu_saved:.1f}%",
            "=" * 40,
        ]
        cache_stats = self.cache.stats()
        lines += [
            f"  L2 缓存命中率: {cache_stats['hit_rate']*100:.1f}%",
            f"  RAM 模板数:    {cache_stats['ram']['entries']}",
            f"  磁盘模板数:    {cache_stats['disk_count']}",
            f"  向量索引总数:  {cache_stats['embeddings']['total_vectors']}",
            "=" * 40,
        ]
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            **self._metrics,
            "cache": self.cache.stats(),
            "rules": self.rules.stats(),
            "llm_config": self.llm.config_summary,
        }

    def load_cache_from_disk(self):
        """Pre-load cached workflows from disk into RAM and FAISS on startup."""
        for wf in self.cache.disk.load_all():
            if wf.hit_count > 0:
                self.cache.ram.put(wf)
                self.cache.embeddings.add(wf)
