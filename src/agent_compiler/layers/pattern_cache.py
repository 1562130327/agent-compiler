"""Layer 2: Pattern cache — embedding similarity matching + workflow execution."""

from __future__ import annotations

import time

from agent_compiler.compiler.compiler import Compiler
from agent_compiler.embeddings.base import EmbeddingProvider
from agent_compiler.storage.cloud_store import CloudStore
from agent_compiler.storage.disk_store import DiskStore
from agent_compiler.storage.ram_store import RamStore
from agent_compiler.core.types import ActionStep, AgentResult, WorkflowTemplate


class PatternCache:
    """Three-tier pattern cache: RAM → Disk(SSD) → Cloud.

    Layer 2 of the agent: matches user input to cached workflow templates
    using embedding similarity. Pure CPU execution when cached.
    """

    def __init__(self, embedding_provider: EmbeddingProvider,
                 cache_dir: str = "./agent_cache"):
        self.embeddings = embedding_provider
        self.ram = RamStore()
        self.disk = DiskStore(cache_dir)
        self.cloud = CloudStore(f"{cache_dir}/cloud")
        self.compiler = Compiler()
        self._metrics = {"hits": 0, "misses": 0}

    def match(self, user_input: str, tool_executor) -> AgentResult | None:
        """Try to match user input to a cached workflow template."""
        t0 = time.perf_counter()

        # Step 1: FAISS similarity search
        matches = self.embeddings.search(user_input, k=5)
        if not matches:
            self._metrics["misses"] += 1
            return None

        # Step 2: Try to load best match (RAM → Disk → Cloud)
        for wf_id, score in matches:
            wf = self._load_workflow(wf_id)
            if wf is None:
                continue

            # Step 3: Execute the cached workflow with CPU
            try:
                params = self._extract_params(user_input, wf)
                steps = self.compiler.instantiate(wf, params)
                results = [tool_executor(s) for s in steps]

                latency = (time.perf_counter() - t0) * 1000
                self._metrics["hits"] += 1
                self.ram.put(wf)
                self.disk.update_hit(wf.id, wf.hit_count)

                return AgentResult(
                    success=True,
                    data={"steps": results, "workflow": wf},
                    source="cache",
                    confidence=round(score, 4),
                    latency_ms=latency,
                    workflow_id=wf.id,
                )
            except Exception as e:
                continue

        self._metrics["misses"] += 1
        return None

    def cache(self, wf: WorkflowTemplate):
        """Store a new compiled workflow in all storage tiers."""
        self.ram.put(wf)
        self.disk.save(wf)
        if wf.embedding is None:
            # Use original input (Chinese) for embedding — keywords alone
            # are too sparse for FAISS n-gram overlap with future queries.
            embed_text = wf.original_input or (" ".join(wf.keywords) if wf.keywords else wf.intent)
            wf.embedding = self.embeddings.encode(embed_text)
        self.embeddings.add(wf)

    def _load_workflow(self, wf_id: str) -> WorkflowTemplate | None:
        """Load workflow from RAM → Disk → Cloud."""
        wf = self.ram.get(wf_id)
        if wf:
            return wf

        wf = self.disk.load(wf_id)
        if wf:
            self.ram.put(wf)
            return wf

        # Cloud restore (optional)
        data = self.cloud.restore(wf_id)
        if data:
            from agent_compiler.core.types import ActionStep, WorkflowTemplate
            wf = WorkflowTemplate(
                id=data["id"], intent=data["intent"],
                steps=[ActionStep(**s) for s in data["steps"]],
                hit_count=data.get("hit_count", 0),
                confidence=data.get("confidence", 1.0),
                params_schema=data.get("params_schema", {}),
                keywords=data.get("keywords", []),
            )
            self.disk.save(wf)
            self.ram.put(wf)
            return wf

        return None

    def _extract_params(self, user_input: str, wf: WorkflowTemplate) -> dict:
        """Extract parameter values from user input for workflow instantiation."""
        params = {}
        for key, schema in wf.params_schema.items():
            default = schema.get("default")
            if default is not None:
                params[key] = default
            # Try to find the value in user input
            import re
            if isinstance(default, str):
                # Look for quoted strings, paths, or specific values in user input
                for val in re.findall(r'["\']([^"\']+)["\']', user_input):
                    if "/" in val or "\\" in val:
                        params[key] = val
                        break
            elif isinstance(default, (int, float)):
                # Look for numbers near keywords
                match = re.search(r'(\d+)\s*(?:个|条|项|次|days?|天|files?|文件)', user_input)
                if match:
                    params[key] = int(match.group(1)) if isinstance(default, int) else float(match.group(1))
        return params

    def stats(self) -> dict:
        total = self._metrics["hits"] + self._metrics["misses"]
        hit_rate = self._metrics["hits"] / total if total > 0 else 0
        return {
            **self._metrics,
            "total_queries": total,
            "hit_rate": round(hit_rate, 4),
            "ram": self.ram.stats(),
            "disk_count": self.disk.count(),
            "embeddings": self.embeddings.stats(),
        }
