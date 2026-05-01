"""Storage layer: RAM (hot) → Disk/SSD (warm) → Cloud (cold)."""

import time
import threading
from collections import OrderedDict
from typing import Generator

import numpy as np

from agent_compiler.core.types import WorkflowTemplate


class RamStore:
    """L1: In-memory LRU cache for hot workflow templates."""

    def __init__(self, max_entries: int = 1000, max_memory_mb: float = 200):
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self._cache: OrderedDict[str, WorkflowTemplate] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, workflow_id: str) -> WorkflowTemplate | None:
        with self._lock:
            wf = self._cache.get(workflow_id)
            if wf:
                self._cache.move_to_end(workflow_id)
                wf.bump()
            return wf

    def put(self, wf: WorkflowTemplate):
        with self._lock:
            if wf.id in self._cache:
                self._cache.move_to_end(wf.id)
                self._cache[wf.id] = wf
                return
            self._cache[wf.id] = wf
            self._evict_if_needed()

    def _evict_if_needed(self):
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)  # FIFO eviction of coldest

    def top_by_hits(self, n: int = 10) -> list[WorkflowTemplate]:
        return sorted(self._cache.values(), key=lambda w: w.hit_count, reverse=True)[:n]

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._cache), "max_entries": self.max_entries}
