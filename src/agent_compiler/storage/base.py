"""Storage abstract base class — defines the interface for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_compiler.core.types import WorkflowTemplate


class Store(ABC):
    """Abstract storage backend for workflow templates.

    Implementations: RamStore (L1 hot), DiskStore (L2 warm), CloudStore (L3 cold)
    """

    @abstractmethod
    def get(self, workflow_id: str) -> WorkflowTemplate | None: ...

    @abstractmethod
    def put(self, wf: WorkflowTemplate): ...

    @abstractmethod
    def delete(self, workflow_id: str): ...

    def stats(self) -> dict:
        return {}
