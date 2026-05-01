"""Abstract base classes for the embedding module.

To add a new embedding backend, subclass EmbeddingProvider and implement:
  - encode(text) -> np.ndarray
  - dim property
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from agent_compiler.core.types import WorkflowTemplate


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface.

    Implementations:
      - LightweightEmbedding: TF-IDF char n-grams, zero downloads
      - NeuralEmbedding: sentence-transformers, higher accuracy
    """

    def __init__(self, similarity_threshold: float = 0.50):
        self.similarity_threshold = similarity_threshold
        self._faiss = None
        self._index: dict[int, str] = {}
        self._next_id = 0

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Generate a normalized embedding vector for a text."""

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts], dtype=np.float32)

    def add(self, wf: WorkflowTemplate):
        """Add a workflow template's embedding to the FAISS index."""
        if wf.embedding is None:
            wf.embedding = self.encode(wf.intent)
        emb = wf.embedding.reshape(1, -1)

        import faiss
        if self._faiss is None:
            self._faiss = faiss.IndexFlatIP(emb.shape[1])

        self._index[self._next_id] = wf.id
        self._next_id += 1
        self._faiss.add(emb)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Search for similar workflow templates. Returns [(workflow_id, score), ...]."""
        if self._faiss is None or self._faiss.ntotal == 0:
            return []

        import faiss
        q_vec = self.encode(query).reshape(1, -1)
        scores, indices = self._faiss.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < self.similarity_threshold:
                continue
            wf_id = self._index.get(int(idx))
            if wf_id:
                results.append((wf_id, float(score)))
        return results

    def save(self, path: str):
        """Persist FAISS index and id mapping to disk."""
        import faiss
        import json
        idx_path = Path(path)
        idx_path.mkdir(parents=True, exist_ok=True)
        if self._faiss is not None:
            faiss.write_index(self._faiss, str(idx_path / "faiss.index"))
        mapping = {"next_id": self._next_id, "index": self._index,
                   "threshold": self.similarity_threshold}
        (idx_path / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False))

    def load(self, path: str):
        """Load FAISS index and id mapping from disk."""
        import faiss
        import json
        idx_path = Path(path)
        faiss_file = idx_path / "faiss.index"
        mapping_file = idx_path / "mapping.json"
        if faiss_file.exists():
            self._faiss = faiss.read_index(str(faiss_file))
        if mapping_file.exists():
            mapping = json.loads(mapping_file.read_text())
            self._next_id = mapping["next_id"]
            self._index = {int(k): v for k, v in mapping["index"].items()}
            self.similarity_threshold = mapping.get("threshold", self.similarity_threshold)

    def stats(self) -> dict:
        total = self._faiss.ntotal if self._faiss else 0
        return {"total_vectors": total, "threshold": self.similarity_threshold}
