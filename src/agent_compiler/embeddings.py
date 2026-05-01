"""Backward compatibility — re-exports from agent_compiler.embeddings."""

from agent_compiler.embeddings.base import EmbeddingProvider

# Legacy name alias
EmbeddingStore = EmbeddingProvider

__all__ = ["EmbeddingProvider", "EmbeddingStore"]
