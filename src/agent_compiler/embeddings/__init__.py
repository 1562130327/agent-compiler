"""Embedding module — pluggable embedding providers.

Built-in providers:
  - LightweightEmbedding: TF-IDF char n-grams, zero downloads, works offline
  - NeuralEmbedding: sentence-transformers model, higher accuracy

To add a custom provider, subclass EmbeddingProvider and implement encode().
"""

from agent_compiler.embeddings.base import EmbeddingProvider
from agent_compiler.embeddings.lightweight import LightweightEmbedding
from agent_compiler.embeddings.neural import NeuralEmbedding

# Legacy alias
EmbeddingStore = EmbeddingProvider

__all__ = ["EmbeddingProvider", "EmbeddingStore",
           "LightweightEmbedding", "NeuralEmbedding"]
