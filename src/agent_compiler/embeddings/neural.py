"""Neural embedding: sentence-transformers model.

Higher accuracy than lightweight mode, but requires ~120MB model download
on first use. Only import this module if you need neural embeddings.
"""

from __future__ import annotations

import numpy as np

from agent_compiler.embeddings.base import EmbeddingProvider


class NeuralEmbedding(EmbeddingProvider):
    """Sentence-transformer embeddings for higher accuracy.

    Requires network access for first download (~120MB).
    Model: paraphrase-multilingual-MiniLM-L12-v2
    """

    def __init__(self,
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 similarity_threshold: float = 0.70):
        super().__init__(similarity_threshold)
        self.model_name = model_name
        self._model = None

    def encode(self, text: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)
