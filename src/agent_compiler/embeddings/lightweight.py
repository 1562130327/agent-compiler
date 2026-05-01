"""Lightweight embedding: TF-IDF with character n-grams.

Zero external downloads. Works fully offline. Good enough for
short-text intent matching in Chinese and English.
"""

from __future__ import annotations

import re

import numpy as np

from agent_compiler.embeddings.base import EmbeddingProvider


class LightweightEmbedding(EmbeddingProvider):
    """TF-IDF char n-gram embeddings. No model downloads needed."""

    _SEED_TEXTS = [
        "查看服务器状态", "检查错误日志并生成报告",
        "分析磁盘空间使用情况", "搜索日志中的关键词",
        "列出目录文件", "查看当前时间", "获取系统运行状态",
        "检查磁盘使用", "查找大文件", "生成汇总报告",
        "check server status", "search error logs and generate report",
        "analyze disk usage", "list files in directory", "get current time",
    ]

    def __init__(self, similarity_threshold: float = 0.50):
        super().__init__(similarity_threshold)
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}

    def encode(self, text: str) -> np.ndarray:
        if not self._vocab:
            self._build_vocab(list(self._SEED_TEXTS) + [text])

        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(len(self._vocab) or 1, dtype=np.float32)

        vec = np.zeros(len(self._vocab), dtype=np.float32)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        for token, count in tf.items():
            idx = self._vocab.get(token)
            if idx is not None:
                tf_val = count / len(tokens)
                idf_val = self._idf.get(token, 1.0)
                vec[idx] = tf_val * idf_val

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def _build_vocab(self, texts: list[str]):
        doc_freq: dict[str, int] = {}
        all_tokens: list[set[str]] = []
        for text in texts:
            tokens = set(self._tokenize(text))
            all_tokens.append(tokens)
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        sorted_tokens = sorted(doc_freq.items(), key=lambda x: -x[1])
        self._vocab = {t: i for i, (t, _) in enumerate(sorted_tokens)}
        n = len(texts)
        self._idf = {}
        for token, df in doc_freq.items():
            self._idf[token] = np.log((n + 1) / (df + 1)) + 1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        tokens = []
        for i in range(len(text) - 1):
            tokens.append(text[i:i+2])
        for i in range(len(text) - 2):
            tokens.append(text[i:i+3])
        for i in range(len(text) - 3):
            tokens.append(text[i:i+4])
        words = re.findall(r'[一-鿿]+|[a-z0-9]+', text)
        tokens.extend(words)
        return tokens
