"""Self-evolving memory system — persistent, auto-extracting, adaptive.

Inspired by Hermes Agent's self-evolving memory: learns from conversations,
automatically categorizes information, and refines knowledge over time.

Three-tier architecture (v0.3):
  Episodic   — conversation summaries, auto-compressed, time-decaying
  Semantic   — facts about user/project (names, preferences, tech stack)
  Procedural — learned operational rules ("when doing X, first do Y")

Self-evolution mechanisms:
  Extract   — LLM extracts key facts from each conversation
  Update    — new facts refine or supersede existing memories
  Merge     — similar memories are combined
  Prune     — outdated/contradicted info is archived
  Strengthen — confirmed info gets higher confidence
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class MemoryTier(Enum):
    """Three-tier memory architecture."""
    EPISODIC = "episodic"       # Recent conversation summaries
    SEMANTIC = "semantic"       # User/project facts
    PROCEDURAL = "procedural"   # Learned operational rules


@dataclass
class MemoryEntry:
    """A single memory record with metadata."""
    id: str
    category: str          # user_profile | project | pattern | feedback | knowledge
    tier: MemoryTier = MemoryTier.SEMANTIC  # episodic | semantic | procedural
    title: str = ""
    content: str = ""
    keywords: list[str] = field(default_factory=list)
    confidence: float = 1.0   # 0-1, increases with confirmation
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    embedding: np.ndarray | None = None
    source: str = ""        # which conversation/interaction created this
    version: int = 1
    decay_factor: float = 1.0  # episodic memories decay over time
    merge_count: int = 0       # how many memories merged into this one

    def touch(self):
        self.access_count += 1
        self.updated_at = time.time()

    def strengthen(self, amount: float = 0.1):
        self.confidence = min(1.0, self.confidence + amount)

    def weaken(self, amount: float = 0.1):
        self.confidence = max(0.0, self.confidence - amount)


class MemoryStore:
    """Disk-backed memory store with FAISS retrieval and self-evolution."""

    def __init__(self, memory_dir: str = "./agent_memory"):
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._memories: dict[str, MemoryEntry] = {}
        self._index: dict[str, int] = {}       # memory_id → FAISS index
        self._faiss = None
        self._next_id = 0
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._vocab_built = False
        self._stats = {"extractions": 0, "merges": 0, "prunes": 0, "consolidations": 0}

        # Load existing memories from disk
        self._load_all()

    # ── CRUD ────────────────────────────────────────────────────────

    def add(self, entry: MemoryEntry) -> str:
        """Add a new memory and index it for retrieval."""
        if entry.id in self._memories:
            # Update existing
            existing = self._memories[entry.id]
            existing.content = entry.content
            existing.keywords = entry.keywords
            existing.updated_at = time.time()
            existing.version += 1
            existing.strengthen()
            self._save_one(existing)
            return entry.id

        self._memories[entry.id] = entry
        self._save_one(entry)
        self._add_to_index(entry)
        return entry.id

    def get(self, memory_id: str) -> MemoryEntry | None:
        entry = self._memories.get(memory_id)
        if entry:
            entry.touch()
            self._save_one(entry)
        return entry

    def delete(self, memory_id: str):
        entry = self._memories.pop(memory_id, None)
        if entry:
            path = self._dir / f"{memory_id}.md"
            if path.exists():
                path.unlink()

    def list_by_category(self, category: str) -> list[MemoryEntry]:
        return [m for m in self._memories.values() if m.category == category]

    def all(self) -> list[MemoryEntry]:
        return list(self._memories.values())

    # ── Retrieval ───────────────────────────────────────────────────

    def search(self, query: str, k: int = 5, min_confidence: float = 0.3,
               tiers: list[MemoryTier] | None = None) -> list[MemoryEntry]:
        """Find relevant memories for a query using FAISS similarity.

        Args:
            query: search query
            k: max results
            min_confidence: minimum confidence threshold
            tiers: optional tier filter (e.g. [MemoryTier.SEMANTIC, MemoryTier.PROCEDURAL])
        """
        if not self._memories or self._faiss is None:
            return self._keyword_search(query, k, min_confidence, tiers)

        q_vec = self._encode(query).reshape(1, -1)
        import faiss
        scores, indices = self._faiss.search(q_vec, min(k * 2, self._faiss.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < 0.2:
                continue
            mem_id = self._index.get(int(idx))
            if mem_id:
                entry = self._memories.get(mem_id)
                if entry and entry.confidence >= min_confidence:
                    if tiers and entry.tier not in tiers:
                        continue
                    entry.touch()
                    # Apply decay factor for episodic memories
                    if entry.tier == MemoryTier.EPISODIC:
                        age_days = (time.time() - entry.created_at) / 86400
                        effective_score = score * max(0.3, entry.decay_factor - age_days * 0.02)
                        if effective_score < 0.15:
                            continue
                    results.append(entry)

        # Supplement with keyword search
        if len(results) < k:
            kw_results = self._keyword_search(query, k - len(results), min_confidence, tiers)
            existing_ids = {r.id for r in results}
            for r in kw_results:
                if r.id not in existing_ids:
                    results.append(r)

        return results[:k]

    def _keyword_search(self, query: str, k: int, min_confidence: float,
                        tiers: list[MemoryTier] | None = None) -> list[MemoryEntry]:
        """Fallback keyword-based search with cross-language matching."""
        query_lower = query.lower()
        # Expand query with cross-language hints
        query_tokens = set(query_lower.split())
        query_tokens.update(re.findall(r'[一-鿿]+', query))  # Chinese chars
        query_tokens.update(re.findall(r'[a-z]{3,}', query_lower))   # English words

        scored = []
        for m in self._memories.values():
            if m.confidence < min_confidence:
                continue
            if tiers and m.tier not in tiers:
                continue
            score = 0
            # Match keywords to query tokens
            for kw in m.keywords:
                kw_lower = kw.lower()
                for qt in query_tokens:
                    if len(qt) >= 2 and (qt in kw_lower or kw_lower in qt):
                        score += 1
                        break
            # Match title/query
            title_lower = m.title.lower()
            for qt in query_tokens:
                if len(qt) >= 2 and qt in title_lower:
                    score += 2
            # Content match
            content_lower = m.content.lower()
            for qt in query_tokens:
                if len(qt) >= 3 and qt in content_lower:
                    score += 1
            if score > 0:
                scored.append((m, score))
        scored.sort(key=lambda x: -x[1])
        return [m for m, _ in scored[:k]]

    def context_for_query(self, query: str, max_tokens: int = 2000,
                          include_procedural: bool = True) -> str:
        """Build a context string of relevant memories for injection into prompt.

        Always includes recent user_profile memories (name, role, preferences)
        since they rarely share keywords with queries like "我叫什么名字".
        Optionally includes relevant procedural rules for the task.
        """
        relevant = self.search(query, k=5,
                              tiers=[MemoryTier.SEMANTIC, MemoryTier.EPISODIC])

        # Always include top user_profile memories regardless of search match
        profile_mems = [m for m in self._memories.values()
                       if m.category == "user_profile" and m.confidence >= 0.5]
        profile_mems.sort(key=lambda m: -m.updated_at)
        seen_ids = {m.id for m in relevant}
        for pm in profile_mems[:3]:
            if pm.id not in seen_ids:
                relevant.append(pm)
                seen_ids.add(pm.id)

        # Include procedural rules for this task type
        if include_procedural:
            proc_rules = self.get_procedural_rules(for_task=query)
            for pr in proc_rules[:3]:
                if pr.id not in seen_ids:
                    relevant.append(pr)
                    seen_ids.add(pr.id)

        if not relevant:
            return ""

        parts = ["[相关记忆]"]
        char_count = 0
        for m in relevant:
            tier_label = {MemoryTier.EPISODIC: "[对话] ",
                         MemoryTier.PROCEDURAL: "[经验] ",
                         MemoryTier.SEMANTIC: ""}.get(m.tier, "")
            snippet = f"### {tier_label}{m.title}\n{m.content[:300]}"
            if char_count + len(snippet) > max_tokens * 2:
                break
            parts.append(snippet)
            char_count += len(snippet)

        return "\n\n".join(parts) if len(parts) > 1 else ""

    # ── Episodic Memory ────────────────────────────────────────────

    def add_episodic(self, user_input: str, assistant_reply: str,
                     summary: str = "") -> MemoryEntry | None:
        """Store a conversation turn as an episodic memory.

        Episodic memories auto-decay and are compressed when too many accumulate.
        """
        if not summary:
            summary = f"用户: {user_input[:200]}\n助手: {assistant_reply[:200]}"

        mem_id = _make_id("episodic", f"{user_input[:80]}:{assistant_reply[:80]}")

        # Check if similar episodic memory exists — merge instead of duplicate
        for existing in self._memories.values():
            if existing.tier == MemoryTier.EPISODIC and _text_similarity(existing.content, summary) > 0.6:
                existing.content = summary if len(summary) > len(existing.content) else existing.content
                existing.updated_at = time.time()
                existing.merge_count += 1
                existing.decay_factor = min(1.0, existing.decay_factor + 0.1)
                self._save_one(existing)
                return existing

        entry = MemoryEntry(
            id=mem_id,
            category="conversation",
            tier=MemoryTier.EPISODIC,
            title=f"对话: {user_input[:60]}",
            content=summary,
            keywords=_extract_chinese_keywords(f"{user_input} {assistant_reply}"),
            confidence=0.7,
            source="episodic",
            decay_factor=1.0,
        )
        self.add(entry)

        # Auto-compress if too many episodic memories
        self._compress_episodic(max_episodes=50)
        return entry

    def _compress_episodic(self, max_episodes: int = 50):
        """Compress old episodic memories when exceeding the limit."""
        episodes = [m for m in self._memories.values()
                    if m.tier == MemoryTier.EPISODIC]
        if len(episodes) <= max_episodes:
            return

        # Sort by age (oldest first), merge oldest 20% into a summary
        episodes.sort(key=lambda m: m.created_at)
        to_merge = episodes[:max(1, len(episodes) - max_episodes + 5)]

        if len(to_merge) >= 2:
            merged_content = "\n---\n".join(
                f"[{time.strftime('%m-%d', time.localtime(e.created_at))}] {e.content[:200]}"
                for e in to_merge
            )
            merged_title = f"压缩对话 {len(to_merge)}条: {to_merge[0].created_at}-{to_merge[-1].created_at}"
            mem_id = _make_id("episodic_compressed", merged_title)

            entry = MemoryEntry(
                id=mem_id,
                category="conversation_summary",
                tier=MemoryTier.EPISODIC,
                title=merged_title[:80],
                content=merged_content[:2000],
                keywords=_extract_chinese_keywords(merged_content),
                confidence=0.5,
                source="auto-compress",
                decay_factor=0.5,
                merge_count=len(to_merge),
            )
            self.add(entry)

            # Delete the merged individual episodes
            for e in to_merge:
                self.delete(e.id)

    # ── Procedural Memory ──────────────────────────────────────────

    def add_procedural(self, rule: str, context: str = "",
                       confidence: float = 0.6) -> MemoryEntry:
        """Store a learned operational rule.

        Called from reflexion and pattern learning to remember
        what approaches work best.
        """
        mem_id = _make_id("procedural", rule[:120])

        # Update existing similar rule
        for existing in self._memories.values():
            if existing.tier == MemoryTier.PROCEDURAL:
                if _text_similarity(existing.content, rule) > 0.5:
                    existing.strengthen(0.15)
                    existing.content = rule if len(rule) > len(existing.content) else existing.content
                    existing.updated_at = time.time()
                    existing.source = f"{existing.source}; reflexion"
                    self._save_one(existing)
                    return existing

        entry = MemoryEntry(
            id=mem_id,
            category="pattern",
            tier=MemoryTier.PROCEDURAL,
            title=f"操作规则: {rule[:60]}",
            content=rule,
            keywords=_extract_chinese_keywords(f"{rule} {context}"),
            confidence=confidence,
            source="reflexion" if "reflexion" in context else "learned",
        )
        self.add(entry)
        return entry

    def get_procedural_rules(self, for_task: str = "") -> list[MemoryEntry]:
        """Get relevant procedural rules, optionally filtered by task."""
        rules = [m for m in self._memories.values()
                 if m.tier == MemoryTier.PROCEDURAL and m.confidence >= 0.3]
        if not for_task:
            return sorted(rules, key=lambda m: -m.confidence)
        return self.search(for_task, k=5, min_confidence=0.3,
                          tiers=[MemoryTier.PROCEDURAL])

    # ── Self-Evolution ──────────────────────────────────────────────

    def extract_from_interaction(self, user_input: str, assistant_reply: str,
                                  llm_provider=None) -> list[MemoryEntry]:
        """Extract memory-worthy facts from a conversation turn.

        Uses LLM (when available) to identify key facts worth remembering,
        then automatically categorizes and stores them. Falls back to
        heuristic extraction when no LLM is available.
        """
        self._stats["extractions"] += 1
        new_entries: list[MemoryEntry] = []

        # ── LLM-based extraction (preferred) ─────────────────
        if llm_provider is not None and hasattr(llm_provider, 'extract_memories'):
            try:
                facts = llm_provider.extract_memories(user_input, assistant_reply)
                for fact in facts:
                    cat = fact.get("category", "knowledge")
                    title = fact.get("title", fact.get("content", "")[:40])
                    content = fact.get("content", "")
                    if len(content) < 3:
                        continue
                    # Map category to tier
                    tier = MemoryTier.SEMANTIC
                    if cat in ("pattern", "feedback"):
                        tier = MemoryTier.PROCEDURAL

                    mem_id = _make_id(cat, content[:80])
                    entry = MemoryEntry(
                        id=mem_id,
                        category=cat,
                        tier=tier,
                        title=title[:80],
                        content=content[:800],
                        keywords=fact.get("keywords", _extract_chinese_keywords(content)),
                        confidence=float(fact.get("confidence", 0.8)),
                        source="llm-extract",
                    )
                    self.add(entry)
                    new_entries.append(entry)
                if new_entries:
                    return new_entries
            except Exception:
                pass  # fall through to heuristic

        # ── Heuristic fallback ──────────────────────────────
        inp = user_input.strip()

        # Only extract explicit "remember" directives heuristically
        remember_patterns = [
            (r'(?:记住[：:]\s*)(.{4,200}?)(?:[。！？\n]|$)',
             "knowledge", "用户告知"),
            (r'(?:我叫|我是|我的名字是|我的职业是|我的工作是)\s*(.{3,100}?)(?:[。！？\n，,\.]|$)',
             "user_profile", "用户信息"),
        ]

        for pattern, cat, desc in remember_patterns:
            matches = re.findall(pattern, inp)
            for m_text in matches:
                m_text = m_text.strip()
                if len(m_text) >= 2:
                    mem_id = _make_id(cat, m_text[:80])
                    entry = MemoryEntry(
                        id=mem_id,
                        category=cat,
                        title=f"{desc}: {m_text[:60]}",
                        content=m_text[:500],
                        keywords=_extract_chinese_keywords(m_text),
                        confidence=0.7,
                        source="heuristic-extract",
                    )
                    self.add(entry)
                    new_entries.append(entry)

        return new_entries

    def consolidate(self, llm_provider=None) -> int:
        """Review and consolidate memories — merge similar, weaken old, prune stale.

        Called periodically (every N interactions) to self-evolve the memory.
        Returns number of changes made.
        """
        self._stats["consolidations"] += 1
        changes = 0

        # Merge very similar memories
        all_mems = self.all()
        for i, m1 in enumerate(all_mems):
            for m2 in all_mems[i + 1:]:
                if m1.category == m2.category and _text_similarity(m1.content, m2.content) > 0.7:
                    # Merge: m2 into m1
                    m1.content += f"\n\n[更新] {m2.content}"
                    m1.keywords = list(set(m1.keywords + m2.keywords))
                    m1.strengthen(0.2)
                    m1.version += 1
                    self._save_one(m1)
                    self.delete(m2.id)
                    self._stats["merges"] += 1
                    changes += 1

        # Weaken old, low-confidence memories
        now = time.time()
        for m in all_mems:
            age_days = (now - m.created_at) / 86400
            if age_days > 30 and m.confidence < 0.5 and m.access_count < 2:
                m.weaken(0.3)
                if m.confidence <= 0.1:
                    self.delete(m.id)
                    self._stats["prunes"] += 1
                    changes += 1
                else:
                    self._save_one(m)

        return changes

    def inject_memories_into_context(self, query: str, messages: list[dict]) -> list[dict]:
        """Prepend relevant memory context to the conversation messages."""
        ctx = self.context_for_query(query)
        if not ctx:
            return messages

        memory_msg = {
            "role": "system",
            "content": f"## 你学到的关于这个用户/项目的知识\n{ctx}\n\n"
                      f"使用这些知识来个性化你的回复。如有冲突，以用户最新表述为准。",
        }

        # Insert after the first system message, or at the beginning
        for i, m in enumerate(messages):
            if m.get("role") == "system":
                messages.insert(i + 1, memory_msg)
                return messages
        messages.insert(0, memory_msg)
        return messages

    # ── Persistence ─────────────────────────────────────────────────

    def _save_one(self, entry: MemoryEntry):
        """Save a single memory entry to a markdown file."""
        path = self._dir / f"{entry.id}.md"
        fm = {
            "id": entry.id,
            "category": entry.category,
            "tier": entry.tier.value,
            "title": entry.title,
            "keywords": entry.keywords,
            "confidence": entry.confidence,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "access_count": entry.access_count,
            "source": entry.source,
            "version": entry.version,
            "decay_factor": entry.decay_factor,
            "merge_count": entry.merge_count,
        }
        body = f"---\n{yaml.dump(fm, allow_unicode=True)}---\n\n{entry.content}"
        path.write_text(body, encoding="utf-8")

    def _load_all(self):
        """Load all memory entries from disk."""
        for path in sorted(self._dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                if text.startswith("---"):
                    _, fm_str, content = text.split("---", 2)
                    fm = yaml.safe_load(fm_str) or {}
                else:
                    fm, content = {}, text

                tier_str = fm.get("tier", "semantic")
                try:
                    tier = MemoryTier(tier_str)
                except ValueError:
                    tier = MemoryTier.SEMANTIC

                entry = MemoryEntry(
                    id=fm.get("id", path.stem),
                    category=fm.get("category", "knowledge"),
                    tier=tier,
                    title=fm.get("title", path.stem),
                    content=content.strip(),
                    keywords=fm.get("keywords", []),
                    confidence=fm.get("confidence", 1.0),
                    created_at=fm.get("created_at", time.time()),
                    updated_at=fm.get("updated_at", time.time()),
                    access_count=fm.get("access_count", 0),
                    source=fm.get("source", ""),
                    version=fm.get("version", 1),
                    decay_factor=fm.get("decay_factor", 1.0),
                    merge_count=fm.get("merge_count", 0),
                )
                self._memories[entry.id] = entry
            except Exception:
                continue

        # Build FAISS index
        for entry in self._memories.values():
            self._add_to_index(entry)

    def _add_to_index(self, entry: MemoryEntry):
        """Add a memory entry to the FAISS index."""
        if entry.id in self._index:
            return  # already indexed

        text = f"{entry.title} {entry.content}"[:500]
        vec = self._encode(text)

        import faiss
        if self._faiss is None:
            self._faiss = faiss.IndexFlatIP(len(vec))
        elif self._faiss.d != len(vec):
            # Dimension mismatch — rebuild index
            self._faiss = faiss.IndexFlatIP(len(vec))
            self._index.clear()

        self._index[entry.id] = self._next_id
        self._next_id += 1
        self._faiss.add(vec.reshape(1, -1))
        entry.embedding = vec

    def _encode(self, text: str) -> np.ndarray:
        """Simple TF-IDF encoding for memory retrieval."""
        tokens = _tokenize(text)
        if not tokens:
            return np.zeros(len(self._vocab) or 128, dtype=np.float32)

        if not self._vocab_built:
            self._build_vocab([text])
            self._vocab_built = True

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
        for text in texts:
            for t in set(_tokenize(text)):
                doc_freq[t] = doc_freq.get(t, 0) + 1
        sorted_tokens = sorted(doc_freq.items(), key=lambda x: -x[1])
        self._vocab = {t: i for i, (t, _) in enumerate(sorted_tokens)}
        n = len(texts)
        self._idf = {}
        for token, df in doc_freq.items():
            self._idf[token] = np.log((n + 1) / (df + 1)) + 1

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        cats = {}
        tiers = {}
        for m in self._memories.values():
            cats[m.category] = cats.get(m.category, 0) + 1
            tiers[m.tier.value] = tiers.get(m.tier.value, 0) + 1
        return {
            "total_memories": len(self._memories),
            "by_category": cats,
            "by_tier": tiers,
            "evolution": self._stats,
        }


# ── Helpers ──────────────────────────────────────────────────────────

def _make_id(category: str, text: str) -> str:
    """Generate a stable memory ID from category + content."""
    import hashlib
    h = hashlib.sha256(f"{category}:{text}".encode()).hexdigest()[:12]
    return f"mem_{h}"


def _extract_chinese_keywords(text: str) -> list[str]:
    """Extract Chinese keywords from text."""
    import re
    tokens = []
    # Chinese character sequences (2+ chars)
    for m in re.findall(r'[一-鿿]{2,}', text):
        tokens.append(m)
    # English words (3+ chars)
    for m in re.findall(r'[a-zA-Z]{3,}', text):
        tokens.append(m.lower())
    return list(set(tokens))[:10]


def _tokenize(text: str) -> list[str]:
    """Tokenize text for TF-IDF encoding."""
    text = text.lower().strip()
    tokens = []
    # Character n-grams
    for i in range(len(text) - 1):
        tokens.append(text[i:i+2])
    for i in range(len(text) - 2):
        tokens.append(text[i:i+3])
    # Words
    words = re.findall(r'[一-鿿]+|[a-z0-9]{2,}', text)
    tokens.extend(words)
    return tokens


def _text_similarity(a: str, b: str) -> float:
    """Simple character-level similarity for memory dedup."""
    a_set = set(_tokenize(a))
    b_set = set(_tokenize(b))
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)
