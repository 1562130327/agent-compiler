"""Self-evolving memory system — Nudge-driven, capacity-limited, snapshot-frozen.

Inspired by Hermes Agent's self-evolving memory architecture:
  Three-tier  — Episodic (session), Semantic (facts), Procedural (skills/rules)
  Nudge engine — Periodic background review extracts & distills, not every turn
  Capacity cap — Strict limits force compression; stale info naturally drops out
  Snapshot freeze — Memory loaded at session start, frozen in prompt for cache stability

Self-evolution cycle:
  User dialog → ... → [~N turns later] → Nudge triggers →
  Background agent reviews snapshot → Extracts worth-keeping facts →
  Writes to MEMORY.md / USER.md / SKILL.md → Next session loads updated files
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


class MemoryTier(Enum):
    """Three-tier memory architecture — maps to cognitive science."""
    EPISODIC = "episodic"       # Working memory: conversation summaries, time-decaying
    SEMANTIC = "semantic"       # Long-term facts: user profile, project info, preferences
    PROCEDURAL = "procedural"   # Skills: learned operational rules and workflows


# ── Capacity limits (Hermes-style: strict caps force compression) ──
MAX_MEMORY_CHARS = 2200       # MEMORY.md total char limit
MAX_USER_CHARS = 1375         # USER.md total char limit
MAX_PROCEDURAL_COUNT = 20     # Max distinct procedural rules

# Nudge intervals
NUDGE_MEMORY_INTERVAL = 10    # Review memory every N user interactions
NUDGE_SKILL_INTERVAL = 10     # Review skills every N tool-call iterations


@dataclass
class MemoryEntry:
    """A single memory record with metadata."""
    id: str
    category: str          # user_profile | project | pattern | feedback | knowledge
    tier: MemoryTier = MemoryTier.SEMANTIC
    title: str = ""
    content: str = ""
    keywords: list[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    embedding: np.ndarray | None = None
    source: str = ""
    version: int = 1
    decay_factor: float = 1.0
    merge_count: int = 0

    def touch(self):
        self.access_count += 1
        self.updated_at = time.time()

    def strengthen(self, amount: float = 0.1):
        self.confidence = min(1.0, self.confidence + amount)

    def weaken(self, amount: float = 0.1):
        self.confidence = max(0.0, self.confidence - amount)


class MemoryStore:
    """Disk-backed memory store with FAISS retrieval and Nudge-driven evolution.

    Key design decisions vs old system:
      - Does NOT write every interaction. Nudge engine batches reviews.
      - Enforces capacity limits; adding past limit requires replace/merge.
      - Freezes memories at session start for prompt cache stability.
      - Uses background agent (spawned by caller) for distillation, not sync extraction.
    """

    def __init__(self, memory_dir: str = "./agent_memory"):
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._memories: dict[str, MemoryEntry] = {}
        self._index: dict[str, int] = {}
        self._faiss = None
        self._next_id = 0
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._vocab_built = False
        self._stats = {"extractions": 0, "merges": 0, "prunes": 0, "consolidations": 0,
                       "nudge_reviews": 0, "nudge_skills": 0}

        # Nudge counters
        self._interaction_count = 0
        self._tool_call_count = 0
        self._nudge_callback: Callable | None = None

        # Snapshot freeze — memories loaded at init, frozen for prompt stability
        self._frozen_semantic_context: str = ""
        self._frozen_procedural_context: str = ""
        self._session_buffered: list[dict] = []  # [{role, content}] buffered this session

        # Load existing memories from disk
        self._load_all()
        self._freeze()

    # ── Nudge Engine ──────────────────────────────────────────────────

    def register_nudge_callback(self, cb: Callable):
        """Register an async callback for background memory review.
        Called with (review_type: str, snapshot: list[dict]) when nudge fires.
        """
        self._nudge_callback = cb

    def tick_interaction(self) -> bool:
        """Call after each user interaction. Returns True if nudge should fire."""
        self._interaction_count += 1
        if self._interaction_count % NUDGE_MEMORY_INTERVAL == 0:
            return True
        return False

    def tick_tool_call(self) -> bool:
        """Call after each tool-call iteration. Returns True if skill nudge should fire."""
        self._tool_call_count += 1
        if self._tool_call_count % NUDGE_SKILL_INTERVAL == 0:
            return True
        return False

    def buffer_interaction(self, user_input: str, assistant_reply: str):
        """Buffer a conversation turn for later review (NOT immediate write)."""
        self._session_buffered.append({
            "role": "user", "content": user_input[:2000],
            "timestamp": time.time(),
        })
        self._session_buffered.append({
            "role": "assistant", "content": assistant_reply[:2000],
            "timestamp": time.time(),
        })
        # Keep buffer bounded
        if len(self._session_buffered) > 100:
            self._session_buffered = self._session_buffered[-60:]

    def get_review_snapshot(self) -> list[dict]:
        """Return the buffered conversation for the background review agent."""
        return list(self._session_buffered)

    def clear_buffer(self):
        self._session_buffered.clear()

    def get_nudge_stats(self) -> dict:
        return {
            "interactions": self._interaction_count,
            "tool_calls": self._tool_call_count,
            "next_memory_nudge": NUDGE_MEMORY_INTERVAL - (self._interaction_count % NUDGE_MEMORY_INTERVAL),
            "next_skill_nudge": NUDGE_SKILL_INTERVAL - (self._tool_call_count % NUDGE_SKILL_INTERVAL),
        }

    # ── Snapshot Freeze ───────────────────────────────────────────────

    def _freeze(self):
        """Freeze current memories into prompt-stable context strings.
        Called once at init. Session writes land on disk but don't affect
        the frozen context until next session — protecting LLM prefix cache.
        """
        # Semantic context
        semantic = [m for m in self._memories.values()
                    if m.tier == MemoryTier.SEMANTIC and m.confidence >= 0.5]
        parts = []
        char_count = 0
        for m in sorted(semantic, key=lambda x: -x.updated_at):
            snippet = f"- {m.title}: {m.content[:200]}"
            if char_count + len(snippet) > MAX_MEMORY_CHARS:
                break
            parts.append(snippet)
            char_count += len(snippet)
        self._frozen_semantic_context = "\n".join(parts) if parts else ""

        # Procedural context
        proc = [m for m in self._memories.values()
                if m.tier == MemoryTier.PROCEDURAL and m.confidence >= 0.3]
        proc_parts = []
        for m in sorted(proc, key=lambda x: -x.confidence)[:MAX_PROCEDURAL_COUNT]:
            proc_parts.append(f"- {m.title}: {m.content[:200]}")
        self._frozen_procedural_context = "\n".join(proc_parts) if proc_parts else ""

    def get_frozen_context(self) -> str:
        """Get the frozen memory context for system prompt injection."""
        parts = []
        if self._frozen_semantic_context:
            parts.append(f"[知识]\n{self._frozen_semantic_context}")
        if self._frozen_procedural_context:
            parts.append(f"[经验]\n{self._frozen_procedural_context}")
        return "\n\n".join(parts) if parts else ""

    def refresh_frozen(self):
        """Re-freeze after disk changes (e.g. after background review writes)."""
        self._load_all()
        self._freeze()

    # ── CRUD ────────────────────────────────────────────────────────

    def add(self, entry: MemoryEntry, enforce_cap: bool = True) -> str | None:
        """Add a memory entry. Returns None if capacity cap blocks it."""
        if entry.id in self._memories:
            existing = self._memories[entry.id]
            existing.content = entry.content
            existing.keywords = entry.keywords
            existing.updated_at = time.time()
            existing.version += 1
            existing.strengthen()
            self._save_one(existing)
            return entry.id

        # Check capacity before adding
        if enforce_cap and entry.tier == MemoryTier.SEMANTIC:
            same_tier = [m for m in self._memories.values()
                        if m.tier == MemoryTier.SEMANTIC]
            total_chars = sum(len(m.content) for m in same_tier)
            if total_chars + len(entry.content) > MAX_MEMORY_CHARS:
                return None  # Caller must prune first

        self._memories[entry.id] = entry
        self._save_one(entry)
        self._add_to_index(entry)
        return entry.id

    def get(self, memory_id: str) -> MemoryEntry | None:
        entry = self._memories.get(memory_id)
        if entry:
            entry.touch()
        return entry

    def delete(self, memory_id: str):
        entry = self._memories.pop(memory_id, None)
        if entry:
            path = self._dir / f"{memory_id}.md"
            if path.exists():
                path.unlink()

    def replace(self, old_text: str, content: str, target: str = "memory") -> str | None:
        """Replace a substring in an existing memory (Hermes-style partial update)."""
        tier_filter = MemoryTier.SEMANTIC if target == "memory" else None
        candidates = [m for m in self._memories.values()
                     if (tier_filter is None or m.tier == tier_filter)
                     and old_text in m.content]
        if candidates:
            entry = candidates[0]
            entry.content = entry.content.replace(old_text, content)
            entry.updated_at = time.time()
            entry.version += 1
            self._save_one(entry)
            return entry.id
        return None

    def list_by_category(self, category: str) -> list[MemoryEntry]:
        return [m for m in self._memories.values() if m.category == category]

    def all(self) -> list[MemoryEntry]:
        return list(self._memories.values())

    # ── Retrieval ───────────────────────────────────────────────────

    def search(self, query: str, k: int = 5, min_confidence: float = 0.3,
               tiers: list[MemoryTier] | None = None) -> list[MemoryEntry]:
        """Find relevant memories using FAISS similarity."""
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
                    if entry.tier == MemoryTier.EPISODIC:
                        age_days = (time.time() - entry.created_at) / 86400
                        effective_score = score * max(0.3, entry.decay_factor - age_days * 0.02)
                        if effective_score < 0.15:
                            continue
                    results.append(entry)

        if len(results) < k:
            kw_results = self._keyword_search(query, k - len(results), min_confidence, tiers)
            existing_ids = {r.id for r in results}
            for r in kw_results:
                if r.id not in existing_ids:
                    results.append(r)

        return results[:k]

    def _keyword_search(self, query: str, k: int, min_confidence: float,
                        tiers: list[MemoryTier] | None = None) -> list[MemoryEntry]:
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        query_tokens.update(re.findall(r'[一-鿿]+', query))
        query_tokens.update(re.findall(r'[a-z]{3,}', query_lower))

        scored = []
        for m in self._memories.values():
            if m.confidence < min_confidence:
                continue
            if tiers and m.tier not in tiers:
                continue
            score = 0
            for kw in m.keywords:
                kw_lower = kw.lower()
                for qt in query_tokens:
                    if len(qt) >= 2 and (qt in kw_lower or kw_lower in qt):
                        score += 1
                        break
            title_lower = m.title.lower()
            for qt in query_tokens:
                if len(qt) >= 2 and qt in title_lower:
                    score += 2
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
        """Build context string of relevant memories for prompt injection.
        Uses frozen context as base, supplements with query-specific results.
        """
        relevant = self.search(query, k=5,
                              tiers=[MemoryTier.SEMANTIC, MemoryTier.EPISODIC])

        profile_mems = [m for m in self._memories.values()
                       if m.category == "user_profile" and m.confidence >= 0.5]
        profile_mems.sort(key=lambda m: -m.updated_at)
        seen_ids = {m.id for m in relevant}
        for pm in profile_mems[:3]:
            if pm.id not in seen_ids:
                relevant.append(pm)
                seen_ids.add(pm.id)

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
        """Store a conversation turn summary (not raw text) as episodic memory."""
        if not summary:
            summary = f"用户讨论了: {user_input[:150]}"

        mem_id = _make_id("episodic", summary[:120])

        for existing in self._memories.values():
            if existing.tier == MemoryTier.EPISODIC and _text_similarity(existing.content, summary) > 0.6:
                existing.content = summary if len(summary) > len(existing.content) else existing.content
                existing.updated_at = time.time()
                existing.merge_count += 1
                existing.decay_factor = min(1.0, existing.decay_factor + 0.1)
                self._save_one(existing)
                return existing

        entry = MemoryEntry(
            id=mem_id, category="conversation", tier=MemoryTier.EPISODIC,
            title=f"对话摘要: {summary[:60]}",
            content=summary[:500],
            keywords=_extract_chinese_keywords(summary),
            confidence=0.7, source="episodic", decay_factor=1.0,
        )
        self.add(entry, enforce_cap=False)
        self._compress_episodic(max_episodes=50)
        return entry

    def _compress_episodic(self, max_episodes: int = 50):
        episodes = [m for m in self._memories.values()
                    if m.tier == MemoryTier.EPISODIC]
        if len(episodes) <= max_episodes:
            return
        episodes.sort(key=lambda m: m.created_at)
        to_merge = episodes[:max(1, len(episodes) - max_episodes + 5)]
        if len(to_merge) >= 2:
            merged_content = "\n---\n".join(
                f"[{time.strftime('%m-%d', time.localtime(e.created_at))}] {e.content[:200]}"
                for e in to_merge
            )
            merged_title = f"压缩对话 {len(to_merge)}条"
            mem_id = _make_id("episodic_compressed", merged_title)
            entry = MemoryEntry(
                id=mem_id, category="conversation_summary", tier=MemoryTier.EPISODIC,
                title=merged_title[:80], content=merged_content[:2000],
                keywords=_extract_chinese_keywords(merged_content),
                confidence=0.5, source="auto-compress", decay_factor=0.5,
                merge_count=len(to_merge),
            )
            self.add(entry, enforce_cap=False)
            for e in to_merge:
                self.delete(e.id)

    # ── Procedural Memory ──────────────────────────────────────────

    def add_procedural(self, rule: str, context: str = "",
                       confidence: float = 0.6) -> MemoryEntry | None:
        mem_id = _make_id("procedural", rule[:120])
        for existing in self._memories.values():
            if existing.tier == MemoryTier.PROCEDURAL:
                if _text_similarity(existing.content, rule) > 0.5:
                    existing.strengthen(0.15)
                    existing.content = rule if len(rule) > len(existing.content) else existing.content
                    existing.updated_at = time.time()
                    existing.source = f"{existing.source}; reflexion"
                    self._save_one(existing)
                    return existing

        # Check procedural count cap
        proc_count = len([m for m in self._memories.values()
                         if m.tier == MemoryTier.PROCEDURAL])
        if proc_count >= MAX_PROCEDURAL_COUNT:
            # Replace lowest-confidence procedural
            lowest = min([m for m in self._memories.values()
                         if m.tier == MemoryTier.PROCEDURAL],
                        key=lambda m: m.confidence)
            self.delete(lowest.id)

        entry = MemoryEntry(
            id=mem_id, category="pattern", tier=MemoryTier.PROCEDURAL,
            title=f"操作规则: {rule[:60]}", content=rule,
            keywords=_extract_chinese_keywords(f"{rule} {context}"),
            confidence=confidence,
            source="reflexion" if "reflexion" in context else "learned",
        )
        return self.add(entry, enforce_cap=False)

    def get_procedural_rules(self, for_task: str = "") -> list[MemoryEntry]:
        rules = [m for m in self._memories.values()
                 if m.tier == MemoryTier.PROCEDURAL and m.confidence >= 0.3]
        if not for_task:
            return sorted(rules, key=lambda m: -m.confidence)
        return self.search(for_task, k=5, min_confidence=0.3,
                          tiers=[MemoryTier.PROCEDURAL])

    # ── Consolidation ───────────────────────────────────────────────

    def consolidate(self, llm_provider=None) -> int:
        """Merge similar memories, weaken stale ones, prune garbage.
        Call periodically (every N interactions) as a lightweight sync pass.
        This is the fast local pass; the heavy LLM distillation happens in
        the background nudge review.
        """
        self._stats["consolidations"] += 1
        changes = 0

        all_mems = self.all()
        for i, m1 in enumerate(all_mems):
            for m2 in all_mems[i + 1:]:
                if m1.category == m2.category and _text_similarity(m1.content, m2.content) > 0.7:
                    m1.content += f"\n\n[更新] {m2.content}"
                    m1.keywords = list(set(m1.keywords + m2.keywords))
                    m1.strengthen(0.2)
                    m1.version += 1
                    self._save_one(m1)
                    self.delete(m2.id)
                    self._stats["merges"] += 1
                    changes += 1

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

    # ── Nudge Review (LLM distillation) ─────────────────────────────

    def nudge_review(self, snapshot: list[dict]) -> int:
        """Called by background agent to review conversation and extract memories.
        The snapshot contains recent conversation turns. This method is the
        entry point for LLM-driven distillation — the caller passes in the
        LLM extraction result.
        """
        self._stats["nudge_reviews"] += 1
        return 0  # Placeholder — actual extraction happens via LLM in agent

    def nudge_skills(self, recent_tool_results: list[dict]) -> int:
        """Called by background agent to review tool-call patterns for skill creation."""
        self._stats["nudge_skills"] += 1
        return 0  # Placeholder

    def extract_facts(self, facts: list[dict]) -> list[MemoryEntry]:
        """Store facts extracted by LLM distillation. Each fact is a dict with:
        {category, tier, title, content, keywords, confidence}
        """
        self._stats["extractions"] += 1
        new_entries = []
        for fact in facts:
            cat = fact.get("category", "knowledge")
            tier_str = fact.get("tier", "semantic")
            try:
                tier = MemoryTier(tier_str)
            except ValueError:
                tier = MemoryTier.SEMANTIC

            if len(fact.get("content", "")) < 3:
                continue

            mem_id = _make_id(cat, fact.get("content", "")[:80])
            entry = MemoryEntry(
                id=mem_id, category=cat, tier=tier,
                title=fact.get("title", "")[:80],
                content=fact.get("content", "")[:800],
                keywords=fact.get("keywords", _extract_chinese_keywords(fact.get("content", ""))),
                confidence=float(fact.get("confidence", 0.8)),
                source="llm-distill",
            )
            result = self.add(entry)
            if result:
                new_entries.append(entry)

        if new_entries:
            self._freeze()  # Re-freeze after new facts
        return new_entries

    # ── Persistence ─────────────────────────────────────────────────

    def _save_one(self, entry: MemoryEntry):
        path = self._dir / f"{entry.id}.md"
        fm = {
            "id": entry.id, "category": entry.category,
            "tier": entry.tier.value, "title": entry.title,
            "keywords": entry.keywords, "confidence": entry.confidence,
            "created_at": entry.created_at, "updated_at": entry.updated_at,
            "access_count": entry.access_count, "source": entry.source,
            "version": entry.version, "decay_factor": entry.decay_factor,
            "merge_count": entry.merge_count,
        }
        body = f"---\n{yaml.dump(fm, allow_unicode=True)}---\n\n{entry.content}"
        path.write_text(body, encoding="utf-8")

    def _load_all(self):
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
                    id=fm.get("id", path.stem), category=fm.get("category", "knowledge"),
                    tier=tier, title=fm.get("title", path.stem),
                    content=content.strip(), keywords=fm.get("keywords", []),
                    confidence=fm.get("confidence", 1.0),
                    created_at=fm.get("created_at", time.time()),
                    updated_at=fm.get("updated_at", time.time()),
                    access_count=fm.get("access_count", 0),
                    source=fm.get("source", ""), version=fm.get("version", 1),
                    decay_factor=fm.get("decay_factor", 1.0),
                    merge_count=fm.get("merge_count", 0),
                )
                self._memories[entry.id] = entry
            except Exception:
                continue
        for entry in self._memories.values():
            self._add_to_index(entry)

    def _add_to_index(self, entry: MemoryEntry):
        if entry.id in self._index:
            return
        text = f"{entry.title} {entry.content}"[:500]
        vec = self._encode(text)
        import faiss
        if self._faiss is None:
            self._faiss = faiss.IndexFlatIP(len(vec))
        elif self._faiss.d != len(vec):
            self._faiss = faiss.IndexFlatIP(len(vec))
            self._index.clear()
        self._index[entry.id] = self._next_id
        self._next_id += 1
        self._faiss.add(vec.reshape(1, -1))
        entry.embedding = vec

    def _encode(self, text: str) -> np.ndarray:
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
        cats, tiers = {}, {}
        for m in self._memories.values():
            cats[m.category] = cats.get(m.category, 0) + 1
            tiers[m.tier.value] = tiers.get(m.tier.value, 0) + 1
        return {
            "total_memories": len(self._memories),
            "by_category": cats,
            "by_tier": tiers,
            "evolution": self._stats,
            "nudge": self.get_nudge_stats(),
            "capacity": {
                "semantic_chars": sum(len(m.content) for m in self._memories.values()
                                     if m.tier == MemoryTier.SEMANTIC),
                "max_semantic": MAX_MEMORY_CHARS,
                "procedural_count": len([m for m in self._memories.values()
                                        if m.tier == MemoryTier.PROCEDURAL]),
                "max_procedural": MAX_PROCEDURAL_COUNT,
            },
        }


# ── Helpers ──────────────────────────────────────────────────────────

def _make_id(category: str, text: str) -> str:
    import hashlib
    h = hashlib.sha256(f"{category}:{text}".encode()).hexdigest()[:12]
    return f"mem_{h}"


def _extract_chinese_keywords(text: str) -> list[str]:
    tokens = []
    for m in re.findall(r'[一-鿿]{2,}', text):
        tokens.append(m)
    for m in re.findall(r'[a-zA-Z]{3,}', text):
        tokens.append(m.lower())
    return list(set(tokens))[:10]


def _tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    tokens = []
    for i in range(len(text) - 1):
        tokens.append(text[i:i+2])
    for i in range(len(text) - 2):
        tokens.append(text[i:i+3])
    words = re.findall(r'[一-鿿]+|[a-z0-9]{2,}', text)
    tokens.extend(words)
    return tokens


def _text_similarity(a: str, b: str) -> float:
    a_set = set(_tokenize(a))
    b_set = set(_tokenize(b))
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)
