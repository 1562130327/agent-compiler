"""Context window management — token-aware, smart compression.

Replaces the naive FIFO message truncation with:
  - Token counting (rough estimation)
  - Priority-based pruning (system > recent > memory > old history)
  - Automatic summarization of old messages
  - Token budget enforcement
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ContextBudget:
    """Token budget for context window management."""
    max_tokens: int = 100_000       # total context window size
    system_reserve: int = 3_000     # reserve for system prompt
    memory_reserve: int = 2_000     # reserve for memory injection
    recent_reserve: int = 8_000     # reserve for recent messages (last N turns)
    tool_output_max: int = 4_000    # max tokens per tool output
    summary_target: int = 500       # target tokens for summarization


class ContextManager:
    """Manages message context with token-aware pruning.

    Usage:
        mgr = ContextManager(budget=ContextBudget(max_tokens=80000))
        mgr.add_message({"role": "user", "content": "hello"})
        ...
        trimmed = mgr.build_context()  # returns pruned message list
    """

    # Rough token estimation: ~1.3 tokens per Chinese char, ~0.75 per English word
    _CHINESE_RE = re.compile(r'[一-鿿㐀-䶿]')
    _WORD_RE = re.compile(r'[a-zA-Z0-9]+')

    def __init__(self, budget: ContextBudget | None = None):
        self.budget = budget or ContextBudget()
        self._messages: list[dict] = []
        self._summary: str = ""  # compressed summary of pruned messages

    def add_message(self, msg: dict):
        """Add a message to the context."""
        self._messages.append(msg)

    def add_messages(self, msgs: list[dict]):
        self._messages.extend(msgs)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for a text string."""
        if not text:
            return 0
        chinese_chars = len(ContextManager._CHINESE_RE.findall(text))
        english_words = len(ContextManager._WORD_RE.findall(text))
        other_chars = len(text) - chinese_chars - sum(
            len(w) for w in ContextManager._WORD_RE.findall(text)
        )
        # Chinese: ~1.3 tokens/char, English: ~1.3 tokens/word, other: ~0.5 tokens/char
        return int(chinese_chars * 1.3 + english_words * 1.3 + other_chars * 0.5)

    @staticmethod
    def estimate_message_tokens(msg: dict) -> int:
        """Estimate tokens for a single message."""
        tokens = ContextManager.estimate_tokens(msg.get("content", "") or "")
        # Tool calls in assistant messages
        if msg.get("tool_calls"):
            import json
            tokens += ContextManager.estimate_tokens(
                json.dumps(msg["tool_calls"], ensure_ascii=False))
        return tokens

    def total_tokens(self) -> int:
        return sum(self.estimate_message_tokens(m) for m in self._messages)

    def build_context(self, system_prompt: str = "",
                      memory_context: str = "") -> list[dict]:
        """Build a context list that fits within the token budget.

        Strategy:
          1. System prompt always included (up to system_reserve)
          2. Memory context included (up to memory_reserve)
          3. Most recent messages included (up to recent_reserve)
          4. Older messages compressed into summary
          5. Tool outputs truncated to tool_output_max
        """
        budget = self.budget
        result: list[dict] = []
        remaining = budget.max_tokens

        # 1. System prompt
        sys_tokens = self.estimate_tokens(system_prompt)
        if sys_tokens > budget.system_reserve:
            system_prompt = self._truncate_text(system_prompt, budget.system_reserve)
        result.append({"role": "system", "content": system_prompt})
        remaining -= min(sys_tokens, budget.system_reserve)

        # 2. Memory context
        if memory_context:
            mem_tokens = self.estimate_tokens(memory_context)
            if mem_tokens > budget.memory_reserve:
                memory_context = self._truncate_text(memory_context, budget.memory_reserve)
            result.append({"role": "system", "content": memory_context})
            remaining -= min(mem_tokens, budget.memory_reserve)

        # 3. Prioritize recent messages
        msgs = list(self._messages)
        recent = msgs[-6:]  # last 3 turns (user+assistant pairs)
        older = msgs[:-6]

        # Add older messages as summary if they exist
        if older:
            if not self._summary:
                self._summary = self._summarize(older)
            if self._summary:
                summary_tokens = min(self.estimate_tokens(self._summary),
                                    budget.summary_target * 2)
                result.append({
                    "role": "system",
                    "content": f"[早前对话摘要]\n{self._summary[:budget.summary_target * 4]}",
                })
                remaining -= summary_tokens

        # 4. Add recent messages (with tool output truncation)
        for msg in recent:
            tokens = self.estimate_message_tokens(msg)
            if tokens > budget.tool_output_max and msg.get("role") in ("tool_result", "tool"):
                content = msg.get("content", "")
                msg = dict(msg)
                msg["content"] = self._truncate_text(content, budget.tool_output_max)

            remaining -= min(tokens, budget.tool_output_max)
            if remaining < 0:
                break
            result.append(msg)

        return result

    def _summarize(self, messages: list[dict]) -> str:
        """Create a brief summary of older messages."""
        parts = []
        for m in messages[:20]:
            role = m.get("role", "?")
            content = (m.get("content") or "")[:200]
            if content.strip():
                parts.append(f"[{role}]: {content.strip()}")
        return "\n".join(parts)

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to roughly max_tokens."""
        if self.estimate_tokens(text) <= max_tokens:
            return text
        # Binary search for cutoff point
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.estimate_tokens(text[:mid]) < max_tokens:
                lo = mid + 1
            else:
                hi = mid
        return text[:lo] + "\n... [已截断]"

    def reset(self):
        self._messages.clear()
        self._summary = ""
