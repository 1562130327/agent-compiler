"""Tree-of-Thought (ToT) — deep reasoning via multi-branch exploration.

Inspired by Yao et al. (2023): "Tree of Thoughts: Deliberate Problem Solving
with Large Language Models" — Game of 24: CoT 4% → ToT 74% (18× improvement).

Generates multiple reasoning branches, evaluates each, backtracks to the best
path. Only activates for complex tasks detected by keyword matching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ThoughtNode:
    """A single node in the thought tree."""
    id: str
    content: str                    # the thought/plan/approach text
    score: float = 0.0              # evaluation score (0-10)
    depth: int = 0
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    is_terminal: bool = False       # True if this is a final solution
    eval_feedback: str = ""         # evaluation feedback text


class ThoughtTree:
    """Manages a tree of thoughts for deep reasoning.

    Usage:
        tree = ThoughtTree(llm_provider, beam_width=2, max_depth=3)
        best_path = tree.solve("如何重构这个500行的函数?")
        # Returns list of ThoughtNode from root to best leaf
    """

    def __init__(self, llm_provider=None, beam_width: int = 2,
                 max_depth: int = 3, num_candidates: int = 4,
                 max_per_expand: int = 3):
        self.llm = llm_provider
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.num_candidates = num_candidates
        self.max_per_expand = max_per_expand
        self.nodes: dict[str, ThoughtNode] = {}
        self._id_counter = 0

    def solve(self, problem: str, context: str = "") -> list[ThoughtNode]:
        """Run Tree-of-Thought reasoning on a problem.

        Returns the best path as a list of ThoughtNode (root → leaf).
        """
        t0 = time.perf_counter()

        # ── Phase 1: Generate initial candidates ──
        root = self._new_node(
            content=f"问题: {problem}",
            depth=0,
        )
        self.nodes[root.id] = root

        candidates = self._generate_thoughts(
            problem, context, num=self.num_candidates,
            parent=None,
        )
        if not candidates:
            return [root]

        # Evaluate initial candidates
        scored = self._evaluate_thoughts(problem, candidates, context)

        # ── Phase 2: Beam search through thought space ──
        active = scored[:self.beam_width]  # top-K nodes
        for node in active:
            node.parent_id = root.id
            root.children_ids.append(node.id)
            self.nodes[node.id] = node

        for depth in range(1, self.max_depth):
            next_candidates: list[ThoughtNode] = []
            for node in active:
                children = self._generate_thoughts(
                    problem, context,
                    parent=node,
                    num=self.max_per_expand,
                )
                for child in children:
                    child.parent_id = node.id
                    child.depth = depth
                    node.children_ids.append(child.id)
                    self.nodes[child.id] = child
                next_candidates.extend(children)

            if not next_candidates:
                break

            # Evaluate and beam
            scored_next = self._evaluate_thoughts(
                problem, next_candidates, context,
            )
            active = scored_next[:self.beam_width]

            # Check if any reached a terminal solution
            terminals = [n for n in active if n.is_terminal]
            if terminals and terminals[0].score >= 7.0:
                break

        # ── Phase 3: Backtrack to find best path ──
        all_leaves = [n for n in self.nodes.values()
                      if not n.children_ids or n.is_terminal]
        if not all_leaves:
            all_leaves = [n for n in self.nodes.values() if n.depth > 0]

        best_leaf = max(all_leaves, key=lambda n: n.score) if all_leaves else root

        path = self._backtrack(best_leaf)
        self._best_path = path
        self._latency_ms = (time.perf_counter() - t0) * 1000
        return path

    def best_path_text(self) -> str:
        """Format the best path as readable text."""
        if not hasattr(self, '_best_path') or not self._best_path:
            return "(未找到推理路径)"
        lines = [f"## Tree-of-Thought 推理路径 ({self._latency_ms:.0f}ms)\n"]
        for i, node in enumerate(self._best_path):
            if i == 0:
                continue  # skip root problem node
            prefix = "  " * (node.depth - 1) + "→"
            lines.append(f"{prefix} **[{node.score:.1f}/10]** {node.content[:200]}")
            if node.eval_feedback:
                lines.append(f"     _{node.eval_feedback[:150]}_")
        return "\n".join(lines)

    # ── Internal methods ────────────────────────────────────────────

    def _new_node(self, content: str, depth: int = 0,
                  parent_id: str | None = None) -> ThoughtNode:
        self._id_counter += 1
        return ThoughtNode(
            id=f"thought_{self._id_counter}",
            content=content,
            depth=depth,
            parent_id=parent_id,
        )

    def _generate_thoughts(self, problem: str, context: str,
                           parent: ThoughtNode | None = None,
                           num: int = 4) -> list[ThoughtNode]:
        """Generate candidate next thoughts using the LLM."""
        parent_text = f"\n当前思路: {parent.content[:300]}" if parent else ""
        context_text = f"\n上下文: {context[:500]}" if context else ""

        prompt = f"""你是一个深度推理系统。为以下问题生成 {num} 个不同的解决思路。

问题: {problem}{context_text}{parent_text}

请生成 {num} 个不同的候选方案。每个方案应该有不同的切入点或策略。
如果某个方案已经可以直接解决问题，标记 is_terminal=true。

输出 ONLY 一个 JSON 数组:
[
  {{"content": "方案1: 具体思路...", "is_terminal": false}},
  {{"content": "方案2: 另一种思路...", "is_terminal": false}},
  ...
]

要求:
- 方案之间要有明显差异（不同角度、不同策略）
- 每个方案要具体、可执行
- content 用中文，控制在200字以内
- 如果方案已经是完整的最终答案，设置 is_terminal=true"""

        try:
            if self.llm is None or self.llm.is_mock:
                return self._heuristic_thoughts(problem, num)

            result = self._llm_json(prompt)
            if isinstance(result, list):
                nodes = []
                for item in result:
                    if isinstance(item, dict) and item.get("content"):
                        node = self._new_node(
                            content=item["content"],
                            depth=(parent.depth + 1) if parent else 1,
                        )
                        node.is_terminal = item.get("is_terminal", False)
                        nodes.append(node)
                return nodes[:num]
        except Exception:
            pass

        return self._heuristic_thoughts(problem, num)

    def _evaluate_thoughts(self, problem: str,
                           candidates: list[ThoughtNode],
                           context: str = "") -> list[ThoughtNode]:
        """Evaluate and score candidate thoughts."""
        if not candidates:
            return []

        if self.llm is None or self.llm.is_mock:
            return self._heuristic_evaluate(candidates)

        candidates_text = "\n".join(
            f"{i+1}. {n.content[:200]}"
            for i, n in enumerate(candidates)
        )

        prompt = f"""评估以下候选方案对问题的适用性。

问题: {problem[:300]}
上下文: {context[:300] if context else "无"}

候选方案:
{candidates_text}

对每个方案打分 (1-10):
- 1-3: 不可行、不相关或有严重缺陷
- 4-6: 基本可行但有不足
- 7-8: 可行且合理
- 9-10: 最佳方案，全面且高效

输出 ONLY 一个 JSON 数组:
[
  {{"index": 1, "score": 8, "feedback": "简短评语（中文）"}},
  ...
]
"""

        try:
            result = self._llm_json(prompt)
            if isinstance(result, list):
                score_map: dict[int, tuple[float, str]] = {}
                for item in result:
                    if isinstance(item, dict):
                        idx = item.get("index", 0)
                        score_map[idx] = (
                            float(item.get("score", 5)),
                            str(item.get("feedback", "")),
                        )

                for i, node in enumerate(candidates):
                    if (i + 1) in score_map:
                        node.score, node.eval_feedback = score_map[i + 1]
                    else:
                        node.score = 5.0
        except Exception:
            return self._heuristic_evaluate(candidates)

        candidates.sort(key=lambda n: -n.score)
        return candidates

    def _heuristic_thoughts(self, problem: str, num: int) -> list[ThoughtNode]:
        """Generate diverse heuristic thoughts without LLM."""
        templates = [
            f"分步骤解决: 将「{problem[:40]}」拆分为独立的子任务逐个处理",
            f"工具优先: 列出解决「{problem[:40]}」所需的关键工具和API",
            f"数据流分析: 追踪「{problem[:40]}」的输入输出数据流",
            f"模式匹配: 寻找与「{problem[:40]}」相似的已知解决方案",
            f"增量迭代: 先做最小可行版本，逐步完善「{problem[:40]}」",
        ]
        nodes = []
        for t in templates[:num]:
            node = self._new_node(content=t, depth=1)
            nodes.append(node)
        return nodes

    def _heuristic_evaluate(self, candidates: list[ThoughtNode]) -> list[ThoughtNode]:
        """Simple heuristic scoring: prefer diverse, specific thoughts."""
        for i, node in enumerate(candidates):
            content = node.content
            # Reward specificity (length), penalize vagueness
            length_score = min(3.0, len(content) / 80)
            diversity_bonus = (len(candidates) - i) * 0.5
            has_specific = 1.5 if any(w in content for w in
                ("步骤", "方法", "使用", "分析", "方案", "策略", "具体")) else 0
            node.score = min(10, 5.0 + length_score + diversity_bonus + has_specific)
        candidates.sort(key=lambda n: -n.score)
        return candidates

    def _backtrack(self, leaf: ThoughtNode) -> list[ThoughtNode]:
        """Trace path from leaf back to root."""
        path: list[ThoughtNode] = []
        current: ThoughtNode | None = leaf
        while current is not None:
            path.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        path.reverse()
        return path

    def _llm_json(self, prompt: str) -> list | dict | None:
        """Lightweight JSON call through LLM provider."""
        if self.llm is None:
            return None
        try:
            return self.llm._llm_json_call(prompt)
        except Exception:
            return None


# ── Complexity detection ──────────────────────────────────────────────

# Keywords that trigger Tree-of-Thought deep reasoning
_TOT_TRIGGER_KEYWORDS = [
    # Chinese
    "分析架构", "重构", "设计模式", "架构设计", "复杂", "困难",
    "规划", "方案设计", "优化策略", "多步骤", "深度", "系统设计",
    # English
    "architecture", "refactor", "complex", "difficult", "design pattern",
    "optimization strategy", "system design", "multi-step", "deep",
]


def should_use_tot(user_input: str) -> bool:
    """Detect if the query warrants Tree-of-Thought deep reasoning."""
    inp_lower = user_input.lower()
    for kw in _TOT_TRIGGER_KEYWORDS:
        if kw.lower() in inp_lower:
            return True
    # Heuristic: very long queries likely need deeper reasoning
    if len(user_input) > 200:
        return True
    return False
