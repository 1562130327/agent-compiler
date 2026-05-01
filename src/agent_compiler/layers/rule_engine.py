"""Layer 1: Rule engine — O(1) keyword/regex matching on CPU."""

from __future__ import annotations

import re
import time
from pathlib import Path

import yaml

from agent_compiler.core.types import ActionStep, AgentResult, Rule


class RuleEngine:
    """Pure CPU rule matcher. Microsecond-level response for known patterns."""

    def __init__(self, rules_path: str | None = None):
        self.rules: list[Rule] = []
        self._compiled: list[tuple[re.Pattern, Rule]] = []
        self._keyword_trie: dict[str, list[Rule]] = {}

        if rules_path and Path(rules_path).exists():
            self.load_rules(rules_path)

    def load_rules(self, path: str):
        """Load rules from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for r in data.get("rules", []):
            rule = Rule(
                name=r["name"],
                keywords=r.get("keywords", []),
                patterns=r.get("patterns", []),
                tool_name=r["tool_name"],
                params=r.get("params", {}),
            )
            self.add_rule(rule)

    def add_rule(self, rule: Rule):
        """Add a single rule to the engine."""
        self.rules.append(rule)
        # Compile regex patterns
        for pat in rule.patterns:
            self._compiled.append((re.compile(pat, re.IGNORECASE), rule))
        # Build keyword trie entries
        for kw in rule.keywords:
            low = kw.lower()
            self._keyword_trie.setdefault(low, []).append(rule)

    def match(self, user_input: str) -> AgentResult | None:
        """Try to match input against rules. Returns result or None if no match."""
        t0 = time.perf_counter()
        input_lower = user_input.lower()

        # Phase 1: keyword match (fastest — O(k) where k=#keywords in input)
        for keyword, rules in self._keyword_trie.items():
            if keyword in input_lower:
                rule = rules[0]  # first match wins
                latency = (time.perf_counter() - t0) * 1000
                return AgentResult(
                    success=True,
                    data={"tool": rule.tool_name, "params": rule.params},
                    source="rule",
                    confidence=1.0,
                    latency_ms=latency,
                    workflow_id=rule.name,
                )

        # Phase 2: regex match (slower but more flexible)
        for pattern, rule in self._compiled:
            if pattern.search(user_input):
                latency = (time.perf_counter() - t0) * 1000
                return AgentResult(
                    success=True,
                    data={"tool": rule.tool_name, "params": rule.params},
                    source="rule",
                    confidence=0.95,
                    latency_ms=latency,
                    workflow_id=rule.name,
                )

        return None

    def promote(self, workflow_template) -> Rule | None:
        """Promote a high-frequency cached workflow to a rule (manual review flag)."""
        if workflow_template.hit_count < 100:
            return None
        rule = Rule(
            name=workflow_template.id,
            keywords=[workflow_template.intent.lower()],
            patterns=[],
            tool_name="execute_workflow",
            params={"workflow_id": workflow_template.id},
        )
        return rule

    def stats(self) -> dict:
        return {
            "total_rules": len(self.rules),
            "compiled_patterns": len(self._compiled),
            "keyword_entries": sum(len(v) for v in self._keyword_trie.values()),
        }
