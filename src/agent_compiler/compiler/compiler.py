"""Workflow compiler: extracts reusable templates from LLM output and parameterizes them.

Two stages:
  1. Extract: LLM output -> structured ActionSteps
  2. Parameterize: replace magic literals with variables

Also extracts keywords from intent + original input for L1 dynamic rule growth.
"""

from __future__ import annotations

import re
from collections import Counter

from agent_compiler.core.types import ActionStep, WorkflowTemplate


class Compiler:
    """Compiles LLM reasoning results into reusable workflow templates."""

    def compile(self, intent: str, steps_data: list[dict],
                original_input: str = "") -> WorkflowTemplate:
        """Convert LLM output to a workflow template."""
        steps = []
        for s in steps_data:
            step = ActionStep(
                tool_name=s.get("tool_name", ""),
                params=s.get("params", {}),
                is_generic=False,
                description=s.get("description", ""),
            )
            steps.append(step)

        wf = WorkflowTemplate(
            id=WorkflowTemplate.generate_id(intent),
            intent=intent,
            steps=steps,
            keywords=self._extract_keywords(intent, original_input),
            original_input=original_input,
        )
        return self.parameterize(wf)

    @staticmethod
    def _extract_keywords(intent: str, original_input: str = "") -> list[str]:
        """Extract distinctive keywords from intent and original input.

        These keywords become L1 dynamic rules — when a future input
        contains them, we skip straight to executing this workflow.
        """
        text = (intent + " " + original_input).lower().strip()

        tokens: list[str] = []

        # Chinese character n-grams (2-4 chars) — captures meaningful phrases
        for n in (2, 3, 4):
            for i in range(len(text) - n + 1):
                chunk = text[i:i + n]
                # Only keep chunks that are primarily CJK
                cjk = sum(1 for c in chunk if '一' <= c <= '鿿')
                if cjk >= n - 1:
                    tokens.append(chunk)

        # English / alphanumeric words
        for w in re.findall(r'[a-z0-9]{2,}', text):
            tokens.append(w)

        # Count and score
        freqs = Counter(tokens)
        total = sum(freqs.values()) or 1

        # Score: frequency * distinctiveness (penalize very common 2-grams)
        scored = []
        for t, f in freqs.items():
            distinctiveness = len(t) / 2  # longer tokens are more distinctive
            score = (f / total) * distinctiveness
            scored.append((t, score))

        scored.sort(key=lambda x: -x[1])

        # Return top 5-8 keywords
        keywords = [t for t, s in scored[:8] if s > 0.02]
        if not keywords:
            keywords = [t for t, s in scored[:3]]
        return keywords[:8]

    def parameterize(self, wf: WorkflowTemplate) -> WorkflowTemplate:
        """Detect concrete values and lift them into parameters.

        Heuristics for "magic literals":
        - File paths (like /path/to/file or C:\\...)
        - Timestamps / dates
        - Numeric limits (top N, last N)
        """
        params_schema = {}
        for step in wf.steps:
            for key, val in list(step.params.items()):
                if isinstance(val, str):
                    param_name = self._extract_param(key, val)
                    if param_name:
                        params_schema[param_name] = {"type": "string", "default": val}
                        step.params[key] = f"${{{param_name}}}"
                        step.is_generic = True
                elif isinstance(val, (int, float)) and key in ("limit", "top_n", "days", "count", "size"):
                    params_schema[key] = {"type": type(val).__name__, "default": val}
                    step.params[key] = f"${{{key}}}"
                    step.is_generic = True

        wf.params_schema = params_schema
        return wf

    def instantiate(self, wf: WorkflowTemplate, params: dict) -> list[ActionStep]:
        """Fill in a parameterized workflow with concrete values.

        Resolves ${param} from `params` dict first, then falls back
        to default values in wf.params_schema.
        """
        import copy
        steps = copy.deepcopy(wf.steps)
        for step in steps:
            for key, val in list(step.params.items()):
                if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                    param_name = val[2:-1]
                    if param_name in params:
                        step.params[key] = params[param_name]
                    elif param_name in wf.params_schema:
                        default = wf.params_schema[param_name].get("default")
                        # Don't pass unresolved placeholders to tools
                        if isinstance(default, str) and default.startswith("${"):
                            step.params[key] = None
                        else:
                            step.params[key] = default
        return steps

    def _extract_param(self, key: str, value: str) -> str | None:
        """Try to extract a parameter name from a string value."""
        if re.search(r'[/\\]', value) and not value.startswith("${"):
            return key
        if re.match(r'^\d{4}-\d{2}-\d{2}', value):
            return key
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', value):
            return key
        if re.match(r'^[\w-]+\.[\w.]+$', value):
            return key
        return None
