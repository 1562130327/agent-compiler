"""Workflow compiler: extracts reusable templates from LLM output and parameterizes them.

Two stages:
  1. Extract: LLM output -> structured ActionSteps
  2. Parameterize: replace magic literals with variables
"""

from __future__ import annotations

import re

from agent_compiler.core.types import ActionStep, WorkflowTemplate


class Compiler:
    """Compiles LLM reasoning results into reusable workflow templates."""

    def compile(self, intent: str, steps_data: list[dict]) -> WorkflowTemplate:
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
        )
        return self.parameterize(wf)

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
                        step.params[key] = wf.params_schema[param_name]["default"]
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
