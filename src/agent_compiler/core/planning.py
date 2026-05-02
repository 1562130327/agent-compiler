"""Planning engine — task decomposition, dependency ordering, plan tracking.

Inspired by Claude Code's Plan mode and OpenClaw's multi-step execution.
Converts complex user requests into executable plans with dependencies.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """One step in an execution plan."""
    id: str
    description: str
    tool_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # step IDs
    status: StepStatus = StepStatus.PENDING
    result: dict | None = None
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    retry_count: int = 0
    max_retries: int = 2

    @property
    def is_ready(self) -> bool:
        return self.status == StepStatus.PENDING

    @property
    def is_terminal(self) -> bool:
        return self.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED)

    @property
    def all_dependencies_met(self) -> bool:
        return True  # checked externally against plan state


@dataclass
class ExecutionPlan:
    """A complete execution plan with ordered steps."""
    id: str
    title: str
    description: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    @property
    def progress(self) -> dict:
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        pending = sum(1 for s in self.steps if s.status == StepStatus.PENDING)
        in_progress = sum(1 for s in self.steps if s.status == StepStatus.IN_PROGRESS)
        return {
            "total": total, "completed": completed, "failed": failed,
            "pending": pending, "in_progress": in_progress,
            "pct": int(completed / total * 100) if total > 0 else 0,
        }

    @property
    def is_complete(self) -> bool:
        return all(s.is_terminal for s in self.steps)

    @property
    def has_failures(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def get_ready_steps(self) -> list[PlanStep]:
        """Get steps that are ready to execute (dependencies satisfied)."""
        completed_ids = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        ready = []
        for s in self.steps:
            if s.status != StepStatus.PENDING:
                continue
            deps_met = all(d in completed_ids for d in s.depends_on)
            if deps_met:
                ready.append(s)
        return ready

    def to_prompt(self) -> str:
        """Format the plan as a prompt for the LLM to follow."""
        lines = [f"## 执行计划: {self.title}"]
        if self.description:
            lines.append(f"目标: {self.description}")
        lines.append("")
        lines.append("### 步骤")
        for i, s in enumerate(self.steps):
            status_icon = {
                StepStatus.PENDING: "⬜",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
            }[s.status]
            deps = f" (依赖: {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"{status_icon} **{i+1}. {s.description}**{deps}")
            if s.error:
                lines.append(f"   错误: {s.error}")
            if s.result:
                r = s.result
                if r.get("data"):
                    lines.append(f"   结果: {str(r['data'])[:200]}")
        lines.append("")
        lines.append(f"进度: {self.progress['completed']}/{self.progress['total']} ({self.progress['pct']}%)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "tool_name": s.tool_name,
                    "params": s.params,
                    "depends_on": s.depends_on,
                    "status": s.status.value,
                    "error": s.error,
                    "retry_count": s.retry_count,
                }
                for s in self.steps
            ],
            "progress": self.progress,
        }


class Planner:
    """Plans multi-step tasks by asking the LLM to decompose a goal.

    Usage:
        planner = Planner(llm_provider)
        plan = planner.plan("重构 agent.py 中的 _run_react_loop 方法")
        # plan.steps = [PlanStep(id="1", description="读文件", ...), ...]
    """

    def __init__(self, llm_provider=None):
        self._llm = llm_provider

    def plan(self, goal: str, available_tools: list[str] | None = None,
             context: str = "") -> ExecutionPlan:
        """Generate an execution plan from a goal description.

        If an LLM provider is available, uses it for intelligent planning.
        Otherwise falls back to simple decomposition.
        """
        if self._llm and not self._llm.is_mock:
            return self._llm_plan(goal, available_tools, context)
        return self._heuristic_plan(goal, available_tools)

    def _llm_plan(self, goal: str, available_tools: list[str] | None,
                  context: str) -> ExecutionPlan:
        """Use LLM to generate a structured plan."""
        tools_str = ", ".join(available_tools) if available_tools else "(unknown)"
        prompt = f"""You are a task planner. Break down the following goal into an ordered sequence of executable steps.

Goal: {goal}

Available tools: {tools_str}

Output ONLY a JSON object with this structure:
{{
    "title": "short plan title",
    "description": "what this plan accomplishes",
    "steps": [
        {{
            "id": "1",
            "description": "what to do in this step",
            "tool_name": "tool to use (or empty if manual)",
            "params": {{}},
            "depends_on": []
        }}
    ]
}}

Rules:
- Each step should be specific and actionable
- Mark dependencies between steps using "depends_on" with step IDs
- Steps can run in parallel if they don't depend on each other
- Use the most appropriate tool for each step
- Output ONLY valid JSON, no markdown"""

        try:
            if hasattr(self._llm, '_llm_json_call'):
                result = self._llm._llm_json_call(prompt)
            else:
                return self._heuristic_plan(goal, available_tools)

            plan_id = f"plan_{int(time.time())}"
            plan = ExecutionPlan(
                id=plan_id,
                title=result.get("title", goal[:60]),
                description=result.get("description", goal),
            )
            for sd in result.get("steps", []):
                plan.steps.append(PlanStep(
                    id=sd.get("id", str(len(plan.steps) + 1)),
                    description=sd.get("description", ""),
                    tool_name=sd.get("tool_name", ""),
                    params=sd.get("params", {}),
                    depends_on=sd.get("depends_on", []),
                ))
            return plan
        except Exception:
            return self._heuristic_plan(goal, available_tools)

    def _heuristic_plan(self, goal: str,
                        available_tools: list[str] | None = None) -> ExecutionPlan:
        """Simple heuristic plan when no LLM is available."""
        plan_id = f"plan_{int(time.time())}"
        plan = ExecutionPlan(
            id=plan_id,
            title=goal[:80],
            description=goal,
        )

        tools = available_tools or []

        # Heuristic: if goal mentions "read" or "file", read first
        if any(w in goal.lower() for w in ("读", "read", "file", "文件", "查看")):
            plan.steps.append(PlanStep(
                id="1",
                description=f"读取相关文件: {goal[:60]}",
                tool_name="read_file" if "read_file" in tools else "",
                params={"path": "."},
            ))

        # Heuristic: if goal mentions "search", "find", "查找"
        if any(w in goal.lower() for w in ("search", "find", "查找", "搜索", "grep")):
            plan.steps.append(PlanStep(
                id=str(len(plan.steps) + 1),
                description=f"搜索相关代码: {goal[:60]}",
                tool_name="search_files" if "search_files" in tools else "",
                params={"directory": ".", "pattern": goal[:30]},
            ))

        # Heuristic: if goal mentions "modify", "change", "edit", "修改"
        if any(w in goal.lower() for w in ("modify", "change", "edit", "修改", "改", "替换")):
            plan.steps.append(PlanStep(
                id=str(len(plan.steps) + 1),
                description=f"编辑文件: {goal[:60]}",
                tool_name="edit_file" if "edit_file" in tools else "write_file",
                params={},
            ))

        # Heuristic: if goal mentions "test", "测试"
        if any(w in goal.lower() for w in ("test", "测试", "验证", "verify", "check")):
            plan.steps.append(PlanStep(
                id=str(len(plan.steps) + 1),
                description=f"运行测试验证: {goal[:60]}",
                tool_name="run_tests" if "run_tests" in tools else "execute_shell",
                params={"path": "."},
            ))

        # If no steps were added, add a generic exploration step
        if not plan.steps:
            plan.steps.append(PlanStep(
                id="1",
                description=f"分析并执行: {goal[:80]}",
                tool_name="",
                params={},
            ))

        # Add dependencies (sequential by default)
        for i in range(1, len(plan.steps)):
            plan.steps[i].depends_on = [plan.steps[i-1].id]

        return plan
