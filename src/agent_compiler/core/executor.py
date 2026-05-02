"""Execution engine — plan-driven execution with progress tracking.

Executes an ExecutionPlan step by step, respecting dependencies,
handling failures, and reporting progress.
"""

from __future__ import annotations

import time
from typing import Callable

from agent_compiler.core.planning import (
    ExecutionPlan, PlanStep, StepStatus,
)
from agent_compiler.core.types import ActionStep
from agent_compiler.tools.registry import ToolRegistry


class ExecutionResult:
    """Result of executing a plan."""

    def __init__(self, plan: ExecutionPlan):
        self.plan = plan
        self.steps_executed = 0
        self.steps_failed = 0
        self.steps_skipped = 0
        self.total_latency_ms = 0.0
        self.messages: list[str] = []

    @property
    def success(self) -> bool:
        return self.steps_failed == 0

    def to_text(self) -> str:
        lines = [
            f"## 执行完成: {self.plan.title}",
            f"步骤: {self.steps_executed} 成功, {self.steps_failed} 失败, "
            f"{self.steps_skipped} 跳过",
            f"耗时: {self.total_latency_ms:.0f}ms",
        ]
        for s in self.plan.steps:
            icon = {StepStatus.COMPLETED: "✅",
                    StepStatus.FAILED: "❌",
                    StepStatus.SKIPPED: "⏭️"}.get(s.status, "")
            lines.append(f"  {icon} {s.description}")
            if s.error:
                lines.append(f"     错误: {s.error}")
        return "\n".join(lines)


class Executor:
    """Executes an ExecutionPlan, managing dependencies and retries.

    Usage:
        executor = Executor(max_parallel=3)
        result = executor.execute(plan)
        print(result.to_text())
    """

    def __init__(self, max_parallel: int = 1,
                 on_progress: Callable[[ExecutionPlan], None] | None = None):
        self.max_parallel = max_parallel
        self.on_progress = on_progress
        self._abort_on_failure = False

    def execute(self, plan: ExecutionPlan,
                abort_on_failure: bool = False) -> ExecutionResult:
        """Execute a plan, respecting dependencies and retries.

        Args:
            plan: The execution plan to run
            abort_on_failure: If True, stop on first failure

        Returns ExecutionResult with summary.
        """
        self._abort_on_failure = abort_on_failure
        result = ExecutionResult(plan)
        t0 = time.perf_counter()

        while not plan.is_complete:
            ready = plan.get_ready_steps()

            if not ready and not plan.is_complete:
                # Check for deadlock (steps pending but none ready)
                pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if pending and all(
                    not self._deps_met(s, plan) for s in pending
                ):
                    # Deadlock: mark remaining as skipped
                    for s in pending:
                        s.status = StepStatus.SKIPPED
                        s.error = "依赖步骤失败，跳过"
                        result.steps_skipped += 1
                    break
                # Otherwise steps might be in progress, continue
                break

            for step in ready[:max(1, self.max_parallel)]:
                self._execute_step(step, result)
                if self._abort_on_failure and step.status == StepStatus.FAILED:
                    self._skip_remaining(plan, result)
                    break

            if self.on_progress:
                self.on_progress(plan)

        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        plan.completed_at = time.time()
        return result

    def _execute_step(self, step: PlanStep, result: ExecutionResult):
        """Execute a single plan step."""
        step.status = StepStatus.IN_PROGRESS
        step.started_at = time.time()

        if not step.tool_name:
            # No tool specified — the step is descriptive only
            step.status = StepStatus.COMPLETED
            step.completed_at = time.time()
            result.steps_executed += 1
            return

        # Execute via ToolRegistry
        action = ActionStep(
            tool_name=step.tool_name,
            params=step.params,
            description=step.description,
        )

        while step.retry_count <= step.max_retries:
            try:
                output = ToolRegistry.execute(action)
                if output.get("success"):
                    step.result = output
                    step.status = StepStatus.COMPLETED
                    result.steps_executed += 1
                else:
                    step.error = output.get("error", "未知错误")
                    step.retry_count += 1
                    if step.retry_count > step.max_retries:
                        step.status = StepStatus.FAILED
                        result.steps_failed += 1
                    else:
                        time.sleep(0.5)  # brief pause before retry
                        continue
                break
            except Exception as e:
                step.error = str(e)
                step.retry_count += 1
                if step.retry_count > step.max_retries:
                    step.status = StepStatus.FAILED
                    result.steps_failed += 1
                else:
                    time.sleep(0.5)

        step.completed_at = time.time()

    def _deps_met(self, step: PlanStep, plan: ExecutionPlan) -> bool:
        completed_ids = {s.id for s in plan.steps if s.status == StepStatus.COMPLETED}
        return all(d in completed_ids for d in step.depends_on)

    def _skip_remaining(self, plan: ExecutionPlan, result: ExecutionResult):
        """Skip all remaining pending steps."""
        for s in plan.steps:
            if s.status == StepStatus.PENDING:
                s.status = StepStatus.SKIPPED
                s.error = "前置步骤失败，已跳过"
                result.steps_skipped += 1


def execute_plan(plan: ExecutionPlan) -> ExecutionResult:
    """Convenience function: execute a plan with default settings."""
    executor = Executor()
    return executor.execute(plan)
