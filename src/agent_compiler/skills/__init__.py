"""Skill system — SKILL.md loading, matching, and invocation.

Skills are modular capability packages defined by SKILL.md files
with YAML frontmatter + Markdown instructions.

Directories searched (in order of priority):
  1. Project:  .agent-compiler/skills/<name>/SKILL.md
  2. User:     ~/.agent-compiler/skills/<name>/SKILL.md
"""

from agent_compiler.skills.loader import SkillLoader, Skill

__all__ = ["SkillLoader", "Skill"]
