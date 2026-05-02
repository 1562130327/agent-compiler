"""Skill loader — discovers and loads SKILL.md files."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Skill:
    """A loaded skill from a SKILL.md file."""
    name: str
    description: str = ""
    body: str = ""                     # Markdown instructions (after frontmatter)
    allowed_tools: list[str] = field(default_factory=list)
    disable_model_invocation: bool = False  # manual /slash only
    argument_hint: str = ""
    version: str = "0.1.0"
    source_path: str = ""              # where the SKILL.md was loaded from
    context: str = "inline"            # "inline" or "fork"

    def resolve_body(self, args: str = "") -> str:
        """Return the skill body with $ARGUMENTS and $N interpolated."""
        body = self.body
        if args:
            body = body.replace("$ARGUMENTS", args)
            parts = args.split()
            for i, p in enumerate(parts):
                body = body.replace(f"${i}", p)
        # Resolve !`command` shell expansions
        def _sub_shell(m: re.Match) -> str:
            cmd = m.group(1)
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True,
                                   text=True, timeout=10, encoding="utf-8", errors="replace")
                return r.stdout.strip()
            except Exception:
                return f"[shell error: {cmd}]"
        body = re.sub(r'!`([^`]+)`', _sub_shell, body)
        return body

    @property
    def is_auto_trigger(self) -> bool:
        """Whether the skill can be auto-invoked by description matching."""
        return not self.disable_model_invocation and bool(self.description)


class SkillLoader:
    """Loads skills from project and user skill directories.

    Directory layout:
        skills/
        ├── my-skill/
        │   ├── SKILL.md          # Required
        │   ├── scripts/          # Optional executables
        │   └── references/       # Optional docs (loaded on demand)
        └── another-skill/
            └── SKILL.md

    Usage:
        loader = SkillLoader()
        loader.discover()  # scans standard dirs
        skill = loader.get("my-skill")
        if skill:
            prompt = skill.resolve_body(args="some args")
    """

    def __init__(self, project_dir: str | None = None):
        self._skills: dict[str, Skill] = {}
        self._project_dir = Path(project_dir) if project_dir else Path.cwd()

    def discover(self, extra_dirs: list[str] | None = None):
        """Scan standard skill directories for SKILL.md files."""
        dirs = []

        # Project skills (highest priority)
        dirs.append(self._project_dir / ".agent-compiler" / "skills")
        # User skills
        dirs.append(Path.home() / ".agent-compiler" / "skills")
        # Extra dirs
        if extra_dirs:
            dirs.extend(Path(d) for d in extra_dirs)

        for skills_dir in dirs:
            if not skills_dir.is_dir():
                continue
            for skill_dir in sorted(skills_dir.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if skill_md.exists():
                    skill = self._load_skill(skill_md)
                    if skill and skill.name not in self._skills:
                        self._skills[skill.name] = skill

    def _load_skill(self, path: Path) -> Skill | None:
        """Parse a SKILL.md file into a Skill object."""
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None

        # Parse YAML frontmatter
        fm: dict[str, Any] = {}
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    pass
                body = parts[2].strip()

        name = fm.get("name", path.parent.name)
        description = fm.get("description", "")
        allowed = fm.get("allowed-tools", [])
        if isinstance(allowed, str):
            allowed = [t.strip() for t in allowed.split(",")]

        return Skill(
            name=name,
            description=description,
            body=body,
            allowed_tools=allowed,
            disable_model_invocation=fm.get("disable-model-invocation", False),
            argument_hint=fm.get("argument-hint", ""),
            version=str(fm.get("version", "0.1.0")),
            source_path=str(path),
            context=fm.get("context", "inline"),
        )

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def list_auto_triggers(self) -> list[Skill]:
        return [s for s in self._skills.values() if s.is_auto_trigger]

    def match(self, user_input: str) -> Skill | None:
        """Find a skill whose description matches the user input.

        Simple keyword matching — the skill description should contain
        trigger phrases that match the user's intent.
        """
        inp_lower = user_input.lower()
        best: tuple[Skill, int] | None = None

        for skill in self._skills.values():
            if not skill.is_auto_trigger:
                continue
            desc_lower = skill.description.lower()
            # Count keyword overlap
            score = sum(1 for word in desc_lower.split()
                       if len(word) > 2 and word in inp_lower)
            if score > 0:
                if best is None or score > best[1]:
                    best = (skill, score)

            # Also check if the skill name itself appears in the input
            if skill.name.lower() in inp_lower:
                score += 10
                if best is None or score > best[1]:
                    best = (skill, score)

        return best[0] if best else None

    def build_context_prompt(self) -> str:
        """Build a system-level prompt listing available skills for the LLM."""
        skills = self.list_all()
        if not skills:
            return ""

        lines = ["## 可用技能 (Skills)"]
        for s in skills:
            trigger = "手动触发" if s.disable_model_invocation else "自动匹配"
            hint = f" 参数: {s.argument_hint}" if s.argument_hint else ""
            lines.append(f"- **{s.name}** ({trigger}){hint}: {s.description[:120]}")
        lines.append("")
        lines.append("使用 Skill 工具来调用技能。你可以在回复中建议用户使用 /skill-name 来手动调用。")
        return "\n".join(lines)
