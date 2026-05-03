"""AI 漫剧数据模型 — 持久化项目数据到磁盘."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StyleConfig:
    """项目全局风格配置 — 审图阶段锁定后所有分镜共用.

    审图时系统自动枚举不同组合生成预览图，用户选择最佳组合后锁定。
    锁定的组合参数（checkpoint/LoRA/sampler等）会贯穿后续所有生图流程。
    """
    project_id: str
    style_name: str = "日漫风格"
    workflow_id: str = ""                         # 生图用的 ComfyUI 工作流 (系统自动选)
    positive_prefix: str = "masterpiece, best quality"
    negative_prompt: str = "低质量, 模糊, 畸形手, 畸形脸, 文字, 水印"
    fixed_params: dict[str, Any] = field(default_factory=dict)  # 锁定参数如 seed, cfg, size

    # Locked combo from review (审图确定的组合，贯穿后续生图)
    locked_checkpoint: str = ""                   # 选定的底模
    locked_loras: list[dict] = field(default_factory=list)     # [{name, strength}]
    locked_sampler: str = ""                      # 选定的采样器
    locked_scheduler: str = ""                    # 选定的调度器
    locked_vae: str = ""                          # 选定的VAE
    locked_combo_params: dict[str, Any] = field(default_factory=dict)  # 锁定时的全部节点参数快照

    character_ref_workflow: str = ""              # 角色参考图工作流
    environment_ref_workflow: str = ""            # 环境参考图工作流
    video_workflow: str = ""                      # 图生视频工作流(如 H18-图生视频)
    preview_images: list[str] = field(default_factory=list)  # 审图预览图列表
    preview_combo_map: list[dict] = field(default_factory=list)  # [{index, checkpoint, loras, sampler, params}]
    selected_preview: int = -1                    # 选中的预览图索引
    is_locked: bool = False
    locked_at: float = 0.0


@dataclass
class Character:
    """角色 — 含三视图级别的外观+服装细节，用于保持一致性."""
    id: str = ""
    name: str = ""
    description: str = ""                         # 性格/背景
    appearance_detail: str = ""                   # 三视图级外观：身高/体型/发型/脸型/肤色
    clothing_details: str = ""                    # 详细服装：款式/颜色/材质/配饰（用于提示词）
    voice_style: str = ""                         # 声音特点
    ref_image_front: str = ""                     # 正面参考图 URL
    ref_image_side: str = ""                      # 侧面参考图 URL
    ref_image_back: str = ""                      # 背面参考图 URL
    ref_image_full: str = ""                      # 全身参考图 URL
    lora_model: str = ""                          # LoRA 模型名
    lora_trigger: str = ""                        # LoRA 触发词
    status: str = "draft"                         # draft | generating | done | failed


@dataclass
class Environment:
    """环境/场景 — 含参考图用于分镜背景."""
    id: str = ""
    name: str = ""
    description: str = ""                         # 详细视觉描述
    ref_image_url: str = ""                       # 环境参考图 URL
    ref_prompt: str = ""                          # 生成时的提示词
    status: str = "draft"


@dataclass
class Panel:
    """分镜 — 含角色/环境映射、台词、配音、视频运动."""
    id: str = ""
    panel_num: int = 0
    scene_description: str = ""                   # 视觉场景描述（提示词基础）
    characters: list[str] = field(default_factory=list)    # 出现的角色 ID
    character_refs: list[str] = field(default_factory=list)  # 角色参考图 URL（用于图生图）
    environment: str = ""                         # 环境 ID
    environment_ref: str = ""                     # 环境参考图 URL（用于图生图）
    dialogue: str = ""                            # 台词文本
    voice_line: str = ""                          # 配音文本（含语气指示）
    emotion: str = ""                             # 情绪/氛围
    camera: str = ""                              # 镜头类型
    action: str = ""                              # 动作描述
    video_motion: str = ""                        # 视频运动描述
    prompt_positive: str = ""                     # 组装后的正向提示词
    prompt_negative: str = ""                     # 组装后的负向提示词
    generated_images: list[str] = field(default_factory=list)  # 分镜图 URL
    generated_video_url: str = ""                 # 视频 URL
    status: str = "draft"                         # draft | prompt_ready | generating | done | failed


@dataclass
class SkillTemplate:
    """从成功 pipeline 自动生成的可复用技能."""
    id: str = ""
    name: str = ""
    description: str = ""                         # 触发短语（逗号分隔）
    pipeline_steps: list[dict] = field(default_factory=list)
    source_project_id: str = ""
    created_at: float = 0.0
    usage_count: int = 0


class ProjectStore:
    """磁盘持久化的项目数据存储.

    目录结构:
        comic_data/
          index.json            — 项目索引
          {project_id}/
            project.json       — 项目元数据
            style.json         — 风格配置
            panels.json        — 分镜列表
            characters.json    — 角色列表
            environments.json  — 环境列表
        comic_skills/
          {skill_name}.json    — 技能模板
    """

    def __init__(self, data_dir: str = "./comic_data"):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._skills_dir = self._dir.parent / "comic_skills"
        self._skills_dir.mkdir(parents=True, exist_ok=True)

    # ── Projects ──────────────────────────────────────────────────

    def create_project(self, project_id: str = "", name: str = "未命名项目",
                       style_name: str = "日漫风格", workflow_id: str = "") -> dict:
        pid = project_id or str(uuid.uuid4())[:8]
        proj = {
            "id": pid, "name": name, "style_name": style_name,
            "workflow_id": workflow_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._proj_dir(pid).mkdir(parents=True, exist_ok=True)
        self._write_json(self._proj_path(pid, "project.json"), proj)
        self._ensure_defaults(pid)
        self._add_to_index(proj)
        return proj

    def get_project(self, project_id: str) -> dict | None:
        p = self._proj_path(project_id, "project.json")
        if p.exists():
            return self._read_json(p)
        return None

    def list_projects(self) -> list[dict]:
        idx = self._read_json(self._dir / "index.json")
        return list(idx.values()) if isinstance(idx, dict) else []

    def delete_project(self, project_id: str) -> bool:
        import shutil
        pdir = self._proj_dir(project_id)
        if pdir.exists():
            shutil.rmtree(pdir)
        self._remove_from_index(project_id)
        return True

    # ── Style ─────────────────────────────────────────────────────

    def save_style(self, project_id: str, config: StyleConfig):
        self._write_json(self._proj_path(project_id, "style.json"), {
            "project_id": project_id, "style_name": config.style_name,
            "workflow_id": config.workflow_id, "positive_prefix": config.positive_prefix,
            "negative_prompt": config.negative_prompt, "fixed_params": config.fixed_params,
            "locked_checkpoint": config.locked_checkpoint,
            "locked_loras": config.locked_loras,
            "locked_sampler": config.locked_sampler,
            "locked_scheduler": config.locked_scheduler,
            "locked_vae": config.locked_vae,
            "locked_combo_params": config.locked_combo_params,
            "character_ref_workflow": config.character_ref_workflow,
            "environment_ref_workflow": config.environment_ref_workflow,
            "video_workflow": config.video_workflow,
            "preview_images": config.preview_images,
            "preview_combo_map": config.preview_combo_map,
            "selected_preview": config.selected_preview,
            "is_locked": config.is_locked, "locked_at": config.locked_at,
        })

    def get_style(self, project_id: str) -> dict | None:
        p = self._proj_path(project_id, "style.json")
        return self._read_json(p) if p.exists() else None

    def lock_style(self, project_id: str) -> dict:
        cfg = self.get_style(project_id) or {}
        cfg["is_locked"] = True
        cfg["locked_at"] = time.time()
        self._write_json(self._proj_path(project_id, "style.json"), cfg)
        return cfg

    def unlock_style(self, project_id: str) -> dict:
        cfg = self.get_style(project_id) or {}
        cfg["is_locked"] = False
        self._write_json(self._proj_path(project_id, "style.json"), cfg)
        return cfg

    # ── Panels ────────────────────────────────────────────────────

    def save_panels(self, project_id: str, panels: list[dict]):
        self._write_json(self._proj_path(project_id, "panels.json"), panels)

    def get_panels(self, project_id: str) -> list[dict]:
        p = self._proj_path(project_id, "panels.json")
        return self._read_json(p) if p.exists() else []

    def update_panel(self, project_id: str, panel_id: str, updates: dict) -> dict | None:
        panels = self.get_panels(project_id)
        for i, p in enumerate(panels):
            if p.get("id") == panel_id:
                panels[i].update(updates)
                self.save_panels(project_id, panels)
                return panels[i]
        return None

    # ── Characters ────────────────────────────────────────────────

    def save_characters(self, project_id: str, chars: list[dict]):
        self._write_json(self._proj_path(project_id, "characters.json"), chars)

    def get_characters(self, project_id: str) -> list[dict]:
        p = self._proj_path(project_id, "characters.json")
        return self._read_json(p) if p.exists() else []

    # ── Environments ──────────────────────────────────────────────

    def save_environments(self, project_id: str, envs: list[dict]):
        self._write_json(self._proj_path(project_id, "environments.json"), envs)

    def get_environments(self, project_id: str) -> list[dict]:
        p = self._proj_path(project_id, "environments.json")
        return self._read_json(p) if p.exists() else []

    # ── Skills ────────────────────────────────────────────────────

    def save_skill(self, template: SkillTemplate) -> dict:
        skill_id = template.id or f"skill_{str(uuid.uuid4())[:8]}"
        data = {
            "id": skill_id, "name": template.name,
            "description": template.description,
            "pipeline_steps": template.pipeline_steps,
            "source_project_id": template.source_project_id,
            "created_at": template.created_at or time.time(),
            "usage_count": template.usage_count,
        }
        self._write_json(self._skills_dir / f"{skill_id}.json", data)
        self._write_skill_md(template)
        return data

    def list_skills(self) -> list[dict]:
        skills = []
        for f in sorted(self._skills_dir.glob("*.json")):
            d = self._read_json(f)
            if d:
                skills.append(d)
        return skills

    # ── Table view helper ─────────────────────────────────────────

    def build_table_view(self, project_id: str) -> dict:
        panels = self.get_panels(project_id)
        chars = {c.get("id", ""): c.get("name", "") for c in self.get_characters(project_id)}
        envs = {e.get("id", ""): e.get("name", "") for e in self.get_environments(project_id)}

        headers = ["分镜#", "场景", "角色", "环境", "台词", "配音", "镜头", "动作", "状态"]
        rows = []
        for p in panels:
            char_names = ", ".join(chars.get(cid, cid) for cid in p.get("characters", []))
            env_name = envs.get(p.get("environment", ""), p.get("environment", ""))
            rows.append([
                p.get("panel_num", 0),
                (p.get("scene_description", "") or "")[:80],
                char_names,
                env_name,
                p.get("dialogue", ""),
                p.get("voice_line", ""),
                p.get("camera", ""),
                p.get("action", ""),
                p.get("status", "draft"),
            ])
        return {"headers": headers, "rows": rows}

    # ── Internal helpers ──────────────────────────────────────────

    def _proj_dir(self, pid: str) -> Path:
        return self._dir / pid

    def _proj_path(self, pid: str, filename: str) -> Path:
        return self._dir / pid / filename

    def _ensure_defaults(self, pid: str):
        for fn in ["panels.json", "characters.json", "environments.json"]:
            p = self._proj_path(pid, fn)
            if not p.exists():
                self._write_json(p, [])

    def _add_to_index(self, proj: dict):
        idx = self._read_json(self._dir / "index.json") or {}
        idx[proj["id"]] = {"id": proj["id"], "name": proj["name"],
                           "created_at": proj["created_at"]}
        self._write_json(self._dir / "index.json", idx)

    def _remove_from_index(self, pid: str):
        idx = self._read_json(self._dir / "index.json") or {}
        idx.pop(pid, None)
        self._write_json(self._dir / "index.json", idx)

    def _write_skill_md(self, template: SkillTemplate):
        """同时生成 SKILL.md 供 agent SkillLoader 自动发现."""
        skill_dir = Path("./.agent-compiler/skills") / template.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        steps_text = "\n".join(
            f"  {i+1}. {s.get('step','')}: {s.get('description','')}"
            for i, s in enumerate(template.pipeline_steps)
        )
        md = f"""---
name: {template.name}
description: "{template.description}"
version: "1.0.0"
context: inline
---

用户想要{template.description.split(',')[0]}。执行完整的 AI 漫剧 pipeline：

{steps_text}

使用项目 {template.source_project_id} 的参数作为默认值。
"""
        (skill_dir / "SKILL.md").write_text(md, encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json(path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
