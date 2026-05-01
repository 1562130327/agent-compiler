"""L2: Disk/SSD-backed storage using SQLite for metadata and FAISS for vectors."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from agent_compiler.core.types import ActionStep, WorkflowTemplate


class DiskStore:
    """Persistent storage for workflow templates on local SSD."""

    def __init__(self, path: str = "./agent_cache"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "workflows.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    intent TEXT,
                    steps_json TEXT,
                    hit_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 1.0,
                    created_at REAL,
                    last_hit_at REAL,
                    params_schema_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hit_count ON workflows(hit_count DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_hit ON workflows(last_hit_at DESC)
            """)

    def save(self, wf: WorkflowTemplate):
        steps_json = json.dumps([
            {"tool_name": s.tool_name, "params": s.params,
             "is_generic": s.is_generic, "description": s.description}
            for s in wf.steps
        ], ensure_ascii=False)
        params_json = json.dumps(wf.params_schema, ensure_ascii=False)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workflows
                (id, intent, steps_json, hit_count, confidence, created_at, last_hit_at, params_schema_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (wf.id, wf.intent, steps_json, wf.hit_count, wf.confidence,
                  wf.created_at, wf.last_hit_at, params_json))

    def load(self, workflow_id: str) -> WorkflowTemplate | None:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM workflows WHERE id = ?", (workflow_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_wf(row)

    def load_all(self) -> list[WorkflowTemplate]:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute("SELECT * FROM workflows").fetchall()
        return [self._row_to_wf(r) for r in rows]

    def update_hit(self, workflow_id: str, hit_count: int):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE workflows SET hit_count = ?, last_hit_at = ? WHERE id = ?",
                (hit_count, time.time(), workflow_id),
            )

    def delete(self, workflow_id: str):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM workflows WHERE id = ?", (workflow_id,))

    def count(self) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            return conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0]

    def _row_to_wf(self, row) -> WorkflowTemplate:
        steps_data = json.loads(row[2])
        steps = [ActionStep(**s) for s in steps_data]
        return WorkflowTemplate(
            id=row[0],
            intent=row[1],
            steps=steps,
            hit_count=row[3],
            confidence=row[4],
            created_at=row[5],
            last_hit_at=row[6],
            params_schema=json.loads(row[7]) if len(row) > 7 else {},
        )
