"""L3: Optional cloud storage for cold data and federated sharing."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agent_compiler.storage.disk_store import DiskStore
from agent_compiler.core.types import WorkflowTemplate


class CloudStore:
    """Cloud archive for cold workflows. Uses local file as S3 stand-in for demo.

    In production, replace _upload_file / _download_file with boto3 calls.
    """

    def __init__(self, bucket_path: str = "./agent_cache/cloud"):
        self.bucket_path = Path(bucket_path)
        self.bucket_path.mkdir(parents=True, exist_ok=True)

    def archive(self, wf: WorkflowTemplate, local_store: DiskStore):
        """Move a cold workflow from local disk to cloud storage."""
        manifest = {
            "id": wf.id,
            "intent": wf.intent,
            "steps": [
                {"tool_name": s.tool_name, "params": s.params,
                 "is_generic": s.is_generic, "description": s.description}
                for s in wf.steps
            ],
            "params_schema": wf.params_schema,
            "hit_count": wf.hit_count,
            "confidence": wf.confidence,
            "archived_at": time.time(),
        }
        fpath = self.bucket_path / f"{wf.id}.json"
        fpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        local_store.delete(wf.id)

    def restore(self, workflow_id: str) -> dict | None:
        """Restore a workflow from cloud storage."""
        fpath = self.bucket_path / f"{workflow_id}.json"
        if not fpath.exists():
            return None
        return json.loads(fpath.read_text(encoding="utf-8"))

    def list_archived(self) -> list[str]:
        return [p.stem for p in self.bucket_path.glob("*.json")]
