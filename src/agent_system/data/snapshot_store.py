# -*- coding: utf-8 -*-
"""冻结模型输入行情快照，保证多模型读取同一份数据。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


class SnapshotStore:
    def __init__(self, root: Path | str = Path("data") / "trend_forecast" / "snapshots") -> None:
        self.root = Path(root)

    def freeze_rows(self, *, run_id: str, rows: Iterable[Dict[str, Any]], metadata: Dict[str, Any] | None = None) -> Dict[str, str]:
        self.root.mkdir(parents=True, exist_ok=True)
        normalized = [self._json_safe(row) for row in rows]
        payload_rows = normalized
        if metadata:
            payload_rows = [{"__snapshot_metadata__": self._json_safe(metadata)}] + normalized
        payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in payload_rows)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        path = self.root / f"{run_id}.{digest[:16]}.jsonl"
        path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        return {"snapshot_uri": str(path), "data_snapshot_id": f"sha256:{digest}"}

    @staticmethod
    def _json_safe(row: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for key, value in row.items():
            if hasattr(value, "isoformat"):
                safe[key] = value.isoformat()
            else:
                safe[key] = value
        return safe

