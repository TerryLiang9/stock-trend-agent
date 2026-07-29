# -*- coding: utf-8 -*-
"""将三策略预测运行记录保存为 JSON 文件。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonRunRecordRepository:
    def __init__(self, root: Path | str = Path("data") / "trend_forecast" / "runs") -> None:
        self.root = Path(root)

    def save_run_record(self, run_id: str, payload: Dict[str, Any]) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{run_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return str(path)

    def get_run_record(self, run_id: str) -> Dict[str, Any] | None:
        """读取单次运行记录；不存在时返回 None。"""
        path = self.root / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

