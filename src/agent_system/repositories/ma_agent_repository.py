# -*- coding: utf-8 -*-
"""均线预测、后验和参数版本的轻量持久化实现。

MVP 使用 JSON 文件，记录结构保持稳定；后续可替换为项目数据库而不改变服务接口。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.agent_system.config.ma_agent import MaParameters
from src.agent_system.schemas.ma_prediction import MaPredictionOutcome, MaTrendPrediction


class MaAgentRepository:
    def __init__(self, root: Path | str = Path("data") / "trend_forecast" / "ma_agent"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_prediction(self, prediction: MaTrendPrediction) -> int:
        key = f"{prediction.symbol}|{prediction.target_date}|{prediction.data_cutoff.isoformat()}|{prediction.parameter_version_id}"
        prediction_id = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
        path = self.root / f"prediction-{prediction_id}.json"
        if not path.exists():
            payload = prediction.model_dump(mode="json")
            payload["prediction_id"] = prediction_id
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return prediction_id

    def list_predictions(self) -> list[MaTrendPrediction]:
        values = []
        for path in sorted(self.root.glob("prediction-*.json")):
            values.append(MaTrendPrediction.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return values

    def save_outcome(self, outcome: MaPredictionOutcome) -> str:
        path = self.root / f"outcome-{outcome.prediction_id}.json"
        path.write_text(json.dumps(outcome.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def list_outcomes(self) -> list[MaPredictionOutcome]:
        return [
            MaPredictionOutcome.model_validate(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.root.glob("outcome-*.json"))
        ]

    def get_parameters(self) -> MaParameters:
        path = self.root / "champion-parameters.json"
        if not path.exists():
            params = MaParameters()
            path.write_text(json.dumps(params.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
            return params
        return MaParameters.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def save_challenger(self, parameters: MaParameters, *, reason: str) -> str:
        version = hashlib.sha256(json.dumps(parameters.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]
        path = self.root / f"challenger-{version}.json"
        path.write_text(json.dumps({"version": version, "reason": reason, "parameters": parameters.model_dump(mode="json")}, ensure_ascii=False, indent=2), encoding="utf-8")
        return version
